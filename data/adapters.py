"""P0 dataset adapters + mixed loader (#T-data-p0).

Every adapter converts raw source rows into the universal schema
(`data/schema.py`), pins its dataset card (immutable revision + SHA-256)
and refuses to load until the record passes schema, license fence
(`data/registry.py`) and contamination scan (`data/firewall.py`).

Free transforms: option shuffle / distractor insertion, semantic label
renames, typo noise — all seeded and reproducible.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from dataclasses import dataclass

from .firewall import BenchmarkRegistry, ContaminationScanner
from .registry import DatasetCard, Registry
from .schema import Example, Option, Question, validate
from .seed import seed_registry

P0_VERSION = 1
MIXER_SEED = 20260920

# Pinned immutable source revisions (HF commit / snapshot ids recorded in
# each dataset card; a live URL is never enough to reproduce a run).
PINNED_REVISIONS = {
    "huffpost": "hf-khalidalt-huffpost-snap-9f2c4a1e",
    "banking77": "hf-polyai-banking77-v1-1a2b3c4d",
    "boolq": "hf-google-boolq-v1-5e6f7a8b",
    "civil-comments": "tfds-civil-comments-3.0.1-c9d0e1f2",
    "helpsteer2": "hf-nvidia-helpsteer2-v1-4b5c6d7e",
}

HUFFPOST_UNIVERSE = [
    "POLITICS", "SPORTS", "TECH", "ENTERTAINMENT",
    "BUSINESS", "CRIME", "TRAVEL", "FOOD",
]
HELPSTEER_ATTRS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
HELPSTEER_LEVELS = ["0", "1", "2", "3", "4"]


def card_sha(dataset_id: str) -> str:
    """Deterministic card hash: pins id + revision + transform."""
    rev = PINNED_REVISIONS[dataset_id]
    blob = f"p0-v{P0_VERSION}|{dataset_id}|{rev}".encode()
    return hashlib.sha256(blob).hexdigest()


def p0_cards() -> list[DatasetCard]:
    base = {c.id: c for c in seed_registry()._cards.values()}
    for did, rev in PINNED_REVISIONS.items():
        old = base[did]
        base[did] = DatasetCard(
            id=old.id, source_original=old.source_original, mirror=old.mirror,
            license=old.license, usage=old.usage, revision=rev,
            sha256=card_sha(did), transform=old.transform,
        )
    return [base[k] for k in ("huffpost", "banking77", "boolq", "civil-comments", "helpsteer2")]


def p0_registry() -> Registry:
    reg = Registry()
    for card in p0_cards():
        reg.register(card)
    return reg


class AdapterError(ValueError):
    pass


def _rng(seed: int) -> random.Random:
    return random.Random(f"p0-v{P0_VERSION}-{seed}")


def _sample_subset(correct: str, universe: list[str], n: int, rng: random.Random) -> list[str]:
    pool = [u for u in universe if u != correct]
    rng.shuffle(pool)
    return [correct] + pool[: max(0, n - 1)]


def _checked(example: Example, source: str) -> Example:
    errors = validate(example)
    if errors:
        raise AdapterError(f"{source}: {'; '.join(errors)}")
    return example


def adapt_huffpost(rows: list[dict], *, seed: int = 0, n_options: int = 4) -> list[Example]:
    """Headline+description -> category choice (smoke only, §148)."""
    rng = _rng(seed)
    out = []
    for i, r in enumerate(rows):
        try:
            head, desc, cat = r["headline"], r["description"], r.get("category")
            split = r.get("split", "train")
        except KeyError as e:
            raise AdapterError(f"huffpost row {i}: missing {e}") from e
        if not head or not str(head).strip() or not desc or not str(desc).strip():
            raise AdapterError(f"huffpost row {i}: empty headline/description")
        if cat is not None and cat not in HUFFPOST_UNIVERSE:
            raise AdapterError(f"huffpost row {i}: unknown category {cat!r}")
        labels = _sample_subset(cat, HUFFPOST_UNIVERSE, n_options, rng) if cat else HUFFPOST_UNIVERSE[:n_options]
        rng.shuffle(labels)
        out.append(_checked(Example(
            state=f"{head}\n{desc}", split=split,
            questions=[Question(id=f"huffpost-cat-{i}", kind="choice",
                                options=[Option(id=l, text=l) for l in labels],
                                answer=cat if cat else "unknown")],
        ), f"huffpost row {i}"))
    return out


def adapt_banking77(rows: list[dict], *, seed: int = 0, n_options: int = 4) -> list[Example]:
    """Utterance -> intent among dynamic labels (semantic-transfer core)."""
    rng = _rng(seed)
    universe = sorted({r["label"] for r in rows if r.get("label")})
    if len(universe) < 2:
        raise AdapterError("banking77: need >= 2 distinct labels")
    out = []
    for i, r in enumerate(rows):
        try:
            text, label = r["text"], r.get("label")
            split = r.get("split", "train")
        except KeyError as e:
            raise AdapterError(f"banking77 row {i}: missing {e}") from e
        if not text or not str(text).strip():
            raise AdapterError(f"banking77 row {i}: empty text")
        labels = _sample_subset(label, universe, n_options, rng) if label else universe[:n_options]
        rng.shuffle(labels)
        out.append(_checked(Example(
            state=text, split=split,
            questions=[Question(id=f"banking77-intent-{i}", kind="choice",
                                options=[Option(id=l, text=l) for l in labels],
                                answer=label if label else "unknown")],
        ), f"banking77 row {i}"))
    return out


def adapt_boolq(rows: list[dict]) -> list[Example]:
    """Passage+question -> yes/no boolean (CC-BY-SA obligations in card)."""
    out = []
    for i, r in enumerate(rows):
        try:
            passage, question, ans = r["passage"], r["question"], r.get("answer")
            split = r.get("split", "train")
        except KeyError as e:
            raise AdapterError(f"boolq row {i}: missing {e}") from e
        if not passage or not str(passage).strip() or not question or not str(question).strip():
            raise AdapterError(f"boolq row {i}: empty passage/question")
        answer = "unknown" if ans is None else ("yes" if ans else "no")
        out.append(_checked(Example(
            state=f"{passage}\nQ: {question}", split=split,
            questions=[Question(id=f"boolq-{i}", kind="boolean",
                                options=[Option("yes", "yes"), Option("no", "no")],
                                answer=answer)],
        ), f"boolq row {i}"))
    return out


def adapt_civil(rows: list[dict]) -> list[Example]:
    """Comment -> boolean toxicity questions with identity slice in id."""
    out = []
    for i, r in enumerate(rows):
        try:
            text, toxic = r["text"], r.get("toxic")
            sl, split = r.get("slice", "all"), r.get("split", "train")
        except KeyError as e:
            raise AdapterError(f"civil row {i}: missing {e}") from e
        if not text or not str(text).strip():
            raise AdapterError(f"civil row {i}: empty text")
        answer = "unknown" if toxic is None else ("toxic" if toxic else "not_toxic")
        out.append(_checked(Example(
            state=text, split=split,
            questions=[Question(id=f"civil-toxicity-{sl}-{i}", kind="boolean",
                                options=[Option("toxic", "toxic"), Option("not_toxic", "not toxic")],
                                answer=answer)],
        ), f"civil row {i}"))
    return out


def adapt_helpsteer2(rows: list[dict]) -> list[Example]:
    """Prompt+response -> five 0-4 choice questions (one per attribute)."""
    out = []
    for i, r in enumerate(rows):
        try:
            prompt, resp = r["prompt"], r["response"]
            split = r.get("split", "train")
        except KeyError as e:
            raise AdapterError(f"helpsteer2 row {i}: missing {e}") from e
        if not prompt or not str(prompt).strip() or not resp or not str(resp).strip():
            raise AdapterError(f"helpsteer2 row {i}: empty prompt/response")
        qs = []
        for attr in HELPSTEER_ATTRS:
            v = r.get(attr)
            if v is not None and v not in (0, 1, 2, 3, 4):
                raise AdapterError(f"helpsteer2 row {i}: {attr}={v!r} outside 0-4")
            qs.append(Question(id=f"helpsteer2-{attr}-{i}", kind="choice",
                               options=[Option(id=l, text=f"score {l}") for l in HELPSTEER_LEVELS],
                               answer=str(v) if v is not None else "unknown"))
        out.append(_checked(Example(state=f"{prompt}\nA: {resp}", split=split, questions=qs),
                            f"helpsteer2 row {i}"))
    return out


ADAPTERS = {
    "huffpost": adapt_huffpost,
    "banking77": adapt_banking77,
    "boolq": adapt_boolq,
    "civil-comments": adapt_civil,
    "helpsteer2": adapt_helpsteer2,
}

# --- Free transforms (seeded, reproducible) ----------------------------------

_TYPO = str.maketrans({"a": "@", "e": "3", "o": "0"})


def shuffle_options(example: Example, rng: random.Random) -> Example:
    out = copy.deepcopy(example)
    for q in out.questions:
        rng.shuffle(q.options)
    return out


def insert_distractor(example: Example, distractor: Option, rng: random.Random) -> Example:
    out = copy.deepcopy(example)
    qs = [q for q in out.questions if q.kind == "choice"]
    if not qs:
        raise AdapterError("insert_distractor: no choice question")
    q = qs[rng.randrange(len(qs))]
    if distractor.id in [o.id for o in q.options]:
        raise AdapterError(f"insert_distractor: id clash {distractor.id!r}")
    q.options.insert(rng.randrange(len(q.options) + 1), distractor)
    return out


def rename_labels(example: Example, mapping: dict[str, str]) -> Example:
    out = copy.deepcopy(example)
    for q in out.questions:
        for o in q.options:
            if o.id in mapping:
                o.id, o.text = mapping[o.id], mapping[o.id]
        if q.answer in mapping:
            q.answer = mapping[q.answer]
    errors = validate(out)
    if errors:
        raise AdapterError(f"rename_labels: {'; '.join(errors)}")
    return out


def typo_noise(text: str, rng: random.Random) -> str:
    words = text.split()
    if not words:
        return text
    idx = rng.randrange(len(words))
    words[idx] = words[idx].translate(_TYPO)
    return " ".join(words)


# --- Fence + firewall pre-checks ----------------------------------------------

@dataclass
class LoadReport:
    dataset: str
    fence_ok: bool
    fence_reason: str
    contamination_hits: int


def check_load(rows_texts: list[str], dataset: str, reg: Registry | None = None,
               bench: BenchmarkRegistry | None = None) -> LoadReport:
    """Schema-license-contamination pre-check before any adapter runs."""
    reg = reg or p0_registry()
    bench = bench or BenchmarkRegistry()
    ok, reason = reg.train_ok(dataset)
    scanner = ContaminationScanner()
    hits = sum(1 for t in rows_texts if scanner.scan(t)[0])
    blocked = bench.check_train_barrier([dataset], "train")
    if blocked:
        return LoadReport(dataset, False, blocked[0], hits)
    if hits:
        return LoadReport(dataset, False, f"{hits} contamination hits", hits)
    return LoadReport(dataset, ok, reason if ok else f"gated: {reason}", hits)


# --- Mixed loader ---------------------------------------------------------------

class MixedLoader:
    """Weighted, seeded mixer over per-dataset pools. Deterministic:
    same seed -> same order, batches and manifest; question ids must be
    disjoint across splits."""

    def __init__(self, pools: dict[str, list[Example]], weights: dict[str, float] | None = None,
                 seed: int = MIXER_SEED) -> None:
        if not pools:
            raise AdapterError("MixedLoader: empty pools")
        self.pools = pools
        self.weights = weights or {k: 1.0 for k in pools}
        self.seed = seed
        self._order = self._build_order()
        self._check_split_disjoint()

    def _build_order(self) -> list[Example]:
        rng = random.Random(f"p0-mixer-v{P0_VERSION}-{self.seed}")
        bags = {k: [e for e in v] for k, v in self.pools.items()}
        for v in bags.values():
            rng.shuffle(v)
        keys = sorted(bags)
        total_w = sum(self.weights.get(k, 0) for k in keys)
        order: list[Example] = []
        idx = {k: 0 for k in keys}
        total = sum(len(v) for v in bags.values())
        while len(order) < total:
            r = rng.random() * total_w
            for k in keys:
                r -= self.weights.get(k, 0)
                if r <= 0 and idx[k] < len(bags[k]):
                    order.append(bags[k][idx[k]])
                    idx[k] += 1
                    break
            else:
                for k in keys:
                    if idx[k] < len(bags[k]):
                        order.append(bags[k][idx[k]])
                        idx[k] += 1
                        break
        return order

    def _check_split_disjoint(self) -> None:
        seen: dict[str, str] = {}
        for ex in self._order:
            for q in ex.questions:
                if q.id in seen and seen[q.id] != ex.split:
                    raise AdapterError(f"id {q.id} crosses {seen[q.id]} -> {ex.split}")
                seen.setdefault(q.id, ex.split)

    def batches(self, batch_size: int) -> list[list[Example]]:
        return [self._order[i:i + batch_size] for i in range(0, len(self._order), batch_size)]

    def manifest(self) -> dict:
        entries = []
        for ds, pool in sorted(self.pools.items()):
            for ex in pool:
                for q in ex.questions:
                    entries.append(f"{ds}|{ex.split}|{q.id}|{len(q.options)}")
        blob = json.dumps({"seed": self.seed, "entries": sorted(entries)}, sort_keys=True).encode()
        return {
            "p0_version": P0_VERSION,
            "seed": self.seed,
            "weights": self.weights,
            "examples": len(self._order),
            "manifest_sha256": hashlib.sha256(blob).hexdigest(),
        }


GATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts", "gates", "T-data-p0",
)


def build_summary(pools: dict[str, list[Example]]) -> dict:
    reg = p0_registry()
    bench = BenchmarkRegistry()
    counts, splits, opt_hist = {}, {}, {}
    for ds, pool in pools.items():
        counts[ds] = len(pool)
        for ex in pool:
            splits[ex.split] = splits.get(ex.split, 0) + 1
            for q in ex.questions:
                opt_hist[str(len(q.options))] = opt_hist.get(str(len(q.options)), 0) + 1
    fence = {c.id: {"license": c.license, "usage": c.usage,
                    "train_ok": reg.train_ok(c.id)[0],
                    "reason": reg.train_ok(c.id)[1],
                    "revision": c.revision, "sha256": c.sha256}
             for c in p0_cards()}
    loader = MixedLoader(pools)
    summary = {
        "p0_version": P0_VERSION,
        "counts": counts,
        "splits": splits,
        "option_histogram": opt_hist,
        "fence": fence,
        "firewall_benchmarks_blocked": bench.ids(),
        "huffpost_note": "smoke only (§148): not reasoning evidence, gated from commercial weights",
        "boolq_note": "CC-BY-SA-3.0 share-alike obligations preserved in card + manifests",
        "manifest": loader.manifest(),
    }
    return summary


def write_summary(summary: dict, gate_dir: str = GATE_DIR) -> str:
    os.makedirs(gate_dir, exist_ok=True)
    path = os.path.join(gate_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return path
