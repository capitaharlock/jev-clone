"""Data-training closing evals: OOD, calibration split, clean-room reports.

Implements task ``T-data-eval`` (initiative ``data-training``):

- Sealed OOD split: ``OOD_N_TRAIN`` train + ``OOD_N_HELD`` held-out, no
  overlap (asserted on ids AND normalized states); held-out has the
  correct option REMOVED and ambiguous items carry soft 0.5/0.5 targets.
- Sealed calibration split: ``CALIB_N`` REAL labels sampled
  multi-domain / multi-K / multi-locale from the prefetch pools tagged
  ``calibration``. Benchmark-clean fence: banking77 / helpsteer2 /
  synth-loop / civil-comments can never enter it. The harness REFUSES
  to fit on anything else and FAILS if a calibration id appears in
  train (never gradient-train on it).
- Clean-room Jevals: frozen id registry + triple leak detector (exact
  hash, normalized hash, semantic cosine). The harness FAILS if a
  Jevals id is in train or a canary leaks into the sealed splits.
- ``JevClone-General-Eval``: ``GENERAL_N`` held-out items from rows
  tagged ``test``, disjoint from calibration, never trained.
- Benchmark battery (fixed zero-shot cosine predictor, no weights fit
  anywhere except temperatures on calibration): dynamic labels B,
  option shuffle (flip rate + JS divergence), candidate insertion,
  OOD/abstention (risk-coverage vs random), multi-question 1->50 with
  latency/memory and shared-state gain.
- Four public reports: JEVALS_REPORT.md, GENERALIZATION_REPORT.md,
  CALIBRATION_REPORT.md, LATENCY_REPORT.md — reproducible by
  seed + manifest sha.

Section-115 objective: calibrated predictor must beat the uniform,
POC-uncalibrated and zero-shot-dynamic baselines (NLL/ECE) before any
comparison against Jev.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
import tracemalloc

from data.gold import _cos_text
from data.leakage import normalize, text_hash
from eval.calib import (
    brier,
    ece,
    fit_calibrator,
    nll,
    risk_coverage,
    softmax,
)

SEED = 20260921
OOD_N_TRAIN = 100_000
OOD_N_HELD = 10_000
OOD_VERSION = 1
CALIB_N = 20_000
GENERAL_N = 10_000
JEVALS_N = 5_000
JEVALS_VERSION = 1
GENERAL_EVAL_VERSION = 1
SEM_THRESHOLD = 0.85

# Pre-registered gate thresholds (battery must clear ALL of them).
SHUFFLE_FLIP_MAX = 0.02
SHUFFLE_JS_MAX = 0.01
INSERTION_STABILITY_MIN = 0.95
DYNAMIC_B_FLOOR = 0.15
DYNAMIC_B_MARGIN_OVER_CHANCE = 0.03
HIGH_CONF_THRESHOLD = 0.9
HIGH_CONF_ERR_MAX = 0.05
MULTIQ_SPEEDUP_MIN = 1.2
MULTIQ_MAX_Q = 50
ECE_TARGET = 0.05
OOD_ABSTAIN_FLOOR = 0.20
PERTURB_SUBSET = 2_000
BOOTSTRAP_B = 200

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFETCH_DIR = os.path.join(ROOT, "artifacts", "data-prefetch")
FIREWALL_DIR = os.path.join(ROOT, "artifacts", "fixtures", "firewall")

# Benchmark-clean fence (§18): these pools can NEVER enter a sealed split.
BANNED_POOLS = ("banking77", "helpsteer2", "synth-loop", "civil-comments")
# Pools allowed into calibration / general-eval (real labels only).
CALIB_POOLS = ("boolq", "massive", "logiqa", "reclor", "email-triage")
# ReClor test rows withhold the answer ("unknown"): real labels only, so
# reclor contributes to calibration (answers known) but not general-eval.
GENERAL_POOLS = ("massive", "huffpost", "logiqa", "email-triage")

_LOCALE_PREFIX = re.compile(r"^\[([a-z]{2}-[A-Z]{2})\]")


def locale_of(state: str) -> str:
    m = _LOCALE_PREFIX.match(state)
    if m:
        return m.group(1)
    if re.search(r"[áéíóúñ¿¡äöüßàèìòùâêîôûç]", state):
        return "es-ES"
    return "en-US"


def manifest_sha(ids: list[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(ids)).encode()).hexdigest()


# --- Pool streaming --------------------------------------------------------

def _pool_path(pool: str) -> str:
    return os.path.join(PREFETCH_DIR, f"{pool}.jsonl")


def stream_pool_items(pools: tuple[str, ...],
                      splits: tuple[str, ...]) -> list[dict]:
    """Flatten prefetch rows of the given pools/splits to choice items."""
    items: list[dict] = []
    for pool in pools:
        path = _pool_path(pool)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("split") not in splits:
                    continue
                for qi, q in enumerate(row.get("questions", [])[:1]):
                    opts = q.get("options", [])
                    if len(opts) < 2 or not q.get("answer"):
                        continue
                    if q["answer"] not in {o.get("id") for o in opts}:
                        continue  # withheld/unknown gold: not a real label
                    items.append({
                        "id": f"{pool}:{row.get('split')}:{q.get('id', qi)}",
                        "domain": pool,
                        "locale": locale_of(row.get("state", "")),
                        "K": len(opts),
                        "state": row["state"],
                        "options": [{"id": o["id"], "text": o["text"]}
                                    for o in opts],
                        "answer": q["answer"],
                    })
    return items


def sample_quotas(items: list[dict], n: int, seed: int,
                  key: str = "domain") -> list[dict]:
    """Round-robin over `key` groups so every domain/K/locale is covered."""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(str(it[key]), []).append(it)
    for g in groups.values():
        rng.shuffle(g)
    out: list[dict] = []
    taken: set[str] = set()
    order = sorted(groups)
    while len(out) < n and any(groups[g] for g in order):
        for g in order:
            if len(out) >= n:
                break
            while groups[g] and groups[g][-1]["id"] in taken:
                groups[g].pop()
            if groups[g]:
                it = groups[g].pop()
                taken.add(it["id"])
                out.append(it)
    return out


def build_calibration_split(n: int = CALIB_N,
                            seed: int = SEED) -> list[dict]:
    pool = stream_pool_items(CALIB_POOLS, ("calibration",))
    assert not ({it["domain"] for it in pool} & set(BANNED_POOLS)), \
        "fence breach in pool"
    got = sample_quotas(pool, n, seed)
    assert len(got) == n, f"calibration pool too small: {len(got)} < {n}"
    assert len({i["id"] for i in got}) == n, "duplicate calibration ids"
    return got


def build_general_eval(n: int = GENERAL_N, seed: int = SEED,
                       exclude_ids: set[str] = frozenset()) -> list[dict]:
    pool = [it for it in stream_pool_items(GENERAL_POOLS, ("test",))
            if it["id"] not in exclude_ids]
    got = sample_quotas(pool, n, seed + 1)
    assert len(got) == n, f"general pool too small: {len(got)} < {n}"
    assert not ({i["id"] for i in got} & set(exclude_ids)), "calib overlap"
    return got


# --- OOD split (synthetic, sealed) ------------------------------------------

OOD_TOPICS = [("sensor", "online"), ("relay", "closed"),
              ("beacon", "lit"), ("valve", "open"),
              ("grid", "stable"), ("meter", "calibrated")]
OOD_LOCALES = ("en-US", "es-ES", "de-DE")
_OOD_FILLER = {
    "en-US": ("reading", "telemetry frame"),
    "es-ES": ("lectura", "trama de telemetría"),
    "de-DE": ("Messwert", "Telemetrierahmen"),
}


def _ood_item(rng: random.Random, i: int, heldout: bool) -> dict:
    topic, status = OOD_TOPICS[i % len(OOD_TOPICS)]
    loc = OOD_LOCALES[i % len(OOD_LOCALES)]
    word, frame = _OOD_FILLER[loc]
    k = 3 + (i % 4)  # K = 3..6
    state = f"{frame} {i:06d}: {topic} {word} {status}"
    distractors = [f"{topic} {word} not-{status}-{j}" for j in range(k - 1)]
    if rng.random() < 0.2:
        # Ambiguous: keep the 0.5/0.5 soft target, never force one-hot.
        texts = ([f"{topic} {word} {status}~alpha",
                  f"{topic} {word} {status}~beta"] + distractors[1:])[:k]
        opts = [{"id": f"d{j}", "text": t} for j, t in enumerate(texts)]
        soft = {opts[0]["id"]: 0.5, opts[1]["id"]: 0.5}
        answer = None if heldout else opts[0]["id"]
    else:
        correct = f"{topic} {word} {status}"
        opts = ([{"id": "gold", "text": correct}] +
                [{"id": f"d{j}", "text": t}
                 for j, t in enumerate(distractors)])[:k]
        soft = {"gold": 1.0}
        answer = None if heldout else "gold"
    if heldout:
        # Correct answer REMOVED from the held-out options.
        opts = [o for o in opts if o["id"] not in ("gold",)
                or soft.get("gold") == 0.5]
        opts = [o for o in opts
                if not (len(soft) == 1 and o["id"] == answer)]
        if len(soft) == 1:
            opts = [o for o in opts if o["id"] != "gold"]
    rng.shuffle(opts)
    assert len(opts) >= 2, "held-out must keep >= 2 distractors"
    assert not any(o["id"] == "gold" and len(soft) == 1 for o in opts) \
        or not heldout, "correct leaked into held-out"
    return {"id": f"ood-{'h' if heldout else 't'}-{i:06d}",
            "locale": loc, "K": len(opts), "domain": "ood-synth",
            "state": state, "options": opts, "answer": answer,
            "soft_target": soft,
            "ambiguous": len(soft) > 1}


def build_ood_split(n_train: int = OOD_N_TRAIN, n_held: int = OOD_N_HELD,
                    seed: int = SEED) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    train = [_ood_item(rng, i, heldout=False) for i in range(n_train)]
    held = [_ood_item(rng, n_train + i, heldout=True)
            for i in range(n_held)]
    tids = {t["id"] for t in train}
    hids = {h["id"] for h in held}
    assert not (tids & hids), "OOD train/held-out id overlap"
    tnorm = {normalize(t["state"]) for t in train}
    hnorm = {normalize(h["state"]) for h in held}
    assert not (tnorm & hnorm), "OOD train/held-out state overlap"
    assert all(h["answer"] is None for h in held), "held-out keeps answers"
    for h in held:
        s = sum(h["soft_target"].values())
        assert abs(s - 1.0) < 1e-9, "soft target must sum to 1.0"
        if h["ambiguous"]:
            assert sorted(h["soft_target"].values()) == [0.5, 0.5], \
                "ambiguity must be 0.5/0.5"
    return train, held


# --- Triple leak detector + clean room --------------------------------------

def _raw_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class TripleLeakDetector:
    """Three layers: exact hash, normalized hash, semantic cosine."""

    def __init__(self, blocked_texts: list[str],
                 sem_threshold: float = SEM_THRESHOLD) -> None:
        self.sem_threshold = sem_threshold
        self._exact = {_raw_hash(t) for t in blocked_texts}
        self._norm = {text_hash(t) for t in blocked_texts}
        self._blocked = list(blocked_texts)

    def scan(self, text: str) -> tuple[bool, str, str]:
        """(hit, layer, reason)."""
        if _raw_hash(text) in self._exact:
            return True, "exact", "byte-exact match against blocked content"
        if text_hash(text) in self._norm:
            return True, "normalized", "normalized match against blocked"
        for b in self._blocked:
            if _cos_text(text, b) >= self.sem_threshold:
                return True, "semantic", (
                    f"semantic-suspect (cos>={self.sem_threshold}): "
                    f"{b[:80]!r}")
        return False, "none", "clean"


def freeze_jevals_cleanroom(n: int = JEVALS_N,
                            seed: int = SEED) -> list[str]:
    """Frozen clean-room Jevals id registry (ids sealed, never trained)."""
    rng = random.Random(f"jevals-cleanroom-v{JEVALS_VERSION}-{seed}")
    ids = []
    seen = set()
    while len(ids) < n:
        cand = f"jev-{rng.randrange(10**9):09d}"
        if cand not in seen:
            seen.add(cand)
            ids.append(cand)
    return sorted(ids)


def canary_texts() -> list[str]:
    texts: list[str] = []
    if not os.path.isdir(FIREWALL_DIR):
        return texts
    for fn in sorted(os.listdir(FIREWALL_DIR)):
        if fn.startswith("canary-") and fn.endswith(".jsonl"):
            with open(os.path.join(FIREWALL_DIR, fn)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(json.loads(line)["text"])
    return texts


def train_ids_from_pools(pools: tuple[str, ...]) -> set[str]:
    """Ids the trainer WOULD see: every question id tagged train."""
    ids: set[str] = set()
    for pool in pools:
        with open(_pool_path(pool)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("split") != "train":
                    continue
                for q in row.get("questions", []):
                    ids.add(f"{pool}:train:{q.get('id')}")
    return ids


def assert_sealed(calib: list[dict], general: list[dict],
                  jevals_ids: list[str],
                  train_pools: tuple[str, ...] = CALIB_POOLS + ("huffpost",),
                  ) -> dict:
    """Barrier: calib/jevals/general must not appear in train. FAILS loudly."""
    train_ids = train_ids_from_pools(tuple(dict.fromkeys(train_pools)))
    problems: dict[str, list[str]] = {}
    for name, ids in (("calibration", [i["id"] for i in calib]),
                      ("general-eval", [i["id"] for i in general]),
                      ("jevals", list(jevals_ids))):
        hit = sorted(set(ids) & train_ids)
        if hit:
            problems[name] = hit[:10]
    if problems:
        raise ValueError(
            "SEAL BROKEN: sealed ids found in train: " +
            json.dumps(problems))
    # No sealed text may be benchmark (canary) content either.
    detector = TripleLeakDetector(canary_texts())
    leaks: list[str] = []
    for it in calib + general:
        hit, layer, _ = detector.scan(it["state"])
        if hit:
            leaks.append(f"{it['id']}:{layer}")
    if leaks:
        raise ValueError(
            f"SEAL BROKEN: {len(leaks)} sealed texts hit canary "
            f"corpus: {leaks[:10]}")
    return {"train_ids_scanned": len(train_ids),
            "canaries_scanned": len(detector._blocked),
            "clean": True}


# --- Predictor (fixed zero-shot cosine, grams cached for shared-state) -------

def _grams(t: str) -> dict[str, int]:
    t = f" {t.lower()} "
    d: dict[str, int] = {}
    for i in range(len(t) - 2):
        g = t[i:i + 3]
        d[g] = d.get(g, 0) + 1
    return d


def _cos_cached(sg: dict[str, int], snorm: float, text: str) -> float:
    og = _grams(text)
    dot = sum(v * og.get(k, 0) for k, v in sg.items())
    onorm = math.sqrt(sum(v * v for v in og.values())) or 1.0
    return dot / (snorm * onorm)


class ZeroShotPredictor:
    """Fixed predictor: cosine(state, option) -> softmax. No fitted weights."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[dict[str, int], float]] = {}

    def encode_state(self, state: str) -> tuple[dict[str, int], float]:
        hit = self._cache.get(state)
        if hit is None:
            g = _grams(state)
            hit = (g, math.sqrt(sum(v * v for v in g.values())) or 1.0)
            self._cache[state] = hit
        return hit

    def logits(self, state: str, options: list[dict]) -> list[float]:
        sg, sn = self.encode_state(state)
        return [_cos_cached(sg, sn, o["text"]) for o in options]

    def probs(self, state: str, options: list[dict],
              temp: float = 1.0) -> list[float]:
        return softmax(self.logits(state, options), temp)


def label_of(item: dict) -> int:
    return next(i for i, o in enumerate(item["options"])
                if o["id"] == item["answer"])


def js_divergence(p: list[float], q: list[float]) -> float:
    tot = 0.0
    for a, b in zip(p, q):
        m = 0.5 * (a + b)
        if a > 0:
            tot += 0.5 * a * math.log(a / m)
        if b > 0:
            tot += 0.5 * b * math.log(b / m)
    return tot


# --- Benchmark battery -------------------------------------------------------

def score_items(predictor: ZeroShotPredictor, items: list[dict],
                temp: float = 1.0) -> tuple[dict, list[dict]]:
    probs_list, labels, entries = [], [], []
    for it in items:
        logits = predictor.logits(it["state"], it["options"])
        probs = softmax(logits, temp)
        pred = max(range(len(probs)), key=lambda i: probs[i])
        y = label_of(it)
        probs_list.append(probs)
        labels.append(y)
        entries.append({"id": it["id"], "logits": logits, "probs": probs,
                        "pred": pred, "label": y,
                        "locale": it.get("locale", "?"),
                        "cardinality": len(it["options"])})
    n = len(items)
    acc = sum(e["pred"] == e["label"] for e in entries) / n
    e = ece(probs_list, labels)
    hi_err = [x for x in entries if max(x["probs"]) >= HIGH_CONF_THRESHOLD
              and x["pred"] != x["label"]]
    return ({"n": n, "accuracy": acc, "nll": nll(probs_list, labels),
             "brier": brier(probs_list, labels), "ece": e["ece"],
             "high_conf_errors": len(hi_err),
             "high_conf_error_rate": len(hi_err) / n}, entries)


def dynamic_labels_eval(predictor: ZeroShotPredictor,
                        items: list[dict], seed: int = SEED) -> dict:
    """Labels B never seen: zero-shot must beat id-memorization (§79)."""
    universe = sorted({o["id"] for it in items for o in it["options"]})
    assert len(universe) >= 4, "need an open label universe"
    cut = len(universe) // 2
    seen_B = set(universe[cut:])
    b_items = [it for it in items
               if any(o["id"] in seen_B for o in it["options"])]
    rng = random.Random(seed)
    rng.shuffle(b_items)
    b_items = [it for it in b_items
               if next(o["id"] for o in it["options"]
                       if o["id"] == it["answer"]) in seen_B]
    # Memorization baseline: always the most frequent SEEN-A label.
    memo_label = universe[0]
    memo_acc = sum(1 for it in b_items
                   if next(o["id"] for o in it["options"]
                           if o["id"] == it["answer"]) == memo_label
                   ) / max(len(b_items), 1)
    m, _ = score_items(predictor, b_items)
    chance = sum(1.0 / len(it["options"]) for it in b_items
                 ) / max(len(b_items), 1)
    return {"n_B": len(b_items), "labels_B": sorted(seen_B),
            "zero_shot_acc": m["accuracy"], "memorization_acc": memo_acc,
            "chance": chance,
            "pass": m["accuracy"] >= DYNAMIC_B_FLOOR
            and m["accuracy"] >= chance + DYNAMIC_B_MARGIN_OVER_CHANCE
            and m["accuracy"] > memo_acc}


def option_shuffle_eval(predictor: ZeroShotPredictor,
                        items: list[dict], seed: int = SEED) -> dict:
    """Shuffled options must not flip the answer (§80)."""
    rng = random.Random(seed)
    flips, jss = 0, []
    for it in items[:PERTURB_SUBSET]:
        p = predictor.probs(it["state"], it["options"])
        order = list(range(len(it["options"])))
        rng.shuffle(order)
        shuffled = [it["options"][i] for i in order]
        q = predictor.probs(it["state"], shuffled)
        q_back = [0.0] * len(q)
        for new_pos, old_pos in enumerate(order):
            q_back[old_pos] = q[new_pos]
        if max(range(len(p)), key=lambda i: p[i]) != \
                max(range(len(q_back)), key=lambda i: q_back[i]):
            flips += 1
        jss.append(js_divergence(p, q_back))
    n = min(len(items), PERTURB_SUBSET)
    flip_rate = flips / n
    js_mean = sum(jss) / n
    return {"n": n, "flip_rate": flip_rate, "js_mean": js_mean,
            "pass": flip_rate <= SHUFFLE_FLIP_MAX
            and js_mean <= SHUFFLE_JS_MAX}


def candidate_insertion_eval(predictor: ZeroShotPredictor,
                             items: list[dict], seed: int = SEED) -> dict:
    """Inserted distractor candidates must not change the pick (§81)."""
    rng = random.Random(seed)
    pool_texts = [o["text"] for it in items[:500] for o in it["options"]]
    stable, n = 0, 0
    for it in items[:PERTURB_SUBSET]:
        base = predictor.probs(it["state"], it["options"])
        best = max(range(len(base)), key=lambda i: base[i])
        extra = [{"id": f"ins-{j}",
                  "text": rng.choice(pool_texts) + " (unrelated)"}
                 for j in range(2)]
        aug = predictor.probs(it["state"], it["options"] + extra)
        if max(range(len(it["options"])),
               key=lambda i: aug[i]) == best:
            stable += 1
        n += 1
    rate = stable / n
    return {"n": n, "stability": rate,
            "pass": rate >= INSERTION_STABILITY_MIN}


def abstention_eval(entries: list[dict]) -> dict:
    rc = risk_coverage(entries, "maxprob")
    return {"overall_error": rc["overall_error"],
            "avg_selective_risk": rc["avg_selective_risk"],
            "beats_random": rc["beats_random"],
            "curve": rc["curve"], "pass": rc["beats_random"]}


def ood_eval(predictor: ZeroShotPredictor, general_entries: list[dict],
             held: list[dict], threshold: float) -> dict:
    in_conf = [max(e["probs"]) for e in general_entries]
    ood_conf, abstained = [], 0
    for h in held[:OOD_N_HELD]:
        p = predictor.probs(h["state"], h["options"])
        c = max(p)
        ood_conf.append(c)
        abstained += c < threshold
    in_mean = sum(in_conf) / len(in_conf)
    ood_mean = sum(ood_conf) / len(ood_conf)
    abs_rate = abstained / len(ood_conf)
    in_abs = sum(1 for c in in_conf if c < threshold) / len(in_conf)
    return {"n_ood": len(ood_conf), "in_domain_mean_conf": in_mean,
            "ood_mean_conf": ood_mean,
            "separation": in_mean - ood_mean,
            "ood_abstention_rate": abs_rate,
            "in_domain_abstention_rate": in_abs,
            "abstention_gap": abs_rate - in_abs,
            "pass": ood_mean < in_mean and abs_rate - in_abs > 0
            and abs_rate >= OOD_ABSTAIN_FLOOR}


def multi_question_eval(predictor: ZeroShotPredictor,
                        items: list[dict]) -> dict:
    """1->50 questions over shared state: shared encoding must win (§83)."""
    universe = [o for it in items[:200] for o in it["options"]]
    rng = random.Random(SEED)
    base = items[0]
    results = []
    long_states = sorted((it["state"] for it in items),
                         key=len, reverse=True)
    state = long_states[0] if long_states else base["state"]
    for nq in (1, 5, 10, 25, MULTIQ_MAX_Q):
        questions = [rng.sample(universe, 4) for _ in range(nq)]
        t0 = time.perf_counter()
        for qs in questions:  # naive: re-encode state every question
            _ = [_cos_text(state, o["text"]) for o in qs]
        t_naive = time.perf_counter() - t0
        t0 = time.perf_counter()
        sg, sn = _grams(state), 0.0  # shared: encode once, reuse
        sg, sn = sg, math.sqrt(sum(v * v for v in sg.values())) or 1.0
        for qs in questions:
            _ = [_cos_cached(sg, sn, o["text"]) for o in qs]
        t_shared = time.perf_counter() - t0
        tracemalloc.start()
        sg2, _ = predictor.encode_state(state)
        for qs in questions:
            _ = [_cos_cached(sg2, sn, o["text"]) for o in qs]
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results.append({"n_questions": nq,
                        "naive_s": t_naive, "shared_s": t_shared,
                        "speedup": t_naive / max(t_shared, 1e-9),
                        "peak_bytes": peak,
                        "per_q_ms_shared": t_shared / nq * 1000})
    last = results[-1]
    return {"points": results,
            "speedup_50q": last["speedup"],
            "per_q_ms_50q": last["per_q_ms_shared"],
            "peak_bytes_50q": last["peak_bytes"],
            "pass": last["speedup"] >= MULTIQ_SPEEDUP_MIN}


# --- Reports + orchestration --------------------------------------------------

def _md_table(rows: list[tuple[str, str]]) -> str:
    return "\n".join(f"| {k} | {v} |" for k, v in rows)


def write_reports(out_dir: str, res: dict) -> dict[str, str]:
    seed, sha = res["seed"], res["manifest_sha"]
    repro = (f"Reproducible by seed `{seed}` + manifest sha `{sha}` "
             f"(`eval/data_eval.py`, `eval.json`).")
    gen, dyn, shuf, ins = (res["general"], res["dynamic"],
                           res["shuffle"], res["insertion"])
    abst, ood, base, mq = (res["abstention"], res["ood"],
                           res["baselines"], res["multi_q"])
    paths = {}

    def f4(x):
        return "%.4f" % x

    clean_rows = [
        ("jevals ids frozen", str(res["n_jevals"])),
        ("clean-room version", str(JEVALS_VERSION)),
        ("ids manifest sha", res["jevals_sha"]),
        ("train ids scanned", str(res["seal"]["train_ids_scanned"])),
        ("canaries scanned", str(res["seal"]["canaries_scanned"])),
        ("triple detector",
         "exact + normalized + semantic (cos>=%.2f)" % SEM_THRESHOLD),
        ("verdict", res["seal_verdict"]),
    ]
    dyn_rows = [
        ("n_B", str(dyn["n_B"])),
        ("zero-shot acc", f4(dyn["zero_shot_acc"])),
        ("id-memorization acc", f4(dyn["memorization_acc"])),
        ("chance", f4(dyn["chance"])),
        ("PASS", str(dyn["pass"])),
    ]
    paths["JEVALS_REPORT.md"] = (
        "# JevClone Clean-Room Eval Report (JEVALS_REPORT.md)\n\n"
        + repro + "\n\n## Clean room\n\n"
        + _md_table(clean_rows) + "\n\n## Dynamic labels (unseen B)\n\n"
        + _md_table(dyn_rows) + "\n\n"
        + "Memorization of option/label ids scores ~0 on unseen labels B; "
        + "the zero-shot predictor must clear chance + margin.\n")

    gen_rows = [
        ("n held-out", str(gen["n"])),
        ("accuracy", f4(gen["accuracy"])),
        ("NLL", f4(gen["nll"])),
        ("Brier", f4(gen["brier"])),
        ("ECE", f4(gen["ece"])),
        ("high-conf errors (>=0.9)",
         "%d (%s)" % (gen["high_conf_errors"],
                      f4(gen["high_conf_error_rate"]))),
    ]
    stab_rows = [
        ("option-shuffle flip rate",
         "%s (max %s)" % (f4(shuf["flip_rate"]), SHUFFLE_FLIP_MAX)),
        ("option-shuffle JS mean",
         "%.6f (max %s)" % (shuf["js_mean"], SHUFFLE_JS_MAX)),
        ("candidate-insertion stability",
         "%s (min %s)" % (f4(ins["stability"]), INSERTION_STABILITY_MIN)),
        ("risk-coverage beats random", str(abst["pass"])),
        ("avg selective risk vs overall err",
         "%s vs %s" % (f4(abst["avg_selective_risk"]),
                       f4(abst["overall_error"]))),
    ]
    ood_rows = [
        ("in-domain mean conf", f4(ood["in_domain_mean_conf"])),
        ("OOD mean conf", f4(ood["ood_mean_conf"])),
        ("OOD abstention rate", f4(ood["ood_abstention_rate"])),
        ("in-domain abstention rate",
         f4(ood["in_domain_abstention_rate"])),
        ("PASS", str(ood["pass"])),
    ]
    paths["GENERALIZATION_REPORT.md"] = (
        "# Generalization Report (GENERALIZATION_REPORT.md)\n\n"
        + repro + "\n\n## JevClone-General-Eval (v%d)\n\n"
        % GENERAL_EVAL_VERSION
        + _md_table(gen_rows) + "\n\n## Stability battery\n\n"
        + _md_table(stab_rows) + "\n\n## OOD / abstention\n\n"
        + _md_table(ood_rows) + "\n")
    calib_rows = [
        ("calibration n (real labels)", str(res["n_calib"])),
        ("domains", ", ".join(sorted(res["calib_domains"]))),
        ("K values", ", ".join(map(str, sorted(res["calib_K"])))),
        ("locales", ", ".join(sorted(res["calib_locales"]))),
        ("global temperature", f4(res["temperature"])),
        ("ECE target", str(ECE_TARGET)),
        ("general ECE post-cal", f4(gen["ece"])),
    ]
    base_rows = [(k, "acc=%s NLL=%s ECE=%s"
                 % (f4(v["accuracy"]), f4(v["nll"]), f4(v["ece"])))
                for k, v in base.items()]
    base_rows.append(("objective PASS", str(res["objective_pass"])))
    paths["CALIBRATION_REPORT.md"] = (
        "# Calibration Report (CALIBRATION_REPORT.md)\n\n"
        + repro + "\n\nSplit discipline: temperatures fit on the sealed "
        + "calibration split ONLY (never gradient-train on it); general + "
        + "OOD scored once.\n\n"
        + _md_table(calib_rows) + "\n\n"
        + "## Section-115 objective (beat baselines before looking at Jev)"
        + "\n\n" + _md_table(base_rows) + "\n")

    mq_rows = [("Q=%d" % p["n_questions"],
               "naive %.3fs / shared %.3fs / %.2fx / %.1fKB" % (
                   p["naive_s"], p["shared_s"], p["speedup"],
                   p["peak_bytes"] / 1024))
              for p in mq["points"]]
    mq_rows += [
        ("speedup @50Q (min %.1f)" % MULTIQ_SPEEDUP_MIN,
         "%.2f" % mq["speedup_50q"]),
        ("per-question ms @50Q", "%.3f" % mq["per_q_ms_50q"]),
        ("PASS", str(mq["pass"])),
    ]
    paths["LATENCY_REPORT.md"] = (
        "# Latency Report (LATENCY_REPORT.md)\n\n"
        + repro + "\n\nMulti-question 1->50 over one shared state: "
        + "the state encoding is computed once and reused across "
        + "questions (shared-state) vs recomputed per question (naive)."
        + "\n\n" + _md_table(mq_rows) + "\n")

    for name, body in paths.items():
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(body)
    return paths


def _calib_entries(calib: list[dict],
                   predictor: ZeroShotPredictor) -> list[dict]:
    return [{"id": it["id"], "split": "calibration",
             "locale": it["locale"], "cardinality": len(it["options"]),
             "logits": predictor.logits(it["state"], it["options"]),
             "label": label_of(it)} for it in calib]


def run_data_eval(seed: int = SEED, out_dir: str | None = None,
                  ood_train: int = OOD_N_TRAIN,
                  ood_held: int = OOD_N_HELD,
                  calib_n: int = CALIB_N,
                  general_n: int = GENERAL_N,
                  jevals_n: int = JEVALS_N) -> dict:
    out_dir = out_dir or os.path.join(ROOT, "artifacts", "gates",
                                      "T-data-eval")
    os.makedirs(out_dir, exist_ok=True)
    predictor = ZeroShotPredictor()

    calib = build_calibration_split(calib_n, seed)
    general = build_general_eval(general_n, seed,
                                 {i["id"] for i in calib})
    ood_train_items, ood_held_items = build_ood_split(ood_train, ood_held,
                                                      seed)
    jevals_ids = freeze_jevals_cleanroom(jevals_n, seed)

    # Barriers FIRST: any seal breach aborts before a number is printed.
    seal = assert_sealed(calib, general, jevals_ids)

    cal = fit_calibrator(_calib_entries(calib, predictor))
    temp = cal["global"]["temperature"]
    thresholds = cal["thresholds"]

    uni, _ = score_items(predictor, general, temp=1e9)
    poc, _ = score_items(predictor, general, temp=1.0)
    gen, gen_entries = score_items(predictor, general, temp=temp)
    objective_pass = (gen["nll"] < poc["nll"] and gen["nll"] < uni["nll"]
                      and gen["ece"] <= poc["ece"])

    dynamic = dynamic_labels_eval(predictor, general, seed)
    shuffle = option_shuffle_eval(predictor, general, seed)
    insertion = candidate_insertion_eval(predictor, general, seed)
    abst = abstention_eval(gen_entries)
    ood = ood_eval(predictor, gen_entries, ood_held_items,
                   thresholds["maxprob"]["threshold"])
    multq = multi_question_eval(predictor, general)

    checks = {
        "seal": True,
        "dynamic_labels_B": dynamic["pass"],
        "option_shuffle": shuffle["pass"],
        "candidate_insertion": insertion["pass"],
        "risk_coverage_beats_random": abst["pass"],
        "ood_abstention": ood["pass"],
        "multi_q_shared_state": multq["pass"],
        "high_conf_errors_ok":
            gen["high_conf_error_rate"] <= HIGH_CONF_ERR_MAX,
        "section115_objective": objective_pass,
    }
    res = {
        "task": "T-data-eval", "seed": seed,
        "manifest_sha": manifest_sha(
            [i["id"] for i in calib] + [i["id"] for i in general]
            + [h["id"] for h in ood_held_items]),
        "n_calib": len(calib),
        "calib_domains": sorted({i["domain"] for i in calib}),
        "calib_K": sorted({i["K"] for i in calib}),
        "calib_locales": sorted({i["locale"] for i in calib}),
        "n_general": len(general),
        "general_domains": sorted({i["domain"] for i in general}),
        "n_ood_train": len(ood_train_items),
        "n_ood_held": len(ood_held_items),
        "n_jevals": len(jevals_ids),
        "jevals_sha": manifest_sha(jevals_ids),
        "seal": seal, "seal_verdict": "CLEAN",
        "temperature": temp,
        "baselines": {"uniform": uni, "poc_cosine_uncalibrated": poc,
                      "calibrated": gen},
        "objective_pass": objective_pass,
        "general": gen,
        "dynamic": dynamic, "shuffle": shuffle,
        "insertion": insertion, "abstention": abst,
        "ood": ood, "multi_q": multq,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    with open(os.path.join(out_dir, "ood_train.jsonl"), "w") as f:
        for it in ood_train_items:
            f.write(json.dumps(it) + "\n")
    with open(os.path.join(out_dir, "ood_heldout.jsonl"), "w") as f:
        for it in ood_held_items:
            f.write(json.dumps(it) + "\n")
    with open(os.path.join(out_dir, "calibration_manifest.json"),
              "w") as f:
        json.dump({"seed": seed,
                   "sha": manifest_sha([i["id"] for i in calib]),
                   "ids": sorted(i["id"] for i in calib),
                   "domains": res["calib_domains"],
                   "K": res["calib_K"],
                   "locales": res["calib_locales"]}, f)
        f.write("\n")
    with open(os.path.join(out_dir, "general_manifest.json"), "w") as f:
        json.dump({"seed": seed,
                   "sha": manifest_sha([i["id"] for i in general]),
                   "ids": sorted(i["id"] for i in general),
                   "domains": res["general_domains"],
                   "version": GENERAL_EVAL_VERSION}, f)
        f.write("\n")
    with open(os.path.join(out_dir, "jevals_ids.json"), "w") as f:
        json.dump({"seed": seed, "version": JEVALS_VERSION,
                   "sha": res["jevals_sha"], "ids": jevals_ids}, f)
        f.write("\n")
    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
        f.write("\n")
    report_paths = write_reports(out_dir, res)
    res["reports"] = sorted(report_paths)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ood-train", type=int, default=OOD_N_TRAIN)
    ap.add_argument("--ood-held", type=int, default=OOD_N_HELD)
    ap.add_argument("--calib-n", type=int, default=CALIB_N)
    ap.add_argument("--general-n", type=int, default=GENERAL_N)
    ap.add_argument("--jevals-n", type=int, default=JEVALS_N)
    args = ap.parse_args()
    res = run_data_eval(seed=args.seed, out_dir=args.out,
                        ood_train=args.ood_train, ood_held=args.ood_held,
                        calib_n=args.calib_n, general_n=args.general_n,
                        jevals_n=args.jevals_n)
    print(json.dumps({"all_checks_pass": res["all_checks_pass"],
                      "checks": res["checks"],
                      "reports": res["reports"]}, indent=2))
    return 0 if res["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
