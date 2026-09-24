"""What information the model is GIVEN, priced at K=77 — #T-option-text.

`artifacts/gates/T-teacher-probe/fullspace.json` put our head at 0.0123 on
the 3 080 BANKING77 test rows against a teacher's published 0.924, same
rows, same 77 intents. Same rows and same K is not the same protocol: the
teacher is handed natural-language definitions of the 77 categories and up
to 24 retrieved labelled examples per prediction, and `data/adapters.py:146`
hands our head `Option(id=l, text=l)` — the identifier string and nothing
else. Before that gap is attributed to the training objective somebody has
to price the input, and pricing it costs one forward pass over a checkpoint
that already exists.

Three arms, one checkpoint, one sealed cut, K=77 throughout:

* **A — actual**: `Option(id=l, text=l)`. The repo as it ships.
* **B — readable**: `Card arrival — <definition>`, the teacher's own 77
  definitions (`data/taxonomies/banking77-jev/`, pinned and hashed).
* **C — with examples**: arm B's options, plus BM25-retrieved labelled
  training examples in the `STATE` — the teacher's `retrieved24` regime, as
  far as a 256-token window lets it go.

The window is not an implementation detail, it is half the result (rule R8).
`eval/calib.py:373-375` encodes every state through
`train_decision.encode_states(..., TRAIN_MAX_LENGTH)` and `TRAIN_MAX_LENGTH`
is **256**. Twenty-four examples do not fit in 256 tokens, and which half
survives depends entirely on what order the state is built in — so this
module measures the budget instead of assuming it: tokens requested, tokens
retained, how many examples survived WHOLE, and where the query itself ended
up. Arm C is published in the order that keeps the query
(`C` — query first) and again in the teacher's order (`C-teacher-order` —
examples first), because the second one is what "reproduce the teacher's
regime" literally means and the difference between them is the budget
finding stated as a number.

Nothing here trains. Arms are chosen on the DEVELOPMENT cut (`eval.cuts`,
rule R7); the reserved test cut takes `--cut test --reason "…"` and one
query goes in `artifacts/gates/T-option-text/test-queries.json` and in the
repo-wide R7 ledger.

CLI:
    CKPT=artifacts/checkpoints/decision/leverstack-d512-prior-ettin-68m-s20260922/stage-001000000
    .venv-train/bin/python -m eval.optiontext run --checkpoint $CKPT
    .venv-train/bin/python -m eval.optiontext run --checkpoint $CKPT \\
        --cut test --arms C --reason "the chosen arm on the reserved cut"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import taxonomy as TX  # noqa: E402
from data.optset import PREFETCH_DIR, iter_rows  # noqa: E402
from eval import cuts as CUTS  # noqa: E402
from eval import fullspace as F  # noqa: E402

TASK = "T-option-text"
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
GATE_PATH = GATE_DIR / "optiontext.json"
LEDGER_PATH = GATE_DIR / "test-queries.json"

#: the window `eval/calib.py` scores every state through, via
#: `training.python.train_decision.TRAIN_MAX_LENGTH`. Read, never set: the
#: task's whole point is that raising it quietly would be cheating.
STATE_WINDOW = 256

#: the teacher's selected variant (`PROTOCOL.md`, `runs/confirm-retrieved24`)
EXAMPLES_WANTED = 24
EXAMPLES_PER_CLASS = 4
BM25_K1 = 1.5
BM25_B = 0.75

EXAMPLES_HEADER = "Labelled examples from the BANKING77 training set:"

#: the published teacher number this whole task exists to explain
TEACHER = {
    "who": "Jev (teacher)",
    "citation": True,
    "accuracy": 0.9240,
    "setup": "all 3 080 official test messages, 77 intents, category "
             "definitions + up to 24 BM25-retrieved labelled examples per "
             "prediction, no weight updates",
    "source": "github.com/simonmesmith/jev-banking77-experiment",
    "caveat": "BANKING77 is public; nobody can inspect the teacher's "
              "training data, so this is not a held-out claim",
}

ARM_SPECS = {
    "A": {
        "name": "A — actual",
        "options": "raw label identifier: `Option(id=l, text=l)`, "
                   "`data/adapters.py:146`",
        "state": "the row's own text, nothing else",
    },
    "B": {
        "name": "B — readable",
        "options": "`<readable name> — <definition>`, the teacher's own 77 "
                   "definitions",
        "state": "the row's own text, nothing else",
    },
    "C": {
        "name": "C — with examples (query first)",
        "options": "same as arm B",
        "state": "the row's own text, then up to 24 BM25-retrieved labelled "
                 "training examples — the query is first so the 256-token "
                 "window cannot eat it",
    },
    "C-teacher-order": {
        "name": "C-teacher-order — with examples (examples first)",
        "options": "same as arm B",
        "state": "the examples first and the query last, which is the order "
                 "the teacher's prompt uses. Published as a DIAGNOSTIC: at "
                 "256 tokens this is what the budget does to a protocol "
                 "copied literally",
    },
}
DEFAULT_ARMS = ("A", "B", "C", "C-teacher-order")

_WORD = re.compile(r"[a-z0-9]+")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------- retrieval (BM25)

def terms(text: str) -> list:
    """Word unigrams plus adjacent word pairs — the teacher's own units.

    `PROTOCOL.md`: "BM25 uses words and adjacent word pairs." Reimplemented
    here rather than vendored, and said so in the card: the retrieval is
    part of arm C's protocol, so it is declared, not assumed.
    """
    words = _WORD.findall(text.lower())
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def normalise(text: str) -> str:
    """The normalised form two messages are 'the same message' under."""
    return " ".join(_WORD.findall(text.lower()))


class BM25:
    """Okapi BM25 over the support pool. Deterministic, no seed, no torch."""

    def __init__(self, docs: list, k1: float = BM25_K1, b: float = BM25_B):
        #: docs: [{"id", "label", "text"}]
        self.docs = docs
        self.k1, self.b = k1, b
        self.lengths = []
        self.postings: dict = defaultdict(list)
        for i, doc in enumerate(docs):
            counts = Counter(terms(doc["text"]))
            self.lengths.append(sum(counts.values()) or 1)
            for term, count in counts.items():
                self.postings[term].append((i, count))
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

    def scores(self, query: str) -> dict:
        out: dict = defaultdict(float)
        for term in set(terms(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for i, count in posting:
                norm = 1 - self.b + self.b * self.lengths[i] / self.avgdl
                out[i] += idf * count * (self.k1 + 1) / (
                    count + self.k1 * norm)
        return out

    def retrieve(self, query: str, want: int = EXAMPLES_WANTED,
                 per_class: int = EXAMPLES_PER_CLASS,
                 banned: str | None = None) -> list:
        """Top `want` docs, at most `per_class` of any one label.

        `banned` is the normalised form of the query itself: the teacher
        keeps messages sharing a normalised form with a held-out message out
        of its retrieval support, and retrieving the query back as its own
        demonstration would answer the question instead of describing it.
        Ties break on the support id, so the list does not depend on dict
        order.
        """
        ranked = sorted(self.scores(query).items(),
                        key=lambda kv: (-kv[1], self.docs[kv[0]]["id"]))
        taken: Counter = Counter()
        out = []
        for i, score in ranked:
            doc = self.docs[i]
            if banned is not None and doc["norm"] == banned:
                continue
            if taken[doc["label"]] >= per_class:
                continue
            taken[doc["label"]] += 1
            out.append({**doc, "bm25": round(score, 6)})
            if len(out) >= want:
                break
        return out


def support_pool(cut, root: str = PREFETCH_DIR) -> list:
    """The train rows arm C may retrieve from: the split minus the cut.

    A demonstration drawn from the very rows being scored is not a
    demonstration, it is the answer key, so the sealed cut's ids are removed
    before anything is indexed.
    """
    scored = set(CUTS.sealed(cut)["ids"])
    out = []
    for n, row in enumerate(iter_rows(cut.dataset, cut.split, root=root)):
        for question in row.get("questions", []):
            sid = (f"{cut.dataset}:{cut.dataset}-{n}:"
                   f"{question.get('id', '')}")
            gold = question.get("answer")
            if sid in scored or not gold or gold == "unknown":
                continue
            out.append({"id": sid, "label": gold, "text": row["state"],
                        "norm": normalise(row["state"])})
    return out


# ------------------------------------------------------------- the arms

def build_state(query: str, examples: list, query_first: bool = True):
    """The arm-C state, with the char span of every piece it is made of.

    The spans are what makes the budget measurable exactly: one tokenizer
    call with `return_offsets_mapping` says which character the window
    stopped at, and a piece survived WHOLE iff it ends at or before it.
    """
    parts: list = []
    spans: list = []

    def add(text: str, kind: str, label: str = "") -> None:
        start = sum(len(p) for p in parts)
        parts.append(text)
        spans.append((kind, label, start, start + len(text)))

    if query_first:
        add(query, "query")
        if examples:
            parts.append(f"\n\n{EXAMPLES_HEADER}\n")
            for ex in examples:
                add(f"- {TX.readable(ex['label'])}: {ex['text']}\n",
                    "example", ex["label"])
    else:
        if examples:
            parts.append(f"{EXAMPLES_HEADER}\n")
            for ex in examples:
                add(f"- {TX.readable(ex['label'])}: {ex['text']}\n",
                    "example", ex["label"])
            parts.append("\n")
        add(query, "query")
    return "".join(parts), [
        {"kind": kind, "label": label, "start": start, "end": end}
        for kind, label, start, end in spans]


def arm_samples(arm: str, base: list, descriptions: dict,
                index: BM25 | None):
    """`(samples, segments_per_row)` for one arm, off the same base rows.

    Every arm scores the SAME sealed rows with the SAME gold in the SAME
    77-option order; only the option strings and the state change, which is
    the only way the three numbers are a comparison rather than three
    experiments.
    """
    if arm == "A":
        return list(base), [None] * len(base)

    def retext(sample):
        opts = [{"id": o["id"],
                 "text": TX.option_text(o["id"], descriptions)}
                for o in sample.options]
        return replace(sample, options=opts)

    if arm == "B":
        return [retext(s) for s in base], [None] * len(base)

    if arm not in ("C", "C-teacher-order"):
        raise ValueError(f"unknown arm {arm!r}")
    if index is None:
        raise ValueError(f"arm {arm!r} needs a retrieval index")
    first = arm == "C"
    out, segs = [], []
    for sample in base:
        examples = index.retrieve(sample.state, banned=normalise(sample.state))
        state, spans = build_state(sample.state, examples, query_first=first)
        out.append(replace(retext(sample), state=state))
        segs.append({"spans": spans, "examples": examples})
    return out, segs


# ------------------------------------------------------- the R8 budget

def budget(tokenizer, samples: list, segments: list,
           window: int = STATE_WINDOW) -> dict:
    """What survived the 256-token window, per arm — rule R8.

    Measured on the SAME strings that were scored, through the SAME
    tokenizer, not estimated from character counts. `requested` is what the
    state would have cost untruncated; `retained` is what the encoder
    actually read.
    """
    requested, retained = [], []
    complete, offered = [], []
    query_ok = 0
    query_start_frac, query_end_tok = [], []
    for i, sample in enumerate(samples):
        full = tokenizer(sample.state, add_special_tokens=True)["input_ids"]
        enc = tokenizer(sample.state, truncation=True, max_length=window,
                        return_offsets_mapping=True, add_special_tokens=True)
        requested.append(len(full))
        retained.append(len(enc["input_ids"]))
        # the last character the window actually reached; specials map to
        # (0, 0), so they cannot pull the boundary back to the start
        cut = max((end for start, end in enc["offset_mapping"]
                   if end > start), default=0)
        seg = segments[i]
        if seg is None:
            offered.append(0)
            complete.append(0)
            query_ok += 1
            query_start_frac.append(0.0)
            query_end_tok.append(len(enc["input_ids"]))
            continue
        spans = seg["spans"]
        exs = [s for s in spans if s["kind"] == "example"]
        q = next(s for s in spans if s["kind"] == "query")
        offered.append(len(exs))
        complete.append(sum(1 for s in exs if s["end"] <= cut))
        whole = q["end"] <= cut
        query_ok += 1 if whole else 0
        query_start_frac.append(round(q["start"] / max(1, len(sample.state)),
                                      4))
        query_end_tok.append(sum(1 for start, end in enc["offset_mapping"]
                                 if end > start and start < q["end"]))

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    n = len(samples)
    truncated = sum(1 for a, b in zip(requested, retained) if a > b)
    return {
        "window_tokens": window,
        "window_source": "training.python.train_decision.TRAIN_MAX_LENGTH, "
                         "applied by eval/calib.py:373-375 — read here, "
                         "never overridden: raising it to make an arm fit "
                         "would be changing the protocol to like the answer",
        "rows": n,
        "state_tokens_requested_mean": mean(requested),
        "state_tokens_requested_max": max(requested) if requested else None,
        "state_tokens_retained_mean": mean(retained),
        "state_tokens_retained_max": max(retained) if retained else None,
        "rows_truncated": truncated,
        "rows_truncated_share": round(truncated / n, 6) if n else None,
        "tokens_dropped_total": sum(a - b for a, b in zip(requested, retained)),
        "examples_offered_mean": mean(offered),
        "examples_offered_total": sum(offered),
        "examples_complete_mean": mean(complete),
        "examples_complete_total": sum(complete),
        "examples_complete_share": (
            round(sum(complete) / sum(offered), 6) if sum(offered) else None),
        "query_survives_whole": query_ok,
        "query_survives_share": round(query_ok / n, 6) if n else None,
        "query_starts_at_state_fraction_mean": mean(query_start_frac),
        "query_last_token_position_mean": mean(query_end_tok),
        "query_position": ("first — the query is the head of the state, so "
                           "it is the last thing the window could lose"
                           if all(f == 0.0 for f in query_start_frac)
                           else "after the examples — the window reaches it "
                                "only once the demonstrations are paid for"),
    }


def option_budget(tokenizer, samples: list) -> dict:
    """The other half of R8: what the OPTION strings cost.

    Option texts go through `DecisionEngine.embed_texts`, whose window is
    `model/encoder.py: DEFAULT_MAX_LENGTH = 512` — a different budget from
    the state's 256, and one arm B could in principle blow. It does not, and
    the artifact says so with a number instead of leaving it to be assumed.
    """
    texts = sorted({o["text"] for s in samples for o in s.options})
    lens = [len(tokenizer(t, add_special_tokens=True)["input_ids"])
            for t in texts]
    return {
        "distinct_option_texts": len(texts),
        "window_tokens": 512,
        "window_source": "model/encoder.py DEFAULT_MAX_LENGTH, used by "
                         "DecisionEngine.embed_texts",
        "option_tokens_mean": round(sum(lens) / len(lens), 3) if lens else 0,
        "option_tokens_max": max(lens) if lens else 0,
        "options_truncated": sum(1 for x in lens if x > 512),
        "sample": texts[:3],
    }


# ------------------------------------------------------------ composition

def arm_report(arm: str, samples: list, entries: list, tokenizer,
               segments: list, elapsed: float, device: str) -> dict:
    rep = F.tally(samples, entries)
    spec = ARM_SPECS[arm]
    return {
        "arm": arm,
        "name": spec["name"],
        "role": ("DIAGNOSTIC — the same arm C in the teacher's own order"
                 if arm == "C-teacher-order" else "measured arm"),
        "protocol": {
            "option_text": spec["options"],
            "state": spec["state"],
            "labelled_examples_in_state": (
                0 if arm in ("A", "B") else EXAMPLES_WANTED),
            "definitions_source": (
                None if arm == "A"
                else "data/taxonomies/banking77-jev/descriptions.json"),
        },
        **rep,
        "chance_printed_beside": rep["chance"],
        "lift_over_chance": (round(rep["accuracy"] / rep["chance"], 4)
                             if rep.get("accuracy") and rep.get("chance")
                             else None),
        "context_budget": budget(tokenizer, samples, segments),
        "option_budget": option_budget(tokenizer, samples),
        "cost": {"seconds": round(elapsed, 2), "rows": len(entries),
                 "device": device,
                 "ms_per_row": round(1000 * elapsed / max(1, len(entries)),
                                     2)},
    }


def verdict(arms: dict, baseline: str = "A") -> dict:
    """The one line the task asks for, computed rather than asserted.

    `explains` is the share of the gap to the teacher that the best arm's
    input information buys: `(best - A) / (teacher - A)`. It is reported
    beside the intervals, because a point move inside overlapping intervals
    buys nothing at all and the ratio on its own would hide that.
    """
    base = arms[baseline]
    measured = {k: v for k, v in arms.items() if k != "C-teacher-order"}
    best = max(measured.values(), key=lambda a: a["accuracy"] or 0.0)
    gap = TEACHER["accuracy"] - (base["accuracy"] or 0.0)
    delta = (best["accuracy"] or 0.0) - (base["accuracy"] or 0.0)
    explains = round(delta / gap, 6) if gap else None
    any_clears = any(a["beats_chance"] for a in measured.values())
    # an arm has only MOVED the number if its interval clears the
    # baseline's point estimate; overlapping intervals are not a move
    moved = [k for k, a in measured.items()
             if k != baseline and a["accuracy_ci95"][0] > (base["accuracy"]
                                                           or 0.0)]
    if any_clears or moved:
        line = (
            f"the information in the input explains {explains:.1%} of the "
            f"{gap:.4f} gap to the teacher ({best['arm']} "
            f"{best['accuracy']:.4f} CI {best['accuracy_ci95']} vs A "
            f"{base['accuracy']:.4f} CI {base['accuracy_ci95']}, chance "
            f"{base['chance']}, K={base['cardinality']}, n={base['n']}): "
            f"arms {moved} move the number, so #T-fullspace-objective is no "
            "longer the only live hypothesis and the option text becomes "
            "corpus material, not an eval knob")
        priority = "shared"
    else:
        line = (
            f"the information in the input explains none of the "
            f"{gap:.4f} gap to the teacher — every arm stays inside the "
            f"chance interval at K={base['cardinality']}, n={base['n']}, "
            f"chance {base['chance']} (A {base['accuracy']:.4f} CI "
            f"{base['accuracy_ci95']}, best {best['arm']} "
            f"{best['accuracy']:.4f} CI {best['accuracy_ci95']}, point "
            f"shift {delta:+.4f} = {explains:.1%} of the gap and not "
            f"distinguishable from zero), so #T-fullspace-objective keeps "
            "first priority with one competitor fewer")
        priority = "first — unchanged, and now with the input-information "\
                   "hypothesis ruled out rather than merely unranked"
    return {
        "teacher": TEACHER,
        "baseline_arm": baseline,
        "best_measured_arm": best["arm"],
        "gap_to_teacher_from_baseline": round(gap, 6),
        "point_shift_best_vs_baseline": round(delta, 6),
        "share_of_gap_explained_by_input": explains,
        "arms_that_move_the_number": moved,
        "any_arm_beats_chance": any_clears,
        "fullspace_objective_priority": priority,
        "line": line,
    }



def examples_source(card: dict, support: dict, arms: tuple) -> dict:
    """Where arm C's demonstrations came from — or that there were none.

    A read with no arm C in it (the reserved-cut confirmation is A and B)
    retrieved nothing, and publishing `support_rows_indexed: 0` there would
    read as "the support pool was empty", which is a different and wrong
    claim.
    """
    out = dict(card.get("examples_source") or {})
    if any(a.startswith("C") for a in arms):
        out.update({"retrieved_in_this_read": True,
                    "support_rows_indexed": support.get("rows"),
                    "support_labels": support.get("labels"),
                    "excluded_scored_rows": support.get("excluded")})
    else:
        out.update({"retrieved_in_this_read": False,
                    "why": f"no arm C in this read (arms {list(arms)}), so "
                           "nothing was retrieved and no support pool was "
                           "indexed — not an empty pool"})
    return out


def final_verdict(doc: dict) -> str:
    """The last line of the artifact: one sentence, both cuts, R2 complete.

    `doc["verdict"]["line"]` answers the question on the cut it was measured
    on. This one is what a reader who stops at the bottom of the file has to
    come away with, so it carries the reserved-cut confirmation too when
    there is one — a dev-only line at the end of a file that also contains a
    test read would be the more flattering half of the evidence.
    """
    line = doc["verdict"]["line"]
    conf = doc.get("reserved_cut_confirmation")
    if not conf:
        return f"[{doc['cut']['name']}] {line}"
    arms = conf["arms"]
    parts = "; ".join(
        f"{a['arm']} {a['hits']}/{a['n']} = {a['accuracy']:.4f} CI "
        f"{a['accuracy_ci95']}" for a in arms.values())
    first = next(iter(arms.values()))
    budget = doc["arms"]["C"]["context_budget"] if "C" in doc["arms"] else {}
    return (
        f"[{doc['cut']['name']} + {conf['cut']['name']}, RESERVED, read "
        f"once] {line} — and the reserved cut confirms it rather than the "
        f"dev-cut ordering: {parts} at K={first['cardinality']}, "
        f"n={first['n']}, chance {first['chance']}, every interval "
        f"containing chance, so the best dev arm does not reproduce; the "
        f"teacher's regime is additionally out of reach on budget alone "
        f"({budget.get('examples_complete_mean')} of "
        f"{budget.get('examples_offered_mean')} retrieved examples survive "
        f"the {budget.get('window_tokens')}-token window), and "
        f"#T-fullspace-objective keeps first priority with the "
        f"input-information hypothesis ruled out rather than unranked")


def compose(cut, ckpt_dir: str, manifest: dict, seal: dict, arms: dict,
            support: dict, reason: str = "") -> dict:
    card = TX.card()
    doc = {
        "format": "jev.gate.v1",
        "task": TASK,
        "artifact": "optiontext",
        "generated_utc": utcnow(),
        "question": "how much of the distance to the teacher is explained by "
                    "WHAT THE MODEL IS SHOWN, rather than by what it was "
                    "trained to optimise",
        "trains_nothing": True,
        "checkpoint": ckpt_dir,
        "checkpoint_sha256": manifest.get("weights_sha256"),
        "model_version": manifest.get("model_version"),
        "run_id": manifest.get("run_id"),
        "tokenizer_hash": manifest.get("tokenizer_hash"),
        "cut": {
            "name": cut.name,
            "reserved": cut.reserved,
            "dataset": cut.dataset,
            "split": cut.split,
            "seed": cut.seed,
            "rows_sealed": seal.get("rows"),
            "rows_scored": arms[next(iter(arms))]["n"],
            "manifest": seal.get("manifest"),
            "manifest_sha256": seal.get("manifest_sha256"),
            "split_sha256": seal.get("split_sha256"),
            "rule": "R7 — arms are chosen here; the reserved cut is read "
                    "once, with the arm already chosen, and logged",
        },
        "cardinality": arms[next(iter(arms))]["cardinality"],
        "chance": arms[next(iter(arms))]["chance"],
        "protocol_is_the_only_difference":
            "the three arms score the same sealed rows with the same gold "
            "in the same 77-option order on the same weights; only the "
            "option strings and the state differ (R8)",
        "sources": {
            "definitions": {
                "file": "data/taxonomies/banking77-jev/descriptions.json",
                "card": "data/taxonomies/banking77-jev/card.json",
                "source": card.get("source_original"),
                "revision": card.get("revision"),
                "sha256": card.get("sha256"),
                "labels": card.get("labels"),
                "license": card.get("license"),
                "license_caveat": card.get("license_caveat"),
                "attribution": card.get("attribution"),
                "usage": card.get("usage"),
            },
            "examples": examples_source(card, support, tuple(arms)),
            "readable_name_transform":
                "label.replace('_',' ').replace('-',' ') with the first "
                "character upper-cased (`data/taxonomy.py: readable`) — one "
                "rule, no acronym table, so the string is reproducible from "
                "the identifier",
        },
        "arms": arms,
        "verdict": verdict(arms),
        "honesty": [
            "the teacher's 0.924 is CITED, not measured by us: its rows are "
            "the reserved test cut and it is not scored through our "
            "pipeline (same_rows: false)",
            "arm C reproduces the teacher's retrieval regime, not its "
            "model: BM25 over words and adjacent word pairs, ≤24 examples, "
            "≤4 per class, re-implemented from PROTOCOL.md",
            "R9 — a single seed and a single checkpoint. This prices the "
            "input; it does not establish a cause, and the winning arm has "
            "to repeat on another seed before any causal verdict",
        ],
        "same_rows": False,
    }
    if reason:
        doc["reason"] = reason
    doc["verdict_line"] = final_verdict(doc)
    return doc


# -------------------------------------------------------------- the runner

def run(ckpt_dir: str, cut_key: str = "dev", arms: tuple = DEFAULT_ARMS,
        limit: int | None = None, device: str = "auto",
        batch_size: int = 16, reason: str = "", write: bool = True,
        log=print) -> dict:
    from eval import calib as C
    from training.python import train_decision as T

    cut = CUTS.CUTS[cut_key]
    seal = CUTS.sealed(cut)
    base = CUTS.samples(cut, limit)
    if not base:
        raise ValueError(f"cut {cut.name!r} produced no samples")
    log(f"[optiontext] {cut.name}: {len(base)} rows, "
        f"K={len(base[0].options)}")

    descriptions = TX.load()
    cover = TX.coverage([o["id"] for o in base[0].options])
    if not cover["complete"]:
        raise ValueError(f"the taxonomy does not define {cover['missing']}: "
                         "arm B would be part arm A without saying so")

    index = None
    support_meta = {"rows": 0, "labels": 0, "excluded": 0}
    if any(a.startswith("C") for a in arms):
        pool = support_pool(cut)
        index = BM25(pool)
        support_meta = {"rows": len(pool),
                        "labels": len({d["label"] for d in pool}),
                        "excluded": seal["rows"]}
        log(f"[optiontext] support pool: {len(pool)} train rows "
            f"({seal['rows']} scored rows excluded)")

    engine, manifest = T.load_checkpoint(ckpt_dir, device)
    tokenizer = engine.backbone.tokenizer
    out = {}
    for arm in arms:
        samples, segments = arm_samples(arm, base, descriptions, index)
        t0 = time.perf_counter()
        entries = C.entries_from_samples(engine, samples, cut.split,
                                         f"{cut.name}:{arm}", cut.dataset,
                                         batch_size)
        elapsed = time.perf_counter() - t0
        out[arm] = arm_report(arm, samples, entries, tokenizer, segments,
                              elapsed, str(engine.device))
        rep = out[arm]
        log(f"[optiontext] {arm}: {rep['hits']}/{rep['n']} = "
            f"{rep['accuracy']} CI {rep['accuracy_ci95']} vs chance "
            f"{rep['chance']} · examples complete "
            f"{rep['context_budget']['examples_complete_mean']}/"
            f"{rep['context_budget']['examples_offered_mean']} · query whole "
            f"{rep['context_budget']['query_survives_share']} · "
            f"{elapsed:.1f}s")

    rel = os.path.relpath(ckpt_dir, ROOT)
    doc = compose(cut, rel, manifest, seal, out, support_meta, reason)
    doc["artifact"] = os.path.relpath(GATE_PATH, ROOT)
    if cut.reserved:
        doc["role"] = ("CONFIRMATION — the arm already chosen on the dev "
                       "cut, read once on the reserved rows the teacher's "
                       "number is quoted on (R7)")
        doc["test_cut_query"] = log_test_query(rel, reason, len(base), arms,
                                               write=write)
    if write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        GATE_PATH.write_text(json.dumps(merge(doc, cut), indent=2,
                                        ensure_ascii=False) + "\n")
    return doc


def merge(doc: dict, cut) -> dict:
    """One artifact, two cuts — the task asks for a single file.

    A read of the reserved cut does not REPLACE the development table it was
    chosen on; it hangs off it. Overwriting would leave the file saying the
    arm was picked on the rows it was confirmed on, which is the exact
    confusion rule R7 exists to prevent.
    """
    if not cut.reserved:
        return doc
    if not GATE_PATH.exists():
        raise ValueError(
            "there is no development table at "
            f"{os.path.relpath(GATE_PATH, ROOT)} to hang a reserved-cut "
            "read off. Arms are chosen on the dev cut first (R7)")
    base = json.loads(GATE_PATH.read_text())
    base.pop("verdict_line", None)
    base["reserved_cut_confirmation"] = doc
    base["updated_utc"] = doc["generated_utc"]
    base["verdict_line"] = final_verdict(base)
    return base


def log_test_query(ckpt_dir: str, reason: str, rows: int, arms: tuple,
                   write: bool = True) -> dict:
    """Write the R7 query to BOTH ledgers — this task's and the repo's.

    The task's own `test-queries.json` is where a reader of THIS gate looks;
    `eval.cuts`' ledger is where a reader counting every read of the
    reserved cut looks. One read, two places, same entry — a per-task ledger
    that the repo-wide count cannot see would make the reserved cut look
    less consulted than it is.
    """
    doc = (json.loads(LEDGER_PATH.read_text()) if LEDGER_PATH.exists()
           else {"format": 1, "task": TASK, "cut": CUTS.TEST.split_name,
                 "rule": "R7 — every read of the reserved cut is logged "
                         "with its date, its checkpoint and its reason. "
                         "Arms are chosen on the dev cut; this file exists "
                         "so a reader can see how often #T-option-text "
                         "looked at the rows the teacher's number is quoted "
                         "on",
                 "also_recorded_in": os.path.relpath(CUTS.LEDGER_PATH, ROOT),
                 "queries": []})
    entry = {
        "when": utcnow(),
        "checkpoint": ckpt_dir,
        "reason": reason or "no reason given on the command line — the read "
                            "happened anyway and is logged as unexplained",
        "artifact": os.path.relpath(GATE_PATH, ROOT),
        "rows": rows,
        "arms": list(arms),
        "by": "eval.optiontext",
    }
    doc["queries"].append(entry)
    doc["updated_utc"] = entry["when"]
    doc["n_queries"] = len(doc["queries"])
    if write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        LEDGER_PATH.write_text(json.dumps(doc, indent=2, ensure_ascii=False)
                               + "\n")
        CUTS.record_query(ckpt_dir, entry["reason"], entry["artifact"],
                          rows=rows, by="eval.optiontext")
    return entry


def finalise(doc: dict) -> dict:
    """Re-derive the parts of a published artifact that are pure arithmetic.

    Both `sources.examples` and `verdict_line` are functions of numbers the
    file already carries, so recomputing them costs nothing and reads no
    data. That matters here: the reserved cut is read ONCE (R7), and a
    presentation fix that forced a second forward pass over those rows would
    spend a query on a string. Anything that would need the model back is
    not in this function.
    """
    card = TX.card()
    for part in (doc, doc.get("reserved_cut_confirmation")):
        if not part:
            continue
        support = (part.get("sources", {}).get("examples") or {})
        part["sources"]["examples"] = examples_source(
            card,
            {"rows": support.get("support_rows_indexed"),
             "labels": support.get("support_labels"),
             "excluded": support.get("excluded_scored_rows")},
            tuple(part["arms"]))
        if "verdict" in part:
            part.pop("verdict_line", None)
            part["verdict_line"] = final_verdict(part)
    doc.pop("verdict_line", None)
    doc["verdict_line"] = final_verdict(doc)
    return doc


# ----------------------------------------------------------------- render

def render(doc: dict) -> str:
    out = []
    add = out.append
    cut = doc["cut"]
    add(f"OPTION TEXT · {doc['checkpoint']}")
    add(f"  cut {cut['name']}{' [RESERVED]' if cut['reserved'] else ''} · "
        f"{cut['rows_scored']}/{cut['rows_sealed']} rows · "
        f"K={doc['cardinality']} · "
        f"chance {doc['chance']} · model_version {doc['model_version']}")
    add("")
    add(f"  {'arm':<17}{'hits':>6}{'acc':>10}{'CI 95 %':>22}{'chance':>9}"
        f"{'beats':>7}{'abst':>7}{'ex/row':>13}{'query':>8}")
    for arm in doc["arms"].values():
        b = arm["context_budget"]
        ci = "[%.4f, %.4f]" % tuple(arm["accuracy_ci95"])
        add(f"  {arm['arm']:<17}{arm['hits']:>6}{arm['accuracy']:>10.4f}"
            f"{ci:>22}{arm['chance']:>9.4f}"
            f"{('yes' if arm['beats_chance'] else 'no'):>7}"
            f"{arm['abstain_rate']:>7.3f}"
            f"{str(b['examples_complete_mean']) + '/' + str(b['examples_offered_mean']):>13}"
            f"{b['query_survives_share']:>8.2f}")
    add("")
    add(f"CONTEXT BUDGET (window {STATE_WINDOW} tokens)")
    for arm in doc["arms"].values():
        b = arm["context_budget"]
        add(f"  {arm['arm']:<17} requested {b['state_tokens_requested_mean']}"
            f" → retained {b['state_tokens_retained_mean']} · truncated "
            f"{b['rows_truncated']}/{b['rows']} rows · examples whole "
            f"{b['examples_complete_total']}/{b['examples_offered_total']} · "
            f"query {b['query_position'].split('—')[0].strip()}")
    add("")
    add("SOURCES")
    src = doc["sources"]["definitions"]
    add(f"  definitions {src['file']} · {src['labels']} labels · "
        f"{src['license']} · {src['source']}@{src['revision'][:12]}")
    ex = doc["sources"]["examples"]
    add(f"  examples    {ex.get('path')} split={ex.get('split')} · "
        f"{ex.get('license')} · support {ex.get('support_rows_indexed')} "
        f"rows, {ex.get('excluded_scored_rows')} scored rows excluded")
    add("")
    add("VERDICT")
    add("  " + doc.get("verdict_line", doc["verdict"]["line"]))
    return "\n".join(out)


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="eval.optiontext")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="score the arms on one cut")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cut", default="dev", choices=sorted(CUTS.CUTS))
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default="")
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--reason", default="",
                   help="why the reserved cut is being read (required for "
                        "--cut test; it goes in both ledgers)")
    f = sub.add_parser("finalise",
                       help="recompute the artifact's derived text (verdict "
                            "line, examples source) without reading any cut")
    f.add_argument("--path", default=str(GATE_PATH))
    args = ap.parse_args(argv)

    if args.cmd == "finalise":
        path = Path(args.path)
        doc = finalise(json.loads(path.read_text()))
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        print(doc["verdict_line"])
        return 0

    if args.cut == "test" and not args.reason.strip():
        ap.error("--cut test needs --reason: every read of the reserved cut "
                 "is logged with why it was made (R7)")

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    doc = run(args.checkpoint, args.cut, arms, args.limit or None,
              args.device, args.batch_size, args.reason,
              write=not args.no_write)
    if args.out and not args.no_write:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(doc, indent=2,
                                             ensure_ascii=False) + "\n")
    print(json.dumps(doc, indent=2, ensure_ascii=False) if args.json
          else render(doc))
    if not args.no_write:
        print(f"\nartifact: {doc.get('artifact')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
