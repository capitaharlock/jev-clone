"""Eval firewall: versioned benchmark registry, contamination scanner and
stress suite (#T-firewall).

Benchmarks are pinned by immutable revision + SHA-256 of their canary
corpus (``artifacts/fixtures/firewall/``); training on them — inputs,
outputs, teachers or hard negatives — is refused. The stress suite
applies deterministic, seeded perturbations (label rename, opaque-ID
trap, shuffle, hard siblings, irrelevant state, evidence-last,
contradictions, multilingual mismatch, typos) and judges GO/NO-GO
against versioned thresholds.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass

from .leakage import LeakageDetector, text_hash

FIREWALL_VERSION = 1
STRESS_SUITE_VERSION = 1
STRESS_SEED = 20260919

# GO/NO-GO thresholds (§128, §170): canaries must ALL be caught,
# clean fixtures may trip the paraphrase layer only rarely.
REQUIRED_DETECTION_RATE = 1.0
MAX_FALSE_POSITIVE_RATE = 0.05
SIM_THRESHOLD = 0.5

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts",
    "fixtures",
    "firewall",
)


@dataclass
class BenchmarkCard:
    id: str
    revision: str  # immutable benchmark revision, never a live pointer
    sha256: str  # hash of the pinned canary corpus for this benchmark
    usage: str = "eval-only"


BENCHMARKS = [
    BenchmarkCard("mmlu-pro", "rev-2024-06-canary-v1", ""),
    BenchmarkCard("gpqa", "rev-2023-12-canary-v1", ""),
    BenchmarkCard("arc-challenge", "rev-v1-canary-v1", ""),
    BenchmarkCard("musr", "rev-2023-10-canary-v1", ""),
    BenchmarkCard("rewardbench2", "rev-2024-08-canary-v1", ""),
    BenchmarkCard("simpleqa", "rev-2024-11-canary-v1", ""),
    BenchmarkCard("openbookqa", "rev-2018-09-canary-v1", ""),
    # Transfer-semantics holdout: OOD probe, never train (license unclear).
    BenchmarkCard("clinc150-oos", "rev-holdout-canary-v1", ""),
    # Reasoning benchmarks: train contamination would void every eval claim.
    # Frozen out of training by #T-halt-contam (audit-2026-09-21, finding G).
    BenchmarkCard("logiqa", "rev-logiqa-i-canary-v1", ""),
    BenchmarkCard("reclor", "rev-reclor-canary-v1", ""),
    # Quarantined synth-loop corpus (finding C): 190 skeletons, split i%10.
    # Not a benchmark — registered so a silent re-add to any train JOBS list
    # fails loudly at the barrier instead of re-contaminating the dashboard.
    BenchmarkCard("synth-loop", "rev-quarantined-20260921", ""),
]


def _canary_files() -> list[str]:
    return sorted(
        os.path.join(FIXTURE_DIR, fn)
        for fn in os.listdir(FIXTURE_DIR)
        if fn.startswith("canary-") and fn.endswith(".jsonl")
    )


def canary_corpus_hash() -> str:
    h = hashlib.sha256()
    for path in _canary_files():
        with open(path, "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def pinned_benchmarks() -> list[BenchmarkCard]:
    """Benchmarks with the live canary-corpus hash filled in."""
    digest = canary_corpus_hash()
    return [
        BenchmarkCard(b.id, b.revision, digest, b.usage) for b in BENCHMARKS
    ]


class BenchmarkRegistry:
    """Versioned, immutable benchmark registry. Train usage is refused."""

    def __init__(self, cards: list[BenchmarkCard] | None = None) -> None:
        self._cards: dict[str, BenchmarkCard] = {}
        for card in cards if cards is not None else pinned_benchmarks():
            self.register(card)

    def register(self, card: BenchmarkCard) -> None:
        errors = self.check_card(card)
        if errors:
            raise ValueError("; ".join(errors))
        self._cards[card.id] = card

    @staticmethod
    def check_card(card: BenchmarkCard) -> list[str]:
        errors: list[str] = []
        if card.usage != "eval-only":
            errors.append(f"benchmark {card.id}: usage must be eval-only")
        if not card.revision or not card.revision.strip():
            errors.append(f"benchmark {card.id}: immutable revision required")
        if not card.sha256 or len(card.sha256) != 64:
            errors.append(f"benchmark {card.id}: sha256 (64 hex) required")
        else:
            try:
                int(card.sha256, 16)
            except ValueError:
                errors.append(f"benchmark {card.id}: sha256 must be hex")
        return errors

    def ids(self) -> list[str]:
        return sorted(self._cards)

    def check_train_barrier(self, ids: list[str], role: str = "train") -> list[str]:
        """No benchmark content may enter train / teachers / hard negatives."""
        return [
            f"{i}: benchmark {i!r} is eval-only, blocked from {role}"
            for i in ids
            if i in self._cards
        ]

    def manifest(self) -> dict:
        entries = [asdict(self._cards[i]) for i in self.ids()]
        blob = json.dumps(entries, sort_keys=True).encode()
        return {
            "firewall_version": FIREWALL_VERSION,
            "benchmarks": entries,
            "manifest_sha256": hashlib.sha256(blob).hexdigest(),
        }


def _load_texts(prefix: str) -> list[str]:
    texts: list[str] = []
    for fn in sorted(os.listdir(FIXTURE_DIR)):
        if fn.startswith(prefix) and fn.endswith(".jsonl"):
            with open(os.path.join(FIXTURE_DIR, fn)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(json.loads(line)["text"])
    return texts


def canary_texts() -> list[str]:
    return _load_texts("canary-")


def clean_texts() -> list[str]:
    return _load_texts("clean-")


class ContaminationScanner(LeakageDetector):
    """Firewall scanner over the benchmark canary corpus (exact +
    normalized + paraphrase layers, inherited from LeakageDetector)."""

    def __init__(self, sim_threshold: float = SIM_THRESHOLD) -> None:
        super().__init__(canary_texts(), sim_threshold=sim_threshold)
        self.firewall_version = FIREWALL_VERSION


def check_job_allowed(job_name: str, registry=None) -> bool:
    """Refuse to train an eval-only benchmark. Raises ValueError (an error,
    never a warning) naming the blocked benchmark and role."""
    reg = registry if registry is not None else BenchmarkRegistry()
    hits = reg.check_train_barrier([job_name], "train")
    if hits:
        raise ValueError("; ".join(hits))
    return True


def blocked_hash_sets(blocked_texts: list[str] | None = None):
    """O(1) lookup sets for the exact (raw sha256) + normalized layers."""
    texts = blocked_texts if blocked_texts is not None else canary_texts()
    raw = {hashlib.sha256(t.encode()).hexdigest() for t in texts}
    norm = {text_hash(t) for t in texts}
    return raw, norm


def reject_train_rows(
    texts: list[str],
    blocked_texts: list[str] | None = None,
    sim_threshold: float = SIM_THRESHOLD,
    max_report: int = 10,
) -> bool:
    """Reject benchmark content hiding in training rows as a hard error.

    Layers: exact hash, normalized hash (both O(1)), then Jaccard-3gram
    paraphrase via LeakageDetector. Any hit raises ValueError — a leak is a
    failure, never a warning. Returns True when the batch is clean.
    """
    texts = list(texts)
    blocked = blocked_texts if blocked_texts is not None else canary_texts()
    raw_set, norm_set = blocked_hash_sets(blocked)
    detector = LeakageDetector(blocked, sim_threshold=sim_threshold)
    hits: list[tuple[str, str]] = []
    for t in texts:
        if hashlib.sha256(t.encode()).hexdigest() in raw_set:
            hits.append((t, "exact match against blocked content"))
        elif text_hash(t) in norm_set:
            hits.append((t, "normalized match against blocked content"))
        else:
            hit, reason = detector.scan(t)
            if hit:
                hits.append((t, reason))
    if hits:
        shown = "; ".join(f"{t[:80]!r} ({r})" for t, r in hits[:max_report])
        extra = f" (+{len(hits) - max_report} more)" if len(hits) > max_report else ""
        raise ValueError(
            f"refusing {len(hits)} training row(s) leaking eval-only "
            f"content: {shown}{extra}"
        )
    return True


# --- Stress suite: deterministic seeded perturbations ----------------------

_WS = re.compile(r"\s+")
_TYPO_MAP = str.maketrans({"a": "@", "e": "3", "i": "1", "o": "0"})


def _rng(seed: int) -> random.Random:
    return random.Random(f"firewall-v{STRESS_SUITE_VERSION}-{seed}")


def label_rename(text: str, rng: random.Random) -> str:
    labels = ["A", "B", "C", "D"]
    shuffled = labels[:]
    rng.shuffle(shuffled)
    out = text
    for src, dst in zip(labels, shuffled):
        out = re.sub(rf"\b{src}\)", f"{dst})", out)
    return out


def opaque_id_trap(text: str, rng: random.Random) -> str:
    token = f"ID-{rng.randint(1000, 9999)}"
    return f"[{token}] {text} (ref {token})"


def shuffle_words(text: str, rng: random.Random) -> str:
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def hard_sibling(text: str, rng: random.Random) -> str:
    return text + f" Not {_pick(['entirely', 'exactly', 'precisely'], rng)} unrelated."


def _pick(options: list[str], rng: random.Random) -> str:
    return options[rng.randrange(len(options))]


def irrelevant_state(text: str, rng: random.Random) -> str:
    fillers = [
        "Weather today is mild with light winds.",
        "The office printer needs more paper.",
        "Traffic on Main Street is normal.",
    ]
    return f"{_pick(fillers, rng)} {text}"


def evidence_last(text: str, rng: random.Random) -> str:
    parts = [p.strip() for p in text.split(".") if p.strip()]
    if len(parts) < 2:
        return text
    key = parts.pop(rng.randrange(len(parts)))
    return ". ".join(parts) + f". Therefore: {key}."


def contradiction(text: str, rng: random.Random) -> str:
    return text + f" Although some claim the {_pick(['opposite', 'reverse'], rng)}."


def multilingual_mismatch(text: str, rng: random.Random) -> str:
    suffix = _pick(
        ["Sin embargo, esto es irrelevante.", "Ceci est sans rapport.", "Dies ist irrelevant."],
        rng,
    )
    return f"{text} {suffix}"


def typos(text: str, rng: random.Random) -> str:
    words = text.split()
    if not words:
        return text
    idx = rng.randrange(len(words))
    words[idx] = words[idx].translate(_TYPO_MAP)
    return " ".join(words)


PERTURBATIONS = {
    "label_rename": label_rename,
    "opaque_id_trap": opaque_id_trap,
    "shuffle_words": shuffle_words,
    "hard_sibling": hard_sibling,
    "irrelevant_state": irrelevant_state,
    "evidence_last": evidence_last,
    "contradiction": contradiction,
    "multilingual_mismatch": multilingual_mismatch,
    "typos": typos,
}


def run_stress_suite(seed: int = STRESS_SEED) -> dict:
    """Deterministic stress run: perturb canaries + cleans, scan, judge."""
    scanner = ContaminationScanner()
    canaries = canary_texts()
    cleans = clean_texts()
    perturbed_hits = 0
    perturbed_total = 0
    per_perturbation: dict[str, dict] = {}
    for name, fn in PERTURBATIONS.items():
        rng = _rng(seed + hash(name) % (2**31))
        hits = sum(1 for t in canaries if scanner.scan(fn(t, rng))[0])
        per_perturbation[name] = {"hits": hits, "total": len(canaries)}
        perturbed_hits += hits
        perturbed_total += len(canaries)
    exact_hits = sum(1 for t in canaries if scanner.scan(t)[0])
    false_positives = sum(1 for t in cleans if scanner.scan(t)[0])
    detection_rate = exact_hits / len(canaries) if canaries else 0.0
    fp_rate = false_positives / len(cleans) if cleans else 0.0
    verdict = (
        "GO"
        if detection_rate >= REQUIRED_DETECTION_RATE and fp_rate <= MAX_FALSE_POSITIVE_RATE
        else "NO-GO"
    )
    return {
        "firewall_version": FIREWALL_VERSION,
        "stress_suite_version": STRESS_SUITE_VERSION,
        "seed": seed,
        "sim_threshold": SIM_THRESHOLD,
        "canaries": {"hits": exact_hits, "total": len(canaries)},
        "detection_rate": detection_rate,
        "clean": {"false_positives": false_positives, "total": len(cleans)},
        "false_positive_rate": fp_rate,
        "perturbed": per_perturbation,
        "verdict": verdict,
    }


def write_raw_results(results: dict, gate_dir: str) -> str:
    os.makedirs(gate_dir, exist_ok=True)
    path = os.path.join(gate_dir, "raw_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    return path
