"""Hard-negative miner: nearest-label ranking over the SAME-taxon pool + teacher adversary.

§46 — a distractor for "favourite colour" is another colour, never
"database" or "volcano". That holds STRUCTURALLY, not by a filter:
candidates are only ever drawn from ``pools[(taxon, prop)]``, the values
this property takes on entities of this taxon in the graph. A colour
question and a volcano question never share a pool, so no ranking mistake
can cross them.

Ranking inside the pool is ``data.optset.difficulty`` — the project's one
notion of label plausibility (char-3gram cosine blended with word-3gram
Jaccard). It is reproduced here only in MEMOIZED form: embeddings and
trigram sets are computed once per pool member instead of once per
comparison, which turns a 3 000-label pool from ~400 s of scoring into a
few seconds. The score is identical and `test_index_score_matches_optset`
pins it to `optset.difficulty`.

The teacher adversary (§§35, 130) only REORDERS candidates the graph
already allows. It cannot introduce a label the graph does not have and
cannot touch the gold: the graph decides the answer, the teacher redacts.
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data.hardneg import _embed  # noqa: E402
from data.leakage import jaccard, trigrams  # noqa: E402
from data.optset import _label_words  # noqa: E402
from tools.data_factory.teachers_local import (  # noqa: E402
    HttpTeacher,
    LocalTeacher,
    teacher_endpoint_configured,
)

#: share of a row's distractors taken from the nearest-label bucket
HARD_FRACTION = 0.5
#: blend weight of the char-3gram cosine against the word-3gram Jaccard
COS_WEIGHT = 0.5
#: candidates scored exactly per gold, after the trigram-overlap prefilter
MAX_CANDIDATES = 384
#: nearest labels cached per gold
NEAR_LIMIT = 64


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _char_trigrams(text: str) -> set[str]:
    """Character 3-grams, the unit `data.hardneg._embed` hashes.

    The SHORTLIST has to use these, not `data.leakage.trigrams`: that one is
    a WORD 3-gram, so "Austria" and "Australia" are each a single token and
    share nothing. Prefiltering on word trigrams therefore shortlisted
    nobody for single-word labels — which is most country names — and the
    miner silently fell back to random same-taxon draws. The SCORE still
    blends both, exactly as `optset.difficulty` defines it.
    """
    t = f" {text.lower()} "
    return {t[i:i + 3] for i in range(len(t) - 2)}


def _sparse(vec: list[float]) -> dict[int, float]:
    return {i: x for i, x in enumerate(vec) if x}


def _sparse_cos(a: dict[int, float], b: dict[int, float]) -> float:
    """`data.hardneg._cos` over the nonzero entries; `_embed` is unit-norm."""
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b[i] for i, x in a.items() if i in b)


def build_pools(store) -> dict[tuple[str, str], list[str]]:
    """(taxon, prop) -> sorted distinct LABELLED value QIDs seen in that taxon.

    Values Wikidata could not label are dropped here rather than downstream:
    a bare "Q100269625" is not an answer a reader could pick, so it is not a
    distractor either.
    """
    pools: dict[tuple[str, str], set[str]] = {}
    for (e, p), vals in store.triples.items():
        taxon = store.taxon_of.get(e)
        if taxon is None:
            continue
        pools.setdefault((taxon, p), set()).update(
            v for v in vals if store.has_label(v))
    return {k: sorted(v) for k, v in pools.items() if v}


class NearestLabelIndex:
    """Nearest-label index over ONE same-taxon value pool.

    A trigram inverted index (the pattern `gen_schemas.dedup` already uses)
    shortlists the candidates that can possibly score above zero, so a gold
    is scored against a few hundred plausible labels instead of the whole
    pool.
    """

    def __init__(self, ids: list[str], labels: dict[str, str],
                 cos_weight: float = COS_WEIGHT):
        self.ids = list(ids)
        self.label = {i: labels.get(i, i) for i in self.ids}
        self.cos_weight = cos_weight
        self._emb: dict[str, dict[int, float]] = {}
        self._tri: dict[str, set[str]] = {}
        self._grams: dict[str, set[str]] = {}
        self._inv: dict[str, set[str]] = {}
        for i in self.ids:
            text = self.label[i]
            self._emb[i] = _sparse(_embed(text))
            self._tri[i] = trigrams(_label_words(text))
            self._grams[i] = _char_trigrams(text)
            for g in self._grams[i]:
                self._inv.setdefault(g, set()).add(i)
        self._ranked: dict[str, list[str]] = {}
        self._distinct = len({_norm(t) for t in self.label.values()})

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def distinct_labels(self) -> int:
        """Pool capacity in DISTINCT wording, which is what an option set needs."""
        return self._distinct

    def score(self, a: str, b: str) -> float:
        """Identical to `data.optset.difficulty(label[a], label[b], cos_weight)`."""
        cos = _sparse_cos(self._emb[a], self._emb[b])
        jac = jaccard(self._tri[a], self._tri[b])
        return self.cos_weight * cos + (1.0 - self.cos_weight) * jac

    def ranked(self, gold: str, limit: int = NEAR_LIMIT) -> list[str]:
        """Same-pool ids nearest the gold, most plausible first. Cached."""
        cached = self._ranked.get(gold)
        if cached is not None:
            return cached
        counts: Counter = Counter()
        for g in self._grams.get(gold, ()):
            for c in self._inv.get(g, ()):
                counts[c] += 1
        counts.pop(gold, None)
        short = [c for c, _ in sorted(counts.items(),
                                      key=lambda kv: (-kv[1], kv[0]))[:MAX_CANDIDATES]]
        short.sort(key=lambda c: (-self.score(gold, c), c))
        out = short[:limit]
        self._ranked[gold] = out
        return out


def build_indexes(store, pools: dict[tuple[str, str], list[str]] | None = None
                  ) -> dict[tuple[str, str], NearestLabelIndex]:
    pools = pools if pools is not None else build_pools(store)
    return {key: NearestLabelIndex(ids, store.labels)
            for key, ids in pools.items() if len(ids) > 1}


class PoolTooSmall(ValueError):
    """The same-taxon pool cannot supply K distinct labels — never widen it."""


def sample_negatives(index: NearestLabelIndex, gold: str, k: int,
                     rng: random.Random,
                     hard_fraction: float = HARD_FRACTION) -> list[str]:
    """k same-taxon hard negatives for `gold`, nearest-label first.

    `hard_fraction` of them come from the nearest-label bucket and the rest
    from the pool at large, so a row mixes confusable and merely-same-type
    distractors. Two candidates that render to the SAME wording are never
    both used: that would make the question unanswerable, not hard.
    """
    if k < 1:
        return []
    if index.distinct_labels - 1 < k:
        raise PoolTooSmall(
            f"pool of {index.distinct_labels} distinct labels cannot supply {k}")
    taken = {_norm(index.label[gold])}
    picks: list[str] = []
    seen: set[str] = {gold}

    n_hard = min(k, int(round(k * hard_fraction)))
    for cand in index.ranked(gold):
        if len(picks) >= n_hard:
            break
        label = _norm(index.label[cand])
        if cand in seen or label in taken:
            continue
        taken.add(label)
        seen.add(cand)
        picks.append(cand)

    # Tail: seeded draws against the whole same-taxon pool. Rejection
    # sampling, so the cost is O(k) instead of O(pool) per question.
    guard = 0
    while len(picks) < k and guard < 64 * k:
        guard += 1
        cand = index.ids[rng.randrange(len(index.ids))]
        label = _norm(index.label[cand])
        if cand in seen or label in taken:
            continue
        taken.add(label)
        seen.add(cand)
        picks.append(cand)
    if len(picks) < k:  # dense-duplicate pool: finish with a deterministic scan
        for cand in index.ids:
            if len(picks) >= k:
                break
            label = _norm(index.label[cand])
            if cand in seen or label in taken:
                continue
            taken.add(label)
            seen.add(cand)
            picks.append(cand)
    if len(picks) < k:
        raise PoolTooSmall(f"only {len(picks)} distinct labels for k={k}")
    return picks


# -- teacher adversary ----------------------------------------------------

_LOGGED = False


def make_adversary(log=print):
    """Teacher for the adversary role.

    STUB rule: with `JEV_TEACHER_ENDPOINT` unset we run the deterministic
    local fixture teacher and say so once on the console. The gold is never
    at stake either way — the adversary only reorders graph values.
    """
    global _LOGGED
    if teacher_endpoint_configured():
        endpoint = os.environ["JEV_TEACHER_ENDPOINT"]
        model = os.environ.get("JEV_TEACHER_MODEL", "qwen")
        if not _LOGGED:
            log(f"[prog-gold] teacher adversary: HTTP endpoint {endpoint} ({model})")
            _LOGGED = True
        return HttpTeacher(endpoint, model), "http"
    if not _LOGGED:
        log("[prog-gold] JEV_TEACHER_ENDPOINT unset — teacher adversary runs on "
            "the local deterministic fixture (gold still comes from the graph)")
        _LOGGED = True
    return LocalTeacher("qwen"), "local-fixture"


def adversarial_reorder(index: NearestLabelIndex, gold: str, picks: list[str],
                        teacher, key: str) -> tuple[list[str], bool]:
    """Let the teacher promote the candidate it finds most confusable.

    Returns (picks, changed). The teacher is shown the gold and the chosen
    distractors and asked to solve; whichever distractor it prefers is moved
    to the front. It can only permute `picks` — it never adds, drops, or
    selects the gold, so a wrong teacher costs ordering, never the label.
    """
    if len(picks) < 2:
        return picks, False
    question = {
        "kind": "choice",
        "options": [{"id": c, "text": index.label[c]} for c in [gold] + picks],
        "gold": gold,
    }
    try:
        vote = teacher.solve("", question, key)
    except Exception:  # noqa: BLE001 — a flaky teacher must not decide data
        return picks, False
    if vote == gold or vote not in picks or vote == picks[0]:
        return picks, False
    out = [vote] + [c for c in picks if c != vote]
    return out, True
