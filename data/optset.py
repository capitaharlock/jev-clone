"""Dynamic option-set sampler over the converted V1 corpus (#T-optset-sampler).

The converters already put good material on disk: `artifacts/data-prefetch/
{banking77,massive,huffpost,boolq}.jsonl` keep state, options and option
TEXTS in the V1 schema. What was missing is the *dataloader* that feeds
`model.decision_head` the contract it expects.

Finding B, second half: HuffPost converts to 4 candidates per row but
`train_baseline.py` trained over the 41 GLOBAL categories, destroying the
dynamic-option contract. This sampler is what stops that from coming back:

* **K is per-row and variable** (`k_min..k_max`, clamped by the pool), never
  the dataset's global label space. `global_space_violations()` is a hard
  check, run by the gate and by `data/test_optset.py`.
* **Hard negatives** come from `data.hardneg.nearest_labels` (char-3gram
  cosine over label embeddings) blended with `data.leakage.jaccard` (word
  overlap, so `card_arrival` vs `card_delivery_estimate` ranks above a
  random intent). The easy/hard mix is a configured target AND a counted
  measurement — never assumed.
* **`unknown` rows**: a configured fraction of rows where the gold is
  ABSENT from the offered options, so the head's `unknown` logit trains on
  real supervision instead of becoming a post-hoc threshold.
* **Option order is reshuffled every epoch** (the option SET of a row is
  stable, seeded by the row; the ORDER is seeded by `(row, epoch)`), which
  is what the pointer head's order-invariance test consumes.

Coverage is the clean, converted P0 corpus only. `synth-loop` (quarantined,
finding C) and `logiqa`/`reclor` (eval-only, finding G) are refused twice
over: by `ALLOWED_DATASETS` and by `data.firewall.check_job_allowed`, which
raises rather than warns.

* **Cardinality regime** (`k_min`/`k_max`, #T-bigk-optsets): the default
  3-8 is the *sampled-options* regime all of phase 1 trained in. Setting
  `k_min = k_max = 0` (`FULL_SPACE`) switches to the *full-space* regime:
  the row offers EVERY label of its own space, so `cross_entropy` over the
  emitted columns already normalises over the whole space plus `unknown`.
  Which regime a run used is declared by `distractor_regime()` and travels
  in the gate — never inferred from a constant in this file.

* **Space-scoped pools** (`space_scoped=`, #T-labelspace-div): the default
  pool of a dataset is its GLOBAL label space, keyed on the option id. That
  is right for a taxonomy (`dbpedia14`: Album, Animal, …) and wrong for a
  corpus whose rows carry their own option set — `swag`'s ids are positions
  (o0..o3), so the pool collapses to four texts taken from whichever row
  defined each id first, and `episodic-div`'s 20 000 miniature taxonomies
  merge into one pool of 99 544 pseudowords. Naming such a dataset here
  makes the ROW's own option set its label space: distractors stay inside
  it, and the per-gold difficulty ranking costs a six-label pass instead of
  a 99 544-label one (measured: 15.7 s per distinct gold before, ~0 after).
  It is opt-in and empty by default, so no existing run changes.

The sampler NEVER rewrites the jsonl and `data/schema.py` stays untouched:
rows are read as they are and batches are built in memory.

Emitted contract (one `Sample` per question), matching
`model.decision_head.DecisionEngine.logits(mem, question, options)`:

    state       str                     encoded once per row
    question    str                     `q["text"]` or, as the head does,
                                        the question id when there is none
    options     list[{"id", "text"}]    K entries, shuffled, K varies
    answer      option id or "unknown"
    gold_index  int                     index into options, or K for
                                        `unknown` — the head's own
                                        convention (logits are [K + 1])

CLI:
    python3 -m data.optset gate
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field

from .firewall import check_job_allowed
from .hardneg import _cos, _embed
from .leakage import jaccard, trigrams
from .mix import MIX_1M_FENCED
from .mix import SOURCES as MIX_SOURCES
from .mix import TRAINABLE_DATASETS

# Clean, converted, train-approved P0 corpus. NOT synth-loop (quarantined,
# finding C) and NOT logiqa/reclor (eval-only, finding G). This is the
# DEFAULT set of a bare `OptionSetSampler()`, not the limit of what may be
# trained: the mixture a run consumes is decided by `data.mix` (which caps
# every source at 15 %, #T-corpus-rebalance) and named explicitly.
ALLOWED_DATASETS = ("banking77", "massive", "huffpost", "boolq")

UNKNOWN_ID = "unknown"
DEFAULT_SEED = 20260921

#: `k_min == k_max == FULL_SPACE` means "K is the size of the row's own
#: label space". Zero cannot be a real K (a row with no options is not a
#: question), so it is free to carry the regime, and
#: `train_decision.cardinality_regime` already reads this exact sentinel.
FULL_SPACE = 0

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFETCH_DIR = os.path.join(ROOT, "artifacts", "data-prefetch")
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", "T-optset-sampler")

# A pool this small IS the question (boolq: yes/no). Such a dataset cannot
# offer an option set different from its global label space, so it is
# exempt from the global-label-space check — explicitly, and reported.
BINARY_POOL_SIZE = 2


@dataclass(frozen=True)
class SamplerConfig:
    """Every knob the gate has to report is a field here, not a constant."""

    #: `k_min`/`k_max`: the per-row cardinality. `FULL_SPACE` in BOTH
    #: (0, 0) is the full-space regime — K is the size of the row's own
    #: label space, whatever that is. Any other pair is the sampled-options
    #: regime, and `k_max` may be as large as the space (77 for banking77,
    #: 41 for huffpost): it is clamped per row by `min(k_max, |space|)`.
    k_min: int = 3
    k_max: int = 8
    #: target share of the distractors of a row drawn from the hard bucket
    hard_fraction: float = 0.5
    #: share of rows emitted with the gold ABSENT (gold_index == K)
    unknown_fraction: float = 0.12
    #: the top `hard_top_ratio` of the difficulty ranking is the hard bucket
    hard_top_ratio: float = 0.25
    hard_top_min: int = 4
    #: weight of the char-3gram cosine against the word-Jaccard term
    cos_weight: float = 0.5
    split: str = "train"
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if not (self.full_space or 2 <= self.k_min <= self.k_max):
            raise ValueError(
                "need 2 <= k_min <= k_max, or k_min == k_max == 0 "
                "(FULL_SPACE) for the full-space regime")
        if not 0.0 <= self.hard_fraction <= 1.0:
            raise ValueError("hard_fraction must be in [0, 1]")
        if not 0.0 <= self.unknown_fraction < 1.0:
            raise ValueError("unknown_fraction must be in [0, 1)")
        if not 0.0 < self.hard_top_ratio <= 1.0:
            raise ValueError("hard_top_ratio must be in (0, 1]")
        if not 0.0 <= self.cos_weight <= 1.0:
            raise ValueError("cos_weight must be in [0, 1]")

    @property
    def full_space(self) -> bool:
        """Is this the full-space regime? (#T-bigk-optsets)"""
        return self.k_min == FULL_SPACE and self.k_max == FULL_SPACE

    def draw_k(self, rng: random.Random, space_size: int) -> int:
        """How many options a row of a `space_size` space offers.

        The `randint` is consumed in BOTH regimes even though the
        full-space answer does not depend on it, so two arms that share a
        seed keep the same random stream per row: the same rows become
        `unknown` rows, and the only difference between the arms is the
        cardinality. That is what makes them comparable.
        """
        k_hi = space_size if self.full_space else min(self.k_max, space_size)
        k_lo = min(self.k_min, k_hi)
        drawn = rng.randint(k_lo, k_hi)
        return space_size if self.full_space else drawn

    def regime(self) -> dict:
        """The cardinality regime, declared — never inferred by a reader."""
        if self.full_space:
            return {
                "mode": "full space",
                "k_min": self.k_min, "k_max": self.k_max,
                "k_per_row": ("|space| for an answerable row; |space| - 1 "
                              "for an `unknown` row, which is the space "
                              "with the gold withheld"),
                "distractor_sampling": ("none — every other label of the "
                                        "row's own space is offered, so "
                                        "the hard and easy buckets are "
                                        "both exhausted"),
                "hard_fraction_applies": False,
                "loss_denominator": ("the [K + 1] columns handed to "
                                     "`cross_entropy` ARE the row's whole "
                                     "space plus `unknown`"),
                "unknown_fraction": self.unknown_fraction,
            }
        return {
            "mode": "sampled options",
            "k_min": self.k_min, "k_max": self.k_max,
            "k_per_row": (f"{self.k_min}-{self.k_max} drawn uniformly, "
                          "clamped by |space|"),
            "distractor_sampling": (
                f"{round(100.0 * self.hard_fraction)} % of the distractors "
                f"from the hard bucket (the top "
                f"{round(100.0 * self.hard_top_ratio)} % of the difficulty "
                f"ranking, at least {self.hard_top_min} labels), the rest "
                "from the easy bucket, both without replacement"),
            "hard_fraction_applies": True,
            "loss_denominator": ("the [K + 1] columns handed to "
                                 "`cross_entropy` — a SUBSET of the space"),
            "unknown_fraction": self.unknown_fraction,
        }


@dataclass
class Sample:
    """One training row for the pointer head. K is whatever fits this row."""

    dataset: str
    row_id: str
    question_id: str
    state: str
    question: str
    options: list[dict] = field(default_factory=list)
    answer: str = UNKNOWN_ID
    gold_index: int = 0
    #: for an `unknown` row: the gold that was deliberately withheld
    dropped_gold: str | None = None
    n_hard: int = 0
    n_easy: int = 0
    epoch: int = 0
    #: name of the label space this row drew from — the dataset for a
    #: global pool, `dataset#n` for a space-scoped one. `cross_space_
    #: violations()` proves every option came from THIS space and no other.
    space: str = ""
    #: size of that space, so `k == space_size` is checkable per sample
    space_size: int = 0

    @property
    def k(self) -> int:
        return len(self.options)

    @property
    def is_unknown(self) -> bool:
        return self.answer == UNKNOWN_ID

    def option_ids(self) -> list[str]:
        return [o["id"] for o in self.options]

    def gold_target(self) -> int:
        """Index for `cross_entropy` against the head's `[K + 1]` logits."""
        return self.gold_index

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(k=self.k, is_unknown=self.is_unknown)
        return d


# -- corpus access (read-only; the jsonl is never rewritten) ---------------

def dataset_path(dataset: str, root: str = PREFETCH_DIR) -> str:
    return os.path.join(root, f"{dataset}.jsonl")


def assert_trainable(datasets: list[str] | tuple[str, ...]) -> list[str]:
    """Registry + firewall barrier. Raises, never warns.

    `check_job_allowed` is the same barrier `#T-halt-contam` installed, so
    a silent re-add of `synth-loop`, `logiqa` or `reclor` fails here. The
    registry is `data.mix.SOURCES`: a corpus that is not in it has no
    family, no origin and no weight, which means the 15 %/30 % caps cannot
    be computed for it — training on it would be unguarded by construction.
    """
    out = []
    for name in datasets:
        if name not in TRAINABLE_DATASETS:
            raise ValueError(
                f"{name!r} is not a registered trainable corpus; registry: "
                f"{list(TRAINABLE_DATASETS)} (synth-loop is quarantined, "
                f"logiqa/reclor are eval-only)")
        check_job_allowed(name)
        out.append(name)
    return out


def source_paths(dataset: str, root: str = PREFETCH_DIR) -> list[str]:
    """Every shard of a source: `synth-v1` is six files, the rest are one."""
    source = MIX_SOURCES.get(dataset)
    paths = (source.paths(root) if source is not None
             else [dataset_path(dataset, root)])
    return [p for p in paths if os.path.exists(p)]


def iter_rows(dataset: str, split: str = "train", limit: int | None = None,
              root: str = PREFETCH_DIR, keep=None):
    """Stream V1 rows of one split. Read-only, one json parse per line.

    `keep(index, row) -> bool` is the mixture's seeded selector
    (`data.mix.MixSpec.keep`): `index` counts the rows of this split in file
    order, so the decision does not depend on how many rows were kept
    before it. A source with several shards is read in registry order.
    """
    n = 0
    index = -1
    for path in source_paths(dataset, root):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if split is not None and row.get("split") != split:
                    continue
                index += 1
                if keep is not None and not keep(index, row):
                    continue
                yield row
                n += 1
                if limit is not None and n >= limit:
                    return


def label_pool(dataset: str, root: str = PREFETCH_DIR) -> list[dict]:
    """The dataset's GLOBAL label space: every option seen in the file.

    This is exactly what must never become a row's option set. It is built
    once so the sampler can prove, per sample, that it did not emit it.
    """
    seen: dict[str, str] = {}
    for row in iter_rows(dataset, split=None, root=root):
        for q in row.get("questions", []):
            for opt in q.get("options", []):
                seen.setdefault(opt["id"], opt.get("text", opt["id"]))
    return [{"id": i, "text": seen[i]} for i in sorted(seen)]


def question_text(question: dict) -> str:
    """The head's own convention: `q["text"]`, else the question id."""
    return question.get("text") or question.get("id", "")


# -- difficulty: hardneg cosine blended with leakage.jaccard ---------------

def _label_words(text: str) -> str:
    """`card_arrival` -> `card arrival`, so word-Jaccard can see siblings."""
    return text.replace("_", " ").replace("-", " ").replace(".", " ")


def difficulty(gold_text: str, cand_text: str,
               cos_weight: float = 0.5) -> float:
    """Blended plausibility of `cand` as a distractor for `gold`.

    `data.hardneg._embed` is a char-3gram embedding (catches `PARENTS` vs
    `PARENTING`); `data.leakage.jaccard` over word 3-grams of the
    de-underscored label catches shared tokens (`card_arrival` vs
    `card_delivery_estimate`). Neither alone covers both label styles.
    """
    cos = _cos(_embed(gold_text), _embed(cand_text))
    jac = jaccard(trigrams(_label_words(gold_text)),
                  trigrams(_label_words(cand_text)))
    return cos_weight * cos + (1.0 - cos_weight) * jac


class DistractorIndex:
    """Per-gold ranking of the pool, split into a hard and an easy bucket.

    Cached per gold label: a pool is at most a few hundred short strings,
    so the whole index costs one pass over the pool per distinct gold.
    """

    def __init__(self, pool: list[dict], config: SamplerConfig):
        self.pool = pool
        self.config = config
        self.ids = [o["id"] for o in pool]
        self.text = {o["id"]: o["text"] for o in pool}
        self._ranked: dict[str, list[str]] = {}

    def hard_size(self, n_candidates: int) -> int:
        return max(0, min(n_candidates,
                          max(self.config.hard_top_min,
                              int(round(n_candidates *
                                        self.config.hard_top_ratio)))))

    def ranked(self, gold_id: str) -> list[str]:
        """Pool ids minus the gold, hardest (most plausible) first."""
        cached = self._ranked.get(gold_id)
        if cached is not None:
            return cached
        gold_text = self.text.get(gold_id, gold_id)
        scored = [
            (i, difficulty(gold_text, self.text[i], self.config.cos_weight))
            for i in self.ids if i != gold_id
        ]
        scored.sort(key=lambda t: (-t[1], t[0]))
        out = [i for i, _ in scored]
        self._ranked[gold_id] = out
        return out

    def buckets(self, gold_id: str) -> tuple[list[str], list[str]]:
        ranked = self.ranked(gold_id)
        cut = self.hard_size(len(ranked))
        return ranked[:cut], ranked[cut:]


# -- the sampler -----------------------------------------------------------

class OptionSetSampler:
    """Emits `Sample`s with variable K, hard negatives and `unknown` rows.

    The option SET of a question is deterministic in `(seed, dataset, row,
    question)` — the curriculum does not wobble between epochs. The option
    ORDER is deterministic in `(seed, epoch, dataset, row, question)`, so
    every epoch reshuffles.
    """

    def __init__(self, datasets: list[str] | tuple[str, ...] = ALLOWED_DATASETS,
                 config: SamplerConfig | None = None,
                 rows_per_dataset: int | None = None,
                 root: str = PREFETCH_DIR,
                 pools: dict | None = None,
                 keep: dict | None = None,
                 quotas: dict | None = None,
                 space_scoped=()):
        """`pools` and `keep` are what a `data.mix.MixSpec` injects.

        `pools[d]` skips the full-file scan `label_pool` would do — for
        `civil-comments` that is a gigabyte re-read per sampler. `keep[d]`
        is the mixture's seeded row selector, so a source enters at its
        capped share instead of at whatever its first N rows happen to be.
        `quotas[d]` is that source's exact sample budget: the selector
        draws with headroom (`data.mix.DRAW_HEADROOM`) and the rows are
        trimmed here in file order, so the realised share IS the planned
        one instead of a binomial draw around it.

        `space_scoped` names the datasets whose LABEL SPACE is the row's
        own option set instead of the dataset's global pool — see the
        module docstring. Empty by default: naming a dataset here is the
        only way a run's option sets change.
        """
        self.config = config or SamplerConfig()
        self.datasets = assert_trainable(datasets)
        self.space_scoped = {d for d in space_scoped if d in self.datasets}
        #: `dataset -> space key -> (pool, DistractorIndex)`, filled lazily
        #: as spaces are met: 20 000 six-label indexes, not one 99 544-label
        #: index whose per-gold ranking never finishes.
        self._spaces: dict[str, dict] = {d: {} for d in self.datasets}
        #: space name -> the ids it contains. A global pool is named after
        #: its dataset; a space-scoped one gets `dataset#n` in first-seen
        #: order. This is what `cross_space_violations` checks against.
        self.space_members: dict[str, set] = {}
        self._space_names: dict[str, dict] = {d: {} for d in self.datasets}
        self.root = root
        self.rows_per_dataset = rows_per_dataset
        self.keep = dict(keep or {})
        self.quotas = dict(quotas or {})
        #: split-relative index of every row kept, parallel to `self.rows`
        self.row_index: dict[str, list[int]] = {}
        self.pools: dict[str, list[dict]] = {
            d: list((pools or {}).get(d) or label_pool(d, root))
            for d in self.datasets}
        self.pool_ids: dict[str, set[str]] = {
            d: {o["id"] for o in p} for d, p in self.pools.items()}
        for d, ids in self.pool_ids.items():
            if d not in self.space_scoped:
                self.space_members[d] = ids
        self.index: dict[str, DistractorIndex] = {
            d: DistractorIndex(p, self.config) for d, p in self.pools.items()}
        self.rows: dict[str, list[dict]] = {
            d: list(iter_rows(d, self.config.split, rows_per_dataset, root,
                              keep=self._row_keep(d)))
            for d in self.datasets}
        for d in self.datasets:
            if self.quotas.get(d):
                self.trim_to_quota(d, self.quotas[d])

    def trim_to_quota(self, dataset: str, quota: int) -> int:
        """Cut the row list at the last row that fits `quota` samples.

        Row granularity, the same rule `data.mix.realise()` applies, so a
        manifest built from the corpus and a stream built by this loader
        stop on exactly the same member.
        """
        selector = self.keep.get(dataset)
        indices = self.row_index.get(dataset, [])
        kept, cut = 0, 0
        for n, row in enumerate(self.rows[dataset]):
            index = indices[n] if indices else n
            if selector is None:
                selected = len(row.get("questions", []))
            else:
                selected = sum(1 for q in row.get("questions", [])
                               if selector(index, q.get("id", ""),
                                           q.get("answer")))
            if not selected:
                continue
            if kept + selected > quota:
                break
            kept += selected
            cut = n + 1
        self.rows[dataset] = self.rows[dataset][:cut]
        if indices:
            self.row_index[dataset] = indices[:cut]
        return kept

    def selected_questions(self, dataset: str) -> int:
        """How many samples this source really emits per epoch.

        Not `sum(len(row.questions))`: a mixture selector decides per
        QUESTION, and `helpsteer2` ships five per row. The caps count
        samples, so this is the number they must be computed on.
        """
        selector = self.keep.get(dataset)
        rows = self.rows.get(dataset, [])
        if selector is None:
            return sum(len(r.get("questions", [])) for r in rows)
        indices = self.row_index.get(dataset, [])
        return sum(1 for n, row in enumerate(rows)
                   for q in row.get("questions", [])
                   if selector(indices[n], q.get("id", ""), q.get("answer")))

    def filter_rows(self, dataset: str, predicate) -> int:
        """Drop rows failing `predicate`, keeping `row_index` in lockstep.

        `#T-train-real`'s holdout cuts rows after loading; if the parallel
        index were not cut with them, the mixture selector would be asked
        about the wrong row and the realised shares would stop matching the
        manifest. Returns how many rows survived.
        """
        rows = self.rows.get(dataset, [])
        index = self.row_index.get(dataset)
        if not index:
            self.rows[dataset] = [r for r in rows if predicate(r)]
        else:
            pairs = [(i, r) for i, r in zip(index, rows) if predicate(r)]
            self.row_index[dataset] = [i for i, _ in pairs]
            self.rows[dataset] = [r for _, r in pairs]
        return len(self.rows[dataset])

    def _row_keep(self, dataset: str):
        """Adapt a mix selector — which decides per QUESTION — to a row.

        A row survives when any of its questions is selected; questions the
        selector drops are filtered again in `epoch()`, so a multi-question
        source (helpsteer2 ships five) is sampled at the question level and
        still counted once per question by the caps.
        """
        selector = self.keep.get(dataset)
        if selector is None:
            return None
        kept = self.row_index.setdefault(dataset, [])

        def _keep(index: int, row: dict) -> bool:
            if not any(selector(index, q.get("id", ""), q.get("answer"))
                       for q in row.get("questions", [])):
                return False
            kept.append(index)
            return True
        return _keep

    # -- per-question composition (seeded by the row, not by the epoch) --
    def _set_rng(self, dataset: str, row_id: str, qid: str) -> random.Random:
        return random.Random(f"{self.config.seed}\x00set\x00{dataset}"
                             f"\x00{row_id}\x00{qid}")

    def _order_rng(self, epoch: int, dataset: str, row_id: str,
                   qid: str) -> random.Random:
        return random.Random(f"{self.config.seed}\x00ord\x00{epoch}"
                             f"\x00{dataset}\x00{row_id}\x00{qid}")

    def space_of(self, dataset: str, question: dict) -> tuple:
        """`(pool, index)` of the label space one question draws from.

        The dataset's global pool, unless the dataset is space-scoped — in
        which case the space IS the question's own option set, cached on
        it so the rows that share a taxonomy share its ranking.
        """
        if dataset not in self.space_scoped:
            return self.pools[dataset], self.index[dataset]
        options = question.get("options", [])
        key = "\x00".join(sorted(o.get("id", "") for o in options))
        cached = self._spaces[dataset].get(key)
        if cached is None:
            pool = [{"id": o.get("id", ""),
                     "text": o.get("text", o.get("id", ""))}
                    for o in options]
            cached = (pool, DistractorIndex(pool, self.config))
            self._spaces[dataset][key] = cached
            name = f"{dataset}#{len(self._spaces[dataset])}"
            self._space_names[dataset][key] = name
            self.space_members[name] = {o["id"] for o in pool}
        return cached

    def space_name(self, dataset: str, question: dict) -> str:
        """Name of the space a question draws from — `space_members` key.

        A global pool is the dataset itself; a space-scoped question is
        named after the FIRST time its option-id set was met, which is
        stable within a run because the row order is.
        """
        if dataset not in self.space_scoped:
            return dataset
        key = "\x00".join(sorted(o.get("id", "")
                                 for o in question.get("options", [])))
        if key not in self._space_names[dataset]:
            self.space_of(dataset, question)  # registers it
        return self._space_names[dataset][key]

    def spaces_seen(self, dataset: str) -> int:
        """Distinct label spaces this sampler has drawn from so far."""
        if dataset not in self.space_scoped:
            return 1
        return len(self._spaces[dataset])

    def compose(self, dataset: str, row: dict, question: dict,
                row_id: str) -> Sample:
        cfg = self.config
        gold_id = question.get("answer")
        pool, index = self.space_of(dataset, question)
        by_id = index.text
        rng = self._set_rng(dataset, row_id, question.get("id", ""))

        k_wanted = cfg.draw_k(rng, len(pool))
        # A row whose own answer is absent or already `unknown` stays an
        # unknown row; the draw is consumed either way so the stream of a
        # row does not depend on its gold.
        gold_missing = gold_id is None or gold_id == UNKNOWN_ID
        is_unknown = rng.random() < cfg.unknown_fraction or gold_missing

        hard, easy = index.buckets("" if gold_missing else gold_id)
        n_available = len(hard) + len(easy)
        n_distract = min(k_wanted if is_unknown else k_wanted - 1, n_available)
        n_hard = min(len(hard), int(round(cfg.hard_fraction * n_distract)))
        n_easy = n_distract - n_hard
        if n_easy > len(easy):  # tiny pool: spill the remainder back to hard
            n_hard = min(len(hard), n_hard + (n_easy - len(easy)))
            n_easy = n_distract - n_hard

        picked = rng.sample(hard, n_hard) + rng.sample(easy, n_easy)
        options = [{"id": i, "text": by_id.get(i, i)} for i in picked]
        if not is_unknown:
            options.append({"id": gold_id, "text": by_id.get(gold_id, gold_id)})

        return Sample(
            dataset=dataset, row_id=row_id,
            question_id=question.get("id", ""),
            state=row["state"], question=question_text(question),
            options=options,
            answer=UNKNOWN_ID if is_unknown else gold_id,
            gold_index=len(options) if is_unknown else len(options) - 1,
            dropped_gold=(gold_id if is_unknown and not gold_missing
                          else None),
            n_hard=n_hard, n_easy=n_easy,
            space=self.space_name(dataset, question),
            space_size=len(pool))

    def epoch(self, epoch: int = 0):
        """Yield every sample of one epoch, options reshuffled for `epoch`."""
        for dataset in self.datasets:
            selector = self.keep.get(dataset)
            indices = self.row_index.get(dataset, [])
            for n, row in enumerate(self.rows[dataset]):
                row_id = f"{dataset}-{n}"
                for question in row.get("questions", []):
                    if selector is not None and not selector(
                            indices[n], question.get("id", ""),
                            question.get("answer")):
                        continue
                    sample = self.compose(dataset, row, question, row_id)
                    self._shuffle(sample, epoch)
                    yield sample

    def _shuffle(self, sample: Sample, epoch: int) -> Sample:
        """Reshuffle the option order for this epoch; keep `gold_index`."""
        rng = self._order_rng(epoch, sample.dataset, sample.row_id,
                              sample.question_id)
        gold = (None if sample.is_unknown
                else sample.options[sample.gold_index]["id"])
        rng.shuffle(sample.options)
        sample.gold_index = (len(sample.options) if gold is None
                             else sample.option_ids().index(gold))
        sample.epoch = epoch
        return sample

    def batches(self, epoch: int = 0, batch_size: int = 16):
        """Variable-K samples grouped into batches; nothing is padded."""
        batch: list[Sample] = []
        for sample in self.epoch(epoch):
            batch.append(sample)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # -- checks the gate and the tests share ----------------------------
    def global_space_violations(self, samples: list[Sample]) -> list[dict]:
        """Samples whose option set IS the dataset's global label space.

        This is the HuffPost 41-category bug. Datasets whose pool is binary
        (boolq: yes/no) cannot satisfy it and are exempt — see
        `binary_exempt_datasets()`; every other dataset must be empty here.

        A space-scoped dataset is exempt for a different reason and it is
        named as such: its option set IS its label space by construction,
        so the check would compare a six-label space against a 99 544-label
        union and pass vacuously. What must hold there is the stronger
        claim `test_optset` makes instead — no option ever comes from
        another space.
        """
        # The full-space regime offers the row's whole space ON PURPOSE
        # (#T-bigk-optsets): this check is not relaxed there, it is
        # REPLACED by `cross_space_violations`, which is the strictly
        # stronger claim — no option ever comes from another space, and no
        # sample from a dataset this sampler was not built on.
        if self.config.full_space:
            return []
        exempt = set(self.fixed_option_datasets()) | self.space_scoped
        out = []
        for s in samples:
            pool = self.pool_ids[s.dataset]
            if s.dataset in exempt:
                continue
            if set(s.option_ids()) == pool:
                out.append({"dataset": s.dataset, "row_id": s.row_id,
                            "k": s.k, "pool_size": len(pool)})
        return out

    def cross_space_violations(self, samples: list[Sample]) -> list[dict]:
        """The firewall claim: no option crosses a space or a dataset.

        Widening the option set is exactly the change that could smuggle a
        label in, so the proof has to be per sample and not per config:

        1. the sample's dataset is one this sampler was built on — and
           every one of those went through `assert_trainable` (registry +
           `data.firewall.check_job_allowed`), so a fenced or eval-only
           corpus cannot be the source of an option;
        2. every option id belongs to the space the row itself drew from,
           named on the sample. For a space-scoped dataset that is the
           row's own taxonomy, not the union of 20 000 of them.

        Empty list = no violation. This is the check the full-space gate
        passes on, in place of `global_space_violations`.
        """
        out = []
        for s in samples:
            if s.dataset not in self.datasets:
                out.append({"why": "sample from a dataset this sampler was "
                                   "not built on",
                            "dataset": s.dataset, "row_id": s.row_id,
                            "allowed_datasets": list(self.datasets)})
                continue
            allowed = self.space_members.get(s.space)
            if allowed is None:
                out.append({"why": "sample names a space that was never "
                                   "registered",
                            "dataset": s.dataset, "row_id": s.row_id,
                            "space": s.space})
                continue
            stray = sorted(set(s.option_ids()) - allowed)
            if stray:
                out.append({"why": "option from outside the row's space",
                            "dataset": s.dataset, "row_id": s.row_id,
                            "space": s.space, "space_size": len(allowed),
                            "stray_options": stray[:5]})
        return out

    def foreign_label_violations(self, samples: list[Sample],
                                 forbidden: dict) -> list[dict]:
        """Options that match a label of a corpus this run must not touch.

        `forbidden` is `{dataset: {label ids}}` — for a `--fence-clean`
        run, the label spaces of `data.mix.MIX_1M_FENCED`. An id shared by
        a fenced dataset and a trained one is reported with both names so
        the reader can see it is a collision and not a leak.
        """
        out = []
        for s in samples:
            own = self.space_members.get(s.space) or set()
            for name, ids in forbidden.items():
                if name in self.datasets:
                    continue
                hit = sorted(set(s.option_ids()) & set(ids))
                if hit:
                    out.append({"dataset": s.dataset, "row_id": s.row_id,
                                "space": s.space, "fenced_dataset": name,
                                "labels": hit[:5],
                                "also_in_own_space": sorted(
                                    set(hit) & own)[:5]})
        return out

    def binary_exempt_datasets(self) -> list[str]:
        return [d for d in self.datasets
                if len(self.pool_ids[d]) <= BINARY_POOL_SIZE]

    def fixed_option_datasets(self) -> list[str]:
        """Sources whose option set IS their label space, by construction.

        Two kinds, both declared rather than inferred: a binary pool
        (boolq's yes/no) and an ordinal scale (`helpsteer2`'s score 0-4,
        registered with `dynamic_options=False` in `data.mix`). Sampling a
        subset of a scale would not make it dynamic, it would make it wrong,
        so these are exempt from the global-label-space check — and their
        share of the mixture is exactly what the 15 % cap exists to bound.
        """
        return [d for d in self.datasets
                if len(self.pool_ids[d]) <= BINARY_POOL_SIZE
                or (d in MIX_SOURCES and not MIX_SOURCES[d].dynamic_options)]

    def composition(self, samples: list[Sample]) -> dict:
        """Counted composition: K distribution, % hard, % unknown."""
        per: dict[str, dict] = {}
        for s in samples:
            d = per.setdefault(s.dataset, {
                "samples": 0, "unknown": 0, "hard": 0, "easy": 0,
                "k_distribution": {}, "pool_size": len(self.pool_ids[s.dataset]),
                "rows": len(self.rows[s.dataset]),
                "binary_pool": len(self.pool_ids[s.dataset]) <= BINARY_POOL_SIZE,
            })
            d["samples"] += 1
            d["unknown"] += int(s.is_unknown)
            d["hard"] += s.n_hard
            d["easy"] += s.n_easy
            d["k_sum"] = d.get("k_sum", 0) + s.k
            d["spaces"] = d.get("spaces", set()) | {s.space}
            # `unknown` rows offer the space MINUS the withheld gold, so
            # "K is the whole space" is |space| - 1 for them. Both count
            # as full cardinality; the offset is the gold, not a cap.
            target = s.space_size - (1 if s.is_unknown else 0)
            d["k_is_space"] = d.get("k_is_space", 0) + int(
                s.space_size > 0 and s.k >= target)
            d["k_distribution"][str(s.k)] = \
                d["k_distribution"].get(str(s.k), 0) + 1
        for d in per.values():
            distract = d["hard"] + d["easy"]
            d["mean_k"] = round(d.pop("k_sum", 0) / max(d["samples"], 1), 4)
            d["spaces_seen"] = len(d.pop("spaces", ()) or ())
            d["pct_k_is_full_space"] = round(
                100.0 * d.pop("k_is_space", 0) / max(d["samples"], 1), 4)
            d["distractors"] = distract
            d["pct_hard_negatives"] = round(
                100.0 * d["hard"] / distract, 4) if distract else 0.0
            d["pct_unknown"] = round(100.0 * d["unknown"] / d["samples"], 4)
            d["max_k"] = max(int(k) for k in d["k_distribution"])
            d["k_distribution"] = dict(
                sorted(d["k_distribution"].items(), key=lambda t: int(t[0])))
        total_hard = sum(d["hard"] for d in per.values())
        total_distract = sum(d["distractors"] for d in per.values())
        total_unknown = sum(d["unknown"] for d in per.values())
        overall_k: dict[str, int] = {}
        for d in per.values():
            for k, n in d["k_distribution"].items():
                overall_k[k] = overall_k.get(k, 0) + n
        return {
            "samples": len(samples),
            "regime": self.config.regime(),
            "mean_k": round(sum(s.k for s in samples) / len(samples), 4)
            if samples else 0.0,
            "pct_k_is_full_space": round(
                100.0 * sum(1 for s in samples if s.space_size > 0
                            and s.k >= s.space_size - (1 if s.is_unknown
                                                       else 0))
                / len(samples), 4) if samples else 0.0,
            "per_dataset": dict(sorted(per.items())),
            "k_distribution": dict(
                sorted(overall_k.items(), key=lambda t: int(t[0]))),
            "pct_hard_negatives": round(
                100.0 * total_hard / total_distract, 4) if total_distract else 0.0,
            "pct_unknown": round(100.0 * total_unknown / len(samples), 4)
            if samples else 0.0,
            "target_hard_fraction": self.config.hard_fraction,
            "target_unknown_fraction": self.config.unknown_fraction,
        }


# -- chi-square (stdlib only) ---------------------------------------------

def _gammainc_upper(s: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(s, x); Numerical Recipes."""
    if x < 0 or s <= 0:
        raise ValueError("gammainc_upper needs s > 0 and x >= 0")
    if x == 0:
        return 1.0
    if x < s + 1.0:  # series for P(s, x), then Q = 1 - P
        term = 1.0 / s
        total = term
        n = s
        for _ in range(1000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return 1.0 - total * math.exp(-x + s * math.log(x) - math.lgamma(s))
    tiny = 1e-300  # continued fraction for Q(s, x)
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h * math.exp(-x + s * math.log(x) - math.lgamma(s))


def chi2_sf(chi2: float, df: int) -> float:
    """P(X > chi2) for X ~ chi-square with `df` degrees of freedom."""
    if df <= 0:
        return 1.0
    return _gammainc_upper(df / 2.0, chi2 / 2.0)


def gold_position_chi2(samples: list[Sample]) -> dict:
    """Is the gold uniform over positions? Chi-square, per K and pooled.

    A prior on the gold position is what let LogiQA score 0.435 off a
    positional shortcut. Answerable samples only (an `unknown` row has no
    gold position). Grouped by K because the uniform expectation is 1/K.
    """
    by_k: dict[int, list[int]] = {}
    for s in samples:
        if s.is_unknown:
            continue
        by_k.setdefault(s.k, [0] * s.k)
        by_k[s.k][s.gold_index] += 1
    per_k, pooled_chi2, pooled_df = {}, 0.0, 0
    for k in sorted(by_k):
        counts = by_k[k]
        n = sum(counts)
        expected = n / k
        chi2 = sum((c - expected) ** 2 / expected for c in counts) \
            if expected > 0 else 0.0
        per_k[str(k)] = {"n": n, "counts": counts, "chi2": round(chi2, 4),
                         "df": k - 1, "p": round(chi2_sf(chi2, k - 1), 6),
                         "min_expected": round(expected, 2)}
        pooled_chi2 += chi2
        pooled_df += k - 1
    return {"n_samples": sum(v["n"] for v in per_k.values()),
            "per_k": per_k, "pooled_chi2": round(pooled_chi2, 4),
            "pooled_df": pooled_df,
            "pooled_p": round(chi2_sf(pooled_chi2, pooled_df), 6)}


def epoch_shuffle_report(sampler: OptionSetSampler, epochs: tuple[int, int],
                         limit: int = 2000) -> dict:
    """Same rows, two epochs: same option SET, different option ORDER."""
    a, b = [], []
    for n, s in enumerate(sampler.epoch(epochs[0])):
        if n >= limit:
            break
        a.append(s)
    for n, s in enumerate(sampler.epoch(epochs[1])):
        if n >= limit:
            break
        b.append(s)
    shufflable = same_set = reordered = gold_moved = 0
    for x, y in zip(a, b):
        if sorted(x.option_ids()) == sorted(y.option_ids()):
            same_set += 1
        if x.k < 2:
            continue
        shufflable += 1
        if x.option_ids() != y.option_ids():
            reordered += 1
        if not x.is_unknown and x.gold_index != y.gold_index:
            gold_moved += 1
    return {"compared": len(a), "epochs": list(epochs),
            "set_stable": same_set == len(a), "same_set": same_set,
            "shufflable": shufflable, "reordered": reordered,
            "reordered_rate": round(reordered / shufflable, 4)
            if shufflable else 0.0,
            "gold_moved": gold_moved}


def firewall_report() -> dict:
    """Proof that the blocked corpora are refused, not merely absent."""
    blocked = {}
    for name in ("synth-loop", "logiqa", "reclor"):
        try:
            assert_trainable([name])
        except ValueError as exc:
            blocked[name] = str(exc)[:120]
        else:  # pragma: no cover - a pass here is a firewall regression
            blocked[name] = ""
    return {"allowed": list(ALLOWED_DATASETS),
            "trainable": list(TRAINABLE_DATASETS),
            "blocked": blocked,
            "all_blocked": all(bool(v) for v in blocked.values())}


# -- gate ------------------------------------------------------------------

MIN_CHI2_SAMPLES = 10_000


def run_gate(rows_per_dataset: int = 4000, config: SamplerConfig | None = None,
             write: bool = True, root: str = PREFETCH_DIR) -> dict:
    """Run the three contract checks and write `gate.json`."""
    t0 = time.perf_counter()
    sampler = OptionSetSampler(config=config, rows_per_dataset=rows_per_dataset,
                               root=root)
    samples = list(sampler.epoch(0))
    composition = sampler.composition(samples)
    violations = sampler.global_space_violations(samples)
    crossings = sampler.cross_space_violations(samples)
    fenced = sampler.foreign_label_violations(
        samples, {d: {o["id"] for o in label_pool(d, root)}
                  for d in MIX_1M_FENCED
                  if d not in sampler.datasets
                  and os.path.exists(dataset_path(d, root))})
    positions = gold_position_chi2(samples)
    shuffle = epoch_shuffle_report(sampler, (0, 1))
    firewall = firewall_report()

    counted_unknown = sum(1 for s in samples if s.is_unknown)
    counted_hard = sum(s.n_hard for s in samples)
    counted_easy = sum(s.n_easy for s in samples)
    mix_tolerance = 2.0  # percentage points against the configured target
    unknown_ok = abs(composition["pct_unknown"]
                     - 100.0 * sampler.config.unknown_fraction) <= mix_tolerance
    hard_ok = counted_hard > 0 and counted_easy > 0

    full = sampler.config.full_space
    checks = {
        "cross_space_firewall": {
            # The check that decides the full-space gate, and a second
            # belt in the sampled regime. Widening the option set is the
            # change that could smuggle a label in; this is the proof it
            # did not.
            "pass": not crossings and not fenced,
            "crossings": len(crossings),
            "examples": crossings[:5],
            "fenced_label_hits": len(fenced),
            "fenced_examples": fenced[:5],
            "fenced_datasets_checked": [d for d in MIX_1M_FENCED
                                        if d not in sampler.datasets],
            "criterion": ("every option id belongs to the space the row "
                          "itself drew from, every sample comes from a "
                          "dataset that passed `assert_trainable`, and no "
                          "option matches a label of a fenced corpus"),
            "spaces_registered": len(sampler.space_members),
        },
        "no_global_label_space": {
            "pass": not violations,
            "applies": not full,
            "why_not": ("the full-space regime offers the row's whole "
                        "space on purpose; `cross_space_firewall` is the "
                        "check that replaces it — strictly stronger"
                        if full else ""),
            "violations": len(violations),
            "examples": violations[:5],
            "exempt_binary_datasets": sampler.binary_exempt_datasets(),
            "exempt_datasets": sampler.fixed_option_datasets(),
            "exempt_reason": (
                "a pool of 2 (boolq yes/no) or an ordinal scale IS the "
                "question: its option set cannot differ from its global "
                "label space"),
            "pool_sizes": {d: len(ids)
                           for d, ids in sorted(sampler.pool_ids.items())},
        },
        "gold_position_uniform": {
            "pass": (positions["n_samples"] >= MIN_CHI2_SAMPLES
                     and positions["pooled_p"] > 0.01
                     and all(v["p"] > 0.01 for v in positions["per_k"].values())),
            "min_samples": MIN_CHI2_SAMPLES,
            "alpha": 0.01,
            **positions,
        },
        "mix_counted": {
            "pass": unknown_ok and hard_ok,
            "counted_unknown_rows": counted_unknown,
            "counted_hard_distractors": counted_hard,
            "counted_easy_distractors": counted_easy,
            "pct_unknown": composition["pct_unknown"],
            "pct_hard_negatives": composition["pct_hard_negatives"],
            "target_unknown_pct": 100.0 * sampler.config.unknown_fraction,
            "target_hard_pct": 100.0 * sampler.config.hard_fraction,
            "tolerance_pp": mix_tolerance,
            "note": ("hard/easy is capped by the pool: boolq has 1 possible "
                     "distractor, so its mix is counted, not targeted"),
        },
        "epoch_shuffle": {
            "pass": shuffle["set_stable"] and shuffle["reordered_rate"] >= 0.5,
            **shuffle,
        },
        "firewall": {"pass": firewall["all_blocked"], **firewall},
    }
    gate = {
        "task": "T-optset-sampler",
        "pass": all(c["pass"] for c in checks.values()),
        "config": asdict(sampler.config),
        "cardinality_regime": sampler.config.regime(),
        "mean_k": composition["mean_k"],
        "pct_k_is_full_space": composition["pct_k_is_full_space"],
        "datasets": sampler.datasets,
        "rows_per_dataset": rows_per_dataset,
        "samples": composition["samples"],
        "k_distribution": composition["k_distribution"],
        "pct_hard_negatives": composition["pct_hard_negatives"],
        "pct_unknown": composition["pct_unknown"],
        "per_dataset": composition["per_dataset"],
        "checks": checks,
        "contract": ("Sample -> DecisionEngine.logits(mem, question, "
                     "options); gold_index == K means unknown"),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if write:
        os.makedirs(GATE_DIR, exist_ok=True)
        with open(os.path.join(GATE_DIR, "gate.json"), "w") as fh:
            json.dump(gate, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return gate


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "gate"
    if cmd not in ("gate",):
        print(f"unknown command {cmd!r}; use gate", file=sys.stderr)
        return 2
    rows = int(argv[2]) if len(argv) > 2 else 4000
    config = None
    if "--full-space" in argv:
        config = SamplerConfig(k_min=FULL_SPACE, k_max=FULL_SPACE)
    gate = run_gate(rows_per_dataset=rows, config=config)
    print(json.dumps({k: v for k, v in gate.items()
                      if k not in ("checks", "per_dataset")},
                     indent=2, sort_keys=True))
    for name, check in gate["checks"].items():
        print(f"[gate] {name}: {'PASS' if check['pass'] else 'FAIL'}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
