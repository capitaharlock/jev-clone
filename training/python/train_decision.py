"""Listwise training of the pointer decision head (#T-train-real).

This replaces `data/train_baseline.py` on the product path. The baseline
learned `state -> clf.classes_`: a fixed global label space, one pickle
per dataset, the question and the option TEXTS thrown away. Here the
output space of a row IS that row's option set, and there is exactly one
model over the whole clean mixture.

The pieces are the upstream ones, used as they are:

* `model.decision_head.PointerDecisionHead` (#T-pointer-head) is the only
  scorer — `K + 1` logits per row, the last one the learned `unknown`.
  No `num_labels x d` matrix exists anywhere; `label_free_report()` is a
  gate check here too.
* `data.optset.OptionSetSampler.batches()` (#T-optset-sampler) is the only
  dataloader — variable K, counted hard negatives, `unknown` rows, option
  order reshuffled per epoch, `synth-loop`/`logiqa`/`reclor` refused by
  the firewall.
* `model.encoder` (#T-torch-stack) is the only backbone, loaded from
  sha256-verified bytes. It stays FROZEN: the trainable surface is the
  2.7 M-parameter head. That is what makes ~1 M samples fit a Mac, and it
  is also why an unseen option text is scored at all — its embedding
  comes from a backbone that was never fitted to this label space.

Loss
----
Listwise cross-entropy over the row's `[K + 1]` logits against
`Sample.gold_index` (which is `K` for an `unknown` row). Never a global
multiclass head. Brier and ECE are logged from step 1, not bolted on at
the end.

Unseen labels
-------------
The metric that decides this project is accuracy on labels whose TEXT
never appeared in training (#T-unseen-labels, finding F). So the trainer
itself carves the label space first:

* a per-dataset holdout of sibling labels (chosen by the same
  `data.optset.difficulty` ranking that builds hard negatives, so the
  holdout is the hard half, not the easy half);
* held-out labels are removed from the training pool AND every row whose
  gold is one of them is removed from training, so the text is absent
  from train both as a gold and as a distractor;
* the unseen eval set offers option sets drawn ONLY from held-out labels,
  so the model cannot win by elimination. Chance is `mean(1 / (K + 1))`.

`boolq` has a two-label pool (yes/no): a holdout is impossible by
construction, so it trains and is reported as `seen`-only, explicitly.

Gate criteria — written before the first measurement
----------------------------------------------------
1. `unseen_beats_chance`: the Wilson 95 % lower bound of unseen accuracy
   is strictly above the mean chance rate of the unseen eval set.
   If it is not, the gate is `pass: false` and the run does NOT scale.
2. `holdout_clean`: no held-out label text (normalised) appears in any
   training sample, as gold or as distractor.
3. `label_free`: the trained head still contains no label-space parameter.
4. `checkpoint_cold_load`: a fresh process loads the safetensors
   checkpoint and answers a real decision with p95 < 500 ms.
5. `manifest_model_version`: the checkpoint manifest carries the
   `model_version` + `tokenizer_hash` pair that `crates/jev-runtime`
   composes its state-cache key from.

Artifacts
---------
* `artifacts/runs/<run_id>/metrics.jsonl` — one line per step, per eval
  and per stage. The `:8794` dashboard headlines the unseen number.
* `artifacts/checkpoints/decision/<run_id>/stage-<samples>/` —
  `model.safetensors` + `tokenizer.json` + `manifest.json`.
* `artifacts/gates/T-train-real/gate.json` — rewritten at every stage
  from the real numbers of that stage.

CLI
---
    .venv-train/bin/python -m training.python.train_decision train \
        --max-samples 1000000
    .venv-train/bin/python -m training.python.train_decision gate \
        --checkpoint artifacts/checkpoints/decision/<run>/stage-<n>
    .venv-train/bin/python -m training.python.train_decision holdout
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field, replace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:  # the trainer runs both as -m and as a script
    sys.path.insert(0, ROOT)

from data import mix as mixmod  # noqa: E402
from data.optset import (ALLOWED_DATASETS, DistractorIndex,  # noqa: E402
                         OptionSetSampler, Sample, SamplerConfig, UNKNOWN_ID,
                         difficulty)

RUNS_DIR = os.path.join(ROOT, "artifacts", "runs")
CKPT_DIR = os.path.join(ROOT, "artifacts", "checkpoints", "decision")
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", "T-train-real")
PREFETCH_DIR = os.path.join(ROOT, "artifacts", "data-prefetch")

DEFAULT_SEED = 1789
#: training truncation; inference keeps `encoder.DEFAULT_MAX_LENGTH`
TRAIN_MAX_LENGTH = 256
#: MPS recompiles on every new shape — bucket T to keep the variant count low
PAD_MULTIPLE = 32
#: the 250 k -> 1 M scale curve the operator decides on, plus two early points
DEFAULT_STAGES = (62_500, 125_000, 250_000, 500_000, 1_000_000)
#: rolling window (samples) for ECE, so a bin estimate exists from step 1
ECE_WINDOW = 2048
ECE_BINS = 15
#: the eval split of each dataset; boolq and helpsteer2 ship `calibration`
EVAL_SPLIT = {"banking77": "test", "massive": "test", "huffpost": "test",
              "boolq": "calibration", "helpsteer2": "calibration",
              "civil-comments": "test", "email-triage": "test",
              "synth-v1": "test", "prog-gold": "test"}
#: rows an eval sampler loads per dataset. Every P0 eval split is smaller
#: than this; it exists so `civil-comments` (97 320 test rows) cannot pull
#: a hundred thousand rows into memory to answer 3 000 eval questions.
EVAL_ROWS_CAP = 20_000
#: share of each label pool held out as "never seen in training"
UNSEEN_LABEL_FRACTION = {"huffpost": 11 / 41, "banking77": 0.25,
                         "massive": 0.25, "boolq": 0.0}
LATENCY_BUDGET_MS = 500.0

_QID_TAIL = re.compile(r"-\d+$")


# -- small numeric helpers -------------------------------------------------

def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple:
    """95 % Wilson score interval — the honest CI for a small unseen set."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def expected_calibration_error(pairs: list, bins: int = ECE_BINS) -> float:
    """ECE over (confidence, correct) pairs, equal-width bins."""
    if not pairs:
        return 0.0
    buckets = [[0, 0.0, 0.0] for _ in range(bins)]
    for conf, correct in pairs:
        b = min(bins - 1, int(conf * bins))
        buckets[b][0] += 1
        buckets[b][1] += conf
        buckets[b][2] += float(correct)
    n = len(pairs)
    return sum(cnt / n * abs(acc / cnt - cf / cnt)
               for cnt, cf, acc in buckets if cnt)


def brier_score(probs: list, gold: int) -> float:
    """Multiclass Brier over the row's `K + 1` outcomes (range 0..2)."""
    return sum((p - (1.0 if i == gold else 0.0)) ** 2
               for i, p in enumerate(probs))


def canonical_question(text: str) -> str:
    """`banking77-intent-8412` -> `banking77-intent`.

    The converted corpus has no question prose: `question_text()` falls
    back to the question id, which for banking77/massive carries the ROW
    INDEX. Feeding that to the encoder is both noise and a memorisation
    handle (a per-row unique token the head could key on). Stripping the
    trailing index leaves one stable question token per dataset — and
    makes the embedding cache hit every time.
    """
    return _QID_TAIL.sub("", text) if text else text


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalise_label(text: str) -> str:
    """Comparison form for the holdout-cleanliness check (text, not id)."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


class MixShortfallError(RuntimeError):
    """The mixture is smaller than the budget the run was asked to train.

    `--max-samples` is DECISIONS SEEN and the mixture is DECISIONS THAT
    EXIST; nothing used to connect the two, so a run whose corpus was
    39 981 rows and whose budget was 1 000 000 quietly took 25 epochs over
    it and reported a 1 M-decision curve. Twenty-five passes over 4 % of a
    corpus is not the same measurement as one pass over all of it, and the
    difference does not show up in any metric the run publishes.

    So a shortfall is now an abort. `--allow-repeat` is the way to say
    "yes, repeat" out loud, and it writes `epochs_over_corpus` into
    `run.json`, `metrics.jsonl` and the run's `mix.json` so the repetition
    is a published number rather than an absence.
    """


# -- label holdout ---------------------------------------------------------

@dataclass
class DatasetHoldout:
    dataset: str
    seen: list = field(default_factory=list)
    unseen: list = field(default_factory=list)
    texts: dict = field(default_factory=dict)
    exempt: str | None = None

    def to_dict(self) -> dict:
        return {"dataset": self.dataset, "n_seen": len(self.seen),
                "n_unseen": len(self.unseen), "seen": self.seen,
                "unseen": self.unseen, "exempt": self.exempt}


def sibling_holdout(pool: list, fraction: float, cos_weight: float = 0.5
                    ) -> tuple:
    """Hold out whole sibling PAIRS, hardest first.

    `#T-unseen-labels` asks for a banking77 holdout of *sibling* intents —
    the pairs `#T-optset-sampler` mines as hard negatives — so that an
    unseen number separates "generalises" from "got the easy half". The
    same rule is applied to every multi-label pool: rank every unordered
    pair by `data.optset.difficulty` and take the hardest pairs until the
    quota is met. Deterministic: no rng, ties broken by id.
    """
    ids = sorted(o["id"] for o in pool)
    text = {o["id"]: o["text"] for o in pool}
    quota = int(round(len(ids) * fraction))
    if quota < 2 or len(ids) - quota < 2:
        return ids, []
    scored = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            scored.append((difficulty(text[a], text[b], cos_weight), a, b))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    unseen: list = []
    taken: set = set()
    for _, a, b in scored:
        if len(unseen) + 2 > quota:
            break
        if a in taken or b in taken:
            continue
        taken.update((a, b))
        unseen.extend((a, b))
    # An odd quota leaves one slot: fill it with the hardest single label
    # not already held out, so the quota is met exactly.
    if len(unseen) < quota:
        for _, a, b in scored:
            cand = a if a not in taken else (b if b not in taken else None)
            if cand is None:
                continue
            taken.add(cand)
            unseen.append(cand)
            if len(unseen) >= quota:
                break
    unseen = sorted(unseen)
    return [i for i in ids if i not in taken], unseen


def build_holdout(datasets=ALLOWED_DATASETS, root: str = PREFETCH_DIR,
                  fractions: dict | None = None,
                  pools: dict | None = None) -> dict:
    """Per-dataset seen/unseen label split. Deterministic, no rng.

    `pools` comes from `data.mix.scan_supply`, which already walked every
    shard: without it this re-reads a gigabyte of `civil-comments` to
    rediscover that its label space is {toxic, not_toxic}.
    """
    from data.optset import label_pool

    fractions = fractions or UNSEEN_LABEL_FRACTION
    out: dict = {}
    for name in datasets:
        pool = (pools or {}).get(name) or label_pool(name, root)
        frac = fractions.get(name, 0.0)
        if len(pool) <= 2 or frac <= 0.0:
            out[name] = DatasetHoldout(
                name, sorted(o["id"] for o in pool), [],
                {o["id"]: o["text"] for o in pool},
                exempt=("a two-label pool (yes/no) IS the question: no "
                        "label can be held out without deleting the task"
                        if len(pool) <= 2 else "holdout disabled"))
            continue
        seen, unseen = sibling_holdout(pool, frac)
        out[name] = DatasetHoldout(name, seen, unseen,
                                   {o["id"]: o["text"] for o in pool})
    return out


def holdout_report(holdout: dict) -> dict:
    return {"per_dataset": {d: h.to_dict() for d, h in sorted(holdout.items())},
            "rule": ("sibling pairs ranked by data.optset.difficulty "
                     "(char-3gram cosine blended with word-Jaccard), "
                     "hardest first — the holdout is the hard half"),
            "fractions": dict(sorted(UNSEEN_LABEL_FRACTION.items()))}


# -- samplers --------------------------------------------------------------

def restrict(sampler: OptionSetSampler, keep: set) -> OptionSetSampler:
    """Narrow a single-dataset sampler to `keep` labels, in place.

    Both the POOL (so `keep` is the only text a distractor can come from)
    and the ROWS (so no row with an out-of-`keep` gold survives) are cut.
    `DistractorIndex` is rebuilt because its ranking is pool-relative.
    """
    dataset = sampler.datasets[0]
    sampler.pools[dataset] = [o for o in sampler.pools[dataset]
                              if o["id"] in keep]
    sampler.pool_ids[dataset] = {o["id"] for o in sampler.pools[dataset]}
    sampler.index[dataset] = DistractorIndex(sampler.pools[dataset],
                                             sampler.config)
    sampler.filter_rows(
        dataset,
        lambda r: bool(r.get("questions")) and all(q.get("answer") in keep
                                                   for q in r["questions"]))
    return sampler


def make_sampler(dataset: str, keep: set, split: str, config: SamplerConfig,
                 rows: int | None = None, root: str = PREFETCH_DIR,
                 pools: dict | None = None, mix_keep=None,
                 quota: int | None = None):
    """One restricted, firewall-checked sampler for one dataset.

    `mix_keep` is `data.mix.MixSpec.keep(dataset)` — the seeded selector
    that brings the source in at its CAPPED share. Without it a source
    enters at its first `rows` rows, which for `civil-comments` is 1.8 M and
    for everyone else is whatever the file happens to start with.
    """
    cfg = SamplerConfig(**{**asdict(config), "split": split})
    sampler = OptionSetSampler(datasets=(dataset,), config=cfg,
                               rows_per_dataset=rows, root=root,
                               pools=pools,
                               keep={dataset: mix_keep} if mix_keep else None,
                               quotas={dataset: quota} if quota else None)
    return restrict(sampler, keep)


class MixtureStream:
    """Round-robin over the per-dataset samplers: ONE model, one mixture.

    Each dataset keeps its own `OptionSetSampler.batches()` — which is what
    makes a batch length-homogeneous (banking77 states are 15 tokens,
    boolq passages are 200) — and the datasets alternate at batch
    granularity, so the optimiser never sees 163 k huffpost rows in a row.
    The dataset with the largest remaining share goes next: deterministic,
    and every dataset finishes its epoch at the same time.
    """

    def __init__(self, samplers: dict, batch_size: int = 32,
                 verify: bool = True, dataset_cap: float | None = None,
                 family_cap: float | None = None):
        """`verify` runs the corpus guardrails on the REALISED mixture.

        Not on the plan: the unseen-label holdout cuts rows after the
        quotas are computed, so the only composition worth checking is the
        one these samplers will actually emit. A breach raises
        `data.mix.MixGuardrailError` and the run stops — finding E existed
        because a 74.6 % share was a number in a report instead of an
        error in a pipeline.

        `dataset_cap`/`family_cap` are for an EXPERIMENT that has to state
        a different cap out loud — the §128 synthetic ablation cannot be
        built at 15 % because dropping `synth-v1` leaves the fenced
        registry covering only 90 % of a mixture. They default to the
        §§65-66 values, they are recorded in `run.json`, and they still
        abort the run when the realised mixture breaches them.
        """
        self.samplers = samplers
        self.batch_size = batch_size
        self.dataset_cap = (mixmod.MAX_DATASET_FRACTION if dataset_cap is None
                            else dataset_cap)
        self.family_cap = (mixmod.MAX_FAMILY_FRACTION if family_cap is None
                           else family_cap)
        self.sizes = {d: s.selected_questions(d) for d, s in samplers.items()}
        self.guardrails = (mixmod.verify_mix(self.sizes, None, self.dataset_cap,
                                             self.family_cap)
                           if verify and self.sizes else None)

    def total(self) -> int:
        return sum(self.sizes.values())

    def epoch(self, epoch: int = 0):
        gens = {d: s.batches(epoch, self.batch_size)
                for d, s in self.samplers.items()}
        served = {d: 0 for d in gens}
        while gens:
            order = sorted(
                gens, key=lambda d: (-(1.0 - served[d] / max(self.sizes[d], 1)), d))
            served_any = False
            for d in order:
                try:
                    batch = next(gens[d])
                except StopIteration:
                    gens.pop(d, None)
                    continue
                served[d] += len(batch)
                served_any = True
                yield batch
                break
            if not served_any and not gens:
                return

    def stream(self, epochs: int = 1_000_000):
        for e in range(epochs):
            yielded = False
            for batch in self.epoch(e):
                yielded = True
                yield e, batch
            if not yielded:
                return


def train_samplers(holdout: dict, config: SamplerConfig,
                   rows: int | None = None, root: str = PREFETCH_DIR,
                   spec=None, pools: dict | None = None) -> dict:
    """One sampler per source of the MIXTURE, not per file on disk.

    With a `data.mix.MixSpec` the datasets, their row budgets and their
    seeded selectors all come from the descriptor, so what trains is what
    the manifest says trains. Without one this keeps the old behaviour:
    every dataset with a seen-label pool, whole, in file order.
    """
    datasets = spec.datasets if spec is not None else list(holdout)
    out = {}
    for d in datasets:
        h = holdout.get(d)
        if h is None or not h.seen:
            continue
        mix_keep = spec.keep(d) if spec is not None else None
        quota = spec.quotas.get(d) if spec is not None else None
        out[d] = make_sampler(d, set(h.seen), "train", config, rows, root,
                              pools=pools, mix_keep=mix_keep, quota=quota)
    return out


def eval_samplers(holdout: dict, which: str, config: SamplerConfig,
                  rows: int | None = None, root: str = PREFETCH_DIR,
                  pools: dict | None = None) -> dict:
    """`seen` or `unseen` eval samplers over each dataset's eval split.

    Eval never emits `unknown` rows: the question asked here is "does it
    point at the right option", and an `unknown` row has no gold position
    to point at. The abstention rate is reported separately.
    """
    cfg = SamplerConfig(**{**asdict(config), "unknown_fraction": 0.0})
    out = {}
    for d, h in holdout.items():
        keep = set(h.seen if which == "seen" else h.unseen)
        if len(keep) < 2:
            continue
        out[d] = make_sampler(d, keep, EVAL_SPLIT.get(d, "test"), cfg,
                              rows if rows is not None else EVAL_ROWS_CAP,
                              root, pools=pools)
    return out


# -- torch is optional at import time, never stubbed -----------------------

try:  # pragma: no cover - exercised by the venv, skipped by the suite
    import torch
    from safetensors.torch import load_file, save_file
    from torch import nn

    from model.decision_head import (DEFAULT_D_MODEL, DEFAULT_HEADS,
                                     DEFAULT_LAYERS, DecisionEngine,
                                     PointerDecisionHead)
    from model.encoder import DEFAULT_BACKBONE, load_backbone
    from model.weights import BACKBONES, weights_dir
    HAVE_TORCH = True
except ImportError as _exc:  # pragma: no cover
    HAVE_TORCH = False
    _IMPORT_ERROR = _exc
    DEFAULT_D_MODEL, DEFAULT_LAYERS, DEFAULT_HEADS = 256, 2, 8
    DEFAULT_BACKBONE = "ettin-68m"


def _require_torch() -> None:
    if not HAVE_TORCH:  # pragma: no cover
        raise RuntimeError(
            f"the real stack is missing ({_IMPORT_ERROR}); run with "
            f".venv-train/bin/python — nothing here is stubbed")


# -- batched encoding (the single-row path is `encoder.encode_state`) ------

def encode_states(backbone, states: list, max_length: int = TRAIN_MAX_LENGTH,
                  pad_multiple: int = PAD_MULTIPLE, grad: bool = False):
    """One backbone forward over a whole batch of states.

    `model.encoder.encode_state` is the single-row contract and stays the
    reference; this is the same forward with a padded batch axis, which is
    the only way ~1 M rows fit a wallclock. `test_train_decision` asserts
    the two agree row by row.

    Padding up to a multiple of `pad_multiple` keeps the number of distinct
    tensor shapes small — MPS recompiles a kernel per shape, and the
    sampler's variable K already supplies enough variants.
    """
    _require_torch()
    enc = backbone.tokenizer(states, return_tensors="pt", padding=True,
                             truncation=True, max_length=max_length)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    t = ids.shape[1]
    if pad_multiple > 1 and t % pad_multiple:
        pad = pad_multiple - t % pad_multiple
        pad_id = getattr(backbone.tokenizer, "pad_token_id", 0) or 0
        ids = nn.functional.pad(ids, (0, pad), value=pad_id)
        mask = nn.functional.pad(mask, (0, pad), value=0)
    ids = ids.to(backbone.device)
    mask = mask.to(backbone.device)
    # frozen backbone: no_grad (not inference_mode, the head trains on top).
    # #T-unfreeze-backbone passes grad=True and the forward keeps its graph.
    ctx = contextlib.nullcontext() if grad else torch.no_grad()
    with ctx:
        tokens = backbone.model(input_ids=ids,
                                attention_mask=mask).last_hidden_state
    return tokens, mask.bool(), int(mask.sum().item())


def batch_embeddings(engine, batch: list, grad: bool = False):
    """Question + option embeddings for a batch, through the shared cache.

    One `embed_texts` call per batch: label texts repeat across rows, so
    after the first few batches this costs a dict lookup, not a forward.
    """
    texts, spans = [], []
    for sample in batch:
        start = len(texts)
        texts.append(canonical_question(sample.question))
        texts.extend(o["text"] for o in sample.options)
        spans.append((start, len(texts)))
    embs = engine.embed_texts(texts, grad=grad)
    return embs, spans


def row_logits(engine, tokens, mask, embs, spans, i: int):
    """`[K + 1]` logits for row `i` of an encoded batch."""
    start, end = spans[i]
    return engine.head(tokens[i:i + 1], mask[i:i + 1],
                       embs[start], embs[start + 1:end])


def pack_options(embs, spans, device):
    """Gather `embed_texts` output into the padded `[B, K_max, H]` block.

    `embs` is the flat `[question, opt, opt, ...]` per row that
    `batch_embeddings` returns; `spans` says where each row starts and
    ends. One `index_select` builds the whole padded cube, so the packing
    itself costs a single kernel rather than a Python slice per row.

    Padding slots gather row 0 instead of being zero-filled — one kernel
    less, and `forward_batch` masks them out of the set attention, the
    row summary and the scores, so their content cannot reach a real
    option (`test_decision_head` asserts exactly that).
    """
    b = len(spans)
    kmax = max(end - start - 1 for start, end in spans)
    idx = torch.zeros((b, kmax), dtype=torch.long)
    omask = torch.zeros((b, kmax), dtype=torch.bool)
    qidx = torch.empty((b,), dtype=torch.long)
    for i, (start, end) in enumerate(spans):
        k = end - start - 1
        qidx[i] = start
        idx[i, :k] = torch.arange(start + 1, end)
        omask[i, :k] = True
    q = embs.index_select(0, qidx.to(device))
    opts = embs.index_select(0, idx.reshape(-1).to(device)).reshape(
        b, kmax, embs.shape[-1])
    return q, opts, omask.to(device)


def batch_prior_penalty(batch: list, counts: dict, kmax: int, device,
                        dtype):
    """`[B, K_max + 1]` of `log(train count)` — the batched candidate 4.

    Same numbers `apply_prior_penalty` subtracts row by row, laid out on
    the padded axis. `unknown` (the last column) and the padding columns
    get 0, which is what the per-row path does with its trailing `0.0`.
    """
    rows = []
    for sample in batch:
        pen = [prior_penalty_for(sample.dataset, o["id"], counts)
               for o in sample.options]
        rows.append(pen + [0.0] * (kmax + 1 - len(pen)))
    return torch.tensor(rows, device=device, dtype=dtype)


def batch_gold(batch: list, kmax: int, device):
    """Gold column per row on the padded axis.

    A row that answers `unknown` points at its own `K_i` in the per-row
    layout; here every `unknown` lives in the same column, `K_max`.
    """
    return torch.tensor(
        [s.gold_index if s.gold_index < len(s.options) else kmax
         for s in batch], device=device, dtype=torch.long)


# -- gen-objective candidates (#T-gen-objective) ---------------------------
#
# Four composable interventions against label-space memorisation. The
# ablation of #T-antiscale-diag attributes 75.4 % of the 250 k -> 1 M fall
# to ranking (axis 4) and measures that 100 % of the 1 M mixture is
# answerable by a text -> label map (axis 1) — so these attack the
# incentive (the objective), not the capacity, and they run BEFORE
# #T-unfreeze-backbone. All four default to OFF and compose: each is a
# pure function over a batch/samples/tensors, wired into `train()` by one
# flag each. Only one is adopted (the one that moves the unseen slope);
# the rest stay as flags, recorded as discarded in the task's gate.

#: marker for synthetic replacement option texts (candidate 1). A real
#: label text never contains a NUL byte, so a replaced text can never
#: collide with the train label space — which is what the test asserts.
NOVEL_OPTION_MARK = "\x00gen-novel"


def apply_label_dropout(batch: list, rate: float,
                        rng: random.Random) -> tuple:
    """Candidate 1: per-episode dropout of option texts.

    Every option text of every row is replaced with probability `rate`
    by a novel sentinel text. Positions, ids, `gold_index` and `answer`
    are untouched, so the row stays answerable — but a memorised
    text -> label map no longer suffices to answer it. Returns the new
    batch plus a stats dict.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("label dropout rate must be in [0, 1]")
    out, replaced, used = [], 0, set()
    for sample in batch:
        options = [dict(o) for o in sample.options]
        for opt in options:
            if rng.random() < rate:
                text = f"{NOVEL_OPTION_MARK}-{rng.randrange(1 << 60):016x}"
                while text in used:
                    text = (f"{NOVEL_OPTION_MARK}-"
                            f"{rng.randrange(1 << 60):016x}")
                used.add(text)
                opt["text"] = text
                replaced += 1
        out.append(replace(sample, options=options))
    return out, {"rows": len(batch), "options_replaced": replaced,
                 "rate": rate}


def resample_distractors(sample: Sample, sampler: OptionSetSampler,
                         rng: random.Random) -> Sample:
    """Candidate 2: episodic few-shot — a fresh distractor set, same gold.

    Reuses the sampler's own pool and difficulty buckets
    (#T-optset-sampler): the redrawn set keeps the row's K and its
    hard/easy split, the gold stays present at a reshuffled position,
    and `unknown` rows stay `unknown` with a redrawn offer set. Calling
    this per step (seeded by the step) is what makes the option SET vary
    across epochs instead of only its order.
    """
    d = sample.dataset
    index = sampler.index[d]
    by_id = index.text
    unknown = sample.is_unknown
    gold_id = None if unknown else sample.answer
    hard, easy = index.buckets("" if gold_id is None else gold_id)
    k = len(sample.options)
    n_distract = k if unknown else k - 1
    n_hard = min(len(hard), sample.n_hard)
    n_easy = min(len(easy), n_distract - n_hard)
    if n_easy < n_distract - n_hard:  # tiny pool: spill back to hard
        n_hard = min(len(hard), n_distract - n_easy)
        n_easy = n_distract - n_hard
    picked = rng.sample(hard, n_hard) + rng.sample(easy, n_easy)
    options = [{"id": i, "text": by_id.get(i, i)} for i in picked]
    if gold_id is not None:
        options.append({"id": gold_id, "text": by_id.get(gold_id, gold_id)})
    rng.shuffle(options)
    gold_index = k if unknown else next(
        i for i, o in enumerate(options) if o["id"] == gold_id)
    return replace(sample, options=options,
                   answer=UNKNOWN_ID if unknown else gold_id,
                   gold_index=gold_index, n_hard=n_hard, n_easy=n_easy)


def contrastive_q_option_loss(q, opts_list: list, golds: list,
                              tau: float = 0.07):
    """Candidate 3: InfoNCE over question <-> option-TEXT similarity.

    Per row, the question vector scores against that row's own option
    vectors (COSINE / tau) with a cross-entropy against the gold position
    — so the signal is question <-> option text, never question -> index
    of a known space. Rows whose gold is absent (`unknown`, gold None)
    are skipped. Pure torch: the gradient flows into the inputs, which
    in `train()` are the head's trainable `q_proj` / `opt_proj`
    outputs over the frozen backbone embeddings.

    Both sides are L2-normalised before the dot product, which is what
    makes `tau` a temperature: the similarity then lives in [-1, 1] and
    the loss is bounded by log(K). Without the normalisation the raw
    projections of a d_model=256 head give dot products in the tens, and
    dividing those by 0.07 produces a term ~35x the listwise
    cross-entropy that grows without bound — measured: the first
    contrastive arm logged loss 39.2 at step 1 rising to 53.0 by step 50
    while every other arm sat at 1.45. That arm was not measuring
    "CE + 0.5 x InfoNCE"; it was measuring a runaway term, so it was
    restarted against this function.
    """
    _require_torch()
    if tau <= 0:
        raise ValueError("contrastive temperature must be positive")
    losses = []
    for b in range(q.shape[0]):
        g = golds[b]
        if g is None:
            continue
        qn = nn.functional.normalize(q[b], dim=-1)
        on = nn.functional.normalize(opts_list[b], dim=-1)
        sims = (on * qn.unsqueeze(0)).sum(-1) / tau
        target = torch.tensor([g], device=sims.device)
        losses.append(nn.functional.cross_entropy(sims.unsqueeze(0),
                                                  target))
    if not losses:
        return q.sum() * 0.0
    return torch.stack(losses).mean()


def fit_label_prior(samplers: dict) -> dict:
    """Candidate 4 (fit): empirical train prior over gold labels.

    Counts gold answers over the training samplers' loaded rows, keyed
    `(dataset, option_id)`. An approximation of what the head sees (the
    mixture selector subsamples per question), but the ranking of heads
    vs tails — which is all the penalty needs — is what it preserves.
    """
    counts: dict = {}
    for d, sampler in samplers.items():
        for row in sampler.rows.get(d, []):
            for question in row.get("questions", []):
                answer = question.get("answer")
                if answer is None or answer == UNKNOWN_ID:
                    continue
                key = (d, answer)
                counts[key] = counts.get(key, 0) + 1
    return counts


def prior_penalty_for(dataset: str, option_id: str, counts: dict) -> float:
    """`log(count)` penalty for one option; 0 for labels unseen in train."""
    return math.log(counts.get((dataset, option_id), 1))


def apply_prior_penalty(logits, sample: Sample, counts: dict,
                        weight: float):
    """Candidate 4 (apply): discount the empirical frequency prior.

    Subtracts `weight * log(train count)` from each of the K option
    logits so a frequent label no longer wins by being frequent; the
    `unknown` logit is untouched. A constant offset that keeps the
    gradient: the operation is differentiable in the logits.
    """
    _require_torch()
    if weight < 0:
        raise ValueError("prior penalty weight must be non-negative")
    if weight == 0 or not counts:
        return logits
    penalty = torch.tensor(
        [prior_penalty_for(sample.dataset, o["id"], counts)
         for o in sample.options] + [0.0], device=logits.device,
        dtype=logits.dtype)
    return logits - weight * penalty


# -- metrics ---------------------------------------------------------------

@dataclass
class RunningMetrics:
    """Accuracy/Brier/NLL as sums; ECE over a rolling window of rows."""

    n: int = 0
    correct: int = 0
    correct_options: int = 0
    loss_sum: float = 0.0
    brier_sum: float = 0.0
    nll_sum: float = 0.0
    chance_sum: float = 0.0
    chance_options_sum: float = 0.0
    k_sum: int = 0
    abstain: int = 0
    window: list = field(default_factory=list)
    per_dataset: dict = field(default_factory=dict)

    def add(self, probs: list, gold: int, dataset: str = "") -> None:
        k1 = len(probs)
        pred = max(range(k1), key=lambda i: probs[i])
        ok = int(pred == gold)
        # Diagnostic, NOT the gate criterion: argmax restricted to the K
        # real options, i.e. "does the pointer RANK the right option first"
        # with the `unknown` logit taken out of the race. A model that
        # abstains on every unfamiliar option set scores 0 above and can
        # still rank well here; the operator needs to see both.
        pred_opt = max(range(k1 - 1), key=lambda i: probs[i]) if k1 > 1 else 0
        self.n += 1
        self.correct += ok
        self.correct_options += int(pred_opt == gold)
        self.brier_sum += brier_score(probs, gold)
        self.nll_sum += -math.log(max(probs[gold], 1e-12))
        self.chance_sum += 1.0 / k1
        self.chance_options_sum += 1.0 / max(k1 - 1, 1)
        self.k_sum += k1 - 1
        self.abstain += int(pred == k1 - 1)
        self.window.append((probs[pred], ok))
        if len(self.window) > ECE_WINDOW:
            del self.window[:len(self.window) - ECE_WINDOW]
        if dataset:
            d = self.per_dataset.setdefault(dataset, [0, 0, 0])
            d[0] += 1
            d[1] += ok
            d[2] += int(pred_opt == gold)

    def report(self) -> dict:
        n = max(self.n, 1)
        lo, hi = wilson_interval(self.correct, self.n)
        olo, ohi = wilson_interval(self.correct_options, self.n)
        return {
            "n": self.n,
            "accuracy": round(self.correct / n, 6),
            "accuracy_ci95": [round(lo, 6), round(hi, 6)],
            "chance": round(self.chance_sum / n, 6),
            "accuracy_options_only": round(self.correct_options / n, 6),
            "accuracy_options_only_ci95": [round(olo, 6), round(ohi, 6)],
            "chance_options_only": round(self.chance_options_sum / n, 6),
            "brier": round(self.brier_sum / n, 6),
            "nll": round(self.nll_sum / n, 6),
            "ece": round(expected_calibration_error(self.window), 6),
            "mean_k": round(self.k_sum / n, 4),
            "abstain_rate": round(self.abstain / n, 6),
            "per_dataset": {
                d: {"n": v[0], "accuracy": round(v[1] / max(v[0], 1), 6),
                    "accuracy_options_only": round(v[2] / max(v[0], 1), 6)}
                for d, v in sorted(self.per_dataset.items())},
        }


# -- the regime a run was trained at ---------------------------------------

def cardinality_regime(config, episodic_resample: bool = False) -> dict:
    """R4 — the cardinality this run trains at, written into the artifact.

    A GO/NO-GO is only valid inside the regime it was measured in, so a
    checkpoint that does not say which regime produced it turns every later
    comparison into a guess. This block travels in `run.json` and in EVERY
    checkpoint manifest, and `eval.scoreboard` prints it above the table
    (`#T-eval-cardinality`, rule R4).
    """
    full = config.k_min == config.k_max == 0
    return {
        "mode": "full label space" if full else "sampled options",
        "k_min": config.k_min,
        "k_max": config.k_max,
        "options_per_row": (f"{config.k_min}-{config.k_max} sampled from the "
                            "row's label space, plus the `unknown` logit"),
        "unknown_fraction": config.unknown_fraction,
        "hard_fraction": config.hard_fraction,
        "resampled_each_epoch": bool(episodic_resample),
        "loss_normalised_over": "the [K + 1] columns handed to "
                                "`cross_entropy` — that is the denominator, "
                                "and it is this K, not the label space",
        "rule": "R4 — a verdict measured in this regime does not transfer "
                "to another one; it is marked non-transferable instead of "
                "inherited",
    }


# -- evaluation ------------------------------------------------------------

def evaluate(engine, samplers: dict, max_samples: int = 4000,
             batch_size: int = 32, max_length: int = TRAIN_MAX_LENGTH) -> dict:
    """Accuracy / ECE / Brier over an eval mixture, no gradients."""
    _require_torch()
    if not samplers:
        return {"n": 0, "skipped": "no eval sampler for this cut"}
    # eval mixtures are not the training corpus: no cap applies
    stream = MixtureStream(samplers, batch_size, verify=False)
    metrics = RunningMetrics()
    engine.head.eval()
    with torch.no_grad():
        for batch in stream.epoch(0):
            tokens, mask, _ = encode_states(
                engine.backbone, [s.state for s in batch], max_length)
            embs, spans = batch_embeddings(engine, batch)
            q_emb, opt_embs, opt_mask = pack_options(embs, spans,
                                                     engine.device)
            kmax = opt_embs.shape[1]
            logits = engine.head.forward_batch(tokens, mask, q_emb,
                                               opt_embs, opt_mask)
            probs_rows = torch.softmax(logits.float(), dim=-1).cpu().tolist()
            for i, sample in enumerate(batch):
                k = len(sample.options)
                metrics.add(probs_rows[i][:k] + [probs_rows[i][kmax]],
                            sample.gold_index, sample.dataset)
            if metrics.n >= max_samples:
                break
    engine.head.train()
    report = metrics.report()
    report["beats_chance"] = bool(report["accuracy_ci95"][0] > report["chance"])
    report["ranking_beats_chance"] = bool(
        report["accuracy_options_only_ci95"][0] > report["chance_options_only"])
    # #T-eval-cardinality: this cut shares its regime with the objective —
    # the sampler's K, not the label space — which is what makes it useful
    # during a run and blind to the transfer the product sells. It is a
    # DIAGNOSTIC and says so wherever it is published, with the n and the
    # mean K that make it readable at all (rules R1/R2).
    report["role"] = "diagnostic"
    report["why_diagnostic"] = (
        f"n={report['n']}, mean K={report.get('mean_k')} — the training "
        "regime's cardinality, not the label space. The primary metric is "
        "full-cardinality accuracy (`eval.scoreboard`); this number may not "
        "head a report")
    return report


def holdout_cleanliness(holdout: dict, samplers: dict,
                        max_samples: int = 20000) -> dict:
    """No held-out label TEXT (normalised) may appear in a train sample.

    Checked on the sampler's own output, not on the config that produced
    it: a restriction bug would show up here as a hit, which is the point.
    """
    banned = {}
    for d, h in holdout.items():
        for label in h.unseen:
            banned.setdefault(normalise_label(h.texts.get(label, label)), []).append(
                f"{d}:{label}")
    hits, checked = [], 0
    per_dataset = max(1, max_samples // max(len(samplers), 1))
    for d, sampler in samplers.items():
        seen_here = 0
        for batch in MixtureStream({d: sampler}, 64,
                                   verify=False).epoch(0):
            for sample in batch:
                checked += 1
                seen_here += 1
                for opt in sample.options:
                    key = normalise_label(opt["text"])
                    if key in banned:
                        hits.append({"dataset": d, "row_id": sample.row_id,
                                     "option": opt["id"],
                                     "label": banned[key][0]})
            if seen_here >= per_dataset:
                break
    return {"pass": not hits, "checked_samples": checked,
            "banned_texts": len(banned), "hits": hits[:10],
            "n_hits": len(hits),
            "note": "compared by normalised TEXT, not by option id"}


# -- checkpoints -----------------------------------------------------------

def tokenizer_source(backbone_id: str) -> str:
    return os.path.join(weights_dir(backbone_id), "tokenizer.json")


def model_version(backbone_id: str, arch: dict, seed: int, samples: int,
                  weights_sha: str, backbone_sha: str | None = None) -> str:
    """The `model_version` half of `jev_runtime::CacheKey`.

    `CacheKey::new(model_version, state_hash, tokenizer_hash)` composes the
    runtime's state cache; a checkpoint that cannot name its own
    `model_version` cannot be cached, so it is built here and written into
    the manifest next to the `tokenizer_hash` the same key needs.
    """
    # A trained encoder (#T-unfreeze-backbone) is part of the model the
    # key names: two runs can share a head sha and differ in the backbone.
    tail = f"-bb{backbone_sha[:8]}" if backbone_sha else ""
    return (f"jev-dec-{backbone_id}-d{arch['d_model']}l{arch['n_layers']}"
            f"h{arch['n_heads']}-s{seed}-n{samples}-{weights_sha[:12]}{tail}")


def save_checkpoint(engine, out_dir: str, run: dict, samples: int,
                    tokens: int, metrics: dict) -> dict:
    """safetensors + tokenizer.json + manifest, in one directory."""
    _require_torch()
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "model.safetensors")
    save_file({k: v.detach().cpu().contiguous()
               for k, v in engine.head.state_dict().items()}, weights_path)
    tok_dst = os.path.join(out_dir, "tokenizer.json")
    shutil.copyfile(tokenizer_source(run["backbone"]["id"]), tok_dst)
    weights_sha = sha256_file(weights_path)
    arch = run["architecture"]
    # #T-unfreeze-backbone: a checkpoint whose encoder was trained is not
    # reproducible from the hub revision any more, so its bytes ship with
    # the head. The counts below are read off the live module, which is
    # what makes `trainable_params` an assertion and not a label.
    bb = dict(run["backbone"])
    named = list(engine.backbone.model.named_parameters())
    bb["state_dict_params"] = sum(v.numel() for _, v in named)
    bb["state_dict_trainable_params"] = sum(v.numel() for _, v in named
                                            if v.requires_grad)
    if not bb.get("frozen", True):
        bb_path = os.path.join(out_dir, "backbone.safetensors")
        save_file({k: v.detach().cpu().contiguous() for k, v
                   in engine.backbone.model.state_dict().items()}, bb_path)
        bb["weights_file"] = "backbone.safetensors"
        bb["weights_sha256"] = sha256_file(bb_path)
    manifest = {
        "task": "T-train-real",
        "run_id": run["run_id"],
        "model_version": model_version(
            run["backbone"]["id"], arch, run["seed"], samples,
            weights_sha, bb.get("weights_sha256")),
        "tokenizer_hash": sha256_file(tok_dst),
        "weights_sha256": weights_sha,
        "weights_file": "model.safetensors",
        "format": "safetensors",
        "head_params": engine.head.n_params(),
        "architecture": arch,
        "backbone": bb,
        "seed": run["seed"],
        "samples_seen": samples,
        "tokens_seen": tokens,
        "loss": "listwise cross-entropy over [K + 1] logits (unknown last)",
        # R4: every checkpoint says what cardinality it was trained at, so a
        # K<=8 run and a full-space run cannot be compared by accident.
        "cardinality_regime": run.get("cardinality_regime"),
        "datasets": run["datasets"],
        "data_manifest": run["data_manifest"],
        "holdout": run["holdout_path"],
        "metrics": metrics,
        "runtime_cache_key": ("jev_runtime::CacheKey(model_version, "
                              "state_hash, tokenizer_hash)"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest


def load_checkpoint(ckpt_dir: str, device: str = "auto"):
    """Cold load: fresh backbone, fresh head, weights from safetensors."""
    _require_torch()
    with open(os.path.join(ckpt_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    arch = manifest["architecture"]
    backbone = load_backbone(manifest["backbone"]["id"], device)
    head = PointerDecisionHead(backbone.hidden_size, arch["d_model"],
                               arch["n_layers"], arch["n_heads"])
    head.load_state_dict(load_file(os.path.join(ckpt_dir,
                                                manifest["weights_file"])))
    head.eval()
    bb_file = manifest["backbone"].get("weights_file")
    if bb_file:  # the encoder was trained: its bytes, not the hub's
        bb_path = os.path.join(ckpt_dir, bb_file)
        if not os.path.exists(bb_path):
            raise FileNotFoundError(
                f"{ckpt_dir} declares a trained backbone ({bb_file}) but "
                "the file is missing: scoring it against the hub revision "
                "would silently evaluate a different model")
        got = sha256_file(bb_path)
        want = manifest["backbone"].get("weights_sha256")
        if want and got != want:
            raise ValueError(f"{bb_file}: sha256 {got} != manifest {want}")
        backbone.model.load_state_dict(load_file(bb_path))
        backbone.model.eval()
    return DecisionEngine(backbone=backbone, head=head), manifest


def measure_decision_latency(engine, samples: list, n: int = 25) -> dict:
    """Real rows, cold state AND option caches, encode included."""
    _require_torch()
    times = []
    for sample in samples[:n]:
        engine._states.clear()
        engine._texts.clear()
        t0 = time.perf_counter()
        mem = engine.encode_state(sample.state)
        engine.score(mem, canonical_question(sample.question), sample.options)
        if engine.device.type == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    if not times:
        return {"runs": 0}
    return {"runs": len(times), "device": str(engine.device),
            "p50_ms": round(times[len(times) // 2], 3),
            "p95_ms": round(times[max(0, int(len(times) * 0.95) - 1)], 3),
            "max_ms": round(times[-1], 3),
            "budget_ms": LATENCY_BUDGET_MS,
            "cache": "cleared per row (cold state + cold options)"}


# -- the training run ------------------------------------------------------

def data_manifest(datasets, root: str = PREFETCH_DIR) -> dict:
    """sha256 of the exact corpus bytes: seed + this = reproducible run."""
    out = {}
    for d in datasets:
        shards = mixmod.SOURCES[d].paths(root) if d in mixmod.SOURCES else [
            os.path.join(root, f"{d}.jsonl")]
        shards = [p for p in shards if os.path.exists(p)]
        out[d] = {"sha256": sha256_file(shards[0]) if len(shards) == 1 else [
            {"path": os.path.relpath(p, ROOT), "sha256": sha256_file(p)}
            for p in shards],
            "bytes": sum(os.path.getsize(p) for p in shards)}
    return out


def build_mix(seed: int, root: str = PREFETCH_DIR, target: int | None = None,
              datasets=None, cap_margin: float = mixmod.CAP_SAFETY_MARGIN,
              weights: dict | None = None, dataset_cap: float | None = None,
              family_cap: float | None = None) -> tuple:
    """Scan, hold labels out, plan the mixture (#T-corpus-rebalance).

    One pass over the corpus answers all three: the label pool of every
    source (which the holdout carves), the per-label gold counts (which say
    how much supply survives that carving) and the shard sha256 the
    manifest pins. Quotas are then planned on the SURVIVING supply, so the
    share a source is granted is the share it can actually deliver.

    `datasets`, `cap_margin` and `weights` are what `tools/mix_1m` passes
    to assemble `decision-mix-clean-1m`: the registry minus the §§18/77
    benchmark fence, the caps at their exact §§65-66 values and the §86
    layer plan as the allocator's pull. The function is deterministic in
    its arguments, so the corpus this builds for a training run IS the one
    the published manifest describes — `members_sha256` proves it.

    Returns `(spec, scan, pools, holdout)`.
    """
    scan = mixmod.scan_supply(datasets, root)
    present = [d for d, s in scan.items() if not s.get("missing")]
    pools = {d: [{"id": i, "text": t}
                 for i, t in sorted(scan[d]["pool"].items())]
             for d in present if scan[d]["pool"]}
    holdout = build_holdout(present, root, pools=pools)
    seen_labels = {d: list(h.seen) for d, h in holdout.items() if h.unseen}
    supply = mixmod.effective_supply(scan, seen_labels)
    spec = mixmod.plan_mix(supply, target, seed, seen_labels, scan,
                           cap_margin, weights, dataset_cap, family_cap)
    return spec, scan, pools, holdout


def lr_at(step: int, total: int, base: float, warmup: int = 200) -> float:
    """Linear warmup, then cosine decay to 10 % of `base`."""
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


class MetricsLog:
    """`artifacts/runs/<run_id>/metrics.jsonl`, one json object per line."""

    def __init__(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "metrics.jsonl")
        self.fh = open(self.path, "a", buffering=1)

    def write(self, record: dict) -> None:
        self.fh.write(json.dumps(record, sort_keys=True) + "\n")

    def close(self) -> None:
        self.fh.close()


def _realised_counts(samplers: dict, stream) -> dict:
    """Count the diversity axes of what the samplers will really emit.

    Same shape and same member digest as `data.mix.realise()`, but read off
    the rows already in memory instead of walking the corpus a second time
    — so the mixture the manifest describes is literally the one this run
    is about to train on, holdout cuts included.
    """
    digest = hashlib.sha256()
    counts = {"dataset": {}, "family": {}, "origin": {}, "qtype": {},
              "lang": {}, "k": {}, "k_bucket": {}, "dynamic": 0, "rows": 0,
              "per_dataset": {}}
    for dataset in sorted(samplers):
        sampler = samplers[dataset]
        source = mixmod.SOURCES[dataset]
        selector = sampler.keep.get(dataset)
        indices = sampler.row_index.get(dataset, [])
        kept = 0
        for n, row in enumerate(sampler.rows[dataset]):
            index = indices[n] if indices else n
            lang = mixmod.lang_of(row.get("state", ""), source.lang_scope)
            for q in row.get("questions", []):
                qid = q.get("id", "")
                if selector is not None and not selector(index, qid,
                                                         q.get("answer")):
                    continue
                options = q.get("options", [])
                k = len(options)
                qtype = mixmod.question_type(
                    q.get("kind", ""), [o.get("text", "") for o in options])
                digest.update(f"{dataset}\x00{index}\x00{qid}\n".encode())
                for axis, key in (("dataset", dataset),
                                  ("family", source.family),
                                  ("origin", source.origin),
                                  ("qtype", qtype), ("lang", lang),
                                  ("k", str(k)),
                                  ("k_bucket", mixmod.k_bucket(k))):
                    counts[axis][key] = counts[axis].get(key, 0) + 1
                counts["dynamic"] += int(source.dynamic_options)
                counts["rows"] += 1
                kept += 1
        counts["per_dataset"][dataset] = {
            "kept": kept, "family": source.family, "origin": source.origin,
            "dynamic_options": source.dynamic_options,
            "weight": source.weight, "note": source.note}
    counts["members_sha256"] = digest.hexdigest()
    counts["stream_sizes"] = dict(sorted(stream.sizes.items()))
    return counts


# -- trainable backbone (#T-unfreeze-backbone) -----------------------------
#
# Every run before this one trained ~2.9 M of pointer head on top of a
# frozen 68 M/149 M encoder, and #T-antiscale-diag attributes 75 % of the
# 250 k -> 1 M fall to ranking. The question these knobs answer is whether
# the slope is a property of the objective alone or of a comparison that
# has to happen two 256-dim layers above the representation. Three
# regimes, same corpus, same seed, same evals: `none` (the honest
# baseline), `last-n` (the top blocks, discriminative LR) and `full`.

UNFREEZE_MODES = ("none", "last-n", "full")


def backbone_layers(model):
    """The encoder's transformer block list, whatever its family calls it."""
    for holder in (model, getattr(model, "encoder", None)):
        if holder is None:
            continue
        for attr in ("layers", "layer"):
            blocks = getattr(holder, attr, None)
            if blocks is not None and len(blocks):
                return blocks
    raise AttributeError("cannot find the block list of "
                         f"{type(model).__name__}")


def set_backbone_trainable(backbone, mode: str = "none",
                           last_n: int = 2) -> dict:
    """Flip `requires_grad` on the encoder and report what it opened.

    The returned dict is what `run.json` and every checkpoint manifest
    publish as the regime. `trainable_params` is counted off the module
    itself — not off the flag that was asked for — so a manifest cannot
    claim a regime the weights do not have.
    """
    if mode not in UNFREEZE_MODES:
        raise ValueError(f"unknown unfreeze mode {mode!r}; "
                         f"known: {list(UNFREEZE_MODES)}")
    model = backbone.model
    for p in model.parameters():
        p.requires_grad_(False)
    blocks = backbone_layers(model)
    opened = []
    if mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        opened = ["*"]
    elif mode == "last-n":
        if last_n < 1:
            raise ValueError("last-n needs at least one layer")
        n = min(last_n, len(blocks))
        for i in range(len(blocks) - n, len(blocks)):
            for p in blocks[i].parameters():
                p.requires_grad_(True)
            opened.append(f"layers.{i}")
        for name in ("final_norm", "norm", "layer_norm"):
            tail = getattr(model, name, None)
            if tail is not None:
                for p in tail.parameters():
                    p.requires_grad_(True)
                opened.append(name)
                break
    model.train(mode != "none")  # dropout follows the regime
    return {"mode": mode,
            "last_n": last_n if mode == "last-n" else None,
            "opened": opened,
            "frozen": mode == "none",
            "n_blocks": len(blocks),
            "params": backbone.params,
            "trainable_params": sum(p.numel() for p in model.parameters()
                                    if p.requires_grad)}


def peak_memory_gb(device) -> float | None:
    """Peak device memory in GiB — the price of a regime, in the summary."""
    kind = getattr(device, "type", str(device))
    try:
        if kind == "mps":
            return round(torch.mps.driver_allocated_memory() / 2 ** 30, 3)
        if kind == "cuda":
            return round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        unit = 1 if sys.platform == "darwin" else 1024  # bytes vs KiB
        return round(rss * unit / 2 ** 30, 3)
    except Exception:
        return None


def train(max_samples: int = 250_000, batch_size: int = 32,
          lr: float = 3e-4, seed: int = DEFAULT_SEED, device: str = "auto",
          run_id: str | None = None, rows_per_dataset: int | None = None,
          eval_samples: int = 3000, log_every: int = 10,
          stages: tuple = DEFAULT_STAGES, backbone_id: str = DEFAULT_BACKBONE,
          max_length: int = TRAIN_MAX_LENGTH, root: str = PREFETCH_DIR,
          d_model: int = DEFAULT_D_MODEL, n_layers: int = DEFAULT_LAYERS,
          n_heads: int = DEFAULT_HEADS, write_gate: bool = True,
          mix_target: int | None = None, use_mix: bool = True,
          fence_clean: bool = False, drop_datasets=(),
          only_datasets=(),
          mix_seed: int | None = None, dataset_cap: float | None = None,
          family_cap: float | None = None,
          allow_repeat: bool = False, label_dropout: float = 0.0,
          episodic_resample: bool = False,
          contrastive_weight: float = 0.0, contrastive_tau: float = 0.07,
          prior_penalty: float = 0.0, unfreeze: str = "none",
          unfreeze_layers: int = 2, backbone_lr: float = 1e-5) -> dict:
    """One model, one capped mixture, listwise loss, stage curve to 1 M."""
    _require_torch()
    t_start = time.perf_counter()
    run_id = run_id or time.strftime("dec-%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(seed)
    if not 0.0 <= label_dropout <= 1.0:
        raise ValueError("label_dropout must be in [0, 1]")
    if contrastive_weight < 0 or prior_penalty < 0:
        raise ValueError("contrastive_weight and prior_penalty must be >= 0")
    if contrastive_tau <= 0:
        raise ValueError("contrastive_tau must be positive")
    gen_rng = random.Random(f"{seed}\x00gen-objective")
    config = SamplerConfig(seed=seed)
    spec = pools = None
    train_rows = rows_per_dataset
    if use_mix:
        if mix_target is None and rows_per_dataset is not None:
            # a smoke run: the same SHARES, a smaller mixture. The row
            # budget cannot be split evenly (helpsteer2 ships five samples
            # per row, the intent family has three sources), so it scales
            # the target and the quotas do the dividing.
            mix_target = rows_per_dataset * len(mixmod.SOURCES)
        # `decision-mix-clean-1m` (#T-mix-1m): the registry minus the
        # §§18/77 benchmark fence, the §86 layer plan as the pull, and the
        # caps at their exact §§65-66 values — the fenced registry covers
        # only 99.75 % of a mixture, so the planner's safety margin would
        # make every clean mixture infeasible instead of merely tight.
        cap_d = (mixmod.MAX_DATASET_FRACTION if dataset_cap is None
                 else dataset_cap)
        cap_f = (mixmod.MAX_FAMILY_FRACTION if family_cap is None
                 else family_cap)
        datasets = cap_margin = weights = None
        if fence_clean:
            datasets = [d for d in mixmod.clean_datasets()
                        if d not in set(drop_datasets)]
            # the §86 corpus is ONE recipe, and `data.mix` holds it: the
            # same target, seed and cap margin `tools.mix_1m.run_mix`
            # publishes the manifest from. Defaulting the target to the
            # cap CEILING here and planning at margin 0.0 is what made the
            # trainer assemble a different corpus from the published one.
            cap_margin = mixmod.CLEAN_1M_CAP_MARGIN
            weights = mixmod.layer_weights(datasets)
            if mix_target is None:
                mix_target = mixmod.CLEAN_1M_TARGET
            if mix_seed is None:
                mix_seed = mixmod.CLEAN_1M_SEED
        elif drop_datasets:
            datasets = [d for d in mixmod.default_datasets()
                        if d not in set(drop_datasets)]
        if only_datasets:
            # an experiment names its corpus out loud (#T-labelspace-div:
            # `--only-dataset episodic-div`); the fence still supplies the
            # recipe defaults, and the weights are re-pulled for the named
            # list so the allocator never keys into another list's pull.
            unknown = [d for d in only_datasets if d not in mixmod.SOURCES]
            if unknown:
                raise ValueError(f"unregistered --only-dataset: {unknown}")
            datasets = [d for d in only_datasets
                        if d not in set(drop_datasets)]
            if fence_clean:
                weights = mixmod.layer_weights(datasets)
        # an explicit cap replaces the §§65-66 value outright, so the
        # mixture is planned against exactly the caps `MixtureStream`
        # verifies it against, and `run.json` records which pair that was
        spec, _scan, pools, holdout = build_mix(
            seed if mix_seed is None else mix_seed,
            root, mix_target, datasets,
            mixmod.CAP_SAFETY_MARGIN if cap_margin is None else cap_margin,
            weights, dataset_cap, family_cap)
        train_rows = None  # the quota is the budget
    else:  # the pre-#T-corpus-rebalance path: uncapped, P0 datasets only
        holdout = build_holdout(root=root)
    holdout_path = os.path.join(run_dir, "holdout.json")
    with open(holdout_path, "w") as fh:
        json.dump(holdout_report(holdout), fh, indent=2, sort_keys=True)
        fh.write("\n")

    tr = train_samplers(holdout, config, train_rows, root, spec, pools)
    prior_counts = fit_label_prior(tr) if prior_penalty > 0 else {}
    ev_seen = eval_samplers(holdout, "seen", config, rows_per_dataset, root,
                            pools)
    ev_unseen = eval_samplers(holdout, "unseen", config, rows_per_dataset,
                              root, pools)
    stream = MixtureStream(tr, batch_size, dataset_cap=cap_d,
                           family_cap=cap_f)
    epoch_size = stream.total()
    # -- the shortfall guardrail (#T-mix-1m) ---------------------------
    # `stream.stream()` loops the mixture forever, so a budget larger than
    # the corpus used to be filled by silently re-reading it. It is an
    # abort now, because the silent version already happened: a run asked
    # for 1 000 000 decisions, got a 39 981-row mixture and published a
    # curve as though it had seen a million distinct ones.
    epochs_over_corpus = round(max_samples / max(epoch_size, 1), 6)
    if epoch_size < max_samples and not allow_repeat:
        raise MixShortfallError(
            f"the mixture realises {epoch_size:,} decisions but "
            f"--max-samples asked to train on {max_samples:,}: the run "
            f"would take {epochs_over_corpus:.2f} epochs over the same "
            f"corpus and report it as {max_samples:,} distinct decisions. "
            f"Build a corpus that covers the budget (--mix-target "
            f"{max_samples:,} or larger), lower --max-samples to "
            f"{epoch_size:,}, or pass --allow-repeat to state the "
            f"repetition out loud — it is then published as "
            f"epochs_over_corpus in run.json, metrics.jsonl and mix.json.")
    budget = {"max_samples": max_samples, "epoch_samples": epoch_size,
              "epochs_over_corpus": epochs_over_corpus,
              "allow_repeat": bool(allow_repeat),
              "covers_budget": epoch_size >= max_samples}
    mix_manifest = None
    if spec is not None:
        mix_manifest = mixmod.build_manifest(
            spec, _realised_counts(tr, stream), _scan)
        mix_manifest["budget"] = budget
        with open(os.path.join(run_dir, "mix.json"), "w") as fh:
            json.dump(mix_manifest, fh, indent=2, sort_keys=True)
            fh.write("\n")

    backbone = load_backbone(backbone_id, device)
    torch.manual_seed(seed)  # the head's init must not depend on load order
    head = PointerDecisionHead(backbone.hidden_size, d_model, n_layers,
                               n_heads)
    engine = DecisionEngine(backbone=backbone, head=head)
    engine.head.train()
    regime = set_backbone_trainable(engine.backbone, unfreeze, unfreeze_layers)
    groups = [{"params": list(engine.head.parameters()),
               "lr": lr, "base_lr": lr, "name": "head"}]
    bb_params = [p for p in engine.backbone.model.parameters()
                 if p.requires_grad]
    if bb_params:  # discriminative LR: the encoder moves far slower
        groups.append({"params": bb_params, "lr": backbone_lr,
                       "base_lr": backbone_lr, "name": "backbone"})
    regime["lr"] = backbone_lr if bb_params else None
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.01)
    trainable = [p for g in groups for p in g["params"]]

    run = {
        "run_id": run_id, "seed": seed, "task": "T-train-real",
        "datasets": sorted(tr),
        "architecture": {"d_model": d_model, "n_layers": n_layers,
                         "n_heads": n_heads,
                         "scorer": "pointer over option text embeddings",
                         "loss": ("listwise cross-entropy over [K + 1]"
                                  + (f" + {contrastive_weight} * q<->opt "
                                     "InfoNCE" if contrastive_weight > 0
                                     else "")
                                  + (f" + prior penalty {prior_penalty}"
                                     if prior_penalty > 0 else ""))},
        "backbone": {"id": backbone.id, "hidden_size": backbone.hidden_size,
                     "params": backbone.params,
                     "frozen": regime["frozen"],
                     "trainable_params": regime["trainable_params"],
                     "lr": regime["lr"],
                     "unfreeze": regime,
                     "revision": BACKBONES[backbone_id]["revision"]},
        "cardinality_regime": cardinality_regime(config, episodic_resample),
        "data_manifest": data_manifest(sorted(tr), root),
        "mix": ({"version": mix_manifest["version"],
                 "seed": mix_manifest["seed"],
                 "target": mix_manifest["target"],
                 "rows": mix_manifest["composition"]["rows"],
                 "by_dataset": mix_manifest["composition"]["by_dataset"],
                 "by_family": mix_manifest["composition"]["by_family"],
                 "dynamic_option_share":
                     mix_manifest["composition"]["dynamic_option_share"],
                 "members_sha256": mix_manifest["members_sha256"],
                 "pass": mix_manifest["checks"]["pass"],
                 "path": os.path.join("artifacts", "runs", run_id,
                                      "mix.json")}
                if mix_manifest else {"enforced": False,
                                      "why": "--no-mix"}),
        "caps": {"dataset": stream.dataset_cap, "family": stream.family_cap,
                 "default": [mixmod.MAX_DATASET_FRACTION,
                             mixmod.MAX_FAMILY_FRACTION],
                 "overridden": [stream.dataset_cap, stream.family_cap]
                 != [mixmod.MAX_DATASET_FRACTION, mixmod.MAX_FAMILY_FRACTION]},
        "fence": {"clean_1m": fence_clean,
                  "fenced_datasets": sorted(mixmod.MIX_1M_FENCED)
                  if fence_clean else [],
                  "dropped": sorted(drop_datasets),
                  "only": sorted(only_datasets),
                  "why": ("§§18, 77: zero Banking77/HelpSteer2/PubMedQA "
                          "rows in train" if fence_clean else
                          "product mix: the benchmark fence is not applied")},
        "holdout_path": os.path.relpath(holdout_path, ROOT),
        "gen_objective": {"label_dropout": label_dropout,
                          "episodic_resample": episodic_resample,
                          "contrastive_weight": contrastive_weight,
                          "contrastive_tau": contrastive_tau,
                          "prior_penalty": prior_penalty,
                          "prior_labels": len(prior_counts)},
        "epoch_samples": epoch_size,
        "max_samples": max_samples,
        "budget": budget,
        "epochs_over_corpus": epochs_over_corpus,
        "allow_repeat": bool(allow_repeat),
        "batch_size": batch_size, "lr": lr, "max_length": max_length,
        "device": str(engine.device),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(run_dir, "run.json"), "w") as fh:
        json.dump(run, fh, indent=2, sort_keys=True)
        fh.write("\n")

    log = MetricsLog(run_dir)
    log.write({"t": "run", **{k: v for k, v in run.items()
                              if k != "data_manifest"}})
    total_steps = max(1, max_samples // batch_size)
    pending = [s for s in sorted(stages) if s <= max_samples]
    if max_samples not in pending:
        pending.append(max_samples)
    curve, stage_records = [], []
    metrics = RunningMetrics()
    seen_samples = tokens_seen = step = 0
    last_ckpt = None
    epochs_run = 0

    for epoch, batch in stream.stream():
        if seen_samples >= max_samples:
            break
        epochs_run = epoch + 1
        step += 1
        for group in opt.param_groups:
            group["lr"] = lr_at(step, total_steps, group["base_lr"])
        if episodic_resample:  # candidate 2: fresh distractor set
            batch = [resample_distractors(s, tr[s.dataset], gen_rng)
                     if s.dataset in tr else s for s in batch]
        if label_dropout > 0:  # candidate 1: novel option texts
            batch, _drop = apply_label_dropout(batch, label_dropout,
                                               gen_rng)
        tokens, mask, n_tok = encode_states(engine.backbone,
                                            [s.state for s in batch],
                                            max_length,
                                            grad=bool(bb_params))
        embs, spans = batch_embeddings(engine, batch, grad=bool(bb_params))
        opt.zero_grad(set_to_none=True)
        # ONE head call for the whole batch (#T-metal-throughput). The
        # row-by-row loop this replaces was ~60 % of a step's wallclock on
        # MPS in dispatch alone; `forward_batch` is the same maths on a
        # padded option axis and `test_decision_head` pins the two
        # together row by row.
        q_emb, opt_embs, opt_mask = pack_options(embs, spans, engine.device)
        kmax = opt_embs.shape[1]
        logits = engine.head.forward_batch(tokens, mask, q_emb, opt_embs,
                                           opt_mask)
        if prior_penalty > 0 and prior_counts:  # candidate 4: prior
            logits = logits - prior_penalty * batch_prior_penalty(
                batch, prior_counts, kmax, engine.device, logits.dtype)
        gold = batch_gold(batch, kmax, engine.device)
        loss = nn.functional.cross_entropy(logits, gold)
        if contrastive_weight > 0:  # candidate 3: q<->opt signal
            contrast_q, contrast_opts, contrast_golds = [], [], []
            for i, sample in enumerate(batch):
                start, end = spans[i]
                contrast_q.append(engine.head.q_proj(embs[start]))
                contrast_opts.append(engine.head.opt_proj(
                    embs[start + 1:end]))
                contrast_golds.append(
                    None if sample.gold_index >= len(sample.options)
                    else sample.gold_index)
            loss = loss + contrastive_weight * contrastive_q_option_loss(
                torch.stack(contrast_q), contrast_opts, contrast_golds,
                contrastive_tau)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        # One device->host transfer per step instead of one per row: the
        # padded columns are exactly 0 after the softmax, so each row is
        # read back as its own `K_i` options plus the shared `unknown`.
        batch_metrics = RunningMetrics()
        probs_cpu = torch.softmax(logits.detach().float(), dim=-1).cpu()
        probs_rows = probs_cpu.tolist()
        for i, sample in enumerate(batch):
            k = len(sample.options)
            probs = probs_rows[i][:k] + [probs_rows[i][kmax]]
            metrics.add(probs, sample.gold_index, sample.dataset)
            batch_metrics.add(probs, sample.gold_index, sample.dataset)
        seen_samples += len(batch)
        tokens_seen += n_tok
        metrics.loss_sum += float(loss.detach()) * len(batch)

        if step % log_every == 0 or step == 1:
            b = batch_metrics.report()
            log.write({
                "t": "step", "step": step, "epoch": epoch,
                "samples": seen_samples, "tokens": tokens_seen,
                "loss": round(float(loss.detach()), 6),
                "accuracy": b["accuracy"], "brier": b["brier"],
                "ece": round(expected_calibration_error(metrics.window), 6),
                "mean_k": b["mean_k"], "abstain_rate": b["abstain_rate"],
                "lr": round(lr_at(step, total_steps, lr), 8),
                "elapsed_s": round(time.perf_counter() - t_start, 2),
                "samples_per_s": round(seen_samples /
                                       max(1e-6, time.perf_counter() - t_start), 2),
            })

        while pending and seen_samples >= pending[0]:
            stage = pending.pop(0)
            record = _stage(engine, run, run_dir, log, stage, seen_samples,
                            tokens_seen, step, ev_seen, ev_unseen,
                            eval_samples, batch_size, max_length, t_start,
                            holdout, tr, write_gate)
            curve.append(record["curve_point"])
            stage_records.append(record)
            last_ckpt = record["checkpoint"]

    summary = {
        "run_id": run_id, "steps": step, "samples_seen": seen_samples,
        "tokens_seen": tokens_seen, "epoch_samples": epoch_size,
        "epochs_over_corpus": epochs_over_corpus,
        "epochs_run": epochs_run,
        "train": metrics.report(),
        "train_loss": round(metrics.loss_sum / max(metrics.n, 1), 6),
        "scale_curve": curve, "stages": stage_records,
        "checkpoint": last_ckpt,
        "cost": {"device": str(engine.device),
                 "backbone": regime,
                 "minutes_per_1m": round(
                     (time.perf_counter() - t_start) / 60.0
                     * 1_000_000 / max(1, seen_samples), 2),
                 "peak_mem_gb": peak_memory_gb(engine.device)},
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    log.write({"t": "summary", **{k: v for k, v in summary.items()
                                  if k != "stages"}})
    log.close()
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return summary


def _stage(engine, run, run_dir, log, stage, samples, tokens, step,
           ev_seen, ev_unseen, eval_samples, batch_size, max_length,
           t_start, holdout, train_sets, write_gate) -> dict:
    """A point on the 250 k -> 1 M curve: eval both cuts, checkpoint, gate."""
    # #T-unfreeze-backbone: with a trainable encoder the memoised states
    # and texts are stale by construction, and its dropout must be off
    # for the measurement. Both are restored before training resumes.
    moving = not run["backbone"].get("frozen", True)
    if moving:
        engine.clear_caches()
        engine.backbone.model.eval()
    seen = evaluate(engine, ev_seen, eval_samples, batch_size, max_length)
    unseen = evaluate(engine, ev_unseen, eval_samples, batch_size, max_length)
    ckpt_dir = os.path.join(CKPT_DIR, run["run_id"], f"stage-{stage:09d}")
    manifest = save_checkpoint(engine, ckpt_dir, run, samples, tokens,
                               {"seen": seen, "unseen": unseen})
    record = {
        "t": "stage", "stage": stage, "step": step, "samples": samples,
        "tokens": tokens, "seen": seen, "unseen": unseen,
        "checkpoint": os.path.relpath(ckpt_dir, ROOT),
        "model_version": manifest["model_version"],
        "elapsed_s": round(time.perf_counter() - t_start, 2),
    }
    log.write(record)
    record["curve_point"] = {
        "samples": samples, "tokens": tokens,
        "unseen_accuracy": unseen.get("accuracy"),
        "unseen_chance": unseen.get("chance"),
        "unseen_ece": unseen.get("ece"),
        "unseen_ranking": unseen.get("accuracy_options_only"),
        "unseen_ranking_chance": unseen.get("chance_options_only"),
        "unseen_abstain_rate": unseen.get("abstain_rate"),
        "seen_accuracy": seen.get("accuracy"),
        "seen_ece": seen.get("ece"),
    }
    if write_gate:
        try:
            write_gate_json(ckpt_dir, holdout, train_sets, seen, unseen,
                            manifest, run, engine)
        except Exception as exc:  # a gate write must never kill a long run
            log.write({"t": "gate-error", "stage": stage, "error": str(exc)})
    if moving:
        engine.clear_caches()
        engine.backbone.model.train()
    return record


# -- gate ------------------------------------------------------------------

def scale_curve(run_id: str) -> list:
    """Read the 250 k -> 1 M curve back out of `metrics.jsonl`."""
    path = os.path.join(RUNS_DIR, run_id, "metrics.jsonl")
    curve = []
    if not os.path.exists(path):
        return curve
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("t") != "stage":
                continue
            curve.append({
                "samples": rec["samples"], "tokens": rec["tokens"],
                "unseen_accuracy": rec["unseen"].get("accuracy"),
                "unseen_chance": rec["unseen"].get("chance"),
                "unseen_ece": rec["unseen"].get("ece"),
                "unseen_ranking": rec["unseen"].get("accuracy_options_only"),
                "unseen_ranking_chance":
                    rec["unseen"].get("chance_options_only"),
                "unseen_abstain_rate": rec["unseen"].get("abstain_rate"),
                "seen_accuracy": rec["seen"].get("accuracy"),
                "seen_ece": rec["seen"].get("ece"),
                "model_version": rec.get("model_version"),
            })
    return curve


def probe_samples(samplers: dict, n: int = 25) -> list:
    out = []
    for sampler in samplers.values():
        for batch in MixtureStream({sampler.datasets[0]: sampler},
                                   16, verify=False).epoch(0):
            out.extend(batch)
            break
        if len(out) >= n:
            break
    return out[:n]


def checkpoint_checks(ckpt_dir: str, live_engine, probe: list,
                      device: str = "cpu") -> dict:
    """Cold reload: same logits, a real decision inside the 500 ms budget."""
    _require_torch()
    cold, manifest = load_checkpoint(ckpt_dir, device)
    max_diff = 0.0
    with torch.no_grad():
        for sample in probe[:8]:
            question = canonical_question(sample.question)
            a = live_engine.score(live_engine.encode_state(sample.state),
                                  question, sample.options)
            b = cold.score(cold.encode_state(sample.state), question,
                           sample.options)
            for x, y in zip(a["logits"] + [a["unknown_logit"]],
                            b["logits"] + [b["unknown_logit"]]):
                max_diff = max(max_diff, abs(x - y))
    latency = measure_decision_latency(cold, probe)
    ok = (max_diff < 5e-3 and latency.get("p95_ms", 1e9) < LATENCY_BUDGET_MS)
    return {"pass": bool(ok), "max_logit_diff": round(max_diff, 8),
            "tolerance": 5e-3, "latency": latency,
            "loaded_from": os.path.relpath(ckpt_dir, ROOT),
            "device": str(cold.device),
            "model_version": manifest["model_version"]}


def compose_gate(ckpt_dir: str, manifest: dict, seen: dict, unseen: dict,
                 holdout: dict, cleanliness: dict, reload_check: dict,
                 label_free: dict, run: dict, write: bool = True) -> dict:
    """Assemble `artifacts/gates/T-train-real/gate.json` from real numbers."""
    unseen_ok = bool(unseen.get("n") and unseen.get("beats_chance"))
    checks = {
        "unseen_beats_chance": {
            "pass": unseen_ok,
            "criterion": ("Wilson 95 % lower bound of unseen accuracy > "
                          "mean chance (1 / (K + 1)) — written before the "
                          "first measurement"),
            "accuracy": unseen.get("accuracy"),
            "ci95": unseen.get("accuracy_ci95"),
            "chance": unseen.get("chance"),
            "n": unseen.get("n"),
            # R2: without the K this accuracy cannot be read at all, and
            # the per-dataset breakdown below inherits it from here.
            "mean_k": unseen.get("mean_k"),
            "per_dataset": unseen.get("per_dataset"),
            "abstain_rate": unseen.get("abstain_rate"),
            "diagnostic_not_a_criterion": {
                "what": ("argmax over the K options only, with the learned "
                         "`unknown` logit taken out of the race — it says "
                         "whether the POINTER ranks an unseen option first, "
                         "separately from whether the model chose to answer "
                         "at all. The gate above is decided by the full "
                         "[K + 1] argmax, as pre-registered."),
                "accuracy_options_only": unseen.get("accuracy_options_only"),
                "ci95": unseen.get("accuracy_options_only_ci95"),
                "chance_options_only": unseen.get("chance_options_only"),
                "ranking_beats_chance": unseen.get("ranking_beats_chance"),
            },
        },
        "holdout_clean": {**cleanliness},
        "label_free": {"pass": label_free["label_free"],
                       "offenders": label_free["offenders"],
                       "head_params": label_free["n_params"]},
        "checkpoint_cold_load": {**reload_check},
        "manifest_model_version": {
            "pass": bool(manifest.get("model_version")
                         and manifest.get("tokenizer_hash")),
            "model_version": manifest.get("model_version"),
            "tokenizer_hash": manifest.get("tokenizer_hash"),
            "consumer": manifest.get("runtime_cache_key"),
        },
    }
    gate = {
        "task": "T-train-real",
        "pass": all(c["pass"] for c in checks.values()),
        "role": {
            "kind": "diagnostic",
            "n": unseen.get("n"),
            "mean_k": unseen.get("mean_k"),
            "cardinality_regime": manifest.get("cardinality_regime"),
            "why": "this gate measures the run in the regime it trains in "
                   "(the sampler's K, not the label space). It says whether "
                   "the run is learning; it does not say what the product "
                   "promises, and it may not head a report",
            "primary_metric": "artifacts/gates/T-eval-cardinality/"
                              "scoreboard.json (#T-eval-cardinality)",
        },
        "run_id": manifest["run_id"],
        "model_version": manifest["model_version"],
        "checkpoint": os.path.relpath(ckpt_dir, ROOT),
        "samples_seen": manifest["samples_seen"],
        "tokens_seen": manifest["tokens_seen"],
        "epoch_samples": run.get("epoch_samples"),
        "device": run.get("device"),
        "seed": manifest["seed"],
        "loss": manifest["loss"],
        "datasets": manifest["datasets"],
        "seen": seen,
        "unseen": unseen,
        "holdout": {d: {"n_seen": len(h.seen), "n_unseen": len(h.unseen),
                        "unseen": h.unseen, "exempt": h.exempt}
                    for d, h in sorted(holdout.items())},
        "scale_curve": scale_curve(manifest["run_id"]),
        "checks": checks,
        "historical_baseline": ("data/train_baseline.py (TF-IDF + "
                                "LogisticRegression, one pickle per dataset) "
                                "is out of the product path and feeds no "
                                "gate"),
        "verdict": ("GO" if all(c["pass"] for c in checks.values())
                    else "NO-GO"),
        "torch": torch.__version__ if HAVE_TORCH else None,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not unseen_ok:
        gate["no_go_reason"] = (
            "unseen-label accuracy does not beat chance: the model has not "
            "learned to score an option by its text. Do not scale.")
    if write:
        os.makedirs(GATE_DIR, exist_ok=True)
        with open(os.path.join(GATE_DIR, "gate.json"), "w") as fh:
            json.dump(gate, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return gate


def write_gate_json(ckpt_dir, holdout, train_sets, seen, unseen, manifest,
                    run, engine) -> dict:
    """Stage hook: the live evals plus a cold reload of what was just saved."""
    probe = probe_samples(train_sets, 25)
    return compose_gate(
        ckpt_dir, manifest, seen, unseen, holdout,
        holdout_cleanliness(holdout, train_sets, 8000),
        checkpoint_checks(ckpt_dir, engine, probe),
        engine.head.label_free_report(), run)


def run_gate(ckpt_dir: str, device: str = "auto", eval_samples: int = 3000,
             rows_per_dataset: int | None = None, root: str = PREFETCH_DIR,
             write: bool = True) -> dict:
    """Standalone gate: everything measured from a COLD checkpoint."""
    _require_torch()
    engine, manifest = load_checkpoint(ckpt_dir, device)
    config = SamplerConfig(seed=manifest["seed"])
    holdout = build_holdout(root=root)
    tr = train_samplers(holdout, config, rows_per_dataset, root)
    seen = evaluate(engine, eval_samplers(holdout, "seen", config,
                                          rows_per_dataset, root),
                    eval_samples)
    unseen = evaluate(engine, eval_samplers(holdout, "unseen", config,
                                            rows_per_dataset, root),
                      eval_samples)
    run = {"epoch_samples": None, "device": str(engine.device),
           "run_id": manifest["run_id"]}
    return compose_gate(ckpt_dir, manifest, seen, unseen, holdout,
                        holdout_cleanliness(holdout, tr, 8000),
                        checkpoint_checks(ckpt_dir, engine,
                                          probe_samples(tr, 25)),
                        engine.head.label_free_report(), run, write)


# -- CLI -------------------------------------------------------------------

def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="train_decision",
                                 description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    t = sub.add_parser("train", help="listwise training run")
    t.add_argument("--max-samples", type=int, default=250_000)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--seed", type=int, default=DEFAULT_SEED)
    t.add_argument("--device", default="auto")
    t.add_argument("--run-id", default=None)
    t.add_argument("--rows-per-dataset", type=int, default=None)
    t.add_argument("--eval-samples", type=int, default=3000)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--max-length", type=int, default=TRAIN_MAX_LENGTH)
    t.add_argument("--backbone", default=DEFAULT_BACKBONE,
                   help=("which sha256-verified backbone to train the head "
                         "on; #T-bakeoff-real trains one short run per "
                         "candidate to fill the Pareto"))
    t.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL,
                   help=("width of the trainable pointer head "
                         "(#T-antiscale-diag axis 3: 512/1024 repeat the "
                         "250 k -> 1 M segment with a 2x/4x head); "
                         "must stay divisible by the head's 8 attention "
                         "heads"))
    t.add_argument("--no-gate", action="store_true",
                   help="do not rewrite artifacts/gates/T-train-real")
    t.add_argument("--mix-target", type=int, default=None,
                   help=("rows in the capped mixture (#T-corpus-rebalance); "
                         "default: the largest the 15 %%/30 %% caps allow"))
    t.add_argument("--dataset-cap", type=float, default=None,
                   help=("state a dataset cap other than the §65 15 %%. The "
                         "§128 synthetic ablation needs it: without "
                         "synth-v1 the fenced registry covers only 90 %% of "
                         "a mixture. The value lands in run.json"))
    t.add_argument("--family-cap", type=float, default=None,
                   help="likewise for the §66 30 %% family cap")
    t.add_argument("--mix-seed", type=int, default=None,
                   help=("seed of the MIXTURE, when it must differ from the "
                         "training seed: passing the seed of a published "
                         "corpus manifest trains exactly that corpus"))
    t.add_argument("--fence-clean", action="store_true",
                   help=("train `decision-mix-clean-1m` (#T-mix-1m): the "
                         "registry minus the §§18/77 benchmark fence "
                         "(banking77/helpsteer2/pubmedqa), pulled by the "
                         "§86 layer plan, caps at their exact values. It "
                         "defaults --mix-target and --mix-seed to the "
                         "published recipe (data.mix.CLEAN_1M_*), so this "
                         "flag alone assembles the corpus whose manifest "
                         "tools/mix_1m publishes"))
    t.add_argument("--drop-dataset", action="append", default=[],
                   metavar="ID",
                   help=("exclude one source from the mixture; repeatable. "
                         "The §128 synthetic-value arm uses it to build the "
                         "no-synthetic baseline"))
    t.add_argument("--only-dataset", action="append", default=[],
                   metavar="ID",
                   help=("train ONLY these registered sources, naming the "
                         "corpus out loud (the #T-labelspace-div curve uses "
                         "it for the experimental episodic-div mix). "
                         "Experimental sources are selectable this way and "
                         "no other way; combines with --fence-clean for "
                         "the recipe defaults and with --dataset-cap / "
                         "--family-cap for the caps the single-source "
                         "mixture states."))
    t.add_argument("--allow-repeat", action="store_true",
                   help=("train more decisions than the mixture holds, by "
                         "looping it. Without this a budget larger than the "
                         "corpus is a MixShortfallError, because the silent "
                         "version of it published a 1 M-decision curve "
                         "measured on 39 981 rows. With it, "
                         "`epochs_over_corpus` is written to run.json, "
                         "metrics.jsonl and the run's mix.json"))
    t.add_argument("--no-mix", action="store_true",
                   help=("train the raw P0 datasets uncapped — the "
                         "pre-rebalance behaviour. The guardrails still run "
                         "and will abort on the imbalance they find"))
    t.add_argument("--label-dropout", type=float, default=0.0,
                   help=("candidate 1 (#T-gen-objective): fraction of "
                         "options per batch whose text is replaced by a "
                         "novel sentinel, so the memorised text->label map "
                         "never suffices"))
    t.add_argument("--episodic-resample", action="store_true",
                   help=("candidate 2 (#T-gen-objective): redraw every row's "
                         "distractor set from the sampler's own buckets at "
                         "every step, gold kept — the option SET varies, "
                         "not just its order"))
    t.add_argument("--contrastive-weight", type=float, default=0.0,
                   help=("candidate 3 (#T-gen-objective): weight of the "
                         "question<->option-text InfoNCE term added to the "
                         "listwise loss"))
    t.add_argument("--contrastive-tau", type=float, default=0.07,
                   help="temperature of the candidate-3 InfoNCE term")
    t.add_argument("--prior-penalty", type=float, default=0.0,
                   help=("candidate 4 (#T-gen-objective): weight "
                         "subtracting the empirical log train-count from "
                         "each option logit, so frequent labels stop "
                         "winning by frequency"))

    t.add_argument("--unfreeze", default="none", choices=UNFREEZE_MODES,
                   help=("#T-unfreeze-backbone: which encoder weights "
                         "train alongside the head. `none` is the frozen "
                         "baseline every run so far used, `last-n` opens "
                         "the top --unfreeze-layers blocks plus the final "
                         "norm, `full` opens all of them. The regime, the "
                         "blocks it opened and the REAL trainable "
                         "parameter count land in run.json and in every "
                         "checkpoint manifest"))
    t.add_argument("--unfreeze-layers", type=int, default=2, metavar="N",
                   help="how many top blocks --unfreeze last-n opens")
    t.add_argument("--backbone-lr", type=float, default=1e-5,
                   help=("LR of the encoder param group — deliberately far "
                         "below --lr: the head is learning a task, the "
                         "encoder is only being nudged"))

    g = sub.add_parser("gate", help="cold gate over a saved checkpoint")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--device", default="auto")
    g.add_argument("--eval-samples", type=int, default=3000)

    h = sub.add_parser("holdout", help="print the unseen-label plan")
    h.add_argument("--json", action="store_true")

    args = ap.parse_args(argv[1:])
    cmd = args.cmd or "holdout"

    if cmd == "holdout":
        report = holdout_report(build_holdout())
        if getattr(args, "json", False):
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            for d, h in sorted(report["per_dataset"].items()):
                print(f"{d:10s} seen={h['n_seen']:3d} unseen={h['n_unseen']:3d}"
                      f"  {h['exempt'] or ''}")
                if h["unseen"]:
                    print(f"           held out: {', '.join(h['unseen'])}")
        return 0

    if cmd == "train":
        summary = train(max_samples=args.max_samples,
                        batch_size=args.batch_size, lr=args.lr,
                        seed=args.seed, device=args.device,
                        run_id=args.run_id,
                        rows_per_dataset=args.rows_per_dataset,
                        eval_samples=args.eval_samples,
                        log_every=args.log_every,
                        max_length=args.max_length,
                        backbone_id=args.backbone,
                        d_model=args.d_model,
                        write_gate=not args.no_gate,
                        mix_target=args.mix_target,
                        use_mix=not args.no_mix,
                        fence_clean=args.fence_clean,
                        drop_datasets=tuple(args.drop_dataset),
                        only_datasets=tuple(args.only_dataset),
                        mix_seed=args.mix_seed,
                        dataset_cap=args.dataset_cap,
                        family_cap=args.family_cap,
                        allow_repeat=args.allow_repeat,
                        label_dropout=args.label_dropout,
                        episodic_resample=args.episodic_resample,
                        contrastive_weight=args.contrastive_weight,
                        contrastive_tau=args.contrastive_tau,
                        prior_penalty=args.prior_penalty,
                        unfreeze=args.unfreeze,
                        unfreeze_layers=args.unfreeze_layers,
                        backbone_lr=args.backbone_lr)
        print(json.dumps({k: v for k, v in summary.items() if k != "stages"},
                         indent=2, sort_keys=True))
        return 0

    gate = run_gate(args.checkpoint, device=args.device,
                    eval_samples=args.eval_samples)
    print(json.dumps({k: v for k, v in gate.items()
                      if k not in ("checks", "holdout")},
                     indent=2, sort_keys=True))
    for name, check in gate["checks"].items():
        print(f"[gate] {name}: {'PASS' if check['pass'] else 'FAIL'}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
