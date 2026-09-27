"""Public decision datasets → universal schema → `episode-v1` (#T-ingest-public).

The generic bridge of `#data-flywheel`: one fetch, one adapter per dataset
into the universal schema (`data/schema.py`: `choice` / `boolean`), and ONE
function, `universal_to_episode(row, seed)`, from there to the contract the
trainer reads (`data/episode_contract.py`). Laya pre-trained on a mix like
this one (AG News, BoolQ, spam, MASSIVE, XNLI…): external volume buys
domain coverage, not sensitivity — that is what `#episodic-data`'s
counterfactual episodes are for.

What the bridge does, and does NOT do (every rule lands in the manifest
under `rules`, per dataset):

* `state`: the row's text, verbatim. Two-field rows are rendered as
  labelled lines (`Prompt:`/`Response:` for HelpSteer2, `Sentence A:`/
  `Sentence B:` — `Frase A:`/`Frase B:` in Spanish — for PAWS-X). A
  passage dataset keeps the passage as state and its own question (BoolQ)
  or hypothesis (XNLI) goes into the question slot.
* `question`: hand-written templates, per dataset and per language, at
  least three each (`TEMPLATES`); the one used is drawn by seed per row so
  the wording is never a shortcut. The index travels as `question_render`.
* `candidates`: opaque ids `c1..cK` over a seeded permutation of the WHOLE
  label space of the dataset (K = 60 for MASSIVE, 77 for BANKING77…).
  `text` is the label NAME as the dataset publishes it (ClassLabel names
  read from the parquet metadata at fetch, never typed here), with `_`
  shown as a space. No dataset in this batch ships label descriptions, so
  none is written — the task forbids inventing them. Two declared
  exceptions: a binary question renders its classes as `Yes`/`No`
  (`Sí`/`No`), and XNLI's three names are translated for Spanish rows
  (`XNLI_NAMES_ES`) — a name, not a description.
* `evidence`: the datasets annotate none and the contract demands a
  literal fragment of the state. A state of ≤ `WHOLE_STATE_CHARS` is its
  own evidence (`heuristic:whole-state`); a longer one gives the sentence
  that overlaps most with the gold option and the question slot
  (`heuristic:max-overlap-sentence`, first sentence on a tie). Heuristic
  and said so on every episode: it serves the contract, not a verifier.
* `family`: the contract's enumeration is closed; each dataset maps to one
  of the five and the finer key goes to `subfamily`
  (`external/<dataset>/<question_type>`).
* `variant_group`: `<dataset>-<row id>`. MASSIVE and PAWS-X publish one id
  per utterance across locales, so the Spanish and English versions of a
  row share their group.
* eval-only: every held-out cut (`test`, or `validation` when the test has
  no labels), every split of a dataset whose licence does not pass the
  fence, and BANKING77 entirely (it is Jev's benchmark and our unseen
  cut). `assert_trainable` refuses any mixture that names one.

Leakage: every published cut is scanned with `data.leakage`'s exact hash
+ word-trigram Jaccard (≥ `LEAK_THRESHOLD`) against the development
battery, the sealed battery (read for disjointness, never opened for
scoring: `battery_sealed.open_sealed` is not called) and the
typed-decisions test cut. A train row that hits, or whose state also sits
in its own dataset's eval-only cut, goes to `rejects.jsonl`.

Pipeline (CPU, no torch, no training):

    # 1. download pinned by revision + sha256, decode → raw jsonl (needs
    #    pyarrow for parquet: any venv with `pyarrow huggingface_hub`)
    PYTHONPATH=. <python-with-pyarrow> -m data.episode_bridge fetch [--dataset K]
    # 2. convert every cut (stdlib only)
    PYTHONPATH=. .venv-train/bin/python -m data.episode_bridge convert [--dataset K]
    # 3. mix manifest (25 % cap) + gate
    PYTHONPATH=. .venv-train/bin/python -m data.episode_bridge gate
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import convert_typed_decisions as TD  # noqa: E402
from data import episode_contract as EC  # noqa: E402
from data import leakage as LK  # noqa: E402
from data.registry import DatasetCard, Registry  # noqa: E402
from data.schema import Example, Option, Question  # noqa: E402
from data.schema import validate as validate_universal  # noqa: E402

TASK = "T-ingest-public"
BRIDGE_VERSION = "episode-bridge-v1"
SEED = 20260927
#: default ceiling of source rows kept from a TRAIN split (seeded choice,
#: recorded in the raw manifest); held-out cuts are never capped
TRAIN_ROW_CAP = 50_000
#: no dataset may bring more than this share of the external mix
MIX_CAP = 0.25
WHOLE_STATE_CHARS = 280
LEAK_THRESHOLD = 0.5  # LeakageDetector's default similarity threshold

RAW_DIR = os.path.join(ROOT, "artifacts", "data-raw", "public")
OUT_DIR = os.path.join(ROOT, "artifacts", "episodes-external")
MIX_DIR = os.path.join(OUT_DIR, "public-mix")
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", TASK)
GATE_PATH = os.path.join(GATE_DIR, "gate.json")

YES_NO = {"en": ("Yes", "No"), "es": ("Sí", "No")}
XNLI_NAMES_ES = {
    "entailment": "implicación",
    "neutral": "neutral",
    "contradiction": "contradicción",
}
HELPSTEER_ATTRS = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
HELPSTEER_LEVELS = ("0", "1", "2", "3", "4")

#: German function words: deepset/prompt-injections mixes English and
#: German rows and the contract only speaks `en`/`es`
_DE_MARKERS = frozenset(
    "der die das und ist nicht ich sie wie für mit auf ein eine einen zu den "
    "von sind werden über auch oder aber wir ihr mir mich dich bitte sollte "
    "welche gibt".split()
)

EvalOnlyLeak = TD.EvalOnlyLeak


class FenceError(ValueError):
    """The licence fence refused a dataset: nothing is converted."""


# ------------------------------------------------------------ the specs


@dataclass(frozen=True)
class SourceFile:
    """One file of one split (and language) at a pinned revision."""

    split: str
    path: str  # repo path, or a full URL for non-Hub sources
    sha256: str  # expected bytes; `fetch` refuses anything else
    fmt: str  # parquet | jsonl.gz | csv
    lang: str = "en"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    hf_id: str
    revision: str
    license: str | None  # what the card declares, verbatim-ish
    license_evidence: str  # where/what we read
    usage: str  # train | eval-only
    usage_reason: str
    question_type: str  # categorical | binary | ordinal | multiple-choice
    family: str
    adapter: str  # name in ADAPTERS
    files: tuple[SourceFile, ...]
    train_split: str | None = "train"
    heldout_splits: tuple[str, ...] = ("test",)
    heldout_note: str = ""
    train_row_cap: int = TRAIN_ROW_CAP
    label_column: str | None = None  # ClassLabel column read at fetch
    config: str = "default"
    card_revision: str | None = None  # when the files live on another ref
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def origin(self) -> str:
        return f"public-{self.key}"

    @property
    def splits(self) -> tuple[str, ...]:
        out = [self.train_split] if self.train_split else []
        return tuple(out + list(self.heldout_splits))

    @property
    def langs(self) -> tuple[str, ...]:
        return tuple(sorted({f.lang for f in self.files}))

    def is_eval_only(self, split: str) -> bool:
        return self.usage != "train" or split != self.train_split


def _hf(split: str, path: str, sha: str, fmt: str = "parquet", lang: str = "en") -> SourceFile:
    return SourceFile(split, path, sha, fmt, lang)


SPECS: dict[str, DatasetSpec] = {
    s.key: s
    for s in (
        DatasetSpec(
            key="massive",
            hf_id="AmazonScience/massive",
            # parquet export of the script dataset (refs/convert/parquet);
            # the card (and its licence) is read at main ff6bd8e4…
            revision="ed58ac423a2f4121720918bf5301577edce4ffd3",
            card_revision="ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740",
            license="CC-BY-4.0",
            license_evidence="card YAML `license: cc-by-4.0` at ff6bd8e4",
            usage="train",
            usage_reason="CC-BY-4.0 passes the fence; attribution kept in the card",
            question_type="categorical",
            family=EC.DESCRIPTION,
            adapter="massive",
            label_column="intent",
            config="en-US+es-ES",
            files=(
                # non-LFS files on the convert ref: sha256 of the bytes as
                # first downloaded on 2026-09-27 (trust on first use)
                _hf("train", "en-US/train/0000.parquet",
                    "9c8b3ce8a96ec3b0d4aee104d8778d9f6bacdbd18f2cea232f6d8333d24401b1"),
                _hf("test", "en-US/test/0000.parquet",
                    "c418da9a5f3a7425b5cf44941bcea032a31040dbda2da85e0dd9323785c7731e"),
                _hf("train", "es-ES/train/0000.parquet",
                    "e8f98e62b3ab88bbf974033647d79a44162bc555c559d0a8a0a201ccba596c0d",
                    lang="es"),
                _hf("test", "es-ES/test/0000.parquet",
                    "ca94eebc836e61e8bc530966c6565bb2c2b9d6184cc910c767c7fa6a9563de19",
                    lang="es"),
            ),
        ),
    )
}

# XNLI: the card declares no licence (no `license:` key in its YAML, and
# `### Licensing Information` reads «[More Information Needed]»). The
# upstream corpus is widely cited as CC-BY-NC-4.0, which the fence would
# refuse anyway. Undeclared → eval-only, converted, annotated.
SPECS["xnli"] = DatasetSpec(
    key="xnli",
    hf_id="facebook/xnli",
    revision="b8dd5d7af51114dbda02c0e3f6133f332186418e",
    license="undeclared",
    license_evidence="card has no `license:` key; «Licensing Information: [More "
    "Information Needed]» at b8dd5d7a",
    usage="eval-only",
    usage_reason="licence not declared in the card → eval-only (register rule)",
    question_type="categorical",
    family=EC.INFERENCE,
    adapter="xnli",
    label_column="label",
    config="en+es",
    train_row_cap=25_000,  # per language: 50 000 rows in all
    files=(
        _hf("train", "en/train-00000-of-00001.parquet",
            "6be63cec9ec932a5790de82caee7ee4d00e5a50ef19b79f23b740732da143424"),
        _hf("test", "en/test-00000-of-00001.parquet",
            "8f0fd1e105091e0f11cd37f9b2bc382f16de9949aa9471e1366b2605ba037167"),
        _hf("train", "es/train-00000-of-00001.parquet",
            "f609c655a0c200168235609e7709bcf3ac48a465b4c6e90380093862a53a3a2a", lang="es"),
        _hf("test", "es/test-00000-of-00001.parquet",
            "c0159c9a734d7d0683d20a9850adcf3e214a49179b6517fea5868744bfd99886", lang="es"),
    ),
)

SPECS.update(
    {
        s.key: s
        for s in (
            DatasetSpec(
                key="paws-x",
                hf_id="google-research-datasets/paws-x",
                revision="4cd8187c404bda33cb1f62b49b001115862acf37",
                license="PAWS-X custom (free use, attribution appreciated)",
                license_evidence="card YAML `license: other`; Licensing Information: «may "
                "be freely used for any purpose, although acknowledgement of Google LLC "
                "… would be appreciated» at 4cd8187c",
                usage="eval-only",
                usage_reason="custom licence not in registry.TRAIN_APPROVED_LICENSES: "
                "the fence refuses train until the operator approves it",
                question_type="binary",
                family=EC.INFERENCE,
                adapter="paws_x",
                config="en+es",
                files=(
                    _hf("train", "en/train-00000-of-00001.parquet",
                        "0de6e8a4ba4bd88ee79c883349d112ec1ca85cda78726b59c215fab5849e49f3"),
                    _hf("test", "en/test-00000-of-00001.parquet",
                        "fdb1e43f1068cf0cec955327cfdedc2d3065f15d3cd673c9d0002e6e87cc7915"),
                    _hf("train", "es/train-00000-of-00001.parquet",
                        "9501c5d9ebac5b7ad95970d1cb1a98798a1b16b6ffa89e48328c20f9c2b04df3",
                        lang="es"),
                    _hf("test", "es/test-00000-of-00001.parquet",
                        "5ba872e113a79c783616e59d3d28ec0e922b6db745d797e5eb95010a32928bee",
                        lang="es"),
                ),
                train_row_cap=25_000,
            ),
            DatasetSpec(
                key="banking77",
                hf_id="PolyAI/banking77",
                # the Hub repo is a loading script over the original GitHub
                # CSVs; the bytes are fetched from that repo at a pinned commit
                revision="57ec275d8078af65b7731c2a98be812d844a6d6b",
                card_revision="90d4e2ee5521c04fc1488f065b8b083658768c57",
                license="CC-BY-4.0",
                license_evidence="card YAML `license: cc-by-4.0` at 90d4e2ee; bytes from "
                "github.com/PolyAI-LDN/task-specific-datasets@57ec275d",
                usage="eval-only",
                usage_reason="Jev's published benchmark (0,870 K=77) and our unseen cut: "
                "eval-only always, whatever the licence",
                question_type="categorical",
                family=EC.DESCRIPTION,
                adapter="banking77",
                files=(
                    SourceFile("train", "https://raw.githubusercontent.com/PolyAI-LDN/"
                               "task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/"
                               "banking_data/train.csv",
                               "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
                               "csv"),
                    SourceFile("test", "https://raw.githubusercontent.com/PolyAI-LDN/"
                               "task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/"
                               "banking_data/test.csv",
                               "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
                               "csv"),
                ),
            ),
            DatasetSpec(
                key="go-emotions",
                hf_id="google-research-datasets/go_emotions",
                revision="add492243ff905527e67aeb8b80c082af02207c3",
                license="Apache-2.0",
                license_evidence="card YAML `license: apache-2.0` at add49224",
                usage="train",
                usage_reason="Apache-2.0 passes the fence",
                question_type="categorical",
                family=EC.DESCRIPTION,
                adapter="go_emotions",
                label_column="labels",
                config="simplified",
                files=(
                    _hf("train", "simplified/train-00000-of-00001.parquet",
                        "b7d74279616ae7c9b8374ab62ea9f9d6504d36a577bb17f745d720dc2b0d4e76"),
                    _hf("test", "simplified/test-00000-of-00001.parquet",
                        "fd0953e535ba2569edc6a1daaa1133f8e4b9071691d540c9bab812fda132bc26"),
                ),
                notes=("multi-label rows publish `acceptable_answers` (any listed "
                       "emotion is correct) instead of a single `answer`",),
            ),
            DatasetSpec(
                key="commonsense-qa",
                hf_id="tau/commonsense_qa",
                revision="94630fe30dad47192a8546eb75f094926d47e155",
                license="MIT",
                license_evidence="card YAML `license: mit` at 94630fe3",
                usage="train",
                usage_reason="MIT passes the fence",
                question_type="multiple-choice",
                family=EC.PRIORITY,
                adapter="commonsense_qa",
                heldout_splits=("validation",),
                heldout_note="the published test split carries no answers; validation "
                "is the held-out cut, eval-only",
                files=(
                    _hf("train", "data/train-00000-of-00001.parquet",
                        "b0449767ed986bfc2ca52b1244a46ef12f732756727f3cb0a4ab69ac8b3d282b"),
                    _hf("validation", "data/validation-00000-of-00001.parquet",
                        "bdbd9bf9cc4d2349b24901038b2ab2f58e10e4e507ad2fd425dca55cd3cb6660"),
                ),
            ),
            DatasetSpec(
                key="prompt-injections",
                hf_id="deepset/prompt-injections",
                revision="4f61ecb038e9c3fb77e21034b22511b523772cdd",
                license="Apache-2.0",
                license_evidence="card YAML `license: apache-2.0` at 4f61ecb0 (the "
                "dataset_info block also says cc-by-4.0; both pass the fence)",
                usage="train",
                usage_reason="Apache-2.0 passes the fence",
                question_type="binary",
                family=EC.DESCRIPTION,
                adapter="prompt_injections",
                files=(
                    _hf("train", "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
                        "2e10bc7ab30f542c97e4e83e2a5683000b5057d25ec10908784c631d44124c04"),
                    _hf("test", "data/test-00000-of-00001-701d16158af87368.parquet",
                        "39ac797cabc157eeed58435a08593b2952bb6cb16fc394a2d383f447cc7b246e"),
                ),
                notes=("the card names no labels: label 1 = injection is read from "
                       "the rows and stated in the question, never in a candidate",
                       "German rows are rejected (heuristic:german-marker): the "
                       "contract speaks en/es only"),
            ),
            DatasetSpec(
                key="helpsteer2",
                hf_id="nvidia/HelpSteer2",
                revision="990b2711a36180dd19d9c94b8627844866f8982a",
                license="CC-BY-4.0",
                license_evidence="card YAML `license: cc-by-4.0` at 990b2711",
                usage="train",
                usage_reason="CC-BY-4.0 passes the fence",
                question_type="ordinal",
                family=EC.DESCRIPTION,
                adapter="helpsteer2",
                heldout_splits=("validation",),
                heldout_note="HelpSteer2 publishes train/validation only; validation is "
                "the held-out cut, eval-only",
                train_row_cap=6_000,  # × 5 attributes = 30 000 decisions, long states
                files=(
                    _hf("train", "train.jsonl.gz",
                        "c0d7e91d738d42e8a08070db26c4c09a9c7631308e1f0fd380ff43d130c9f713",
                        "jsonl.gz"),
                    _hf("validation", "validation.jsonl.gz",
                        "610eeb5289494d613c4c0f70aade2df8df0b499f3a24e76d232f74e6909d010a",
                        "jsonl.gz"),
                ),
            ),
            DatasetSpec(
                key="ag-news",
                hf_id="fancyzhx/ag_news",
                revision="eb185aade064a813bc0b7f42de02595523103ca4",
                license="unknown (declared by the card)",
                license_evidence="card YAML `license: unknown`; Licensing Information: "
                "«[More Information Needed]» at eb185aad",
                usage="eval-only",
                usage_reason="licence not declared (the card says unknown) → eval-only",
                question_type="categorical",
                family=EC.DESCRIPTION,
                adapter="ag_news",
                label_column="label",
                files=(
                    _hf("train", "data/train-00000-of-00001.parquet",
                        "fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee"),
                    _hf("test", "data/test-00000-of-00001.parquet",
                        "71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb"),
                ),
            ),
            DatasetSpec(
                key="boolq",
                hf_id="google/boolq",
                revision="35b264d03638db9f4ce671b711558bf7ff0f80d5",
                license="CC-BY-SA-3.0",
                license_evidence="card YAML `license: cc-by-sa-3.0`; «BoolQ is released "
                "under the Creative Commons Share-Alike 3.0 license» at 35b264d0",
                usage="eval-only",
                usage_reason="share-alike is gated by registry.TRAIN_GATED['boolq'] until "
                "the operator documents the SA obligations",
                question_type="binary",
                family=EC.INFERENCE,
                adapter="boolq",
                heldout_splits=("validation",),
                heldout_note="BoolQ publishes train/validation only",
                files=(
                    _hf("train", "data/train-00000-of-00001.parquet",
                        "4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623"),
                    _hf("validation", "data/validation-00000-of-00001.parquet",
                        "52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8"),
                ),
            ),
            DatasetSpec(
                key="emotion",
                hf_id="dair-ai/emotion",
                revision="cab853a1dbdf4c42c2b3ef2173804746df8825fe",
                license="research-only (card)",
                license_evidence="card YAML `license: other`; «The dataset should be used "
                "for educational and research purposes only.» at cab853a1",
                usage="eval-only",
                usage_reason="research-only licence → eval-only (Laya holds it out too)",
                question_type="categorical",
                family=EC.DESCRIPTION,
                adapter="emotion",
                label_column="label",
                config="split",
                files=(
                    _hf("train", "split/train-00000-of-00001.parquet",
                        "10817f0f2ea42358bc62f69a09dfb8bd71701727df6d5a387bea742f3ea06417"),
                    _hf("test", "split/test-00000-of-00001.parquet",
                        "6f8407fa1ca9c310f55781f082ed73812f6551e8dda2c61973123a121869245b"),
                ),
            ),
        )
    }
)

# ------------------------------------------------------------ templates

#: hand-written, ≥ 3 per dataset and language; `{…}` slots come from the row
TEMPLATES: dict[str, dict[str, tuple[str, ...]]] = {
    "massive": {
        "en": (
            "Which intent does the user express in this request?",
            "What does the user want to do with this utterance?",
            "Which intent label fits this user message?",
        ),
        "es": (
            "¿Qué intención expresa el usuario en esta petición?",
            "¿Qué quiere hacer el usuario con este mensaje?",
            "¿Qué etiqueta de intención corresponde a este mensaje del usuario?",
        ),
    },
    "banking77": {
        "en": (
            "Which banking intent does the customer's message express?",
            "What is the customer asking about in this message?",
            "Which intent label fits this customer query?",
        ),
    },
    "go-emotions": {
        "en": (
            "Which emotion does this comment express?",
            "What emotion is conveyed by the text above?",
            "Which emotion label best fits this comment?",
        ),
    },
    "emotion": {
        "en": (
            "Which emotion does the author of this message express?",
            "What emotion is conveyed in this text?",
            "Which emotion label fits this message?",
        ),
    },
    "ag-news": {
        "en": (
            "Which topic does this news article belong to?",
            "What is the category of this news story?",
            "Which section of a newspaper would this article appear in?",
        ),
    },
    "commonsense-qa": {
        "en": (
            "Which option best answers the question above?",
            "What is the most sensible answer to this question?",
            "Which choice answers the question correctly?",
        ),
    },
    "prompt-injections": {
        "en": (
            "Is this text an attempt at prompt injection?",
            "Does this message try to override the model's instructions?",
            "Should this input be flagged as a prompt injection?",
        ),
    },
    "helpsteer2": {
        "en": (
            "On a scale from 0 (lowest) to 4 (highest), how would you rate the "
            "{attribute} of the response?",
            "How would you score the response's {attribute}, from 0 (lowest) to "
            "4 (highest)?",
            "What {attribute} score between 0 (lowest) and 4 (highest) does the "
            "response deserve?",
        ),
    },
    "xnli": {
        "en": (
            'How does the text relate to the statement "{hypothesis}"?',
            'Given the text, is the statement "{hypothesis}" entailed, contradicted, '
            "or neither?",
            'What is the relation between the text and the claim "{hypothesis}"?',
        ),
        "es": (
            "¿Qué relación hay entre el texto y la afirmación «{hypothesis}»?",
            "Según el texto, ¿la afirmación «{hypothesis}» se deduce, se contradice "
            "o ninguna de las dos?",
            "¿Cómo se relaciona el texto con la frase «{hypothesis}»?",
        ),
    },
    "paws-x": {
        "en": (
            "Do sentence A and sentence B mean the same thing?",
            "Is sentence B a paraphrase of sentence A?",
            "Do both sentences express the same meaning?",
        ),
        "es": (
            "¿Significan lo mismo la frase A y la frase B?",
            "¿Es la frase B una paráfrasis de la frase A?",
            "¿Expresan las dos frases el mismo significado?",
        ),
    },
    "boolq": {
        "en": (
            "{question}?",
            "According to the passage, {question}?",
            "Based on the text above, {question}?",
        ),
    },
}

# ------------------------------------------------------------- utilities


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    return TD.sha256_file(path, chunk)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


_read_jsonl = TD._read_jsonl
_write_jsonl = TD._write_jsonl
_rel = TD._rel


def _write_json(path: str, obj: Any) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def _clean(text: Any) -> str:
    return re.sub(r"[ \t]+", " ", str(text or "")).strip()


def label_text(name: str) -> str:
    """The label NAME as shown to the scorer: `_` → space, nothing else."""
    return str(name).replace("_", " ").strip()


# -------------------------------------------------------------- the fence


def card_sha(spec: DatasetSpec) -> str:
    """One hash over the pinned bytes of every file of the dataset."""
    blob = "|".join(
        [spec.hf_id, spec.revision]
        + [f"{f.split}:{f.lang}:{f.path}:{f.sha256}" for f in spec.files]
    )
    return sha256_text(blob)


def dataset_card(spec: DatasetSpec) -> DatasetCard:
    return DatasetCard(
        id=spec.key,
        source_original=f"https://huggingface.co/datasets/{spec.hf_id}",
        mirror=spec.hf_id,
        license=spec.license or "",
        usage=spec.usage,
        revision=spec.revision,
        sha256=card_sha(spec),
        transform=BRIDGE_VERSION,
    )


def fence(spec: DatasetSpec) -> dict:
    """`data.registry`'s fence over this dataset. Raises `FenceError` when
    the card cannot even be registered (no declared licence, no revision)
    or when a dataset declared `train` fails the licence check: in both
    cases nothing is converted. An `eval-only` dataset is converted, every
    cut marked eval-only."""
    reg = Registry()
    card = dataset_card(spec)
    try:
        reg.register(card)
    except ValueError as exc:
        raise FenceError(f"{spec.key}: fence refused the card: {exc}") from exc
    ok, reason = reg.train_ok(spec.key)
    if spec.usage == "train" and not ok:
        raise FenceError(f"{spec.key}: declared train but the fence says: {reason}")
    return {
        "license": spec.license,
        "usage": spec.usage,
        "train_ok": ok,
        "fence_reason": reason,
        "card_sha256": card.sha256,
    }


# ---------------------------------------------------- the universal rows


def universal_row(
    *,
    dataset: str,
    row_id: str,
    lang: str,
    split: str,
    state: str,
    qid: str,
    kind: str,
    options: list[tuple[str, str]],
    answer: str | None,
    group: str | None = None,
    slots: dict | None = None,
    acceptable: list[str] | None = None,
) -> dict:
    """A universal-schema example (one question) plus the bridge's `meta`.

    `options` are `(label, text)`: the label is the dataset's own key (it
    becomes the universal option id), the text is what the scorer reads.
    The universal `split` is `train` or `test` (the schema's vocabulary);
    the source split name travels in `meta.split`.
    """
    ex = Example(
        state=state,
        split="train" if split == "train" else "test",
        questions=[
            Question(
                id=qid,
                kind=kind,
                options=[Option(id=lab, text=txt) for lab, txt in options],
                answer=answer,
            )
        ],
    )
    row = asdict(ex)
    row["meta"] = {
        "dataset": dataset,
        "row_id": str(row_id),
        "lang": lang,
        "split": split,
        "group": group or f"{dataset}-{row_id}",
        "slots": dict(slots or {}),
        "acceptable": list(acceptable) if acceptable else None,
    }
    return row


def _example_of(row: dict) -> Example:
    return Example(
        state=row.get("state", ""),
        split=row.get("split", "train"),
        questions=[
            Question(
                id=q["id"],
                kind=q["kind"],
                options=[Option(**o) for o in q.get("options", [])],
                answer=q.get("answer"),
                teacher_conf=q.get("teacher_conf"),
                weights=q.get("weights"),
            )
            for q in row.get("questions", [])
        ],
    )


# ------------------------------------------------------------- evidence

_SENT = re.compile(r"(?<=[.!?…])\s+|\n+")
_WORD = re.compile(r"\w+", re.UNICODE)


def _toks(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if len(w) >= 3}


def pick_evidence(state: str, hints: Iterable[str]) -> tuple[str, str]:
    """`(fragment, method)`: the whole state when short, else the sentence
    that overlaps most with the hints (gold text, question slot)."""
    s = state.strip()
    if len(s) <= WHOLE_STATE_CHARS:
        return s, "heuristic:whole-state"
    sents = [x.strip() for x in _SENT.split(s) if x and x.strip()]
    if not sents:
        return s, "heuristic:whole-state"
    ht: set[str] = set()
    for h in hints:
        ht |= _toks(h)
    best, best_score = sents[0], -1
    for sent in sents:
        score = len(ht & _toks(sent))
        if score > best_score:
            best, best_score = sent, score
    return best, "heuristic:max-overlap-sentence"


# ----------------------------------------------------------- the bridge


def _rng(seed: int, *parts: str) -> random.Random:
    return random.Random("\x00".join([str(seed), *parts]))


def is_reject(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("rejected") is True


def _reject(row: dict, reasons: list[str], stage: str) -> dict:
    meta = row.get("meta", {}) if isinstance(row, dict) else {}
    qs = row.get("questions") if isinstance(row, dict) else None
    qid = qs[0].get("id") if isinstance(qs, list) and qs and isinstance(qs[0], dict) else "?"
    return {
        "rejected": True,
        "id": f"{meta.get('dataset', '?')}-{meta.get('lang', '?')}-{qid}",
        "dataset": meta.get("dataset"),
        "row_id": meta.get("row_id"),
        "lang": meta.get("lang"),
        "split": meta.get("split"),
        "stage": stage,
        "reasons": reasons,
    }


def universal_to_episode(row: dict, seed: int = SEED) -> dict:
    """One universal-schema row (one question) → an `episode-v1` dict, or a
    reject (`{"rejected": True, "reasons": [...]}`; see `is_reject`).

    Nothing is fixed by hand: a row the universal schema refuses, a
    dataset with no templates for the row's language, or an episode the
    contract refuses, all come back as rejects with their reasons.
    """
    if not isinstance(row, dict) or "meta" not in row:
        return _reject(row if isinstance(row, dict) else {}, ["row has no bridge meta"], "bridge")
    meta = row["meta"]
    dataset = meta.get("dataset")
    spec = SPECS.get(dataset)
    if spec is None:
        return _reject(row, [f"unknown dataset {dataset!r}"], "bridge")
    errs = validate_universal(_example_of(row))
    if errs:
        return _reject(row, [f"universal schema: {e}" for e in errs], "universal")
    if len(row["questions"]) != 1:
        return _reject(row, ["bridge takes exactly one question per row"], "bridge")
    q = row["questions"][0]
    lang = meta.get("lang")
    templates = TEMPLATES.get(dataset, {}).get(lang)
    if not templates:
        return _reject(row, [f"no templates for {dataset}/{lang}"], "bridge")
    row_id, qid = str(meta.get("row_id")), str(q["id"])
    t_idx = _rng(seed, dataset, lang, row_id, qid, "question").randrange(len(templates))
    try:
        question = templates[t_idx].format(**meta.get("slots", {}))
    except (KeyError, IndexError) as exc:
        return _reject(row, [f"template slot missing: {exc}"], "bridge")
    question = re.sub(r"\s+", " ", question).strip()

    options = q["options"]
    order = list(range(len(options)))
    _rng(seed, dataset, lang, row_id, qid, "candidates").shuffle(order)
    cands = [{"id": f"c{k + 1}", "text": options[j]["text"]} for k, j in enumerate(order)]
    label_to_id = {options[j]["id"]: f"c{k + 1}" for k, j in enumerate(order)}

    acceptable = meta.get("acceptable")
    answer = acceptable_ids = None
    if acceptable:
        missing = [a for a in acceptable if a not in label_to_id]
        if missing:
            return _reject(row, [f"acceptable labels not options: {missing}"], "bridge")
        acceptable_ids = sorted(label_to_id[a] for a in acceptable)
        gold_labels = list(acceptable)
    else:
        if q.get("answer") in (None, "unknown"):
            return _reject(row, ["row has no gold answer"], "bridge")
        answer = label_to_id[q["answer"]]
        gold_labels = [q["answer"]]
    gold_texts = [o["text"] for o in options if o["id"] in gold_labels]
    evidence, how = pick_evidence(row["state"], gold_texts + list(meta.get("slots", {}).values()))
    qtype = spec.question_type
    source_split = meta.get("split", "train")
    ep = {
        "id": f"pub-{dataset}-{lang}-{source_split}-{row_id}-{qid}",
        "schema_version": EC.SCHEMA_VERSION,
        "state": row["state"],
        "question": question,
        "candidates": cands,
        "answer": answer,
        "acceptable_answers": acceptable_ids,
        "preference": None,
        "evidence": evidence,
        "evidence_origin": how,
        "family": spec.family,
        "subfamily": f"external/{dataset}/{qtype}",
        "question_type": qtype,
        "question_render": f"template[{t_idx}]",
        "lang": lang,
        "origin": spec.origin,
        "generator_seed": seed,
        "generator_version": BRIDGE_VERSION,
        "variant_group": meta.get("group") or f"{dataset}-{row_id}",
        "split": source_split,
        "eval_only": spec.is_eval_only(source_split),
        "license": spec.license,
        "source": {
            "dataset": spec.hf_id,
            "config": spec.config,
            "revision": spec.revision,
            "split": source_split,
            "row_id": row_id,
            "question": qid,
            "labels": gold_labels,
        },
    }
    reasons = EC.validate(ep)
    if reasons:
        return _reject(row, reasons, "validate")
    return ep


# ------------------------------------------------------------ adapters
# raw row (as decoded at fetch) → universal rows. `names` are the dataset's
# ClassLabel names read from the parquet metadata (or the CSV universe).


def _looks_german(text: str) -> bool:
    words = set(re.findall(r"[a-zäöüß]+", text.lower()))
    return bool(re.search(r"[äöüß]", text.lower())) or len(words & _DE_MARKERS) >= 2


def _cat_options(names: list[str]) -> list[tuple[str, str]]:
    return [(n, label_text(n)) for n in names]


def _binary(lang: str) -> list[tuple[str, str]]:
    yes, no = YES_NO[lang]
    return [("yes", yes), ("no", no)]


def adapt_massive(raw: dict, split: str, lang: str, names: list[str], idx: int) -> list[dict]:
    return [universal_row(
        dataset="massive", row_id=raw["id"], lang=lang, split=split,
        state=_clean(raw["utt"]), qid="intent", kind="choice",
        options=_cat_options(names), answer=names[int(raw["intent"])],
        group=f"massive-{raw['id']}",
    )]


def adapt_banking77(raw: dict, split: str, lang: str, names: list[str], idx: int) -> list[dict]:
    return [universal_row(
        dataset="banking77", row_id=f"{split}{idx}", lang=lang, split=split,
        state=_clean(raw["text"]), qid="intent", kind="choice",
        options=_cat_options(names), answer=raw["category"],
    )]


def _single_label(dataset: str, raw: dict, split: str, lang: str,
                  names: list[str], idx: int, qid: str) -> list[dict]:
    return [universal_row(
        dataset=dataset, row_id=str(idx), lang=lang, split=split,
        state=_clean(raw["text"]), qid=qid, kind="choice",
        options=_cat_options(names), answer=names[int(raw["label"])],
    )]


def adapt_ag_news(raw, split, lang, names, idx):
    return _single_label("ag-news", raw, split, lang, names, idx, "topic")


def adapt_emotion(raw, split, lang, names, idx):
    return _single_label("emotion", raw, split, lang, names, idx, "emotion")


def adapt_go_emotions(raw: dict, split: str, lang: str, names: list[str], idx: int) -> list[dict]:
    labels = [names[int(i)] for i in raw["labels"]]
    if not labels:
        raise ValueError("row carries no emotion label")
    return [universal_row(
        dataset="go-emotions", row_id=raw["id"], lang=lang, split=split,
        state=_clean(raw["text"]), qid="emotion", kind="choice",
        options=_cat_options(names), answer=labels[0],
        acceptable=labels if len(labels) > 1 else None,
    )]


def adapt_commonsense_qa(raw: dict, split: str, lang: str, names: list[str], idx: int) -> list[dict]:
    ch = raw["choices"]
    opts = [(str(lab), _clean(txt)) for lab, txt in zip(ch["label"], ch["text"])]
    return [universal_row(
        dataset="commonsense-qa", row_id=raw["id"], lang=lang, split=split,
        state=_clean(raw["question"]), qid="answer", kind="choice",
        options=opts, answer=str(raw["answerKey"]) or None,
    )]


def adapt_prompt_injections(raw: dict, split: str, lang: str, names: list[str], idx: int) -> list[dict]:
    text = _clean(raw["text"])
    if _looks_german(text):
        raise ValueError("lang: row looks German (heuristic:german-marker), contract is en/es")
    return [universal_row(
        dataset="prompt-injections", row_id=f"{split}{idx}", lang=lang, split=split,
        state=text, qid="injection", kind="boolean", options=_binary(lang),
        answer="yes" if int(raw["label"]) == 1 else "no",
    )]


def adapt_helpsteer2(raw: dict, split: str, lang: str, names: list[str], idx: int) -> list[dict]:
    prompt, resp = str(raw["prompt"]).strip(), str(raw["response"]).strip()
    rid = sha256_text(prompt + "\x00" + resp)[:16]
    state = f"Prompt: {prompt}\n\nResponse: {resp}"
    out = []
    for attr in HELPSTEER_ATTRS:
        v = raw.get(attr)
        out.append(universal_row(
            dataset="helpsteer2", row_id=rid, lang=lang, split=split,
            state=state, qid=attr, kind="choice",
            options=[(lv, lv) for lv in HELPSTEER_LEVELS],
            answer=None if v is None else str(int(v)),
            slots={"attribute": attr},
        ))
    return out


def adapt_xnli(raw: dict, split: str, lang: str, names: list[str], idx: int) -> list[dict]:
    label = int(raw["label"])
    if label < 0:
        raise ValueError("unlabelled pair (label -1)")
    shown = (lambda n: XNLI_NAMES_ES[n]) if lang == "es" else label_text
    return [universal_row(
        dataset="xnli", row_id=f"{split}{idx}", lang=lang, split=split,
        state=_clean(raw["premise"]), qid="relation", kind="choice",
        options=[(n, shown(n)) for n in names], answer=names[label],
        slots={"hypothesis": _clean(raw["hypothesis"])},
    )]


def adapt_paws_x(raw: dict, split: str, lang: str, names: list[str], idx: int) -> list[dict]:
    a, b = ("Frase A", "Frase B") if lang == "es" else ("Sentence A", "Sentence B")
    s1, s2 = _clean(raw["sentence1"]), _clean(raw["sentence2"])
    if not s1 or not s2 or s1.upper() == "NS" or s2.upper() == "NS":
        raise ValueError("empty or NS (untranslated) sentence")
    return [universal_row(
        dataset="paws-x", row_id=raw["id"], lang=lang, split=split,
        state=f"{a}: {s1}\n{b}: {s2}", qid="paraphrase", kind="boolean",
        options=_binary(lang), answer="yes" if int(raw["label"]) == 1 else "no",
        group=f"paws-x-{split}-{raw['id']}",
    )]


def adapt_boolq(raw: dict, split: str, lang: str, names: list[str], idx: int) -> list[dict]:
    q = _clean(raw["question"]).rstrip("?").strip()
    return [universal_row(
        dataset="boolq", row_id=f"{split}{idx}", lang=lang, split=split,
        state=_clean(raw["passage"]), qid="answer", kind="boolean",
        options=_binary(lang), answer="yes" if raw["answer"] else "no",
        slots={"question": q},
    )]


ADAPTERS: dict[str, Callable[..., list[dict]]] = {
    "massive": adapt_massive,
    "banking77": adapt_banking77,
    "ag_news": adapt_ag_news,
    "emotion": adapt_emotion,
    "go_emotions": adapt_go_emotions,
    "commonsense_qa": adapt_commonsense_qa,
    "prompt_injections": adapt_prompt_injections,
    "helpsteer2": adapt_helpsteer2,
    "xnli": adapt_xnli,
    "paws_x": adapt_paws_x,
    "boolq": adapt_boolq,
}


def convert_raw_rows(
    spec: DatasetSpec, rows: Iterable[dict], split: str, lang: str,
    names: list[str], seed: int = SEED,
) -> tuple[list[dict], list[dict]]:
    """Raw rows of one split/language → (episodes, rejects)."""
    adapter = ADAPTERS[spec.adapter]
    episodes: list[dict] = []
    rejects: list[dict] = []
    for idx, raw in enumerate(rows):
        ridx = raw.get("_row", idx) if isinstance(raw, dict) else idx
        try:
            urows = adapter(raw, split, lang, names, ridx)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            rejects.append({
                "rejected": True, "id": f"{spec.key}-{lang}-{split}-{ridx}",
                "dataset": spec.key, "row_id": str(ridx), "lang": lang,
                "split": split, "stage": "adapt",
                "reasons": [f"{type(exc).__name__}: {exc}"],
            })
            continue
        for urow in urows:
            out = universal_to_episode(urow, seed)
            (rejects if is_reject(out) else episodes).append(out)
    return episodes, rejects


# -------------------------------------------------------------- the fetch


def _download(spec: DatasetSpec, f: SourceFile) -> str:
    if f.path.startswith("https://"):
        cache = os.path.join(RAW_DIR, spec.key, "_download")
        os.makedirs(cache, exist_ok=True)
        dest = os.path.join(cache, os.path.basename(f.path))
        if not os.path.exists(dest) or sha256_file(dest) != f.sha256:
            with urllib.request.urlopen(f.path, timeout=120) as resp:  # noqa: S310
                data = resp.read()
            with open(dest, "wb") as fh:
                fh.write(data)
        return dest
    from huggingface_hub import hf_hub_download  # optional dep, fetch only

    return hf_hub_download(spec.hf_id, f.path, repo_type="dataset", revision=spec.revision)


def _decode(path: str, fmt: str) -> tuple[list[dict], dict[str, list[str]]]:
    """Rows + ClassLabel names found in the file's HF metadata."""
    if fmt == "parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "decoding parquet needs pyarrow, which the hash-locked .venv-train "
                "does not carry: run `fetch` from a venv with `pyarrow "
                "huggingface_hub`; `convert` and `gate` are stdlib"
            ) from exc
        table = pq.read_table(path)
        names: dict[str, list[str]] = {}
        md = table.schema.metadata or {}
        if b"huggingface" in md:
            feats = json.loads(md[b"huggingface"]).get("info", {}).get("features", {})
            for col, feat in feats.items():
                if isinstance(feat, dict) and feat.get("_type") == "ClassLabel":
                    names[col] = list(feat["names"])
                elif isinstance(feat, dict) and feat.get("_type") == "Sequence":
                    inner = feat.get("feature", {})
                    if inner.get("_type") == "ClassLabel":
                        names[col] = list(inner["names"])
                elif isinstance(feat, list) and feat and feat[0].get("_type") == "ClassLabel":
                    names[col] = list(feat[0]["names"])
        return table.to_pylist(), names
    if fmt == "jsonl.gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()], {}
    if fmt == "csv":
        with open(path, encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(io.StringIO(fh.read()))), {}
    raise ValueError(f"unknown format {fmt!r}")


def select_rows(n: int, cap: int, seed: int, key: str) -> list[int]:
    """Seeded choice of `cap` row indices out of `n`, returned in file order."""
    if n <= cap:
        return list(range(n))
    ranked = sorted(range(n), key=lambda i: sha256_text(f"{seed}\x00{key}\x00{i}"))
    return sorted(ranked[:cap])


def raw_path(spec: DatasetSpec, split: str, lang: str, raw_dir: str = RAW_DIR) -> str:
    return os.path.join(raw_dir, spec.key, f"{split}.{lang}.jsonl")


def fetch(spec: DatasetSpec, raw_dir: str = RAW_DIR, seed: int = SEED) -> dict:
    """Download every file at the pinned revision, verify its sha256,
    decode, cap train rows by seed, write `<split>.<lang>.jsonl`."""
    fence(spec)  # a dataset the fence refuses is not even downloaded
    base = os.path.join(raw_dir, spec.key)
    os.makedirs(base, exist_ok=True)
    manifest: dict[str, Any] = {
        "task": TASK, "dataset": spec.key, "hf_id": spec.hf_id,
        "revision": spec.revision, "card_revision": spec.card_revision,
        "license": spec.license, "fetched_utc": _utcnow(),
        "decoder": sys.executable, "seed": seed, "files": {}, "label_names": {},
    }
    decoded: dict[tuple[str, str], list[dict]] = {}
    for f in spec.files:
        path = _download(spec, f)
        digest = sha256_file(path)
        if digest != f.sha256:
            raise ValueError(
                f"{spec.key} {f.path} at {spec.revision}: sha256 {digest} != pinned "
                f"{f.sha256}; refusing to decode"
            )
        rows, names = _decode(path, f.fmt)
        if spec.label_column and spec.label_column in names:
            prev = manifest["label_names"].get(spec.label_column)
            if prev is not None and prev != names[spec.label_column]:
                raise ValueError(f"{spec.key}: label names differ between files")
            manifest["label_names"][spec.label_column] = names[spec.label_column]
        decoded[(f.split, f.lang)] = rows
        manifest["files"][f"{f.split}.{f.lang}"] = {
            "source": f.path, "format": f.fmt, "sha256": digest,
            "bytes": os.path.getsize(path), "rows_available": len(rows),
        }
    if spec.adapter == "banking77":  # CSV: the universe is the categories seen
        manifest["label_names"]["category"] = sorted(
            {r["category"] for rows in decoded.values() for r in rows if r.get("category")}
        )
    for (split, lang), rows in decoded.items():
        cap = spec.train_row_cap if split == spec.train_split else len(rows)
        keep = select_rows(len(rows), cap, seed, f"{spec.key}/{split}/{lang}")
        kept = []
        for i in keep:
            row = dict(rows[i])
            row["_row"] = i
            kept.append(row)
        out = raw_path(spec, split, lang, raw_dir)
        n = _write_jsonl(out, kept)
        manifest["files"][f"{split}.{lang}"].update({
            "raw": _rel(out), "raw_sha256": sha256_file(out), "rows_kept": n,
            "row_cap": cap if cap < len(rows) else None,
            "selection": "all rows" if cap >= len(rows) else
            f"sha256(seed\\0{spec.key}/{split}/{lang}\\0index) ascending, first {cap}, file order",
        })
    _write_json(os.path.join(base, "manifest.json"), manifest)
    return manifest


def raw_manifest(spec: DatasetSpec, raw_dir: str = RAW_DIR) -> dict:
    path = os.path.join(raw_dir, spec.key, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{_rel(path)} missing: run `python -m data.episode_bridge fetch --dataset "
            f"{spec.key}` from a venv with pyarrow first"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def names_for(spec: DatasetSpec, raw: dict) -> list[str]:
    if spec.adapter == "banking77":
        return list(raw["label_names"]["category"])
    if spec.label_column:
        return list(raw["label_names"][spec.label_column])
    return []


# ------------------------------------------------------------ the leakage


class TrigramIndex:
    """`data.leakage`'s two layers (normalised exact hash + word-trigram
    Jaccard ≥ threshold) over an inverted index, so 50 000 rows against
    ~1 000 blocked states is seconds, not the O(n·m) of `LeakageDetector`.
    Same functions, same threshold, same verdicts."""

    def __init__(self, texts: dict[str, list[str]], threshold: float = LEAK_THRESHOLD):
        self.threshold = threshold
        self.hash_src: dict[str, str] = {}
        self.tri: list[tuple[str, set[str]]] = []
        self.index: dict[str, list[int]] = {}
        self.sizes = {src: len(v) for src, v in texts.items()}
        for src, items in texts.items():
            for t in items:
                self.hash_src.setdefault(LK.text_hash(t), src)
                grams = LK.trigrams(t)
                k = len(self.tri)
                self.tri.append((src, grams))
                for g in grams:
                    self.index.setdefault(g, []).append(k)

    def scan(self, text: str) -> tuple[str, str] | None:
        """`(source, kind)` of the first hit, or None when clean."""
        src = self.hash_src.get(LK.text_hash(text))
        if src is not None:
            return src, "exact"
        grams = LK.trigrams(text)
        shared: dict[int, int] = {}
        for g in grams:
            for k in self.index.get(g, ()):
                shared[k] = shared.get(k, 0) + 1
        for k, s in shared.items():
            src, bg = self.tri[k]
            if s / (len(grams) + len(bg) - s) >= self.threshold:
                return src, "paraphrase"
        return None


def blocked_texts() -> dict[str, list[str]]:
    """The cuts no train row may resemble, read as files (the sealed
    battery is read for disjointness exactly like `battery_dev` does; it
    is not opened: no ledger, no scoring)."""
    from data import battery_dev, battery_sealed

    out: dict[str, list[str]] = {}
    for name, path in (("battery-dev", battery_dev.BATTERY),
                       ("battery-sealed", battery_sealed.SEALED)):
        out[name] = sorted({ep["state"] for ep in _read_jsonl(path)})
    td = os.path.join(TD.OUT_DIR, "test", "episodes.jsonl")
    out["typed-decisions-test"] = (
        sorted({ep["state"] for ep in _read_jsonl(td)}) if os.path.exists(td) else []
    )
    return out


def leak_filter(
    episodes: list[dict], index: TrigramIndex, own_eval_hashes: set[str] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Split episodes into (kept, leak rejects, report). A state is scanned
    once; every episode of a leaking state goes."""
    verdict: dict[str, tuple[str, str] | None] = {}
    kept: list[dict] = []
    rejects: list[dict] = []
    report: dict[str, Any] = {"n_states": 0, "hits": {}, "own_eval_overlap": 0}
    for ep in episodes:
        st = ep["state"]
        if st not in verdict:
            hit = index.scan(st)
            if hit is None and own_eval_hashes and LK.text_hash(st) in own_eval_hashes:
                hit = ("own-eval-cut", "exact")
            verdict[st] = hit
            report["n_states"] += 1
            if hit is not None:
                key = f"{hit[0]}:{hit[1]}"
                report["hits"][key] = report["hits"].get(key, 0) + 1
                if hit[0] == "own-eval-cut":
                    report["own_eval_overlap"] += 1
        hit = verdict[st]
        if hit is None:
            kept.append(ep)
        else:
            rejects.append({
                "rejected": True, "id": ep["id"], "dataset": ep["origin"],
                "row_id": ep["source"]["row_id"], "lang": ep["lang"], "split": ep["split"],
                "stage": "leak",
                "reasons": [f"state {'matches' if hit[1] == 'exact' else 'resembles'} "
                            f"{hit[0]} ({hit[1]}, trigram Jaccard >= {index.threshold})"],
            })
    report["hits"] = dict(sorted(report["hits"].items()))
    report["n_episodes_removed"] = len(rejects)
    return kept, rejects, report


# ------------------------------------------------------------ the convert


def _count(episodes: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for ep in episodes:
        v = str(ep.get(key))
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def reasons_histogram(rejects: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rj in rejects:
        for reason in rj.get("reasons", []):
            key = re.sub(r"'[^']*'|\"[^\"]*\"|\d+", "…", reason)[:160]
            out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def convert_dataset(
    spec: DatasetSpec, seed: int = SEED, raw_dir: str = RAW_DIR, out_dir: str = OUT_DIR,
    index: TrigramIndex | None = None,
) -> dict[str, dict]:
    """Every cut of one dataset → episodes/rejects/manifest per split.
    Held-out cuts go first: the train split is filtered against them."""
    fence_rec = fence(spec)
    raw = raw_manifest(spec, raw_dir)
    if raw.get("revision") != spec.revision:
        raise ValueError(f"{spec.key}: raw fetched at {raw.get('revision')}, spec pins {spec.revision}")
    names = names_for(spec, raw)
    index = index or TrigramIndex(blocked_texts())
    manifests: dict[str, dict] = {}
    own_eval: set[str] = set()
    order = list(spec.heldout_splits) + ([spec.train_split] if spec.train_split else [])
    for split in order:
        t0 = time.time()
        episodes: list[dict] = []
        rejects: list[dict] = []
        sources: dict[str, Any] = {}
        for lang in spec.langs:
            key = f"{split}.{lang}"
            if key not in raw["files"]:
                continue
            entry = raw["files"][key]
            path = raw_path(spec, split, lang, raw_dir)
            digest = sha256_file(path)
            if digest != entry["raw_sha256"]:
                raise ValueError(f"{_rel(path)}: sha256 {digest} != {entry['raw_sha256']} at fetch")
            eps, rjs = convert_raw_rows(spec, _read_jsonl(path), split, lang, names, seed)
            episodes.extend(eps)
            rejects.extend(rjs)
            sources[lang] = entry
        is_train = split == spec.train_split
        kept, leak_rj, leak = leak_filter(episodes, index, own_eval if is_train else None)
        if is_train:
            # a train row that leaks is removed; a held-out row is only counted
            episodes, rejects = kept, rejects + leak_rj
        else:
            own_eval |= {LK.text_hash(ep["state"]) for ep in episodes}
        base = os.path.join(out_dir, spec.key, split)
        os.makedirs(base, exist_ok=True)
        ep_path = os.path.join(base, "episodes.jsonl")
        rj_path = os.path.join(base, "rejects.jsonl")
        n_ep = _write_jsonl(ep_path, episodes)
        n_rj = _write_jsonl(rj_path, rejects)
        eval_only = spec.is_eval_only(split)
        man = {
            "task": TASK,
            "converter_version": BRIDGE_VERSION,
            "origin": spec.origin,
            "dataset": spec.hf_id,
            "key": spec.key,
            "config": spec.config,
            "revision": spec.revision,
            "card_revision": spec.card_revision,
            "license": spec.license,
            "license_evidence": spec.license_evidence,
            "usage": spec.usage,
            "usage_reason": spec.usage_reason,
            "fence": fence_rec,
            "split": split,
            "eval_only": eval_only,
            "eval_only_reason": None if not eval_only else (
                spec.usage_reason if spec.usage != "train"
                else f"published held-out cut ({split}) of {spec.hf_id}: never train"
                + (f" — {spec.heldout_note}" if spec.heldout_note else "")),
            "seed": seed,
            "command": "PYTHONPATH=. .venv-train/bin/python -m data.episode_bridge "
            f"convert --dataset {spec.key} --seed {seed}",
            "source": sources,
            "contract": EC.manifest_record(n_ep, seed, BRIDGE_VERSION, {split: n_ep}),
            "rules": {
                "family": spec.family,
                "question_type": spec.question_type,
                "templates": TEMPLATES[spec.key],
                "candidate_text": "label name as published (ClassLabel metadata / "
                "CSV / row choices), '_' shown as space; no descriptions exist in "
                "this dataset, none written",
                "binary_render": {k: list(v) for k, v in YES_NO.items()}
                if spec.question_type == "binary" else None,
                "label_names_es": XNLI_NAMES_ES if spec.key == "xnli" else None,
                "evidence": f"heuristic:whole-state if <= {WHOLE_STATE_CHARS} chars, "
                "else heuristic:max-overlap-sentence",
                "notes": list(spec.notes),
                "label_names": names or None,
            },
            "leakage": {
                "method": "data.leakage normalize/text_hash + word-trigram Jaccard "
                f">= {index.threshold} (inverted index)",
                "blocked": index.sizes,
                "own_eval_cut_checked": is_train and bool(spec.heldout_splits),
                "action": "removed from train" if is_train else "counted only",
                **leak,
            },
            "n_rows_raw": sum(s.get("rows_kept", 0) for s in sources.values()),
            "n_episodes": n_ep,
            "n_rejects": n_rj,
            "per_lang": _count(episodes, "lang"),
            "rejects_by_stage": _count(rejects, "stage"),
            "rejects_by_reason": reasons_histogram(rejects),
            "files": {
                "episodes.jsonl": {"rows": n_ep, "sha256": sha256_file(ep_path)},
                "rejects.jsonl": {"rows": n_rj, "sha256": sha256_file(rj_path)},
            },
            "elapsed_s": round(time.time() - t0, 2),
            "built_utc": _utcnow(),
            "status": "published",
        }
        _write_json(os.path.join(base, "manifest.json"), man)
        manifests[split] = man
    return manifests


def load_split(key: str, split: str, out_dir: str = OUT_DIR) -> tuple[list[dict], dict]:
    """Episodes of a published cut, verified against their manifest sha."""
    base = os.path.join(out_dir, key, split)
    with open(os.path.join(base, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    path = os.path.join(base, "episodes.jsonl")
    digest = sha256_file(path)
    if digest != man["files"]["episodes.jsonl"]["sha256"]:
        raise ValueError(f"{_rel(path)}: sha256 {digest} moved under its manifest seal")
    return _read_jsonl(path), man


# ------------------------------------------------ the eval-only barrier


def eval_only_cuts() -> dict[str, str]:
    """`{"<key>/<split>": reason}` for every eval-only cut of every spec."""
    out = {}
    for spec in SPECS.values():
        for split in spec.splits:
            if spec.is_eval_only(split):
                out[f"{spec.key}/{split}"] = (
                    spec.usage_reason if spec.usage != "train"
                    else f"published held-out cut of {spec.hf_id}")
    return dict(sorted(out.items()))


def is_eval_only_ref(item: Any) -> bool:
    """A path under `episodes-external/<key>/<eval-only split>`, a manifest
    or mix entry with `eval_only: true` (or origin+split naming such a
    cut), or an episode of one. The typed-decisions test cut is refused by
    its own converter's rule too."""
    if TD.is_eval_only_ref(item):
        return True
    cuts = eval_only_cuts()
    if isinstance(item, str):
        norm = item.replace("\\", "/").rstrip("/")
        for cut in cuts:
            if f"episodes-external/{cut}" in norm or norm.endswith(cut):
                return True
        return False
    if isinstance(item, dict):
        if item.get("eval_only") is True:
            return True
        origin = item.get("origin")
        key = item.get("key") or (origin[len("public-"):] if isinstance(origin, str)
                                  and origin.startswith("public-") else None)
        split = item.get("split")
        if key and split and f"{key}/{split}" in cuts:
            return True
        for k in ("path", "dir", "episodes", "manifest", "source"):
            v = item.get(k)
            if isinstance(v, str) and is_eval_only_ref(v):
                return True
    return False


def assert_trainable(sources: Any) -> list[str]:
    """Refuse a mixture that names ANY eval-only cut (public or
    typed-decisions test) by path, manifest or episode. `excluded` keys
    are skipped: a manifest records there why a cut is absent."""
    hits: list[str] = []

    def walk(node: Any, where: str) -> None:
        if isinstance(node, str):
            if is_eval_only_ref(node):
                hits.append(where)
        elif isinstance(node, dict):
            if is_eval_only_ref(node):
                hits.append(where)
                return
            for k, v in node.items():
                if k == "excluded":
                    continue
                if isinstance(v, (dict, list, tuple)) or (
                    isinstance(v, str) and k in TD._PATH_KEYS
                ):
                    walk(v, f"{where}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, sub in enumerate(node):
                walk(sub, f"{where}[{i}]")

    walk(sources, "sources")
    if hits:
        raise EvalOnlyLeak(
            "eval-only cut(s) named as training data at: " + ", ".join(hits)
            + " (public held-out cuts, BANKING77, licence-fenced datasets and "
            "typed-decisions test never train)"
        )
    return hits


# ----------------------------------------------------------- the mix cap


def cap_counts(counts: dict[str, int], cap: float = MIX_CAP) -> dict[str, int]:
    """Largest integer counts ≤ the available ones such that no dataset is
    more than `cap` of the total. Needs ≥ ceil(1/cap) non-empty sources."""
    live = {k: v for k, v in counts.items() if v > 0}
    if live and len(live) * cap < 1:
        raise ValueError(f"{len(live)} sources cannot satisfy a {cap:.0%} cap")
    out = dict(counts)
    for _ in range(10_000):
        total = sum(out.values())
        over = [k for k, v in out.items() if v > cap * total + 1e-9]
        if not over:
            return out
        for k in over:
            others = total - out[k]
            # v ≤ cap·(others + v)  ⇔  v ≤ cap·others / (1 − cap)
            out[k] = int(cap * others / (1 - cap))
    raise RuntimeError("cap did not converge")


def build_mix_manifest(out_dir: str = OUT_DIR, cap: float = MIX_CAP) -> dict:
    """The declared external mix: every trainable train cut, its sha and
    its share under the cap. `#T-loop-trainer` applies it."""
    members = []
    for spec in SPECS.values():
        if spec.usage != "train" or not spec.train_split:
            continue
        eps, man = load_split(spec.key, spec.train_split, out_dir)
        members.append({
            "key": spec.key,
            "origin": spec.origin,
            "split": spec.train_split,
            "path": _rel(os.path.join(out_dir, spec.key, spec.train_split, "episodes.jsonl")),
            "sha256": man["files"]["episodes.jsonl"]["sha256"],
            "license": spec.license,
            "n_available": len(eps),
            "per_lang": _count(eps, "lang"),
        })
    capped = cap_counts({m["key"]: m["n_available"] for m in members}, cap)
    total = sum(capped.values())
    for m in members:
        m["n_under_cap"] = capped[m["key"]]
        m["share"] = round(capped[m["key"]] / total, 4) if total else 0.0
        m["selection"] = ("all episodes" if capped[m["key"]] == m["n_available"] else
                          "first n_under_cap by sha256(seed\\0episode id), applied by the trainer")
    manifest = {
        "task": TASK,
        "kind": "external-public-mix",
        "bridge_version": BRIDGE_VERSION,
        "cap_per_dataset": cap,
        "cap_rule": "no dataset contributes more than cap × total; counts are the "
        "largest that satisfy it (data.episode_bridge.cap_counts)",
        "members": members,
        "n_available": sum(m["n_available"] for m in members),
        "n_under_cap": total,
        "excluded": eval_only_cuts(),
        "built_utc": _utcnow(),
        "command": "PYTHONPATH=. .venv-train/bin/python -m data.episode_bridge gate",
    }
    assert_trainable(manifest)  # refuses before anything is written
    manifest["members_sha256"] = sha256_text(json.dumps(
        [(m["key"], m["sha256"], m["n_under_cap"]) for m in members]))
    return manifest


# ---------------------------------------------------------------- the gate


def _check(name: str, ok: bool, detail: Any) -> dict:
    return {"name": name, "pass": bool(ok), "detail": detail}


def measure_gate(out_dir: str = OUT_DIR) -> dict:
    """Counts measured from the published files, never from memory."""
    per_ds: dict[str, Any] = {}
    langs_seen: dict[str, set] = {}
    all_valid = True
    eval_marks_ok = True
    train_leaks = 0
    for spec in SPECS.values():
        entry: dict[str, Any] = {
            "hf_id": spec.hf_id, "revision": spec.revision, "license": spec.license,
            "license_evidence": spec.license_evidence, "usage": spec.usage,
            "usage_reason": spec.usage_reason, "family": spec.family,
            "question_type": spec.question_type, "splits": {},
        }
        for split in spec.splits:
            eps, man = load_split(spec.key, split, out_dir)
            rep = EC.batch_validate(eps)
            all_valid &= rep["n_valid"] == len(eps) and not rep["duplicate_ids"]
            ev = spec.is_eval_only(split)
            eval_marks_ok &= man["eval_only"] == ev and all(e["eval_only"] is ev for e in eps)
            leak = man["leakage"]
            if split == spec.train_split:
                # after removal, a re-scan must find nothing
                idx = TrigramIndex(blocked_texts())
                train_leaks += sum(1 for st in {e["state"] for e in eps} if idx.scan(st))
            for e in eps:
                langs_seen.setdefault(spec.key, set()).add(e["lang"])
            entry["splits"][split] = {
                "manifest": _rel(os.path.join(out_dir, spec.key, split, "manifest.json")),
                "episodes_sha256": man["files"]["episodes.jsonl"]["sha256"],
                "eval_only": man["eval_only"],
                "n_rows_raw": man["n_rows_raw"],
                "n_episodes": len(eps),
                "n_valid": rep["n_valid"],
                "n_rejects": man["n_rejects"],
                "per_lang": _count(eps, "lang"),
                "n_groups": len({e["variant_group"] for e in eps}),
                "rejects_by_stage": man["rejects_by_stage"],
                "leakage": {k: leak[k] for k in ("hits", "n_states", "own_eval_overlap",
                                                 "n_episodes_removed", "action")},
                "row_cap": {lang: s.get("row_cap") for lang, s in man["source"].items()},
            }
        per_ds[spec.key] = entry
    mix = build_mix_manifest(out_dir)
    mix_path = _write_json(os.path.join(MIX_DIR, "manifest.json"), mix)
    es_datasets = sorted(k for k, v in langs_seen.items() if "es" in v)
    refused = []
    for cut in eval_only_cuts():
        try:
            assert_trainable([os.path.join("artifacts", "episodes-external", cut)])
        except EvalOnlyLeak:
            refused.append(cut)
    max_share = max((m["share"] for m in mix["members"]), default=0.0)
    checks = [
        _check("at_least_8_datasets_converted", len(per_ds) >= 8, {"n": len(per_ds)}),
        _check("at_least_2_with_spanish_rows", len(es_datasets) >= 2, es_datasets),
        _check("all_published_episodes_validate", all_valid, {}),
        _check("eval_only_marks_match_spec", eval_marks_ok, {}),
        _check("banking77_entirely_eval_only",
               all(per_ds["banking77"]["splits"][s]["eval_only"] for s in SPECS["banking77"].splits),
               list(SPECS["banking77"].splits)),
        _check("no_blocked_state_left_in_any_train_split", train_leaks == 0,
               {"states_hitting_after_filter": train_leaks}),
        _check("every_eval_only_cut_refused_by_assert_trainable",
               sorted(refused) == sorted(eval_only_cuts()),
               {"refused": len(refused), "eval_only_cuts": len(eval_only_cuts())}),
        _check("mix_cap_respected", max_share <= MIX_CAP + 1e-9,
               {"max_share": max_share, "cap": MIX_CAP}),
        _check("revisions_pinned", all(len(s.revision) == 40 for s in SPECS.values()), {}),
    ]
    return {
        "task": TASK,
        "status": "measured",
        "measured_utc": _utcnow(),
        "bridge_version": BRIDGE_VERSION,
        "schema_version": EC.SCHEMA_VERSION,
        "schema_sha": EC.schema_sha(),
        "accuracy": None,
        "note": "data only: nothing is trained or evaluated in this task",
        "command": "PYTHONPATH=. .venv-train/bin/python -m data.episode_bridge gate",
        "datasets": per_ds,
        "datasets_with_spanish": es_datasets,
        "consumable": {
            "definition": "validated, leak-filtered train episodes of datasets whose "
            "licence passes the fence (usage train); eval-only cuts excluded",
            "total_episodes": mix["n_available"],
            "total_under_25pct_cap": mix["n_under_cap"],
            "per_dataset": {m["key"]: {"available": m["n_available"],
                                       "under_cap": m["n_under_cap"],
                                       "share": m["share"], "per_lang": m["per_lang"]}
                            for m in mix["members"]},
            "mix_manifest": _rel(mix_path),
            "mix_members_sha256": mix["members_sha256"],
        },
        "eval_only_cuts": eval_only_cuts(),
        "checks": checks,
        "verdict": "PASS" if all(c["pass"] for c in checks) else "FAIL",
    }


# ---------------------------------------------------------------- the CLI


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="data.episode_bridge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "convert"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", choices=sorted(SPECS), action="append")
        p.add_argument("--seed", type=int, default=SEED)
    sub.add_parser("gate")
    args = ap.parse_args(argv)
    if args.cmd in ("fetch", "convert"):
        keys = args.dataset or list(SPECS)
        index = TrigramIndex(blocked_texts()) if args.cmd == "convert" else None
        for key in keys:
            spec = SPECS[key]
            if args.cmd == "fetch":
                man = fetch(spec, seed=args.seed)
                for k, f in man["files"].items():
                    print(f"[fetch] {key} {k}: {f['rows_kept']}/{f['rows_available']} rows "
                          f"(sha256 {f['sha256'][:12]}…)")
            else:
                for split, man in convert_dataset(spec, seed=args.seed, index=index).items():
                    print(f"[convert] {key}/{split}: {man['n_episodes']} episodes, "
                          f"{man['n_rejects']} rejects, eval_only={man['eval_only']} "
                          f"leak={man['leakage']['hits']} ({man['elapsed_s']} s)")
        return 0
    gate = measure_gate()
    path = _write_json(GATE_PATH, gate)
    for c in gate["checks"]:
        print(f"[gate] {'ok  ' if c['pass'] else 'FAIL'} {c['name']} {c['detail']}")
    print(f"[gate] consumable {gate['consumable']['total_episodes']} episodes "
          f"({gate['consumable']['total_under_25pct_cap']} under the 25 % cap)")
    print(f"[gate] {gate['verdict']} → {_rel(path)}")
    return 0 if gate["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
