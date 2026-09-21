"""Five independent corpora for `decision-mix-clean-1m` (#T-mix-1m, stdlib only).

Why this file exists
--------------------
The §§65-66 caps say no dataset may exceed 15 % of a mixture and no family
30 %. With seven clean sources that ceiling is arithmetic, not opinion: six
of them sat EXACTLY at the 15 % cap and the seventh (`email-triage`, 4 000
rows) ate the 5 % of slack, so the whole fenced registry admitted 39 970
rows — 4 % of the 1 M §86 asks for. No seed and no target size moves that
number. Only independent supply does.

So this module converts five corpora that are (a) downloadable without
credentials, (b) permissively licensed, (c) disjoint from the §§18/77
benchmark fence (Banking77 / HelpSteer2 / PubMedQA) and from the frozen
Jevals id registry, and (d) big enough to sit at the cap instead of under
it:

* `dbpedia14`   — 560 k DBpedia ontology abstracts, 14 curated classes.
                  A SECOND source in the `topic` family, which is what
                  lifts that family from one cap unit to two.
* `snli`        — 550 k premise/hypothesis pairs. Fills the §86 `nli`
                  layer, which had no supply at all, and emits a boolean
                  question next to the 3-way one so the §48 Noul share
                  does not collapse as the corpus grows.
* `goemotions`  — 58 k Reddit comments, 28 emotions, 211 k rater
                  judgements. A 28-label pool is where a wide option set
                  is a real question and not a two-way guess.
* `detox-attack`— 115 k Wikipedia talk-page comments, ~10 annotators
                  each. This is the §48 **Score** supply: HelpSteer2 was
                  the only ordinal corpus in the registry and it is
                  fenced, so Score sat at 0 % against a 15 % target. The
                  gold here is not a model's opinion — it is the measured
                  fraction of human annotators who called the comment an
                  attack, bucketed into a five-point scale, and it is
                  re-derivable from the raw TSV by counting rows.
* `swag`        — 73 k grounded video captions with four continuations.
                  The §86 `adversarial` layer, which had no supply at all:
                  SWAG's distractors are not noise, they are candidates
                  that survived ADVERSARIAL FILTERING against an ensemble
                  of models, which is the published construction of the
                  corpus and not a label this converter invented.

Hard is MEASURED, never asserted
--------------------------------
`data.mix.is_hard` counts a question as hard when K >= 9 or when the
generator labelled it hard. Emitting wide option sets full of random
labels would satisfy that letter and teach nothing. So every hard row
here takes its distractors from the TOP of the pool's difficulty ranking
(`data.optset.difficulty`: char-3gram cosine blended with word-Jaccard,
the same function the sampler's hard bucket uses) and carries the
measured similarity it was built at — `quality.nn_mean` / `quality.nn_min`
— so "this slice is hard" is a number in the row, checkable after the
fact, and not a flag the converter set about itself.

Licences (see `.meshkore/docs/source-register.md` for the full cards)
--------------------------------------------------------------------
dbpedia14 CC-BY-SA-3.0 + GFDL · snli CC-BY-SA-4.0 · goemotions Apache-2.0
· detox-attack CC0-1.0 · swag MIT. The two share-alike sets carry the same
obligation BoolQ already carries and are recorded next to it.

Usage::

    python3 -m data.convert_widen fetch          # what to download, where
    python3 -m data.convert_widen dbpedia14
    python3 -m data.convert_widen snli
    python3 -m data.convert_widen goemotions
    python3 -m data.convert_widen detox-attack
    python3 -m data.convert_widen swag
    python3 -m data.convert_widen all
"""
from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.adapters import Example, Option, Question, _checked  # noqa: E402
from data.optset import difficulty  # noqa: E402

RAW = ROOT / "artifacts" / "data-raw"
OUT_DIR = ROOT / "artifacts" / "data-prefetch"

#: bump when a question shape here changes: the shards are pinned by
#: sha256 in every mixture manifest, so a silent reshape is a silent
#: invalidation of every manifest that names them.
WIDEN_VERSION = 1

#: where the bytes come from. No credentials, no HF token, no login.
DOWNLOADS = {
    "dbpedia14": [("https://s3.amazonaws.com/fast-ai-nlp/dbpedia_csv.tgz",
                   "dbpedia/dbpedia_csv.tgz")],
    "snli": [("https://nlp.stanford.edu/projects/snli/snli_1.0.zip",
              "snli/snli_1.0.zip")],
    "goemotions": [
        (f"https://storage.googleapis.com/gresearch/goemotions/data/"
         f"full_dataset/goemotions_{i}.csv", f"goemotions/goemotions_{i}.csv")
        for i in (1, 2, 3)],
    "detox-attack": [
        ("https://ndownloader.figshare.com/files/7554634",
         "wiki-detox/7554634.tsv"),
        ("https://ndownloader.figshare.com/files/7554637",
         "wiki-detox/7554637.tsv")],
    "swag": [
        ("https://raw.githubusercontent.com/rowanz/swagaf/master/data/"
         "train.csv", "swag/train.csv"),
        ("https://raw.githubusercontent.com/rowanz/swagaf/master/data/"
         "val.csv", "swag/val.csv")],
}

_WS = re.compile(r"\s+")
# a Wikipedia talk-page comment and a DBpedia abstract both blow past the
# default 128 KiB csv field limit; a truncated field would be a silent
# corpus corruption, so the limit is raised rather than caught.
csv.field_size_limit(1 << 24)
#: the five-point ordinal scale, spelled exactly the way `data.mix.
#: question_type` recognises a Score question (`^score \d+$`)
SCORE_OPTIONS = [Option(id=str(i), text=f"score {i}") for i in range(5)]


# -- deterministic helpers -------------------------------------------------

def _u(*parts) -> float:
    """A stateless uniform in [0, 1) keyed on the row, not on a stream.

    Same reason `data.mix.keep_row` is a sha256 and not `random`: a
    conversion that depends on iteration order cannot be reproduced from a
    shard the way the manifest promises it can.
    """
    key = "\x00".join(str(p) for p in ("widen", WIDEN_VERSION, *parts))
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def _shuffled(items: list, *parts) -> list:
    """A deterministic permutation, keyed the same way."""
    return [x for _, x in sorted((_u(*parts, i), x)
                                 for i, x in enumerate(items))]


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


class _Ranker:
    """Pool labels ranked by measured plausibility as a distractor.

    One pass over the pool per distinct gold label, cached: the pools here
    are 3-28 short strings, so the whole ranking of a corpus costs at most
    28 passes no matter how many rows it has.
    """

    def __init__(self, pool: dict) -> None:
        self.pool = dict(pool)          # id -> text
        self._cache: dict = {}

    def ranked(self, gold: str) -> list:
        """`[(id, measured_difficulty)]`, hardest distractor first."""
        hit = self._cache.get(gold)
        if hit is None:
            gold_text = self.pool[gold]
            hit = sorted(((i, round(difficulty(gold_text, t), 6))
                          for i, t in self.pool.items() if i != gold),
                         key=lambda p: (-p[1], p[0]))
            self._cache[gold] = hit
        return hit

    def draw(self, gold: str, k: int, hard: bool, *key) -> tuple:
        """`(option_ids, quality)` for one row. K includes the gold.

        `hard=True` takes the top `k - 1` of the ranking — the measured
        nearest neighbourhood of the gold label. `hard=False` takes a
        deterministic sample of the WHOLE ranking, so the easy tier is not
        secretly the same question with fewer options.
        """
        ranking = self.ranked(gold)
        if hard:
            chosen = ranking[:k - 1]
        else:
            chosen = _shuffled(ranking, *key)[:k - 1]
        scores = [s for _, s in chosen]
        options = _shuffled([gold] + [i for i, _ in chosen], "opt", *key)
        quality = {
            "difficulty": "hard" if hard else "easy",
            "source": "measured",
            "distractors": ("nearest-%d of %d by data.optset.difficulty"
                            % (k - 1, len(ranking)) if hard else
                            "uniform over the %d-label pool" % len(ranking)),
            "nn_mean": round(sum(scores) / len(scores), 6) if scores else 0.0,
            "nn_min": round(min(scores), 6) if scores else 0.0,
            "nn_max": round(max(scores), 6) if scores else 0.0,
            "pool_size": len(ranking) + 1,
        }
        return options, quality


def _row(example: Example, quality_by_qid: dict | None = None) -> dict:
    """Validate against the V1 schema, then attach the measured `quality`.

    `data.schema.Question` has no `quality` field and should not grow one
    for this: the field is a GENERATOR's note about a row (#T-prog-gold
    writes the same key), not part of the contract a head consumes. So the
    row is checked as a schema dataclass first and annotated after, which
    keeps the validation real.
    """
    out = dataclasses.asdict(example)
    for q in out["questions"]:
        note = (quality_by_qid or {}).get(q["id"])
        if note:
            q["quality"] = note
    return out


class _Writer:
    """Shard writer that dedups whole QUESTIONS and counts what it wrote.

    The key is the normalised state plus every question's option texts and
    gold, not the state alone: `swag` cuts several rows from one video
    caption, so two rows can share a state and still be different
    questions with different answers. Deduping on the state would have
    thrown 21 103 of them away as if they were copies.
    """

    def __init__(self, dataset: str, out_dir: Path = OUT_DIR) -> None:
        self.dataset = dataset
        self.path = Path(out_dir) / f"{dataset}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w")
        self._seen: set = set()
        self.written = 0
        self.questions = 0
        self.duplicates = 0
        self.by_split: dict = collections.Counter()
        self.by_qtype: dict = collections.Counter()

    def add(self, row: dict) -> bool:
        signature = [_norm(row["state"])]
        for q in row["questions"]:
            signature.append(str(q.get("answer")))
            signature.extend(sorted(_norm(o["text"]) for o in q["options"]))
        key = hashlib.sha256("\x00".join(signature).encode()).hexdigest()
        if key in self._seen:
            self.duplicates += 1
            return False
        self._seen.add(key)
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.written += 1
        self.questions += len(row["questions"])
        self.by_split[row["split"]] += 1
        for q in row["questions"]:
            texts = [o["text"] for o in q["options"]]
            self.by_qtype[_qtype(q["kind"], texts)] += 1
        return True

    def close(self) -> dict:
        self._fh.close()
        try:
            where = str(self.path.relative_to(ROOT))
        except ValueError:      # a fixture directory outside the repo
            where = str(self.path)
        return {"dataset": self.dataset, "out": where,
                "rows": self.written, "questions": self.questions,
                "duplicates_dropped": self.duplicates,
                "by_split": dict(sorted(self.by_split.items())),
                "by_qtype": dict(sorted(self.by_qtype.items())),
                "sha256": _sha256(self.path),
                "bytes": self.path.stat().st_size}


def _qtype(kind: str, texts: list) -> str:
    from data.mix import question_type
    return question_type(kind, texts)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


# -- dbpedia14 -------------------------------------------------------------

#: 45 % of the rows are a four-way question, the rest are the wide regime
#: (K 9-13) the §68 hard slice is made of. 13 and not 14: a row whose
#: option set IS the dataset's global label space is the HuffPost bug
#: `data.optset.global_space_violations` exists to catch.
DBPEDIA_HARD_SHARE = 0.55
DBPEDIA_K_EASY = 4
DBPEDIA_K_HARD = (9, 13)


def convert_dbpedia(raw: Path | None = None, out_dir: Path = OUT_DIR,
                    limit: int | None = None) -> dict:
    base = Path(raw or (RAW / "dbpedia" / "dbpedia_csv"))
    classes = [c.strip() for c in
               (base / "classes.txt").read_text().splitlines() if c.strip()]
    ranker = _Ranker({c: c for c in classes})
    writer = _Writer("dbpedia14", out_dir)
    for split, fn in (("train", "train.csv"), ("test", "test.csv")):
        path = base / fn
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for i, rec in enumerate(csv.reader(fh)):
                if limit is not None and writer.written >= limit:
                    break
                if len(rec) != 3:
                    continue
                gold = classes[int(rec[0]) - 1]
                state = f"{rec[1].strip()}\n{rec[2].strip()}"
                if not state.strip():
                    continue
                key = ("dbpedia14", split, i)
                hard = _u(*key, "tier") < DBPEDIA_HARD_SHARE
                k = (DBPEDIA_K_HARD[0] + int(_u(*key, "k") * (
                    DBPEDIA_K_HARD[1] - DBPEDIA_K_HARD[0] + 1))
                    if hard else DBPEDIA_K_EASY)
                ids, quality = ranker.draw(gold, k, hard, *key)
                qid = f"dbpedia14-ontology-{split}-{i}"
                ex = _checked(Example(
                    state=state, split=split,
                    questions=[Question(
                        id=qid, kind="choice", answer=gold,
                        options=[Option(id=o, text=o) for o in ids])]),
                    f"dbpedia14 row {i}")
                writer.add(_row(ex, {qid: quality}))
    return writer.close()


# -- snli ------------------------------------------------------------------

SNLI_LABELS = {"entailment": "entailment", "neutral": "neutral",
               "contradiction": "contradiction"}
SNLI_SPLITS = {"snli_1.0_train.jsonl": "train",
               "snli_1.0_dev.jsonl": "calibration",
               "snli_1.0_test.jsonl": "test"}


def convert_snli(raw: Path | None = None, out_dir: Path = OUT_DIR,
                 limit: int | None = None) -> dict:
    """Premise/hypothesis -> a 3-way relation AND a boolean entailment.

    Two questions per pair on purpose. The 3-way one is the NLI task; the
    boolean one is what keeps the §48 Noul share alive. Without it the only
    binary supply in the registry is `civil-comments` and `boolq`, and a
    corpus that grows past a few hundred thousand rows drifts below the
    30 % Noul target no matter how it is allocated.
    """
    path = Path(raw or (RAW / "snli" / "snli_1.0.zip"))
    writer = _Writer("snli", out_dir)
    labels = sorted(SNLI_LABELS)
    with zipfile.ZipFile(path) as zf:
        for member, split in SNLI_SPLITS.items():
            name = next((n for n in zf.namelist()
                         if n.endswith(member) and "__MACOSX" not in n), None)
            if name is None:
                continue
            with zf.open(name) as fh:
                for i, line in enumerate(fh):
                    if limit is not None and writer.written >= limit:
                        break
                    rec = json.loads(line)
                    gold = rec.get("gold_label")
                    # "-" means the five annotators reached no majority:
                    # there is no gold to point at, so the pair is dropped
                    # rather than given one.
                    if gold not in SNLI_LABELS:
                        continue
                    s1 = (rec.get("sentence1") or "").strip()
                    s2 = (rec.get("sentence2") or "").strip()
                    if not s1 or not s2:
                        continue
                    state = f"Premise: {s1}\nHypothesis: {s2}"
                    rel = f"snli-relation-{split}-{i}"
                    ent = f"snli-entails-{split}-{i}"
                    ex = _checked(Example(
                        state=state, split=split, questions=[
                            Question(id=rel, kind="choice", answer=gold,
                                     options=[Option(id=lbl, text=lbl)
                                              for lbl in labels]),
                            Question(id=ent, kind="boolean",
                                     answer="yes" if gold == "entailment"
                                     else "no",
                                     options=[Option(id="yes", text="yes"),
                                              Option(id="no", text="no")]),
                        ]), f"snli row {i}")
                    writer.add(_row(ex))
    return writer.close()


# -- goemotions ------------------------------------------------------------

#: a rater judgement is kept only when the winning emotion has at least
#: this many votes AND beats the runner-up outright: a 1-1 split is not a
#: gold label, it is an unlabelled row.
GOEMOTIONS_MIN_VOTES = 2
GOEMOTIONS_K_EASY = 4
GOEMOTIONS_K_HARD = 12


def _goemotions_votes(files: list) -> tuple:
    """`(texts, votes, raters, unclear)` folded over the annotation rows."""
    texts: dict = {}
    votes: dict = collections.defaultdict(collections.Counter)
    raters: dict = collections.Counter()
    unclear: dict = collections.Counter()
    emotions: list = []
    for path in files:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not emotions:
                emotions = list(reader.fieldnames[9:])
            for rec in reader:
                eid = rec["id"]
                texts.setdefault(eid, rec["text"])
                raters[eid] += 1
                if rec["example_very_unclear"] == "True":
                    unclear[eid] += 1
                for emotion in emotions:
                    if rec[emotion] == "1":
                        votes[eid][emotion] += 1
    return texts, votes, raters, unclear, emotions


def convert_goemotions(raw: Path | None = None, out_dir: Path = OUT_DIR,
                       limit: int | None = None) -> dict:
    base = Path(raw or (RAW / "goemotions"))
    files = sorted(base.glob("goemotions_*.csv"))
    texts, votes, raters, unclear, emotions = _goemotions_votes(files)
    ranker = _Ranker({e: e.replace("_", " ") for e in emotions})
    writer = _Writer("goemotions", out_dir)
    for eid in sorted(texts):
        if limit is not None and writer.written >= limit:
            break
        if unclear[eid] * 2 >= raters[eid]:
            continue
        tally = votes[eid].most_common(2)
        if not tally or tally[0][1] < GOEMOTIONS_MIN_VOTES:
            continue
        if len(tally) > 1 and tally[1][1] == tally[0][1]:
            continue
        gold = tally[0][0]
        # no official split ships with the full annotation dump, so the
        # split is a deterministic 90/10 on the comment id — the same rule
        # `data/convert_huffpost.py` uses.
        split = "train" if _u("goemotions", eid, "split") < 0.9 else "test"
        easy_id = f"goemotions-emotion-{eid}"
        hard_id = f"goemotions-emotion-hard-{eid}"
        easy_ids, easy_q = ranker.draw(gold, GOEMOTIONS_K_EASY, False,
                                       "goemotions", eid)
        hard_ids, hard_q = ranker.draw(gold, GOEMOTIONS_K_HARD, True,
                                       "goemotions", eid)
        ex = _checked(Example(
            state=texts[eid], split=split, questions=[
                Question(id=easy_id, kind="choice", answer=gold,
                         options=[Option(id=o, text=o.replace("_", " "))
                                  for o in easy_ids]),
                Question(id=hard_id, kind="choice", answer=gold,
                         options=[Option(id=o, text=o.replace("_", " "))
                                  for o in hard_ids]),
            ]), f"goemotions {eid}")
        for note in (easy_q, hard_q):
            note["votes"] = tally[0][1]
            note["raters"] = raters[eid]
        writer.add(_row(ex, {easy_id: easy_q, hard_id: hard_q}))
    return writer.close()


# -- detox-attack ----------------------------------------------------------

DETOX_COMMENTS = "7554634.tsv"
DETOX_ANNOTATIONS = "7554637.tsv"
DETOX_SPLITS = {"train": "train", "dev": "calibration", "test": "test"}
#: the two aspects every comment is asked about, plus one that is mostly
#: zero and is therefore BALANCED rather than taken whole (below).
DETOX_ASPECTS = ("attack", "recipient_attack")
DETOX_BALANCED = "third_party_attack"


def detox_bucket(fraction: float) -> int:
    """Annotator agreement -> the five-point ordinal.

    0 annotators, then quartiles of the ones who said yes. The gold is
    re-derivable by counting rows of the annotation TSV, which is what
    makes this ordinal supply verifiable instead of asserted.
    """
    if fraction <= 0.0:
        return 0
    if fraction <= 0.25:
        return 1
    if fraction <= 0.5:
        return 2
    if fraction <= 0.75:
        return 3
    return 4


def convert_detox(raw: Path | None = None, out_dir: Path = OUT_DIR,
                  limit: int | None = None) -> dict:
    base = Path(raw or (RAW / "wiki-detox"))
    comments: dict = {}
    with open(base / DETOX_COMMENTS, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh, delimiter="\t"):
            comments[rec["rev_id"]] = (
                rec["comment"].replace("NEWLINE_TOKEN", "\n")
                              .replace("TAB_TOKEN", "\t").strip(),
                DETOX_SPLITS.get(rec["split"], "train"))
    positives: dict = collections.defaultdict(collections.Counter)
    workers: dict = collections.Counter()
    with open(base / DETOX_ANNOTATIONS, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh, delimiter="\t"):
            rev = rec["rev_id"]
            workers[rev] += 1
            for aspect in (*DETOX_ASPECTS, DETOX_BALANCED):
                if float(rec[aspect]) >= 0.5:
                    positives[rev][aspect] += 1

    # the balanced aspect: every non-zero comment, plus as many zeros,
    # drawn deterministically. An ordinal whose gold is 80 % "score 0"
    # teaches "answer 0", which is not the calibration axis §48 wants.
    nonzero = [r for r in comments
               if workers[r] and positives[r][DETOX_BALANCED] > 0]
    zeros = sorted(r for r in comments
                   if workers[r] and positives[r][DETOX_BALANCED] == 0)
    keep_zero = set(sorted(zeros, key=lambda r: _u("detox-balance", r)
                           )[:len(nonzero)])
    balanced = set(nonzero) | keep_zero

    writer = _Writer("detox-attack", out_dir)
    for rev in sorted(comments, key=lambda r: int(r)):
        if limit is not None and writer.written >= limit:
            break
        text, split = comments[rev]
        n = workers[rev]
        if not text or not n:
            continue
        aspects = list(DETOX_ASPECTS)
        if rev in balanced:
            aspects.append(DETOX_BALANCED)
        questions = []
        notes = {}
        for aspect in aspects:
            bucket = detox_bucket(positives[rev][aspect] / n)
            qid = f"detox-{aspect}-{rev}"
            questions.append(Question(id=qid, kind="choice",
                                      answer=str(bucket),
                                      options=list(SCORE_OPTIONS)))
            notes[qid] = {"source": "measured", "aspect": aspect,
                          "annotators": n,
                          "positive": positives[rev][aspect],
                          "scale": "0-4 quintile of annotator agreement"}
        ex = _checked(Example(state=text, split=split, questions=questions),
                      f"detox {rev}")
        writer.add(_row(ex, notes))
    return writer.close()


# -- swag ------------------------------------------------------------------

SWAG_SPLITS = {"train.csv": "train", "val.csv": "calibration"}


def convert_swag(raw: Path | None = None, out_dir: Path = OUT_DIR,
                 limit: int | None = None) -> dict:
    """Context -> which of four continuations really happened next.

    The option ids are positional (`o0..o3`, the #T-prog-gold convention)
    because the candidates are per-row TEXT and not a label pool: there is
    no global label space here to hold out or to accidentally emit whole.

    Nothing marks these rows hard beyond the corpus itself: SWAG's three
    distractors per row are LM continuations kept only if an ensemble of
    models could not separate them from the real one (adversarial
    filtering, Zellers et al. 2018). That property belongs to the source,
    so it is recorded as the source's — `quality.source = "adversarial
    filtering (SWAG)"` — and the §86 layer, not as a difficulty this
    converter measured.
    """
    base = Path(raw or (RAW / "swag"))
    writer = _Writer("swag", out_dir)
    for fn, split in SWAG_SPLITS.items():
        path = base / fn
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for i, rec in enumerate(csv.DictReader(fh)):
                if limit is not None and writer.written >= limit:
                    break
                try:
                    label = int(rec["label"])
                except (KeyError, TypeError, ValueError):
                    continue        # test.csv ships unlabelled; skip it
                # the STARTPHRASE is the state: `sent1` alone repeats
                # across the rows cut from one video caption, and deduping
                # on it would throw away most of the corpus.
                context = (rec.get("startphrase")
                           or f"{rec.get('sent1', '')} "
                              f"{rec.get('sent2', '')}").strip()
                endings = [(rec.get(f"ending{j}") or "").strip()
                           for j in range(4)]
                if not context or not all(endings) or not 0 <= label < 4:
                    continue
                qid = f"swag-next-{split}-{i}"
                ex = _checked(Example(
                    state=context, split=split,
                    questions=[Question(
                        id=qid, kind="choice", answer=f"o{label}",
                        options=[Option(id=f"o{j}", text=t)
                                 for j, t in enumerate(endings)])]),
                    f"swag row {i}")
                writer.add(_row(ex, {qid: {
                    "difficulty": "hard",
                    "source": "adversarial filtering (SWAG)",
                    "distractors": ("LM continuations retained only where a "
                                    "model ensemble could not separate them "
                                    "from the real one"),
                    "gold_source": rec.get("gold-source", ""),
                    "pool_size": 4}}))
    return writer.close()


CONVERTERS = {"dbpedia14": convert_dbpedia, "snli": convert_snli,
              "goemotions": convert_goemotions, "detox-attack": convert_detox,
              "swag": convert_swag}


def fetch_plan() -> dict:
    """What to download and where it lands. No credentials anywhere."""
    return {name: [{"url": url, "path": f"artifacts/data-raw/{dest}",
                    "present": (RAW / dest).exists()}
                   for url, dest in urls]
            for name, urls in sorted(DOWNLOADS.items())}


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="convert_widen",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("target", choices=[*sorted(CONVERTERS), "all", "fetch"])
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N rows (smoke runs only)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv[1:])

    if args.target == "fetch":
        print(json.dumps(fetch_plan(), indent=2, sort_keys=True))
        return 0
    names = sorted(CONVERTERS) if args.target == "all" else [args.target]
    report = {}
    for name in names:
        report[name] = CONVERTERS[name](out_dir=Path(args.out_dir),
                                        limit=args.limit)
        print(json.dumps(report[name], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
