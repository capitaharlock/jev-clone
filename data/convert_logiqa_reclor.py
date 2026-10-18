"""Convert LogiQA 2.0 + ReClor raw dumps -> universal schema JSONL (stdlib only).

Closes T-massive-huff §124: teach decision WITHOUT generating or storing
explanations. Both sources are logical-reasoning choice tasks
(passage/context + question + options + gold index). The adapter keeps
state/question/options/gold only; reasoning-type flags (LogiQA `type`
dict) and any explanation-like fields are dropped on the floor, and the
tests assert the serialized output carries no explanation text.

Sources (prefetch raw dumps, already split train/calibration/test):
  in : artifacts/data-prefetch/logiqa20.raw.jsonl  ({"split","text": <inner JSON>})
  in : artifacts/data-prefetch/reclor.raw.jsonl    ({"split","context","question",
                                                    "answers": <py-list repr>,
                                                    "label": int, "id_string": str})
  out: artifacts/data-prefetch/logiqa.jsonl
  out: artifacts/data-prefetch/reclor.jsonl

Usage: python3 data/convert_logiqa_reclor.py
"""

from __future__ import annotations

import ast
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data.adapters as A  # noqa: E402
from data.adapters import Example, Option, Question  # noqa: E402

LOGIQA_RAW = ROOT / "artifacts" / "data-prefetch" / "logiqa20.raw.jsonl"
RECLOR_RAW = ROOT / "artifacts" / "data-prefetch" / "reclor.raw.jsonl"
LOGIQA_OUT = ROOT / "artifacts" / "data-prefetch" / "logiqa.jsonl"
RECLOR_OUT = ROOT / "artifacts" / "data-prefetch" / "reclor.jsonl"

# Belt-and-braces: any of these substrings in serialized output fails the run.
FORBIDDEN = ("explanation", "rationale", "chain-of-thought", "cot")


LOGIQA_LABEL_NORM = {"entailed": "entailment", "not entailed": "not_entailment"}


def _logiqa_mc(i: int, r: dict, inner: dict) -> Example:
    options = list(inner["options"])
    answer = int(inner["answer"])
    if not options or not (0 <= answer < len(options)):
        raise A.AdapterError(f"logiqa row {i}: bad options/answer")
    passage = str(inner.get("text", "")).strip()
    question = str(inner.get("question", "")).strip()
    if not passage or not question or any(not str(o).strip() for o in options):
        raise A.AdapterError(f"logiqa row {i}: empty passage/question/option")
    opts = [Option(id=f"opt{j}", text=str(o)) for j, o in enumerate(options)]
    return A._checked(Example(
        state=f"{passage}\n{question}",
        split=r.get("split", "train"),
        questions=[Question(id=f"logiqa-{inner.get('id', inner.get('example_id', i))}",
                            kind="choice", options=opts, answer=f"opt{answer}")],
    ), f"logiqa row {i}")


def _logiqa_nli(i: int, r: dict, inner: dict) -> Example:
    """NLI shapes -> boolean yes/no via the shared bigsrc convention."""
    from data.bigsrc import _nli_boolean  # absolute: also runs as __main__ script
    if "major_premise" in inner:
        major = inner.get("major_premise")
        major_txt = " ".join(major) if isinstance(major, list) else str(major or "")
        state = (f"{major_txt.strip()}\n{str(inner.get('minor_premise', '')).strip()}\n"
                 f"Conclusion: {str(inner.get('conclusion', '')).strip()}")
    else:
        state = (f"{str(inner.get('premise', '')).strip()}\n"
                 f"Hypothesis: {str(inner.get('hypothesis', '')).strip()}")
    if len(state.strip()) < len("Conclusion: ") + 1:
        raise A.AdapterError(f"logiqa row {i}: empty NLI state")
    label = LOGIQA_LABEL_NORM.get(str(inner.get("label", "")).strip(),
                                  inner.get("label"))
    questions = _nli_boolean(i, None, label, r.get("split", "train"),
                             "logiqa20", prefix="logiqa")
    return A._checked(Example(state=state, split=r.get("split", "train"),
                              questions=questions), f"logiqa row {i}")


def adapt_logiqa(rows: list[dict]) -> tuple[list[Example], dict]:
    """Parse inner-JSON rows; returns (examples, skipped counts by reason).

    The LogiQA2.0 dump mixes multiple-choice shapes ({answer, options,
    question, text}) with NLI shapes ({major/minor_premise, conclusion,
    label} and {premise, hypothesis, label}); both convert, reasoning-type
    flags are always dropped (§124).
    """
    out, skipped = [], {"truncated": 0, "degenerate": 0}
    for i, r in enumerate(rows):
        try:
            inner = json.loads(r["text"])
        except Exception:
            skipped["truncated"] += 1  # truncated dump line (§124 delta, in audit)
            continue
        try:
            if "options" in inner:
                out.append(_logiqa_mc(i, r, inner))
            elif "label" in inner and ("premise" in inner or "major_premise" in inner):
                out.append(_logiqa_nli(i, r, inner))
            else:
                raise A.AdapterError(f"logiqa row {i}: unknown inner shape "
                                     f"{sorted(inner.keys())}")
        except A.AdapterError as e:
            if "empty" in str(e):
                skipped["degenerate"] += 1  # empty passage/question/option
                continue
            raise
    return out, skipped


def adapt_reclor(rows: list[dict]) -> list[Example]:
    out = []
    for i, r in enumerate(rows):
        try:
            options = ast.literal_eval(r["answers"])
            label = int(r["label"])
        except Exception as e:
            raise A.AdapterError(f"reclor row {i}: unparsable answers/label ({e})") from e
        if not isinstance(options, list) or not options:
            raise A.AdapterError(f"reclor row {i}: bad options")
        # label -1 = hidden gold (test split clean room): keep the input,
        # mark the answer unknown like adapt_huffpost does (§124).
        if label != -1 and not (0 <= label < len(options)):
            raise A.AdapterError(f"reclor row {i}: bad label")
        context = str(r.get("context", "")).strip()
        question = str(r.get("question", "")).strip()
        if not context or not question or any(not str(o).strip() for o in options):
            raise A.AdapterError(f"reclor row {i}: empty context/question/option")
        opts = [Option(id=f"opt{j}", text=str(o)) for j, o in enumerate(options)]
        out.append(A._checked(Example(
            state=f"{context}\n{question}",
            split=r.get("split", "train"),
            questions=[Question(id=f"reclor-{r.get('id_string', i)}", kind="choice",
                                options=opts,
                                answer=f"opt{label}" if label != -1 else "unknown")],
        ), f"reclor row {i}"))
    return out


def field_names(obj) -> list[str]:
    names = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            names.append(str(k))
            names.extend(field_names(v))
    elif isinstance(obj, list):
        for v in obj:
            names.extend(field_names(v))
    return names


def write(examples: list[Example], path: Path) -> None:
    with open(path, "w") as fout:
        for ex in examples:
            doc = dataclasses.asdict(ex)
            # §124: no explanation *fields* may be stored. (Prose may of
            # course contain ordinary words like "explanation", so only
            # field names are scanned.)
            keys = [k.lower() for k in field_names(doc)]
            if any(f in k for k in keys for f in FORBIDDEN):
                raise A.AdapterError(f"{path.name}: explanation field in output")
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")


def main() -> None:
    logiqa_rows = [json.loads(l) for l in open(LOGIQA_RAW) if l.strip()]
    reclor_rows = [json.loads(l) for l in open(RECLOR_RAW) if l.strip()]
    logiqa_ex, logiqa_skipped = adapt_logiqa(logiqa_rows)
    reclor_ex = adapt_reclor(reclor_rows)
    write(logiqa_ex, LOGIQA_OUT)
    write(reclor_ex, RECLOR_OUT)
    print(json.dumps({"job": "logiqa", "status": "converted",
                      "rows": len(logiqa_rows), "examples": len(logiqa_ex),
                      "skipped": logiqa_skipped,
                      "reasons": {"truncated": "truncated inner-JSON dump lines",
                                  "degenerate": "empty passage/question/option"},
                      "out": str(LOGIQA_OUT)}))
    print(json.dumps({"job": "reclor", "status": "converted",
                      "rows": len(reclor_rows), "examples": len(reclor_ex),
                      "skipped": len(reclor_rows) - len(reclor_ex),
                      "out": str(RECLOR_OUT)}))


if __name__ == "__main__":
    main()
