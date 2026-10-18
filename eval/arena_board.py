"""Professional arena board for the pointer against JEv.

Every row names the cut, the chance rate, and whether each score was
measured on that cut or copied from a citation. A blank JEv cell means
no same-protocol number exists. The sealed battery is not opened here.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = "T-arena-board"

PUBLIC = (
    ("xnli", "test", "XNLI", "entailment, English and Spanish, 3-way"),
    ("paws-x", "test", "PAWS-X", "paraphrase, English and Spanish, yes/no"),
    ("boolq", "validation", "BoolQ", "yes/no over a passage"),
    ("ag-news", "test", "AG News", "4-way topic"),
    ("emotion", "test", "Emotion", "affect label"),
    ("go-emotions", "test", "GoEmotions", "emotion label"),
    ("commonsense-qa", "validation", "CommonsenseQA", "commonsense, 5-way"),
    ("helpsteer2", "validation", "HelpSteer2", "ordinal quality, 0 to 4"),
    ("prompt-injections", "test", "Prompt injection", "binary security flag"),
    ("massive", "test", "MASSIVE", "intent, English and Spanish"),
    ("banking77", "test", "BANKING77", "77-way intent, official test"),
)

FAMILIES = (
    "attribute_comparison",
    "description_classification",
    "extraction_paraphrase",
    "priority_decision",
    "textual_inference_negation",
)


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _row(**fields) -> dict:
    base = {
        "id": "",
        "title": "",
        "n": None,
        "k": None,
        "chance": None,
        "protocol": "",
        "clone": None,
        "clone_kind": "unscored",
        "clone_note": "",
        "jev": None,
        "jev_kind": "absent",
        "jev_note": "",
        "qwen": None,
    }
    base.update(fields)
    return base


def _family_rows(clone: dict, jev: dict) -> list[dict]:
    rows = []
    clone_fam = clone.get("by_family") or {}
    jev_fam = ((jev.get("report") or {}).get("by_family") or {})
    step = clone.get("adapter_step")
    for name in FAMILIES:
        cell = clone_fam.get(name) or {}
        jev_cell = ((jev_fam.get(name) or {}).get("ranking") or {})
        rows.append(_row(
            id=f"battery-dev:{name}",
            title=name.replace("_", " "),
            n=cell.get("n") or jev_cell.get("n") or 80,
            k="2/3/8",
            chance=0.3375,
            protocol="battery-dev family slice, same rows",
            clone=cell.get("forced"),
            clone_kind="measured" if cell.get("forced") is not None else "unscored",
            clone_note=f"written order, adapter step {step}" if step else "",
            jev=jev_cell.get("accuracy"),
            jev_kind="measured" if jev_cell.get("accuracy") is not None else "absent",
            jev_note="same 400-row development cut" if jev_cell else "",
        ))
    return rows


def _k_rows(clone: dict) -> list[dict]:
    grouped = {2: [0, 0], 3: [0, 0], 8: [0, 0]}
    for key, cell in (clone.get("by_k") or {}).items():
        if not isinstance(cell, dict) or "k=" not in key:
            continue
        k = int(key.rsplit("k=", 1)[1])
        if k not in grouped:
            continue
        grouped[k][0] += int(cell.get("hits") or 0)
        grouped[k][1] += int(cell.get("n") or 0)
    chance = {2: 0.5, 3: 1 / 3, 8: 0.125}
    rows = []
    step = clone.get("adapter_step")
    for k, (hits, n) in grouped.items():
        score = (hits / n) if n else None
        rows.append(_row(
            id=f"battery-dev:k={k}",
            title=f"Decision battery K={k}",
            n=n or None,
            k=str(k),
            chance=chance[k],
            protocol="battery-dev rows with this candidate count",
            clone=score,
            clone_kind="measured" if score is not None else "unscored",
            clone_note=f"written order, adapter step {step}" if step else "",
            jev_kind="absent",
            jev_note="JEv published the mixed-K cut, not this slice alone",
        ))
    return rows


def _public_rows() -> list[dict]:
    rows = []
    root = ROOT / "artifacts" / "episodes-external"
    for key, split, title, protocol in PUBLIC:
        man = _read(root / key / split / "manifest.json")
        n = man.get("n_episodes")
        cited = key == "banking77"
        rows.append(_row(
            id=key,
            title=title,
            n=n,
            k="77" if key == "banking77" else None,
            chance=(1 / 77) if key == "banking77" else None,
            protocol=protocol + (", eval-only" if man.get("eval_only") else ""),
            clone_kind="unscored",
            clone_note="4B pointer has not been scored on this cut",
            jev=0.924 if cited else None,
            jev_kind="cited" if cited else "absent",
            jev_note=("public BANKING77 experiment, richer prompt, not this pipeline"
                      if cited else "no same-protocol JEv score"),
        ))
    return rows


def build_board() -> list[dict]:
    clone = _read(ROOT / "artifacts" / "checkpoints" / "pointer" / "loop"
                  / "battery_regression.json")
    jev = _read(ROOT / "artifacts" / "gates" / "T-teacher-probe" / "battery.json")
    qwen = _read(ROOT / "artifacts" / "gates" / "T-qwen38-ref" / "gate.json")
    ranking = (jev.get("report") or {}).get("ranking") or {}
    qwen_acc = (((qwen.get("columns") or {}).get("qwen3.8") or {})
                .get("forced") or {}).get("accuracy")
    step = clone.get("adapter_step")
    head = _row(
        id="battery-dev",
        title="Decision battery",
        n=clone.get("n") or ranking.get("n") or 400,
        k="2/3/8",
        chance=0.3375,
        protocol="same 400 development rows, English and Spanish",
        clone=clone.get("forced"),
        clone_kind="measured" if clone.get("forced") is not None else "unscored",
        clone_note=f"written order, adapter step {step}" if step else "",
        jev=ranking.get("accuracy") if jev.get("n_scored") == 400 else None,
        jev_kind="measured" if jev.get("n_scored") == 400 else "absent",
        jev_note="remote teacher, same rows and candidate sets",
        qwen=qwen_acc,
    )
    return [head, *_family_rows(clone, jev), *_k_rows(clone), *_public_rows()]


def write_board(path: Path | None = None) -> Path:
    dest = path or (ROOT / "artifacts" / "gates" / TASK / "board.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"task": TASK, "arenas": build_board()}
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def main() -> None:
    dest = write_board()
    rows = build_board()
    measured = sum(1 for row in rows if row["clone_kind"] == "measured")
    print(f"[arena] {len(rows)} arenas, {measured} with a clone score -> {dest}")


if __name__ == "__main__":
    main()
