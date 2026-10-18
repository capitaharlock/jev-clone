"""Build a fresh, auditable listwise mixture for one training cycle.

The first listwise supervisor accidentally reused batch 48 and the verified
pilot on every cycle. This builder rotates through untouched rule batches and
adds fresh rule-backed replay for all five families. No episode is repeated
until all eligible rule batches have been consumed once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from data import episode_contract as EC
from data import episode_gen as EG
from data import rule_variety as RV

ROOT = Path(__file__).resolve().parents[2]
RULE_ROOT = ROOT / "artifacts" / "episodes-rule"
EXHAUSTED_RULE_BATCHES = {48}
GENERAL_PER_CYCLE = 3000
ALIGNED_START_CYCLE = 62
ALIGNED_FORMS = {
    EC.COMPARISON: "superlative",
    EC.PRIORITY: "chain2",
}
ALIGNED_ROWS_PER_CELL = {2: 1200, 3: 900, 8: 900}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_rule_batches(rule_root: Path = RULE_ROOT) -> list[Path]:
    batches = []
    for path in sorted(rule_root.glob("batch-*")):
        try:
            number = int(path.name.split("-")[-1])
        except ValueError:
            continue
        if number in EXHAUSTED_RULE_BATCHES:
            continue
        if (path / "episodes.jsonl").exists():
            batches.append(path)
    return batches


def rule_batch_for_cycle(cycle: int, rule_root: Path = RULE_ROOT) -> Path:
    batches = eligible_rule_batches(rule_root)
    if not batches:
        raise ValueError("no unexhausted rule batches are available")
    # Cycle 29 is the first audited cycle and starts at batch 1.
    index = max(0, int(cycle) - 29) % len(batches)
    return batches[index]


def aligned_numeric_rows(
        cycle: int, rule_root: Path = RULE_ROOT, *,
        rows_per_cell: dict[int, int] = ALIGNED_ROWS_PER_CELL,
        start_cycle: int = ALIGNED_START_CYCLE) -> tuple[list[dict], dict]:
    """Select whole variant groups with the battery's family/lang/K balance."""
    cells: dict[tuple, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list))
    sources = eligible_rule_batches(rule_root)
    for directory in sources:
        for row in _read_jsonl(directory / "episodes.jsonl"):
            family = row["family"]
            trace = row.get("rule_trace") or {}
            if trace.get("form") != ALIGNED_FORMS.get(family):
                continue
            key = (family, row["lang"], len(row["candidates"]))
            if key[2] not in rows_per_cell:
                continue
            cells[key][row["variant_group"]].append(row)

    selected: list[dict] = []
    cell_counts = {}
    round_index = max(0, int(cycle) - start_cycle)
    for family in (EC.COMPARISON, EC.PRIORITY):
        for lang in ("en", "es"):
            for k, row_target in rows_per_cell.items():
                key = (family, lang, k)
                complete = {
                    group: rows for group, rows in cells[key].items()
                    if len(rows) == len(RV.VARIANTS)
                }
                group_target = row_target // len(RV.VARIANTS)
                ordered = sorted(
                    complete,
                    key=lambda group: hashlib.sha256(
                        f"aligned-v1\0{group}".encode()).digest())
                if len(ordered) < group_target:
                    raise ValueError(
                        f"not enough aligned groups for {key}: "
                        f"{len(ordered)} < {group_target}")
                offset = (round_index * group_target) % len(ordered)
                chosen = (ordered + ordered)[offset:offset + group_target]
                rows = [row for group in chosen for row in complete[group]]
                selected.extend(RV.relayout_battery(row) for row in rows)
                cell_counts[f"{family}/{lang}/K{k}"] = len(rows)
    return selected, {
        "profile": "battery-aligned-numeric-v1",
        "forms": dict(ALIGNED_FORMS),
        "rows_per_family_lang_k": cell_counts,
        "source_batches": [int(p.name.split("-")[-1]) for p in sources],
        "round": round_index,
    }


def _census(rows: list[dict]) -> dict:
    return {
        "by_family": dict(sorted(Counter(r["family"] for r in rows).items())),
        "by_lang": dict(sorted(Counter(r["lang"] for r in rows).items())),
        "by_k": dict(sorted(Counter(str(len(r["candidates"]))
                                    for r in rows).items())),
        "groups": len({r["variant_group"] for r in rows}),
    }


def build(cycle: int, out: Path, *, rule_root: Path = RULE_ROOT,
          general_n: int = GENERAL_PER_CYCLE) -> dict:
    out = Path(out)
    mix_file = out / "episodes.jsonl"
    manifest_file = out / "manifest.json"
    if mix_file.exists() and manifest_file.exists():
        return json.loads(manifest_file.read_text())

    rule_rows, aligned = aligned_numeric_rows(cycle, rule_root)

    general_dir = out.parent / "general"
    general_file = general_dir / "episodes.jsonl"
    if not general_file.exists():
        EG.run(
            n=general_n,
            seed=30300000 + int(cycle),
            prose="local",
            teacher="stub",
            out_dir=str(general_dir),
            families=[EC.DESCRIPTION, EC.EXTRACTION, EC.INFERENCE],
            variety=0,
            group_prefix=f"listwise{int(cycle):04d}-",
            layout="battery",
        )
    general_rows = _read_jsonl(general_file)
    rows = rule_rows + general_rows
    verdict = EC.batch_validate(rows)
    if verdict["n_invalid"] or verdict["duplicate_ids"]:
        raise ValueError(
            f"invalid listwise mix: {verdict['n_invalid']} invalid, "
            f"{len(verdict['duplicate_ids'])} duplicate ids")

    out.mkdir(parents=True, exist_ok=True)
    with mix_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    id_digest = hashlib.sha256(
        "\n".join(sorted(r["id"] for r in rows)).encode()).hexdigest()
    manifest = {
        "format": "jev.listwise-mix.v3",
        "cycle": int(cycle),
        "repeat_policy": {
            "max_exposures_per_episode": 4,
            "selection": "whole groups from a battery-aligned rotating pool",
            "exhausted_rule_batches": sorted(EXHAUSTED_RULE_BATCHES),
            "eligible_batches": len(eligible_rule_batches(rule_root)),
        },
        "sources": {
            "aligned_numeric_rule": {
                "n": len(rule_rows),
                **aligned,
            },
            "fresh_general_replay": {
                "dir": str(general_dir),
                "n": len(general_rows),
                "sha256": _sha(general_file),
                "generator": (
                    "episode_gen local/stub, battery layout, "
                    "three nonnumeric families"),
            },
        },
        "n_episodes": len(rows),
        "episode_ids_sha256": id_digest,
        "episodes_sha256": _sha(mix_file),
        **_census(rows),
    }
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--general-n", type=int, default=GENERAL_PER_CYCLE)
    args = parser.parse_args(argv)
    doc = build(args.cycle, Path(args.out), general_n=args.general_n)
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
