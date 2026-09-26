"""Batería de desarrollo — constructor y validador (#T-battery-dev).

Los 400 casos viven en `data/battery_dev.jsonl`, escritos a mano en 10
lotes (familia x idioma) contra la mezcla declarada en
`data/battery_dev_manifest.json` — mezcla escrita y fechada ANTES de
generar un solo caso. Este módulo no genera prosa: audita lo que hay.

Lo que verifica, y los tests exigen en vez de suponer:

* los 400 casos pasan el validador de `#T-episode-contract`;
* el reparto real familia x idioma x K coincide con la mezcla
  declarada dentro de la tolerancia escrita de antemano;
* el gold cumple la regla determinista con semilla (`GOLD_SALT`):
  la regla fijó el SLOT antes de existir la prosa; el autor escribió
  hechos que hacen correcto ese slot. Recomputable por cualquiera;
* cada `variant_group` es contrafactual de verdad: el control
  conserva el gold del original y la variante decisiva lo mueve;
* disciplina de namespace `dev-` / `sealed-` y, cuando el corte
  sellado exista, intersección de grupos = 0.

CPU puro. Validar los 400 casos tarda milisegundos.

Run:
    PYTHONPATH=. python3 -m data.battery_dev --gate
    PYTHONPATH=. pytest data/test_battery_dev.py -q
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = "T-battery-dev"

BATTERY = os.path.join(ROOT, "data", "battery_dev.jsonl")
MANIFEST = os.path.join(ROOT, "data", "battery_dev_manifest.json")
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", "T-battery-dev")

#: La regla que fijó los slots de gold antes de existir la prosa.
#: orig = H(grupo, "orig") mod K; decisive = (orig + step) mod K con
#: step = 1 + (H(grupo, "flip") mod (K-1)); control = orig.
GOLD_SALT = "battery-dev-gold-v1"

#: Dónde buscará el corte sellado cuando exista (#T-battery-sealed).
SEALED_GLOBS = (
    os.path.join(ROOT, "artifacts", "gates", "T-battery-sealed", "*.jsonl"),
    os.path.join(ROOT, "data", "battery_sealed.jsonl"),
)

ROLE_OF_SUFFIX = {"a": "orig", "b": "decisive", "c": "control"}
DEV_PREFIX = re.compile(r"^dev-")
SEALED_PREFIX = re.compile(r"^sealed-")


def load_battery(path: str = BATTERY) -> list[dict]:
    """Lee el corte; una línea, un episodio."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_manifest(path: str = MANIFEST) -> dict:
    """La mezcla declarada (fechada antes de generar)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def battery_sha(episodes: list[dict] | None = None) -> str:
    """Sha del corte: la versión con la que se mide."""
    if episodes is None:
        with open(BATTERY, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    payload = "\n".join(
        json.dumps(ep, sort_keys=True, ensure_ascii=False) for ep in episodes
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _h(group: str, role: str) -> int:
    digest = hashlib.sha256(
        f"{GOLD_SALT}\x00{group}\x00{role}".encode("utf-8")
    ).hexdigest()
    return int(digest, 16)


def expected_slots(variant_group: str, k: int) -> dict[str, int]:
    """Slots 1-based que la regla dicta para cada rol del grupo."""
    orig = _h(variant_group, "orig") % k
    step = 1 + _h(variant_group, "flip") % (k - 1)
    decisive = (orig + step) % k
    return {"orig": orig + 1, "decisive": decisive + 1, "control": orig + 1}


def role_of(episode_id: str) -> str | None:
    """El sufijo del id (a/b/c) dice el rol contrafactual."""
    if not isinstance(episode_id, str) or len(episode_id) < 2:
        return None
    return ROLE_OF_SUFFIX.get(episode_id[-1])


def check_gold(episodes: list[dict]) -> list[str]:
    """El gold escrito coincide con el slot que la regla dictó."""
    problems = []
    for ep in episodes:
        role = role_of(ep.get("id", ""))
        if role is None:
            problems.append(f"{ep.get('id')}: no role suffix (a/b/c)")
            continue
        k = len(ep.get("candidates", []))
        want = expected_slots(ep["variant_group"], k)[role]
        if ep.get("answer") != f"c{want}":
            problems.append(
                f"{ep.get('id')}: answer {ep.get('answer')!r} "
                f"breaks gold rule (role {role} wants c{want})"
            )
    return problems


def check_counterfactual_shape(episodes: list[dict]) -> list[str]:
    """Control conserva el gold; decisiva lo mueve; grupo de 2-3."""
    per_group: dict[str, list[dict]] = {}
    for ep in episodes:
        per_group.setdefault(ep["variant_group"], []).append(ep)
    problems = []
    for group, eps in sorted(per_group.items()):
        roles = sorted(role_of(ep.get("id", "")) or "?" for ep in eps)
        if roles not in (["decisive", "orig"], ["control", "decisive", "orig"]):
            problems.append(f"{group}: roles {roles}, want pair or triple")
            continue
        gold = {role_of(ep["id"]): ep.get("answer") for ep in eps}
        if "control" in gold and gold["control"] != gold["orig"]:
            problems.append(f"{group}: control gold moved ({gold})")
        if gold["decisive"] == gold["orig"]:
            problems.append(f"{group}: decisive gold did not move ({gold})")
    return problems


def check_distribution(episodes: list[dict], manifest: dict) -> list[str]:
    """Reparto real contra mezcla declarada (tolerancia del manifest)."""
    from collections import Counter

    tol = manifest.get("tolerance", {})
    cell_tol = 2
    try:
        cell_tol = int(
            re.search(r"\d+", tol.get("cell_family_lang_k", "")).group()
        )
    except (AttributeError, TypeError):
        pass
    declared = {
        (c["family"], c["lang"], 2): c["k2"]
        for c in manifest["declared_mix"]["cells"]
    }
    declared.update(
        {(c["family"], c["lang"], 3): c["k3"] for c in manifest["declared_mix"]["cells"]}
    )
    declared.update(
        {(c["family"], c["lang"], 8): c["k8"] for c in manifest["declared_mix"]["cells"]}
    )
    real = Counter(
        (ep["family"], ep["lang"], len(ep["candidates"])) for ep in episodes
    )
    problems = []
    for cell in sorted(set(declared) | set(real)):
        delta = real.get(cell, 0) - declared.get(cell, 0)
        if abs(delta) > cell_tol:
            problems.append(f"cell {cell}: real {real.get(cell, 0)} "
                            f"vs declared {declared.get(cell, 0)}")
    totals = manifest["declared_mix"]
    if len(episodes) != totals["cases_total"]:
        problems.append(f"n {len(episodes)} vs declared {totals['cases_total']}")
    return problems


def check_namespace(episodes: list[dict]) -> list[str]:
    """Disciplina de prefijos: dev- aquí, sealed- nunca aquí."""
    problems = []
    for ep in episodes:
        group = ep.get("variant_group", "")
        if not DEV_PREFIX.match(group):
            problems.append(f"{ep.get('id')}: group {group!r} lacks dev- prefix")
        if SEALED_PREFIX.match(group):
            problems.append(f"{ep.get('id')}: group {group!r} uses sealed- prefix")
    return problems


def find_sealed_files() -> list[str]:
    """Los ficheros del corte sellado, si ya existen."""
    found = []
    for pattern in SEALED_GLOBS:
        found.extend(glob.glob(pattern))
    return sorted(found)


def check_sealed_disjoint(episodes: list[dict]) -> tuple[str, list[str]]:
    """Intersección de grupos con el sellado: 0, o pendiente si no existe.

    Devuelve (estado, motivos): estado "absent" (sin sellado aún, regla
    de prefijos como única garantía), "clean" o "leak".
    """
    dev_groups = {ep["variant_group"] for ep in episodes}
    sealed_groups: set[str] = set()
    for path in find_sealed_files():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        sealed_groups.add(json.loads(line)["variant_group"])
                    except (KeyError, ValueError):
                        pass
    if not find_sealed_files():
        return "absent", []
    shared = sorted(dev_groups & sealed_groups)
    if shared:
        return "leak", [f"shared variant_group with sealed: {g}" for g in shared]
    return "clean", []


def run_gate(outdir: str = GATE_DIR) -> dict[str, Any]:
    """Ejecuta el gate de la task y escribe `gate.json` con valores MEDIDOS."""
    from . import episode_contract as EC

    episodes = load_battery()
    manifest = load_manifest()
    validation = EC.batch_validate(episodes)
    invalid_detail = [
        {"id": ep.get("id"), "reasons": reasons}
        for ep, reasons in zip(episodes, validation["per_episode"])
        if reasons
    ]
    gold_problems = check_gold(episodes)
    shape_problems = check_counterfactual_shape(episodes)
    dist_problems = check_distribution(episodes, manifest)
    ns_problems = check_namespace(episodes)
    sealed_state, sealed_problems = check_sealed_disjoint(episodes)
    verdict = (
        "GO"
        if not (
            invalid_detail
            or gold_problems
            or shape_problems
            or dist_problems
            or ns_problems
            or sealed_problems
        )
        else "NO-GO"
    )
    gate = {
        "task": TASK,
        "verdict": verdict,
        "battery_sha256": battery_sha(),
        "manifest_version": manifest.get("version"),
        "manifest_date": manifest.get("date_declared"),
        "schema_version": EC.SCHEMA_VERSION,
        "schema_sha": EC.schema_sha(),
        "measured": {
            "n": validation["n"],
            "n_valid": validation["n_valid"],
            "n_invalid": validation["n_invalid"],
            "duplicate_ids": validation["duplicate_ids"],
            "gold_rule_violations": len(gold_problems),
            "counterfactual_shape_violations": len(shape_problems),
            "distribution_violations": len(dist_problems),
            "namespace_violations": len(ns_problems),
            "sealed_state": sealed_state,
            "sealed_shared_groups": len(sealed_problems),
        },
        "problems": {
            "invalid": invalid_detail,
            "gold": gold_problems,
            "shape": shape_problems,
            "distribution": dist_problems,
            "namespace": ns_problems,
            "sealed": sealed_problems,
        },
    }
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "gate.json"), "w", encoding="utf-8") as fh:
        json.dump(gate, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return gate


def main() -> None:
    """CLI mínima: `--gate` escribe el gate; sin flags, resume el corte."""
    parser = argparse.ArgumentParser(description="Development battery auditor")
    parser.add_argument("--gate", action="store_true", help="write gate.json")
    args = parser.parse_args()
    if args.gate:
        gate = run_gate()
        print(json.dumps(gate["measured"], indent=1))
        print("verdict:", gate["verdict"])
        return
    episodes = load_battery()
    print(f"{len(episodes)} episodes, "
          f"{len({e['variant_group'] for e in episodes})} groups")


if __name__ == "__main__":
    main()
