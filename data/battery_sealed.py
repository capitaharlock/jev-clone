"""Batería sellada — auditor y guardián de apertura única (#T-battery-sealed).

Los 600 casos viven en `data/battery_sealed.jsonl`, escritos a mano contra
la mezcla declarada en `data/battery_sealed_manifest.json` — mezcla,
thresholds de solape y sal del gold escritos y fechados ANTES de redactar
un solo caso. El diagnóstico K=20/77 vive aparte en
`data/battery_diag.jsonl` y nunca entra en la meta del 70 %. Este módulo
no genera prosa: audita lo que hay y custodia la apertura.

Lo que verifica, y los tests exigen en vez de suponer:

* los 600 casos pasan el validador de `#T-episode-contract`;
* el reparto real familia x idioma x K coincide con la mezcla
  declarada dentro de la tolerancia escrita de antemano;
* el gold cumple la regla determinista con SU propia semilla
  (`GOLD_SALT`, distinta de la de desarrollo): la regla fijó el SLOT
  antes de existir la prosa;
* cada `variant_group` es contrafactual de verdad y ninguno se
  comparte con desarrollo (intersección = 0, medida, no comentada);
* el solape entidad/espacio con desarrollo está por debajo del umbral
  que el manifest escribió antes de medir (`overlap_rule`);
* el diagnóstico K=20/77 se reporta en su propio bloque:
  `build_target_report` solo lee el corte sellado y RECHAZA
  cualquier mezcla;
* apertura única: solo `#T-ce-confirm` abre el sellado, una sola vez;
  el runner registra sha + flag de consumido y la segunda apertura
  es REFUSADA. Nada de esta task lo abre (los tests usan copias
  temporales, nunca el ledger real).

CPU puro. Validar los 600 casos tarda milisegundos.

Run:
    PYTHONPATH=. python3 -m data.battery_sealed --gate
    PYTHONPATH=. pytest data/test_battery_sealed.py -q
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = "T-battery-sealed"

SEALED = os.path.join(ROOT, "data", "battery_sealed.jsonl")
MANIFEST = os.path.join(ROOT, "data", "battery_sealed_manifest.json")
DEV_BATTERY = os.path.join(ROOT, "data", "battery_dev.jsonl")
DIAG = os.path.join(ROOT, "data", "battery_diag.jsonl")
DIAG_MANIFEST = os.path.join(ROOT, "data", "battery_diag_manifest.json")
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", "T-battery-sealed")
OPENED = os.path.join(GATE_DIR, "opened.json")

#: La regla que fijó los slots de gold antes de existir la prosa.
#: Sal propia (distinta de la de desarrollo): mismo mecanismo,
#: distinto sorteo. orig = H(grupo, "orig") mod K;
#: decisive = (orig + step) mod K con step = 1 + H(grupo, "flip") mod (K-1).
GOLD_SALT = "battery-sealed-gold-v1"

#: Sal del diagnóstico K=20/77 (corte aparte, mismo mecanismo).
DIAG_GOLD_SALT = "battery-diag-gold-v1"

#: Solo este task abre el sellado, y una sola vez.
AUTHORIZED_OPENER = "T-ce-confirm"

#: K del corte de meta (el 70 % se mide aquí) y del diagnóstico.
TARGET_KS = (2, 3, 8)
DIAG_KS = (20, 77)

SEALED_PREFIX = re.compile(r"^sealed-")
DEV_PREFIX = re.compile(r"^dev-")
DIAG_PREFIX = re.compile(r"^diag-")

#: entity-v1: tokens alfabéticos de longitud >= 5 del `state`,
#: menos estas funcionales. Declarado en el manifest ANTES de medir;
#: el gate lo reproduce al céntimo.
STOPWORDS = frozenset(
    """
    todos todas para pero porque como donde cuando entre sobre hasta desde
    esta este estos estas con sin sus nuestra nuestro durante mediante cada
    otro otra otros otras mismo misma mismos mismas hace hacen hacia tiene
    tienen despues siempre nunca tambien tampoco puede deben debe mientras
    aunque pues cual cuales cuanto cuantos quien quienes
    about with from that this these those which their them they have does
    were been under over after before every other same such than then there
    when where while both each into would could should never always today
    tonight still again first second third fourth fifth sixth seventh eighth
    against along seven eight three
    """.split()
)


class SealedAccessDenied(Exception):
    """Otro task que no sea #T-ce-confirm intentó abrir el sellado."""


class SealedAlreadyConsumed(Exception):
    """El sellado ya se abrió una vez: la segunda apertura se RECHAZA."""


def load_sealed(path: str = SEALED) -> list[dict]:
    """Lee el corte sellado; una línea, un episodio."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_manifest(path: str = MANIFEST) -> dict:
    """La mezcla declarada (fechada antes de redactar)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_diag(path: str = DIAG) -> list[dict]:
    """El diagnóstico K=20/77: corte aparte, nunca meta."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def file_sha(path: str = SEALED) -> str:
    """Sha del fichero: la versión con la que se mide."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _h(salt: str, group: str, role: str) -> int:
    digest = hashlib.sha256(
        f"{salt}\x00{group}\x00{role}".encode("utf-8")
    ).hexdigest()
    return int(digest, 16)


def expected_slots(variant_group: str, k: int) -> dict[str, int]:
    """Slots 1-based que la regla sellada dicta para cada rol."""
    orig = _h(GOLD_SALT, variant_group, "orig") % k
    step = 1 + _h(GOLD_SALT, variant_group, "flip") % (k - 1)
    decisive = (orig + step) % k
    return {"orig": orig + 1, "decisive": decisive + 1, "control": orig + 1}


def diag_expected_slots(variant_group: str, k: int) -> dict[str, int]:
    """Slots del diagnóstico (su propia sal, mismo mecanismo)."""
    orig = _h(DIAG_GOLD_SALT, variant_group, "orig") % k
    step = 1 + _h(DIAG_GOLD_SALT, variant_group, "flip") % (k - 1)
    decisive = (orig + step) % k
    return {"orig": orig + 1, "decisive": decisive + 1, "control": orig + 1}


def check_gold(episodes: list[dict]) -> list[str]:
    """El gold escrito coincide con el slot que la regla sellada dictó."""
    from .battery_dev import role_of

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
                f"breaks sealed gold rule (role {role} wants c{want})"
            )
    return problems


def check_diag_gold(episodes: list[dict]) -> list[str]:
    """El gold del diagnóstico cumple SU regla (sal propia)."""
    from .battery_dev import role_of

    problems = []
    for ep in episodes:
        role = role_of(ep.get("id", ""))
        if role is None:
            problems.append(f"{ep.get('id')}: no role suffix (a/b/c)")
            continue
        k = len(ep.get("candidates", []))
        want = diag_expected_slots(ep["variant_group"], k)[role]
        if ep.get("answer") != f"c{want}":
            problems.append(
                f"{ep.get('id')}: answer {ep.get('answer')!r} "
                f"breaks diag gold rule (role {role} wants c{want})"
            )
    return problems


def check_namespace(episodes: list[dict]) -> list[str]:
    """Disciplina de prefijos: sealed- aquí; dev-/diag- nunca aquí."""
    problems = []
    for ep in episodes:
        group = ep.get("variant_group", "")
        if not SEALED_PREFIX.match(group):
            problems.append(f"{ep.get('id')}: group {group!r} lacks sealed- prefix")
        if DEV_PREFIX.match(group) or DIAG_PREFIX.match(group):
            problems.append(f"{ep.get('id')}: group {group!r} uses foreign prefix")
    return problems


def check_dev_disjoint(episodes: list[dict]) -> list[str]:
    """Intersección de variant_group con desarrollo: 0, medido."""
    sealed_groups = {ep["variant_group"] for ep in episodes}
    dev_groups: set[str] = set()
    with open(DEV_BATTERY, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                dev_groups.add(json.loads(line)["variant_group"])
    return [
        f"shared variant_group with dev: {g}"
        for g in sorted(sealed_groups & dev_groups)
    ]


def entity_types(episodes: list[dict]) -> set[str]:
    """entity-v1: tipos del manifest (state, token >= 5, sin STOPWORDS)."""
    out: set[str] = set()
    for ep in episodes:
        toks = re.findall(r"[a-záéíóúñü]+", ep.get("state", "").lower())
        out.update(w for w in toks if len(w) >= 5 and w not in STOPWORDS)
    return out


def candidate_texts(episodes: list[dict]) -> set[str]:
    """Textos de candidato normalizados (minúsculas, espacio colapsado)."""
    return {
        re.sub(r"\s+", " ", c.get("text", "").lower()).strip()
        for ep in episodes
        for c in ep.get("candidates", [])
    }


def measure_overlap(episodes: list[dict]) -> dict[str, Any]:
    """Solape con desarrollo contra los umbrales del manifest."""
    manifest = load_manifest()
    rule = manifest["overlap_rule"]
    with open(DEV_BATTERY, encoding="utf-8") as fh:
        dev = [json.loads(line) for line in fh if line.strip()]
    sealed_types, dev_types = entity_types(episodes), entity_types(dev)
    entity_overlap = len(sealed_types & dev_types) / len(sealed_types)
    sealed_cands, dev_cands = candidate_texts(episodes), candidate_texts(dev)
    cand_overlap = len(sealed_cands & dev_cands) / len(sealed_cands)
    entity_thr = float(rule["entity"].split("<=")[1].strip())
    return {
        "entity_overlap": entity_overlap,
        "entity_threshold": entity_thr,
        "entity_pass": entity_overlap <= entity_thr,
        "entity_n_sealed": len(sealed_types),
        "entity_n_shared": len(sealed_types & dev_types),
        "candidate_overlap": cand_overlap,
        "candidate_pass": cand_overlap == 0,
        "candidate_n_shared": len(sealed_cands & dev_cands),
    }


def is_consumed(ledger_path: str = OPENED) -> bool:
    """True si el sellado ya se abrió (flag registrado)."""
    return os.path.exists(ledger_path)


def open_sealed(
    task_id: str,
    sealed_path: str = SEALED,
    ledger_path: str = OPENED,
) -> list[dict]:
    """Abre el sellado UNA vez, solo para #T-ce-confirm.

    Registra el sha + flag de consumido y devuelve los episodios.
    Cualquier otro task recibe `SealedAccessDenied`; una segunda
    apertura recibe `SealedAlreadyConsumed` (sale non-zero en CLI).
    """
    if task_id != AUTHORIZED_OPENER:
        raise SealedAccessDenied(
            f"sealed cut opens only for {AUTHORIZED_OPENER}, not {task_id!r}"
        )
    if os.path.exists(ledger_path):
        with open(ledger_path, encoding="utf-8") as fh:
            record = json.load(fh)
        raise SealedAlreadyConsumed(
            f"sealed cut already consumed by {record.get('opened_by')} "
            f"at {record.get('opened_at_utc')} (sha {record.get('battery_sha256')})"
        )
    with open(sealed_path, encoding="utf-8") as fh:
        episodes = [json.loads(line) for line in fh if line.strip()]
    with open(sealed_path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    with open(ledger_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "battery_sha256": sha,
                "opened_by": task_id,
                "opened_at_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "consumed": True,
            },
            fh,
            indent=1,
        )
        fh.write("\n")
    return episodes


def build_target_report(
    episodes: list[dict] | None = None,
    sealed_path: str = SEALED,
) -> dict[str, Any]:
    """El bloque de la meta del 70 %: SOLO el corte sellado K=2/3/8.

    Lee únicamente el fichero sellado y RECHAZA (`ValueError`) todo
    episodio diag- o con K fuera de {2, 3, 8}. El diagnóstico se
    reporta en su propio bloque, jamás promediado aquí.
    """
    if episodes is None:
        episodes = load_sealed(sealed_path)
    from collections import Counter

    problems = []
    for ep in episodes:
        if DIAG_PREFIX.match(ep.get("variant_group", "")):
            problems.append(f"{ep.get('id')}: diag case in target block")
        if len(ep.get("candidates", [])) not in TARGET_KS:
            problems.append(f"{ep.get('id')}: K outside target set {TARGET_KS}")
    if problems:
        raise ValueError(
            "target block mixes non-target cases: " + "; ".join(problems[:5])
        )
    k_counts = Counter(len(ep["candidates"]) for ep in episodes)
    return {
        "cut": "sealed-target",
        "n": len(episodes),
        "k_counts": {str(k): k_counts.get(k, 0) for k in TARGET_KS},
        "families": sorted({ep["family"] for ep in episodes}),
        "battery_sha256": file_sha(sealed_path),
        "diag_included": False,
    }


def run_gate(outdir: str = GATE_DIR) -> dict[str, Any]:
    """Ejecuta el gate de la task y escribe `gate.json` con valores MEDIDOS."""
    from . import episode_contract as EC
    from .battery_dev import check_counterfactual_shape, check_distribution

    episodes = load_sealed()
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
    disjoint_problems = check_dev_disjoint(episodes)
    overlap = measure_overlap(episodes)

    diag = load_diag()
    diag_manifest = json.load(open(DIAG_MANIFEST, encoding="utf-8"))
    diag_validation = EC.batch_validate(diag)
    diag_invalid = [
        {"id": ep.get("id"), "reasons": reasons}
        for ep, reasons in zip(diag, diag_validation["per_episode"])
        if reasons
    ]
    diag_gold_problems = check_diag_gold(diag)
    diag_ks = sorted({len(ep["candidates"]) for ep in diag})
    diag_ns = [
        f"{ep.get('id')}: group {ep.get('variant_group')!r} lacks diag- prefix"
        for ep in diag
        if not DIAG_PREFIX.match(ep.get("variant_group", ""))
    ]

    from collections import Counter

    k_counts = Counter(len(ep["candidates"]) for ep in episodes)
    consumed = is_consumed()
    checks = {
        "n": validation["n"] == 600,
        "validity": validation["n_invalid"] == 0,
        "gold": not gold_problems,
        "shape": not shape_problems,
        "distribution": not dist_problems,
        "namespace": not ns_problems,
        "dev_disjoint": not disjoint_problems,
        "entity_overlap": overlap["entity_pass"],
        "candidate_overlap": overlap["candidate_pass"],
        "k_distribution": (
            k_counts.get(2, 0) == 240
            and k_counts.get(3, 0) == 180
            and k_counts.get(8, 0) == 180
        ),
        "diag_valid": diag_validation["n_invalid"] == 0,
        "diag_gold": not diag_gold_problems,
        "diag_ks": diag_ks == [20, 77],
        "diag_namespace": not diag_ns,
    }
    verdict = "GO" if all(checks.values()) else "NO-GO"
    gate = {
        "task": TASK,
        "verdict": verdict,
        "battery_sha256": file_sha(),
        "manifest_version": manifest.get("version"),
        "manifest_date": manifest.get("date_declared"),
        "schema_version": EC.SCHEMA_VERSION,
        "schema_sha": EC.schema_sha(),
        "consumed": consumed,
        "checks": checks,
        "measured": {
            "n": validation["n"],
            "n_valid": validation["n_valid"],
            "n_invalid": validation["n_invalid"],
            "duplicate_ids": validation["duplicate_ids"],
            "gold_rule_violations": len(gold_problems),
            "counterfactual_shape_violations": len(shape_problems),
            "distribution_violations": len(dist_problems),
            "namespace_violations": len(ns_problems),
            "dev_shared_groups": len(disjoint_problems),
            "entity_overlap": round(overlap["entity_overlap"], 4),
            "entity_threshold": overlap["entity_threshold"],
            "entity_n_sealed": overlap["entity_n_sealed"],
            "entity_n_shared": overlap["entity_n_shared"],
            "candidate_overlap": round(overlap["candidate_overlap"], 4),
            "candidate_n_shared": overlap["candidate_n_shared"],
            "k_counts": {str(k): k_counts.get(k, 0) for k in sorted(k_counts)},
            "diag_n": diag_validation["n"],
            "diag_n_valid": diag_validation["n_valid"],
            "diag_gold_violations": len(diag_gold_problems),
            "diag_ks": diag_ks,
            "diag_namespace_violations": len(diag_ns),
            "diag_manifest_version": diag_manifest.get("version"),
        },
        "problems": {
            "invalid": invalid_detail,
            "gold": gold_problems,
            "shape": shape_problems,
            "distribution": dist_problems,
            "namespace": ns_problems,
            "dev_disjoint": disjoint_problems,
            "diag_invalid": diag_invalid,
            "diag_gold": diag_gold_problems,
            "diag_namespace": diag_ns,
        },
    }
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "gate.json"), "w", encoding="utf-8") as fh:
        json.dump(gate, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return gate


def main() -> None:
    """CLI mínima: `--gate` escribe el gate; `--open` abre (solo T-ce-confirm)."""
    parser = argparse.ArgumentParser(description="Sealed battery auditor")
    parser.add_argument("--gate", action="store_true", help="write gate.json")
    parser.add_argument("--open", metavar="TASK", help="open sealed cut once")
    args = parser.parse_args()
    if args.gate:
        gate = run_gate()
        print(json.dumps(gate["measured"], indent=1))
        print("verdict:", gate["verdict"])
        return
    if args.open:
        try:
            episodes = open_sealed(args.open)
        except (SealedAccessDenied, SealedAlreadyConsumed) as exc:
            print(f"REFUSED: {exc}")
            raise SystemExit(1)
        print(f"opened {len(episodes)} sealed episodes for {args.open}")
        return
    episodes = load_sealed()
    print(f"{len(episodes)} sealed episodes, "
          f"{len({e['variant_group'] for e in episodes})} groups, "
          f"consumed={is_consumed()}")


if __name__ == "__main__":
    main()
