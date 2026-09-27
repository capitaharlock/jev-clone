"""Prueba de valor del generador por regla (#T-numeric-gen, punto 5).

Antes de meter el generador en el bucle hay que saber si sirve. La task
lo fija: entrenar el smoke con 5 000 episodios por regla MÁS el piloto
verificado, el mismo comando que `smoke.json` con `--budget 5000`, y
medir dev por familia y el holdout del piloto por familia. Con la
predicción escrita ANTES (`PREDICTION`, en este fichero y en el gate).

Tres subcomandos:

* `predict` — escribe `PREDICTION` en la sección `value_proof_prediction`
  del gate de la task. Se ejecuta antes de la mezcla y antes del entreno;
  si la sección ya está, no se toca (una predicción reescrita después de
  ver una cifra no es una predicción).
* `mix`     — construye el directorio de entreno: una muestra POR GRUPO
  del lote por regla, equilibrada entre las cuatro celdas
  (familia × idioma), más el piloto verificado entero. Manifest con el
  sha256 de cada fuente y el conteo por celda.
* `measure` — puntúa el holdout del piloto con el control SIN AJUSTAR y
  con el checkpoint del smoke, sobre las MISMAS filas, y publica la
  diferencia pareada por familia. Las cifras salen de
  `eval.metrics_suite.report`; el IC95 % de la diferencia, del bootstrap
  pareado determinista de `training.python.ce_finetune`.
* `gate`    — junta dev (de `smoke.json`, que ya lo mide) con el holdout
  y firma el veredicto por la regla predeclarada.

El holdout del piloto NO es dev: es el 20 % por grupo de lo que entra en
el entreno (`training.python.ce_finetune.provisional_split`, misma
semilla), así que se puede recomputar sin volver a entrenar. Es la única
medición que decide aquí, porque es la que `#T-ce-finetune` dejó clavada
en 0,574 → 0,574.

    python -m eval.numeric_value predict
    python -m eval.numeric_value mix --rule artifacts/episodes-rule/batch-0001 \\
        --pilot artifacts/episodes-qwen/pilot-2k/verified.jsonl --n-rule 5000 \\
        --out artifacts/episodes-rule/mix-smoke
    python -m eval.numeric_value measure --run artifacts/checkpoints/ce/numeric-5k \\
        --mix artifacts/episodes-rule/mix-smoke
    python -m eval.numeric_value gate --run artifacts/checkpoints/ce/numeric-5k
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import episode_contract as EC  # noqa: E402
from data import rule_variety as RV  # noqa: E402
from eval import calib as C  # noqa: E402
from eval import metrics_suite as MS  # noqa: E402
from eval import preflight_refs as PR  # noqa: E402
from model import ce_scorer as CE  # noqa: E402

TASK = "T-numeric-gen"
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
FORMAT = "jev.numeric-value.v1"

#: El corte que decide aquí, con su nombre propio.
CUT_NAME = "pilot-holdout"

#: La familia que la task manda mirar: la que no se aprende.
TARGET_FAMILY = EC.COMPARISON

#: LA PREDICCIÓN, escrita el 2026-09-27 antes de construir la mezcla,
#: antes del entreno y antes de cualquier cifra de este gate.
PREDICTION = {
    "written_utc": "2026-09-27",
    "written_before": "the mix directory, the smoke run and every figure "
                      "in this gate",
    "what_is_measured": (
        "the smoke of #T-ce-finetune with --budget 5000 over 5 000 rule "
        "episodes (data.rule_variety, batch 1) plus the verified pilot "
        "(artifacts/episodes-qwen/pilot-2k/verified.jsonl), against the "
        "SAME untuned control (--checkpoint none) on the same rows: "
        "development battery by family, and the pilot's own holdout by "
        "family"),
    "what_is_already_known": {
        "untuned control on dev": "forced 0.6225, counterfactual 0.343 "
                                  "(artifacts/gates/T-ce-finetune/"
                                  "eval-none.json)",
        "1 000 decisions moved the whole": "forced +0.055 [0.015, 0.098], "
                                          "counterfactual +0.121 "
                                          "[0.043, 0.200] (smoke.json)",
        "and did not move this family": "attribute_comparison 0.574 -> "
                                       "0.574 on the pilot's own holdout "
                                       "(#T-ce-finetune)",
    },
    "expect": {
        "priority_decision": "moves on dev and on the holdout: it already "
                             "moved in distribution in the pilot "
                             "(0.62 -> 0.78)",
        "attribute_comparison": "does NOT move on the pilot holdout. Five "
                                "times the volume of exactly this family, "
                                "with 15 attributes, mixed number formats "
                                "and ordinal questions, is still read by a "
                                "107 M encoder that has to compare two "
                                "numbers inside one sequence",
    },
    "go_if": (
        "attribute_comparison forced accuracy on the pilot holdout "
        "improves against the untuned control with a paired bootstrap "
        "CI95 whose lower bound is above 0 (B = 2000, seed fixed, the "
        "same rows scored twice)"),
    "no_go_if": (
        "it does not. Then the volume is not the bottleneck and the "
        "backbone is: it is written in this gate and handed to "
        "#T-backbone-ladder without waiting for the loop "
        "(docs/bucle-infinito.md §4, r = 3)"),
    "not_the_decision": (
        "the development battery by family and by language is published "
        "beside it, but dev is 400 rows shared with every other arm and "
        "this task's own question is the in-distribution holdout"),
    "bootstrap": {"B": 2000, "seed": 20260927, "unit": "paired row"},
}


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path) -> str:
    """Relativo al repo cuando lo es; absoluto cuando no (tests en /tmp)."""
    got = Path(path).resolve()
    try:
        return str(got.relative_to(ROOT))
    except ValueError:
        return str(got)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ------------------------------------------------------------- la mezcla
def is_rule(ep: dict) -> bool:
    return ep.get("generator_version") == RV.VARIETY_VERSION


def sample_by_group(episodes: list, n: int, seed: int) -> list:
    """`n` episodios del lote, POR GRUPO y equilibrados entre celdas.

    Cortar los `n` primeros del lote daría una sola celda: el plan sale
    ordenado por (familia, idioma), así que los 5 000 primeros de 50 000
    serían todos `attribute_comparison/en`. Se toman grupos enteros —un
    contrafactual separado de su base no es un contrafactual— en ronda
    entre las cuatro celdas, con el orden de grupos sorteado por semilla.
    """
    cells: dict = {}
    for ep in episodes:
        cells.setdefault(f"{ep['family']}/{ep['lang']}", {}) \
            .setdefault(ep["variant_group"], []).append(ep)
    order = []
    for name in sorted(cells):
        groups = sorted(cells[name])
        random.Random(f"{FORMAT}-{seed}-{name}").shuffle(groups)
        order.append((name, groups))
    out: list = []
    idx = 0
    while len(out) < n and any(idx < len(g) for _, g in order):
        for name, groups in order:
            if idx >= len(groups) or len(out) >= n:
                continue
            out.extend(cells[name][groups[idx]])
        idx += 1
    return out


def mix(rule_dir, pilot, n_rule: int, out, seed: int = 20260927) -> dict:
    """Escribe `<out>/episodes.jsonl` = muestra por regla + piloto entero."""
    rule_path = Path(rule_dir) / "episodes.jsonl"
    pilot_path = Path(pilot)
    rule_eps = [ep for ep in _read_jsonl(rule_path) if is_rule(ep)]
    pilot_eps = _read_jsonl(pilot_path)
    taken = sample_by_group(rule_eps, n_rule, seed)
    episodes = taken + pilot_eps
    verdict = EC.batch_validate(episodes)
    if verdict["n_invalid"] or verdict["duplicate_ids"]:
        raise ValueError(
            f"the mix is not publishable: {verdict['n_invalid']} invalid, "
            f"{len(verdict['duplicate_ids'])} duplicate ids "
            f"{verdict['duplicate_ids'][:4]}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ep_path = out / "episodes.jsonl"
    with open(ep_path, "w", encoding="utf-8") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep, ensure_ascii=False) + "\n")

    def census(eps: list) -> dict:
        got: dict = {}
        for ep in eps:
            got[f"{ep['family']}/{ep['lang']}"] = \
                got.get(f"{ep['family']}/{ep['lang']}", 0) + 1
        return dict(sorted(got.items()))

    manifest = {
        "task": TASK, "format": FORMAT, "generated_utc": utcnow(),
        "command": (f"python -m eval.numeric_value mix --rule {_rel(rule_dir)}"
                    f" --pilot {_rel(pilot_path)} --n-rule {n_rule} "
                    f"--out {_rel(out)}"),
        "sample_seed": seed,
        "sources": {
            "rule": {"dir": _rel(rule_dir), "file": _rel(rule_path),
                     "sha256": sha256_file(rule_path),
                     "n_available": len(rule_eps), "n_taken": len(taken),
                     "taken_by": "whole variant groups, round-robin over "
                                 "the four family/lang cells, group order "
                                 "seeded",
                     "by_cell": census(taken)},
            "pilot": {"file": _rel(pilot_path),
                      "sha256": sha256_file(pilot_path),
                      "n_taken": len(pilot_eps), "by_cell": census(pilot_eps)},
        },
        "n_episodes": len(episodes),
        "episodes_sha256": sha256_file(ep_path),
        "episodes_versioned": False,
        "by_cell": census(episodes),
        "groups": len({ep["variant_group"] for ep in episodes}),
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return manifest


# ------------------------------------------------- el holdout del piloto
def holdout_of(mix_dir, split_seed: int, share: float) -> dict:
    """El holdout del entreno, RECOMPUTADO — no hace falta reentrenar.

    `provisional_split` reparte por `variant_group` con una función pura
    de (grupo, semilla), así que el mismo 20 % sale otra vez desde el
    mismo directorio y la misma semilla.
    """
    episodes = _read_jsonl(Path(mix_dir) / "episodes.jsonl")
    cuts = (("train", 1.0 - share), ("holdout", share))
    parts = EC.split_by_group(episodes, split_seed, cuts)
    hold = parts["holdout"]
    return {
        "all": hold,
        "pilot": [ep for ep in hold if not is_rule(ep)],
        "rule": [ep for ep in hold if is_rule(ep)],
        "split": {"seed": split_seed, "cuts": [list(c) for c in cuts],
                  "by": "variant_group (data.episode_contract.assign_split)",
                  "n_train": len(parts["train"]), "n_holdout": len(hold)},
    }


def tracking_of(episodes: list, rows: list, source: str) -> dict:
    """El control de seguimiento, medido en ESTE corte.

    `eval.preflight_refs.tracking_control` lee los papeles del id de la
    batería de desarrollo (`orig`/`decisive`), que estos episodios no
    tienen. La propiedad es la misma y aquí se mide por el grupo: dos
    miembros del mismo `variant_group` cuyo gold NO coincide son un par
    de seguimiento, y la elección debería moverse con el gold.
    """
    by_id = {r["row_id"]: r for r in rows}
    per_group: dict = {}
    for ep in episodes:
        row = by_id.get(ep["id"])
        if row is not None:
            per_group.setdefault(ep["variant_group"], []).append(row)
    followed = same = 0
    for _group, members in sorted(per_group.items()):
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a["gold_index"] == b["gold_index"]:
                    continue
                moved = MS._forced(MS._row(a)) != MS._forced(MS._row(b))
                followed += bool(moved)
                same += (not moved)
    n = followed + same
    return {
        "n_pairs": n, "followed": followed, "same_slot": same,
        "rate": round(followed / n, 6) if n else None,
        "how": "pairs inside a variant_group whose rule gold differs; the "
               "pick has to move with it",
        "source": source,
    }


def temperature_of(rows: list, cut: str) -> dict:
    """Una temperatura ajustada en la mitad de los grupos y verificada en
    la otra — la misma receta que `eval.preflight_refs.dev_temperature`,
    con las etiquetas de ESTE corte."""
    fit = [r for r in rows if PR._half(r["variant_group"]) == "fit"]
    ver = [r for r in rows if PR._half(r["variant_group"]) == "verify"]
    if not fit or not ver:
        raise PR.ProtocolMismatch(
            f"{cut} did not split into two halves: a temperature fitted "
            "and verified on the same rows is a fit, not a calibration")
    logits = [[math.log(max(p, 1e-12)) for p in r["probs"][:r["k"]]]
              for r in fit]
    grid = C._fit_temp_grid(logits, [r["gold_index"] for r in fit])
    return {
        "temperature": grid["temperature"],
        "fitted_on": f"{cut} fit half ({len(fit)} rows, "
                     f"{len({r['variant_group'] for r in fit})} groups)",
        "verified_on": f"{cut} verify half ({len(ver)} rows, "
                       f"{len({r['variant_group'] for r in ver})} groups)",
        "strategy": "argmax over the K weights; no abstention threshold is "
                    "fitted here — that is #T-battery-calib",
        "threshold": None,
        "split_rule": "by variant_group, sha256 parity, so a group never "
                      "straddles the two halves",
        "nll_before": grid["nll_before"], "nll_after": grid["nll_after"],
        "not_the_products_calibration": "a temperature fitted inside this "
                                        "cut to make the report "
                                        "publishable; the product's is "
                                        "#T-battery-calib",
    }


def cut_descriptor(episodes: list, name: str, split: dict,
                   mix_manifest: dict) -> dict:
    payload = "\n".join(sorted(ep["id"] for ep in episodes))
    return {
        "name": name, "reserved": False, "split": "holdout",
        "task": TASK,
        "source": "the 20 % by variant_group of the smoke's own training "
                  "mix, recomputed from the mix directory and the split "
                  "seed — never dev, never the sealed cut",
        "mix": mix_manifest.get("command"),
        "mix_sha256": mix_manifest.get("episodes_sha256"),
        "split_seed": split["seed"],
        "rows_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "sealed_cut_read": False,
    }


def score_rows(scorer, episodes: list) -> list:
    rows, _trace = PR.nli_column(episodes, score=scorer.score_pairs)
    for row, ep in zip(rows, episodes):
        row["lang"] = ep["lang"]
        row["source"] = "rule" if is_rule(ep) else "pilot"
    return rows


def report_of(rows: list, episodes: list, name: str, split: dict,
              mix_manifest: dict, model_version: str, permutation: dict,
              notes: list) -> dict:
    """La suite sobre este corte. Ninguna cifra la calcula este módulo."""
    doc = MS.report(
        rows, cut=cut_descriptor(episodes, name, split, mix_manifest),
        model_version=model_version, task=TASK,
        calibration=temperature_of(rows, name),
        permutation=permutation,
        tracking=tracking_of(episodes, rows,
                             source=f"{TASK} on {name}"),
        notes=notes)
    return MS.require(doc)


def _by(rows_c: list, rows_t: list, key: str, seed: int, reps: int) -> dict:
    """Diferencia pareada por bucket, con el bootstrap de `ce_finetune`."""
    from training.python import ce_finetune as CF

    right_c = [int(MS._forced_right(MS._row(r))) for r in rows_c]
    right_t = [int(MS._forced_right(MS._row(r))) for r in rows_t]
    out = {}
    for name in sorted({str(r[key]) for r in rows_c}):
        idx = [i for i, r in enumerate(rows_c) if str(r[key]) == name]
        boot = CF._paired_bootstrap([right_c[i] for i in idx],
                                    [right_t[i] for i in idx], seed, reps)
        boot.update({
            "control": round(sum(right_c[i] for i in idx) / len(idx), 6),
            "tuned": round(sum(right_t[i] for i in idx) / len(idx), 6),
        })
        boot["moves"] = bool(boot["ci95"][0] is not None
                             and boot["ci95"][0] > 0)
        out[name] = boot
    return out


def measure(run, mix_dir, weights: str = "", device: str = "mps",
            split_seed: int = 0, share: float = 0.0,
            batch_size: int = 0) -> dict:
    """Control SIN AJUSTAR y checkpoint del smoke sobre el MISMO holdout."""
    from training.python import ce_finetune as CF

    run, mix_dir = Path(run), Path(mix_dir)
    train_manifest = json.loads(
        (run / "train_manifest.json").read_text(encoding="utf-8"))
    mix_manifest = json.loads(
        (mix_dir / "manifest.json").read_text(encoding="utf-8"))
    hyper = train_manifest["hyperparams"]
    split_seed = split_seed or train_manifest["split"]["seed"]
    share = share or dict(train_manifest["split"]["cuts"])["holdout"]
    weights = weights or train_manifest["weights"]
    batch_size = batch_size or CE.DEFAULT_BATCH_SIZE
    hold = holdout_of(mix_dir, split_seed, share)
    if not hold["pilot"]:
        raise ValueError("the recomputed holdout has no pilot episode: the "
                         "split seed or the mix does not match the run")

    columns = {}
    for name, init in (("control", CF.CONTROL),
                       ("tuned", str(run / "model"))):
        tokenizer, model, entail, source = CF.load_model(init, weights, device)
        version = f"{TASK}:{name}"
        scorer = CF.LoadedPairScorer(tokenizer, model, entail, device,
                                     version, batch_size=batch_size)
        perm = {"measured": True,
                **CE.check_permutation_invariance(
                    CE.CrossEncoderScorer(scorer), CE.CONTRACT_CASE),
                "source": f"{TASK} — model.ce_scorer."
                          "check_permutation_invariance on the contract "
                          f"case, {name} weights, same run"}
        rows = score_rows(scorer, hold["all"])
        columns[name] = {
            "init": source, "model_version": version, "rows": rows,
            "report": report_of(
                rows, hold["all"], CUT_NAME, hold["split"], mix_manifest,
                version, perm,
                notes=[f"{TASK}: {name} column on the pilot's own holdout, "
                       "the cut #T-ce-finetune left at 0.574 -> 0.574",
                       f"init: {source}",
                       "the sealed cut was not read"]),
        }
        del model, tokenizer

    c_rows, t_rows = columns["control"]["rows"], columns["tuned"]["rows"]
    if [r["row_id"] for r in c_rows] != [r["row_id"] for r in t_rows]:
        raise PR.ProtocolMismatch("the two columns are not the same rows")
    seed = PREDICTION["bootstrap"]["seed"]
    reps = PREDICTION["bootstrap"]["B"]
    pilot_ids = {ep["id"] for ep in hold["pilot"]}
    c_pilot = [r for r in c_rows if r["row_id"] in pilot_ids]
    t_pilot = [r for r in t_rows if r["row_id"] in pilot_ids]
    doc = {
        "task": TASK, "format": FORMAT, "generated_utc": utcnow(),
        "run": _rel(run), "mix": _rel(mix_dir), "device": device,
        "weights": weights,
        "budget_decisions": hyper["budget_decisions"],
        "decisions_consumed": train_manifest["decisions_consumed"],
        "length_report": train_manifest["length_report"],
        "holdout": {
            "n": len(hold["all"]), "n_pilot": len(hold["pilot"]),
            "n_rule": len(hold["rule"]), **hold["split"]},
        "columns": {name: {k: v for k, v in col.items() if k != "rows"}
                    for name, col in columns.items()},
        "whole_holdout": {
            "by_family": _by(c_rows, t_rows, "family", seed, reps),
            "by_source": _by(c_rows, t_rows, "source", seed + 2, reps)},
        "pilot_holdout": {
            "n": len(c_pilot),
            "by_family": _by(c_pilot, t_pilot, "family", seed + 1, reps),
            "by_lang": _by(c_pilot, t_pilot, "lang", seed + 3, reps)},
    }
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    path = GATE_DIR / "holdout.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n")
    print(f"[value] wrote {_rel(path)}")
    for fam, node in doc["pilot_holdout"]["by_family"].items():
        print(f"[value] pilot holdout {fam}: {node['control']:.4f} -> "
              f"{node['tuned']:.4f} delta {node['point']:+.4f} "
              f"ci{node['ci95']} (n={node['n']})")
    return doc


# ---------------------------------------------------------------- el gate
def predict() -> dict:
    """Escribe la predicción, y sólo si no había una."""
    gate = RV.read_gate()
    if isinstance(gate.get("value_proof_prediction"), dict):
        print("[value] the prediction was already written; not touched")
        return gate
    return RV.write_gate("value_proof_prediction",
                         {**PREDICTION, "pass": None,
                          "status": "written-before-measuring"})


def gate(run, smoke=None, holdout=None) -> dict:
    """El veredicto por la regla predeclarada: decide el holdout."""
    smoke = Path(smoke or GATE_DIR / "smoke.json")
    holdout = Path(holdout or GATE_DIR / "holdout.json")
    dev = json.loads(smoke.read_text(encoding="utf-8"))
    hold = json.loads(holdout.read_text(encoding="utf-8"))
    target = hold["pilot_holdout"]["by_family"].get(TARGET_FAMILY)
    if target is None:
        raise ValueError(f"the pilot holdout has no {TARGET_FAMILY} rows")
    moves = bool(target["moves"])
    payload = {
        "measured_utc": utcnow(),
        "prediction": "value_proof_prediction (this gate, written first)",
        "rule": {"go_if": PREDICTION["go_if"],
                 "no_go_if": PREDICTION["no_go_if"]},
        "run": hold["run"], "mix": hold["mix"],
        "budget_decisions": hold["budget_decisions"],
        "decisions_consumed": hold["decisions_consumed"],
        "dev": {
            "source": _rel(smoke),
            "verdict": dev.get("comparison", dev).get("verdict"),
            "forced": dev.get("comparison", dev).get("forced"),
            "counterfactual_joint": dev.get("comparison",
                                            dev).get("counterfactual_joint"),
            "by_family": dev.get("comparison", dev).get("by_family"),
            "by_lang": dev.get("comparison", dev).get("by_lang"),
        },
        "pilot_holdout": {
            "source": _rel(holdout),
            "n": hold["pilot_holdout"]["n"],
            "by_family": hold["pilot_holdout"]["by_family"],
            "by_lang": hold["pilot_holdout"]["by_lang"],
        },
        "whole_holdout": hold["whole_holdout"],
        "target_family": TARGET_FAMILY,
        "target_moves": moves,
        "power": {
            "n_target_rows": target["n"],
            "what_it_can_see": "a paired CI95 clear of 0 on n rows needs "
                               "roughly a 2/sqrt(n) shift, so this many "
                               "in-distribution rows of one family cannot "
                               "resolve a small move",
            "higher_powered_reading": hold["whole_holdout"]["by_family"].get(
                TARGET_FAMILY),
            "why_it_does_not_decide": "the whole holdout mixes the rule "
                                      "batch's own rows with the pilot's; "
                                      "the pre-registered rule names the "
                                      "PILOT holdout, and it is not "
                                      "rewritten after seeing a figure",
        },
        "verdict": "GO" if moves else "NO-GO",
        "pass": moves,
    }
    if not moves:
        wide = hold["whole_holdout"]["by_family"].get(TARGET_FAMILY) or {}
        payload["bottleneck"] = {
            "is": "backbone",
            "why": (f"{TARGET_FAMILY} did not move on its OWN holdout with "
                    f"{hold['decisions_consumed']} decisions: "
                    f"{target['n']} in-distribution pilot rows went "
                    f"{target['control']} -> {target['tuned']} (paired delta "
                    f"{target['point']} ci{target['ci95']}), and the whole "
                    f"holdout's {wide.get('n')} rows of the family — eight "
                    f"times the power — went {wide.get('control')} -> "
                    f"{wide.get('tuned')} (delta {wide.get('point')} "
                    f"ci{wide.get('ci95')}), still not clear of 0. Five "
                    "times the volume of exactly this family is not what is "
                    "missing"),
            "hand_to": "#T-backbone-ladder",
            "not_touched_here": "this task does not change the backbone; "
                                "it records the measurement and hands it "
                                "over (docs/bucle-infinito.md §4, r = 3)",
        }
    return RV.write_gate("value_proof", payload)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.numeric_value")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("predict", help="write the prediction before measuring")
    m = sub.add_parser("mix", help="build the smoke's training mix")
    m.add_argument("--rule", required=True)
    m.add_argument("--pilot", required=True)
    m.add_argument("--n-rule", type=int, default=5000)
    m.add_argument("--out", required=True)
    m.add_argument("--seed", type=int, default=20260927)
    v = sub.add_parser("measure", help="score the pilot holdout twice")
    v.add_argument("--run", required=True)
    v.add_argument("--mix", required=True)
    v.add_argument("--device", default="mps")
    v.add_argument("--weights", default="")
    v.add_argument("--batch-size", type=int, default=0)
    g = sub.add_parser("gate", help="sign the value-proof verdict")
    g.add_argument("--run", required=True)
    g.add_argument("--smoke", default=None)
    g.add_argument("--holdout", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "predict":
        predict()
        return 0
    if args.cmd == "mix":
        manifest = mix(args.rule, args.pilot, args.n_rule, args.out,
                       seed=args.seed)
        print(f"mix n={manifest['n_episodes']} "
              f"rule={manifest['sources']['rule']['n_taken']} "
              f"pilot={manifest['sources']['pilot']['n_taken']} "
              f"-> {_rel(args.out)}")
        print(json.dumps(manifest["by_cell"], ensure_ascii=False))
        return 0
    if args.cmd == "measure":
        measure(args.run, args.mix, weights=args.weights,
                device=args.device, batch_size=args.batch_size)
        return 0
    got = gate(args.run, smoke=args.smoke, holdout=args.holdout)
    print(json.dumps({"verdict": got["value_proof"]["verdict"],
                      "target_moves": got["value_proof"]["target_moves"]}))
    return 0 if got["value_proof"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
