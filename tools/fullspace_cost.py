"""Coste de la cabeza a CIENTOS, y la ablación de scoring independiente.

`#T-bigk-optsets` ya publicó el coste a K=8/41/77 (`artifacts/gates/
T-bigk-optsets/cost.json`) y esta task **lo reutiliza en vez de repetirlo**.
Lo que falta para decidir por escrito si el objetivo conserva
`model/decision_head.py` o lo cambia son dos cosas:

1. **K en cientos** con la cabeza actual — el régimen que los negativos
   in-batch pedirían si los batches fueran mixtos;
2. la **ablación de scoring independiente** (`set_attention=False`): la
   misma cabeza sin atención entre candidatos, que es la alternativa que la
   task obliga a etiquetar como CAMBIO DE ARQUITECTURA y no como «la misma
   cabeza con otro denominador».

Cada medida corre en su PROPIO subproceso, y por dos razones medidas: el
pool de MPS no se devuelve al sistema dentro de un proceso
(`torch.mps.driver_allocated_memory()` es una marca de agua alta), así que
medir varios K seguidos publica la memoria del mayor en todos; y la primera
medida después de un K grande sale lenta por presión de memoria — en la
primera corrida de este arnés, K=8 salió a 258 filas/s y K=41 a 449, que es
imposible y era el orden, no la cardinalidad. Los K se recorren emparejados
(cabeza actual y ablación seguidas, mismo K), así que la deriva de la
máquina golpea a los dos lados del par.

El arnés del paso es el de `tools.bigk_cost` (mismo paso completo, misma
mediana de 8 pasos tras 2 de calentamiento, mismo B=64).

Honestidad de los textos: el arnés sólo necesita K cadenas distintas — el
coste depende de K, de la longitud y de B, no de qué diga cada etiqueta.
Los espacios entrenables juntos dan 175 textos distintos, así que por
encima de eso la lista se cicla con un sufijo numérico para que cada
columna siga siendo una clave de caché distinta. Es un arnés de coste: no
se escribe checkpoint y no se lee ningún corpus vallado.

CLI:
    PYTHONPATH=. .venv-train/bin/python -m tools.fullspace_cost --device mps
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from data.optset import PREFETCH_DIR, label_pool  # noqa: E402
from model.decision_head import (DecisionEngine,  # noqa: E402
                                 PointerDecisionHead)
from model.encoder import load_backbone, pick_device  # noqa: E402
from tools.bigk_cost import (BATCH, D_MODEL, MAX_LENGTH,  # noqa: E402
                             N_HEADS, N_LAYERS, STEPS, WARMUP, batch_at,
                             measure, states)
from training.python.train_decision import peak_memory_gb  # noqa: E402

TASK = "T-fullspace-objective"
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", TASK)
BIGK_COST = os.path.join(ROOT, "artifacts", "gates", "T-bigk-optsets",
                         "cost.json")
BACKBONE = "ettin-68m"

#: todos los espacios entrenables de `decision-mix-clean-1m`
TEXT_DATASETS = ("massive", "huffpost", "goemotions", "dbpedia14",
                 "prog-gold", "synth-v1", "detox-attack", "snli", "swag",
                 "email-triage", "boolq", "civil-comments")
#: los K nuevos de la cabeza actual — 8/41/77 se reutilizan de bigk
NEW_KS = (154, 308)
#: la ablación se mide en todo el barrido: es un par, no una cifra suelta
ABLATION_KS = (8, 41, 77, 154, 308)


def option_texts(n: int, root: str = PREFETCH_DIR) -> tuple:
    """`n` cadenas distintas, las reales primero y luego cicladas."""
    real, seen = [], set()
    for dataset in TEXT_DATASETS:
        try:
            pool = label_pool(dataset, root)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        for opt in pool:
            text = opt.get("text") or opt["id"]
            if text not in seen:
                seen.add(text)
                real.append(text)
    out = list(real)
    i = 0
    while len(out) < n:
        out.append(f"{real[i % len(real)]} {i // len(real) + 2}")
        i += 1
    return out, len(real)


def bench(dev, backbone, k: int, set_attention: bool, rows, texts) -> dict:
    torch.manual_seed(1789)
    head = PointerDecisionHead(backbone.hidden_size, D_MODEL, N_LAYERS,
                               N_HEADS, set_attention=set_attention)
    engine = DecisionEngine(backbone=backbone, head=head)
    engine.head.train()
    engine.clear_caches()
    optim = torch.optim.AdamW(engine.head.parameters(), lr=3e-4,
                              weight_decay=0.01)
    batch = batch_at(k, rows, texts)
    for _ in range(WARMUP):
        measure(engine, batch, optim, dev)
    marks = [measure(engine, batch, optim, dev) for _ in range(STEPS)]
    step_s = sorted(m["step_s"] for m in marks)[len(marks) // 2]
    head_s = sorted(m["head_s"] for m in marks)[len(marks) // 2]
    out = {
        "k": k, "batch_size": BATCH, "set_attention": set_attention,
        "rows_per_s": round(BATCH / step_s, 2),
        "ms_per_step": round(1000 * step_s, 1),
        "ms_per_step_head_only": round(1000 * head_s, 1),
        "head_share_of_step": round(head_s / step_s, 4),
        "peak_mem_gb": peak_memory_gb(dev),
        "minutes_per_62500_rows": round(62_500 * step_s / BATCH / 60, 2),
        "minutes_per_1m_rows": round(1_000_000 * step_s / BATCH / 60, 1),
        "head_params": engine.head.n_params(),
        "steps_timed": STEPS, "warmup_steps": WARMUP,
    }
    del engine, head, optim
    return out


def one(device: str, k: int, set_attention: bool) -> dict:
    """UNA medida, en este proceso: lo que el subproceso ejecuta."""
    dev = pick_device(device)
    backbone = load_backbone(BACKBONE, dev)
    texts, _n_real = option_texts(k)
    return bench(dev, backbone, k, set_attention, states(BATCH), texts)


def in_subprocess(device: str, k: int, set_attention: bool) -> dict:
    """La misma medida, aislada — pool de MPS limpio y sin orden previo."""
    cmd = [sys.executable, "-m", "tools.fullspace_cost", "--one", str(k),
           "--device", device]
    if not set_attention:
        cmd.append("--independent")
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                         text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def noise_check(per_k: dict) -> dict:
    """¿La tabla se puede leer como una curva, o sólo como pares?

    Estas medidas comparten la máquina con lo que haya corriendo (la nota
    de honestidad de `#T-bigk-optsets` ya dijo que un job paralelo cuesta
    un tercio del throughput). Un `rows_per_s` que SUBE al subir K es
    imposible por construcción y delata esa contaminación, así que se
    detecta aquí y se publica en el artefacto en vez de dejar que el
    lector tome la tabla por una curva limpia.
    """
    ks = sorted(per_k, key=int)
    inversions = [
        {"from_k": int(a), "to_k": int(b),
         "rows_per_s": [per_k[a]["rows_per_s"], per_k[b]["rows_per_s"]]}
        for a, b in zip(ks, ks[1:])
        if per_k[b]["rows_per_s"] > per_k[a]["rows_per_s"]]
    return {
        "monotone_in_k": not inversions,
        "inversions": inversions,
        "reading": ("rows/s can only fall as K grows; an inversion is "
                    "machine noise, not cardinality. Where there are "
                    "inversions, read the same-K PAIRS (measured back to "
                    "back) and not the curve" if inversions else
                    "no inversion: the table reads as a curve"),
    }


def compose(reused: dict, current: dict, ablation: dict, failures: dict,
            dev, n_real: int) -> dict:
    merged = dict(reused["per_k"])
    merged.update(current)
    speedup = {k: round(ablation[k]["rows_per_s"] / merged[k]["rows_per_s"], 2)
               for k in ablation if k in merged}
    head_ratio = {
        k: round(ablation[k]["ms_per_step_head_only"]
                 / merged[k]["ms_per_step_head_only"], 2)
        for k in ablation if k in merged}
    return {
        "format": "jev.gate.v1",
        "task": TASK,
        "artifact": "cost",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(dev),
        "architecture": {"backbone": BACKBONE, "d_model": D_MODEL,
                         "n_layers": N_LAYERS, "n_heads": N_HEADS,
                         "frozen_backbone": True, "batch_size": BATCH,
                         "max_length": MAX_LENGTH},
        "what_is_measured": (
            "one full training step through `tools.bigk_cost.measure`: "
            "state encode (no grad), option text embeddings through the "
            "cache, `forward_batch`, `cross_entropy`, backward, clip, "
            f"optimiser step. Median of {STEPS} steps after {WARMUP} warmups"),
        "current_head_set_attention": {
            "reused_from": {
                "artifact": "artifacts/gates/T-bigk-optsets/cost.json",
                "generated_utc": reused["generated_utc"],
                "ks": sorted(reused["per_k"], key=int),
                "why": "R5/economy — already measured on this machine with "
                       "this same harness; it is reused, not repeated",
            },
            "measured_here": sorted(current, key=int),
            "per_k": merged,
            "noise_check": noise_check(merged),
        },
        "independent_scoring_ablation": {
            "what": "the same head with `set_attention=False`: no attention "
                    "among candidates, so a logit depends only on (state, "
                    "question, this option's text)",
            "label": "CHANGE OF ARCHITECTURE — an arm that adopts this is "
                     "not `the same head with another denominator`, and any "
                     "verdict of the current head does not transfer to it "
                     "(R4, R9)",
            "per_k": ablation,
            "noise_check": noise_check(ablation),
            "rows_per_s_speedup_vs_set_attention": speedup,
            "head_ms_ratio_vs_set_attention": head_ratio,
            "how_to_read": (
                "only the SAME-K pair is a comparison: the two sides were "
                "measured back to back, so machine drift hits both. The "
                "column down the K axis is not a clean curve wherever "
                "`noise_check` reports an inversion"),
        },
        "failures": failures,
        "honesty": [
            f"the harness needs K distinct strings; {n_real} real label "
            "texts exist across the trainable spaces, and above that the "
            "list is cycled with a numeric suffix so every column stays a "
            "distinct cache key. Cost depends on K, on token length and on "
            "B, not on what each label says",
            "no checkpoint is written, no fenced corpus is read, and the "
            "weights are thrown away",
            "every measurement runs in its OWN subprocess: the MPS pool is "
            "a high-water mark inside a process, so measuring several K in "
            "one run publishes the largest K's memory for all of them, and "
            "the first measurement after a big K comes out slow from memory "
            "pressure. The K values are walked in pairs (current head and "
            "ablation back to back at the same K) so machine drift hits "
            "both sides of the pair",
            "this Mac was NOT idle: another agent was working in parallel "
            "while these ran. `#T-bigk-optsets` already measured that a "
            "parallel job costs about a third of the throughput, and "
            "`noise_check` reports the inversions that proves it here. The "
            "same-K pairs survive that; the curve down the K axis does not",
            "the reused K=8/41/77 rows for the current head were measured "
            "on a different day; the pair inside a single K is the "
            "comparison that does not depend on the machine's state",
        ],
    }


def run(device: str = "auto", write: bool = True, log=print) -> dict:
    dev = pick_device(device)
    _texts, n_real = option_texts(max(NEW_KS + ABLATION_KS))
    log(f"[cost] device={dev} real label texts={n_real}")

    with open(BIGK_COST, encoding="utf-8") as fh:
        reused = json.load(fh)

    current, ablation, failures = {}, {}, {}
    for k in sorted(set(NEW_KS) | set(ABLATION_KS)):
        for want, store in ((True, current), (False, ablation)):
            if want and k not in NEW_KS:
                continue
            if not want and k not in ABLATION_KS:
                continue
            tag = "set_attn  " if want else "independent"
            try:
                store[str(k)] = in_subprocess(device, k, want)
            except (subprocess.CalledProcessError, ValueError) as exc:
                detail = getattr(exc, "stderr", "") or str(exc)
                failures[f"{tag.strip()}/K={k}"] = detail[-300:]
                log(f"[cost] {tag} K={k}: FAILED")
                continue
            log(f"[cost] {tag} K={k}: {store[str(k)]['rows_per_s']} rows/s, "
                f"head {store[str(k)]['ms_per_step_head_only']} ms, peak "
                f"{store[str(k)]['peak_mem_gb']} GiB")

    doc = compose(reused, current, ablation, failures, dev, n_real)
    if write:
        os.makedirs(GATE_DIR, exist_ok=True)
        with open(os.path.join(GATE_DIR, "cost.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
    return doc


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="tools.fullspace_cost")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--one", type=int, help="measure ONE K and print it")
    ap.add_argument("--recompose", action="store_true",
                    help="rebuild the artifact from the measurements it "
                         "already holds — no re-measuring")
    ap.add_argument("--independent", action="store_true",
                    help="with --one: the set_attention=False ablation")
    args = ap.parse_args(argv[1:])
    if args.one:
        print(json.dumps(one(args.device, args.one, not args.independent)))
        return 0
    if args.recompose:
        path = os.path.join(GATE_DIR, "cost.json")
        with open(path, encoding="utf-8") as fh:
            old = json.load(fh)
        with open(BIGK_COST, encoding="utf-8") as fh:
            reused = json.load(fh)
        cur = old["current_head_set_attention"]
        doc = compose(reused,
                      {k: cur["per_k"][k] for k in cur["measured_here"]},
                      old["independent_scoring_ablation"]["per_k"],
                      old.get("failures", {}), old["device"],
                      option_texts(1)[1])
        doc["generated_utc"] = old["generated_utc"]
        doc["recomposed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime())
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        print(json.dumps({
            "set_attn_noise":
                doc["current_head_set_attention"]["noise_check"],
            "ablation_noise":
                doc["independent_scoring_ablation"]["noise_check"],
            "speedup": doc["independent_scoring_ablation"]
            ["rows_per_s_speedup_vs_set_attention"]},
            indent=2, sort_keys=True))
        return 0
    doc = run(args.device, write=not args.no_write)
    print(json.dumps({"failures": doc["failures"],
                      "speedup": doc["independent_scoring_ablation"]
                      ["rows_per_s_speedup_vs_set_attention"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
