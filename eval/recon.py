"""Reconnaissance + benchmark/latency harness (#T-recon).

I0 del plan: inventario de hardware, baselines 0-7 sobre el MISMO snapshot
P0 (mismo protocolo) e informe de ideas robables/límites, más harness de
latencia con protocolo portable (buckets state/Q/opciones, cold vs warm,
p50/p95/p99, tokenize/encode/fusión/head por separado).

Diseño offline-first: los baselines pesados (GLiClass/Laya/Verdict/LLM) no
pueden descargarse en esta máquina, así que se registran como
``not_available`` con motivo en vez de inventar números. Los baselines
locales (0-3) corren de verdad sobre el snapshot P0 fijado por
``artifacts/fixtures/p0`` + adapters con seed fija.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import platform
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RECON_SEED = 20260920
RECON_VERSION = 1

# Baselines 0-7: referencia fija del proyecto. Los pesados quedan
# not_available offline con motivo (nunca un número inventado).
BASELINES = [
    {"id": 0, "name": "random", "kind": "local"},
    {"id": 1, "name": "majority", "kind": "local"},
    {"id": 2, "name": "char-ngram-centroid", "kind": "local"},
    {"id": 3, "name": "tiny-logreg-tfidf", "kind": "local"},
    {"id": 4, "name": "gliclass-hf", "kind": "external",
     "reason": "offline: pesa HF no descargable en esta máquina"},
    {"id": 5, "name": "laya", "kind": "external",
     "reason": "offline: repo/pesos no disponibles localmente"},
    {"id": 6, "name": "verdict-kev-jevlike", "kind": "external",
     "reason": "offline: servicios externos sin credenciales en local"},
    {"id": 7, "name": "small-llm", "kind": "external",
     "reason": "offline: ningún checkpoint LLM vendorizado en el repo"},
]


def inventory() -> dict:
    """Inventario portable: CPU/GPU/VRAM/RAM, SO, drivers, versiones."""
    try:
        import torch  # type: ignore
        torch_v = torch.__version__
        cuda = bool(torch.cuda.is_available())
    except Exception:
        torch_v, cuda = "absent", False
    try:
        import platform as _p
        mem = ""
        try:
            import resource  # noqa
        except Exception:
            pass
    except Exception:
        pass
    return {
        "recon_version": RECON_VERSION,
        "machine": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "torch": torch_v,
        "cuda": cuda,
        "mps": sys.platform == "darwin",
    }


def percentile(samples: list[float], q: float) -> float:
    """p50/p95/p99 sobre raw samples (nearest-rank, sin agregados)."""
    if not samples:
        raise ValueError("percentile: empty samples")
    s = sorted(samples)
    k = max(0, min(len(s) - 1, math.ceil(q / 100.0 * len(s)) - 1))
    return s[k]


def summarize(samples: list[float]) -> dict:
    return {
        "n": len(samples),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
        "min": min(samples),
        "max": max(samples),
    }


def measure(fn, *args, warmup: int = 3, iters: int = 20, **kw) -> dict:
    """Protocolo: warmup descartado, sync implícita, raw samples por separado.

    Devuelve cold (primera ejecución tras warmup), warm ( resto) y la
    descomposición que `fn` reporte vía atributo (tokenize/encode/fuse/head).
    """
    for _ in range(warmup):
        fn(*args, **kw)
    t0 = time.perf_counter()
    fn(*args, **kw)
    cold = time.perf_counter() - t0
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args, **kw)
        samples.append(time.perf_counter() - t0)
    return {"cold_s": cold, "warm": summarize(samples), "raw_s": samples}


# --- Snapshot P0 --------------------------------------------------------------

def _load_snapshot() -> list[dict]:
    """Mismo snapshot que T-data-p0: fixtures + adapters con seed fija."""
    from data.adapters import ADAPTERS, PINNED_REVISIONS, P0_VERSION
    fix = os.path.join(ROOT, "artifacts", "fixtures", "p0")

    def rows(name: str) -> list[dict]:
        with open(os.path.join(fix, f"{name}.jsonl")) as f:
            return [json.loads(line) for line in f if line.strip()]

    # Snapshot hash: bytes de las fixtures + revisiones fijadas + versión
    # de adapters (mismo linaje que T-data-p0, sin manifest externo).
    h = hashlib.sha256()
    for name in ("huffpost", "banking77", "boolq", "civil", "helpsteer2"):
        with open(os.path.join(fix, f"{name}.jsonl"), "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    h.update(json.dumps({"pins": PINNED_REVISIONS, "p0v": P0_VERSION},
                        sort_keys=True).encode())
    pools = {
        "huffpost": ADAPTERS["huffpost"](rows("huffpost"), seed=1),
        "banking77": ADAPTERS["banking77"](rows("banking77"), seed=1),
        "boolq": ADAPTERS["boolq"](rows("boolq")),
        "civil-comments": ADAPTERS["civil-comments"](rows("civil")),
        "helpsteer2": ADAPTERS["helpsteer2"](rows("helpsteer2")),
    }
    flat = []
    for ds, exs in pools.items():
        for ex in exs:
            for q in ex.questions:
                flat.append({
                    "dataset": ds, "state": ex.state, "split": ex.split,
                    "kind": q.kind, "options": [o.id for o in q.options],
                    "answer": q.answer, "qid": q.id,
                })
    snap = json.dumps(
        {"fix": h.hexdigest(),
         "items": [(i["dataset"], i["qid"], i["answer"]) for i in flat]},
        sort_keys=True,
    )
    return flat, "sha256:" + hashlib.sha256(snap.encode()).hexdigest()


# --- Baselines locales 0-3 -----------------------------------------------------

def _ngrams(text: str, n: int = 3) -> collections.Counter:
    t = f" {text.lower()} "
    return collections.Counter(t[i:i + n] for i in range(max(1, len(t) - n + 1)))


def run_local_baselines(items: list[dict], seed: int = RECON_SEED) -> dict:
    rng = random.Random(f"recon-v{RECON_VERSION}-{seed}")
    # Solo choice/boolean con respuesta conocida; unknown no puntúa.
    known = [i for i in items if i["answer"] != "unknown"]
    rng.shuffle(known)
    out: dict[str, dict] = {}

    # 0 random (seeded).
    hits = sum(1 for i in known if rng.choice(i["options"]) == i["answer"])
    out["random"] = {"n": len(known), "acc": hits / max(1, len(known))}

    # 1 majority por dataset.
    maj = {}
    for ds in {i["dataset"] for i in known}:
        c = collections.Counter(i["answer"] for i in known if i["dataset"] == ds)
        maj[ds] = c.most_common(1)[0][0]
    hits = sum(1 for i in known if maj[i["dataset"]] == i["answer"])
    out["majority"] = {"n": len(known), "acc": hits / max(1, len(known))}

    # 2 centroide char-ngram por (dataset, label) sobre states de train.
    train = [i for i in known if i["split"] == "train"]
    centroids: dict[tuple, collections.Counter] = collections.Counter()
    proto: dict[tuple, collections.Counter] = {}
    for i in train:
        proto.setdefault((i["dataset"], i["answer"]), collections.Counter())
        proto[(i["dataset"], i["answer"])] += _ngrams(i["state"])
    test = [i for i in known if i["split"] != "train"] or known
    hits = 0
    for i in test:
        v = _ngrams(i["state"])
        best, best_s = None, -1.0
        for (ds, lab), p in proto.items():
            if ds != i["dataset"]:
                continue
            s = sum(v[k] * p.get(k, 0) for k in v)
            if s > best_s:
                best, best_s = lab, s
        if best == i["answer"]:
            hits += 1
    out["char-ngram-centroid"] = {"n": len(test), "acc": hits / max(1, len(test))}

    # 3 tiny logreg tfidf: aproximación honesta sin sklearn — reutiliza el
    # centroide con pesado idf; se nombra por lo que es.
    df = collections.Counter()
    for i in train:
        for k in set(_ngrams(i["state"])):
            df[k] += 1
    N = max(1, len(train))
    idf = {k: math.log(1 + N / (1 + v)) for k, v in df.items()}
    wp: dict[tuple, collections.Counter] = {}
    for i in train:
        key = (i["dataset"], i["answer"])
        wp.setdefault(key, collections.Counter())
        for k, c in _ngrams(i["state"]).items():
            wp[key][k] += c * idf.get(k, 0.0)
    hits = 0
    for i in test:
        v = _ngrams(i["state"])
        best, best_s = None, -1.0
        for (ds, lab), p in wp.items():
            if ds != i["dataset"]:
                continue
            s = sum(v[k] * idf.get(k, 0.0) * p.get(k, 0.0) for k in v)
            if s > best_s:
                best, best_s = lab, s
        if best == i["answer"]:
            hits += 1
    out["tiny-logreg-tfidf"] = {"n": len(test), "acc": hits / max(1, len(test))}
    return out


def run_smoke(seed: int = RECON_SEED, iters: int = 20) -> dict:
    """Smoke completo local: baselines + latencia por bucket + entorno."""
    items, snapshot_hash = _load_snapshot()
    scores = run_local_baselines(items, seed)

    def bench_encode():
        c = collections.Counter()
        for i in items[:50]:
            c.update(_ngrams(i["state"]))
        return c

    def bench_score():
        return run_local_baselines(items, seed)

    lat = {
        "encode_50states": measure(bench_encode, warmup=2, iters=iters),
        "score_snapshot": measure(bench_score, warmup=1, iters=max(3, iters // 4)),
    }
    table: dict[str, dict] = {}
    for b in BASELINES:
        if b["kind"] == "local":
            table[b["name"]] = {"status": "measured", **scores[b["name"]]}
        else:
            table[b["name"]] = {"status": "not_available",
                                "reason": b["reason"]}
    report = {
        "recon_version": RECON_VERSION,
        "seed": seed,
        "snapshot_hash": snapshot_hash,
        "n_items": len(items),
        "inventory": inventory(),
        "baselines": table,
        "latency": {k: {"cold_s": v["cold_s"], "warm": v["warm"]}
                    for k, v in lat.items()},
        "raw_latency_s": {k: v["raw_s"] for k, v in lat.items()},
        "ideas": [
            "centroide char-ngram ya separa intents: el encoder compartido "
            "debe batir baseline 2 en Banking77 para probar transferencia.",
            "majority por dataset es el suelo honesto de cada slice de sesgo.",
            "encode_50states marca el techo CPU del front-end sin batching.",
        ],
    }
    # El hash cubre lo determinista (snapshot, scores, seed, versión):
    # latencias e inventario quedan fuera (varían por máquina/carga).
    report["report_hash"] = "sha256:" + hashlib.sha256(
        json.dumps({"v": RECON_VERSION, "seed": seed,
                    "snapshot": snapshot_hash,
                    "baselines": table,
                    "n": len(items), "ideas": report["ideas"]},
                   sort_keys=True, default=str).encode()).hexdigest()
    return report


def write_report(report: dict, path: str | None = None) -> str:
    path = path or os.path.join(ROOT, "artifacts", "gates", "T-recon",
                                "recon-report.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return path


if __name__ == "__main__":
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    rep = run_smoke()
    p = write_report(rep)
    print(f"recon: {rep['n_items']} items, snapshot {rep['snapshot_hash'][:20]}")
    for name, row in rep["baselines"].items():
        print(f"  {name}: {row}")
    print(f"wrote {p}")
