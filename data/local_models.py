"""Host Qwen tags: profesor (razonamiento) vs volumen (datasets pequenos).

Este clon aterrizo en un MacBook Pro M5 Pro (48 GB, Metal). El roadmap
nombra `qwen3.8:27b-mlx` como profesor medido. En esta maquina Ollama
puede no tener esa etiqueta: a 2026-09-27 solo habia
`qwen3.6:27b-q8_0` con system prompt de otro proyecto (Atlas / QuantLab).
Llama no esta instalado. El resolver elige lo que hay, no inventa un tag.

Roles:
- profesor / juez / verificador: 27B si existe (calidad de eleccion).
- volumen / parafrasis / datasets chicos: 8B o 4B si existen (latencia).

`QWEN_MODEL` y `QWEN_VOLUME_MODEL` ganan siempre. El probe
`python3 -m data.local_models probe` escribe tiempos en
`artifacts/gates/T-host-qwen-ladder/speed.json`.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Iterable

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TASK = "T-host-qwen-ladder"
GATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts", "gates", TASK)

#: Orden de preferencia. El primero instalado gana.
TEACHER_PREF = (
    "qwen3.8:27b-mlx",
    "qwen3.8:27b",
    "qwen3.8:27b-q4_K_M",
    "qwen3.6:27b-mlx",
    "qwen3.6:27b-q8_0",
)
VOLUME_PREF = (
    "qwen3:8b",
    "qwen3:8b-q4_K_M",
    "qwen3:4b",
    "qwen3:4b-q4_K_M",
)

#: Neutraliza personas colgadas en un Modelfile ajeno (Atlas, etc.).
TEACHER_SYSTEM = (
    "You are a local decision teacher. Follow the user instruction "
    "exactly. Reply with the requested format only. No persona, no "
    "trading talk, no tool chatter."
)

ATLAS_MARKERS = ("atlas-qwen", "You are Atlas", "QuantLab", "Wall-first")


def list_tags(timeout: int = 5) -> list[str]:
    """Nombres que Ollama reporta en /api/tags. Lista vacia si no hay servidor."""
    try:
        req = urllib.request.Request(f"{OLLAMA}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            doc = json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []
    names = []
    for m in doc.get("models") or []:
        n = m.get("name") or m.get("model")
        if n:
            names.append(n)
    return names


def _first_installed(order: Iterable[str], installed: Iterable[str]) -> str | None:
    have = set(installed)
    for name in order:
        if name in have:
            return name
    return None


def teacher_model(installed: list[str] | None = None) -> str:
    pinned = os.environ.get("QWEN_MODEL", "").strip()
    if pinned:
        return pinned
    tags = list_tags() if installed is None else installed
    return _first_installed(TEACHER_PREF, tags) or TEACHER_PREF[0]


def volume_model(installed: list[str] | None = None) -> str:
    pinned = os.environ.get("QWEN_VOLUME_MODEL", "").strip()
    if pinned:
        return pinned
    tags = list_tags() if installed is None else installed
    picked = _first_installed(VOLUME_PREF, tags)
    if picked:
        return picked
    return teacher_model(tags)


def inspect_modelfile(name: str, timeout: int = 10) -> dict:
    """show el Modelfile. Marca si el system prompt es de otro proyecto."""
    body = json.dumps({"name": name}).encode()
    try:
        req = urllib.request.Request(
            f"{OLLAMA}/api/show", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            doc = json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        return {"name": name, "ok": False, "error": type(e).__name__}
    blob = json.dumps(doc, ensure_ascii=False)
    foreign = any(m in blob for m in ATLAS_MARKERS)
    return {
        "name": name,
        "ok": True,
        "foreign_system_prompt": foreign,
        "parameter_size": (doc.get("details") or {}).get("parameter_size"),
        "quantization": (doc.get("details") or {}).get("quantization_level"),
        "family": (doc.get("details") or {}).get("family"),
    }


def _chat(model: str, user: str, num_predict: int, timeout: int) -> dict:
    t0 = time.time()
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0.0, "num_predict": num_predict},
    }
    try:
        req = urllib.request.Request(
            f"{OLLAMA}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
        text = (out.get("message") or {}).get("content") or ""
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        return {
            "ok": True,
            "model": model,
            "latency_s": round(time.time() - t0, 2),
            "sample": text.strip()[:200],
        }
    except Exception as e:
        return {
            "ok": False,
            "model": model,
            "latency_s": round(time.time() - t0, 2),
            "error": f"{type(e).__name__}: {e}",
        }


#: Un episodio minimo de eleccion forzada: mide al profesor, no a la prosa.
_TEACHER_USER = (
    'Reply with ONLY {"choice":"a","evidence":"3 euros"}.\n'
    "Facts: bus 3 euros, train 9 euros.\n"
    "Question: cheapest?\n"
    "Options:\n- a: bus\n- b: train"
)
_VOLUME_USER = (
    "Rewrite in two English sentences, keep EVERY number: "
    "The bus costs 3 euros. Reply with ONLY the sentences."
)


def probe(timeout: int = 180) -> dict:
    """Tiempos en frio/caliente del profesor y del modelo de volumen."""
    tags = list_tags()
    teacher = teacher_model(tags)
    volume = volume_model(tags)
    models = []
    for name in (teacher, volume):
        if name not in models:
            models.append(name)
    rows = []
    for name in models:
        info = inspect_modelfile(name)
        cold = _chat(name, _TEACHER_USER if name == teacher else _VOLUME_USER,
                     num_predict=64, timeout=timeout)
        warm = _chat(name, _TEACHER_USER if name == teacher else _VOLUME_USER,
                     num_predict=64, timeout=timeout)
        role = []
        if name == teacher:
            role.append("teacher")
        if name == volume:
            role.append("volume")
        rows.append({
            "name": name,
            "role": role,
            "installed": name in tags,
            "modelfile": info,
            "cold": cold,
            "warm": warm,
        })
    doc = {
        "task": TASK,
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "note": "M5 Pro 48 GB: train the decision encoder on MPS, not CPU. "
                    "CPU is the unit-test device.",
            "ollama": OLLAMA,
            "tags": tags,
            "llama_installed": any("llama" in t.lower() for t in tags),
        },
        "resolved": {"teacher": teacher, "volume": volume},
        "runs": rows,
    }
    os.makedirs(GATE_DIR, exist_ok=True)
    path = os.path.join(GATE_DIR, "speed.json")
    with open(path, "w") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    doc["wrote"] = path
    return doc


def main(argv: list | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python3 -m data.local_models")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tags", help="list Ollama names")
    sub.add_parser("resolve", help="print teacher and volume picks")
    p = sub.add_parser("probe", help="time teacher vs volume and write the gate")
    p.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args(argv)
    if args.cmd == "tags":
        print(json.dumps(list_tags(), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "resolve":
        tags = list_tags()
        print(json.dumps({
            "tags": tags,
            "teacher": teacher_model(tags),
            "volume": volume_model(tags),
        }, ensure_ascii=False, indent=2))
        return 0
    doc = probe(timeout=args.timeout)
    print(json.dumps({
        "resolved": doc["resolved"],
        "wrote": doc["wrote"],
        "runs": [
            {"name": r["name"], "role": r["role"],
             "cold_s": r["cold"].get("latency_s"),
             "warm_s": r["warm"].get("latency_s"),
             "ok": r["cold"].get("ok") and r["warm"].get("ok"),
             "foreign_system": (r["modelfile"] or {}).get("foreign_system_prompt")}
            for r in doc["runs"]
        ],
    }, ensure_ascii=False, indent=2))
    return 0 if all(r["cold"].get("ok") for r in doc["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
