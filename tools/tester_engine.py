"""Tester engine: prueba el modelo v0 en terminal con juez LLM y minería de sets.

Uso interactivo:
    .venv-train/bin/python tools/tester_engine.py --run <ts> --task massive

Una sola prueba:
    .venv-train/bin/python tools/tester_engine.py --run <ts> --task massive \\
        --once --text "wake me up at 7am" --emit-training

Forward testing automático (inventa inputs ordenados, evalúa y puntúa):
    .venv-train/bin/python tools/tester_engine.py --task massive --auto --emit-training
    .venv-train/bin/python tools/tester_engine.py --task massive --auto 4 --judge none

Muestra por prueba: request JSON, top-5 del modelo, latencia (ms) y
veredicto del juez (Qwen local vía ollama; Gemini Flash si hay
GEMINI_API_KEY; heurística si no hay juez). Con --emit-training las
pruebas aceptadas se añaden como filas en schema canónico a
artifacts/tester/mined/ con matriz de cobertura por slice
(task_kind x dificultad x locale) para variedad estructurada.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SESS_DIR = ROOT / "artifacts" / "tester" / "sessions"
MINE_DIR = ROOT / "artifacts" / "tester" / "mined"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("TESTER_JUDGE_MODEL", "qwen3.6:27b-mlx")


def load_pack(run: str, task: str):
    with open(ROOT / "artifacts" / "runs" / run / "models" / f"{task}.pkl", "rb") as fh:
        return pickle.load(fh)


def latest_run() -> str:
    runs = sorted((ROOT / "artifacts" / "runs").iterdir())
    if not runs:
        raise SystemExit("no runs under artifacts/runs/")
    return runs[-1].name


def classify_slice(text: str) -> dict:
    loc = "es-ES" if re.search(r"[áéíóúñ¿¡]", text) else "en-US"
    n = len(text.split())
    diff = "short" if n <= 4 else ("long" if n > 12 else "mid")
    kind = "boolean" if re.search(r"\?|^(is|are|do|does|can|should)\b", text, re.I) else "choice"
    return {"locale": loc, "difficulty": diff, "kind": kind}


def judge_qwen(prompt: str, pred: str, probs: list, timeout: int = 120) -> dict:
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": (
            "You judge a classifier. Reply ONLY JSON "
            '{"verdict":"ok|wrong|unsure","reason":"<short>"}.\n'
            f"Input: {prompt}\nPrediction: {pred} probs={probs}"
        ),
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    txt = out.get("response", "")
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            j = json.loads(m.group(0))
            if j.get("verdict") in ("ok", "wrong", "unsure"):
                return {"judge": f"qwen:{OLLAMA_MODEL}", "verdict": j["verdict"],
                        "reason": str(j.get("reason", ""))[:300]}
        except json.JSONDecodeError:
            pass
    return {"judge": f"qwen:{OLLAMA_MODEL}", "verdict": "unsure", "reason": txt[:300]}


def judge_gemini(prompt: str, pred: str, probs: list, timeout: int = 60) -> dict:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("no GEMINI_API_KEY")
    model = os.environ.get("TESTER_GEMINI_MODEL", "gemini-2.0-flash")
    body = json.dumps({"contents": [{"parts": [{
        "text": 'Reply ONLY JSON {"verdict":"ok|wrong|unsure","reason":"<short>"}. '
                f"Input: {prompt}\nPrediction: {pred} probs={probs}"}]}]}).encode()
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
           f":generateContent?key={key}")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    try:
        txt = out["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return {"judge": f"gemini:{model}", "verdict": "unsure", "reason": "bad-response"}
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            j = json.loads(m.group(0))
            if j.get("verdict") in ("ok", "wrong", "unsure"):
                return {"judge": f"gemini:{model}", "verdict": j["verdict"],
                        "reason": str(j.get("reason", ""))[:300]}
        except json.JSONDecodeError:
            pass
    return {"judge": f"gemini:{model}", "verdict": "unsure", "reason": txt[:300]}


def judge_heuristic(prompt: str, pred: str, top_p: float) -> dict:
    v = "ok" if top_p >= 0.5 else "unsure"
    return {"judge": "heuristic", "verdict": v,
            "reason": f"top_p={top_p:.2f}, no LLM judge reachable"}


def predict(pack, task: str, text: str) -> dict:
    vec, clf = pack["vectorizer"], pack["clf"]
    request = {"task": task, "text": text, "top_k": 5,
               "ts": datetime.now(timezone.utc).isoformat()}
    t0 = time.perf_counter()
    proba = clf.predict_proba(vec.transform([text]))[0]
    dt_ms = (time.perf_counter() - t0) * 1000
    labels = list(clf.classes_)
    ranked = sorted(zip(labels, [float(p) for p in proba]), key=lambda kv: -kv[1])[:5]
    pred, top_p = ranked[0]
    probs = [round(p, 3) for _, p in ranked]
    return {"request": request, "prediction": pred, "top5": ranked,
            "latency_ms": round(dt_ms, 2), "top_p": top_p, "probs": probs,
            "slice": classify_slice(text)}


def judge_result(text: str, pred: str, probs: list, top_p: float, judge: str,
                 timeout: int) -> dict:
    if judge == "qwen":
        try:
            return judge_qwen(text, pred, probs, timeout=timeout)
        except Exception as e:
            j = judge_heuristic(text, pred, top_p)
            j["reason"] += f" (qwen error: {e})"
            return j
    elif judge == "gemini":
        try:
            return judge_gemini(text, pred, probs, timeout=min(timeout, 60))
        except Exception as e:
            j = judge_heuristic(text, pred, top_p)
            j["reason"] += f" (gemini error: {e})"
            return j
    elif judge == "none":
        return judge_heuristic(text, pred, top_p)
    else:  # auto: gemini si hay key, si no qwen, si no heurística
        if os.environ.get("GEMINI_API_KEY"):
            try:
                return judge_gemini(text, pred, probs, timeout=min(timeout, 60))
            except Exception:
                try:
                    return judge_qwen(text, pred, probs, timeout=timeout)
                except Exception as e:
                    j = judge_heuristic(text, pred, top_p)
                    j["reason"] += f" (auto fallback: {e})"
                    return j
        else:
            try:
                return judge_qwen(text, pred, probs, timeout=timeout)
            except Exception as e:
                j = judge_heuristic(text, pred, top_p)
                j["reason"] += f" (qwen error: {e})"
                return j


def run_once(pack, task: str, text: str, judge: str, timeout: int = 120) -> dict:
    base = predict(pack, task, text)
    j = judge_result(text, base["prediction"], base["probs"], base["top_p"],
                     judge, timeout)
    base["judgment"] = j
    del base["top_p"], base["probs"]
    return base
    if judge == "qwen":
        try:
            j = judge_qwen(text, pred, probs)
        except Exception as e:
            j = judge_heuristic(text, pred, top_p)
            j["reason"] += f" (qwen error: {e})"
    elif judge == "gemini":
        try:
            j = judge_gemini(text, pred, probs)
        except Exception as e:
            j = judge_heuristic(text, pred, top_p)
            j["reason"] += f" (gemini error: {e})"
    elif judge == "none":
        j = judge_heuristic(text, pred, top_p)
    else:  # auto: gemini si hay key, si no qwen, si no heurística
        if os.environ.get("GEMINI_API_KEY"):
            try:
                j = judge_gemini(text, pred, probs)
            except Exception:
                try:
                    j = judge_qwen(text, pred, probs)
                except Exception as e:
                    j = judge_heuristic(text, pred, top_p)
                    j["reason"] += f" (auto fallback: {e})"
        else:
            try:
                j = judge_qwen(text, pred, probs)
            except Exception as e:
                j = judge_heuristic(text, pred, top_p)
                j["reason"] += f" (qwen error: {e})"
    return {"request": request, "prediction": pred, "top5": ranked,
            "latency_ms": round(dt_ms, 2), "judgment": j,
            "slice": classify_slice(text)}


def print_result(res: dict) -> None:
    print(f"REQUEST {json.dumps(res['request'], ensure_ascii=False)}")
    print(f"MODEL pred={res['prediction']} latency_ms={res['latency_ms']}")
    for lab, p in res["top5"]:
        print(f"  {lab}: {p:.3f}")
    j = res["judgment"]
    print(f"JUDGE [{j['judge']}] verdict={j['verdict']} reason={j['reason']}")
    print(f"SLICE {res['slice']}")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_canonical(task: str, text: str, res: dict, i: int) -> dict:
    opts = [{"id": lab, "text": lab} for lab, _ in res["top5"][:4]]
    if res["prediction"] not in [o["id"] for o in opts]:
        opts[0] = {"id": res["prediction"], "text": res["prediction"]}
    kind = res["slice"]["kind"]
    return {"state": text, "questions": [{
        "id": f"tester-mined-{i}", "kind": kind, "options": opts,
        "answer": res["prediction"],
        "teacher_conf": round(res["top5"][0][1], 3),
        "judge": res["judgment"],
    }], "split": "train"}


POINTS = {"ok": 2, "unsure": 1, "wrong": 0}

# Suite ordenada para forward testing: de fácil a difícil, alternando
# locale en-US/es-ES. Cubre los slices (kind x dificultad x locale) para
# que cada tanda componga variedad estructurada minable.
AUTO_SUITES = {
    "massive": [
        "set an alarm for 7am",
        "play some music",
        "what is the weather today",
        "turn off the lights",
        "call mom please",
        "remind me to buy milk tomorrow morning",
        "set a timer for twenty five minutes for the pasta",
        "pon una alarma a las siete de la mañana",
        "reproduce música de los ochenta en el salón",
        "qué tiempo hará mañana en Barcelona, ¿lloverá?",
        "book a flight from Madrid to Berlin next Friday evening",
        "cancela mi reunión de las tres y avisa al equipo por email",
        "translate 'the early bird catches the worm' to Spanish and explain the idiom meaning",
        "quiero que compares tarifa luz regulada vs libre y me digas cuál me conviene con mi consumo",
    ],
    "boolq": [
        "is water wet",
        "do birds fly south in winter",
        "is Barcelona the capital of Spain",
        "can penguins fly or do they only swim",
        "¿es Madrid la capital de España?",
        "does photosynthesis occur at night when there is no sunlight available",
        "¿puede la fotosíntesis producirse de noche sin luz solar en plantas de interior?",
        "is it true that quantum entanglement allows faster-than-light communication between particles",
    ],
    "civil-comments": [
        "great article, thanks for sharing",
        "I disagree with this point completely",
        "este artículo es muy interesante, gracias",
        "you are an idiot and nobody likes you, shut up",
        "eres un imbécil, cállate, nadie te quiere aquí",
        "While I respect your view, the data on page 3 contradicts the conclusion about taxes",
    ],
    "helpsteer2": [
        "write a haiku about the sea",
        "explain photosynthesis simply",
        "escribe un haiku sobre el mar",
        "write a polite but firm email declining a job offer with three specific reasons",
        "redacta un email formal rechazando una oferta con tres motivos concretos y propuesta alternativa",
    ],
}


try:
    from data.intent.email_triage import FORWARD_CASES as _ET_CASES
    AUTO_SUITES["email-triage"] = list(_ET_CASES)
except Exception:
    pass


def build_auto_suite(task: str, n: int | None) -> list:
    base = AUTO_SUITES.get(task, AUTO_SUITES["massive"])
    if n is None or n < 0:
        return list(base)
    if n == 0:
        return []
    if n <= len(base):
        return base[:n]
    # Si piden más casos que la base, rota con sufijos numerados (orden estable).
    out = list(base)
    i = 1
    while len(out) < n:
        for t in base:
            if len(out) >= n:
                break
            out.append(f"{t} (variante {i})")
        i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Tester engine con juez LLM")
    ap.add_argument("--run", default=None)
    ap.add_argument("--task", default="massive")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--text", default=None)
    ap.add_argument("--judge", default="auto", choices=["auto", "qwen", "gemini", "none"])
    ap.add_argument("--emit-training", action="store_true")
    ap.add_argument("--auto", type=int, default=None, nargs="?", const=-1, metavar="N",
                    help="forward testing: inventa inputs ordenados y corre solo (solo --auto=toda la suite, --auto N=N casos)")
    ap.add_argument("--judge-timeout", type=int, default=120,
                    help="timeout en segundos por llamada al juez LLM (default 120)")
    args = ap.parse_args()

    run = args.run or latest_run()
    pack = load_pack(run, args.task)
    print(f"tester-engine run={run} task={args.task} judge={args.judge}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sess_path = SESS_DIR / f"{stamp}.jsonl"
    mined_path = MINE_DIR / f"{stamp}.jsonl"
    coverage: dict[str, int] = {}
    n = 0

    def handle(text: str) -> None:
        nonlocal n
        res = run_once(pack, args.task, text, args.judge, args.judge_timeout)
        print_result(res)
        append_jsonl(sess_path, res)
        key = f"{res['slice']['kind']}/{res['slice']['difficulty']}/{res['slice']['locale']}"
        coverage[key] = coverage.get(key, 0) + 1
        n += 1
        if args.emit_training and res["judgment"]["verdict"] == "ok":
            append_jsonl(mined_path, to_canonical(args.task, text, res, n))

    if args.once:
        if not args.text:
            raise SystemExit("--once requiere --text")
        handle(args.text)
    elif args.auto is not None:
        suite = build_auto_suite(args.task, args.auto)
        total, mx = 0, 2 * len(suite)
        verdicts: dict[str, int] = {}
        lat: list[float] = []
        print(f"AUTO forward-testing: {len(suite)} casos ordenados (fácil→difícil, en/es)",
              flush=True)
        for i, text in enumerate(suite, 1):
            # Fase 1: predicción inmediata (ms) — se muestra ANTES de llamar
            # al juez para que el caso avance 1 a 1 sin pausas aparentes.
            base = predict(pack, args.task, text)
            print(f"\n[{i}/{len(suite)}] {text}", flush=True)
            print(f"  pred       {base['prediction']:<22} "
                  f"p={base['top5'][0][1]:.3f}   lat={base['latency_ms']}ms",
                  flush=True)
            alts = "   ".join(f"{lab} {p:.3f}" for lab, p in base["top5"][1:])
            if alts:
                print(f"  alt        {alts}", flush=True)
            print(f"  juez       {args.judge} evaluando… (timeout {args.judge_timeout}s)",
                  flush=True)
            # Fase 2: veredicto del juez (lento: Qwen 27b tarda ~30-60s/caso).
            t0 = time.perf_counter()
            j = judge_result(text, base["prediction"], base["probs"],
                             base["top_p"], args.judge, args.judge_timeout)
            judge_ms = (time.perf_counter() - t0) * 1000
            res = {**base, "judgment": j,
                   "judge_latency_ms": round(judge_ms, 1)}
            del res["top_p"], res["probs"]
            pts = POINTS.get(j["verdict"], 0)
            total += pts
            verdicts[j["verdict"]] = verdicts.get(j["verdict"], 0) + 1
            lat.append(res["latency_ms"])
            append_jsonl(sess_path, {**res, "auto_idx": i, "auto_points": pts})
            key = f"{res['slice']['kind']}/{res['slice']['difficulty']}/{res['slice']['locale']}"
            coverage[key] = coverage.get(key, 0) + 1
            n += 1
            if args.emit_training and j["verdict"] == "ok":
                append_jsonl(mined_path, to_canonical(args.task, text, res, n))
            print(f"  veredicto  {j['verdict']:<22} "
                  f"+{pts}pts   acum {total}/{mx}   juez_lat={judge_ms / 1000:.1f}s",
                  flush=True)
            reason = (j.get("reason") or "").strip()
            if reason:
                print(f"             {reason[:160]}", flush=True)
        avg = sum(lat) / len(lat) if lat else 0.0
        print(f"\nTOTAL {total}/{mx} pts · media {total/len(suite):.2f}/caso · "
              f"veredictos={json.dumps(verdicts, ensure_ascii=False)} · lat_media={avg:.1f}ms")
    else:
        print("REPL: escribe texto + Enter. Comandos: /quit /coverage")
        while True:
            try:
                line = input("test> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line in ("/quit", "/q", ":q"):
                break
            if line == "/coverage":
                print(json.dumps(coverage, indent=1, ensure_ascii=False))
                continue
            handle(line)
    print(f"session: {sess_path} ({n} pruebas)")
    if args.emit_training:
        print(f"mined: {mined_path}")
    print(f"coverage: {json.dumps(coverage, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
