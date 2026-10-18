"""Verificador separado, cuarentena y muestra humana (#T-episode-verify).

Toma los episodios PUBLICADOS por ``data.episode_gen`` y les aplica tres
filtros distintos, en este orden, sin resolver nunca un desacuerdo a ojo:

1. **Regla, donde exista.** Cada familia deja una ``rule_trace``; el gold se
   recalcula desde ella (``comparison_gold`` / ``priority_gold`` para lo
   numérico, hecho/categoría/forma lógica para lo demás) y se compara con el
   ``answer`` publicado. El contrato ``episode-v1`` se vuelve a validar, y un
   estado que arrastre el marcador del prompt del generador (``KEEP:``) es un
   atajo, no un episodio: cae en cuarentena con motivo.
2. **Verificador separado.** Un prompt DISTINTO del profesor del generador
   (``episode_gen.teacher_structured_batch`` pide JSON con evidencia y ve el
   estado como «Facts»); éste es un auditor que recibe contexto + pregunta +
   opciones en orden barajado, SIN el gold, y contesta ``<n>: <id>`` (o
   ``none`` si nada lo sostiene). Misma familia de modelo (Qwen local): NO es
   verdad independiente, y por eso el desacuerdo se DESCARTA a cuarentena en
   vez de arbitrarse.
3. **Muestra humana estratificada** por familia × idioma × tipo de variante,
   con el tamaño por celda escrito ANTES de muestrear (``HUMAN_PER_CELL``).
   Este módulo no la revisa: ``human_error`` queda ``null`` y el gate no lo
   firma hasta que el operador la devuelva.

Y el embudo, reconstruible desde manifests (``build_funnel``): generado →
aceptado → publicado → verificado → **consumido por un entreno**. Consumido
se lee de los ``train_manifest.json`` bajo ``artifacts/checkpoints`` que
citen este run; hoy no hay ninguno y la cifra es 0, escrita como tal.

Salidas (todas en el directorio del run)::

    verified.jsonl        episodios que pasan los tres filtros + verify_trace
    quarantine.jsonl      {"episode", "reasons", "verify_trace"} — artefacto
    verify_manifest.json  versión, prompt sha, modelo, conteos, comando
    human_sample.jsonl    la muestra, human_error: null, awaiting-operator
    funnel.json           el embudo por familia/idioma, con sus fuentes

Correr la verificación contra Qwen SOLO cuando el servidor no esté
sirviendo al piloto (un run de Qwen a la vez)::

    PYTHONPATH=. .venv-train/bin/python -m data.episode_verify run \\
        --dir artifacts/episodes-qwen/pilot-2k --batch 20 --concurrency 2
    PYTHONPATH=. .venv-train/bin/python -m data.episode_verify gate \\
        --dir artifacts/episodes-qwen/pilot-2k
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data import episode_contract as EC  # noqa: E402
from data import episode_gen as EG  # noqa: E402

VERIFIER_VERSION = "episode-verify-v1"
TASK = "T-episode-verify"
GATE_PATH = os.path.join(ROOT, "artifacts", "gates", TASK, "gate.json")
CHECKPOINTS_ROOT = os.path.join(ROOT, "artifacts", "checkpoints")

#: Tamaño de la muestra humana POR CELDA (familia × idioma × variante),
#: fijado aquí antes de muestrear: 5 × 5 familias × 2 idiomas × 2 variantes
#: = 100 filas. Cambiarlo después de ver el error es elegir la cifra.
HUMAN_PER_CELL = 5
HUMAN_SEED = 20260927
VARIANT_TYPES = ("base", "counterfactual")

#: Marcadores del prompt del generador que jamás deben viajar en un estado:
#: ``KEEP: <evidencia>`` regala la respuesta al estudiante.
PROMPT_LEAK_MARKERS = ("KEEP:",)

MODEL = EG.MODEL
OLLAMA = EG.OLLAMA


# -- CI ---------------------------------------------------------------------
def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple:
    """IC95 % de Wilson — el honesto para celdas pequeñas."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def cell_of(ep: dict) -> str:
    return f"{ep['family']}/{ep['lang']}"


def variant_type(ep: dict) -> str:
    """``base`` o ``counterfactual`` según el plan de ``episode_gen``.

    ``_work_items`` consume dos índices consecutivos por caso — par = base,
    impar = contrafactual — y el índice cierra el ``id``. IDs de otra forma
    (fixture a mano) quedan ``unknown``: no se adivina.
    """
    tail = str(ep.get("id", "")).rsplit("-", 1)[-1]
    if not tail.isdigit():
        return "unknown"
    return VARIANT_TYPES[int(tail) % 2]


# -- 1. regla, donde exista ---------------------------------------------------
def rule_expected(ep: dict) -> str | None:
    """El gold que la regla de la familia recalcula desde ``rule_trace``.

    ``None`` cuando el episodio no trae traza reconocible: la regla no
    aplica y lo dice, no lo aprueba.
    """
    rt = ep.get("rule_trace") or {}
    cands = ep.get("candidates") or []
    ids = [c.get("id") for c in cands]
    fam = ep.get("family")
    try:
        if fam == EC.COMPARISON and "attrs" in rt and "criterion" in rt:
            return ids[EG.comparison_gold(rt["attrs"], rt["criterion"])]
        if fam == EC.PRIORITY and "attrs" in rt:
            return ids[EG.priority_gold(rt["attrs"])]
        if fam == EC.EXTRACTION and "fact" in rt:
            hits = [c["id"] for c in cands
                    if _norm(c.get("text")) == _norm(rt["fact"])]
            return hits[0] if len(hits) == 1 else None
        if fam == EC.DESCRIPTION and "category" in rt:
            head = _norm(rt["category"])
            hits = [c["id"] for c in cands
                    if _norm(c.get("text")).split(":", 1)[0] == head]
            return hits[0] if len(hits) == 1 else None
        if fam == EC.INFERENCE and "form" in rt:
            return ids[0] if rt["form"] == 0 else ids[1]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return None


def rule_verify(ep: dict) -> dict:
    """Filtro 1: contrato + regla + fuga de prompt. Motivos, nunca opiniones."""
    reasons = [f"contract: {r}" for r in EC.validate(ep)]
    for m in PROMPT_LEAK_MARKERS:
        if m in str(ep.get("state", "")) or m in str(ep.get("question", "")):
            reasons.append(f"prompt-leak: state carries {m!r} from the "
                           "generator prompt")
    expected = rule_expected(ep)
    applies = expected is not None
    agree = (expected == ep.get("answer")) if applies else None
    if applies and not agree:
        reasons.append(f"rule: recomputed gold {expected!r} != published "
                       f"{ep.get('answer')!r}")
    return {"applies": applies, "expected": expected, "agree": agree,
            "reasons": reasons}


# -- 2. verificador separado --------------------------------------------------
_AUDITOR_RULE = (
    "You are an independent auditor checking multiple-choice items. You "
    "never see an answer key. For EACH item, using ONLY the text inside "
    "that item, decide which option the Context supports as the answer to "
    "the Question. If no option is supported, answer none."
)


def _shuffled(cands: list[dict], key: str) -> list[dict]:
    order = list(cands)
    random.Random(f"{VERIFIER_VERSION}\x00{key}").shuffle(order)
    return order


def verifier_prompt(cases: list[dict]) -> str:
    """El prompt del auditor: contexto + pregunta + opciones barajadas.

    Ni gold, ni evidencia, ni «Facts», ni JSON: nada de lo que el profesor
    del generador ve o devuelve. Las opciones van en orden barajado por
    episodio (semilla estable) para que la posición no sea la respuesta.
    """
    blocks = []
    for i, c in enumerate(cases, 1):
        opts = "\n".join(
            f"  ({o['id']}) {o['text']}"
            for o in _shuffled(c["candidates"], c.get("id", str(i))))
        blocks.append(f"## Item {i}\nContext: {c['state']}\n"
                      f"Question: {c['question']}\nOptions:\n{opts}")
    return (
        f"{_AUDITOR_RULE}\n\nReply with exactly {len(cases)} lines and "
        f"nothing else, one per item and in order, each in the form "
        f"`<item number>: <option id>` (or `<item number>: none`).\n\n"
        + "\n\n".join(blocks)
    )


def prompt_sha() -> str:
    """Huella del prompt del auditor — va al manifest y al gate."""
    probe = [{"id": "sha", "state": "S", "question": "Q?",
              "candidates": [{"id": "a", "text": "A"},
                             {"id": "b", "text": "B"}]}]
    return hashlib.sha256(verifier_prompt(probe).encode("utf-8")).hexdigest()


def _chat(prompt: str, num_predict: int, timeout: int) -> str | None:
    """Una llamada a Ollama a temperatura 0; ``None`` si falla."""
    body = json.dumps({
        "model": MODEL, "stream": False, "think": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.0, "num_predict": num_predict},
    }).encode()
    try:
        req = urllib.request.Request(
            f"{OLLAMA}/api/chat", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())["message"]["content"]
        if "</think>" in out:
            out = out.split("</think>", 1)[1]
        return out.strip()
    except Exception:
        return None


_LINE = re.compile(r"^\s*(?:item\s*)?(\d+)\s*[:.)\-]\s*\(?([A-Za-z0-9_.:~-]+)\)?",
                   re.IGNORECASE)


def parse_verdicts(out: str | None, n: int) -> list:
    """``<n>: <id>`` por línea → ``n`` huecos; el que falte es ``None``."""
    got: list = [None] * n
    if not out:
        return got
    for line in out.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        k = int(m.group(1)) - 1
        if 0 <= k < n and got[k] is None:
            got[k] = m.group(2)
    return got


def _verdict(raw: str | None, ids: list, reply: str | None) -> dict:
    trace = {"backend": "qwen-local-auditor", "prompt_version": VERIFIER_VERSION}
    if reply is None:
        trace.update({"status": "no-reply", "choice": None})
    elif raw is None:
        trace.update({"status": "missing", "choice": None})
    elif raw.lower() == "none":
        trace.update({"status": "abstain", "choice": None})
    elif raw in ids:
        trace.update({"status": "verdict", "choice": raw})
    else:
        trace.update({"status": "invalid", "choice": None, "raw": raw[:40]})
    return trace


def verify_batch(cases: list[dict], timeout: int = 1800) -> list[dict]:
    """Veredicto del auditor para MUCHOS episodios en una petición.

    Cada hueco vuelve con su ``status``: ``verdict`` (id válido),
    ``abstain``, ``invalid`` (id inventado), ``missing`` (línea ausente) o
    ``no-reply``. Sólo ``verdict`` puede coincidir con el gold; todo lo
    demás es cuarentena con motivo, nunca un hueco corrido.
    """
    if not cases:
        return []
    reply = _chat(verifier_prompt(cases), num_predict=12 * len(cases) + 16,
                  timeout=timeout)
    raws = parse_verdicts(reply, len(cases))
    return [_verdict(r, [o["id"] for o in c["candidates"]], reply)
            for r, c in zip(raws, cases)]


def _tick(done: int, total: int, t0: float, items: int) -> None:
    el = time.time() - t0
    print(f"  verify {done}/{total} reqs · {items} items · "
          f"{items / el if el else 0.0:.3f} item/s", flush=True)


def verifier_traces(eps: list[dict], batch: int, concurrency: int) -> list:
    jobs = [list(range(s, min(s + batch, len(eps))))
            for s in range(0, len(eps), batch)]
    out: list = [None] * len(eps)

    def one(idxs):
        return idxs, verify_batch([eps[i] for i in idxs])

    t0, seen = time.time(), 0
    with cf.ThreadPoolExecutor(max(1, concurrency)) as ex:
        for k, (idxs, traces) in enumerate(ex.map(one, jobs), 1):
            seen += len(idxs)
            _tick(k, len(jobs), t0, seen)
            for i, tr in zip(idxs, traces):
                out[i] = tr
    return out


# -- decisión: verificado o cuarentena -----------------------------------------
def decide(ep: dict, rule: dict, trace: dict | None) -> tuple[str, list[str]]:
    """``("verified"|"quarantine", reasons)``. Conservador por diseño."""
    reasons = list(rule["reasons"])
    if trace is None:
        reasons.append("verifier: not run")
    elif trace.get("status") != "verdict":
        reasons.append(f"verifier: {trace.get('status')}")
    elif trace["choice"] != ep.get("answer"):
        reasons.append(f"verifier: chose {trace['choice']!r}, published "
                       f"gold is {ep.get('answer')!r}")
    return ("quarantine" if reasons else "verified"), reasons


def _reason_key(reason: str) -> str:
    """Motivo → clave contable: ``rule:disagree``, ``verifier:abstain``…"""
    head, _, tail = reason.partition(":")
    tail = tail.strip()
    if head in ("rule", "verifier") and ("!=" in tail or tail.startswith("chose")):
        return f"{head}:disagree"
    if head == "contract":
        return "contract:" + tail.split("(")[0].split(":")[0].strip()[:48]
    return f"{head}:{tail[:32]}"


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


# -- 3. muestra humana estratificada -----------------------------------------
def human_sample(verified: list[dict], per_cell: int = HUMAN_PER_CELL,
                 seed: int = HUMAN_SEED) -> list[dict]:
    """``per_cell`` por familia × idioma × variante, con semilla; sin revisar.

    Cada fila lleva lo que el operador necesita para juzgar el gold (estado,
    pregunta, opciones, gold, evidencia) y ``human_error: null``: nadie más
    que el operador lo rellena.
    """
    cells: dict[tuple, list[dict]] = {}
    for ep in verified:
        cells.setdefault((ep["family"], ep["lang"], variant_type(ep)), []) \
            .append(ep)
    rows = []
    for key in sorted(cells):
        pool = sorted(cells[key], key=lambda e: e["id"])
        rng = random.Random(f"{VERIFIER_VERSION}\x00human\x00{seed}\x00{key}")
        for ep in rng.sample(pool, min(per_cell, len(pool))):
            rows.append({
                "id": ep["id"], "family": ep["family"], "lang": ep["lang"],
                "variant_type": variant_type(ep),
                "state": ep["state"], "question": ep["question"],
                "candidates": ep["candidates"], "gold": ep.get("answer"),
                "evidence": ep.get("evidence"),
                "human_error": None, "human_note": None,
                "status": "awaiting-operator",
            })
    return rows


# -- embudo desde manifests -----------------------------------------------------
def _sha(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _count_cells(rows: list[dict]) -> dict:
    out: dict = {}
    for ep in rows:
        out[cell_of(ep)] = out.get(cell_of(ep), 0) + 1
    return dict(sorted(out.items()))


def consumed_from_train_manifests(out: str, checkpoints_root: str,
                                  verified: list[dict]) -> dict:
    """Consumido = lo que un ``train_manifest.json`` declara haber leído
    de ESTE run: ``episodes.source`` igual al directorio (o
    ``episodes.manifest_sha`` igual al sha del manifest) y, o bien
    ``episodes.consumed_ids`` (auditable id a id), o bien
    ``episodes.consumed_by_cell``. Sin manifest que lo cite: 0, por celda.
    """
    rel = os.path.relpath(os.path.abspath(out), ROOT)
    msha = _sha(os.path.join(out, "manifest.json"))
    by_id = {ep["id"]: ep for ep in verified}
    cells: dict = {}
    found = []
    pattern = os.path.join(checkpoints_root, "**", "train_manifest.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            tm = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ep_block = tm.get("episodes") or {}
        src = ep_block.get("source")
        cites = (src and os.path.normpath(str(src)) == os.path.normpath(rel)) \
            or (msha and ep_block.get("manifest_sha") == msha)
        if not cites:
            continue
        n_here = 0
        if isinstance(ep_block.get("consumed_ids"), list):
            for eid in ep_block["consumed_ids"]:
                ep = by_id.get(eid)
                if ep is None:
                    continue
                cells[cell_of(ep)] = cells.get(cell_of(ep), 0) + 1
                n_here += 1
        elif isinstance(ep_block.get("consumed_by_cell"), dict):
            for k, v in ep_block["consumed_by_cell"].items():
                cells[k] = cells.get(k, 0) + int(v)
                n_here += int(v)
        found.append({"path": os.path.relpath(path, ROOT),
                      "sha256": _sha(path), "n_consumed": n_here})
    return {"total": sum(cells.values()), "by_cell": dict(sorted(cells.items())),
            "train_manifests": found}


def build_funnel(out: str, checkpoints_root: str = CHECKPOINTS_ROOT) -> dict:
    """generado → aceptado → publicado → verificado → consumido, por celda.

    Todo sale de ficheros del run y de los manifests de entreno; ninguna
    cifra se pasa por argumento. ``generado`` = items que el generador
    terminó (publicados + rechazados); ``aceptado`` = los que pasaron
    contrato + profesor (= publicados en ``episodes.jsonl``; la etapa se
    deja explícita porque el día que se acepte sin publicar, se verá).
    """
    manifest = json.load(open(os.path.join(out, "manifest.json")))
    episodes = _read_jsonl(os.path.join(out, "episodes.jsonl"))
    rejects = [r["episode"] for r in _read_jsonl(os.path.join(out, "rejects.jsonl"))
               if isinstance(r.get("episode"), dict)]
    verified = _read_jsonl(os.path.join(out, "verified.jsonl"))
    quarantine = _read_jsonl(os.path.join(out, "quarantine.jsonl"))
    vm_path = os.path.join(out, "verify_manifest.json")
    consumed = consumed_from_train_manifests(out, checkpoints_root, verified)
    gen_cells = _count_cells(episodes + rejects)
    stages = {
        "planned": {"total": manifest.get("n"),
                    "by_cell": manifest.get("planned_split", {})},
        "generated": {"total": len(episodes) + len(rejects),
                      "by_cell": gen_cells},
        "accepted": {"total": len(episodes), "by_cell": _count_cells(episodes),
                     "rejected": len(rejects),
                     "rejected_by_cell": _count_cells(rejects)},
        "published": {"total": len(episodes),
                      "by_cell": _count_cells(episodes)},
        "verified": {"total": len(verified), "by_cell": _count_cells(verified),
                     "quarantined": len(quarantine),
                     "quarantined_by_cell": _count_cells(
                         [q["episode"] for q in quarantine])},
        "consumed": consumed,
    }
    return {
        "task": TASK, "dir": os.path.relpath(os.path.abspath(out), ROOT),
        "stages": stages,
        "sources": {
            "manifest.json": _sha(os.path.join(out, "manifest.json")),
            "episodes.jsonl": _sha(os.path.join(out, "episodes.jsonl")),
            "rejects.jsonl": _sha(os.path.join(out, "rejects.jsonl")),
            "verify_manifest.json": _sha(vm_path),
            "verified.jsonl": _sha(os.path.join(out, "verified.jsonl")),
            "checkpoints_root": os.path.relpath(
                os.path.abspath(checkpoints_root), ROOT),
        },
    }


# -- run --------------------------------------------------------------------
def run(out: str, batch: int = 20, concurrency: int = 2, limit: int = 0,
        command: str = "", checkpoints_root: str = CHECKPOINTS_ROOT) -> dict:
    """Los tres filtros sobre ``episodes.jsonl`` de ``out``; manifest antes."""
    episodes = _read_jsonl(os.path.join(out, "episodes.jsonl"))
    if limit:
        episodes = episodes[:limit]
    vm_path = os.path.join(out, "verify_manifest.json")
    manifest = {
        "task": TASK, "verifier_version": VERIFIER_VERSION,
        "generator_manifest_sha": _sha(os.path.join(out, "manifest.json")),
        "episodes_sha": _sha(os.path.join(out, "episodes.jsonl")),
        "n_input": len(episodes), "qwen_model": MODEL,
        "prompt_sha": prompt_sha(), "batch": batch, "concurrency": concurrency,
        "human_per_cell": HUMAN_PER_CELL, "human_seed": HUMAN_SEED,
        "command": command or f"python3 -m data.episode_verify run --dir "
                              f"{os.path.relpath(out, ROOT)} --batch {batch} "
                              f"--concurrency {concurrency}",
        "status": "planned",
    }
    _write_json(vm_path, manifest)

    t0 = time.time()
    rules = [rule_verify(ep) for ep in episodes]
    traces = verifier_traces(episodes, batch, concurrency)
    verified, quarantine = [], []
    agree: dict = {}
    reasons_count: dict = {}
    for ep, rule, tr in zip(episodes, rules, traces):
        kind, reasons = decide(ep, rule, tr)
        row = dict(ep)
        row["verify_trace"] = {"rule": rule, "verifier": tr,
                               "verifier_version": VERIFIER_VERSION}
        c = agree.setdefault(cell_of(ep), {"n": 0, "verdicts": 0, "agree": 0,
                                           "rule_applies": 0, "rule_agree": 0})
        c["n"] += 1
        if tr and tr.get("status") == "verdict":
            c["verdicts"] += 1
            c["agree"] += int(tr["choice"] == ep.get("answer"))
        if rule["applies"]:
            c["rule_applies"] += 1
            c["rule_agree"] += int(bool(rule["agree"]))
        if kind == "verified":
            verified.append(row)
        else:
            quarantine.append({"episode": ep, "reasons": reasons,
                               "verify_trace": row["verify_trace"]})
            for r in reasons:
                key = _reason_key(r)
                reasons_count[key] = reasons_count.get(key, 0) + 1
    _write_jsonl(os.path.join(out, "verified.jsonl"), verified)
    _write_jsonl(os.path.join(out, "quarantine.jsonl"), quarantine)
    sample = human_sample(verified)
    _write_jsonl(os.path.join(out, "human_sample.jsonl"), sample)
    manifest.update({
        "status": "published", "n_verified": len(verified),
        "n_quarantine": len(quarantine),
        "quarantine_reasons": dict(sorted(reasons_count.items())),
        "agreement_by_cell": dict(sorted(agree.items())),
        "human_sample_n": len(sample), "elapsed_s": round(time.time() - t0, 1),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    _write_json(vm_path, manifest)
    _write_json(os.path.join(out, "funnel.json"),
                build_funnel(out, checkpoints_root))
    return manifest


# -- gate ---------------------------------------------------------------------
def _agreement_table(verified: list[dict], quarantine: list[dict]) -> dict:
    """Acuerdo generador↔verificador por celda: n, acuerdos, tasa, IC95 %.

    Sobre TODO lo verificado + cuarentena (los episodios que recibieron
    veredicto), no sólo sobre lo que pasó — si no, la tasa sería 1 por
    construcción.
    """
    rows = [(ep, ep["verify_trace"]) for ep in verified] + \
           [(q["episode"], q["verify_trace"]) for q in quarantine]
    cells: dict = {}
    for ep, vt in rows:
        tr = vt.get("verifier") or {}
        c = cells.setdefault(cell_of(ep), {"n": 0, "verdicts": 0, "agree": 0})
        c["n"] += 1
        if tr.get("status") == "verdict":
            c["verdicts"] += 1
            c["agree"] += int(tr.get("choice") == ep.get("answer"))
    out: dict = {}
    for key, c in sorted(cells.items()):
        lo, hi = wilson_interval(c["agree"], c["n"])
        out[key] = {**c, "rate": round(c["agree"] / c["n"], 4) if c["n"] else None,
                    "ci95": [round(lo, 4), round(hi, 4)]}
    tot = {"n": sum(c["n"] for c in cells.values()),
           "verdicts": sum(c["verdicts"] for c in cells.values()),
           "agree": sum(c["agree"] for c in cells.values())}
    lo, hi = wilson_interval(tot["agree"], tot["n"])
    by_family: dict = {}
    for key, c in cells.items():
        fam = key.split("/")[0]
        f = by_family.setdefault(fam, {"n": 0, "agree": 0})
        f["n"] += c["n"]
        f["agree"] += c["agree"]
    for f in by_family.values():
        lo_f, hi_f = wilson_interval(f["agree"], f["n"])
        f.update({"rate": round(f["agree"] / f["n"], 4) if f["n"] else None,
                  "ci95": [round(lo_f, 4), round(hi_f, 4)]})
    return {"by_cell": out, "by_family": dict(sorted(by_family.items())),
            "overall": {**tot, "rate": round(tot["agree"] / tot["n"], 4)
                        if tot["n"] else None,
                        "ci95": [round(lo, 4), round(hi, 4)]}}


def manipulated_gold_check(verified: list[dict]) -> dict:
    """Medido sobre la salida real: cambiar el gold de cada verificado a otro
    candidato y volver a decidir con su traza guardada → todos a cuarentena.
    Sin Qwen: la regla y el veredicto ya están en ``verify_trace``."""
    n = caught = 0
    for ep in verified:
        ids = [c["id"] for c in ep["candidates"]]
        if len(ids) < 2:
            continue
        bad = dict(ep)
        bad["answer"] = next(i for i in ids if i != ep["answer"])
        kind, _ = decide(bad, rule_verify(bad), ep["verify_trace"]["verifier"])
        n += 1
        caught += kind == "quarantine"
    return {"pass": bool(n and caught == n), "n_manipulated": n,
            "n_quarantined": caught}


def measure_gate(out: str, checkpoints_root: str = CHECKPOINTS_ROOT) -> dict:
    vm_path = os.path.join(out, "verify_manifest.json")
    vm = json.load(open(vm_path)) if os.path.exists(vm_path) else {}
    verified = _read_jsonl(os.path.join(out, "verified.jsonl"))
    quarantine = _read_jsonl(os.path.join(out, "quarantine.jsonl"))
    sample = _read_jsonl(os.path.join(out, "human_sample.jsonl"))
    funnel_path = os.path.join(out, "funnel.json")
    on_disk = json.load(open(funnel_path)) if os.path.exists(funnel_path) else None
    rebuilt = build_funnel(out, checkpoints_root)
    if on_disk is not None:
        _write_json(funnel_path, rebuilt)

    cells_in_sample: dict = {}
    for row in sample:
        k = f"{row['family']}/{row['lang']}/{row['variant_type']}"
        cells_in_sample[k] = cells_in_sample.get(k, 0) + 1
    reviewed = [r for r in sample if r.get("human_error") is not None]
    human = {
        "pass": None, "status": "awaiting-operator",
        "per_cell_declared": HUMAN_PER_CELL, "seed": HUMAN_SEED,
        "n_rows": len(sample), "cells": dict(sorted(cells_in_sample.items())),
        "n_reviewed": len(reviewed), "human_error": None,
        "file": os.path.relpath(os.path.join(out, "human_sample.jsonl"), ROOT),
    }
    gate = {
        "task": TASK,
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dir": os.path.relpath(os.path.abspath(out), ROOT),
        "verifier_version": VERIFIER_VERSION,
        "prompt_sha": prompt_sha(), "qwen_model": vm.get("qwen_model"),
        "verify_manifest_sha": _sha(vm_path),
        "command": vm.get("command"), "elapsed_s": vm.get("elapsed_s"),
        "checks": {
            "manipulated_gold_quarantined": manipulated_gold_check(verified),
            "funnel_from_manifests": {
                "pass": bool(on_disk is not None
                             and on_disk["stages"] == rebuilt["stages"]),
                "stages_total": {k: v.get("total")
                                 for k, v in rebuilt["stages"].items()},
                "consumed_train_manifests":
                    rebuilt["stages"]["consumed"]["train_manifests"],
            },
            "agreement_published": {
                "pass": bool(verified or quarantine),
                **_agreement_table(verified, quarantine),
            },
            "quarantine_is_artifact": {
                "pass": bool(vm.get("status") == "published"
                             and all(q.get("reasons") for q in quarantine)),
                "n_quarantine": len(quarantine),
                "reasons": vm.get("quarantine_reasons"),
            },
            "human_error_published": human,
        },
    }
    gate["pass"] = all(c["pass"] for c in gate["checks"].values()
                       if c.get("pass") is not None)
    gate["resolved"] = all(c.get("pass") is not None
                           for c in gate["checks"].values())
    os.makedirs(os.path.dirname(GATE_PATH), exist_ok=True)
    _write_json(GATE_PATH, gate)
    return gate


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m data.episode_verify")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="rule + separate verifier + sample + funnel")
    r.add_argument("--dir", required=True)
    r.add_argument("--batch", type=int, default=20)
    r.add_argument("--concurrency", type=int, default=2)
    r.add_argument("--limit", type=int, default=0,
                   help="verify only the first N episodes (smoke)")
    r.add_argument("--job-id", default="datagen")
    g = sub.add_parser("gate", help="measure the verification gate")
    g.add_argument("--dir", required=True)
    f = sub.add_parser("funnel", help="rebuild funnel.json from manifests")
    f.add_argument("--dir", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        m = run(args.dir, batch=args.batch, concurrency=args.concurrency,
                limit=args.limit, command=" ".join(sys.argv))
        print(f"verified={m['n_verified']} quarantine={m['n_quarantine']} "
              f"elapsed={m['elapsed_s']}s -> {args.dir}")
        return 0 if m["n_verified"] else 1
    if args.cmd == "gate":
        gate = measure_gate(args.dir)
        print(json.dumps({k: v.get("pass") for k, v in gate["checks"].items()},
                         sort_keys=True))
        return 0 if gate["pass"] else 1
    funnel = build_funnel(args.dir)
    _write_json(os.path.join(args.dir, "funnel.json"), funnel)
    print(json.dumps({k: v.get("total") for k, v in funnel["stages"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
