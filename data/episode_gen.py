"""Episode generator (#T-episode-gen): Qwen writes PROSE, rules fix FACTS.

Replaces ``data.labelgen``'s template filling. Five families (ES+EN),
conforming to the ``episode-v1`` contract in ``data.episode_contract``
(which this module CONSUMES — it never redefines the schema).

Division of labour, non-negotiable:

* **Attributes travel in ``state`` (or in code catalogs).** Nothing is
  extracted from Qwen's world knowledge: catalogue facts live in the
  ``_CATALOG`` tables below, the seeded rule samples them, Qwen only
  paraphrases them into natural prose.
* **Gold by rule in the numeric/comparative families.**
  ``comparison_gold()`` / ``priority_gold()`` are pure deterministic
  functions of the sampled attributes; the generator cannot publish a
  gold the rule contradicts. Perturbing a state attribute flips the
  gold (gate test 2 proves it on real output).
* **Teacher = structured choice among valid option ids + evidence
  span**, stored in ``episode["teacher_trace"]``. Any confidence
  percentage the LLM writes is DISCARDED (``confidence_discarded``)
  and never stored as a probability.

Throughput: Qwen-local (27B, ~2-4 tok/s) is the bottleneck. Bulk runs
go through the ``datagen`` daemon job, never inside a turn — the
``EPISODE_GEN_BULK=1`` escape is set by that job and by nothing else.
``--concurrency`` overlaps the round-trips up to ``OLLAMA_NUM_PARALLEL``
without changing WHAT is generated (build plan is pure in seed+idx)::

    EPISODE_GEN_BULK=1 nice -n 10 python3 -m data.episode_gen run \\
        --n 2000 --seed 20260926 --prose qwen --teacher qwen \\
        --concurrency 4 --out artifacts/episodes-qwen/pilot-2k
    python3 -m data.episode_gen gate --dir artifacts/episodes-qwen/pilot-2k
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data import episode_contract as EC  # noqa: E402

GENERATOR_VERSION = "episode-gen-v1"
TASK = "T-episode-gen"
PILOT_N = 2000
#: Episodes published per chunk — the granularity of the progress line and
#: of what survives on disk if a long run is interrupted.
PUBLISH_CHUNK = 100
PILOT_SEED = 20260926
GATE_PATH = os.path.join(ROOT, "artifacts", "gates", TASK, "gen.json")

MODEL = os.environ.get("QWEN_MODEL", "qwen3.8:27b-mlx")
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

#: Temperatura del profesor. El generador MUESTREA (0,7) para que dos
#: tandas no escriban el mismo episodio; el verificador DECIDE en greedy
#: (0,0, `data.episode_verify`). Es un parámetro y no una constante
#: enterrada porque el mismo profesor se mide a las dos temperaturas
#: (#T-qwen38-ref): un control de permutación a 0,7 no puede separar la
#: sensibilidad al orden del ruido del muestreo, y a 0 sí.
TEMPERATURE = 0.7

#: Labelgen markers that must NEVER appear in an episode (gate test 3).
LABELGEN_MARKERS = ("Field report", "Cross-check against")


# -- declared split ------------------------------------------------------
def declared_split(n: int, seed: int) -> dict[tuple[str, str], int]:
    """Per-family/per-language quota, DECLARED before generating.

    ``n`` episodes over the 10 (family, lang) cells; the remainder goes
    to cells in seeded-shuffle order. Pure function of (n, seed).
    """
    cells = [(f, lang) for f in EC.FAMILIES for lang in EC.LANGS]
    base, rem = divmod(n, len(cells))
    order = cells[:]
    random.Random(f"{GENERATOR_VERSION}\x00split\x00{seed}").shuffle(order)
    plan = {c: base for c in cells}
    for c in order[:rem]:
        plan[c] += 1
    return plan


# -- gold rules (pure, seeded only through their inputs) ------------------
def comparison_gold(attrs: list[dict], criterion: str) -> int:
    """Index of the winner: argmin on the criterion attribute.

    ``criterion`` is ``"price"`` (cheapest) or ``"duration"`` (fastest).
    Ties break toward the lower index — explicit, deterministic.
    """
    key = "price_eur" if criterion == "price" else "duration_h"
    if criterion not in ("price", "duration"):
        raise ValueError(f"unknown comparison criterion {criterion!r}")
    best = min(range(len(attrs)), key=lambda i: (attrs[i][key], i))
    return best


def priority_gold(attrs: list[dict]) -> int:
    """Prioritise duration; on equal duration the lower price wins."""
    best = min(
        range(len(attrs)),
        key=lambda i: (attrs[i]["duration_h"], attrs[i]["price_eur"], i),
    )
    return best


# -- catalogs: the facts Qwen never invents --------------------------------
# Each entry: (state_bits, question_bits) per language. Numbers and
# winners are sampled by the seeded rule, never by the model.
_CMP_ITEMS = {
    "es": [
        ("tren", "autobús", "euros", "horas"),
        ("vuelo A", "vuelo B", "euros", "horas"),
        ("ruta norte", "ruta sur", "euros", "minutos"),
        ("curso de mañana", "curso del viernes", "plazas", "euros"),
    ],
    "en": [
        ("train", "bus", "euros", "hours"),
        ("flight A", "flight B", "euros", "hours"),
        ("northern route", "southern route", "euros", "minutes"),
        ("morning course", "Friday course", "seats", "euros"),
    ],
}

_EXT_FACTS = {
    "es": [
        ("el libro «El faro del norte»", "la estantería 3",
         "la estantería 5", "la sala de lectura"),
        ("el paquete 42B", "el almacén de Lisboa",
         "el almacén de Oporto", "la nave central"),
        ("la llave del archivo", "el cajón 2",
         "el cajón 4", "la oficina"),
    ],
    "en": [
        ("parcel 42B", "the Lisbon depot",
         "the Porto depot", "the main hall"),
        ("the archive key", "drawer 2",
         "drawer 4", "the office"),
        ("bus 12", "bay 3",
         "bay 7", "the station"),
    ],
}

_DESC_CATS = {
    "es": [
        ("Zampullín", "ave acuática pequeña que nada y apenas vuela",
         "tiene plumas, pico plano y nada en grupo",
         "caza palomas en vuelo con garras curvas"),
        ("Alcotán", "rapaz diurna que caza en vuelo",
         "caza palomas en vuelo con garras curvas",
         "nada en grupo con pico plano"),
        ("Avutarda", "ave terrestre grande que no nada",
         "corre por la llanura y no nada",
         "nada en grupo con pico plano"),
    ],
    "en": [
        ("Sodium", "soft metal, melts at 98 degrees, reacts with water",
         "melts below 100 degrees and conducts electricity well",
         "hard metal, inert in water"),
        ("Copper", "hard metal, melts above 1000 degrees, inert in water",
         "hard metal, inert in water",
         "melts below 100 degrees"),
        ("Sulfur", "brittle yellow non-metal, poor conductor",
         "brittle yellow powder, does not conduct",
         "melts below 100 degrees and conducts well"),
    ],
}

_INF_FORMS = {
    # (premise_tpl, question_tpl, gold_yes_tpl, gold_no_tpl): the LOGICAL
    # FORM fixes the gold; the rule picks the form, Qwen only paraphrases.
    "es": [
        ("Todos los {group} tienen {prop}. {who} es {member}.",
         "¿{who_q} tiene {prop_q}?",
         "Sí, porque todos los {group} lo tienen",
         "No, no hay información suficiente"),
        ("Ningún {group} nuevo recibe {prop}. {who} se hizo {member} en 2023.",
         "¿{who_q} tiene {prop_q}?",
         "Sí, porque todos los {group} lo tienen",
         "No, no hay información suficiente"),
        ("La mayoría de los {group} tienen {prop}, pero el registro de {who} falta.",
         "¿{who_q} tiene {prop_q}?",
         "Sí, con certeza",
         "No se puede saber por el registro"),
    ],
    "en": [
        ("Every {group} leaves with {prop}. {who} left at dawn.",
         "Did {who_q} leave with {prop_q}?",
         "Yes, every {group} does",
         "It cannot be determined from the log"),
        ("No new {group} gets {prop}. {who} joined in 2023.",
         "Did {who_q} leave with {prop_q}?",
         "Yes, every {group} does",
         "It cannot be determined from the log"),
        ("Most {group} leave with {prop}, but the log for {who} is missing.",
         "Did {who_q} leave with {prop_q}?",
         "Yes, it certainly did",
         "It cannot be determined from the log"),
    ],
}

_INF_FILL = {
    "es": [("socios del club", "llavero azul", "Marta",
             "socia del club desde 2023", "Marta", "llavero azul")],
    "en": [("delivery truck", "a full tank", "truck 7",
             "a delivery truck", "truck 7", "a full tank")],
}

_Q_TPL = {
    "attribute_comparison": {
        "es": {"price": "¿Cuál es la opción más barata para viajar?",
               "duration": "¿Cuál es la opción más rápida para viajar?"},
        "en": {"price": "Which option is the cheapest way to travel?",
               "duration": "Which option is the fastest way to travel?"},
    },
    "priority_decision": {
        "es": "¿Qué opción hay que elegir priorizando la duración, "
              "y a igualdad de duración el menor precio?",
        "en": "Prioritising duration, and with equal duration the lower "
              "price, which option should be chosen?",
    },
}


# -- Qwen: prose only -------------------------------------------------------
def _ollama_chat(messages: list[dict], num_predict: int,
                 timeout: int,
                 temperature: float = TEMPERATURE) -> str | None:
    """One sequential Ollama call; None on any failure (fallback path)."""
    body = json.dumps({
        "model": MODEL, "stream": False, "think": False,
        "messages": messages,
        "options": {"temperature": float(temperature),
                    "num_predict": num_predict},
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


def probe_qwen(timeout: int = 100) -> dict:
    """One tiny call proving the Qwen path is live (not a full episode)."""
    t0 = time.time()
    out = _ollama_chat(
        [{"role": "user",
          "content": "Reply with exactly: qwen-ok"}],
        num_predict=16, timeout=timeout)
    ok = out is not None and "qwen-ok" in out
    return {"reachable": ok, "model": MODEL,
            "latency_s": round(time.time() - t0, 1),
            "sample": (out or "")[:80]}


#: What a paraphrase must carry through untouched. The evidence span is
#: part of it: `episode-v1` demands `evidence` be a literal fragment of
#: `state`, so prose that rewrites the span away is prose that cannot be
#: published (measured 2026-09-27: it was 46 % of the first pilot chunk).
_KEEP_RULE = ("Keep EVERY number, name and place exactly as written, "
              "and repeat the KEEP fragment word for word inside your "
              "sentences. ")


def _paraphrase_prompt(lang: str, body: str, tail: str) -> str:
    return (f"Rewrite {body} as 2-3 natural sentences in "
            f"{'Spanish' if lang == 'es' else 'English'}. {_KEEP_RULE}{tail}")


def qwen_paraphrase(facts: str, lang: str, timeout: int = 120,
                    keep: str = "") -> str | None:
    """Ask Qwen to redact prose from OUR facts. Returns state prose or None.

    The facts (numbers, names, winner) are fixed by the rule; Qwen only
    rephrases. ``keep`` is the evidence span that must survive verbatim.
    The caller verifies it did — the prompt asks, the check decides.
    """
    prompt = _paraphrase_prompt(
        lang, "these facts",
        f"Reply with ONLY the rewritten sentences.\n{facts}"
        + (f"\nKEEP: {keep}" if keep else ""))
    return _ollama_chat([{"role": "user", "content": prompt}],
                        num_predict=220, timeout=timeout)


def teacher_structured(state: str, question: str,
                       candidates: list[dict],
                       timeout: int = 120,
                       temperature: float = TEMPERATURE) -> dict:
    """Ask the teacher for a STRUCTURED choice among valid option ids.

    Returns {"choice", "evidence", "backend", "confidence_discarded"}.
    Any confidence/percentage in the reply is dropped on the floor —
    never stored, never used as a probability.
    """
    ids = [c["id"] for c in candidates]
    opts = "\n".join(f"- {c['id']}: {c['text']}" for c in candidates)
    prompt = (
        f"Read the facts, then reply with ONLY one JSON object "
        f"{{\"choice\": <one of {ids}>, \"evidence\": <exact short span "
        f"copied from the facts>}}. No other keys, no prose.\n"
        f"Facts: {state}\nQuestion: {question}\nOptions:\n{opts}"
    )
    out = _ollama_chat([{"role": "user", "content": prompt}],
                       num_predict=120, timeout=timeout,
                       temperature=temperature)
    if out is None:
        return {"backend": "qwen-local", "model": MODEL, "confidence_discarded": True,
                "reject": "no teacher reply"}
    return _teacher_verdict(_first_json(out), state, ids, out)


def _strip_confidence(raw: str) -> str:
    """Drop any confidence/probability the model volunteered, on the floor."""
    raw = re.sub(r'"confidence"[^,}]*,?', "", raw)
    raw = re.sub(r'"probability"[^,}]*,?', "", raw)
    return re.sub(r"\d+\s*%", "", raw)


def _first_json(out: str) -> dict | None:
    """The first JSON object in ``out``, confidence keys already dropped."""
    out = _strip_confidence(out)
    start, end = out.find("{"), out.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        got = json.loads(out[start:end + 1])
    except ValueError:
        return None
    return got if isinstance(got, dict) else None


def _teacher_verdict(got: dict | None, state: str, ids: list,
                     raw: str) -> dict:
    """The ONE acceptance rule: a valid option id + a span really in the facts.

    Shared by the single-item and the batched teacher so a batched run can
    never be laxer than a sequential one.
    """
    trace: dict = {"backend": "qwen-local", "model": MODEL, "confidence_discarded": True}
    if got is None:
        trace["reject"] = f"unparseable: {raw[:120]!r}"
        return trace
    choice, ev = got.get("choice"), got.get("evidence", "")
    norm_state = re.sub(r"\s+", " ", state.lower())
    if choice in ids and isinstance(ev, str) and ev.strip() \
            and re.sub(r"\s+", " ", ev.lower()) in norm_state:
        trace.update({"choice": choice, "evidence": ev.strip()})
        return trace
    trace["reject"] = f"invalid choice/evidence: {raw[:120]!r}"
    return trace


# -- Qwen, MANY items per request -------------------------------------------
# Ollama costs ~19 s of fixed overhead per request against ~3 s of compute
# for one paraphrase (measured 2026-09-27 on qwen3.6:27b-mlx). Batching is
# the only lever that matters: 6.0 s/item at batch 12 vs 21 s/item alone.
def _numbered_lines(out: str | None, n: int) -> list:
    """Split a ``1. … 2. …`` reply into ``n`` slots; missing ones are None."""
    got: list = [None] * n
    if not out:
        return got
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s*[.)\-]\s*(.+)", line)
        if not m:
            continue
        k = int(m.group(1)) - 1
        if 0 <= k < n and got[k] is None:
            got[k] = m.group(2).strip()
    return got


def qwen_paraphrase_batch(facts: list, lang: str, timeout: int = 1800,
                          keeps: list | None = None) -> list:
    """Paraphrase MANY fact-sets in one request. ``None`` per missing item.

    Same contract as ``qwen_paraphrase``: Qwen only rephrases, ``keeps[i]``
    is the evidence span item ``i`` must carry through verbatim, and the
    caller still checks that it — and every rule-critical token — did.
    """
    if not facts:
        return []
    keeps = keeps or [""] * len(facts)
    if len(facts) == 1:
        return [qwen_paraphrase(facts[0], lang, keep=keeps[0])]
    listing = "\n".join(
        f"{i + 1}. {f}" + (f"\n   KEEP: {k}" if k else "")
        for i, (f, k) in enumerate(zip(facts, keeps)))
    prompt = _paraphrase_prompt(
        lang, "EACH numbered fact-set below",
        f"Reply with ONLY one line per item, in the same order, each line "
        f"starting with its number and a period. "
        f"{len(facts)} lines, nothing else.\n{listing}")
    return _numbered_lines(
        _ollama_chat([{"role": "user", "content": prompt}],
                     num_predict=140 * len(facts), timeout=timeout),
        len(facts))


def teacher_structured_batch(cases: list, timeout: int = 1800) -> list:
    """Structured choice + evidence for MANY cases in one request.

    Every item is accepted by ``_teacher_verdict`` — the same rule as the
    single-item path — so a reply that drifts, skips an item or invents an
    id lands as a reject with its reason, never as a silent gold.
    """
    if not cases:
        return []
    if len(cases) == 1:
        c = cases[0]
        return [teacher_structured(c["state"], c["question"],
                                   c["candidates"], timeout)]
    blocks = []
    for i, c in enumerate(cases, 1):
        opts = "\n".join(f"  - {o['id']}: {o['text']}"
                          for o in c["candidates"])
        blocks.append(f"### Item {i}\nFacts: {c['state']}\n"
                      f"Question: {c['question']}\nOptions:\n{opts}")
    prompt = (
        f"For EACH of the {len(cases)} items below, pick one option id and "
        f"copy the exact short span from THAT item's Facts that justifies "
        f"it. Reply with ONLY one JSON object per line, in the same order, "
        f'each as {{"item": <n>, "choice": <an option id of that item>, '
        f'"evidence": <exact span from that item\'s Facts>}}. '
        f"{len(cases)} lines, no prose.\n\n" + "\n\n".join(blocks)
    )
    out = _ollama_chat([{"role": "user", "content": prompt}],
                       num_predict=90 * len(cases), timeout=timeout)
    if out is None:
        return [{"backend": "qwen-local", "model": MODEL, "confidence_discarded": True,
                 "reject": "no teacher reply"} for _ in cases]
    by_item: dict = {}
    for line in _strip_confidence(out).splitlines():
        s, e = line.find("{"), line.rfind("}")
        if s < 0 or e < s:
            continue
        try:
            got = json.loads(line[s:e + 1])
        except ValueError:
            continue
        if isinstance(got, dict) and isinstance(got.get("item"), int):
            by_item.setdefault(got["item"], got)
    return [_teacher_verdict(by_item.get(i), c["state"],
                             [o["id"] for o in c["candidates"]], out)
            for i, c in enumerate(cases, 1)]


# -- episode builders (rule fixes facts; prose local or Qwen) ---------------
def _base(family: str, lang: str, seed: int, idx: int,
          group: str) -> dict:
    return {
        "schema_version": EC.SCHEMA_VERSION,
        "family": family, "lang": lang, "generator_seed": seed,
        "generator_version": GENERATOR_VERSION, "variant_group": group,
        "id": f"ep-{group}-{idx:02d}",
    }


def _maybe_qwen(facts: str, local: str, lang: str, prose: str,
                tokens: list[str], sink: list | None = None) -> tuple[str, str]:
    """Use Qwen prose only if every rule-critical token survived in it.

    With a ``sink`` the call is DEFERRED: the request is recorded and the
    local prose is returned as a placeholder, so the caller can put many
    paraphrases in one Qwen request (see ``_fill_prose``). One request per
    episode is what makes the pilot cost 18 h instead of 5 (measured:
    ~19 s of fixed per-request overhead against ~3 s of actual compute).
    """
    if prose != "qwen":
        return local, "local-fallback"
    if sink is not None:
        sink.append({"facts": facts, "lang": lang, "tokens": list(tokens),
                     "local": local})
        return local, "pending-qwen"
    got = qwen_paraphrase(facts, lang)
    if got and all(t in got for t in tokens):
        return got, "qwen-local"
    return local, "local-fallback"


def build_comparison(lang: str, seed: int, idx: int, group: str,
                     rng: random.Random, flip: bool,
                     prose: str = "local", sink: list | None = None) -> dict:
    a, b, unit, dur = rng.choice(_CMP_ITEMS[lang])
    p1 = rng.randint(15, 60)
    p2 = rng.randint(15, 60)
    if p2 == p1:
        p2 += 7
    d1 = rng.randint(1, 6)
    d2 = rng.randint(1, 6)
    if d2 == d1:
        d2 += 2
    criterion = rng.choice(["price", "duration"])
    attrs = [{"price_eur": p1, "duration_h": d1},
             {"price_eur": p2, "duration_h": d2}]
    if flip:  # counterfactual: perturb one attribute so the gold flips
        key = "price_eur" if criterion == "price" else "duration_h"
        w = comparison_gold(attrs, criterion)
        lo = min(a[key] for a in attrs)
        attrs[1 - w][key] = lo - (3 if key == "price_eur" else 1)
        # refresh locals for prose
        p1, p2 = attrs[0]["price_eur"], attrs[1]["price_eur"]
        d1, d2 = attrs[0]["duration_h"], attrs[1]["duration_h"]
    gold = comparison_gold(attrs, criterion)
    if lang == "es":
        local = (f"El billete de {a} cuesta {p1} euros y tarda {d1} {dur}. "
                 f"El billete de {b} cuesta {p2} euros y tarda {d2} {dur}. "
                 "Ambos salen cada hora.")
        t1, t2 = f"{a}, a {p1} euros", f"{b}, a {p2} euros"
    else:
        local = (f"The {a} costs {p1} {unit} and takes {d1} {dur}. "
                 f"The {b} costs {p2} {unit} and takes {d2} {dur}. "
                 "Both leave every hour.")
        t1, t2 = f"{a}, {p1} {unit}", f"{b}, {p2} {unit}"
    tokens = [str(p1), str(p2), a, b]
    state, origin = _maybe_qwen(local, local, lang, prose, tokens, sink)
    question = _Q_TPL["attribute_comparison"][lang][criterion]
    cands = [{"id": "c1", "text": t1}, {"id": "c2", "text": t2}]
    key = "price_eur" if criterion == "price" else "duration_h"
    ev_val = attrs[gold][key]
    evidence = next(
        s for s in re.split(r"\. ", state)
        if str(ev_val) in s)
    ep = _base(EC.COMPARISON, lang, seed, idx, group)
    ep.update({
        "state": state, "question": question, "candidates": cands,
        "answer": cands[gold]["id"], "evidence": evidence,
        "origin": origin,
        "rule_trace": {"attrs": attrs, "criterion": criterion,
                       "rule": "argmin"},
    })
    return ep


def build_priority(lang: str, seed: int, idx: int, group: str,
                   rng: random.Random, flip: bool,
                   prose: str = "local", sink: list | None = None) -> dict:
    d1 = rng.randint(1, 4)
    d2 = rng.randint(1, 4)
    if not flip and d2 == d1:
        d2 += 1
    if flip:
        d1 = d2 = rng.randint(1, 4)  # counterfactual: tie → price decides
    p1 = rng.randint(100, 200)
    p2 = rng.randint(100, 200)
    if p2 == p1:
        p2 += 30
    attrs = [{"price_eur": p1, "duration_h": d1},
             {"price_eur": p2, "duration_h": d2}]
    gold = priority_gold(attrs)
    if lang == "es":
        local = (f"El vuelo A dura {d1} horas y cuesta {p1} euros. "
                 f"El vuelo B dura {d2} horas y cuesta {p2} euros. "
                 "Ambos llegan el mismo día.")
        t1, t2 = f"El vuelo A: {d1} horas, {p1} euros", \
            f"El vuelo B: {d2} horas, {p2} euros"
    else:
        local = (f"Offer X takes {d1} hours and costs {p1} euros. "
                 f"Offer Y takes {d2} hours and costs {p2} euros. "
                 "Both start Monday.")
        t1, t2 = f"Offer X: {d1} hours, {p1} euros", \
            f"Offer Y: {d2} hours, {p2} euros"
    tokens = [str(p1), str(p2), str(d1), str(d2)]
    state, origin = _maybe_qwen(local, local, lang, prose, tokens, sink)
    question = _Q_TPL["priority_decision"][lang]
    cands = [{"id": "c1", "text": t1}, {"id": "c2", "text": t2}]
    evidence = next(
        s for s in re.split(r"\. ", state)
        if str(attrs[gold]["duration_h"]) in s)
    ep = _base(EC.PRIORITY, lang, seed, idx, group)
    ep.update({
        "state": state, "question": question, "candidates": cands,
        "answer": cands[gold]["id"], "evidence": evidence,
        "origin": origin,
        "rule_trace": {"attrs": attrs, "rule": "duration-then-price"},
    })
    return ep


def build_extraction(lang: str, seed: int, idx: int, group: str,
                     rng: random.Random, flip: bool,
                     prose: str = "local", sink: list | None = None) -> dict:
    ent, loc_a, loc_b, place = rng.choice(_EXT_FACTS[lang])
    loc = loc_b if flip else loc_a  # counterfactual moves the object
    other = loc_a if flip else loc_b
    variants = {
        "es": [
            f"{ent[0].upper() + ent[1:]} está en {loc} de {place}. "
            f"{other[0].upper() + other[1:]} solo guarda material antiguo.",
            f"Hay que buscar {ent} en {loc} de {place}; "
            f"en {other} no queda nada de eso.",
        ],
        "en": [
            f"{ent[0].upper() + ent[1:]} is at {loc} of {place}. "
            f"{other[0].upper() + other[1:]} holds only old stock.",
            f"Look for {ent} at {loc} of {place}; "
            f"there is none left at {other}.",
        ],
    }
    local = rng.choice(variants[lang])
    state, origin = _maybe_qwen(local, local, lang, prose, [ent, loc], sink)
    q_tpl = {
        "es": f"¿Dónde hay que buscar {ent}?",
        "en": f"Where should one look for {ent}?",
    }
    cands = [{"id": "c1", "text": loc_a}, {"id": "c2", "text": loc_b}]
    gold = "c2" if flip else "c1"
    ep = _base(EC.EXTRACTION, lang, seed, idx, group)
    ep.update({
        "state": state, "question": q_tpl[lang], "candidates": cands,
        "answer": gold, "evidence": loc,
        "origin": origin,
        "rule_trace": {"fact": loc, "rule": "explicit-location"},
    })
    return ep


def build_description(lang: str, seed: int, idx: int, group: str,
                      rng: random.Random, flip: bool,
                      prose: str = "local", sink: list | None = None) -> dict:
    cats = _DESC_CATS[lang]
    gi = rng.randrange(len(cats))
    if flip:
        gi = (gi + 1) % len(cats)  # counterfactual: other traits, other cat
    name, defi, traits, _alt = cats[gi]
    obs = {
        "es": f"El ejemplar observado {traits}. Pesa poco y se mueve de día.",
        "en": f"The observed sample {traits}. It is light and active by day.",
    }
    local = obs[lang]
    state, origin = _maybe_qwen(local, local, lang, prose, [traits], sink)
    cands = [{"id": f"c{i + 1}", "text": f"{n}: {d}"}
             for i, (n, d, _t, _a) in enumerate(cats)]
    q = {"es": "¿A qué categoría pertenece lo observado?",
         "en": "Which category best matches the observation?"}[lang]
    ep = _base(EC.DESCRIPTION, lang, seed, idx, group)
    ep.update({
        "state": state, "question": q, "candidates": cands,
        "answer": f"c{gi + 1}", "evidence": traits,
        "origin": origin,
        "rule_trace": {"category": name, "rule": "trait-match"},
    })
    return ep


def build_inference(lang: str, seed: int, idx: int, group: str,
                    rng: random.Random, flip: bool,
                    prose: str = "local", sink: list | None = None) -> dict:
    forms = _INF_FORMS[lang]
    fi = 0 if not flip else (1 if rng.random() < 0.5 else 2)
    ptpl, qtpl, yes_t, no_t = forms[fi]
    fill = rng.choice(_INF_FILL[lang])
    grp, prop, who, member, who_q, prop_q = fill
    premise = ptpl.format(group=grp, prop=prop, who=who, member=member,
                          who_q=who_q, prop_q=prop_q)
    tails = {
        "es": " El registro se revisó esta mañana.",
        "en": " The log was reviewed this morning.",
    }
    local = premise + tails[lang]
    state, origin = _maybe_qwen(local, local, lang, prose, [who, grp], sink)
    question = qtpl.format(who_q=who_q, prop_q=prop_q)
    # Logical form fixes the gold: universal-affirmative → yes (c1);
    # negated or hedged premise → insufficient/negative (c2).
    gold = "c1" if fi == 0 else "c2"
    cands = [{"id": "c1", "text": yes_t.format(group=grp)},
             {"id": "c2", "text": no_t.format(group=grp)}]
    # Evidence: a fragment of the state that justifies the gold.
    frag = premise.split(". ")[0]
    ep = _base(EC.INFERENCE, lang, seed, idx, group)
    ep.update({
        "state": state, "question": question, "candidates": cands,
        "answer": gold, "evidence": frag,
        "origin": origin,
        "rule_trace": {"form": fi, "rule": "quantifier-form"},
    })
    return ep


BUILDERS = {
    EC.EXTRACTION: build_extraction,
    EC.COMPARISON: build_comparison,
    EC.DESCRIPTION: build_description,
    EC.INFERENCE: build_inference,
    EC.PRIORITY: build_priority,
}


# -- run --------------------------------------------------------------------
def _rng(seed: int, idx: int) -> random.Random:
    return random.Random(f"{GENERATOR_VERSION}\x00{seed}\x00{idx}")


def _work_items(queue: list[tuple[str, str]], n: int) -> list[tuple]:
    """The deterministic build plan: ``(idx, family, lang, group, flip)``.

    Pure function of the queue — pairs of base+counterfactual share a
    ``variant_group``. Extracted so the generation loop can run the Qwen
    round-trips concurrently without touching what gets generated.
    """
    items: list[tuple] = []
    idx = case = 0
    while idx < n:
        family, lang = queue[idx]
        group = f"{family[:4]}-{lang}-c{case:04d}"
        for flip in (False, True):
            if idx >= n:
                break
            fam2, lang2 = queue[idx]
            g2 = group if (fam2, lang2) == (family, lang) else \
                f"{fam2[:4]}-{lang2}-c{case:04d}"
            items.append((idx, fam2, lang2, g2, flip))
            idx += 1
        case += 1
    return items


def _build_episode(item: tuple, seed: int, prose: str,
                   sink: list | None = None) -> dict:
    """Build ONE episode. Pure in ``(seed, idx, flip)``.

    With ``sink`` the Qwen paraphrase is deferred (see ``_maybe_qwen``);
    the episode comes back carrying local prose and ``origin`` set to
    ``pending-qwen`` until ``_fill_prose`` resolves it.
    """
    idx, family, lang, group, flip = item
    return BUILDERS[family](lang, seed, idx, group, _rng(seed, idx),
                            flip, prose, sink)


def _kept(text: str, tokens: list) -> bool:
    """Did every rule-critical token survive the paraphrase?

    Compared with the contract's own normalisation (lowercase, collapsed
    spaces), because the builders capitalise the first word of the prose:
    an exact-case check made the extraction family fall back to local
    prose 100 % of the time and nobody noticed — Qwen never wrote it.
    """
    return all(EC.evidence_is_fragment(t, text) for t in tokens)


def _tick(phase: str, done: int, total: int, t0: float, items: int) -> None:
    """One line per finished Qwen request — the run's only honest pulse."""
    el = time.time() - t0
    rate = items / el if el else 0.0
    print(f"  {phase} {done}/{total} reqs · {items} items · "
          f"{rate:.3f} item/s", flush=True)


def _fill_prose(eps: list, sinks: list, batch: int, concurrency: int) -> None:
    """Resolve the deferred paraphrases in place, batched per language.

    Qwen prose is kept only when every rule-critical token AND the
    evidence span survived it; otherwise the episode keeps the local
    prose. Losing the span is not a rejected episode — it is an episode
    published with the prose the rule wrote. Which one each episode
    ended up with is published per episode in ``origin`` and counted in
    the manifest's ``prose_origins``.
    """
    groups: dict = {}
    for i, sk in enumerate(sinks):
        if sk:
            groups.setdefault(sk[0]["lang"], []).append(i)
    jobs = [(lang, idxs[s:s + batch])
            for lang, idxs in sorted(groups.items())
            for s in range(0, len(idxs), batch)]

    def one(job):
        lang, idxs = job
        return idxs, qwen_paraphrase_batch(
            [sinks[i][0]["facts"] for i in idxs], lang,
            keeps=[eps[i]["evidence"] for i in idxs])

    t0, seen = time.time(), 0
    with cf.ThreadPoolExecutor(max(1, concurrency)) as ex:
        for k, (idxs, got) in enumerate(ex.map(one, jobs), 1):
            seen += len(idxs)
            _tick("prose", k, len(jobs), t0, seen)
            for i, text in zip(idxs, got):
                req = sinks[i][0]
                # The evidence span is checked with the CONTRACT's own
                # normalisation, so prose this accepts is prose the
                # validator accepts — no episode is lost to a stray space.
                ok = bool(text) and _kept(text, req["tokens"]) \
                    and EC.evidence_is_fragment(eps[i]["evidence"], text)
                eps[i]["state"] = text if ok else req["local"]
                eps[i]["origin"] = "qwen-local" if ok else "local-fallback"


def _teacher_traces(eps: list, teacher: str, batch: int,
                    concurrency: int) -> list:
    """One trace per episode — batched Qwen, or ``None`` for the stub path."""
    if teacher != "qwen":
        return [None] * len(eps)
    out: list = [None] * len(eps)
    jobs = [list(range(s, min(s + batch, len(eps))))
            for s in range(0, len(eps), batch)]

    def one(idxs):
        return idxs, teacher_structured_batch(
            [{"state": eps[i]["state"], "question": eps[i]["question"],
              "candidates": eps[i]["candidates"]} for i in idxs])

    t0, seen = time.time(), 0
    with cf.ThreadPoolExecutor(max(1, concurrency)) as ex:
        for k, (idxs, traces) in enumerate(ex.map(one, jobs), 1):
            seen += len(idxs)
            _tick("teacher", k, len(jobs), t0, seen)
            for i, tr in zip(idxs, traces):
                out[i] = tr
    return out


def _finish_episode(ep: dict, trace: dict | None) -> tuple:
    """Attach the teacher trace and validate. ``("ok"|"reject", payload)``."""
    if trace is not None:
        if "choice" in trace:
            trace["gold_rule"] = ep["answer"]
            trace["agree"] = trace["choice"] == ep["answer"]
        else:
            # Rule wins on numeric families; otherwise reject loudly.
            trace["gold_rule"] = ep["answer"]
            if ep["family"] in (EC.COMPARISON, EC.PRIORITY):
                trace["choice"] = ep["answer"]
                trace["agree"] = True
                trace["fallback"] = "rule-wins"
            else:
                return "reject", {"episode": ep, "reasons": [
                    f"teacher failed: {trace.get('reject')}"]}
        ep["teacher_trace"] = trace
    else:
        ep["teacher_trace"] = {
            "backend": "stub-local", "choice": ep["answer"],
            "evidence": ep["evidence"], "gold_rule": ep["answer"],
            "agree": True, "confidence_discarded": True,
        }
    reasons = EC.validate(ep)
    # No template markers, no dataset-id questions — belt and braces on top
    # of the validator (gate test 3 also measures it).
    for m in LABELGEN_MARKERS:
        if m in ep["state"] or m in ep["question"]:
            reasons.append(f"labelgen template marker: {m!r}")
    return ("reject", {"episode": ep, "reasons": reasons}) if reasons \
        else ("ok", ep)


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(n: int, seed: int, prose: str = "local", teacher: str = "stub",
        out_dir: str = "", command: str = "", concurrency: int = 1,
        batch: int = 1, resume: bool = False) -> dict:
    """Generate ``n`` episodes. Manifest is written BEFORE generating.

    Cases come in base+counterfactual pairs sharing a ``variant_group``.
    Teacher trace is attached per episode; rejects land in rejects.jsonl
    with reasons, never dropped silently.

    ``resume=True`` continues a run that died in ``out_dir``: the build
    plan is a pure function of ``(n, seed)``, so with the SAME ``n`` and
    ``seed`` the items already on disk (episodes + rejects, one line
    each) are skipped and the files are opened in append mode. A
    different ``n`` or ``seed`` is refused: it would be another plan. Without it a relaunch overwrites what the
    previous run published (measured 2026-09-27: 1 394 episodes lost
    otherwise).
    """
    out = out_dir or os.path.join(ROOT, "artifacts", "episodes-qwen",
                                  f"seed-{seed}")
    os.makedirs(out, exist_ok=True)
    plan = declared_split(n, seed)
    manifest = {
        "task": TASK, "generator_version": GENERATOR_VERSION,
        "contract_task": EC.TASK, "schema_version": EC.SCHEMA_VERSION,
        "schema_sha": EC.schema_sha(), "seed": seed, "n": n,
        "command": command or f"python3 -m data.episode_gen run --n {n} "
                              f"--seed {seed} --prose {prose} "
                              f"--teacher {teacher}",
        "status": "planned",
        "planned_split": {f"{f}/{lang}": c
                          for (f, lang), c in sorted(plan.items())},
        "qwen_model": MODEL,
        "prose_backend": prose, "teacher_backend": teacher,
    }
    ep_path = os.path.join(out, "episodes.jsonl")
    rj_path = os.path.join(out, "rejects.jsonl")
    n_episodes = n_rejects = 0
    realised: dict[str, int] = {}
    origins: dict[str, int] = {}
    start_at = 0
    if resume and os.path.exists(ep_path):
        mpath = os.path.join(out, "manifest.json")
        prev = json.load(open(mpath)) if os.path.exists(mpath) else {}
        if prev.get("n", n) != n or prev.get("seed", seed) != seed:
            raise ValueError(
                f"resume needs the same plan: on disk n={prev.get('n')} "
                f"seed={prev.get('seed')}, asked n={n} seed={seed}")
        prev_eps = _read_jsonl(ep_path)
        prev_rj = _read_jsonl(rj_path) if os.path.exists(rj_path) else []
        start_at = len(prev_eps) + len(prev_rj)
        n_episodes, n_rejects = len(prev_eps), len(prev_rj)
        for payload in prev_eps:
            key = f"{payload['family']}/{payload['lang']}"
            realised[key] = realised.get(key, 0) + 1
            org = payload.get("origin", "unknown")
            origins[org] = origins.get(org, 0) + 1
        manifest["resumed_from"] = start_at
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    # Flatten plan into an ordered episode list: pairs per case.
    queue: list[tuple[str, str]] = []
    for (family, lang), count in sorted(plan.items()):
        queue += [(family, lang)] * count
    items = _work_items(queue, n)

    t0 = time.time()
    every = max(1, min(25, n // 20))

    def _emit(fh_ep, fh_rj, done: int, res: tuple) -> None:
        nonlocal n_episodes, n_rejects
        kind, payload = res
        if kind == "ok":
            fh_ep.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fh_ep.flush()
            n_episodes += 1
            key = f"{payload['family']}/{payload['lang']}"
            realised[key] = realised.get(key, 0) + 1
            org = payload.get("origin", "unknown")
            origins[org] = origins.get(org, 0) + 1
        else:
            fh_rj.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fh_rj.flush()
            n_rejects += 1
        if done % every == 0 or done == n:
            el = time.time() - t0
            rate = (done - start_at) / el if el else 0.0
            eta = (n - done) / rate if rate else float("inf")
            print(f"progress {done}/{n} ok={n_episodes} rej={n_rejects} "
                  f"rate={rate:.3f} ep/s eta={eta / 3600:.2f} h",
                  flush=True)

    # Three phases per chunk: build (pure, no network) → Qwen prose and
    # Qwen teacher in BATCHED requests → validate and publish. Chunking
    # keeps episodes.jsonl growing on disk while the run is alive, so the
    # progress line below is a measurement and not a promise.
    done = start_at
    mode = "a" if start_at else "w"
    with open(ep_path, mode) as fh_ep, open(rj_path, mode) as fh_rj:
        for start in range(start_at, len(items), PUBLISH_CHUNK):
            part = items[start:start + PUBLISH_CHUNK]
            sinks = [[] for _ in part]
            eps = [_build_episode(it, seed, prose, sk)
                   for it, sk in zip(part, sinks)]
            if prose == "qwen":
                _fill_prose(eps, sinks, batch, concurrency)
            traces = _teacher_traces(eps, teacher, batch, concurrency)
            for ep, tr in zip(eps, traces):
                done += 1
                _emit(fh_ep, fh_rj, done, _finish_episode(ep, tr))

    manifest.update({
        "status": "published", "n_episodes": n_episodes,
        "n_rejects": n_rejects, "splits": realised,
        "concurrency": concurrency, "batch": batch,
        "prose_origins": origins,
        "elapsed_s": round(time.time() - t0, 1),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return {"out": out, "n_episodes": n_episodes,
            "n_rejects": n_rejects, "splits": realised,
            "prose_origins": origins}


def load_episodes(out: str) -> list[dict]:
    path = os.path.join(out, "episodes.jsonl")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# -- gate: the three Verification tests, measured for real -------------------
def _volume_check(out: str, n_published: int, job_id: str) -> dict:
    """The volume box, signed from the published directory or not at all.

    C7: no ``pass: true`` without a measured file in the gate's own
    directory. The figure, the seed, the generator version and the
    manifest digest all come from ``manifest.json`` next to the
    episodes — never from an argument, never from a guess.
    """
    mpath = os.path.join(out, "manifest.json")
    if not os.path.exists(mpath):
        return {"pass": None, "status": "awaiting-operator-compute",
                "n_published": n_published, "required": PILOT_N,
                "job_id": job_id}
    raw = open(mpath, "rb").read()
    manifest = json.loads(raw)
    if manifest.get("status") != "published":
        return {"pass": None, "status": "awaiting-operator-compute",
                "n_published": n_published, "required": PILOT_N,
                "manifest_status": manifest.get("status"), "job_id": job_id}
    return {
        "pass": bool(n_published >= PILOT_N), "status": "measured",
        "n_published": n_published, "required": PILOT_N,
        "dir": os.path.relpath(out, ROOT),
        "manifest_sha": hashlib.sha256(raw).hexdigest(),
        "manifest_n_episodes": manifest.get("n_episodes"),
        "n_rejects": manifest.get("n_rejects"),
        "seed": manifest.get("seed"),
        "generator_version": manifest.get("generator_version"),
        "prose_backend": manifest.get("prose_backend"),
        "prose_origins": manifest.get("prose_origins"),
        "elapsed_s": manifest.get("elapsed_s"),
        "job_id": job_id,
    }


def measure_gate(out: str, job_id: str = "episode-gen-pilot") -> dict:
    """Measure the task's Verification gate on a published directory.

    No invented numbers: every check runs over ``episodes.jsonl`` (and
    ``rejects.jsonl``). The volume check cannot be signed here — it is
    The volume box is signed from ``manifest.json`` when the directory
    is published, and stays ``null``/awaiting-operator-compute when it
    is not.
    """
    episodes = load_episodes(out)
    rejects: list[dict] = []
    rpath = os.path.join(out, "rejects.jsonl")
    if os.path.exists(rpath):
        with open(rpath, encoding="utf-8") as fh:
            rejects = [json.loads(line) for line in fh if line.strip()]

    # 1. 100% validator pass; rejects carry reasons.
    per = EC.batch_validate(episodes)
    rejects_ok = all(
        isinstance(r.get("reasons"), list) and r["reasons"]
        for r in rejects)
    t1 = {"pass": bool(per["n_invalid"] == 0 and rejects_ok),
          "n_valid": per["n_valid"], "n_invalid": per["n_invalid"],
          "n_rejects_recorded": len(rejects),
          "rejects_have_reasons": rejects_ok}

    # 2. Perturbing a state attribute flips the rule-computed gold.
    tested = flipped = 0
    for ep in episodes:
        rt = ep.get("rule_trace") or {}
        if ep["family"] == EC.COMPARISON and "attrs" in rt:
            attrs = [dict(a) for a in rt["attrs"]]
            crit = rt["criterion"]
            before = comparison_gold(attrs, crit)
            key = "price_eur" if crit == "price" else "duration_h"
            lo = min(a[key] for a in attrs)
            attrs[1 - before][key] = lo - 3
            after = comparison_gold(attrs, crit)
            tested += 1
            flipped += (after != before)
        elif ep["family"] == EC.PRIORITY and "attrs" in rt:
            attrs = [dict(a) for a in rt["attrs"]]
            before = priority_gold(attrs)
            attrs[1 - before]["duration_h"] = \
                attrs[before]["duration_h"] - 1
            after = priority_gold(attrs)
            tested += 1
            flipped += (after != before)
    t2 = {"pass": bool(tested > 0 and flipped == tested),
          "n_tested": tested, "n_flipped": flipped}

    # 3. No labelgen template marker, no dataset-id question.
    bad_marker = [ep.get("id") for ep in episodes
                  if any(m in ep.get("state", "") or m in ep.get("question", "")
                         for m in LABELGEN_MARKERS)]
    bad_q = [ep.get("id") for ep in episodes
             if EC.is_dataset_id_question(ep.get("question", ""))]
    t3 = {"pass": bool(not bad_marker and not bad_q and episodes),
          "n_checked": len(episodes),
          "with_marker": bad_marker, "dataset_id_questions": bad_q}

    gate = {
        "task": TASK, "measured_utc":
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dir": os.path.relpath(out, ROOT),
        "generator_version": GENERATOR_VERSION,
        "schema_sha": EC.schema_sha(),
        "checks": {
            "validator_pass_100pc": t1,
            "rule_gold_flips_on_perturb": t2,
            "no_template_marker_no_dataset_id": t3,
            "volume_ge_2000": _volume_check(out, len(episodes), job_id),
        },
    }
    # A check still awaiting compute is `null` and does not vote; every
    # check that HAS been measured does.
    gate["pass"] = all(c["pass"] for c in gate["checks"].values()
                       if c.get("pass") is not None)
    os.makedirs(os.path.dirname(GATE_PATH), exist_ok=True)
    with open(GATE_PATH, "w") as fh:
        json.dump(gate, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return gate


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m data.episode_gen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="generate + publish a batch")
    r.add_argument("--n", type=int, default=25)
    r.add_argument("--seed", type=int, default=PILOT_SEED)
    r.add_argument("--prose", choices=["local", "qwen"], default="local")
    r.add_argument("--teacher", choices=["stub", "qwen"], default="stub")
    r.add_argument("--out", default="")
    r.add_argument("--concurrency", type=int, default=1,
                   help="parallel Qwen round-trips (<= OLLAMA_NUM_PARALLEL)")
    r.add_argument("--batch", type=int, default=1,
                   help="items per Qwen request (1 = one request each)")
    r.add_argument("--resume", action="store_true",
                   help="continue a dead run in --out: skip what is on disk")
    g = sub.add_parser("gate", help="measure the verification gate")
    g.add_argument("--dir", required=True)
    g.add_argument("--job-id", default="episode-gen-pilot")
    sub.add_parser("probe", help="one tiny Qwen reachability call")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        if args.n > 25 and args.prose == "qwen" \
                and os.environ.get("EPISODE_GEN_BULK") != "1":
            print("refusing: bulk Qwen runs go through the daemon job, "
                  "not a turn (compute embargo). The `datagen` job sets "
                  "EPISODE_GEN_BULK=1.", file=sys.stderr)
            return 2
        stats = run(n=args.n, seed=args.seed, prose=args.prose,
                    teacher=args.teacher, out_dir=args.out,
                    concurrency=args.concurrency, batch=args.batch,
                    resume=args.resume)
        print(f"episodes={stats['n_episodes']} "
              f"rejects={stats['n_rejects']} -> {stats['out']}")
        return 0 if stats["n_episodes"] else 1
    if args.cmd == "gate":
        gate = measure_gate(args.dir, args.job_id)
        print(json.dumps(
            {k: v.get("pass") for k, v in gate["checks"].items()},
            sort_keys=True))
        return 0 if gate["pass"] else 1
    print(json.dumps(probe_qwen(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
