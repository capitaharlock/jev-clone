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

Throughput: Qwen-local (~5 tok/s) serves ONE sequential request at a
time (max 1 concurrent by construction). Bulk runs go through the
stopped ``episode-gen-pilot`` daemon job, never inside a turn::

    nice -n 10 python3 -m data.episode_gen run --n 2000 --seed 20260926 \\
        --prose qwen --teacher qwen --out artifacts/episodes-qwen/pilot-2k
    python3 -m data.episode_gen gate --dir artifacts/episodes-qwen/pilot-2k
"""

from __future__ import annotations

import argparse
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
PILOT_SEED = 20260926
GATE_PATH = os.path.join(ROOT, "artifacts", "gates", TASK, "gen.json")

MODEL = os.environ.get("QWEN_MODEL", "qwen3.6:27b-mlx")
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

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
                 timeout: int) -> str | None:
    """One sequential Ollama call; None on any failure (fallback path)."""
    body = json.dumps({
        "model": MODEL, "stream": False, "think": False,
        "messages": messages,
        "options": {"temperature": 0.7, "num_predict": num_predict},
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


def qwen_paraphrase(facts: str, lang: str, timeout: int = 120) -> str | None:
    """Ask Qwen to redact prose from OUR facts. Returns state prose or None.

    The facts (numbers, names, winner) are fixed by the rule; Qwen only
    rephrases. The caller verifies the rule-critical tokens survived.
    """
    prompt = (
        f"Rewrite these facts as 2-3 natural sentences in "
        f"{'Spanish' if lang == 'es' else 'English'}. "
        f"Keep EVERY number, name and place exactly as written. "
        f"Reply with ONLY the rewritten sentences.\n{facts}"
    )
    return _ollama_chat([{"role": "user", "content": prompt}],
                        num_predict=220, timeout=timeout)


def teacher_structured(state: str, question: str,
                       candidates: list[dict],
                       timeout: int = 120) -> dict:
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
                       num_predict=120, timeout=timeout)
    trace: dict = {"backend": "qwen-local", "confidence_discarded": True}
    if out:
        # Strip any "confidence"/"probability"/"%" the model may add.
        out = re.sub(r'"confidence"[^,}]*,?', "", out)
        out = re.sub(r'"probability"[^,}]*,?', "", out)
        out = re.sub(r"\d+\s*%", "", out)
        try:
            start, end = out.find("{"), out.rfind("}")
            got = json.loads(out[start:end + 1])
            choice, ev = got.get("choice"), got.get("evidence", "")
            norm_state = re.sub(r"\s+", " ", state.lower())
            if choice in ids and isinstance(ev, str) and ev.strip() \
                    and re.sub(r"\s+", " ", ev.lower()) in norm_state:
                trace.update({"choice": choice, "evidence": ev.strip()})
                return trace
            trace["reject"] = f"invalid choice/evidence: {out[:120]!r}"
        except (ValueError, AttributeError) as e:
            trace["reject"] = f"unparseable: {e}; {out[:120]!r}"
    else:
        trace["reject"] = "no teacher reply"
    return trace


# -- episode builders (rule fixes facts; prose local or Qwen) ---------------
def _base(family: str, lang: str, seed: int, idx: int,
          group: str) -> dict:
    return {
        "schema_version": EC.SCHEMA_VERSION,
        "family": family, "lang": lang, "generator_seed": seed,
        "generator_version": GENERATOR_VERSION, "variant_group": group,
        "id": f"ep-{group}-{idx:02d}",
    }


def _maybe_qwen(facts: str, local: str, lang: str,
                prose: str, tokens: list[str]) -> tuple[str, str]:
    """Use Qwen prose only if every rule-critical token survived in it."""
    if prose != "qwen":
        return local, "local-fallback"
    got = qwen_paraphrase(facts, lang)
    if got and all(t in got for t in tokens):
        return got, "qwen-local"
    return local, "local-fallback"


def build_comparison(lang: str, seed: int, idx: int, group: str,
                     rng: random.Random, flip: bool,
                     prose: str = "local") -> dict:
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
    state, origin = _maybe_qwen(local, local, lang, prose, tokens)
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
                   prose: str = "local") -> dict:
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
    state, origin = _maybe_qwen(local, local, lang, prose, tokens)
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
                     prose: str = "local") -> dict:
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
    state, origin = _maybe_qwen(local, local, lang, prose, [ent, loc])
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
                      prose: str = "local") -> dict:
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
    state, origin = _maybe_qwen(local, local, lang, prose, [traits])
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
                    prose: str = "local") -> dict:
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
    state, origin = _maybe_qwen(local, local, lang, prose, [who, grp])
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


def run(n: int, seed: int, prose: str = "local", teacher: str = "stub",
        out_dir: str = "", command: str = "") -> dict:
    """Generate ``n`` episodes. Manifest is written BEFORE generating.

    Cases come in base+counterfactual pairs sharing a ``variant_group``.
    Teacher trace is attached per episode; rejects land in rejects.jsonl
    with reasons, never dropped silently.
    """
    out = out_dir or os.path.join(ROOT, "artifacts", "episodes-qwen",
                                  f"seed-{seed}")
    os.makedirs(out, exist_ok=True)
    plan = declared_split(n, seed)
    manifest = {
        "task": TASK, "generator_version": GENERATOR_VERSION,
        "contract_task": EC.TASK, "schema_version": EC.SCHEMA_VERSION,
        "schema_sha": EC.schema_sha(), "seed": seed,
        "command": command or f"python3 -m data.episode_gen run --n {n} "
                              f"--seed {seed} --prose {prose} "
                              f"--teacher {teacher}",
        "status": "planned",
        "planned_split": {f"{f}/{lang}": c
                          for (f, lang), c in sorted(plan.items())},
        "qwen_model": MODEL,
    }
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    # Flatten plan into an ordered episode list: pairs per case.
    queue: list[tuple[str, str]] = []
    for (family, lang), count in sorted(plan.items()):
        queue += [(family, lang)] * count
    episodes: list[dict] = []
    rejects: list[dict] = []
    idx = 0
    case = 0
    while idx < n:
        family, lang = queue[idx]
        group = f"{family[:4]}-{lang}-c{case:04d}"
        for flip in (False, True):
            if idx >= n:
                break
            fam2, lang2 = queue[idx]
            g2 = group if (fam2, lang2) == (family, lang) else \
                f"{fam2[:4]}-{lang2}-c{case:04d}"
            ep = BUILDERS[fam2](lang2, seed, idx, g2,
                                _rng(seed, idx), flip, prose)
            if teacher == "qwen":
                trace = teacher_structured(
                    ep["state"], ep["question"], ep["candidates"])
                if "choice" in trace:
                    trace["gold_rule"] = ep["answer"]
                    trace["agree"] = trace["choice"] == ep["answer"]
                else:
                    # Rule wins on numeric families; otherwise reject loudly.
                    trace["gold_rule"] = ep["answer"]
                    if fam2 in (EC.COMPARISON, EC.PRIORITY):
                        trace["choice"] = ep["answer"]
                        trace["agree"] = True
                        trace["fallback"] = "rule-wins"
                    else:
                        rejects.append({"episode": ep, "reasons": [
                            f"teacher failed: {trace.get('reject')}"]})
                        idx += 1
                        continue
                ep["teacher_trace"] = trace
            else:
                ep["teacher_trace"] = {
                    "backend": "stub-local", "choice": ep["answer"],
                    "evidence": ep["evidence"], "gold_rule": ep["answer"],
                    "agree": True, "confidence_discarded": True,
                }
            reasons = EC.validate(ep)
            # No template markers, no dataset-id questions — belt and braces
            # on top of the validator (gate test 3 also measures it).
            for m in LABELGEN_MARKERS:
                if m in ep["state"] or m in ep["question"]:
                    reasons.append(f"labelgen template marker: {m!r}")
            if reasons:
                rejects.append({"episode": ep, "reasons": reasons})
            else:
                episodes.append(ep)
            idx += 1
        case += 1

    with open(os.path.join(out, "episodes.jsonl"), "w") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "rejects.jsonl"), "w") as fh:
        for rj in rejects:
            fh.write(json.dumps(rj, ensure_ascii=False) + "\n")
    realised: dict[str, int] = {}
    for ep in episodes:
        realised[f"{ep['family']}/{ep['lang']}"] = \
            realised.get(f"{ep['family']}/{ep['lang']}", 0) + 1
    manifest.update({
        "status": "published", "n_episodes": len(episodes),
        "n_rejects": len(rejects), "splits": realised,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return {"out": out, "n_episodes": len(episodes),
            "n_rejects": len(rejects), "splits": realised}


def load_episodes(out: str) -> list[dict]:
    path = os.path.join(out, "episodes.jsonl")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# -- gate: the three Verification tests, measured for real -------------------
def measure_gate(out: str, job_id: str = "episode-gen-pilot") -> dict:
    """Measure the task's Verification gate on a published directory.

    No invented numbers: every check runs over ``episodes.jsonl`` (and
    ``rejects.jsonl``). The volume check cannot be signed here — it is
    recorded as null/awaiting-operator-compute.
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
            "volume_ge_2000": {
                "pass": None, "status": "awaiting-operator-compute",
                "n_published": len(episodes), "required": PILOT_N,
                "job_id": job_id,
            },
        },
    }
    gate["pass"] = all(c.get("pass") for k, c in gate["checks"].items()
                       if k != "volume_ge_2000")
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
    g = sub.add_parser("gate", help="measure the verification gate")
    g.add_argument("--dir", required=True)
    g.add_argument("--job-id", default="episode-gen-pilot")
    sub.add_parser("probe", help="one tiny Qwen reachability call")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        if args.n > 25 and args.prose == "qwen":
            print("refusing: bulk Qwen runs go through the daemon job, "
                  "not a turn (compute embargo)", file=sys.stderr)
            return 2
        stats = run(n=args.n, seed=args.seed, prose=args.prose,
                    teacher=args.teacher, out_dir=args.out)
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
