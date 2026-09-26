"""Validar la MECÁNICA sobreajustando 48 casos inequívocos (#T-ce-mechanics).

Paso 3 del piloto y el más barato de todos. Antes de generar 5.000
decisiones y antes de ocupar la GPU durante horas, comprobar que la
tubería **puede** aprender: ajustar 48 ejemplos inequívocos con el scorer
compartido de `model/ce_scorer.py` y exigir >95 % sobre esos mismos
ejemplos.

LO QUE ESTA CIFRA NO ES
----------------------

**No mide generalización.** Se entrena y se mide sobre el MISMO conjunto,
a propósito: es un test de tubería, no de aprendizaje. Un 100 % aquí no
predice nada sobre desarrollo, y citarlo como calidad del producto es un
error de lectura, no una interpretación optimista. La primera cifra con
significado externo es la de `#T-ce-finetune` contra el mismo checkpoint
sin ajustar en la batería de `#T-battery-dev`.

Lo que sí demuestra, y es lo que la fase 1 tardó semanas en descartar:
que los gradientes llegan, que el gold está donde el trainer cree, que el
tokenizado no corta el estado, que la máscara de candidatos no mezcla
filas de K distinto, y que el formato de hipótesis es EL MISMO en entreno
y en evaluación.

EL CONJUNTO, Y POR QUÉ ES VÁLIDO COMO PRUEBA
--------------------------------------------

Un sobreajuste sólo prueba algo si el checkpoint pelado NO acierta ya los
casos (punto 2 del gate). Por eso el conjunto se construye sobre las dos
familias que la medición sin entrenar de `#T-ce-scorer` resuelve peor
—comparación de atributos (3/6 y 1/6) y decisiones con prioridades (0/4 y
2/4)— y deja fuera extracción y descripciones, donde un NLI sin ajustar
ya llega (6/8–8/8). Los umbrales están escritos ANTES de medir:

* `OVERFIT_TARGET = 0.95` — el sobreajuste tiene que superarlo.
* `SET_VALIDITY_MAX = 0.60` — si el checkpoint pelado pasa de ahí, el
  conjunto NO vale y el gate falla pidiendo uno más difícil.
* `MIN_MARGIN = 0.35` — y la distancia entre las dos cifras tiene que ser
  esa, como mínimo, para que «claramente por debajo» signifique algo.

Los casos de comparación y prioridad se derivan de tablas de números
(`ITEM_SETS`), no se teclean: el gold lo calcula la REGLA (precio mínimo,
duración máxima, duración máxima con desempate por precio) y
`validate_set()` lo RECALCULA leyendo los números del texto renderizado
del candidato —el mismo que ve el modelo— y exige que el ganador sea
único. Un gold mal alineado no puede entrar por un error de tecleo, que es
exactamente el fallo que este paso busca. Los de inferencia y negación sí
están escritos a mano, con sus tres variantes (sí / no / no lo dice) por
escenario, y el artefacto publica cuántos golds están verificados por
regla y cuántos no.

CLI (el comando del gate, reproducible — el mismo que lleva el job parado
`ce-overfit-mechanics` del daemon, que arranca el operador cuando quiera):
    PYTHONPATH=. .venv-train/bin/python -m training.python.ce_overfit \\
        overfit --device cpu --seed 20260926
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.calib import wilson_interval  # noqa: E402
from model import ce_scorer as CE  # noqa: E402

TASK = "T-ce-mechanics"
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
GATE_PATH = GATE_DIR / "overfit.json"

#: El checkpoint del piloto para este paso: multilingüe (el conjunto es
#: ES+EN), barato, y SIN contaminación declarada — `modernbert-zeroshot-v2`
#: lleva BANKING77 en su mezcla y aquí no hace falta abrir ese brazo.
DEFAULT_WEIGHTS = "minilmv2-l6-mnli-xnli"

DEFAULT_SEED = 20260926
DEFAULT_EPOCHS = 60
DEFAULT_LR = 2e-5
DEFAULT_DECISIONS_PER_BATCH = 8
DEFAULT_EVAL_EVERY = 5

#: Umbrales del gate, escritos antes de la primera medición.
OVERFIT_TARGET = 0.95
SET_VALIDITY_MAX = 0.60
MIN_MARGIN = 0.35

#: Tolerancia de las comprobaciones de tubería que se hacen con pesos
#: reales sobre el dispositivo del entreno. No es `CE.PERMUTATION_TOL`
#: porque aquí el lote CAMBIA a propósito (es lo que se está probando) y
#: en MPS el orden de reducción depende de con quién comparte batch una
#: fila. Las versiones exactas de estas mismas comprobaciones viven en
#: `test_ce_overfit.py` con un scorer determinista.
BATCH_TOL = 1e-3

NEG_INF = float("-inf")


# -- el conjunto ---------------------------------------------------------

@dataclass(frozen=True)
class ItemSet:
    """Una tabla de candidatos con sus números. El gold sale de la REGLA."""

    key: str
    lang: str
    noun: str
    #: (nombre, precio en euros, duración en años)
    items: tuple[tuple[str, int, int], ...]


ITEM_SETS = (
    ItemSet("routers", "en", "router",
            (("Router Nimbus", 45, 3), ("Router Cirrus", 70, 6),
             ("Router Stratus", 55, 4))),
    ItemSet("kettles", "en", "kettle",
            (("Kettle Ember", 30, 2), ("Kettle Flint", 22, 5),
             ("Kettle Glow", 48, 3))),
    ItemSet("chairs", "en", "chair",
            (("Chair Pinewood", 90, 7), ("Chair Oakline", 120, 4),
             ("Chair Birchset", 75, 5))),
    ItemSet("drills", "en", "drill",
            (("Drill Hammerhead", 60, 3), ("Drill Falcon", 38, 8),
             ("Drill Wren", 99, 6))),
    ItemSet("awnings", "en", "awning",
            (("Awning Solano", 210, 9), ("Awning Levante", 260, 6),
             ("Awning Poniente", 185, 4))),
    ItemSet("printers", "en", "printer",
            (("Printer Atlas", 150, 4), ("Printer Basalt", 110, 7),
             ("Printer Cobalt", 130, 5), ("Printer Delta", 95, 3))),
    ItemSet("routers-es", "es", "router",
            (("Router Nimbo", 52, 4), ("Router Cirro", 78, 7),
             ("Router Estrato", 64, 2))),
    ItemSet("hervidores", "es", "hervidor",
            (("Hervidor Brasa", 34, 3), ("Hervidor Pedernal", 27, 6),
             ("Hervidor Rescoldo", 41, 2))),
    ItemSet("sillas", "es", "silla",
            (("Silla Pinar", 95, 8), ("Silla Robledo", 128, 5),
             ("Silla Abedul", 82, 3))),
    ItemSet("taladros", "es", "taladro",
            (("Taladro Martillo", 66, 4), ("Taladro Halcón", 43, 9),
             ("Taladro Chochín", 104, 7))),
    ItemSet("toldos", "es", "toldo",
            (("Toldo Solano", 230, 10), ("Toldo Levante", 275, 7),
             ("Toldo Poniente", 196, 5))),
    ItemSet("impresoras", "es", "impresora",
            (("Impresora Atlas", 160, 5), ("Impresora Basalto", 118, 8),
             ("Impresora Cobalto", 139, 6), ("Impresora Delta", 101, 4))),
)

#: Las tablas de la familia `priority_decision`. La mitad trae EMPATE en
#: duración —ahí manda el desempate por precio— y la otra mitad no, para
#: que acertar exija aplicar la regla y no una heurística fija.
PRIORITY_SETS = (
    ItemSet("batteries-tie", "en", "battery",
            (("Battery Kestrel", 12, 4), ("Battery Lynx", 30, 9),
             ("Battery Marten", 18, 9))),
    ItemSet("hoses-tie", "en", "hose",
            (("Hose Aqualine", 44, 6), ("Hose Rivulet", 29, 6),
             ("Hose Tidewell", 51, 3))),
    ItemSet("pumps-tie", "en", "pump",
            (("Pump Anvil", 310, 5), ("Pump Bellows", 275, 8),
             ("Pump Crucible", 240, 8), ("Pump Dross", 199, 2))),
    ItemSet("pilas-tie", "es", "pila",
            (("Pila Cernícalo", 14, 5), ("Pila Lince", 33, 10),
             ("Pila Marta", 21, 10))),
    ItemSet("mangueras-tie", "es", "manguera",
            (("Manguera Acequia", 47, 7), ("Manguera Regato", 31, 7),
             ("Manguera Marea", 56, 4))),
    ItemSet("bombas-tie", "es", "bomba",
            (("Bomba Yunque", 325, 6), ("Bomba Fuelle", 288, 9),
             ("Bomba Crisol", 252, 9), ("Bomba Escoria", 208, 3))),
    ItemSet("lamps-plain", "en", "lamp",
            (("Lamp Quartz", 40, 3), ("Lamp Granite", 65, 7),
             ("Lamp Slate", 52, 5))),
    ItemSet("fans-plain", "en", "fan",
            (("Fan Zephyr", 25, 2), ("Fan Gale", 58, 6),
             ("Fan Breeze", 39, 4))),
    ItemSet("scanners-plain", "en", "scanner",
            (("Scanner Onyx", 140, 3), ("Scanner Jasper", 175, 8),
             ("Scanner Flint", 120, 5), ("Scanner Agate", 155, 6))),
    ItemSet("lamparas-plain", "es", "lámpara",
            (("Lámpara Cuarzo", 44, 4), ("Lámpara Granito", 69, 8),
             ("Lámpara Pizarra", 57, 6))),
    ItemSet("ventiladores-plain", "es", "ventilador",
            (("Ventilador Céfiro", 28, 3), ("Ventilador Vendaval", 61, 7),
             ("Ventilador Brisa", 43, 5))),
    ItemSet("escaneres-plain", "es", "escáner",
            (("Escáner Ónice", 148, 4), ("Escáner Jaspe", 182, 9),
             ("Escáner Sílex", 127, 6), ("Escáner Ágata", 163, 7))),
)

#: Textos por idioma. Nada de esto se improvisa por caso: el formato de
#: premisa e hipótesis lo pone `model.ce_scorer`, y esto es sólo el
#: contenido del estado y la pregunta.
CATALOGUE_STATE = {
    "en": ("Catalogue: the shop lists these {noun}s this week and every "
           "one of them ships tomorrow."),
    "es": ("Catálogo: la tienda lista estos {noun}s esta semana y todos "
           "se envían mañana."),
}
PRIORITY_STATE = {
    "en": ("Purchasing rule: prioritise duration; if two options last the "
           "same, take the lower price."),
    "es": ("Norma de compras: prioriza la duración; si dos opciones duran "
           "lo mismo, elige la de menor precio."),
}
CHEAPER_Q = {"en": "Which {noun} is cheaper?",
             "es": "¿Qué {noun} es más barato?"}
LONGER_Q = {"en": "Which {noun} lasts longer?",
            "es": "¿Qué {noun} dura más?"}
PRIORITY_Q = {"en": "Which {noun} should purchasing buy?",
              "es": "¿Qué {noun} debe comprar el departamento?"}
ITEM_TEXT = {"en": "{name}: {price} euros, lasts {years} years",
             "es": "{name}: {price} euros, dura {years} años"}

#: Inferencia textual y negación: aquí sí, escritos a mano, y con las tres
#: variantes del mismo escenario para que una preferencia fija por un
#: texto («sí») no pueda acertar el grupo entero.
INFERENCE_CASES = (
    {"id": "backup-en-yes", "group": "backup-en", "lang": "en", "gold": "c1",
     "state": ("Operations log: the nightly backup ran on Sunday at 02:10 "
               "and finished without errors."),
     "question": "Did the nightly backup run on Sunday?"},
    {"id": "backup-en-no", "group": "backup-en", "lang": "en", "gold": "c2",
     "state": ("Operations log: the nightly backup did not run on Sunday; "
               "the job was disabled for maintenance."),
     "question": "Did the nightly backup run on Sunday?"},
    {"id": "backup-en-unstated", "group": "backup-en", "lang": "en",
     "gold": "c3",
     "state": ("Operations log: on Sunday the disks were replaced and the "
               "rack was relabelled."),
     "question": "Did the nightly backup run on Sunday?"},
    {"id": "appt-en-yes", "group": "appt-en", "lang": "en", "gold": "c1",
     "state": ("Clinic note: the patient called on Tuesday and cancelled "
               "the appointment."),
     "question": "Did the patient cancel the appointment?"},
    {"id": "appt-en-no", "group": "appt-en", "lang": "en", "gold": "c2",
     "state": ("Clinic note: the appointment was not cancelled by the "
               "patient; the clinic moved it because the doctor was ill."),
     "question": "Did the patient cancel the appointment?"},
    {"id": "appt-en-unstated", "group": "appt-en", "lang": "en", "gold": "c3",
     "state": ("Clinic note: the appointment was booked on Tuesday and a "
               "consulting room was assigned."),
     "question": "Did the patient cancel the appointment?"},
    {"id": "acta-es-si", "group": "acta-es", "lang": "es", "gold": "c1",
     "state": ("Registro de obra: el acta se firmó el viernes por la "
               "mañana ante los dos aparejadores."),
     "question": "¿Se firmó el acta el viernes?"},
    {"id": "acta-es-no", "group": "acta-es", "lang": "es", "gold": "c2",
     "state": ("Registro de obra: el acta no se firmó el viernes; la "
               "reunión se aplazó a la semana siguiente."),
     "question": "¿Se firmó el acta el viernes?"},
    {"id": "acta-es-nodice", "group": "acta-es", "lang": "es", "gold": "c3",
     "state": ("Registro de obra: el viernes se descargó el material y se "
               "montó el andamio."),
     "question": "¿Se firmó el acta el viernes?"},
    {"id": "importe-es-si", "group": "importe-es", "lang": "es",
     "gold": "c1",
     "state": ("Expediente: el importe se devolvió al cliente el día 3 por "
               "transferencia."),
     "question": "¿Se devolvió el importe al cliente?"},
    {"id": "importe-es-no", "group": "importe-es", "lang": "es",
     "gold": "c2",
     "state": ("Expediente: el importe no se devolvió al cliente; la "
               "reclamación se cerró sin abono."),
     "question": "¿Se devolvió el importe al cliente?"},
    {"id": "importe-es-nodice", "group": "importe-es", "lang": "es",
     "gold": "c3",
     "state": ("Expediente: el día 3 se revisó la factura y se adjuntó el "
               "albarán."),
     "question": "¿Se devolvió el importe al cliente?"},
)
INFERENCE_OPTIONS = {
    "en": (("c1", "Yes, it happened"), ("c2", "No, it did not happen"),
           ("c3", "The log does not say")),
    "es": (("c1", "Sí, ocurrió"), ("c2", "No, no ocurrió"),
           ("c3", "El registro no lo dice")),
}


def _candidates(iset: ItemSet) -> tuple[CE.Candidate, ...]:
    return tuple(
        CE.Candidate(f"c{i + 1}", ITEM_TEXT[iset.lang].format(
            name=name, price=price, years=years))
        for i, (name, price, years) in enumerate(iset.items))


def _argmin_unique(values: list[int]) -> int:
    """El índice del mínimo, y sólo si es ÚNICO."""
    lo = min(values)
    hits = [i for i, v in enumerate(values) if v == lo]
    if len(hits) != 1:
        raise ValueError(f"ambiguous minimum in {values}")
    return hits[0]


def _argmax_unique(values: list[int]) -> int:
    hi = max(values)
    hits = [i for i, v in enumerate(values) if v == hi]
    if len(hits) != 1:
        raise ValueError(f"ambiguous maximum in {values}")
    return hits[0]


def _priority_winner(iset: ItemSet) -> int:
    """La regla del estado, aplicada: duración máxima, empate a precio."""
    years = [y for _, _, y in iset.items]
    best = max(years)
    tied = [i for i, y in enumerate(years) if y == best]
    if len(tied) == 1:
        return tied[0]
    prices = [iset.items[i][1] for i in tied]
    return tied[_argmin_unique(prices)]


def overfit_decisions() -> list[CE.Decision]:
    """Los 48 casos, con el gold calculado por la regla de cada familia."""
    out: list[CE.Decision] = []

    for iset in ITEM_SETS:
        cands = _candidates(iset)
        prices = [p for _, p, _ in iset.items]
        years = [y for _, _, y in iset.items]
        state = CATALOGUE_STATE[iset.lang].format(noun=iset.noun)
        for kind, question, gold_index in (
                ("cheap", CHEAPER_Q[iset.lang].format(noun=iset.noun),
                 _argmin_unique(prices)),
                ("long", LONGER_Q[iset.lang].format(noun=iset.noun),
                 _argmax_unique(years))):
            out.append(CE.Decision(
                state=state, question=question, candidates=cands,
                family=CE.COMPARISON, lang=iset.lang,
                gold=cands[gold_index].id,
                meta={"id": f"{iset.key}-{kind}", "group": iset.key,
                      "rule": ("unique minimum price" if kind == "cheap"
                               else "unique maximum duration")}))

    for iset in PRIORITY_SETS:
        cands = _candidates(iset)
        gold_index = _priority_winner(iset)
        out.append(CE.Decision(
            state=PRIORITY_STATE[iset.lang],
            question=PRIORITY_Q[iset.lang].format(noun=iset.noun),
            candidates=cands, family=CE.PRIORITY, lang=iset.lang,
            gold=cands[gold_index].id,
            meta={"id": f"{iset.key}-buy", "group": iset.key,
                  "rule": "maximum duration, ties broken by lower price"}))

    for case in INFERENCE_CASES:
        cands = tuple(CE.Candidate(cid, text)
                      for cid, text in INFERENCE_OPTIONS[case["lang"]])
        out.append(CE.Decision(
            state=case["state"], question=case["question"],
            candidates=cands, family=CE.INFERENCE, lang=case["lang"],
            gold=case["gold"],
            meta={"id": case["id"], "group": case["group"],
                  "rule": "hand-written entailment / negation / unstated"}))
    return out


#: Los números de un candidato, leídos del MISMO texto que el modelo ve.
#: `(precio, duración)`, en el orden en que `ITEM_TEXT` los escribe.
ITEM_NUMBERS = re.compile(r":\s*(\d+)\s+euros,\s*(?:lasts|dura)\s*(\d+)")


def numbers_from_text(text: str) -> tuple[int, int]:
    """`(precio, años)` del texto renderizado, o un error si no están."""
    hit = ITEM_NUMBERS.search(text)
    if not hit:
        raise ValueError(f"no price/duration in candidate text {text!r}")
    return int(hit.group(1)), int(hit.group(2))


#: La regla de cada caso derivado, aplicada sobre `(precio, años)`.
#: `meta["rule"]` es la clave, así que el texto de la regla que viaja al
#: artefacto y la función que decide el gold son la MISMA cosa.
RULES = {
    "unique minimum price":
        lambda rows: _argmin_unique([p for p, _ in rows]),
    "unique maximum duration":
        lambda rows: _argmax_unique([y for _, y in rows]),
    "maximum duration, ties broken by lower price":
        lambda rows: _priority_rule(rows),
}


def _priority_rule(rows: list[tuple[int, int]]) -> int:
    best = max(y for _, y in rows)
    tied = [i for i, (_, y) in enumerate(rows) if y == best]
    if len(tied) == 1:
        return tied[0]
    return tied[_argmin_unique([rows[i][0] for i in tied])]


def validate_set(decisions: list[CE.Decision]) -> dict:
    """Inequívocos, en el rango del gate, y con el gold donde toca.

    Para las familias derivadas recalcula el gold aplicando la regla a los
    números que lee del TEXTO RENDERIZADO del candidato —no de la tabla
    que lo generó— y exige que coincida con el gold de la decisión: un id
    tecleado a mano, un texto que no cuadra con su fila, o una regla mal
    aplicada, saltan aquí. Los casos de inferencia se escriben a mano por
    construcción y no tienen regla que recalcular; se cuentan aparte para
    que el artefacto diga cuántos golds están verificados por regla.
    """
    n = len(decisions)
    if not 32 <= n <= 64:
        raise ValueError(f"the gate asks for 32-64 cases, this set has {n}")
    seen_ids = set()
    rule_checked = 0
    for dec in decisions:
        case_id = dec.meta["id"]
        if case_id in seen_ids:
            raise ValueError(f"duplicate case id {case_id!r}")
        seen_ids.add(case_id)
        if dec.gold is None:
            raise ValueError(f"{case_id} has no gold")
        texts = [c.text for c in dec.candidates]
        if len(set(texts)) != len(texts):
            raise ValueError(f"{case_id} repeats a candidate text")
        rule = dec.meta.get("rule")
        if rule in RULES:
            rows = [numbers_from_text(t) for t in texts]
            want = dec.candidates[RULES[rule](rows)].id
            if want != dec.gold:
                raise ValueError(
                    f"{case_id}: gold is {dec.gold!r} but {rule!r} applied to "
                    f"the rendered candidate texts gives {want!r}")
            rule_checked += 1
    families = sorted({d.family for d in decisions})
    langs = sorted({d.lang for d in decisions})
    if set(langs) != set(CE.LANGS):
        raise ValueError(f"the set must cover {CE.LANGS}, covers {langs}")
    return {"n": n, "families": families, "langs": langs,
            "ks": sorted({d.k for d in decisions}),
            "chance": sum(1.0 / d.k for d in decisions) / n,
            "golds_recomputed_from_text": rule_checked,
            "golds_hand_written": n - rule_checked,
            "unambiguous": ("the gold of every derived case is recomputed by "
                            "its family rule from the numbers in the rendered "
                            "candidate text; a typed id cannot enter")}


def set_fingerprint(decisions: list[CE.Decision]) -> str:
    """Huella del conjunto: los pares RENDERIZADOS y sus golds.

    Cubre el formato y el contenido a la vez, y es lo que permite decir
    si dos ejecuciones del gate midieron lo mismo.
    """
    h = hashlib.sha256()
    for dec in decisions:
        h.update(dec.meta["id"].encode("utf-8"))
        h.update(b"\x00")
        h.update(str(dec.gold).encode("utf-8"))
        for premise, hypothesis in CE.render_pairs(dec):
            h.update(premise.encode("utf-8"))
            h.update(b"\x01")
            h.update(hypothesis.encode("utf-8"))
            h.update(b"\x02")
    return h.hexdigest()[:16]


# -- la tubería: un solo camino para entreno y evaluación ----------------

#: El aplanado del entreno ES el del evaluador: `model.ce_scorer` es el
#: único sitio del árbol donde se renderiza un par y donde se decide cómo
#: se tokeniza. `eval/ce_nograd.py` llama a estas dos mismas funciones, así
#: que no hay dos formatos de hipótesis que puedan separarse — lo
#: comprueba `test_train_and_eval_render_the_same_strings`.
flatten = CE.flatten_pairs
encode = CE.encode_pairs


def gold_indices(decisions: list[CE.Decision]) -> list[int]:
    """El índice del gold EN EL ORDEN EN QUE SE HAN RENDERIZADO los pares.

    El gold es un id opaco, y el orden de los candidatos cambia en cada
    época a propósito. Resolverlo aquí, contra la misma tupla que
    `flatten` acaba de recorrer, es lo que impide el desalineamiento.
    """
    out = []
    for dec in decisions:
        ids = [c.id for c in dec.candidates]
        if dec.gold not in ids:
            raise ValueError(f"gold {dec.gold!r} not among {ids}")
        out.append(ids.index(dec.gold))
    return out


def stack_scores(z, spans: list[tuple[int, int]]):
    """De `[N]` escalares planos a `[B, Kmax]` con `-inf` en el relleno.

    La máscara: las columnas que una fila no tiene valen `-inf`, así que
    el softmax les da exactamente 0 y no hay probabilidad que se escape a
    un candidato que no existe. Sin esto, una fila de K=3 en un lote con
    filas de K=4 competiría contra una cuarta columna ajena.
    """
    kmax = max(hi - lo for lo, hi in spans)
    out = z.new_full((len(spans), kmax), NEG_INF)
    for row, (lo, hi) in enumerate(spans):
        out[row, :hi - lo] = z[lo:hi]
    return out


def forward_logits(model, tokenizer, decisions: list[CE.Decision], device,
                   entail_index: int, max_length: int = CE.DEFAULT_MAX_LENGTH):
    """`[B, Kmax]` puntuaciones enmascaradas, y los golds alineados."""
    import torch
    pairs, spans = flatten(decisions)
    enc, report = encode(tokenizer, pairs, device, max_length)
    z = model(**enc).logits[:, entail_index]
    golds = torch.tensor(gold_indices(decisions), dtype=torch.long,
                         device=device)
    return stack_scores(z, spans), golds, report


def evaluate(model, tokenizer, decisions: list[CE.Decision], device,
             entail_index: int, batch: int = DEFAULT_DECISIONS_PER_BATCH,
             max_length: int = CE.DEFAULT_MAX_LENGTH) -> dict:
    """Acierto sobre las decisiones dadas, por el MISMO camino del entreno."""
    import torch
    was_training = model.training
    model.eval()
    rows = []
    total_nll = 0.0
    with torch.no_grad():
        for start in range(0, len(decisions), batch):
            chunk = decisions[start:start + batch]
            scores, golds, _ = forward_logits(model, tokenizer, chunk, device,
                                              entail_index, max_length)
            probs = torch.softmax(scores.float(), dim=-1)
            best = scores.argmax(dim=-1)
            for i, dec in enumerate(chunk):
                ids = [c.id for c in dec.candidates]
                pred = ids[int(best[i])]
                p_gold = float(probs[i, int(golds[i])])
                total_nll += -math.log(max(p_gold, 1e-12))
                rows.append({
                    "id": dec.meta["id"], "group": dec.meta["group"],
                    "family": dec.family, "lang": dec.lang, "k": dec.k,
                    "gold": dec.gold, "argmax_id": pred,
                    "correct": pred == dec.gold,
                    "p_gold": round(p_gold, 6),
                    "p_pad_leak": round(
                        float(probs[i, dec.k:].sum()) if probs.shape[1] > dec.k
                        else 0.0, 12)})
    if was_training:
        model.train()
    hits = sum(1 for r in rows if r["correct"])
    n = len(rows)
    lo, hi = wilson_interval(hits, n)
    return {"n": n, "correct": hits, "accuracy": hits / n, "ci95": [lo, hi],
            "chance": sum(1.0 / r["k"] for r in rows) / n,
            "mean_nll": total_nll / n,
            "max_pad_leak": max(r["p_pad_leak"] for r in rows),
            "by_family": _bucket(rows, "family"),
            "by_lang": _bucket(rows, "lang"),
            "by_k": _bucket(rows, "k"),
            "wrong": [r["id"] for r in rows if not r["correct"]],
            "rows": rows}


def _bucket(rows: list[dict], key: str) -> dict:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(str(row[key]), []).append(row)
    out = {}
    for name, group in sorted(buckets.items()):
        hits = sum(1 for r in group if r["correct"])
        out[name] = {"n": len(group), "correct": hits,
                     "accuracy": hits / len(group),
                     "chance": sum(1.0 / r["k"] for r in group) / len(group)}
    return out


def shuffled_candidates(dec: CE.Decision, rng: random.Random) -> CE.Decision:
    """La misma decisión con los candidatos en otro orden.

    Se hace CADA época: si el gold se resolviera por posición en vez de
    por id, la pérdida no bajaría y este paso lo vería. Es también la
    prueba de que el orden no lleva información.
    """
    import dataclasses
    cands = list(dec.candidates)
    rng.shuffle(cands)
    return dataclasses.replace(dec, candidates=tuple(cands))


# -- las cuatro comprobaciones de tubería, con pesos reales --------------

def pipeline_checks(model, tokenizer, decisions: list[CE.Decision], device,
                    entail_index: int, seed: int) -> dict:
    """Los cuatro fallos que el gate nombra, medidos, no supuestos.

    1. gold desalineado — permutar los candidatos no puede mover la
       probabilidad del gold, porque el gold se resuelve por id;
    2. máscara de candidatos — una fila de K=3 puntúa igual sola que en un
       lote con filas de K=4, y el relleno se queda a probabilidad 0;
    3. formato entreno/eval — los strings que tokeniza el entreno y los que
       construye el evaluador son los MISMOS para las mismas decisiones, y
       los dos salen de las mismas dos funciones de `model/ce_scorer.py`;
    4. truncado del estado — ningún par del conjunto pierde tokens.
    """
    import torch
    rng = random.Random(seed)
    model.eval()
    with torch.no_grad():
        # (1) gold por id, no por posición
        sample = decisions[:8]
        base, golds, report = forward_logits(model, tokenizer, sample, device,
                                             entail_index)
        p_base = torch.softmax(base.float(), dim=-1).gather(
            1, golds.view(-1, 1)).squeeze(1)
        permuted = [shuffled_candidates(d, rng) for d in sample]
        perm_scores, perm_golds, _ = forward_logits(
            model, tokenizer, permuted, device, entail_index)
        p_perm = torch.softmax(perm_scores.float(), dim=-1).gather(
            1, perm_golds.view(-1, 1)).squeeze(1)
        gold_delta = float((p_base - p_perm).abs().max())

        # (2) la máscara: K=3 sola frente a K=3 mezclada con K=4
        k3 = [d for d in decisions if d.k == 3][:4]
        k4 = [d for d in decisions if d.k == 4][:2]
        if not k3 or not k4:
            raise ValueError(
                "the mask check needs rows of two different K in the set; "
                f"found K=3 x{len(k3)} and K=4 x{len(k4)} — without both, the "
                "check passes without measuring anything")
        alone, _, _ = forward_logits(model, tokenizer, k3, device,
                                     entail_index)
        mixed, _, _ = forward_logits(model, tokenizer, k3 + k4, device,
                                     entail_index)
        p_alone = torch.softmax(alone.float(), dim=-1)
        p_mixed = torch.softmax(mixed.float(), dim=-1)[:len(k3)]
        mask_delta = float(
            (p_alone - p_mixed[:, :p_alone.shape[1]]).abs().max())
        pad_leak = (float(p_mixed[:len(k3), 3:].sum())
                    if p_mixed.shape[1] > 3 else 0.0)

        # (3) entreno y evaluación, los MISMOS strings: los pares que este
        #     trainer va a tokenizar, contra los que el evaluador
        #     (`eval.ce_nograd`, vía `CE.flatten_pairs`) construiría para
        #     las mismas decisiones. Se comparan los strings, no los
        #     tensores: la divergencia que este paso busca es de texto.
        from eval import ce_nograd as NG

        train_pairs, _ = flatten(sample)
        eval_pairs, _ = NG.CE.flatten_pairs(sample)
        same_pairs = train_pairs == eval_pairs
        one_render_path = (flatten is NG.CE.flatten_pairs
                           and encode is NG.CE.encode_pairs)

    all_pairs, _ = flatten(decisions)
    full_report = CE.length_report(tokenizer, all_pairs)
    return {
        "gold_alignment": {
            "max_abs_p_gold_delta": gold_delta, "tolerance": BATCH_TOL,
            "n_decisions": len(sample), "pass": gold_delta <= BATCH_TOL,
            "note": ("the gold is resolved by candidate id against the same "
                     "tuple flatten() walked, so permuting cannot move it")},
        "candidate_mask": {
            "max_abs_prob_delta": mask_delta, "pad_probability": pad_leak,
            "tolerance": BATCH_TOL, "k3": len(k3), "k4": len(k4),
            "pass": mask_delta <= BATCH_TOL and pad_leak <= 1e-9},
        "train_eval_format": {
            "same_rendered_pairs": same_pairs,
            "n_pairs_compared": len(train_pairs),
            "single_render_path": one_render_path,
            "render_path": "model.ce_scorer.flatten_pairs -> render_pair",
            "encode_path": "model.ce_scorer.encode_pairs",
            "hypothesis_format_id": CE.HYPOTHESIS_FORMAT_ID,
            "format_fingerprint": CE.format_fingerprint(),
            "pass": (same_pairs and one_render_path
                     and CE.format_fingerprint() == CE.HYPOTHESIS_FORMAT_ID),
            "note": ("the strings the trainer tokenises are compared with the "
                     "ones eval/ce_nograd.py builds for the same decisions, "
                     "and both go through the same two functions in "
                     "model/ce_scorer.py — there is no second template")},
        "state_truncation": dict(full_report, first_batch=report),
    }


# -- el ajuste -----------------------------------------------------------

def finetune(weights: str = DEFAULT_WEIGHTS, device: str = "mps",
             seed: int = DEFAULT_SEED, epochs: int = DEFAULT_EPOCHS,
             lr: float = DEFAULT_LR,
             decisions_per_batch: int = DEFAULT_DECISIONS_PER_BATCH,
             eval_every: int = DEFAULT_EVAL_EVERY,
             max_length: int = CE.DEFAULT_MAX_LENGTH) -> dict:
    """Sobreajusta el conjunto y devuelve las dos cifras y los chequeos."""
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    from model.weights import require_verified

    random.seed(seed)
    torch.manual_seed(seed)

    decs = overfit_decisions()
    validity = validate_set(decs)
    path = require_verified(weights)
    dev = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, dtype=torch.float32)
    entail_index = CE._entailment_index(model.config)
    model.to(dev)

    checks = pipeline_checks(model, tokenizer, decs, dev, entail_index, seed)
    baseline = evaluate(model, tokenizer, decs, dev, entail_index,
                        decisions_per_batch, max_length)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    history = []
    t0 = time.perf_counter()
    fitted = baseline
    epochs_used = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(decs)))
        rng.shuffle(order)
        epoch_loss, steps = 0.0, 0
        for start in range(0, len(order), decisions_per_batch):
            chunk = [shuffled_candidates(decs[i], rng)
                     for i in order[start:start + decisions_per_batch]]
            scores, golds, _ = forward_logits(model, tokenizer, chunk, dev,
                                              entail_index, max_length)
            loss = torch.nn.functional.cross_entropy(scores, golds)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            epoch_loss += float(loss.detach())
            steps += 1
        epochs_used = epoch
        row = {"epoch": epoch, "loss": epoch_loss / max(steps, 1)}
        if epoch % eval_every == 0 or epoch == epochs:
            fitted = evaluate(model, tokenizer, decs, dev, entail_index,
                              decisions_per_batch, max_length)
            row["accuracy"] = fitted["accuracy"]
        history.append(row)
        if row.get("accuracy") == 1.0:
            break
    wall = time.perf_counter() - t0
    if not history:
        raise ValueError("--epochs must be >= 1: nothing was fitted")
    if "accuracy" not in history[-1]:
        fitted = evaluate(model, tokenizer, decs, dev, entail_index,
                          decisions_per_batch, max_length)
        history[-1]["accuracy"] = fitted["accuracy"]

    return {
        "checkpoint": {
            "id": weights, "params": sum(p.numel()
                                         for p in model.parameters()),
            "n_classes": int(model.config.num_labels),
            "entail_index": entail_index, "trainable": "all parameters"},
        "set": dict(validity, fingerprint=set_fingerprint(decs)),
        "hyperparams": {"seed": seed, "epochs_requested": epochs,
                        "epochs_used": epochs_used, "lr": lr,
                        "decisions_per_batch": decisions_per_batch,
                        "weight_decay": 0.01, "grad_clip": 1.0,
                        "optimizer": "AdamW", "device": device,
                        "max_length": max_length,
                        "score_mode": CE.ENTAIL_LOGIT,
                        "loss": "listwise cross-entropy over the row's K "
                                "masked candidate scores"},
        "train_seconds": round(wall, 2),
        "baseline_untrained": baseline,
        "fitted": fitted,
        "history": history,
        "pipeline_checks": checks,
    }


def gate(result: dict) -> dict:
    """Los dos requisitos duros, más los cuatro chequeos de tubería."""
    fitted = result["fitted"]["accuracy"]
    base = result["baseline_untrained"]["accuracy"]
    checks = result["pipeline_checks"]
    overfit_ok = fitted > OVERFIT_TARGET
    set_ok = base <= SET_VALIDITY_MAX
    margin_ok = (fitted - base) >= MIN_MARGIN
    pipeline_ok = all(c.get("pass", True) for c in checks.values())
    return {
        "overfit_above_target": {"value": fitted, "target": OVERFIT_TARGET,
                                 "pass": overfit_ok},
        "set_is_valid_evidence": {
            "untrained_accuracy": base, "max_allowed": SET_VALIDITY_MAX,
            "pass": set_ok,
            "note": ("if the bare checkpoint already solved these cases the "
                     "set would prove nothing about the pipeline: the gate "
                     "fails and asks for a harder set")},
        "margin": {"value": fitted - base, "min": MIN_MARGIN,
                   "pass": margin_ok},
        "pipeline_checks": {"pass": pipeline_ok,
                            "failed": [k for k, c in checks.items()
                                       if not c.get("pass", True)]},
        "pass": overfit_ok and set_ok and margin_ok and pipeline_ok,
    }


def command_line(args) -> str:
    """El comando EXACTO, con todo lo que mueve la cifra.

    `--eval-every` entra porque la parada temprana ocurre en una época de
    evaluación: cambiarlo cambia cuántas épocas se corren.
    """
    return (f"PYTHONPATH=. .venv-train/bin/python -m "
            f"training.python.ce_overfit overfit --device {args.device} "
            f"--seed {args.seed} --epochs {args.epochs} --lr {args.lr} "
            f"--decisions-per-batch {args.decisions_per_batch} "
            f"--eval-every {args.eval_every} "
            f"--weights {args.weights}")


def run(args) -> dict:
    result = finetune(weights=args.weights, device=args.device,
                      seed=args.seed, epochs=args.epochs, lr=args.lr,
                      decisions_per_batch=args.decisions_per_batch,
                      eval_every=args.eval_every)
    report = {
        "task": TASK,
        #: `measured` = esta cifra salió de una ejecución real. El gate que
        #: hay en el árbol antes de la primera ejecución lleva
        #: `awaiting-operator-compute` y `pass: null`, para que no se pueda
        #: confundir un hueco con un resultado.
        "status": "measured",
        "generated_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "command": command_line(args),
        "seed": args.seed,
        "not_generalization": (
            "TRAINED AND MEASURED ON THE SAME 48 CASES, on purpose. This "
            "figure is a pipeline check: gradients arrive, the gold is where "
            "the trainer thinks it is, the state is not truncated, the "
            "candidate mask does not mix rows of different K, and the "
            "hypothesis format is identical in training and evaluation. It "
            "is NOT a measure of generalization, it does not estimate any "
            "product accuracy, and it may not be compared with the 70% or "
            "with any external figure. The first figure with external "
            "meaning is #T-ce-finetune against the same untrained "
            "checkpoint on the #T-battery-dev development battery."),
        "hypothesis_format": {
            "frozen_id": CE.HYPOTHESIS_FORMAT_ID,
            "computed": CE.format_fingerprint(),
            "hypothesis": CE.HYPOTHESIS, "premise": CE.PREMISE,
            "context_header": CE.CONTEXT_HEADER,
            "comparative_context_by_family": CE.COMPARATIVE_CONTEXT,
            "score_mode": CE.DEFAULT_SCORE_MODE,
            "truncation": CE.TRUNCATION_STRATEGY,
            "frozen_for": ("the rest of the pilot: #T-ce-finetune and "
                           "#T-ce-confirm publish this same id or they are "
                           "not comparable with this gate"),
            "how_it_is_enforced": (
                "model.ce_scorer.format_fingerprint() must equal "
                "HYPOTHESIS_FORMAT_ID; model/test_ce_scorer.py fails "
                "otherwise, and both the trainer and the evaluator render "
                "through model.ce_scorer.render_pairs — there is no second "
                "template in the tree")},
        **result,
    }
    report["gate"] = gate(result)
    if not args.no_write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(GATE_PATH, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True,
                      ensure_ascii=False)
            fh.write("\n")
    return report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["overfit", "describe"])
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--decisions-per-batch", type=int,
                    default=DEFAULT_DECISIONS_PER_BATCH)
    ap.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv[1:])

    if args.command == "describe":
        decs = overfit_decisions()
        info = dict(validate_set(decs), fingerprint=set_fingerprint(decs))
        print(json.dumps(info, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    report = run(args)
    base = report["baseline_untrained"]
    fit = report["fitted"]
    print(f"[overfit] set: {report['set']['n']} cases "
          f"fingerprint={report['set']['fingerprint']} "
          f"chance={report['set']['chance']:.3f}")
    print(f"[overfit] untrained: {base['correct']}/{base['n']} = "
          f"{base['accuracy']:.3f}")
    print(f"[overfit] fitted:    {fit['correct']}/{fit['n']} = "
          f"{fit['accuracy']:.3f} "
          f"({report['hyperparams']['epochs_used']} epochs, "
          f"{report['train_seconds']}s)")
    for name, check in report["pipeline_checks"].items():
        print(f"[overfit] {name}: "
              f"{'PASS' if check.get('pass', True) else 'FAIL'}")
    print(f"[overfit] gate: {'PASS' if report['gate']['pass'] else 'FAIL'}")
    if not args.no_write:
        print(f"[overfit] wrote {GATE_PATH.relative_to(ROOT)}")
    return 0 if report["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
