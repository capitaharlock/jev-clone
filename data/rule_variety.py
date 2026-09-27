"""Variedad por regla para las dos familias numéricas (#T-numeric-gen).

El primer brazo dirigido de la escalera (`docs/bucle-infinito.md` §4). La
comparación de atributos no se aprende ni en el holdout del propio piloto
(0,574 → 0,574 con 1 000 decisiones, `#T-ce-finetune`), y en
`attribute_comparison` y `priority_decision` el gold sale de una REGLA,
no de un profesor: con prosa local se generan miles de episodios por
minuto y sin Qwen de por medio.

Lo que este módulo añade sobre `data.episode_gen`, que sólo sabía
escribir `precio × duración` sobre dos billetes:

* **15 atributos** (`ATTRS`), cada uno con su unidad, su dirección
  (`min`/`max`), su verbo y su superlativo en ES y EN. Los valores viven
  en la unidad MENOR y son enteros: la comparación es exacta y el
  renderizado es lo único que varía.
* **Formatos numéricos mezclados** (`render_value`): «1.200 €»,
  «1,2 mil euros», «1.200,50 €», «dos horas y cuarto», «3 h 5 min»,
  «185 minutos». El formato es cosmético — nunca cambia el entero que
  compara la regla.
* **Formas de pregunta** (`FORMS`): superlativo, ORDINAL («el segundo
  más caro»), umbral («entre los que tienen un precio por debajo de…»),
  empate con desempate explícito; y en prioridad, dos y tres criterios
  encadenados y un umbral que descarta antes de ordenar.
* **Una sola regla de gold** (`rule_gold`): criterios encadenados sobre
  enteros + filtros + rango. `comparison_gold`/`priority_gold` de
  `episode_gen` son el caso particular de dos opciones y un criterio.
* **Cuatro variantes por grupo** (`VARIANTS`): `base`, `number` (cambia
  UN número), `rule` (cambia el criterio en comparación, el ORDEN de los
  criterios en prioridad) y `paraphrase` (cambia plantilla, formato y
  orden de las frases). Las tres primeras tienen gold distinto entre sí
  por construcción; la paráfrasis tiene el MISMO gold. `apply_variant`
  no devuelve nada que no cumpla eso: comprueba y ajusta.
* **Vocabulario de entidades disjunto** de `data/battery_dev.jsonl` y
  del sellado (`ENTITY_NOUNS` × `CODENAMES`: instrumentos, material de
  laboratorio y de escalada con nombres de mineral). `leakage_report`
  lo comprueba contra AMBAS con `data.leakage`.

**Posición del gold.** Los valores de cada atributo se sortean SIN
REPOSICIÓN (`rng.sample`), así que no hay dos opciones con el mismo
valor en un criterio salvo cuando la forma `tie` lo fuerza a propósito —
y ahí decide el segundo criterio, que sí es distinto. El orden es
siempre TOTAL y estricto: el desempate por índice de `rule_gold` no
llega a ejecutarse nunca en lo publicado. Con valores intercambiables
entre posiciones, la posición del gold es uniforme por simetría, no por
un reequilibrado posterior. `POSITION_CHI2` fija la prueba y su valor
crítico ANTES de medir; `position_chi2()` la ejecuta.

CPU puro, sin red, sin torch. Uso:

    python3 -m data.rule_variety publish --batch 1 --n 50000
    python3 -m data.rule_variety gate --dir artifacts/episodes-rule/batch-0001
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import episode_contract as EC  # noqa: E402
from data import leakage as LK  # noqa: E402

TASK = "T-numeric-gen"
VARIETY_VERSION = "rule-variety-v1"

#: Las dos familias con gold por regla. Las otras tres siguen en
#: `data.episode_gen`: su gold no es aritmético y no se amplía aquí.
FAMILIES = (EC.COMPARISON, EC.PRIORITY)

#: Cardinalidades del corte de desarrollo y del sellado, las mismas.
KS = (2, 3, 8)

#: Las cuatro variantes de un grupo. El orden importa: `base` primero
#: porque las otras tres se derivan de ella.
VARIANTS = ("base", "number", "rule", "paraphrase")

#: Variedad mínima que la task exige (>= 12 atributos).
MIN_VARIETY = 12

#: Raíz de los lotes por regla. La semilla de un lote ES su número.
BATCH_ROOT = os.path.join(ROOT, "artifacts", "episodes-rule")

#: La prueba de uniformidad de la posición del gold, ESCRITA ANTES de
#: medir (2026-09-27, antes del primer lote). Una χ² de bondad de ajuste
#: por cada K, con `df = K - 1` y α = 0,01 a una cola. Se rechaza la
#: uniformidad si el estadístico supera el valor crítico. No hay
#: reequilibrado: si falla, el generador está sesgado y se arregla.
POSITION_CHI2 = {
    "written_utc": "2026-09-27",
    "test": "Pearson chi-square goodness-of-fit against the uniform over "
            "the K candidate positions, one test per K",
    "df": "K - 1",
    "alpha": 0.01,
    "critical_values": {"1": 6.635, "2": 9.210, "7": 18.475},
    "reject_if": "statistic > critical value for its df",
    "why_uniform": "attribute values are drawn without replacement per "
                   "criterion, so positions are exchangeable and the "
                   "index tiebreak of rule_gold never fires",
    "min_n_per_k": 200,
}

#: Valores críticos de la χ² a α = 0,01 para los df que usamos.
_CHI2_CRIT = {1: 6.635, 2: 9.210, 7: 18.475}


# -- atributos: 15, con unidad, dirección, verbo y superlativo ------------
# `lo`/`hi`/`step` están en la unidad MENOR y son enteros: la regla
# compara enteros y el formato sólo decide cómo se escriben.
def _a(key, kind, better, lo, hi, step, scale, unit_es, unit_en,
       noun_es, gender, noun_en, vp_es, vp_en, pred=None, adj=None):
    return {"key": key, "kind": kind, "better": better, "lo": lo, "hi": hi,
            "step": step, "scale": scale,
            "unit": {"es": unit_es, "en": unit_en},
            "noun": {"es": noun_es, "en": noun_en}, "gender": gender,
            "vp": {"es": vp_es, "en": vp_en},
            "pred": pred or {}, "adj": adj or {}}


ATTRS = (
    _a("price_eur", "money", "min", 1500, 480000, 500, 100, "euros", "euros",
       "precio", "m", "price", "cuesta {v}", "costs {v}",
       pred={"es": {"min": "es el más barato", "max": "es el más caro"},
             "en": {"min": "is the cheapest", "max": "is the most expensive"}},
       adj={"es": {"min": "barato", "max": "caro"},
            "en": {"min": "cheapest", "max": "most expensive"}}),
    _a("duration_min", "time", "min", 15, 600, 15, 1, "minutos", "minutes",
       "duración", "f", "duration", "tarda {v}", "takes {v}",
       pred={"es": {"min": "es el más rápido", "max": "es el más lento"},
             "en": {"min": "is the fastest", "max": "is the slowest"}},
       adj={"es": {"min": "rápido", "max": "lento"},
            "en": {"min": "fastest", "max": "slowest"}}),
    _a("mass_g", "plain", "min", 120, 4800, 20, 1, "gramos", "grams",
       "peso", "m", "weight", "pesa {v}", "weighs {v}",
       pred={"es": {"min": "es el más ligero", "max": "es el más pesado"},
             "en": {"min": "is the lightest", "max": "is the heaviest"}},
       adj={"es": {"min": "ligero", "max": "pesado"},
            "en": {"min": "lightest", "max": "heaviest"}}),
    _a("distance_m", "plain", "min", 100, 20000, 100, 1000,
       "kilómetros", "kilometres", "distancia", "f", "distance",
       "está a {v}", "sits {v} away",
       pred={"es": {"min": "está más cerca", "max": "está más lejos"},
             "en": {"min": "is the closest", "max": "is the farthest"}},
       adj={"es": {"min": "cercano", "max": "lejano"},
            "en": {"min": "closest", "max": "farthest"}}),
    _a("battery_min", "time", "max", 60, 1440, 30, 1, "minutos", "minutes",
       "batería", "f", "battery life", "aguanta {v} de batería",
       "lasts {v} on battery",
       pred={"es": {"min": "aguanta menos batería",
                    "max": "aguanta más batería"},
             "en": {"min": "lasts least on battery",
                    "max": "lasts longest on battery"}}),
    _a("noise_db", "plain", "min", 180, 900, 10, 10, "decibelios",
       "decibels", "ruido", "m", "noise", "suena a {v}", "runs at {v}",
       pred={"es": {"min": "suena más bajo", "max": "suena más alto"},
             "en": {"min": "is the quietest", "max": "is the loudest"}},
       adj={"es": {"min": "silencioso", "max": "ruidoso"},
            "en": {"min": "quietest", "max": "loudest"}}),
    _a("weight_kg", "plain", "min", 1000, 45000, 250, 1000, "kilos",
       "kilos", "carga", "f", "load", "carga {v}", "hauls {v}"),
    _a("capacity_seats", "plain", "max", 4, 120, 2, 1, "plazas", "seats",
       "capacidad", "f", "capacity", "admite {v}", "takes {v}"),
    _a("rating_pt", "plain", "max", 50, 100, 1, 10, "puntos", "points",
       "valoración", "f", "rating", "puntúa {v}", "scores {v}"),
    _a("lead_days", "plain", "min", 1, 45, 1, 1, "días", "days",
       "plazo", "m", "lead time", "llega en {v}", "arrives in {v}"),
    _a("wait_min", "time", "min", 5, 240, 5, 1, "minutos", "minutes",
       "espera", "f", "wait", "hace esperar {v}", "waits {v}"),
    _a("power_w", "plain", "max", 5000, 240000, 1000, 1000, "kilovatios",
       "kilowatts", "potencia", "f", "power", "da {v}", "delivers {v}"),
    _a("storage_gb", "plain", "max", 64, 4096, 64, 1, "gigas",
       "gigabytes", "almacenamiento", "m", "storage",
       "guarda {v}", "holds {v}"),
    _a("warranty_mo", "plain", "max", 6, 72, 1, 1, "meses", "months",
       "garantía", "f", "warranty", "lleva {v} de garantía",
       "carries a {v} warranty"),
    _a("consumption_wh", "plain", "min", 1000, 90000, 500, 1000,
       "kilovatios hora", "kilowatt-hours", "consumo", "m",
       "consumption", "consume {v}", "draws {v}"),
)

ATTR_BY_KEY = {a["key"]: a for a in ATTRS}

#: Los atributos con adjetivo propio son los únicos que admiten forma
#: ORDINAL: «el segundo más caro» necesita un adjetivo, no una perífrasis.
ORDINAL_KEYS = tuple(a["key"] for a in ATTRS if a["adj"])


# -- vocabulario de entidades, disjunto de dev y del sellado -------------
# dev habla de buses, cabañas, vuelos, hoteles, portátiles, planes y
# talleres; el sellado, de aljibes, tinajas, dehesas y colmenas. Esto no
# se solapa con ninguno de los dos: instrumental de laboratorio, música
# y escalada, nombrado con minerales. `leakage_report` lo comprueba.
ENTITY_NOUNS = {
    "es": ("espectrómetro", "criostato", "autoclave", "plotter",
           "amplificador", "ecualizador", "fagot", "clarinete",
           "trombón", "oboe", "mosquetón", "arnés", "piolet",
           "micrófono", "altavoz"),
    "en": ("spectrometer", "cryostat", "autoclave", "plotter",
           "amplifier", "equaliser", "bassoon", "clarinet",
           "trombone", "oboe", "carabiner", "harness", "ice axe",
           "microphone", "loudspeaker"),
}

#: Minerales, no piedras de joyería: el sellado tiene una escena de
#: joyero con ámbar, jade y ónice, y dev describe un violonchelo — los
#: cuatro términos salieron en `leakage_report` y se cambiaron (medido el
#: 2026-09-27, no supuesto).
CODENAMES = {
    "es": ("Cinabrio", "Cuarzo", "Olivino", "Obsidiana", "Pirita",
           "Granate", "Azurita", "Malaquita", "Zafiro", "Topacio",
           "Fluorita", "Turmalina"),
    "en": ("Cinnabar", "Quartz", "Olivine", "Obsidian", "Pyrite",
           "Garnet", "Azurite", "Malachite", "Sapphire", "Topaz",
           "Fluorite", "Tourmaline"),
}

#: Plantillas de estado. La primera es la desnuda; las demás añaden un
#: marco que no aporta ningún hecho decisivo.
STATE_TPL = {
    "es": (
        "{clauses}",
        "Comparativa de este mes. {clauses}",
        "{clauses} Todos están disponibles esta semana.",
        "Del inventario del taller: {clauses}",
        "{clauses} Los datos son de la última revisión.",
        "Sobre la mesa hay varias opciones. {clauses}",
        "Anotado en la ficha de compras: {clauses}",
        "{clauses} Nada más consta en la ficha.",
    ),
    "en": (
        "{clauses}",
        "This month's comparison. {clauses}",
        "{clauses} All of them are available this week.",
        "From the workshop inventory: {clauses}",
        "{clauses} The figures are from the latest check.",
        "Several options are on the bench. {clauses}",
        "Noted on the purchase sheet: {clauses}",
        "{clauses} Nothing else is on the sheet.",
    ),
}

#: Formas de pregunta por familia. `min_k` filtra las que no tienen
#: sentido con dos opciones (no hay «el segundo más caro» entre dos).
FORMS = {
    EC.COMPARISON: (
        {"name": "superlative", "min_k": 2, "criteria": 1, "rank": 1},
        {"name": "ordinal", "min_k": 3, "criteria": 1, "rank": 2},
        {"name": "threshold", "min_k": 3, "criteria": 1, "rank": 1,
         "filter": True},
        {"name": "tie", "min_k": 2, "criteria": 2, "rank": 1, "tie": True},
    ),
    EC.PRIORITY: (
        {"name": "chain2", "min_k": 2, "criteria": 2, "rank": 1,
         "tie": True},
        {"name": "chain3", "min_k": 3, "criteria": 3, "rank": 1,
         "tie": True},
        {"name": "threshold_chain", "min_k": 3, "criteria": 2, "rank": 1,
         "filter": True, "tie": True},
    ),
}

#: Cuántos atributos caben en el estado según K — MEDIDO, no estimado.
#: El scorer tiene una ventana de 512 tokens y `model.ce_scorer.
#: length_report` se NIEGA a recortar el estado en silencio. Con K = 8 y
#: tres atributos por opción, 419 de 17 328 pares del primer lote pasaban
#: de 512 (máximo 570; medido el 2026-09-27 con el tokenizador de
#: minilmv2-l6-mnli-xnli). Con dos atributos caben todos, así que las
#: formas que necesitan tres —umbral y tres criterios encadenados— viven
#: en K = 3, no en K = 8. Es una restricción del backbone, y sube con él
#: (`#T-backbone-ladder`).
MAX_STATE_ATTRS = {2: 4, 3: 3, 8: 2}

#: Cota del estado en caracteres, para que un cambio de plantilla no se
#: coma la ventana sin que nadie lo note (el test la exige).
MAX_STATE_CHARS = 900

_ORD = {"es": {2: "segundo", 3: "tercero"},
        "en": {2: "second", 3: "third"}}

_WORD_TIME = {
    30: {"es": "media hora", "en": "half an hour"},
    45: {"es": "tres cuartos de hora", "en": "three quarters of an hour"},
    75: {"es": "una hora y cuarto", "en": "an hour and a quarter"},
    90: {"es": "una hora y media", "en": "an hour and a half"},
    135: {"es": "dos horas y cuarto", "en": "two and a quarter hours"},
    150: {"es": "dos horas y media", "en": "two and a half hours"},
    195: {"es": "tres horas y cuarto", "en": "three and a quarter hours"},
    210: {"es": "tres horas y media", "en": "three and a half hours"},
}

STYLES = {
    "money": ("plain", "group", "decimal", "thousand"),
    "time": ("plain_h", "plain_min", "h_min", "words"),
    "plain": ("plain", "group", "decimal"),
}


# -- renderizado de números: cosmético, nunca aritmético ------------------
def _group(digits: str, lang: str) -> str:
    """Separador de millares: «1.200» en ES, «1,200» en EN."""
    sep = "." if lang == "es" else ","
    out = []
    while len(digits) > 3:
        out.insert(0, digits[-3:])
        digits = digits[:-3]
    out.insert(0, digits)
    return sep.join(out)


def _decimals(value: int, scale: int, lang: str, places: int) -> str:
    whole, rest = divmod(value, scale)
    frac = str(rest * (10 ** places) // scale).zfill(places)
    comma = "," if lang == "es" else "."
    return f"{_group(str(whole), lang)}{comma}{frac}"


def render_value(attr: dict, value: int, lang: str, style: str) -> str:
    """El valor en prosa. Devuelve el mismo entero escrito de otra forma.

    Un estilo que no aplica a este valor (mil sin ser múltiplo de mil,
    «dos horas y cuarto» de un valor que no está en `_WORD_TIME`) cae al
    estilo llano de su familia: el formato no puede inventar un número.
    """
    kind, scale = attr["kind"], attr["scale"]
    unit = attr["unit"][lang]
    if kind == "money":
        cents = value
        sym = " €" if lang == "es" else " euros"
        if style == "thousand" and cents % 10000 == 0 and cents >= 100000:
            whole = cents // 100
            txt = _decimals(whole, 1000, lang, 1).rstrip("0").rstrip(",.") \
                if whole % 1000 else _group(str(whole // 1000), lang)
            word = "mil euros" if lang == "es" else "thousand euros"
            return f"{txt} {word}"
        if style == "decimal" and cents % 100:
            return _decimals(cents, 100, lang, 2) + sym
        if cents % 100 == 0:
            euros = cents // 100
            if style == "group":
                return _group(str(euros), lang) + sym
            return f"{euros}{sym}"
        return _decimals(cents, 100, lang, 2) + sym
    if kind == "time":
        mins = value
        hour = "horas" if lang == "es" else "hours"
        one_h = "hora" if lang == "es" else "hour"
        if style == "words" and mins in _WORD_TIME:
            return _WORD_TIME[mins][lang]
        if style == "h_min" and mins >= 60 and mins % 60:
            return f"{mins // 60} h {mins % 60} min"
        if style == "plain_h" and mins % 60 == 0:
            h = mins // 60
            return f"{h} {one_h if h == 1 else hour}"
        return f"{mins} {unit}"
    whole, rest = divmod(value, scale)
    if scale == 1:
        return f"{_group(str(whole), lang) if style == 'group' else whole} {unit}"
    if rest == 0 and style != "decimal":
        return f"{_group(str(whole), lang) if style == 'group' else whole} {unit}"
    places = 1 if scale == 10 else 2
    return f"{_decimals(value, scale, lang, places)} {unit}"


# -- la regla del gold: una sola, para las dos familias -------------------
def _sign(direction: str) -> int:
    if direction not in ("min", "max"):
        raise ValueError(f"unknown direction {direction!r}: min or max")
    return 1 if direction == "min" else -1


def _passes(attrs_i: dict, filters) -> bool:
    for key, op, bound in filters:
        value = attrs_i[key]
        if op == "le" and not value <= bound:
            return False
        if op == "ge" and not value >= bound:
            return False
        if op not in ("le", "ge"):
            raise ValueError(f"unknown filter op {op!r}: le or ge")
    return True


def eligible(attrs: list, filters=()) -> list:
    """Los índices que pasan los umbrales, en orden."""
    return [i for i in range(len(attrs)) if _passes(attrs[i], filters)]


def rule_gold(attrs: list, criteria, filters=(), rank: int = 1) -> int:
    """Índice ganador: criterios encadenados sobre el conjunto elegible.

    `criteria` es una tupla de `(clave, "min"|"max")` en orden de
    prioridad — el primero manda, el siguiente sólo deshace empates.
    `filters` descarta antes de ordenar, `rank` pide el n-ésimo mejor
    (1 = el mejor, 2 = «el segundo más…»). El desempate final por índice
    está para que la función sea total; lo publicado nunca lo necesita
    (valores sorteados sin reposición).
    """
    pool = eligible(attrs, filters)
    if not pool:
        raise ValueError("no candidate passes the thresholds")
    if not criteria:
        raise ValueError("rule_gold needs at least one criterion")
    order = sorted(pool, key=lambda i: (
        tuple(_sign(d) * attrs[i][k] for k, d in criteria), i))
    return order[min(max(rank, 1), len(order)) - 1]


# -- muestreo del caso base ----------------------------------------------
def state_attrs(form: dict) -> int:
    """Cuántos atributos pide una forma en el estado.

    Siempre al menos DOS: la variante `rule` de comparación cambia el
    criterio y sin un segundo atributo no habría a qué cambiarlo. El
    umbral añade el suyo.
    """
    return max(form["criteria"], 2) + (1 if form.get("filter") else 0)


def _pool(seq, variety: int):
    """Los primeros `variety` elementos: la variedad es un dial, no un sí/no."""
    return list(seq[:max(1, min(int(variety), len(seq)))])


#: Escalones reservados en cada extremo de la rejilla. El caso base no
#: sortea nunca los extremos, así que el contrafactual `number` SIEMPRE
#: tiene sitio para pasar por delante del ganador o por detrás del
#: perseguidor. Sin este margen, 10 de 192 000 casos se quedaban sin
#: contrafactual posible (medido el 2026-09-27 con el lote de 50 000:
#: `lead_days` a 1 con `lo = 1`, `capacity_seats` a 120 con `hi = 120`),
#: y el generador fallaba en vez de publicar una variante que no mueve el
#: gold — el fallo correcto, pero un fallo.
GRID_MARGIN = 3


def _sample_values(rng: random.Random, attr: dict, k: int) -> list:
    """`k` valores DISTINTOS del atributo. Sin reposición, siempre."""
    grid = list(range(attr["lo"], attr["hi"] + 1, attr["step"]))
    if len(grid) >= k + 2 * GRID_MARGIN:
        grid = grid[GRID_MARGIN:len(grid) - GRID_MARGIN]
    if len(grid) < k:
        raise ValueError(f"{attr['key']}: grid of {len(grid)} for K={k}")
    return rng.sample(grid, k)


def base_case(family: str, lang: str, k: int, rng: random.Random,
              variety: int) -> dict:
    """El caso base: entidades, atributos, criterios, umbral y forma.

    Puro en `rng`. Devuelve una especificación, no un episodio: las
    variantes se derivan de ella y el renderizado va aparte.
    """
    if family not in FAMILIES:
        raise ValueError(f"{family!r} has no rule gold; {FAMILIES}")
    attrs_pool = _pool(ATTRS, variety)
    # El presupuesto de atributos del estado es el mínimo de dos cosas: la
    # ventana del scorer (`MAX_STATE_ATTRS`) y los atributos que la
    # variedad pedida pone sobre la mesa. Con `--variety 2` no hay tres
    # atributos que enseñar, así que la forma de umbral no existe.
    budget = min(MAX_STATE_ATTRS.get(k, 3), len(attrs_pool))
    forms = [f for f in FORMS[family]
             if k >= f["min_k"] and state_attrs(f) <= budget]
    if not forms:
        raise ValueError(
            f"variety {variety} leaves no {family} form for K={k}: "
            f"{len(attrs_pool)} attributes, budget {budget}")
    form = rng.choice(forms)
    n_crit = form["criteria"]
    if form["name"] == "ordinal":
        # «El segundo más caro» necesita un adjetivo, y la variante
        # `rule` cambia el criterio: los DOS atributos del estado tienen
        # que admitir forma ordinal, no sólo el primero.
        attrs_pool = [a for a in attrs_pool if a["key"] in ORDINAL_KEYS] \
            or [ATTR_BY_KEY[k] for k in ORDINAL_KEYS[:2]]
    first = rng.choice(attrs_pool)
    rest = [a for a in attrs_pool if a["key"] != first["key"]]
    rng.shuffle(rest)
    # Siempre al menos DOS atributos en el estado: la variante `rule` de
    # comparación cambia el criterio, y sin un segundo atributo no hay a
    # qué cambiarlo.
    need = max(n_crit, 2) + (1 if form.get("filter") else 0)
    chosen = [first] + rest[:need - 1]
    if len(chosen) < need:
        raise ValueError(f"variety {variety} is too small for {form['name']}")
    criteria = [(a["key"], a["better"]) for a in chosen[:n_crit]]
    # El umbral va SIEMPRE sobre un atributo que no es el primer
    # criterio, así que cambiar un número del criterio no puede mover a
    # nadie dentro o fuera del conjunto elegible.
    filters: list = []
    values = {a["key"]: _sample_values(rng, a, k) for a in chosen}
    attrs = [{a["key"]: values[a["key"]][i] for a in chosen}
             for i in range(k)]
    if form.get("filter"):
        fattr = chosen[-1]
        col = sorted(values[fattr["key"]])
        # La mediana como cota: pasan entre 2 y K-1, así que el umbral
        # hace trabajo (ni descarta a todos ni a nadie).
        if fattr["better"] == "min":
            op, bound = "le", col[max(1, k // 2)]
        else:
            op, bound = "ge", col[min(k - 2, k // 2)]
        filters = [(fattr["key"], op, bound)]
        keep = eligible(attrs, filters)
        if not 2 <= len(keep) <= k - 1:
            filters = []
    if form.get("tie"):
        # Empate DELIBERADO en el primer criterio entre los dos mejores:
        # el desempate explícito de la pregunta es el que decide.
        key, direction = criteria[0]
        pool = eligible(attrs, filters) or list(range(k))
        rank = sorted(pool, key=lambda i: _sign(direction) * attrs[i][key])
        if len(rank) >= 2:
            attrs[rank[1]][key] = attrs[rank[0]][key]
    spec = {
        "family": family, "lang": lang, "k": k, "form": form["name"],
        "noun": rng.choice(_pool(ENTITY_NOUNS[lang], variety)),
        "names": rng.sample(_pool(CODENAMES[lang], max(variety, k)), k),
        "keys": [a["key"] for a in chosen],
        "attrs": attrs, "criteria": criteria, "filters": filters,
        "rank": form["rank"] if k >= form["rank"] else 1,
        "styles": {a["key"]: rng.choice(_pool(STYLES[a["kind"]], variety))
                   for a in chosen},
        "template": rng.randrange(len(_pool(STATE_TPL[lang], variety))),
        "clause_order": rng.sample(range(k), k),
        "cf_kind": "base",
    }
    spec["gold"] = rule_gold(spec["attrs"], spec["criteria"],
                             spec["filters"], spec["rank"])
    return spec


# -- variantes: dos cambian el gold, una no puede cambiarlo --------------
def _regold(spec: dict) -> dict:
    spec["gold"] = rule_gold(spec["attrs"], spec["criteria"],
                             spec["filters"], spec["rank"])
    return spec


def _copy(spec: dict) -> dict:
    out = dict(spec)
    out["attrs"] = [dict(a) for a in spec["attrs"]]
    out["criteria"] = list(spec["criteria"])
    out["filters"] = list(spec["filters"])
    out["styles"] = dict(spec["styles"])
    return out


def _shift(spec: dict, idx: int, key: str, direction: str, beyond: int,
           better: bool) -> bool:
    """Mueve `attrs[idx][key]` a la rejilla, por delante o por detrás de
    `beyond`, sin coincidir con ningún otro valor de la columna."""
    attr = ATTR_BY_KEY[key]
    step = attr["step"]
    taken = {a[key] for i, a in enumerate(spec["attrs"]) if i != idx}
    sgn = _sign(direction) * (1 if better else -1)
    for mult in range(1, 200):
        cand = beyond - sgn * step * mult
        if attr["lo"] <= cand <= attr["hi"] and cand not in taken:
            spec["attrs"][idx][key] = cand
            return True
    return False


def _beat(spec: dict, target: int, gold: int) -> None:
    """Un solo número cambia y `target` queda por delante del gold.

    Si la rejilla no da para mejorar a `target` (el gold ya está en el
    tope del atributo), se EMPEORA el número del gold: sigue siendo un
    único número el que cambia, y el orden se invierte igual.
    """
    key, direction = spec["criteria"][0]
    if _shift(spec, target, key, direction, spec["attrs"][gold][key],
              better=True):
        return
    if _shift(spec, gold, key, direction, spec["attrs"][target][key],
              better=False):
        return
    raise ValueError(f"cannot separate {key}: grid exhausted at "
                     f"{spec['attrs'][gold][key]}")


def apply_variant(base: dict, kind: str, rng: random.Random) -> dict:
    """La variante `kind` del caso base, con su gold ya comprobado.

    * `number`    — un solo número cambia y el gold cambia con él.
    * `rule`      — en comparación cambia el criterio; en prioridad
      cambia SÓLO el orden de los criterios. El gold cambia.
    * `paraphrase`— plantilla, formato y orden de las frases. Ni un
      número, ni un criterio, ni el gold.

    Comprobado, no supuesto: si el cambio no mueve el gold, la función
    insiste de forma determinista hasta que lo mueve, y si no puede,
    falla en vez de publicar un contrafactual que no lo es.
    """
    if kind not in VARIANTS:
        raise ValueError(f"unknown variant {kind!r}: {VARIANTS}")
    if kind == "base":
        return _copy(base)
    spec = _copy(base)
    spec["cf_kind"] = kind
    gold = base["gold"]
    if kind == "paraphrase":
        tpls = len(STATE_TPL[spec["lang"]])
        spec["template"] = (base["template"] + 1 + rng.randrange(tpls - 1)) \
            % tpls
        spec["styles"] = {
            key: _rotate(STYLES[ATTR_BY_KEY[key]["kind"]],
                         base["styles"][key])
            for key in base["styles"]}
        spec["clause_order"] = list(reversed(base["clause_order"]))
        _regold(spec)
        if spec["gold"] != gold:
            raise ValueError("a paraphrase moved the gold: not a paraphrase")
        return spec
    if kind == "number":
        pool = eligible(spec["attrs"], spec["filters"])
        others = [i for i in pool if i != gold] or \
            [i for i in range(spec["k"]) if i != gold]
        # SORTEADO, no en orden de índice: recorrer 0, 1, 2… y quedarse
        # con el primero que mueva el gold empuja el gold nuevo hacia las
        # primeras posiciones (medido el 2026-09-27: 188 de 668 en la
        # posición 0 con K=8, χ² 173 contra un crítico de 18,5).
        rng.shuffle(others)
        for target in others:
            probe = _copy(spec)
            try:
                _beat(probe, target, gold)
            except ValueError:
                continue  # rejilla agotada para ESE par; se prueba otro
            _regold(probe)
            if probe["gold"] != gold:
                return probe
        raise ValueError("no single number moves the gold")
    # kind == "rule"
    if spec["family"] == EC.PRIORITY:
        spec["criteria"] = [spec["criteria"][1], spec["criteria"][0]] \
            + list(spec["criteria"][2:])
        _regold(spec)
        if spec["gold"] != gold:
            return spec
        # El orden sólo manda si los criterios discrepan: se fuerza la
        # discrepancia moviendo el SEGUNDO criterio, no el primero.
        key, direction = spec["criteria"][0]
        targets = [i for i in range(spec["k"]) if i != gold]
        rng.shuffle(targets)
        for target in targets:
            probe = _copy(spec)
            probe["criteria"] = list(spec["criteria"])
            try:
                _beat_key(probe, target, gold, key, direction)
            except ValueError:
                continue
            _regold(probe)
            if probe["gold"] != gold:
                return probe
        raise ValueError("swapping the criteria order does not move the gold")
    used = {k for k, _ in spec["criteria"]} | {f[0] for f in spec["filters"]}
    swaps = [k for k in spec["keys"] if k not in used] \
        or [k for k in spec["keys"] if k != spec["criteria"][0][0]]
    rng.shuffle(swaps)
    for key in swaps:
        probe = _copy(spec)
        probe["criteria"] = [(key, ATTR_BY_KEY[key]["better"])] \
            + list(spec["criteria"][1:])
        if probe["form"] == "ordinal" and key not in ORDINAL_KEYS:
            continue
        _regold(probe)
        if probe["gold"] != gold:
            return probe
    for key in swaps:
        probe = _copy(spec)
        probe["criteria"] = [(key, ATTR_BY_KEY[key]["better"])] \
            + list(spec["criteria"][1:])
        if probe["form"] == "ordinal" and key not in ORDINAL_KEYS:
            continue
        targets = [i for i in range(spec["k"]) if i != gold]
        rng.shuffle(targets)
        for target in targets:
            probe2 = _copy(probe)
            probe2["criteria"] = list(probe["criteria"])
            try:
                _beat(probe2, target, gold)
            except ValueError:
                continue
            _regold(probe2)
            if probe2["gold"] != gold:
                return probe2
    raise ValueError("no criterion change moves the gold")


def _beat_key(spec: dict, target: int, gold: int, key: str,
              direction: str) -> None:
    saved = spec["criteria"]
    spec["criteria"] = [(key, direction)]
    try:
        _beat(spec, target, gold)
    finally:
        spec["criteria"] = saved


def _rotate(seq, current):
    seq = list(seq)
    return seq[(seq.index(current) + 1) % len(seq)] if current in seq \
        else seq[0]


# -- renderizado: de la especificación al episodio ------------------------
def _pred(attr: dict, direction: str, lang: str) -> str:
    """El predicado del superlativo. Perífrasis cuando no hay adjetivo."""
    got = (attr["pred"].get(lang) or {}).get(direction)
    if got:
        return got
    noun = attr["noun"][lang]
    if lang == "es":
        return f"tiene {'menos' if direction == 'min' else 'más'} {noun}"
    return f"has the {'least' if direction == 'min' else 'most'} {noun}"


def _ranked_noun(attr: dict, direction: str, lang: str) -> str:
    """«el menor precio» / «la mayor batería» / «the lowest price»."""
    noun = attr["noun"][lang]
    if lang == "es":
        art = "el" if attr["gender"] == "m" else "la"
        return f"{art} {'menor' if direction == 'min' else 'mayor'} {noun}"
    return f"the {'lowest' if direction == 'min' else 'highest'} {noun}"


def _join(bits: list, lang: str) -> str:
    """«a, b y c» / «a, b and c». Un solo sitio, prosa y opciones igual."""
    if len(bits) == 1:
        return bits[0]
    joiner = " y " if lang == "es" else " and "
    return ", ".join(bits[:-1]) + joiner + bits[-1]


def entity(spec: dict, i: int) -> str:
    return f"{spec['noun']} {spec['names'][i]}"


def clause(spec: dict, i: int) -> str:
    """La frase de una opción, con todos los atributos que van en el estado."""
    lang = spec["lang"]
    bits = []
    for key in spec["keys"]:
        attr = ATTR_BY_KEY[key]
        value = render_value(attr, spec["attrs"][i][key], lang,
                             spec["styles"][key])
        bits.append(attr["vp"][lang].format(v=value))
    head = ("El" if lang == "es" else "The") + f" {entity(spec, i)}"
    return f"{head} {_join(bits, lang)}"


def state_of(spec: dict) -> str:
    tpls = STATE_TPL[spec["lang"]]
    clauses = " ".join(f"{clause(spec, i)}." for i in spec["clause_order"])
    return tpls[spec["template"] % len(tpls)].format(clauses=clauses)


def _threshold_words(spec: dict, discard: bool) -> str:
    """El umbral en prosa. `discard=True` lo dice al revés: qué se descarta."""
    key, op, bound = spec["filters"][0]
    attr = ATTR_BY_KEY[key]
    value = render_value(attr, bound, spec["lang"], spec["styles"][key])
    keep_below = op == "le"
    if discard:
        # El descarte es el COMPLEMENTO estricto de lo que se conserva:
        # «por encima de X» es exactamente `> X`, sin ambigüedad.
        side = "por encima de" if keep_below else "por debajo de"
        if spec["lang"] == "es":
            return f"{attr['noun']['es']} {side} {value}"
        return f"{attr['noun']['en']} {'above' if keep_below else 'below'} " \
               f"{value}"
    # Lo que se conserva es inclusivo (`<=` / `>=`), así que se dice
    # inclusivo: «por debajo de 50 €» no significa «50 € o menos».
    if spec["lang"] == "es":
        side = "o menos" if keep_below else "o más"
        return f"{attr['noun']['es']} de {value} {side}"
    return f"{attr['noun']['en']} of {value} " \
           f"{'or less' if keep_below else 'or more'}"


def question_of(spec: dict) -> str:
    """La pregunta de la forma del caso. Prosa con el criterio explícito."""
    lang, noun, form = spec["lang"], spec["noun"], spec["form"]
    first = ATTR_BY_KEY[spec["criteria"][0][0]]
    dir1 = spec["criteria"][0][1]
    if form == "superlative":
        pred = _pred(first, dir1, lang)
        return f"¿Qué {noun} {pred}?" if lang == "es" \
            else f"Which {noun} {pred}?"
    if form == "ordinal":
        ordinal = _ORD[lang][spec["rank"]]
        adj = first["adj"][lang][dir1]
        return f"¿Qué {noun} es el {ordinal} más {adj}?" if lang == "es" \
            else f"Which {noun} is the {ordinal} {adj}?"
    if form == "threshold":
        pred = _pred(first, dir1, lang)
        cut = _threshold_words(spec, discard=False)
        return f"¿Qué {noun} {pred}, entre los de {cut}?" if lang == "es" \
            else f"Which {noun} {pred}, among those with {cut}?"
    if form == "tie":
        pred1 = _pred(first, dir1, lang)
        second = ATTR_BY_KEY[spec["criteria"][1][0]]
        pred2 = _pred(second, spec["criteria"][1][1], lang)
        if lang == "es":
            return (f"¿Qué {noun} {pred1} y, a igualdad de "
                    f"{first['noun']['es']}, {pred2}?")
        return (f"Which {noun} {pred1} and, with equal "
                f"{first['noun']['en']}, {pred2}?")
    chain = [_ranked_noun(ATTR_BY_KEY[k], d, lang)
             for k, d in spec["criteria"]]
    if lang == "es":
        parts = [f"primero {chain[0]}"]
        parts += [f"después {c}" for c in chain[1:-1]]
        parts.append(f"y sólo para deshacer empates {chain[-1]}")
        body = f"Criterios en este orden: {', '.join(parts)}."
        if not spec["filters"]:
            return f"{body} ¿Qué {noun} sale elegido?"
        head = f"Descarta los de {_threshold_words(spec, discard=True)}. "
        return f"{head}{body} ¿Qué {noun} sale elegido, entre los demás?"
    parts = [f"first {chain[0]}"]
    parts += [f"then {c}" for c in chain[1:-1]]
    parts.append(f"and only to break ties {chain[-1]}")
    body = f"Criteria in this order: {', '.join(parts)}."
    head = (f"Discard those with {_threshold_words(spec, discard=True)}. "
            if spec["filters"] else "")
    tail = ", among the rest" if spec["filters"] else ""
    return f"{head}{body} Which {noun} comes out chosen{tail}?"


def candidates_of(spec: dict) -> list:
    """Las K opciones, en el orden de `attrs`, con los valores decisivos.

    El contrato marca `COMPARATIVE_CONTEXT` True en las dos familias: la
    información decisiva vive DENTRO de las opciones, así que cada una
    lleva sus valores de los criterios y del umbral.
    """
    lang = spec["lang"]
    shown = [k for k, _ in spec["criteria"]]
    shown += [f[0] for f in spec["filters"] if f[0] not in shown]
    out = []
    for i in range(spec["k"]):
        vals = [render_value(ATTR_BY_KEY[k], spec["attrs"][i][k], lang,
                             spec["styles"][k]) for k in shown]
        head = ("El" if lang == "es" else "The") + f" {entity(spec, i)}"
        out.append({"id": f"c{i + 1}",
                    "text": f"{head}, {_join(vals, lang)}"})
    return out


def rule_trace_of(spec: dict) -> dict:
    return {
        "rule": "chained-argmin over integer minor units",
        "variety_version": VARIETY_VERSION,
        "attrs": spec["attrs"], "criteria": [list(c) for c in spec["criteria"]],
        "filters": [list(f) for f in spec["filters"]], "rank": spec["rank"],
        "form": spec["form"], "k": spec["k"], "cf_kind": spec["cf_kind"],
        "keys": list(spec["keys"]), "styles": dict(spec["styles"]),
        "gold_index": spec["gold"],
        "entities": [entity(spec, i) for i in range(spec["k"])],
    }


def trace_gold(rule_trace: dict) -> int:
    """El gold recomputado desde `rule_trace` — ida y vuelta por JSON."""
    return rule_gold(rule_trace["attrs"],
                     [tuple(c) for c in rule_trace["criteria"]],
                     [tuple(f) for f in rule_trace["filters"]],
                     int(rule_trace.get("rank", 1)))


def case_rngs(seed: int, group: str, variant: str) -> tuple:
    """`(rng del caso, rng de la variante)`.

    El caso se sortea con el GRUPO, no con el índice del episodio: las
    cuatro variantes de un grupo tienen que partir del mismo caso base o
    no son contrafactuales de nada.
    """
    return (random.Random(f"{VARIETY_VERSION}\x00{seed}\x00{group}"),
            random.Random(f"{VARIETY_VERSION}\x00{seed}\x00{group}\x00{variant}"))


def build(family: str, lang: str, seed: int, idx: int, group: str,
          rng: random.Random, variant, prose: str = "local",
          sink: list | None = None, k: int = 2,
          variety: int = MIN_VARIETY, prose_hook=None) -> dict:
    """Un episodio de la familia numérica, con variedad y contrafactual.

    Firma compatible con los `build_*` de `data.episode_gen`; `rng` se
    ignora a propósito (ver `case_rngs`). `prose_hook` es
    `episode_gen._maybe_qwen`, inyectado para no importar en círculo.
    """
    kind = variant if isinstance(variant, str) else \
        ("number" if variant else "base")
    base_rng, var_rng = case_rngs(seed, group, kind)
    base = base_case(family, lang, k, base_rng, variety)
    spec = apply_variant(base, kind, var_rng)
    local = state_of(spec)
    question = question_of(spec)
    cands = candidates_of(spec)
    evidence = clause(spec, spec["gold"])
    tokens = [spec["names"][spec["gold"]]] + [
        render_value(ATTR_BY_KEY[key], spec["attrs"][spec["gold"]][key],
                     lang, spec["styles"][key])
        for key, _ in spec["criteria"]]
    if prose_hook is None:
        state, origin = local, "rule-local"
    else:
        state, origin = prose_hook(local, local, lang, prose, tokens, sink)
    return {
        "schema_version": EC.SCHEMA_VERSION,
        "family": family, "lang": lang, "generator_seed": seed,
        "generator_version": VARIETY_VERSION, "variant_group": group,
        "id": f"ep-{group}-{idx:02d}",
        "state": state, "question": question, "candidates": cands,
        "answer": cands[spec["gold"]]["id"], "evidence": evidence,
        "origin": origin, "rule_trace": rule_trace_of(spec),
    }


# -- posición del gold: la χ² escrita antes de medir ----------------------
def gold_position(ep: dict) -> int:
    ids = [c["id"] for c in ep["candidates"]]
    return ids.index(ep["answer"])


def position_chi2(episodes: list) -> dict:
    """χ² de uniformidad de la posición del gold, una por K.

    La regla y sus valores críticos son `POSITION_CHI2`, escritos antes
    del primer lote. Una celda con menos de `min_n_per_k` filas no vota:
    publica su estadístico y se declara sin potencia, no «uniforme».
    """
    counts: dict = {}
    for ep in episodes:
        k = len(ep["candidates"])
        counts.setdefault(k, [0] * k)[gold_position(ep)] += 1
    by_k = {}
    for k, col in sorted(counts.items()):
        n = sum(col)
        exp = n / k
        stat = sum((c - exp) ** 2 / exp for c in col) if exp else 0.0
        df = k - 1
        crit = _CHI2_CRIT.get(df)
        enough = n >= POSITION_CHI2["min_n_per_k"]
        by_k[str(k)] = {
            "n": n, "counts": col, "expected_per_cell": round(exp, 3),
            "statistic": round(stat, 4), "df": df, "critical_0.01": crit,
            "uniform": (None if (crit is None or not enough)
                        else bool(stat <= crit)),
            "status": ("measured" if (crit is not None and enough)
                       else "underpowered"),
        }
    voted = [v["uniform"] for v in by_k.values() if v["uniform"] is not None]
    return {"rule": POSITION_CHI2, "by_k": by_k,
            "pass": bool(voted) and all(voted)}


# -- contrafactuales: medidos sobre lo publicado -------------------------
#: Qué tiene que pasarle al gold en cada variante. `True` = cambia.
CF_EXPECT = {"base": None, "number": True, "rule": True, "paraphrase": False}


def counterfactual_report(episodes: list) -> dict:
    """Por grupo: `number` y `rule` mueven el gold, `paraphrase` no.

    Se recomputa el gold desde `rule_trace` con la MISMA función que lo
    publicó (`rule_gold`), así que un episodio cuyo `answer` no salga de
    la regla cuenta como fallo aquí, no como una curiosidad.
    """
    groups: dict = {}
    for ep in episodes:
        rt = ep.get("rule_trace") or {}
        if "criteria" not in rt:
            continue
        groups.setdefault(ep["variant_group"], {})[rt["cf_kind"]] = ep
    mismatched, wrong = [], []
    tested = {kind: 0 for kind in CF_EXPECT}
    ok = {kind: 0 for kind in CF_EXPECT}
    for group, variants in sorted(groups.items()):
        for kind, ep in sorted(variants.items()):
            if trace_gold(ep["rule_trace"]) != gold_position(ep):
                mismatched.append(ep["id"])
        base = variants.get("base")
        if base is None:
            continue
        base_pos = gold_position(base)
        for kind, expect in CF_EXPECT.items():
            ep = variants.get(kind)
            if ep is None or expect is None:
                continue
            tested[kind] += 1
            moved = gold_position(ep) != base_pos
            if moved == expect:
                ok[kind] += 1
            else:
                wrong.append({"group": group, "kind": kind, "id": ep["id"],
                              "moved": moved, "expected_move": expect})
    measured = {k: {"n": tested[k], "ok": ok[k], "expected_move": CF_EXPECT[k]}
                for k in ("number", "rule", "paraphrase")}
    return {
        "n_groups": len(groups),
        "gold_recomputes_from_rule": {
            "pass": not mismatched,
            "n_checked": sum(len(v) for v in groups.values()),
            "mismatched": mismatched[:8]},
        "by_variant": measured,
        "failures": wrong[:8],
        "pass": bool(groups) and not mismatched and not wrong
        and all(v["n"] > 0 for v in measured.values()),
    }


# -- fuga: vocabulario y paráfrasis, contra dev Y sellado ----------------
DEV_BATTERY = os.path.join(ROOT, "data", "battery_dev.jsonl")
SEALED_BATTERY = os.path.join(ROOT, "data", "battery_sealed.jsonl")


def vocabulary() -> list:
    """Los términos del vocabulario de entidades, normalizados."""
    terms = set()
    for lang in ("es", "en"):
        terms.update(ENTITY_NOUNS[lang])
        terms.update(CODENAMES[lang])
    return sorted(LK.normalize(t) for t in terms)


def _battery_texts(path: str) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            ep = json.loads(line)
            out.append(ep.get("state", ""))
            out.append(ep.get("question", ""))
            out.extend(c.get("text", "") for c in ep.get("candidates", []))
    return [t for t in out if t]


def leakage_report(episodes: list, sample_n: int = 400,
                   seed: int = 20260927) -> dict:
    """Dos capas contra las DOS baterías: vocabulario y paráfrasis.

    1. Ningún término del vocabulario de entidades aparece en dev ni en
       el sellado — ésa es la garantía fuerte, y es exacta.
    2. `data.leakage.LeakageDetector` sobre una muestra sembrada de
       estados y preguntas: hash exacto y Jaccard de 3-gramas >= 0,5. La
       muestra es lo que hace la comprobación viable en un lote de
       50 000; su n se publica, no se esconde.

    El sellado se abre sólo para esto: se leen sus textos para PROHIBIRLOS,
    no se puntúa ni se mide nada en él (`docs/bucle-infinito.md` §5).
    """
    layers = {}
    for name, path in (("battery_dev", DEV_BATTERY),
                       ("battery_sealed", SEALED_BATTERY)):
        texts = _battery_texts(path)
        if not texts:
            layers[name] = {"present": False, "vocabulary_hits": [],
                            "paraphrase_hits": [], "pass": None}
            continue
        blob = LK.normalize(" | ".join(texts))
        hits = [t for t in vocabulary() if t in blob]
        rng = random.Random(f"{VARIETY_VERSION}-leak-{seed}")
        pool = list(episodes)
        take = pool if len(pool) <= sample_n else rng.sample(pool, sample_n)
        det = LK.LeakageDetector(texts)
        sampled = ([ep["state"] for ep in take]
                   + [ep["question"] for ep in take])
        para = det.scan_all(sampled)
        layers[name] = {
            "present": True, "n_blocked_texts": len(texts),
            "vocabulary_hits": hits,
            "n_sampled_texts": len(sampled), "sample_seed": seed,
            "paraphrase_hits": [{"text": t[:90], "why": why}
                                for t, why in para[:6]],
            "n_paraphrase_hits": len(para),
            "pass": not hits and not para,
        }
    voted = [v["pass"] for v in layers.values() if v["pass"] is not None]
    return {"n_vocabulary_terms": len(vocabulary()), "layers": layers,
            "sim_threshold": 0.5,
            "pass": bool(voted) and all(voted)}


# -- variedad realizada: lo que de verdad salió --------------------------
def variety_report(episodes: list) -> dict:
    """Atributos, formatos, formas y unidades REALIZADOS en el lote."""
    attrs, styles, forms, nouns = set(), set(), set(), set()
    ks, langs = set(), set()
    for ep in episodes:
        rt = ep.get("rule_trace") or {}
        if "criteria" not in rt:
            continue
        attrs.update(rt["keys"])
        styles.update(f"{k}:{v}" for k, v in rt["styles"].items())
        forms.add(f"{ep['family']}/{rt['form']}")
        nouns.add(rt["entities"][0].rsplit(" ", 1)[0])
        ks.add(rt["k"])
        langs.add(ep["lang"])
    return {
        "n_attributes": len(attrs), "attributes": sorted(attrs),
        "n_number_styles": len(styles),
        "n_question_forms": len(forms), "question_forms": sorted(forms),
        "n_entity_nouns": len(nouns), "ks": sorted(ks),
        "langs": sorted(langs),
        "min_attributes_required": MIN_VARIETY,
        "pass": (len(attrs) >= MIN_VARIETY and len(forms) >= 6
                 and sorted(ks) == list(KS)
                 and sorted(langs) == sorted(EC.LANGS)),
    }


# -- publicar un lote y firmar su gate -----------------------------------
GATE_PATH = os.path.join(ROOT, "artifacts", "gates", TASK, "gate.json")


def batch_dir(batch: int) -> str:
    return os.path.join(BATCH_ROOT, f"batch-{batch:04d}")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def publish(batch: int, n: int, variety: int = MIN_VARIETY,
            out_dir: str = "", prose: str = "local",
            teacher: str = "stub", sample_n: int = 400) -> dict:
    """Publica el lote `batch` con semilla = `batch` y firma su manifest.

    El manifest es lo único versionado: lleva el comando, la semilla, la
    variedad, el reparto declarado y el sha256 de `episodes.jsonl`, que
    es lo que hace el lote reconstruible sin guardar las filas.
    """
    from data import episode_gen as EG  # aquí, para no importar en círculo

    if batch < 1:
        raise ValueError("el número de lote empieza en 1 (es la semilla)")
    out = out_dir or batch_dir(batch)
    command = (f"python3 -m data.rule_variety publish --batch {batch} "
               f"--n {n} --variety {variety}")
    stats = EG.run(n=n, seed=batch, prose=prose, teacher=teacher,
                   out_dir=out, command=command, families=list(FAMILIES),
                   variety=variety, group_prefix=f"rule{batch:04d}-")
    episodes = EG.load_episodes(out)
    ep_path = os.path.join(out, "episodes.jsonl")
    mpath = os.path.join(out, "manifest.json")
    manifest = json.load(open(mpath, encoding="utf-8"))
    manifest.update({
        "task": TASK, "batch": batch, "variety": variety,
        "variety_version": VARIETY_VERSION,
        "families": list(FAMILIES), "ks": list(KS),
        "variants": list(VARIANTS),
        "seed_is_batch_number": True,
        "episodes_sha256": _sha256_file(ep_path),
        "episodes_versioned": False,
        "rebuild": command,
        "variety_realised": variety_report(episodes),
        "gold_position_chi2": position_chi2(episodes),
        "counterfactuals": counterfactual_report(episodes),
        "leakage": leakage_report(episodes, sample_n=sample_n),
    })
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return {"out": out, "n_episodes": stats["n_episodes"],
            "n_rejects": stats["n_rejects"], "manifest": mpath,
            "variety": manifest["variety_realised"],
            "chi2": manifest["gold_position_chi2"],
            "counterfactuals": manifest["counterfactuals"],
            "leakage": manifest["leakage"]}


def read_gate() -> dict:
    if os.path.exists(GATE_PATH):
        with open(GATE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {"task": TASK}


def write_gate(section: str, payload: dict) -> dict:
    """Escribe UNA sección del gate de la task sin pisar las demás.

    El veredicto se recalcula desde las secciones que hay: una sección
    aún sin medir es `null` y no vota (regla C7).
    """
    gate = read_gate()
    gate["task"] = TASK
    gate[section] = payload
    votes = [gate[key].get("pass") for key in sorted(gate)
             if isinstance(gate.get(key), dict) and "pass" in gate[key]]
    voted = [v for v in votes if v is not None]
    gate["pass"] = bool(voted) and all(voted)
    os.makedirs(os.path.dirname(GATE_PATH), exist_ok=True)
    with open(GATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(gate, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return gate


def gate_generator(out: str, required: int = 50000,
                   sample_n: int = 400) -> dict:
    """La sección `generator` del gate, medida sobre el lote publicado."""
    from data import episode_gen as EG

    episodes = EG.load_episodes(out)
    mpath = os.path.join(out, "manifest.json")
    manifest = json.load(open(mpath, encoding="utf-8"))
    per = EC.batch_validate(episodes)
    checks = {
        "validator_pass_100pc": {
            "pass": per["n_invalid"] == 0 and not per["duplicate_ids"],
            "n": per["n"], "n_invalid": per["n_invalid"],
            "duplicate_ids": per["duplicate_ids"][:5]},
        "variety": variety_report(episodes),
        "gold_position_chi2": position_chi2(episodes),
        "counterfactuals": counterfactual_report(episodes),
        "leakage": leakage_report(episodes, sample_n=sample_n),
        "volume": {
            "pass": len(episodes) >= required,
            "n_published": len(episodes), "required": required,
            "dir": os.path.relpath(out, ROOT),
            "seed": manifest.get("seed"),
            "episodes_sha256": manifest.get("episodes_sha256"),
            "manifest_sha256": _sha256_file(mpath)},
    }
    payload = {
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                      time.gmtime()),
        "dir": os.path.relpath(out, ROOT),
        "variety_version": VARIETY_VERSION,
        "schema_sha": EC.schema_sha(),
        "command": manifest.get("command"),
        "checks": checks,
        "pass": all(v["pass"] for v in checks.values()
                    if v.get("pass") is not None),
    }
    return write_gate("generator", payload)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m data.rule_variety")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish", help="publish one rule batch (seed = batch)")
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--variety", type=int, default=MIN_VARIETY)
    p.add_argument("--out", default="")
    p.add_argument("--sample-n", type=int, default=400)
    g = sub.add_parser("gate", help="measure the generator gate on a batch")
    g.add_argument("--dir", required=True)
    g.add_argument("--required", type=int, default=50000)
    g.add_argument("--sample-n", type=int, default=400)
    args = ap.parse_args(argv)

    if args.cmd == "publish":
        got = publish(args.batch, args.n, variety=args.variety,
                      out_dir=args.out, sample_n=args.sample_n)
        print(f"episodes={got['n_episodes']} rejects={got['n_rejects']} "
              f"-> {got['out']}")
        print(json.dumps({"variety": got["variety"]["pass"],
                          "chi2": got["chi2"]["pass"],
                          "counterfactuals": got["counterfactuals"]["pass"],
                          "leakage": got["leakage"]["pass"]}, sort_keys=True))
        return 0 if got["n_episodes"] else 1
    gate = gate_generator(args.dir, required=args.required,
                          sample_n=args.sample_n)
    print(json.dumps({k: v.get("pass") for k, v
                      in gate["generator"]["checks"].items()},
                     sort_keys=True))
    return 0 if gate["generator"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


def perturb_flips(rule_trace: dict) -> int:
    """1 si mover UN número del primer criterio cambia el gold de la regla.

    Lo que mide el test 2 del gate de `data.episode_gen` sobre las filas
    con variedad: el generador no puede publicar un gold que la regla
    contradiga, y un número del criterio que manda tiene que moverlo.
    """
    attrs = [dict(a) for a in rule_trace["attrs"]]
    criteria = [tuple(c) for c in rule_trace["criteria"]]
    filters = [tuple(f) for f in rule_trace["filters"]]
    rank = int(rule_trace.get("rank", 1))
    before = rule_gold(attrs, criteria, filters, rank)
    key, direction = criteria[0]
    step = ATTR_BY_KEY[key]["step"]
    pool = eligible(attrs, filters)
    others = [i for i in pool if i != before] or \
        [i for i in range(len(attrs)) if i != before]
    for target in others:
        probe = [dict(a) for a in attrs]
        probe[target][key] = (attrs[before][key]
                              - _sign(direction) * step * 3)
        try:
            if rule_gold(probe, criteria, filters, rank) != before:
                return 1
        except ValueError:
            continue
    return 0
