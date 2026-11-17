"""Synthetic rungs that lead up to the real product comparison task.

Every rung computes its gold from the printed numbers, so no model decides
an answer. The rungs add one difficulty at a time:

    r0-most         two options, one nutrient, always "the most"
    r1-pair         two options, one nutrient, most or least
    r2-panel        2 to 4 full nutrition panels, no product names
    r3-units        3 to 5 named panels, half of them print sodium in mg
    r4-constraints  a cap on a second nutrient that excludes the naive pick

The last rung, r5-real, is the Open Food Facts set from `data.off_products`
and is read from disk, not generated here.
"""
from __future__ import annotations

import random

NUTRIENTS = ("energy", "fat", "saturates", "sugars", "fibre", "protein", "salt")
RANGES = {"energy": (0, 900), "fat": (0, 100), "saturates": (0, 60),
          "sugars": (0, 100), "fibre": (0, 30), "protein": (0, 60),
          "salt": (0, 8)}
LABELS = {
    "en": {"energy": "energy", "fat": "fat", "saturates": "of which saturates",
           "sugars": "sugars", "fibre": "fibre", "protein": "protein",
           "salt": "salt", "sodium": "sodium"},
    "es": {"energy": "energía", "fat": "grasas",
           "saturates": "de las cuales saturadas", "sugars": "azúcares",
           "fibre": "fibra", "protein": "proteínas", "salt": "sal",
           "sodium": "sodio"},
}
ASKED = {
    "en": {"energy": "energy", "fat": "fat", "saturates": "saturated fat",
           "sugars": "sugar", "fibre": "fibre", "protein": "protein",
           "salt": "salt"},
    "es": {"energy": "energía", "fat": "grasa", "saturates": "grasa saturada",
           "sugars": "azúcar", "fibre": "fibra", "protein": "proteína",
           "salt": "sal"},
}
EXTREME = {
    "en": {True: ["Which product has the most {n} per 100 g?",
                  "Which product gives the most {n}?",
                  "I want the option with the highest {n}. Which one?"],
           False: ["Which product has the least {n} per 100 g?",
                   "Which product gives the least {n}?",
                   "I want the option with the lowest {n}. Which one?"]},
    "es": {True: ["¿Qué producto tiene más {n} por 100 g?",
                  "¿Cuál aporta más {n}?",
                  "Quiero la opción con más {n}. ¿Cuál es?"],
           False: ["¿Qué producto tiene menos {n} por 100 g?",
                   "¿Cuál aporta menos {n}?",
                   "Quiero la opción con menos {n}. ¿Cuál es?"]},
}
CONSTRAINED = {
    "en": "Among the products with at most {x} g of {c} per 100 g, "
          "which has the {w} {n}?",
    "es": "Entre los productos con como máximo {x} g de {c} por 100 g, "
          "¿cuál tiene {w} {n}?",
}
WORDS = {"en": {True: "most", False: "least"},
         "es": {True: "más", False: "menos"}}
STATES = {
    "en": "Comparing {item} on the shelf. Nutrition values are per 100 g "
          "as printed on each pack.",
    "es": "Comparando {item} en el lineal. Los valores nutricionales son por "
          "100 g, tal como figuran en cada envase.",
}
ITEMS = {"en": ["breakfast cereals", "yoghurts", "crackers", "soups",
                "granola bars", "biscuits", "sauces", "breads"],
         "es": ["cereales de desayuno", "yogures", "galletas saladas",
                "sopas", "barritas de cereales", "galletas", "salsas",
                "panes"]}
BRANDS = ["Nordvik", "La Huerta", "Alba", "Kestrel", "Molino Real", "Brightfield",
          "Casa Mar", "Oakden", "Verda", "Tisbury", "El Prado", "Hollin"]
MARGIN = 0.15
RUNGS = ("r0-most", "r1-pair", "r2-panel", "r3-units", "r4-constraints")


def _value(rng: random.Random, nutrient: str) -> float:
    low, high = RANGES[nutrient]
    value = rng.uniform(low, high)
    if nutrient == "energy":
        return float(round(value))
    return round(value, rng.choice((0, 1)))


def _fmt(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:.1f}"


def _separated(values: list[float], best: int) -> bool:
    """The winner beats every rival by the same 15% margin as the real set."""
    top = values[best]
    rivals = [v for i, v in enumerate(values) if i != best]
    if not rivals:
        return False
    scale = max(abs(top), 1e-9)
    return all(abs(top - v) / scale >= MARGIN and abs(top - v) >= 0.2
               for v in rivals)


def _pick(values: list[float], most: bool) -> int:
    return max(range(len(values)), key=lambda i: values[i] * (1 if most else -1))


def _panel(rng: random.Random, lang: str, panel: dict, sodium: bool,
           name: str | None) -> str:
    labels = LABELS[lang]
    parts = []
    for nutrient in NUTRIENTS:
        if nutrient == "energy":
            parts.append(f"{labels['energy']} {_fmt(panel['energy'])} kcal")
        elif nutrient == "salt" and sodium:
            parts.append(f"{labels['sodium']} {round(panel['salt'] * 400)} mg")
        else:
            parts.append(f"{labels[nutrient]} {_fmt(panel[nutrient])} g")
    body = ", ".join(parts) + "."
    return f"{name}. {body}" if name else body


def _episode(rid: str, lang: str, state: str, question: str,
             texts: list[str], gold: int, rung: str) -> dict:
    return {
        "id": rid,
        "state": state,
        "question": question,
        "candidates": [{"id": f"c{i}", "text": t} for i, t in enumerate(texts)],
        "answer": f"c{gold}",
        "lang": lang,
        "capability": "product_specs",
        "origin": f"curriculum:{rung}",
    }


def _name(rng: random.Random, lang: str, item: str) -> str:
    return f"{rng.choice(BRANDS)} · {item.capitalize()} {rng.randint(1, 99)}"


def make(rung: str, rng: random.Random, index: int) -> dict:
    lang = rng.choice(("en", "es"))
    item = rng.choice(ITEMS[lang])
    state = STATES[lang].format(item=item)
    # A half-most, half-least mix cancels the gradient of any magnitude
    # feature at random init, so the first rung only asks for the most.
    most = rung == "r0-most" or rng.random() < 0.5
    target = rng.choice(NUTRIENTS)
    rid = f"{rung}-{index}"
    if rung in ("r0-most", "r1-pair"):
        while True:
            values = [_value(rng, target) for _ in range(2)]
            gold = _pick(values, most)
            if _separated(values, gold):
                break
        unit = "kcal" if target == "energy" else "g"
        texts = [f"{LABELS[lang][target]} {_fmt(v)} {unit}" for v in values]
        question = rng.choice(EXTREME[lang][most]).format(n=ASKED[lang][target])
        return _episode(rid, lang, state, question, texts, gold, rung)
    k = rng.choice((2, 3, 4)) if rung == "r2-panel" else rng.choice((3, 4, 5))
    sodium_mix = rung in ("r3-units", "r4-constraints")
    named = rung != "r2-panel"
    if rung == "r4-constraints" and rng.random() < 0.6:
        return _constrained(rid, lang, state, item, k, most, rng)
    while True:
        panels = [{n: _value(rng, n) for n in NUTRIENTS} for _ in range(k)]
        values = [p[target] for p in panels]
        gold = _pick(values, most)
        if _separated(values, gold):
            break
    texts = [_panel(rng, lang, p, sodium_mix and rng.random() < 0.5,
                    _name(rng, lang, item) if named else None) for p in panels]
    question = rng.choice(EXTREME[lang][most]).format(n=ASKED[lang][target])
    return _episode(rid, lang, state, question, texts, gold, rung)


def _constrained(rid: str, lang: str, state: str, item: str, k: int,
                 most: bool, rng: random.Random) -> dict:
    nutrients = [n for n in NUTRIENTS if n != "energy"]
    while True:
        cap, target = rng.sample(nutrients, 2)
        panels = [{n: _value(rng, n) for n in NUTRIENTS} for _ in range(k)]
        caps = sorted(p[cap] for p in panels)
        limit = round(rng.uniform(caps[1], caps[-1]), 1)
        if any(abs(p[cap] - limit) <= max(0.1 * limit, 0.2) for p in panels):
            continue
        allowed = [i for i, p in enumerate(panels) if p[cap] <= limit]
        if len(allowed) < 2 or len(allowed) == k:
            continue
        values = [panels[i][target] for i in allowed]
        best = _pick(values, most)
        if not _separated(values, best):
            continue
        gold = allowed[best]
        outside = [i for i in range(k) if i not in allowed]
        # The trap: an excluded product that would win without the cap.
        if not any((panels[i][target] > panels[gold][target]) == most
                   for i in outside):
            continue
        break
    texts = [_panel(rng, lang, p, rng.random() < 0.5, _name(rng, lang, item))
             for p in panels]
    question = CONSTRAINED[lang].format(
        x=_fmt(limit), c=ASKED[lang][cap], w=WORDS[lang][most],
        n=ASKED[lang][target])
    return _episode(rid, lang, state, question, texts, gold, "r4-constraints")


def batch(rung: str, rows: int, seed: int, replay: float = 0.0) -> list[dict]:
    """Rows of one rung, with a share of earlier rungs so they stay learned."""
    rng = random.Random(seed)
    earlier = list(RUNGS[:RUNGS.index(rung)]) if rung in RUNGS else list(RUNGS)
    out = []
    for index in range(rows):
        source = rung
        if earlier and rng.random() < replay:
            source = rng.choice(earlier)
        out.append(make(source, rng, seed * 1_000_000 + index))
    return out
