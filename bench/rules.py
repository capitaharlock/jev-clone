"""A hand-written program that answers the benchmark from the text alone.

It parses the nutrient, the direction and the optional cap out of the
question, reads every card's numbers, converts sodium in mg to salt in g,
and picks the extreme. It knows the question templates, so its score is
the ceiling of the task, not a fair rival: it shows that every gold label
follows from what the model is given.
"""
from __future__ import annotations

import re

NUTRIENTS = {
    "en": [("saturated fat", "saturates"), ("sugar", "sugars"),
           ("salt", "salt"), ("fat", "fat"), ("protein", "protein"),
           ("fibre", "fibre"), ("calories", "energy")],
    "es": [("grasa saturada", "saturates"), ("azúcar", "sugars"),
           ("sal", "salt"), ("grasa", "fat"), ("proteína", "protein"),
           ("fibra", "fibre"), ("calorías", "energy")],
}
CARD = {
    "energy": r"(?:energy|energía) ([\d.]+) kcal",
    "fat": r"(?:fat|grasas) ([\d.]+) g",
    "saturates": r"(?:of which saturates|de las cuales saturadas) ([\d.]+) g",
    "sugars": r"(?:sugars|azúcares) ([\d.]+) g",
    "fibre": r"(?:fibre|fibra) ([\d.]+) g",
    "protein": r"(?:protein|proteínas) ([\d.]+) g",
    "salt": r"(?:salt|sal) ([\d.]+) g",
    "sodium": r"(?:sodium|sodio) ([\d.]+) mg",
}
LEAST = re.compile(r"\b(least|lowest|minimum|menos|mínimo)\b")
CAP = re.compile(r"(?:at most|at or below|como máximo|pasar de) ([\d.]+) g")


def language(question: str) -> str:
    return "es" if re.search(r"[¿áéíóúñ]", question) else "en"


def mentions(question: str, lang: str) -> list[str]:
    """Nutrients named in the question, in reading order."""
    text = question.lower()
    found = []
    taken = [False] * len(text)
    for word, key in NUTRIENTS[lang]:
        for match in re.finditer(rf"\b{re.escape(word)}\b", text):
            if not any(taken[match.start():match.end()]):
                found.append((match.start(), key))
                for i in range(match.start(), match.end()):
                    taken[i] = True
    return [key for _, key in sorted(found)]


def panel(text: str) -> dict[str, float]:
    # The body starts at the last "energy"; a product name may say "Energy".
    start = max(text.lower().rfind("energy "), text.lower().rfind("energía "))
    body = text[start:] if start >= 0 else text
    values = {}
    for key, pattern in CARD.items():
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            values[key] = float(match.group(1))
    if "salt" not in values and "sodium" in values:
        values["salt"] = values["sodium"] * 2.5 / 1000
    return values


def decide(row: dict) -> str | None:
    """The id the rules pick, or None when the question does not parse."""
    question = row["question"]
    lang = language(question)
    named = mentions(question, lang)
    if not named:
        return None
    target = named[-1]
    least = bool(LEAST.search(question.lower()))
    cards = [(c["id"], panel(c["text"])) for c in row["candidates"]]
    cap = CAP.search(question)
    if cap and len(named) >= 2:
        limit, capped = float(cap.group(1)), named[0]
        cards = [(i, v) for i, v in cards if v.get(capped, 1e9) <= limit]
    cards = [(i, v) for i, v in cards if target in v]
    if not cards:
        return None
    pick = min if least else max
    return pick(cards, key=lambda item: item[1][target])[0]
