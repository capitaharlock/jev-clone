"""Product comparison episodes built from the Open Food Facts dump (ODbL 1.0).

Every option is a real product card with its printed nutrition per 100 g.
Gold is computed from those numbers by a program, and an episode is kept
only when the winner beats the runner-up by a clear margin, so no label
comes from a model or an annotator.

Leakage control works at the product level. Each barcode is hashed to the
training or the evaluation side before any episode is built, so a product
never appears on both sides. Evaluation groups are hashed a second time:
the lower quarter is the sealed cut, which this repository never writes.

Some cards print sodium in milligrams instead of salt in grams. Reading
them needs a unit conversion (salt = 2.5 x sodium), not a lookup.

A cap is checked on the measured values, but a card prints them rounded.
When a product sits on the cap after rounding (fibre 0.03 g printed as
"0 g" under "at most 0 g") the text no longer decides the answer. Such rows
are marked `decidable: false` and left out of the written files.

    python -m data.off_products path/to/en.openfoodfacts.org.products.csv.gz
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from data.io import valid_episode, write_json, write_jsonl

ATTRS = {
    "sugars": ("sugars_100g", 1.0),
    "salt": ("salt_100g", 0.1),
    "fat": ("fat_100g", 1.0),
    "saturated_fat": ("saturated-fat_100g", 0.5),
    "proteins": ("proteins_100g", 1.0),
    "fiber": ("fiber_100g", 0.5),
    "energy": ("energy-kcal_100g", 20.0),
}
REL_MARGIN = 0.15
COUNTRIES = {"Spain": "es", "United States": "en", "United Kingdom": "en"}
DEV_SHARE = 0.2
SEALED_SHARE = 0.25
MAX_PER_CATEGORY = 400
SEED = 20261015

NAMES = {
    "en": {"sugars": "sugar", "salt": "salt", "fat": "fat",
           "saturated_fat": "saturated fat", "proteins": "protein",
           "fiber": "fibre", "energy": "calories"},
    "es": {"sugars": "azúcar", "salt": "sal", "fat": "grasa",
           "saturated_fat": "grasa saturada", "proteins": "proteína",
           "fiber": "fibra", "energy": "calorías"},
}
EXTREME = {
    "en": {"min": ["Which product has the least {a} per 100 g?",
                   "I want the option with the lowest {a}. Which one?",
                   "Which product keeps {a} to a minimum?"],
           "max": ["Which product has the most {a} per 100 g?",
                   "I want the option with the highest {a}. Which one?",
                   "Which product gives the most {a}?"]},
    "es": {"min": ["¿Qué producto tiene menos {a} por 100 g?",
                   "Quiero la opción con menos {a}. ¿Cuál es?",
                   "¿Qué producto reduce al mínimo el contenido de {a}?"],
           "max": ["¿Qué producto tiene más {a} por 100 g?",
                   "Quiero la opción con más {a}. ¿Cuál es?",
                   "¿Qué producto aporta más {a}?"]},
}
CONSTRAINED = {
    "en": ["Among the products with at most {limit} g of {c} per 100 g, "
           "which has the {dir} {a}?",
           "{c_cap} must stay at or below {limit} g per 100 g. Within that "
           "limit, which product has the {dir} {a}?"],
    "es": ["Entre los productos con como máximo {limit} g de {c} por 100 g, "
           "¿cuál tiene {dir} {a}?",
           "El contenido de {c} no puede pasar de {limit} g por 100 g. Dentro "
           "de ese límite, ¿qué producto tiene {dir} {a}?"],
}
DIRECTION = {"en": {"min": "lowest", "max": "highest"},
             "es": {"min": "menos", "max": "más"}}
STATE = {
    "en": "Comparing {cat} on the shelf. Nutrition values are per 100 g as "
          "printed on each pack.",
    "es": "Comparando {cat} en el lineal. Los valores nutricionales son por "
          "100 g, tal como figuran en cada envase.",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unit(text: str) -> float:
    return int(_sha(text)[:8], 16) / 0xFFFFFFFF


def sealed(group_id: str) -> bool:
    """The salted hash that keeps the sealed cut out of this repository."""
    return _unit(f"v3-sealed/{group_id}") < SEALED_SHARE


def _float(text: str | None) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value == value else None


def _fmt(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}".rstrip("0").rstrip(".")


def read_products(path: Path):
    """Yield products with a full, internally consistent nutrition panel."""
    csv.field_size_limit(sys.maxsize)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            countries = (row.get("countries_en") or "").split(",")
            language = next((COUNTRIES[c] for c in countries
                             if c in COUNTRIES), None)
            name = (row.get("product_name") or "").strip()
            category = (row.get("main_category_en") or "").strip()
            if not language or not name or not category or ":" in category:
                continue
            values = {key: _float(row.get(column))
                      for key, (column, _) in ATTRS.items()}
            if any(value is None for value in values.values()):
                continue
            if any(not 0 <= values[key] <= 100
                   for key in values if key != "energy"):
                continue
            if not 0 < values["energy"] <= 900:
                continue
            # Rows whose sodium and salt disagree carry a typing error in
            # one of the two, and a sodium card would then lie.
            sodium = _float(row.get("sodium_100g"))
            if sodium is None or abs(sodium * 2.5 - values["salt"]) > \
                    max(0.05, 0.1 * values["salt"]):
                continue
            yield {
                "code": row["code"], "name": name[:80],
                "brand": (row.get("brands") or "").split(",")[0].strip()[:40],
                "quantity": (row.get("quantity") or "").strip()[:20],
                "category": category,
                "group": ((row.get("categories_en") or "").split(",")[0]
                          .strip() or category),
                "language": language, "values": values, "sodium": sodium,
            }


def card(product: dict, language: str, sodium_mg: bool) -> str:
    v = product["values"]
    head = " · ".join(x for x in (product["brand"], product["name"]) if x)
    if product["quantity"]:
        head += f" ({product['quantity']})"
    if language == "es":
        salt = (f"sodio {_fmt(product['sodium'] * 1000)} mg" if sodium_mg
                else f"sal {_fmt(v['salt'])} g")
        body = (f"energía {_fmt(v['energy'])} kcal, grasas {_fmt(v['fat'])} g, "
                f"de las cuales saturadas {_fmt(v['saturated_fat'])} g, "
                f"azúcares {_fmt(v['sugars'])} g, fibra {_fmt(v['fiber'])} g, "
                f"proteínas {_fmt(v['proteins'])} g, {salt}")
    else:
        salt = (f"sodium {_fmt(product['sodium'] * 1000)} mg" if sodium_mg
                else f"salt {_fmt(v['salt'])} g")
        body = (f"energy {_fmt(v['energy'])} kcal, fat {_fmt(v['fat'])} g, "
                f"of which saturates {_fmt(v['saturated_fat'])} g, "
                f"sugars {_fmt(v['sugars'])} g, fibre {_fmt(v['fiber'])} g, "
                f"protein {_fmt(v['proteins'])} g, {salt}")
    return f"{head}. {body}."


def printed(product: dict, nutrient: str, sodium_mg: bool) -> float:
    """The value a reader recovers from the card, after its rounding."""
    if nutrient == "salt" and sodium_mg:
        return float(_fmt(product["sodium"] * 1000)) * 2.5 / 1000
    return float(_fmt(product["values"][nutrient]))


def clear_winner(products: list[dict], attr: str, direction: str):
    ranked = sorted(products, key=lambda p: p["values"][attr],
                    reverse=direction == "max")
    best, second = ranked[0]["values"][attr], ranked[1]["values"][attr]
    gap = abs(best - second)
    if gap < ATTRS[attr][1] or gap < REL_MARGIN * max(best, second):
        return None
    return ranked[0]


def make_episode(products: list[dict], language: str, rng: random.Random,
                 category: str) -> dict | None:
    k = len(products)
    attr = rng.choice(list(ATTRS))
    direction = rng.choice(["min", "max"])
    names = NAMES[language]
    kind = "extreme"
    winner = constraint = None
    if rng.random() < 0.6:
        choices = [a for a in ATTRS if a not in (attr, "energy")]
        rng.shuffle(choices)
        for option in choices:
            values = sorted(p["values"][option] for p in products)
            limit = round((values[k // 2 - 1] + values[k // 2]) / 2, 1)
            feasible = [p for p in products if p["values"][option] <= limit]
            if not 2 <= len(feasible) < k:
                continue
            # Kept only when the cap changes the answer: the overall
            # winner is excluded, so ignoring the cap is always wrong.
            overall = clear_winner(products, attr, direction)
            local = clear_winner(feasible, attr, direction)
            if local and overall and overall is not local:
                winner, kind, constraint = local, "constrained", option
                template = rng.choice(CONSTRAINED[language])
                question = template.format(
                    limit=_fmt(limit), c=names[constraint],
                    c_cap=names[constraint].capitalize(),
                    dir=DIRECTION[language][direction], a=names[attr])
                break
    if winner is None:
        winner = clear_winner(products, attr, direction)
        if winner is None:
            return None
        question = rng.choice(EXTREME[language][direction]).format(
            a=names[attr])
    sodium_cards = attr == "salt" or (kind == "constrained"
                                      and constraint == "salt")
    use_mg = sodium_cards and rng.random() < 0.4
    texts = [card(p, language, use_mg and i % 2 == 0)
             for i, p in enumerate(products)]
    if len(set(texts)) != k:
        return None
    decidable = kind == "extreme" or all(
        (printed(p, constraint, use_mg and i % 2 == 0)
         <= float(_fmt(limit))) == (p["values"][constraint] <= limit)
        for i, p in enumerate(products))
    state = STATE[language].format(cat=category.lower())
    codes = "+".join(sorted(p["code"] for p in products))
    row = {
        "id": f"v3-prod-{_sha(codes + question)[:16]}",
        "state": state,
        "question": question,
        "candidates": [{"id": f"c{i}", "text": t} for i, t in enumerate(texts)],
        "answer": f"c{products.index(winner)}",
        "group_id": f"off/{codes}",
        "lang": language,
        "capability": "product_specs",
        "domain": f"food_{winner['group'].lower().replace(' ', '_')[:40]}",
        "difficulty": kind + ("+sodium_mg" if use_mg else ""),
        "decidable": decidable,
        "origin": "off-computed",
        "license": "ODbL-1.0",
        "source_id": "openfoodfacts",
    }
    return row if valid_episode(row) else None


def build(path: Path, out_dir: Path, per_category: int = 12,
          seed: int = SEED) -> dict:
    """Write `train.jsonl` and `dev-pool.jsonl`; return a manifest.

    The RNG is drawn in a fixed order over sorted pools, so the same dump
    and seed give byte-identical files.
    """
    rng = random.Random(seed)
    pools = defaultdict(list)
    seen = set()
    for product in read_products(path):
        if product["code"] in seen:
            continue
        seen.add(product["code"])
        split = "dev" if _unit("off/" + product["code"]) < DEV_SHARE else "train"
        key = (split, product["language"], product["category"])
        if len(pools[key]) < MAX_PER_CATEGORY:
            pools[key].append(product)
    rows = {"dev": [], "train": []}
    for (split, language, category), products in sorted(pools.items()):
        if len(products) < 6:
            continue
        budget = per_category if split == "dev" else per_category * 4
        for _ in range(budget * 2):
            if budget <= 0:
                break
            k = rng.choice([3, 4, 4, 5])
            episode = make_episode(rng.sample(products, k), language, rng,
                                   category)
            if episode:
                rows[split].append(episode)
                budget -= 1
    # Undecidable rows are dropped only here, after every RNG draw, so the
    # surviving rows are the same ones an unfiltered build would write.
    train = [row for row in rows["train"] if row["decidable"]]
    dev_pool = [row for row in rows["dev"]
                if row["decidable"] and not sealed(row["group_id"])]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "dev-pool.jsonl", dev_pool)
    manifest = {
        "source": {"path": Path(path).name, "license": "ODbL-1.0"},
        "seed": seed,
        "products": len(seen),
        "train_rows": len(train),
        "train_undecidable_dropped": len(rows["train"]) - len(train),
        "dev_pool_rows": len(dev_pool),
        "dev_sealed_or_undecidable_dropped": len(rows["dev"]) - len(dev_pool),
        "by_difficulty": dict(Counter(r["difficulty"] for r in train)),
        "by_language": dict(Counter(r["lang"] for r in train)),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    cli.add_argument("dump", type=Path,
                     help="en.openfoodfacts.org.products.csv.gz")
    cli.add_argument("--out", type=Path, default=Path("work/off"))
    cli.add_argument("--per-category", type=int, default=12)
    cli.add_argument("--seed", type=int, default=SEED)
    args = cli.parse_args(argv)
    print(json.dumps(build(args.dump, args.out, args.per_category, args.seed),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
