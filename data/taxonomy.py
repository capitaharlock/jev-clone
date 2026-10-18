"""Label-text material: what a label is CALLED and what it MEANS.

`data/adapters.py:146` builds every option as `Option(id=l, text=l)`, so the
only thing the head ever reads about a label is its identifier —
`direct_debit_payment_not_recognised`. That is a decision the repo made by
omission, not on purpose, and #T-option-text is the measurement that prices
it. Pricing it needs the other half: what those 77 categories actually mean,
in words, from a source somebody else can check.

So a taxonomy here is a directory under `data/taxonomies/<id>/` holding

* `descriptions.json` — `{label: definition}`, stored byte for byte as the
  upstream published it, and
* `card.json` — where it came from, at which immutable revision, under which
  licence, with the caveats that licence does not cover.

`load()` verifies the sha256 in the card against the bytes on disk before it
returns anything. A definitions file that drifted from its card is not a
source any more, it is a local edit with a citation stapled to it, and the
whole point of the arm is that somebody else can re-fetch the same bytes.

Stdlib only: this is read by tests that must run without the torch stack.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_DIR = ROOT / "data" / "taxonomies"

#: the taxonomy #T-option-text measures: the teacher's own 77 definitions
BANKING77 = "banking77-jev"


class TaxonomyError(RuntimeError):
    """The material on disk does not match the card that describes it."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def card(name: str = BANKING77) -> dict:
    """The provenance card: source, revision, sha256, licence, caveats."""
    path = TAXONOMY_DIR / name / "card.json"
    if not path.exists():
        raise TaxonomyError(f"no taxonomy {name!r} at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load(name: str = BANKING77, verify: bool = True) -> dict:
    """`{label: definition}` for one taxonomy, checked against its card.

    `verify=False` exists for the test that has to build a taxonomy in a
    tmpdir; nothing that publishes a number may use it.
    """
    meta = card(name)
    path = TAXONOMY_DIR / name / "descriptions.json"
    if verify:
        got = sha256_file(path)
        if got != meta.get("sha256"):
            raise TaxonomyError(
                f"{path} hashes to {got} but its card declares "
                f"{meta.get('sha256')}: the file was edited locally, so it "
                "is no longer the source the card cites. Re-fetch it from "
                f"{meta.get('mirror')} or fix the card on purpose")
    out = json.loads(path.read_text(encoding="utf-8"))
    if len(out) != meta.get("labels"):
        raise TaxonomyError(
            f"{path} carries {len(out)} labels, the card says "
            f"{meta.get('labels')}")
    return out


def readable(label: str) -> str:
    """`card_arrival` -> `Card arrival`. One rule, no acronym table.

    A hand-kept map of special cases (`atm` -> `ATM`, `pin` -> `PIN`) would
    read better and would also be an undeclared choice inside a measurement
    whose whole subject is what the model is shown. The transform is one
    line so the artifact can state it exactly and anybody can reproduce the
    string from the identifier.
    """
    words = label.replace("_", " ").replace("-", " ").strip()
    return words[:1].upper() + words[1:] if words else label


def option_text(label: str, descriptions: dict) -> str:
    """The arm-B option string: readable name, em dash, definition.

    Falls back to the readable name alone for a label the taxonomy does not
    cover — loudly absent rather than silently `label: label`, which would
    make arm B partly arm A without saying so.
    """
    desc = (descriptions or {}).get(label)
    name = readable(label)
    return f"{name} — {desc}" if desc else name


def coverage(labels, name: str = BANKING77) -> dict:
    """How much of a label space this taxonomy actually defines."""
    desc = load(name)
    labels = list(labels)
    missing = sorted(set(labels) - set(desc))
    return {
        "taxonomy": name,
        "labels_in_space": len(set(labels)),
        "labels_defined": len(set(labels)) - len(missing),
        "missing": missing,
        "complete": not missing,
    }


def main(argv: list) -> int:
    name = argv[0] if argv else BANKING77
    meta = card(name)
    desc = load(name)
    print(json.dumps({
        "taxonomy": name,
        "labels": len(desc),
        "source": meta.get("source_original"),
        "revision": meta.get("revision"),
        "sha256": meta.get("sha256"),
        "license": meta.get("license"),
        "license_caveat": meta.get("license_caveat"),
        "sample": {k: desc[k] for k in sorted(desc)[:3]},
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
