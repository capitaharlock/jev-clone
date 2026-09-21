"""Triple store + gold checker: gold is verified by graph query, never teacher vote."""
from __future__ import annotations

import re

from .fetch import TAXONS, load_taxon

#: Wikidata returns the bare QID when an item has no es/en label. Such a
#: value is not a candidate answer — "¿Cuál es el país de X?" cannot be
#: answered with "Q100269625" — so it never enters a pool or a gold slot.
_BARE_QID = re.compile(r"Q\d+")


class TripleStore:
    """(entity, prop) -> set of value QIDs, built from the cached fetch."""

    def __init__(self):
        self.triples: dict[tuple[str, str], set[str]] = {}
        self.labels: dict[str, str] = {}
        self.amounts: dict[str, float] = {}
        self.taxon_of: dict[str, str] = {}

    @classmethod
    def load(cls) -> "TripleStore":
        store = cls()
        for name, spec in TAXONS.items():
            try:
                payload = load_taxon(name)
            except FileNotFoundError:
                continue
            for e, rec in payload["entities"].items():
                store.taxon_of[e] = name
                store.labels[e] = rec["label"]
                if rec.get("amt") is not None:
                    store.amounts[e] = rec["amt"]
                for p, vals in rec["props"].items():
                    key = (e, p)
                    s = store.triples.setdefault(key, set())
                    for v in vals:
                        s.add(v["qid"])
                        store.labels[v["qid"]] = v["label"]
        return store

    def has_label(self, qid: str) -> bool:
        """False when Wikidata gave us no human-readable label for this item."""
        return not _BARE_QID.fullmatch(self.labels.get(qid, qid))

    def labelled_values(self, entity: str, prop: str) -> list[str]:
        """Sorted values of (entity, prop) that actually have a label."""
        return sorted(v for v in self.values(entity, prop) if self.has_label(v))

    def values(self, entity: str, prop: str) -> set[str]:
        return set(self.triples.get((entity, prop), set()))

    def check(self, entity: str, prop: str, value: str) -> bool:
        """True iff the graph contains this triple (hand-tampered gold fails)."""
        return value in self.triples.get((entity, prop), set())


def verify_record(rec: dict, store: TripleStore) -> list[str]:
    """Checker for built records: every graph-sourced answer must resolve."""
    errors: list[str] = []
    g = rec.get("gold", {})
    if g.get("kind") == "graph":
        if g.get("negated"):
            # counterfactual/false boolean: the denied value must NOT hold
            if store.check(g["entity"], g["prop"], g["value"]):
                errors.append(f"negated gold holds in graph: {g['entity']} {g['prop']} {g['value']}")
        elif not store.check(g["entity"], g["prop"], g["value"]):
            errors.append(f"gold not in graph: {g['entity']} {g['prop']} {g['value']}")
    elif g.get("kind") == "graph-amount":
        amt = store.amounts.get(g["entity"])
        if amt is None:
            errors.append(f"no amount in graph for {g['entity']}")
        elif (amt > g["threshold"]) != g["above"]:
            errors.append(f"amount check failed for {g['entity']}: {amt} vs {g['threshold']}")
    else:
        errors.append(f"unknown gold kind: {g.get('kind')}")
    return errors


def verify_answer_binding(q: dict, store: TripleStore) -> list[str]:
    """The answer KEY must follow from the graph, not just the gold pointer.

    `verify_record` checks that the triple a question points at exists. That
    alone does not catch the cheapest tampering there is: leaving the gold
    pointer intact and editing `answer` to another option. This binds the two
    — the option the answer names must be the one carrying the gold's label
    (choice), or the polarity the graph implies (boolean).
    """
    errors: list[str] = []
    gold = q.get("gold", {})
    answer = q.get("answer")
    options = q.get("options") or []
    ids = [o.get("id") for o in options]
    if answer not in ids:
        return [f"answer {answer!r} is not an option id"]

    if q.get("kind") == "choice":
        if gold.get("kind") != "graph":
            return [f"choice question with non-graph gold: {gold.get('kind')}"]
        want = store.labels.get(gold.get("value"))
        got = next(o.get("text") for o in options if o.get("id") == answer)
        if want != got:
            errors.append(
                f"answer points at {got!r} but the graph gold is {want!r}")
        if sum(1 for o in options if o.get("text") == want) != 1:
            errors.append(f"gold label {want!r} is not unique among the options")
    elif q.get("kind") == "boolean":
        if gold.get("kind") == "graph":
            expected = "no" if gold.get("negated") else "yes"
        elif gold.get("kind") == "graph-amount":
            expected = "yes" if gold.get("above") else "no"
        else:
            return [f"boolean question with unknown gold kind: {gold.get('kind')}"]
        if answer != expected:
            errors.append(f"boolean answer {answer!r} contradicts the graph "
                          f"(expected {expected!r})")
    else:
        errors.append(f"unknown question kind: {q.get('kind')}")
    return errors


def verify_question(q: dict, store: TripleStore) -> list[str]:
    """Full check of one built question: graph triple AND answer binding."""
    return verify_record(q, store) + verify_answer_binding(q, store)
