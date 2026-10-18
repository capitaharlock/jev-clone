"""Wikidata CC0 fetcher: ID-only SPARQL (fast) + labels via Action API.

The public SPARQL label service takes ~40 s for 10 rows, so we never use
`SERVICE wikibase:label`. Triples are cached under artifacts/data-raw/wikidata/
so the pilot is re-runnable offline and the gold is checkable (§104).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "jev-pilot/1.0 (research data pipeline)"}
CACHE = Path(__file__).resolve().parent.parent.parent / "artifacts" / "data-raw" / "wikidata"

TAXONS = {
    "city": {
        "type": "Q515", "es": "ciudad", "indef": "una",
        "props": [("P17", "país", "el"), ("P131", "entidad administrativa", "la")],
        "query": ("SELECT ?e ?v1 ?v2 WHERE { ?e wdt:P31 wd:Q515; wdt:P17 ?v1. "
                  "OPTIONAL { ?e wdt:P131 ?v2. } } LIMIT 4000"),
    },
    "human": {
        "type": "Q5", "es": "persona", "indef": "una",
        "props": [("P106", "ocupación", "la"), ("P27", "país de ciudadanía", "el")],
        "query": ("SELECT ?e ?v1 ?v2 WHERE { ?e wdt:P31 wd:Q5; wdt:P106 ?v1. "
                  "OPTIONAL { ?e wdt:P27 ?v2. } } LIMIT 4000"),
    },
    "film": {
        "type": "Q11424", "es": "película", "indef": "una",
        "props": [("P57", "dirección", "la"), ("P136", "género", "el")],
        "query": ("SELECT ?e ?v1 ?v2 WHERE { ?e wdt:P31 wd:Q11424; wdt:P57 ?v1. "
                  "OPTIONAL { ?e wdt:P136 ?v2. } } LIMIT 4000"),
    },
    "book": {
        "type": "Q571", "es": "libro", "indef": "un",
        "props": [("P50", "autoría", "la"), ("P364", "idioma original", "el")],
        "query": ("SELECT ?e ?v1 ?v2 WHERE { ?e wdt:P31 wd:Q571; wdt:P50 ?v1. "
                  "OPTIONAL { ?e wdt:P364 ?v2. } } LIMIT 4000"),
    },
    "mountain": {
        "type": "Q8502", "es": "montaña", "indef": "una",
        "props": [("P17", "país", "el")],
        "query": ("SELECT ?e ?v1 ?amt WHERE { ?e wdt:P31 wd:Q8502; wdt:P17 ?v1. "
                  "?e p:P2044 ?st. ?st psv:P2044 ?vn. ?vn wikibase:quantityAmount ?amt; "
                  "wikibase:quantityUnit wd:Q11573. } LIMIT 4000"),
    },
}


def _get(url: str, timeout: int = 120, retries: int = 6):
    import urllib.error

    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                wait = int(retry_after) + 5 if retry_after else 60 * (i + 1)
                time.sleep(wait)
            else:
                time.sleep(5 * (i + 1))
        except Exception as exc:  # noqa: BLE001 — flaky public endpoint
            last = exc
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"fetch failed after {retries} tries: {url[:120]} — {last}")


def sparql_ids(query: str) -> list[dict]:
    url = "https://query.wikidata.org/sparql?query=" + urllib.parse.quote(query) + "&format=json"
    data = _get(url, timeout=180)
    return data["results"]["bindings"]


def qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def fetch_labels(qids: list[str]) -> dict[str, str]:
    """QID -> label (es preferred, en fallback, else QID). Batched ×50."""
    out: dict[str, str] = {}
    uniq = sorted(set(qids))
    for i in range(0, len(uniq), 50):
        if i:
            time.sleep(1)  # polite pacing against the shared API
        chunk = uniq[i:i + 50]
        url = ("https://www.wikidata.org/w/api.php?action=wbgetentities&ids="
               + "|".join(chunk) + "&props=labels&languages=es|en&format=json")
        data = _get(url, timeout=60)
        for q, ent in data.get("entities", {}).items():
            labels = ent.get("labels", {})
            if "es" in labels:
                out[q] = labels["es"]["value"]
            elif "en" in labels:
                out[q] = labels["en"]["value"]
            else:
                out[q] = q
    return out


def fetch_taxon(name: str) -> dict:
    """Fetch one taxon; returns cache-shaped dict and writes it to disk."""
    spec = TAXONS[name]
    rows = sparql_ids(spec["query"])
    ents: dict[str, dict] = {}
    need: set[str] = set()
    for r in rows:
        e = qid(r["e"]["value"])
        v1 = qid(r["v1"]["value"])
        rec = ents.setdefault(e, {"qid": e, "props": {p: [] for p, *_ in spec["props"]},
                                  "amt": None})
        if v1 not in rec["props"][spec["props"][0][0]]:
            rec["props"][spec["props"][0][0]].append(v1)
        need.add(e)
        need.add(v1)
        if "v2" in r and len(spec["props"]) > 1:
            v2 = qid(r["v2"]["value"])
            if v2 not in rec["props"][spec["props"][1][0]]:
                rec["props"][spec["props"][1][0]].append(v2)
            need.add(v2)
        if "amt" in r:
            try:
                rec["amt"] = float(r["amt"]["value"])
            except ValueError:
                pass
    labels = fetch_labels(sorted(need))
    for e, rec in ents.items():
        rec["label"] = labels.get(e, e)
        for p, vals in rec["props"].items():
            rec["props"][p] = [{"qid": v, "label": labels.get(v, v)} for v in vals]
    payload = {"taxon": name, "entities": ents}
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False))
    return payload


def load_taxon(name: str) -> dict:
    return json.loads((CACHE / f"{name}.json").read_text())
