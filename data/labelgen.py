"""Label-space factory: Qwen-designed taxonomies, template-populated rows.

#T-labelspace-factory. The product needs MANY label spaces, each used WHOLE
as the denominator — not paraphrases of existing rows (`data.qwen_convert`
grows text, not spaces) and not pseudoword mini-taxonomies (`data.episodic`
already covers that corner). This module asks the local Qwen
(`qwen3.6:27b-mlx` via ollama, job `ollama-serve`) for ONE taxonomy per
call — a domain, K sibling labels with definitions, and the plausible
confusions between them — and then populates rows with a deterministic
template. One LLM call per SPACE, not per row: ~200 output tokens buy
~12 training rows.

Resumable by construction: the request cache (`cache/<sha>.json`, keyed on
the exact prompt) makes a re-run free, `spaces.jsonl` appends one line per
accepted space, and `run` skips hashes already accepted. Kill it, restart
it, it continues where it left off.

Every space enters through the same fences as any corpus:

* `data.registry.Registry` — one `DatasetCard` per space (CC0-1.0, `train`
  usage, sha256 of the space spec), persisted to `registry.json`.
* `data.firewall.reject_train_rows` — every generated text screened against
  the eval canaries BEFORE persist; a hit rejects the space, never warns.
* `data.hardneg._embed` — pairwise neighbourhood of the space published to
  `neighborhood.json`, so a gate can separate "discarded the obvious" from
  "separated siblings".

Layout under `artifacts/labelspace-qwen/`::

    cache/<sha>.json      raw Qwen reply + parsed space (or the error)
    spaces.jsonl          one accepted space per line (the resume pointer)
    shards/lq-000.jsonl   populated rows, 250 spaces per shard
    registry.json         per-space cards + registry manifest
    neighborhood.json     per-space neighbour stats + global histogram
    factory-mix.json      planned quotas + realised verification

Throughput note: the 27 B MLX model evaluates ~5 tok/s, so one space costs
~60 s and 2 000 spaces is ~35 h — the bulk runs as the daemon job
`labelspace-factory`, never inside a turn. `run --n` validates the pipe on
a handful of spaces first.

CLI::

    python3 -m data.labelgen run --n 2000 --seed 20260924
    python3 -m data.labelgen publish            # registry+firewall+neighbourhood+mix
    python3 -m data.labelgen mix --target 62000 # plan + verify the factory mix
"""
from __future__ import annotations

import argparse
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

from data import hardneg  # noqa: E402
from data import mix as mixmod  # noqa: E402
from data.firewall import reject_train_rows  # noqa: E402
from data.registry import DatasetCard, Registry  # noqa: E402

#: bump when the prompt, the template or the recipe change: spaces from an
#: older version are a different corpus wearing the same name.
LABELGEN_VERSION = "labelgen-v1"
SOURCE_ID = "labelspace-qwen"
OUT_DIR = os.path.join(ROOT, "artifacts", "labelspace-qwen")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
SPACES_PATH = os.path.join(OUT_DIR, "spaces.jsonl")
SHARD_DIR = os.path.join(OUT_DIR, "shards")
REGISTRY_PATH = os.path.join(OUT_DIR, "registry.json")
NEIGHBORHOOD_PATH = os.path.join(OUT_DIR, "neighborhood.json")
MIX_PATH = os.path.join(OUT_DIR, "factory-mix.json")

MODEL = os.environ.get("QWEN_MODEL", "qwen3.6:27b-mlx")
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

#: spaces the task asks for; rows are template-cheap, LLM calls are not.
N_TARGET = 2000
ROWS_PER_SPACE = 12
SPACES_PER_SHARD = 250
#: sibling counts per space, seeded per space (the denominator varies).
K_CHOICES = (4, 5, 6)
#: two spaces merge when their closest labels read near-identical.
DEDUP_JACCARD = 0.75

SYSTEM = (
    "You design miniature taxonomies for a text classifier. Reply with ONLY "
    "one JSON object, no prose, no code fences, with keys: 'domain' (short "
    "domain name, snake_case), 'labels' (array of exactly {k} objects with "
    "'name' (2-4 words) and 'def' (one line, under 20 words, never repeating "
    "the name verbatim)), 'confusions' (array of exactly 3 triples "
    "[name_a, name_b, reason] naming label pairs that are easy to confuse "
    "and why, in under 15 words)."
)

# -- the domain catalog: FIELDS x CONTEXTS hints, 48 x 44 = 2112 ---------
FIELDS = (
    "harbor berth assignment", "library reshelving tickets",
    "bakery oven scheduling", "vineyard pest reports",
    "taxi dispatch priority", "museum exhibit rotation",
    "ski lift maintenance", "beehive health inspection",
    "data center cooling alerts", "restaurant table turnover",
    "ferry route delays", "greenhouse irrigation zones",
    "parking garage incidents", "airport baggage routing",
    "cinema projector faults", "swimming pool chemical checks",
    "elevator service calls", "farmers market stall placement",
    "campground reservations", "bike-share rebalancing",
    "observatory observation slots", "aquarium tank maintenance",
    "theater seat upgrades", "warehouse forklift tasks",
    "post office sorting errors", "dental appointment triage",
    "car wash queue management", "laundromat machine faults",
    "hostel bunk assignment", "food truck permits",
    "marina fuel dock scheduling", "trailhead parking passes",
    "ski rental returns", "botanical garden pruning",
    "lighthouse lamp checks", "wind turbine inspections",
    "solar farm panel faults", "bus depot cleaning",
    "tram signal faults", "subway lost property",
    "zoo feeding schedules", "planetarium show bookings",
    "archive box retrieval", "print shop job queue",
    "locksmith callouts", "chimney sweep appointments",
    "gutter cleaning routes", "septic tank servicing",
)
CONTEXTS = (
    "in a tidal river port", "during a night shift",
    "on a holiday weekend", "under a heat wave",
    "after a software migration", "in a mountain village",
    "on a small island", "during a festival week",
    "with a reduced crew", "after a storm warning",
    "in a historic district", "during peak tourist season",
    "under a budget freeze", "after a merger",
    "in a rural county", "during exam season",
    "on a construction site", "in a coastal town",
    "during power rationing", "after a recall notice",
    "in a border town", "during a transit strike",
    "under a new regulation", "after a flood",
    "in a desert outpost", "during a cold snap",
    "on a university campus", "in a hospital wing",
    "after an audit", "during a product launch",
    "in a heritage building", "under quarantine rules",
    "after a cyberattack", "in a high-rise complex",
    "during a marathon event", "on a film set",
    "in an intake center", "after a landslide",
    "during a drought", "in an arctic station",
    "on a cruise ship", "during a blackout",
    "in a night market", "after a hailstorm",
)


def domain_hints(seed: int, n: int = N_TARGET) -> list[tuple[str, int]]:
    """Seeded (field, context, K) triples — the only per-space randomness
    that is NOT the model's: same seed, same request sequence."""
    rng = random.Random(f"{LABELGEN_VERSION}\x00hints\x00{seed}")
    combos = [(f, c) for f in FIELDS for c in CONTEXTS]
    rng.shuffle(combos)
    out = []
    for i in range(n):
        f, c = combos[i % len(combos)]
        k = K_CHOICES[random.Random(
            f"{LABELGEN_VERSION}\x00k\x00{seed}\x00{i}").randrange(
            len(K_CHOICES))]
        out.append((f"{f} {c}", k))
    return out


def cache_key(hint: str, k: int) -> str:
    system = SYSTEM.format(k=k)
    return hashlib.sha256(
        f"{LABELGEN_VERSION}\x00{MODEL}\x00{system}\x00{hint}".encode()
    ).hexdigest()


def qwen_space(hint: str, k: int, timeout: int = 300) -> dict:
    """One taxonomy from the model, or {"error": ...}. Cached by prompt."""
    key = cache_key(hint, k)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    body = json.dumps({
        "model": MODEL, "stream": False, "think": False,
        "messages": [
            {"role": "system", "content": SYSTEM.format(k=k)},
            {"role": "user",
             "content": f"Design a taxonomy of: {hint}."},
        ],
        "options": {"temperature": 0.8, "num_predict": 700},
    }).encode()
    try:
        req = urllib.request.Request(
            f"{OLLAMA}/api/chat", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())["message"]["content"]
        if "</think>" in out:
            out = out.split("</think>", 1)[1]
        if out.strip().startswith("```"):
            out = out.strip().split("\n", 1)[1].rsplit("```", 1)[0]
        start, end = out.find("{"), out.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"no JSON in reply: {out[:120]!r}")
        result: dict = {"space": json.loads(out[start:end + 1]),
                        "hint": hint, "k": k, "model": MODEL}
    except Exception as e:  # noqa: BLE001 — the cache must record failures too
        result = {"error": f"{type(e).__name__}: {e}",
                  "hint": hint, "k": k, "model": MODEL}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(result, fh, ensure_ascii=False)
        fh.write("\n")
    return result


# -- validation -----------------------------------------------------------
def validate_space(space: dict) -> list[str]:
    """Errors in a parsed taxonomy; empty means accept."""
    errors: list[str] = []
    if not isinstance(space, dict):
        return ["space is not an object"]
    domain = space.get("domain", "")
    if not isinstance(domain, str) or not domain.strip() or len(domain) > 60:
        errors.append("domain must be a non-empty string <= 60 chars")
    labels = space.get("labels", [])
    if not isinstance(labels, list) or not (
            min(K_CHOICES) <= len(labels) <= max(K_CHOICES)):
        errors.append(
            f"labels must be a list of {min(K_CHOICES)}..{max(K_CHOICES)}")
        return errors
    seen: set[str] = set()
    for i, lab in enumerate(labels):
        if not isinstance(lab, dict):
            errors.append(f"label {i} is not an object")
            continue
        name, defi = lab.get("name", ""), lab.get("def", "")
        if not isinstance(name, str) or not name.strip() or len(name) > 40:
            errors.append(f"label {i}: bad name")
        elif name.casefold() in seen:
            errors.append(f"label {i}: duplicate name {name!r}")
        else:
            seen.add(name.casefold())
        if (not isinstance(defi, str) or not defi.strip()
                or len(defi.split()) > 40):
            errors.append(f"label {i}: def must be non-empty <= 40 words")
    names = {str(lab.get('name', '')).casefold() for lab in labels
             if isinstance(lab, dict)}
    confs = space.get("confusions", [])
    if not isinstance(confs, list) or len(confs) < 2:
        errors.append("confusions must be a list of >= 2 triples")
    else:
        for i, triple in enumerate(confs):
            if (not isinstance(triple, (list, tuple)) or len(triple) != 3
                    or not all(isinstance(x, str) and x.strip()
                               for x in triple)):
                errors.append(f"confusion {i}: must be [name, name, reason]")
                continue
            a, b = triple[0].casefold(), triple[1].casefold()
            if a == b or a not in names or b not in names:
                errors.append(
                    f"confusion {i}: names must be two DISTINCT labels")
    return errors


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def label_jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_duplicate(space: dict, accepted: list[dict]) -> tuple[bool, str]:
    """A space merges when any of its labels reads near-identical to an
    accepted one — otherwise two "different" spaces are one pool."""
    names = [str(lab.get("name", "")) for lab in space.get("labels", [])]
    for prev in accepted:
        for had in [str(lab.get("name", ""))
                    for lab in prev.get("labels", [])]:
            for name in names:
                if label_jaccard(name, had) >= DEDUP_JACCARD:
                    return True, f"{name!r} ~ {had!r}"
    return False, ""


# -- template population --------------------------------------------------
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-") or "x"


def rows_of(space: dict, space_id: str, seed: int) -> list[dict]:
    """Deterministic template rows: the WHOLE space is every row's options.

    The state describes the gold label and quotes one confusable sibling as
    a cross-check line, so the row rewards separating siblings, not
    spotting a lone keyword. Same (seed, space) = same rows.
    """
    labels = space["labels"]
    k = len(labels)
    confs = [tuple(c) for c in space.get("confusions", [])]
    rows = []
    for j in range(ROWS_PER_SPACE):
        rng = random.Random(f"{LABELGEN_VERSION}\x00row\x00{seed}"
                            f"\x00{space_id}\x00{j}")
        gold_idx = j % k
        gold = labels[gold_idx]
        sibs = [lab for i, lab in enumerate(labels) if i != gold_idx]
        sib = rng.choice(sibs)
        reason = ""
        for a, b, why in confs:
            if {a.casefold(), b.casefold()} == {str(gold["name"]).casefold(),
                                                str(sib["name"]).casefold()}:
                reason = f" ({why})"
                break
        lines = [f"Field report {space_id}-{j}: {gold['def']}",
                 f"Cross-check against {sib['name']}: "
                 f"{sib['def']}{reason}"]
        if rng.random() < 0.5:
            other = rng.choice(sibs)
            lines.append(f"Logged under {space['domain']}: "
                         f"see also {other['name']}.")
        rng.shuffle(lines)
        state = "\n".join(lines)
        options = [{"id": f"{space_id}--{_slug(str(lab['name']))}",
                    "text": str(lab["name"])} for lab in labels]
        rng.shuffle(options)
        answer = f"{space_id}--{_slug(str(gold['name']))}"
        rows.append({
            "state": state,
            "questions": [{
                "id": f"{space_id}-{j:02d}",
                "kind": "choice",
                "question": f"Which {space['domain']} category applies?",
                "options": options,
                "answer": answer,
            }],
            "split": "train",
        })
    return rows


def space_texts(space: dict, rows: list[dict]) -> list[str]:
    texts = [str(space.get("domain", ""))]
    for lab in space.get("labels", []):
        texts += [str(lab.get("name", "")), str(lab.get("def", ""))]
    for triple in space.get("confusions", []):
        texts += [str(x) for x in triple]
    for row in rows:
        texts.append(row["state"])
        for q in row["questions"]:
            texts.append(str(q.get("question", "")))
            texts += [o["text"] for o in q["options"]]
    return [t for t in texts if t.strip()]


# -- registry + neighbourhood ----------------------------------------------
def space_sha(space: dict) -> str:
    return hashlib.sha256(json.dumps(
        space, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def card_of(space_id: str, space: dict) -> DatasetCard:
    sha = space_sha(space)
    return DatasetCard(
        id=f"{SOURCE_ID}/{space_id}",
        source_original="local Qwen qwen3.6:27b-mlx taxonomy (ollama job)",
        mirror="none — generated, no upstream",
        license="CC0-1.0",
        usage="train",
        revision=f"{LABELGEN_VERSION}@{sha[:12]}",
        sha256=sha,
        transform=(f"{LABELGEN_VERSION}: qwen space spec + deterministic "
                   f"template rows x{ROWS_PER_SPACE}"),
    )


def neighbourhood(space: dict) -> dict:
    """Pairwise sibling closeness via the repo's char-3gram embedding.

    `min_nn_cos` is the number a gate thresholds on: near 1.0 the space is
    won by separating siblings, near 0.0 by discarding the obvious.
    """
    vecs = {str(lab["name"]): hardneg._embed(
        f"{lab['name']}. {lab['def']}") for lab in space["labels"]}
    names = sorted(vecs)
    nn: dict[str, dict] = {}
    hardest: tuple[str, str, float] | None = None
    for a in names:
        best, best_s = "", -1.0
        for b in names:
            if b == a:
                continue
            s = round(sum(x * y for x, y in zip(vecs[a], vecs[b])), 6)
            if s > best_s:
                best, best_s = b, s
        nn[a] = {"nearest": best, "cos": best_s}
        if hardest is None or best_s > hardest[2]:
            hardest = (a, best, best_s)
    cosines = [v["cos"] for v in nn.values()]
    return {
        "k": len(names),
        "per_label": nn,
        "hardest_pair": list(hardest) if hardest else [],
        "space_min_nn": round(min(cosines), 6) if cosines else 0.0,
        "space_mean_nn": round(sum(cosines) / len(cosines), 6)
        if cosines else 0.0,
    }


# -- run: the resumable factory loop ----------------------------------------
def load_accepted() -> list[dict]:
    accepted = []
    if os.path.exists(SPACES_PATH):
        with open(SPACES_PATH) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    accepted.append(json.loads(line))
    return accepted


def run(n: int = N_TARGET, seed: int = 20260924, log=None) -> dict:
    """Generate until `n` spaces are accepted. Every prompt is cached, every
    space screened by the firewall before it lands in `spaces.jsonl`."""
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    accepted = load_accepted()
    have_hashes = {a["prompt_sha"] for a in accepted}
    stats = {"requested": 0, "cached": 0, "accepted": len(accepted),
             "rejected": 0, "firewall_rejected": 0, "errors": 0}
    hints = domain_hints(seed, max(n * 2, n + 8))
    hi = 0
    with open(SPACES_PATH, "a") as out:
        while len(accepted) < n and hi < len(hints):
            hint, k = hints[hi]
            hi += 1
            key = cache_key(hint, k)
            if key in have_hashes:
                stats["cached"] += 1
                continue
            stats["requested"] += 1
            reply = qwen_space(hint, k)
            if "error" in reply or "space" not in reply:
                stats["errors"] += 1
                continue
            space = reply["space"]
            if validate_space(space):
                stats["rejected"] += 1
                continue
            dup, _ = is_duplicate(space, accepted)
            if dup:
                stats["rejected"] += 1
                continue
            idx = len(accepted)
            space_id = f"lq-{idx:04d}"
            rows = rows_of(space, space_id, seed)
            try:
                reject_train_rows(space_texts(space, rows))
            except ValueError:
                stats["firewall_rejected"] += 1
                continue
            accepted.append({"space_id": space_id, "prompt_sha": key,
                             "hint": hint, "k": k, "seed": seed,
                             "version": LABELGEN_VERSION, **space,
                             "_rows_cached": len(rows)})
            have_hashes.add(key)
            out.write(json.dumps(accepted[-1], ensure_ascii=False) + "\n")
            out.flush()
            stats["accepted"] = len(accepted)
            if log and len(accepted) % 25 == 0:
                log(f"[labelgen] {len(accepted)}/{n} spaces")
    stats["elapsed_s"] = round(time.time() - t0, 1)
    return stats


def write_shards(accepted: list[dict] | None = None,
                 seed: int = 20260924) -> dict:
    """Populate template rows for every accepted space into shard files."""
    accepted = accepted if accepted is not None else load_accepted()
    os.makedirs(SHARD_DIR, exist_ok=True)
    # idempotent: shard content is a pure function of spaces.jsonl, so any
    # stale shard is removed and rebuilt instead of appended to.
    for fn in sorted(os.listdir(SHARD_DIR)):
        if fn.startswith(f"{SOURCE_ID}-") and fn.endswith(".jsonl"):
            os.remove(os.path.join(SHARD_DIR, fn))
    paths, n_rows = [], 0
    for s, entry in enumerate(accepted):
        space = {k: entry[k] for k in
                 ("domain", "labels", "confusions") if k in entry}
        rows = rows_of(space, entry["space_id"], entry.get("seed", seed))
        shard = s // SPACES_PER_SHARD
        path = os.path.join(SHARD_DIR, f"{SOURCE_ID}-{shard:03d}.jsonl")
        with open(path, "a") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
        if path not in paths:
            paths.append(path)
    shards = []
    for p in sorted(paths):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            while True:
                block = fh.read(1 << 20)
                if not block:
                    break
                h.update(block)
        shards.append({"path": os.path.relpath(p, ROOT),
                       "bytes": os.path.getsize(p), "sha256": h.hexdigest()})
    return {"shards": shards, "n_rows": n_rows,
            "n_spaces": len(accepted)}


def publish(seed: int = 20260924) -> dict:
    """Registry cards + firewall verdict + neighbourhood, from spaces.jsonl.

    Rebuilds every derived artifact from scratch: idempotent, re-runnable
    as the factory job appends more spaces.
    """
    t0 = time.time()
    accepted = load_accepted()
    reg = Registry()
    cards = []
    for entry in accepted:
        space = {k: entry[k] for k in
                 ("domain", "labels", "confusions") if k in entry}
        card = card_of(entry["space_id"], space)
        reg.register(card)  # raises on a bad card: publish fails loudly
        cards.append(card.__dict__)
    ok, _ = reg.train_ok(cards[0]["id"]) if cards else (True, "empty")
    if cards and not ok:
        raise ValueError("registry fence refuses the factory cards")
    firewall_texts: list[str] = []
    for entry in accepted:
        space = {k: entry[k] for k in
                 ("domain", "labels", "confusions") if k in entry}
        firewall_texts += space_texts(
            space, rows_of(space, entry["space_id"], entry.get("seed", seed)))
    firewall_clean = reject_train_rows(firewall_texts) if firewall_texts \
        else True
    neigh = []
    for entry in accepted:
        space = {k: entry[k] for k in
                 ("domain", "labels", "confusions") if k in entry}
        stat = neighbourhood(space)
        stat["space_id"] = entry["space_id"]
        neigh.append(stat)
    hist: dict[str, int] = {}
    for stat in neigh:
        b = f"{int(stat['space_min_nn'] * 10) / 10:.1f}"
        hist[b] = hist.get(b, 0) + 1
    shard_info = write_shards(accepted, seed)
    manifest = reg.manifest([c["id"] for c in cards])
    with open(REGISTRY_PATH, "w") as fh:
        json.dump({"version": LABELGEN_VERSION, "task": "T-labelspace-factory",
                   "n_spaces": len(cards), "cards": cards,
                   "manifest": manifest,
                   "firewall": {"screened_texts": len(firewall_texts),
                                "clean": firewall_clean}},
                  fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    with open(NEIGHBORHOOD_PATH, "w") as fh:
        json.dump({"version": LABELGEN_VERSION, "n_spaces": len(neigh),
                   "min_nn_hist": dict(sorted(hist.items())),
                   "mean_min_nn": round(sum(s["space_min_nn"]
                                            for s in neigh) / len(neigh), 6)
                   if neigh else 0.0,
                   "mean_mean_nn": round(sum(s["space_mean_nn"]
                                             for s in neigh) / len(neigh), 6)
                   if neigh else 0.0,
                   "spaces": neigh},
                  fh, indent=2, sort_keys=True)
        fh.write("\n")
    return {"n_spaces": len(accepted), "registry": REGISTRY_PATH,
            "neighbourhood": NEIGHBORHOOD_PATH,
            "shards": shard_info, "firewall_clean": firewall_clean,
            "elapsed_s": round(time.time() - t0, 1)}


# -- the factory mix ----------------------------------------------------------
#: explicit selection: the new spaces + episodic-div + the human corpus.
#: `labelspace-qwen` and `episodic-div` are experimental, so they never leak
#: into a default scan — this list names them out loud instead.
FACTORY_DATASETS = (
    "labelspace-qwen", "episodic-div",
    "banking77", "massive", "huffpost", "dbpedia14", "goemotions", "swag",
)
FACTORY_TARGET = 62000
FACTORY_SEED = 20260924


def factory_mix(target: int = FACTORY_TARGET,
                seed: int = FACTORY_SEED) -> dict:
    """Plan + realise + verify a mixture of the new spaces with episodic-div
    and the human corpus. Raises on any quota breach — a mix that breaks
    `data/mix.py` is not a deliverable."""
    t0 = time.time()
    scan = mixmod.scan_supply(list(FACTORY_DATASETS))
    missing = [d for d in FACTORY_DATASETS
               if scan.get(d, {}).get("missing")]
    if missing:
        raise ValueError(f"factory mix: missing supply for {missing}")
    supply = mixmod.effective_supply(scan)
    target_requested = target
    try:
        spec = mixmod.plan_mix(supply, target=target, seed=seed)
    except mixmod.MixInfeasible:
        # the factory is still grinding: the new spaces cannot fill their
        # dataset ceiling yet, and nobody else may use that headroom. Plan
        # the largest feasible mix instead and say so — re-running `mix`
        # as spaces accumulate converges to the requested target.
        feasible = mixmod.max_feasible_target(
            supply,
            mixmod.MAX_DATASET_FRACTION - mixmod.CAP_SAFETY_MARGIN,
            mixmod.MAX_FAMILY_FRACTION - mixmod.CAP_SAFETY_MARGIN)
        if feasible <= 0:
            raise
        target = feasible
        spec = mixmod.plan_mix(supply, target=target, seed=seed)
    counts = mixmod.realise(spec)
    guard = mixmod.verify_mix(counts.get("dataset", {}), counts["rows"])
    comp = mixmod.check_composition(counts)
    origin = counts.get("origin", {})
    total = counts["rows"]
    human_share = origin.get("human", 0) / total if total else 0.0
    if human_share < mixmod.MIN_HUMAN_FRACTION - 1e-9:
        raise mixmod.MixGuardrailError(
            f"factory mix: human share {human_share:.4f} below "
            f"MIN_HUMAN_FRACTION {mixmod.MIN_HUMAN_FRACTION}")
    if not comp["pass"]:
        failed = [k for k, v in comp["checks"].items() if not v["pass"]]
        raise mixmod.MixGuardrailError(
            f"factory mix: composition checks fail: {failed}")
    manifest = mixmod.build_manifest(spec, counts, scan, time.time() - t0)
    manifest["task"] = "T-labelspace-factory"
    manifest["corpus"] = "decision-mix-labelfactory-62k"
    try:
        with open(REGISTRY_PATH) as fh:
            n_factory_spaces = json.load(fh)["n_spaces"]
    except (OSError, ValueError, KeyError):
        n_factory_spaces = 0
    out = {
        "task": "T-labelspace-factory",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": target, "seed": seed,
        "target_requested": target_requested,
        "target_capped_by_supply": target != target_requested,
        "diversity_vs_baseline": {
            "clean_1m_shared_spaces": 9,
            "clean_1m_shared_share": 0.830976,
            "clean_1m_verdict": "pobre",
            "baseline_artifact":
                "artifacts/gates/T-labelspace-div/inventory.json",
            "factory_spaces": n_factory_spaces,
            "factory_quota_rows": spec.quotas.get(SOURCE_ID, 0),
            "why": ("each factory space is used WHOLE as the denominator, "
                    "so factory spaces add 1:1 to the sampler-visible "
                    "space count once space_scoped; the baseline's 9 "
                    "shared taxonomies cover 83.1 % of its rows"),
        },
        "quotas": spec.quotas,
        "realised_rows": total,
        "by_origin": dict(sorted(origin.items())),
        "human_share": round(human_share, 6),
        "synthetic_share": round(origin.get("synthetic", 0) / total, 6)
        if total else 0.0,
        "verify_mix_breaches": guard["breaches"],
        "composition_pass": comp["pass"],
        "manifest": manifest,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(MIX_PATH, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m data.labelgen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="generate spaces until N accepted")
    r.add_argument("--n", type=int, default=N_TARGET)
    r.add_argument("--seed", type=int, default=20260924)
    p = sub.add_parser("publish", help="registry + firewall + neighbourhood")
    p.add_argument("--seed", type=int, default=20260924)
    m = sub.add_parser("mix", help="plan + realise + verify the factory mix")
    m.add_argument("--target", type=int, default=FACTORY_TARGET)
    m.add_argument("--seed", type=int, default=FACTORY_SEED)
    args = ap.parse_args(argv)

    def _log(msg: str) -> None:
        print(msg, flush=True)

    if args.cmd == "run":
        stats = run(n=args.n, seed=args.seed, log=_log)
        print(f"accepted={stats['accepted']} requested={stats['requested']} "
              f"rejected={stats['rejected']} "
              f"firewall_rejected={stats['firewall_rejected']} "
              f"errors={stats['errors']} "
              f"elapsed_s={stats['elapsed_s']}")
        return 0 if stats["accepted"] >= args.n else 1
    if args.cmd == "publish":
        out = publish(seed=args.seed)
        print(f"spaces={out['n_spaces']} rows={out['shards']['n_rows']} "
              f"firewall_clean={out['firewall_clean']} "
              f"elapsed_s={out['elapsed_s']}")
        return 0
    out = factory_mix(target=args.target, seed=args.seed)
    print(f"rows={out['realised_rows']} human={out['human_share']} "
          f"synthetic={out['synthetic_share']} "
          f"composition_pass={out['composition_pass']} -> "
          f"{os.path.relpath(MIX_PATH, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
