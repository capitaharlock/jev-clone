"""Continued LoRA on TinyJev 4B v1, the pointer recipe that cleared the gate.

The published runs that beat the 280M listwise encoder (Kev, TinyJev E11)
train a Qwen backbone with a rank-16 LoRA and a pointer head. The loss is
cross-entropy over the options of one question, in one forward pass. Option
order is shuffled on every view, and a quarter of the views add a
"none of these" option, half of those with the gold removed.

This run starts from the merged TinyJev-4B v1 weights (Qwen3-4B-Base plus
that head). It does not read battery_dev or the sealed cut. A 10% holdout
of rule-episode groups is the live monitor only. It is not a promotion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import signal
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import episode_contract as EC  # noqa: E402

TASK = "T-open-decision-backbone"
RULE_ROOT = ROOT / "artifacts" / "episodes-rule"
TINYJEV_SRC = ROOT / "tmp" / "jev-research" / "tinyjev"
HUB = (Path.home() / ".cache" / "huggingface" / "hub"
       / "models--AnkitAI--TinyJev-4B")
DEV_GROUP = re.compile(r"^dev-")
SEALED_GROUP = re.compile(r"^sealed-")
NONE_ID = "n0"
NONE_TEXT = {
    "en": "None of the listed options",
    "es": "Ninguna de las opciones listadas",
}
HEAD_DIM = 256
OPTION_PAD = 9
LORA_RANK = 16
LORA_ALPHA = 32
LORA_KEYS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
FAMILY_WEIGHT = {
    EC.COMPARISON: 2,
    EC.PRIORITY: 2,
}
HOLDOUT_CUTS = (("train", 0.9), ("holdout", 0.1))
SPLIT_SEED = 20261003
LR = 2e-5
WARMUP_STEPS = 100
EXPECTED_LORA_LAYERS = 252
DEFAULT_SAVE_EVERY = 10


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def positives_of(ep: dict) -> list[str]:
    """Candidate ids that count as a hit, for the three answer shapes."""
    if ep.get("answer") is not None:
        return [str(ep["answer"])]
    if ep.get("acceptable_answers"):
        return [str(a) for a in ep["acceptable_answers"]]
    return [str(ep["preference"]["order"][0])]


def guard_training_cut(episodes: list[dict]) -> None:
    """Dev and sealed variant groups never enter this trainer."""
    bad = [
        ep.get("id") for ep in episodes
        if DEV_GROUP.match(str(ep.get("variant_group", "")))
        or SEALED_GROUP.match(str(ep.get("variant_group", "")))
    ]
    if bad:
        raise RuntimeError(
            f"{len(bad)} episodes carry a dev or sealed group "
            f"({bad[:4]})")


def decision_view(ep: dict, rng: random.Random, *, p_none: float,
                  shuffle: bool) -> tuple[list[dict], list[str]]:
    """One training view: shuffled options, sometimes a none-option pair."""
    cands = [{"id": str(c["id"]), "text": c["text"]} for c in ep["candidates"]]
    if shuffle:
        rng.shuffle(cands)
    gold = positives_of(ep)
    if p_none > 0 and rng.random() < p_none:
        none_id = NONE_ID
        if any(c["id"] == none_id for c in cands):
            none_id = NONE_ID + "x"
        if rng.random() < 0.5:
            kept = [c for c in cands if c["id"] not in set(gold)]
            if kept:
                cands = kept
                gold = [none_id]
        lang = ep.get("lang") if ep.get("lang") in NONE_TEXT else "en"
        cands.append({"id": none_id, "text": NONE_TEXT[lang]})
    if len(cands) < 2 or not gold:
        raise ValueError(f"{ep.get('id')}: view has no decision")
    return cands, gold


def pointer_nll(logits, gold_mask, valid=None):
    """Listwise NLL. One gold column is ordinary cross-entropy.

    Padded option columns (valid = 0) are pushed out of the softmax.
    """
    import mlx.core as mx
    neg = mx.array(-1e9, dtype=logits.dtype)
    if valid is not None:
        logits = mx.where(valid > 0, logits, neg)
    pos = mx.where(gold_mask > 0, logits, neg)
    return mx.mean(mx.logsumexp(logits, axis=-1) - mx.logsumexp(pos, axis=-1))


def load_rule_episodes(root: Path, limit: int) -> list[dict]:
    episodes = []
    paths = sorted(root.glob("batch-*/episodes.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no rule episodes under {root}")
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                episodes.append(json.loads(line))
                if limit and len(episodes) >= limit:
                    guard_training_cut(episodes)
                    return episodes
    guard_training_cut(episodes)
    return episodes


def load_episode_files(paths: list[Path]) -> list[dict]:
    """A prepared mix, read whole. The guard still runs on every row."""
    episodes = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            episodes.extend(json.loads(line) for line in handle if line.strip())
    too_wide = [ep.get("id") for ep in episodes
                if len(ep["candidates"]) > OPTION_PAD]
    if too_wide:
        raise RuntimeError(
            f"{len(too_wide)} episodes exceed the option pad ({too_wide[:4]})")
    guard_training_cut(episodes)
    return episodes


def resolve_weights() -> Path:
    ref = HUB / "refs" / "v1"
    if not ref.is_file():
        raise FileNotFoundError(
            f"{ref} missing; the v1 snapshot has to be cached first")
    snap = (HUB / "snapshots" / ref.read_text(encoding="utf-8").strip())
    if not (snap / "model.safetensors").exists():
        raise FileNotFoundError(snap)
    return snap


def load_family(source: Path, weights: Path):
    sys.path.insert(0, str(source))
    from tinyjev.families.pointer import PointerFamily
    manifest = json.loads((weights / "tinyjev.json").read_text(encoding="utf-8"))
    return PointerFamily(weights, manifest)


def encode_decision(family, episode_id: str, state: str, question: str,
                    cands: list[dict], gold: list[str]) -> dict:
    record = {
        "id": episode_id,
        "state": state,
        "questions": [{
            "id": "decision",
            "type": "choice",
            "instructions": question,
            "criteria": {c["id"]: c["text"] for c in cands},
        }],
    }
    enc = family.encode(record)
    question_enc = enc.questions[0]
    ids = list(enc.prefix) + list(enc.rows[0])
    index = {key: i for i, key in enumerate(question_enc["keys"])}
    missing = [g for g in gold if g not in index]
    if missing:
        raise ValueError(f"{episode_id}: gold {missing} not in options")
    return {
        "ids": ids,
        "decide": question_enc["decide"],
        "opts": question_enc["opts"],
        "gold": [index[g] for g in gold],
        "k": len(question_enc["keys"]),
    }


def collate(rows: list[dict], pad_id: int):
    import mlx.core as mx
    width = max(len(row["ids"]) for row in rows)
    width = ((width + 127) // 128) * 128
    if any(row["k"] > OPTION_PAD for row in rows):
        raise ValueError(f"option count exceeds {OPTION_PAD}")
    tokens = [
        row["ids"] + [pad_id] * (width - len(row["ids"])) for row in rows
    ]
    opts, mask, valid = [], [], []
    for row in rows:
        pad_n = OPTION_PAD - row["k"]
        opts.append(list(row["opts"]) + [0] * pad_n)
        gmask = [0.0] * OPTION_PAD
        for g in row["gold"]:
            gmask[g] = 1.0
        mask.append(gmask)
        valid.append([1.0] * row["k"] + [0.0] * pad_n)
    return (
        mx.array(tokens),
        mx.array([row["decide"] for row in rows]),
        mx.array(opts),
        mx.array(mask),
        mx.array(valid),
    )


def batch_nll(model, tokens, decide, opts, gold_mask, valid):
    import mlx.core as mx
    hidden = model.backbone(tokens).astype(mx.float32)
    batch = mx.arange(hidden.shape[0])
    query = model.head_q(hidden[batch, decide])
    key = model.head_k(hidden[batch[:, None], opts])
    logits = mx.sum(key * query[:, None, :], axis=-1) / math.sqrt(HEAD_DIM)
    return pointer_nll(logits, gold_mask, valid)


def build_student(weights: Path):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3 import Qwen3Model
    from mlx_lm.tuner.lora import LoRALinear
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from tinyjev.backends.mlx_backend import qwen3_args

    config = json.loads((weights / "config.json").read_text(encoding="utf-8"))

    class Trainable(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Qwen3Model(qwen3_args(config))
            self.head_q = nn.Linear(config["hidden_size"], HEAD_DIM, bias=True)
            self.head_k = nn.Linear(config["hidden_size"], HEAD_DIM, bias=True)

    model = Trainable()
    loaded = mx.load(str(weights / "model.safetensors"))
    model.backbone.load_weights(list(loaded.items()))
    head = mx.load(str(weights / "head.safetensors"))
    model.head_q.weight = head["q.weight"]
    model.head_q.bias = head["q.bias"]
    model.head_k.weight = head["k.weight"]
    model.head_k.bias = head["k.bias"]
    mx.eval(model.parameters())
    model.backbone.freeze()
    linear_to_lora_layers(
        model.backbone,
        len(model.backbone.layers),
        {
            "rank": LORA_RANK,
            "scale": LORA_ALPHA / LORA_RANK,
            "dropout": 0.0,
            "keys": list(LORA_KEYS),
        },
    )
    n_lora = sum(
        1 for _, module in model.named_modules()
        if isinstance(module, LoRALinear))
    if n_lora != EXPECTED_LORA_LAYERS:
        sample = [k for k, _ in model.backbone.layers[0].named_modules()][:12]
        raise RuntimeError(
            f"expected {EXPECTED_LORA_LAYERS} LoRA linears, found {n_lora}; "
            f"layer keys start {sample}")
    grad_checkpoint(model.backbone.layers[0])
    return model


def trainable_count(model) -> int:
    from mlx.utils import tree_flatten
    return sum(int(value.size) for _, value in tree_flatten(
        model.trainable_parameters()))


def save_adapter(model, path: Path) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(path), flat)


def load_adapter(model, path: Path) -> None:
    if path.is_file():
        model.load_weights(str(path), strict=False)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_tensor_tree(tree, path: Path) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.safetensors")
    mx.save_safetensors(str(tmp), dict(tree_flatten(tree)))
    tmp.replace(path)


def load_optimizer(optimizer, model, path: Path) -> bool:
    """Keep saved Adam moments to overlay after the first gradient tree exists.

    Replacing the state before that tree exists drops any parameter the
    checkpoint did not track, and the next update crashes.
    """
    if not path.is_file():
        return False
    import mlx.core as mx
    optimizer._pending_moments = dict(mx.load(str(path)).items())
    return True


def apply_pending_moments(optimizer, gradients) -> None:
    pending = getattr(optimizer, "_pending_moments", None)
    if not pending:
        return
    import mlx.core as mx
    optimizer.init(gradients)
    covered = 0
    for key, value in pending.items():
        node = optimizer.state
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[int(part)] if isinstance(node, list) else node[part]
        have = node.get(parts[-1]) if isinstance(node, dict) else None
        if getattr(have, "shape", None) != getattr(value, "shape", None):
            continue
        node[parts[-1]] = value
        covered += 1
    optimizer._initialized = True
    mx.eval(optimizer.state)
    optimizer._pending_moments = None
    print(f"optimizer moments {covered}/{len(pending)}", flush=True)


def save_training_checkpoint(model, optimizer, out: Path,
                             progress: dict) -> None:
    save_tensor_tree(model.trainable_parameters(),
                     out / "adapters.safetensors")
    save_tensor_tree(optimizer.state, out / "optimizer.safetensors")
    write_json(out / "checkpoint.json", progress)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def groups_of(episodes: list[dict]) -> dict[int, list[int]]:
    pools: dict[int, list[int]] = defaultdict(list)
    for index, ep in enumerate(episodes):
        weight = FAMILY_WEIGHT.get(ep["family"], 1)
        k = len(ep["candidates"])
        pools[k].extend([index] * weight)
    return pools


def take_batch(pools: dict[int, list[int]], batch: int,
               rng: random.Random) -> list[int]:
    ready = [k for k, ids in pools.items() if len(ids) >= batch]
    if not ready:
        ready = [k for k, ids in pools.items() if ids]
        if not ready:
            return []
    k = max(ready, key=lambda key: len(pools[key]))
    ids = pools[k]
    rng.shuffle(ids)
    take, pools[k] = ids[:batch], ids[batch:]
    return take


def encode_rows(family, episodes, indices, rng, pool, *, train: bool,
                shuffle: bool | None = None):
    views = []
    do_shuffle = train if shuffle is None else shuffle
    for index in indices:
        ep = episodes[index]
        cands, gold = decision_view(
            ep, rng, p_none=0.25 if train else 0.0, shuffle=do_shuffle)
        views.append((ep, cands, gold))

    def one(view):
        ep, cands, gold = view
        return encode_decision(
            family, ep["id"], ep["state"], ep["question"], cands, gold)

    return list(pool.map(one, views))


def pointer_logits(model, tokens, decide, opts, valid):
    import mlx.core as mx
    hidden = model.backbone(tokens).astype(mx.float32)
    batch = mx.arange(hidden.shape[0])
    query = model.head_q(hidden[batch, decide])
    key = model.head_k(hidden[batch[:, None], opts])
    logits = mx.sum(key * query[:, None, :], axis=-1) / math.sqrt(HEAD_DIM)
    neg = mx.array(-1e9, dtype=logits.dtype)
    return mx.where(valid > 0, logits, neg)


def predict_hits(model, tokens, decide, opts, gold_mask, valid) -> tuple[int, int]:
    import mlx.core as mx
    logits = pointer_logits(model, tokens, decide, opts, valid)
    pred = mx.argmax(logits, axis=-1)
    rows = mx.arange(logits.shape[0])
    picked = gold_mask[rows, pred]
    mx.eval(picked)
    hits = int(mx.sum(picked).item())
    return hits, int(logits.shape[0])


def fixed_hits(model, family, episodes, pad_id, pool, batch: int, *,
               shuffle: bool = False, seed: int = 0) -> list[bool]:
    """Score each episode. Written order unless shuffle is set."""
    import mlx.core as mx
    model.eval()
    flags = []
    rng = random.Random(seed)
    for start in range(0, len(episodes), batch):
        chunk = episodes[start:start + batch]
        rows = encode_rows(
            family, chunk, list(range(len(chunk))), rng, pool,
            train=False, shuffle=shuffle)
        tokens, decide, opts, mask, valid = collate(rows, pad_id)
        logits = pointer_logits(model, tokens, decide, opts, valid)
        pred = mx.argmax(logits, axis=-1)
        picked = mask[mx.arange(logits.shape[0]), pred]
        mx.eval(picked)
        flags.extend(bool(item) for item in picked.tolist())
        mx.clear_cache()
    model.train()
    return flags


def battery_report(episodes: list[dict], hits: list[bool]) -> dict:
    """Forced accuracy of one fixed order, split by family and K."""
    if len(episodes) != len(hits):
        raise ValueError("one hit flag per episode")
    cells = {}
    for ep, hit in zip(episodes, hits):
        key = (ep["family"], len(ep["candidates"]))
        cell = cells.setdefault(key, {"hits": 0, "n": 0})
        cell["n"] += 1
        cell["hits"] += int(hit)
    by_family = {}
    by_capability = {}
    by_domain = {}
    by_capability_domain = {}
    for (family, _k), cell in cells.items():
        bucket = by_family.setdefault(family, {"hits": 0, "n": 0})
        bucket["hits"] += cell["hits"]
        bucket["n"] += cell["n"]
    for ep, hit in zip(episodes, hits):
        for buckets, key in (
            (by_capability, ep.get("capability")),
            (by_domain, ep.get("domain")),
            (by_capability_domain, (
                f"{ep.get('capability')}:{ep.get('domain')}"
                if ep.get("capability") and ep.get("domain") else None)),
        ):
            if key:
                bucket = buckets.setdefault(key, {"hits": 0, "n": 0})
                bucket["hits"] += int(hit)
                bucket["n"] += 1

    def rates(buckets):
        return {
            key: {
                **bucket,
                "forced": bucket["hits"] / bucket["n"],
                "wilson95_lower": wilson_lower(bucket["hits"], bucket["n"]),
            }
            for key, bucket in sorted(buckets.items())
        }

    n = len(episodes)
    got = sum(int(hit) for hit in hits)
    return {
        "n": n,
        "hits": got,
        "forced": got / n if n else None,
        "by_family": rates(by_family),
        "by_capability": rates(by_capability),
        "by_domain": rates(by_domain),
        "by_capability_domain": rates(by_capability_domain),
        "by_k": {
            f"{family}:k={k}": {**cell, "forced": cell["hits"] / cell["n"]}
            for (family, k), cell in sorted(cells.items())
        },
    }


def wilson_lower(hits: int, n: int, z: float = 1.959964) -> float:
    if not n:
        return 0.0
    p = hits / n
    denominator = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(
        p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denominator


def holdout_accuracy(model, family, episodes, pools, batch, pad_id,
                     pool, limit: int) -> float | None:
    import mlx.core as mx
    model.eval()
    hits = 0
    seen = 0
    work = {k: list(v) for k, v in pools.items()}
    rng = random.Random(0)
    while seen < limit:
        indices = take_batch(work, batch, rng)
        if len(indices) < batch:
            break
        rows = encode_rows(
            family, episodes, indices, rng, pool, train=False)
        tokens, decide, opts, mask, valid = collate(rows, pad_id)
        got, n = predict_hits(model, tokens, decide, opts, mask, valid)
        hits += got
        seen += n
        mx.clear_cache()
    model.train()
    if not seen:
        return None
    return hits / seen


def set_metal_budget() -> dict:
    import mlx.core as mx
    from mlx_lm.utils import maybe_set_recommended_wired_limit
    previous = maybe_set_recommended_wired_limit()
    info = dict(mx.device_info())
    cap = info.get("max_recommended_working_set_size")
    if cap:
        mx.set_cache_limit(int(cap * 0.25))
    info["previous_wired_limit"] = previous
    return info


def probe_batch(model, family, episodes, pools, pad_id, pool, cap: int,
                loss_and_grad) -> int:
    """Largest batch whose backward stays inside 80% of the Metal budget.

    Gradients are computed and dropped. The weights are not updated.
    """
    import mlx.core as mx
    if not cap:
        return 4
    chosen = 4
    rng = random.Random(1)
    work = {k: list(v) for k, v in pools.items()}
    for batch in (16, 32, 48, 64, 96, 128, 160):
        indices = take_batch({k: list(v) for k, v in work.items()}, batch, rng)
        if len(indices) < batch:
            break
        try:
            mx.reset_peak_memory()
            rows = encode_rows(
                family, episodes, indices, rng, pool, train=True)
            arrays = collate(rows, pad_id)
            loss, grads = loss_and_grad(model, *arrays)
            from mlx.utils import tree_flatten
            mx.eval(loss, *[value for _, value in tree_flatten(grads)])
            peak = mx.get_peak_memory()
            print(f"probe batch {batch} peak_gb {peak / 2**30:.2f}", flush=True)
            del loss, grads
            mx.clear_cache()
            if peak > 0.55 * cap:
                break
            chosen = batch
        except Exception as exc:
            print(f"probe batch {batch} stopped: {exc}", flush=True)
            mx.clear_cache()
            break
    return chosen


def train(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    import mlx.core as mx
    import mlx.optimizers as opt
    from mlx.nn.utils import value_and_grad
    from mlx.utils import tree_map

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.live:
        flag = ROOT / "artifacts" / "loop" / "TRAINING"
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("pointer\n", encoding="utf-8")
    prior = read_json(out / "state.json")
    prior.update({
        "status": "starting",
        "pid": os.getpid(),
        "updated_utc": utcnow(),
    })
    write_json(out / "state.json", prior)
    weights = resolve_weights()
    info = set_metal_budget()
    cap = int(info.get("max_recommended_working_set_size") or 0)
    print(f"weights {weights}", flush=True)
    print(f"metal { {k: info.get(k) for k in ('device_name', 'architecture', 'max_recommended_working_set_size', 'memory_size')} }", flush=True)
    sources = [Path(p) for p in args.episodes.split(",") if p.strip()]
    if sources:
        episodes = load_episode_files(sources)
    else:
        episodes = load_rule_episodes(RULE_ROOT, args.limit)
    parts = EC.split_by_group(episodes, SPLIT_SEED, HOLDOUT_CUTS)
    train_eps = parts["train"]
    hold_eps = parts["holdout"]
    print(
        f"episodes {len(episodes)} train {len(train_eps)} "
        f"holdout {len(hold_eps)}",
        flush=True)
    print("loading backbone", flush=True)
    family = load_family(TINYJEV_SRC, weights)
    model = build_student(weights)
    adapter = out / "adapters.safetensors"
    load_adapter(model, adapter)
    if hasattr(model.backbone, "embed_tokens"):
        model.backbone.embed_tokens.freeze()
    checkpoint_path = out / "checkpoint.json"
    checkpoint = read_json(checkpoint_path)
    n_train = trainable_count(model)
    print(f"trainable_params {n_train}", flush=True)
    if not 20_000_000 <= n_train <= 80_000_000:
        raise RuntimeError(f"unexpected trainable count {n_train}")
    model.train()
    lr = args.lr or LR
    warmup_steps = args.warmup if args.warmup >= 0 else WARMUP_STEPS
    flat = opt.linear_schedule(lr, lr, 10**9)
    if warmup_steps:
        warmup = opt.linear_schedule(0.0, lr, warmup_steps)
        base_schedule = opt.join_schedules([warmup, flat], [warmup_steps])
    else:
        base_schedule = flat
    resume_offset = int(checkpoint.get("step") or 0)
    optimizer_path = out / "optimizer.safetensors"
    schedule = base_schedule
    if resume_offset and not optimizer_path.is_file():
        schedule = lambda local_step: base_schedule(local_step + resume_offset)
    optimizer = opt.AdamW(learning_rate=schedule, weight_decay=0.01,
                          bias_correction=True)
    optimizer_restored = load_optimizer(optimizer, model, optimizer_path)
    loss_and_grad = value_and_grad(model, batch_nll)
    workers = max(4, (os.cpu_count() or 8) // 2)
    pool = ThreadPoolExecutor(max_workers=workers)
    train_pools = groups_of(train_eps)
    hold_pools = groups_of(hold_eps)
    batch = args.batch or probe_batch(
        model, family, train_eps, train_pools, family.pad_token_id, pool,
        cap, loss_and_grad)
    print(f"batch {batch} workers {workers}", flush=True)
    counts = Counter(ep["family"] for ep in train_eps)
    manifest_path = out / "train_manifest.json"
    previous_manifest = read_json(manifest_path)
    write_json(manifest_path, {
        "task": TASK,
        "started_utc": previous_manifest.get("started_utc") or utcnow(),
        "resumed_utc": utcnow(),
        "resume_count": int(previous_manifest.get("resume_count") or 0) + 1,
        "resume_step": int(checkpoint.get("step") or 0),
        "optimizer_restored": optimizer_restored,
        "init": args.init or "TinyJev-4B-v1",
        "sources": [str(p) for p in sources] or [str(RULE_ROOT)],
        "weights": str(weights),
        "loss": "listwise NLL over pointer logits (CE when one gold)",
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_keys": list(LORA_KEYS),
        "lr": lr,
        "warmup_steps": warmup_steps,
        "p_none": 0.25,
        "family_weight": FAMILY_WEIGHT,
        "split_seed": SPLIT_SEED,
        "holdout_fraction": 0.1,
        "n_train": len(train_eps),
        "n_holdout": len(hold_eps),
        "by_family": dict(counts),
        "batch": batch,
        "save_every": args.save_every,
        "trainable_params": n_train,
        "battery": "excluded",
        "sealed": "excluded",
    })
    stop = {"flag": False}

    def request_stop(signum, _frame):
        stop["flag"] = True
        print(f"stop requested ({signum})", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    step = int(checkpoint.get("step") or 0)
    resume_step = step
    decisions = int(checkpoint.get("decisions") or 0)
    epoch = int(checkpoint.get("epoch") or 0)
    best = checkpoint.get("best_holdout")
    elapsed_base = float(checkpoint.get("elapsed_s") or 0.0)
    rng = random.Random(args.seed + step)
    started = time.time()
    while not stop["flag"]:
        if args.steps and step >= args.steps:
            break
        if not any(len(v) >= batch for v in train_pools.values()):
            epoch += 1
            train_pools = groups_of(train_eps)
        indices = take_batch(train_pools, batch, rng)
        if len(indices) < batch:
            epoch += 1
            train_pools = groups_of(train_eps)
            continue
        t0 = time.time()
        rows = encode_rows(
            family, train_eps, indices, rng, pool, train=True)
        arrays = collate(rows, family.pad_token_id)
        saved_params = model.trainable_parameters()
        saved_state = None
        try:
            loss, grads = loss_and_grad(model, *arrays)
            apply_pending_moments(optimizer, grads)
            saved_state = tree_map(lambda value: value, optimizer.state)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)
        except RuntimeError as exc:
            if "Memory" not in str(exc) and "Resource limit" not in str(exc):
                raise
            # A failed eval leaves lazy graphs in the weights and moments;
            # retrying on top of them grows the graph until Metal runs out
            # of buffer handles.
            model.update(saved_params)
            if saved_state is not None:
                optimizer.state = saved_state
            loss = grads = None
            mx.clear_cache()
            batch = max(8, batch // 2)
            print(f"oom, batch now {batch}: {str(exc)[:120]}", flush=True)
            continue
        mx.clear_cache()
        step += 1
        decisions += batch
        elapsed = time.time() - t0
        tokens = int(arrays[0].shape[0] * arrays[0].shape[1])
        row = {
            "step": step,
            "epoch": epoch,
            "loss": float(loss.item()),
            "decisions": decisions,
            "tokens": tokens,
            "step_s": round(elapsed, 3),
            "tok_s": round(tokens / elapsed, 1) if elapsed else None,
            "ts": utcnow(),
        }
        evaluated = (step % args.eval_every == 0
                     or (args.steps and step >= args.steps))
        if evaluated:
            score = holdout_accuracy(
                model, family, hold_eps, hold_pools, batch,
                family.pad_token_id, pool, args.eval_rows)
            row["holdout_forced"] = score
            if score is not None and (best is None or score >= best):
                best = score
                save_adapter(model, out / "best.safetensors")
        total_elapsed = elapsed_base + time.time() - started
        if evaluated or step % args.save_every == 0:
            save_training_checkpoint(model, optimizer, out, {
                "step": step,
                "decisions": decisions,
                "epoch": epoch,
                "best_holdout": best,
                "elapsed_s": round(total_elapsed, 1),
                "updated_utc": utcnow(),
            })
        append_jsonl(out / "curve.jsonl", row)
        note = (
            f"step {step} loss {row['loss']:.4f} "
            f"{row['tok_s']} tok/s"
        )
        if row.get("holdout_forced") is not None:
            note += f" holdout {row['holdout_forced']:.4f}"
        write_json(out / "state.json", {
            "status": "training" if not stop["flag"] else "stopped",
            "pid": os.getpid(),
            "step": step,
            "session_step": step - resume_step,
            "resumed_from_step": resume_step,
            "epoch": epoch,
            "decisions": decisions,
            "train_loss": row["loss"],
            "holdout_forced": row.get("holdout_forced", best),
            "best_holdout": best,
            "batch": batch,
            "tok_s": row["tok_s"],
            "updated_utc": utcnow(),
            "elapsed_s": round(total_elapsed, 1),
            "model": "TinyJev-4B-v1",
            "note": note,
            "battery": "excluded",
        })
        print(note, flush=True)
    save_training_checkpoint(model, optimizer, out, {
        "step": step,
        "decisions": decisions,
        "epoch": epoch,
        "best_holdout": best,
        "elapsed_s": round(elapsed_base + time.time() - started, 1),
        "updated_utc": utcnow(),
    })
    state_path = out / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["status"] = "stopped"
        state["pid"] = os.getpid()
        state["updated_utc"] = utcnow()
        write_json(state_path, state)
    pool.shutdown(wait=False, cancel_futures=True)


def eval_battery(args: argparse.Namespace) -> None:
    """Score battery_dev with the saved adapter. The rows are not trained."""
    from data import battery_dev as BD

    out = Path(args.out)
    weights = resolve_weights()
    set_metal_budget()
    family = load_family(TINYJEV_SRC, weights)
    model = build_student(weights)
    adapter = out / "adapters.safetensors"
    if not adapter.is_file():
        raise FileNotFoundError(adapter)
    load_adapter(model, adapter)
    if args.episodes:
        from data import benchmark_v2 as BV2
        source = Path(args.episodes)
        episodes = BV2.load_battery(
            source, limit_per_capability=args.limit_per_capability)
        battery_name = str(source)
        battery_sha = BV2.battery_sha(source)
    else:
        episodes = BD.load_battery()
        battery_name = "data/battery_dev.jsonl"
        battery_sha = BD.battery_sha(episodes)
    if any(len(ep["candidates"]) > OPTION_PAD for ep in episodes):
        raise RuntimeError("battery row exceeds the option pad")
    workers = max(4, (os.cpu_count() or 8) // 2)
    pool = ThreadPoolExecutor(max_workers=workers)
    started = time.time()
    batch = args.batch or 8
    hits = fixed_hits(
        model, family, episodes, family.pad_token_id, pool, batch)
    report = battery_report(episodes, hits)
    orders = [{"order": "written", "forced": report["forced"],
               "hits": report["hits"]}]
    for index in range(args.orders):
        shuffled = fixed_hits(
            model, family, episodes, family.pad_token_id, pool, batch,
            shuffle=True, seed=args.seed + index + 1)
        counted = sum(int(hit) for hit in shuffled)
        orders.append({
            "order": f"shuffle-{index + 1}",
            "seed": args.seed + index + 1,
            "hits": counted,
            "forced": counted / len(episodes),
        })
    checkpoint = read_json(out / "checkpoint.json")
    payload = {
        "task": TASK,
        "role": "regression",
        "decides": False,
        "battery": battery_name,
        "n": report["n"],
        "sha256": battery_sha,
        "adapter_step": checkpoint.get("step"),
        "order": "written",
        "shuffle": False,
        "orders": orders,
        "seconds": round(time.time() - started, 1),
        "measured_utc": utcnow(),
        **report,
    }
    dest = out / args.report
    write_json(dest, payload)
    pool.shutdown(wait=False, cancel_futures=True)
    print(
        f"battery {report['hits']}/{report['n']} "
        f"{report['forced']:.4f} step {payload['adapter_step']}",
        flush=True)
    for family_name, cell in report["by_family"].items():
        print(
            f"{family_name} {cell['hits']}/{cell['n']} {cell['forced']:.4f}",
            flush=True)
    for order in orders[1:]:
        print(
            f"{order['order']} {order['hits']}/{report['n']} "
            f"{order['forced']:.4f}",
            flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "eval-battery"))
    parser.add_argument("--out", default="artifacts/checkpoints/pointer/loop")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-rows", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--orders", type=int, default=0)
    parser.add_argument("--episodes", default="",
                        help="comma-separated jsonl files instead of the "
                             "rule stream")
    parser.add_argument("--limit-per-capability", type=int, default=0,
                        help="evaluation-only cap for each benchmark v2 "
                             "capability")
    parser.add_argument("--report", default="battery_regression.json",
                        help="evaluation report name inside --out")
    parser.add_argument("--init", default="",
                        help="label recorded for the adapter the run starts "
                             "from")
    parser.add_argument("--lr", type=float, default=0.0,
                        help="peak learning rate; 0 keeps LR")
    parser.add_argument("--warmup", type=int, default=-1,
                        help="warmup steps; -1 keeps WARMUP_STEPS")
    args = parser.parse_args()
    if args.command == "eval-battery":
        eval_battery(args)
        return
    train(args)


if __name__ == "__main__":
    main()
