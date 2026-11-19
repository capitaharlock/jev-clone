"""One training run: episodes in, a checkpoint folder and a report out.

The objective is cross-entropy over the K option logits plus the unknown
logit. Options are shuffled in every step, so the model cannot learn a
favourite position. Without `--init` the run starts from random weights.

    python -m asimovia.training.train --episodes train.jsonl --eval dev.jsonl \
        --out runs/demo --steps 600
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..data.io import read_jsonl, sha256_file, valid_episode
from ..metrics import chance, wilson_lower
from ..model.backbone import DecisionModel, ModelConfig
from ..model.checkpoint import load_config, load_weights, save
from ..model.inference import batch_tensors, group_by_options


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@torch.inference_mode()
def forced_hits(
    model: DecisionModel, rows: list[dict], device: torch.device, batch_size: int = 32
) -> int:
    model.eval()
    found = 0
    for chunk in group_by_options(rows, batch_size):
        inputs, gold = batch_tensors(model, chunk, device)
        found += int((model(*inputs)[:, :-1].argmax(-1) == gold).sum())
    return found


def shuffled(rows: list[dict], rng: random.Random) -> list[dict]:
    views = []
    for row in rows:
        options = [dict(c) for c in row["candidates"]]
        rng.shuffle(options)
        views.append({**row, "candidates": options})
    return views


def schedule(step: int, steps: int, warmup: int) -> float:
    """Linear warmup, then cosine from 1.0 down to a 0.1 floor."""
    if not warmup:
        return 1.0
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, steps - warmup))
    return 0.1 + 0.45 * (1 + math.cos(math.pi * progress))


def build_model(args: argparse.Namespace) -> DecisionModel:
    if args.init:
        config = replace(load_config(args.init.parent), dropout=args.dropout)
        if args.max_length > config.max_length:
            config = replace(config, max_length=args.max_length)
        model = DecisionModel(config)
        load_weights(model, args.init)
        return model
    return DecisionModel(
        ModelConfig(
            max_length=args.max_length,
            d_model=args.d_model,
            n_heads=args.heads,
            n_layers=args.layers,
            d_ff=args.d_ff,
            pointer_layers=args.pointer_layers,
            dropout=args.dropout,
        )
    )


def log(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = [row for path in args.episodes for row in read_jsonl(path) if valid_episode(row)]
    holdout = [row for row in read_jsonl(args.eval) if valid_episode(row)]
    if args.eval_limit:
        holdout = holdout[: args.eval_limit]
    if len(rows) < 2 or not holdout:
        raise ValueError("need at least two training rows and one eval row")
    device = torch.device(args.device)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: schedule(step, args.steps, args.warmup)
    )
    chunks = group_by_options(rows, args.batch_size, args.seed)
    view_rng = random.Random(args.seed + 1)
    watch = holdout[:400]
    started = time.time()
    first_loss = loss = None
    model.train()
    for step in range(1, args.steps + 1):
        chunk = shuffled(chunks[(step - 1) % len(chunks)], view_rng)
        inputs, gold = batch_tensors(model, chunk, device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(*inputs), gold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if first_loss is None:
            first_loss = float(loss.detach())
        elapsed = round(time.time() - started, 1)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            log({"step": step, "loss": float(loss.detach()), "elapsed_s": elapsed})
        if args.eval_every and step % args.eval_every == 0:
            found = forced_hits(model, watch, device)
            log(
                {
                    "step": step,
                    "watch_forced": found / len(watch),
                    "watch_rows": len(watch),
                    "elapsed_s": elapsed,
                }
            )
            model.train()
    hits = forced_hits(model, holdout, device)
    train_sample = rows[: args.train_score_rows] if args.train_score_rows else rows
    result = {
        **model.lineage(),
        "initialization": "continued" if args.init else "random",
        "continued_from": (f"{args.init.parent.name}/{args.init.name}" if args.init else None),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "train_rows": len(rows),
        "train_accuracy": forced_hits(model, train_sample, device) / len(train_sample),
        "first_loss": first_loss,
        "last_loss": float(loss.detach()),
        "holdout_rows": len(holdout),
        "holdout_forced_hits": hits,
        "holdout_forced_accuracy": hits / len(holdout),
        "holdout_forced_chance": chance(holdout),
        "holdout_forced_wilson_lower": wilson_lower(hits, len(holdout)),
        "elapsed_s": round(time.time() - started, 1),
        "sources": [
            {"path": Path(p).name, "sha256": sha256_file(p)} for p in [*args.episodes, args.eval]
        ],
    }
    save(model, args.out)
    (args.out / "train.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    cli.add_argument("--episodes", type=Path, nargs="+", required=True)
    cli.add_argument("--eval", type=Path, required=True)
    cli.add_argument("--eval-limit", type=int, default=0)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--init", type=Path, help="model.safetensors to continue from")
    cli.add_argument("--steps", type=int, default=1000)
    cli.add_argument("--batch-size", type=int, default=8)
    cli.add_argument("--lr", type=float, default=3e-4)
    cli.add_argument("--warmup", type=int, default=100)
    cli.add_argument("--weight-decay", type=float, default=0.01)
    cli.add_argument("--dropout", type=float, default=0.0)
    cli.add_argument("--max-length", type=int, default=1536)
    cli.add_argument("--d-model", type=int, default=384)
    cli.add_argument("--heads", type=int, default=8)
    cli.add_argument("--layers", type=int, default=6)
    cli.add_argument("--d-ff", type=int, default=1536)
    cli.add_argument("--pointer-layers", type=int, default=2)
    cli.add_argument("--seed", type=int, default=1789)
    cli.add_argument("--device", default=default_device())
    cli.add_argument("--log-every", type=int, default=100)
    cli.add_argument("--eval-every", type=int, default=0)
    cli.add_argument("--train-score-rows", type=int, default=500)
    return cli


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(train(parser().parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
