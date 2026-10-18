"""Train the publishable decision model without an external checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from model.own_backbone import OwnDecisionModel, OwnModelConfig


def read_episodes(paths: list[Path], limit: int = 0) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                candidates = row.get("candidates") or []
                ids = [candidate.get("id") for candidate in candidates]
                if (row.get("state") and row.get("question") and candidates
                        and row.get("answer") in ids + ["unknown"]):
                    rows.append(row)
                    if limit and len(rows) >= limit:
                        return rows
    return rows


def row_tensors(model: OwnDecisionModel, row: dict, max_length: int,
                device: torch.device) -> tuple[tuple[torch.Tensor, ...], int]:
    tokenizer = model.tokenizer
    state = tokenizer.batch([row["state"]], max_length, device)
    question = tokenizer.batch([row["question"]], max_length, device)
    options = row["candidates"]
    option_ids, option_tokens = tokenizer.batch(
        [candidate["text"] for candidate in options], max_length, device)
    present = torch.ones((1, len(options)), dtype=torch.bool, device=device)
    ids = [candidate["id"] for candidate in options]
    gold = len(options) if row["answer"] == "unknown" \
        else ids.index(row["answer"])
    inputs = (*state, *question, option_ids.unsqueeze(0),
              option_tokens.unsqueeze(0), present)
    return inputs, gold


@torch.inference_mode()
def accuracy(model: OwnDecisionModel, rows: list[dict], max_length: int,
             device: torch.device) -> float:
    model.eval()
    hits = 0
    for row in rows:
        inputs, gold = row_tensors(model, row, max_length, device)
        hits += int(model(*inputs).argmax(-1).item() == gold)
    return hits / len(rows) if rows else 0.0


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_episodes(args.episodes, args.limit)
    if len(rows) < 2:
        raise ValueError("at least two valid episodes are required")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    split = max(1, min(len(rows) - 1, round(len(rows) * 0.9)))
    train_rows, holdout = rows[:split], rows[split:]
    config = OwnModelConfig(
        max_length=args.max_length,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        d_ff=args.d_ff,
        pointer_layers=args.pointer_layers,
        dropout=args.dropout,
    )
    device = torch.device(args.device)
    model = OwnDecisionModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    first_loss = None
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        row = train_rows[(step - 1) % len(train_rows)]
        inputs, gold = row_tensors(model, row, args.max_length, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(*inputs)
        loss = torch.nn.functional.cross_entropy(
            logits, torch.tensor([gold], device=device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.detach())
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({"step": step, "loss": float(loss.detach()),
                              "elapsed_s": round(time.time() - started, 1)}),
                  flush=True)
    result = {
        **model.lineage(),
        "task": "T-own-backbone",
        "steps": args.steps,
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout),
        "first_loss": first_loss,
        "last_loss": float(loss.detach()),
        "holdout_accuracy": accuracy(model, holdout, args.max_length, device),
        "device": str(device),
        "seed": args.seed,
        "elapsed_s": round(time.time() - started, 1),
        "sources": [{"path": str(path), "sha256": digest(path)}
                    for path in args.episodes],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tensors = {name: value.detach().cpu().contiguous()
               for name, value in model.state_dict().items()}
    save_file(tensors, args.out / "model.safetensors")
    (args.out / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    (args.out / "train.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser()
    cli.add_argument("--episodes", type=Path, nargs="+", required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--steps", type=int, default=1000)
    cli.add_argument("--limit", type=int, default=0)
    cli.add_argument("--seed", type=int, default=1789)
    cli.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                     else "cpu")
    cli.add_argument("--max-length", type=int, default=256)
    cli.add_argument("--d-model", type=int, default=384)
    cli.add_argument("--heads", type=int, default=8)
    cli.add_argument("--layers", type=int, default=6)
    cli.add_argument("--d-ff", type=int, default=1536)
    cli.add_argument("--pointer-layers", type=int, default=2)
    cli.add_argument("--dropout", type=float, default=0.1)
    cli.add_argument("--lr", type=float, default=3e-4)
    cli.add_argument("--weight-decay", type=float, default=0.01)
    cli.add_argument("--log-every", type=int, default=10)
    return cli


if __name__ == "__main__":
    summary = train(parser().parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
