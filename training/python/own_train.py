"""Train the publishable decision model without an external checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
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
def hits(model: OwnDecisionModel, rows: list[dict], max_length: int,
         device: torch.device) -> int:
    model.eval()
    found = 0
    for row in rows:
        inputs, gold = row_tensors(model, row, max_length, device)
        found += int(model(*inputs).argmax(-1).item() == gold)
    return found


def accuracy(model: OwnDecisionModel, rows: list[dict], max_length: int,
             device: torch.device) -> float:
    return hits(model, rows, max_length, device) / len(rows) if rows else 0.0


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def option_texts(row: dict) -> set[str]:
    return {normalise(candidate["text"]) for candidate in row["candidates"]}


def gold_text(row: dict) -> str | None:
    if row["answer"] == "unknown":
        return None
    for candidate in row["candidates"]:
        if candidate["id"] == row["answer"]:
            return normalise(candidate["text"])
    return None


def split_unseen(rows: list[dict], fraction: float,
                 seed: int) -> tuple[list[dict], list[dict], int]:
    """Keep evaluation options completely absent from training.

    A mixed row is dropped. An unseen row uses only held-out texts, so the
    model cannot win by recognising one familiar distractor.
    """
    if not 0 < fraction < 1:
        raise ValueError("unseen fraction must be between 0 and 1")
    texts = sorted({text for row in rows for text in option_texts(row)})
    if len(texts) < 2:
        raise ValueError("unseen split needs at least two option texts")
    held_n = min(len(texts) - 1, max(1, round(len(texts) * fraction)))
    held = set(random.Random(seed).sample(texts, held_n))
    train, unseen = [], []
    for row in rows:
        used = option_texts(row)
        gold = gold_text(row)
        if used <= held and gold in held:
            unseen.append(row)
        elif used.isdisjoint(held):
            train.append(row)
    if not train or not unseen:
        raise ValueError("unseen split left one side empty")
    return train, unseen, len(held)


def leaked_texts(train_rows: list[dict], unseen_rows: list[dict]) -> list[str]:
    seen = {text for row in train_rows for text in option_texts(row)}
    return sorted({gold for row in unseen_rows
                   if (gold := gold_text(row)) in seen})


def chance(rows: list[dict]) -> float:
    return sum(1 / (len(row["candidates"]) + 1) for row in rows) / len(rows)


def wilson_lower(successes: int, n: int, z: float = 1.959964) -> float:
    if n <= 0:
        return 0.0
    p = successes / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / (1 + z * z / n))


def batches(rows: list[dict], size: int, seed: int) -> list[list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(len(row["candidates"]), []).append(row)
    rng = random.Random(seed)
    chunks = []
    for pool in grouped.values():
        rng.shuffle(pool)
        chunks.extend(pool[start:start + size]
                      for start in range(0, len(pool), size))
    rng.shuffle(chunks)
    return chunks


def batch_tensors(model: OwnDecisionModel, rows: list[dict], max_length: int,
                  device: torch.device) -> tuple[tuple[torch.Tensor, ...],
                                                 torch.Tensor]:
    width = len(rows[0]["candidates"])
    if any(len(row["candidates"]) != width for row in rows):
        raise ValueError("a batch must share its option count")
    tokenizer = model.tokenizer
    state = tokenizer.batch([row["state"] for row in rows], max_length, device)
    question = tokenizer.batch([row["question"] for row in rows],
                               max_length, device)
    flat = [candidate["text"] for row in rows for candidate in row["candidates"]]
    option_ids, option_tokens = tokenizer.batch(flat, max_length, device)
    batch = len(rows)
    golds = []
    for row in rows:
        ids = [candidate["id"] for candidate in row["candidates"]]
        golds.append(width if row["answer"] == "unknown"
                     else ids.index(row["answer"]))
    inputs = (
        *state,
        *question,
        option_ids.view(batch, width, -1),
        option_tokens.view(batch, width, -1),
        torch.ones((batch, width), dtype=torch.bool, device=device),
    )
    return inputs, torch.tensor(golds, device=device)


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
    held_texts = 0
    if args.unseen_fraction:
        train_rows, holdout, held_texts = split_unseen(
            rows, args.unseen_fraction, args.seed)
    else:
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
    chunks = batches(train_rows, args.batch_size, args.seed)
    for step in range(1, args.steps + 1):
        inputs, gold = batch_tensors(
            model, chunks[(step - 1) % len(chunks)], args.max_length, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(*inputs)
        loss = torch.nn.functional.cross_entropy(logits, gold)
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
        "held_out_texts": held_texts,
        "batch_size": args.batch_size,
        "first_loss": first_loss,
        "last_loss": float(loss.detach()),
        "train_accuracy": accuracy(model, train_rows, args.max_length, device),
        "holdout_hits": hits(model, holdout, args.max_length, device),
        "device": str(device),
        "seed": args.seed,
        "elapsed_s": round(time.time() - started, 1),
        "sources": [{"path": str(path), "sha256": digest(path)}
                    for path in args.episodes],
    }
    result["holdout_accuracy"] = (
        result["holdout_hits"] / result["holdout_rows"])
    result["holdout_chance"] = chance(holdout)
    result["holdout_wilson_lower"] = wilson_lower(
        result["holdout_hits"], result["holdout_rows"])
    leaks = leaked_texts(train_rows, holdout) if args.unseen_fraction else []
    result["checks"] = {
        "loss_decreased": result["last_loss"] < result["first_loss"],
        "random_initialization": not result["external_checkpoint"],
    }
    if args.unseen_fraction:
        result["checks"].update({
            "holdout_clean": not leaks,
            "unseen_beats_chance": (
                result["holdout_wilson_lower"] > result["holdout_chance"]),
        })
    else:
        result["checks"]["train_memorized"] = result["train_accuracy"] >= 0.95
    result["pass"] = all(result["checks"].values())
    args.out.mkdir(parents=True, exist_ok=True)
    tensors = {name: value.detach().cpu().contiguous()
               for name, value in model.state_dict().items()}
    save_file(tensors, args.out / "model.safetensors")
    (args.out / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    (args.out / "train.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.gate:
        args.gate.parent.mkdir(parents=True, exist_ok=True)
        args.gate.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser()
    cli.add_argument("--episodes", type=Path, nargs="+", required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--gate", type=Path)
    cli.add_argument("--steps", type=int, default=1000)
    cli.add_argument("--batch-size", type=int, default=8)
    cli.add_argument("--unseen-fraction", type=float, default=0.0)
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
