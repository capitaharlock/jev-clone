"""Train and evaluate the one-sequence listwise scorer (#T-listwise-format)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shlex
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import listwise_scorer as LS  # noqa: E402
from training.python import ce_finetune as CF  # noqa: E402
from training.python import ce_overfit as OF  # noqa: E402

TASK = "T-listwise-format"
DEFAULT_WEIGHTS = "mdeberta-v3-base-xnli"
DEFAULT_BATTERY = ROOT / "data" / "battery_dev.jsonl"
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _dump(path, doc: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=2,
                                     sort_keys=True) + "\n")


def command_line(args) -> str:
    parts = ["python", "-m", "training.python.listwise_finetune",
             args.command]
    for key, value in vars(args).items():
        if key == "command" or value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        parts.append(flag if value is True else
                     f"{flag} {shlex.quote(str(value))}")
    return " ".join(parts)


def gold_indices(decisions):
    import torch

    out = []
    for decision in decisions:
        ids = [candidate.id for candidate in decision.candidates]
        out.append(ids.index(decision.gold))
    return torch.tensor(out, dtype=torch.long)


def param_groups(model: LS.ListwiseModel, lr_encoder: float,
                 lr_head: float) -> list[dict]:
    return [
        {"params": model.encoder.parameters(), "lr": lr_encoder,
         "name": "encoder"},
        {"params": model.head.parameters(), "lr": lr_head, "name": "head"},
    ]


def lr_factor(step: int, total: int, warmup: float, floor: float) -> float:
    """Warmup then linear decay. A flat rate makes each cycle end worse
    than its own midpoint, which is how earlier runs threw away their peak."""
    warm = max(1, int(total * warmup))
    if step < warm:
        return (step + 1) / warm
    span = max(1, total - warm)
    done = min(1.0, (step - warm) / span)
    return floor + (1.0 - floor) * (1.0 - done)


def evaluate_holdout(model, tokenizer, decisions, device,
                     batch_size: int) -> dict:
    import torch

    was_training = model.training
    model.eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(decisions), batch_size):
            chunk = decisions[start:start + batch_size]
            scores, _ = LS.forward_decisions(
                model, tokenizer, chunk, device, canonical_order=True)
            best = scores.argmax(dim=-1)
            for row, decision in enumerate(chunk):
                ids = [candidate.id for candidate in decision.candidates]
                pred = ids[int(best[row])]
                rows.append(pred == decision.gold)
    if was_training:
        model.train()
    hits = sum(rows)
    n = len(rows)
    lo, hi = OF.wilson_interval(hits, n)
    return {"n": n, "hits": hits, "accuracy": hits / n,
            "ci95": [lo, hi]}


def evaluate(args) -> dict:
    tokenizer, model, source = LS.load(
        args.weights, args.device,
        None if args.checkpoint == "none" else args.checkpoint)
    scorer = LS.ListwisePairScorer(
        tokenizer, model, args.device,
        f"{TASK}:{Path(args.checkpoint).name}",
        batch_size=args.decisions_per_batch)
    checkpoint = {
        **source,
        "model_version": scorer.id,
        "weights": args.weights,
    }
    doc = CF.evaluate_battery(
        scorer, args.battery, checkpoint=checkpoint, device=args.device)
    doc["task"] = TASK
    doc["format"] = LS.FORMAT
    _dump(args.out, doc)
    print("[listwise] " + CF.figure_line(doc), flush=True)
    return doc


def train(args) -> dict:
    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    loaded = CF.load_episodes(args.episodes)
    episodes = loaded["episodes"]
    split = None
    holdout_eps = []
    if args.holdout > 0:
        split = CF.provisional_split(
            episodes, args.split_seed,
            (("train", 1.0 - args.holdout), ("holdout", args.holdout)))
        episodes = split["parts"]["train"]
        holdout_eps = split["parts"]["holdout"]
        del split["parts"]
    CF.guard_training_cut(episodes)
    decisions = [CF.decision_of(ep) for ep in episodes]
    holdout = [CF.decision_of(ep) for ep in holdout_eps]
    if not decisions:
        raise ValueError("no valid training decisions")

    tokenizer, model, source = LS.load(
        args.weights, args.device,
        None if args.init == "none" else args.init)
    device = torch.device(args.device)
    lengths = LS.length_report(tokenizer, decisions)
    print(f"[listwise] max sequence {lengths['max_tokens']}/"
          f"{lengths['max_length']}", flush=True)

    groups = param_groups(model, args.lr_encoder, args.lr_head)
    optimizer = torch.optim.AdamW(
        groups, weight_decay=WEIGHT_DECAY, foreach=False)
    total_steps = max(1, math.ceil(args.budget / args.decisions_per_batch))
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_factor(step, total_steps, args.warmup, args.lr_floor))
    scorer = LS.ListwisePairScorer(
        tokenizer, model, device, f"{TASK}:{out.name}",
        batch_size=args.eval_batch_size)
    checkpoint = {
        "kind": "checkpoint",
        "dir": _rel(out / "model"),
        "run": out.name,
        "init": source,
        "model_version": scorer.id,
        "weights": args.weights,
    }
    curve_path = out / "curve.jsonl"
    curve_path.write_text("")
    rng = random.Random(args.seed)
    consumed = steps = epoch = 0
    consumed_ids: list[str] = []
    next_eval = args.eval_every
    loss_sum = 0.0
    loss_steps = 0
    curve = []
    t0 = time.perf_counter()
    eval_seconds = 0.0
    best = {"forced": None, "decisions_consumed": None, "steps": None,
            "tag": None}
    stale_evals = 0

    def do_eval(tag: str):
        nonlocal loss_sum, loss_steps, eval_seconds, stale_evals
        started = time.perf_counter()
        doc = CF.evaluate_battery(
            scorer, args.battery,
            checkpoint={**checkpoint, "decisions_consumed": consumed},
            device=args.device)
        figures = doc["figures"]
        row = {
            "tag": tag,
            "decisions_consumed": consumed,
            "steps": steps,
            "epoch": epoch,
            "train_loss": (round(loss_sum / loss_steps, 6)
                           if loss_steps else None),
            "forced": figures["forced"]["accuracy"],
            "forced_ci95": figures["forced"]["accuracy_ci95"],
            "counterfactual_joint":
                figures["counterfactual_joint"]["accuracy"],
            "by_family": {
                family: node["accuracy"]
                for family, node in
                figures["macro_by_family"]["by_family"].items()
            },
            "by_k": {
                key: node["accuracy"]
                for key, node in figures["by_k"].items()
            },
        }
        if holdout:
            measured = evaluate_holdout(
                model, tokenizer, holdout, device, args.eval_batch_size)
            row["holdout_forced"] = measured["accuracy"]
            row["holdout_ci95"] = measured["ci95"]
            row["holdout_n"] = measured["n"]
        improved = best["forced"] is None or row["forced"] > best["forced"]
        row["retained"] = improved
        if improved:
            best.update({"forced": row["forced"],
                         "decisions_consumed": consumed,
                         "steps": steps, "tag": tag})
            LS.save(model, tokenizer, out / "model", args.weights)
            _dump(out / "eval-dev.json", doc)
            stale_evals = 0
        else:
            stale_evals += 1
        with curve_path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        curve.append(row)
        loss_sum = 0.0
        loss_steps = 0
        eval_seconds += time.perf_counter() - started
        kept = "kept" if improved else f"discarded, best {best['forced']:.4f}"
        print(f"[listwise] eval@{consumed} {CF.figure_line(doc)} [{kept}]",
              flush=True)
        return doc

    while consumed < args.budget and epoch < args.epochs:
        epoch += 1
        order = list(range(len(decisions)))
        rng.shuffle(order)
        model.train()
        for start in range(0, len(order), args.decisions_per_batch):
            if consumed >= args.budget:
                break
            take = order[start:start + args.decisions_per_batch]
            take = take[:args.budget - consumed]
            chunk = [OF.shuffled_candidates(decisions[i], rng) for i in take]
            consumed_ids.extend(str(decisions[i].meta.get("id")) for i in take)
            scores, _ = LS.forward_decisions(
                model, tokenizer, chunk, device, canonical_order=False)
            positives = CF.positive_mask(chunk, scores.shape[1])
            loss = CF.listwise_loss(scores, positives)
            if args.pairwise_weight:
                loss = loss + args.pairwise_weight * CF.pairwise_loss(
                    scores, positives)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            consumed += len(chunk)
            steps += 1
            loss_sum += float(loss.detach())
            loss_steps += 1
            if steps % 10 == 0:
                print(f"[listwise] step {steps} consumed {consumed} "
                      f"loss {loss_sum / loss_steps:.4f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
            if consumed >= next_eval and consumed < args.budget:
                do_eval("scheduled")
                next_eval += args.eval_every
                if stale_evals >= args.patience:
                    break
                model.train()
        if stale_evals >= args.patience:
            print(f"[listwise] stopping at {consumed}: {stale_evals} evals "
                  f"without beating {best['forced']:.4f}", flush=True)
            break

    stopped_early = stale_evals >= args.patience
    final = do_eval("final")
    consumed_counts = Counter(consumed_ids)
    manifest = {
        "task": TASK,
        "format": LS.FORMAT,
        "status": "measured",
        "run": out.name,
        "generated_utc": utcnow(),
        "command": command_line(args),
        "weights": args.weights,
        "init": source,
        "device": args.device,
        "seed": args.seed,
        "decisions_consumed": consumed,
        "steps": steps,
        "epochs_run": epoch,
        "train_seconds": round(time.perf_counter() - t0 - eval_seconds, 2),
        "eval_seconds": round(eval_seconds, 2),
        "hyperparams": {
            "lr_encoder": args.lr_encoder,
            "lr_head": args.lr_head,
            "lr_schedule": (f"warmup {args.warmup} of {total_steps} steps, "
                            f"linear decay to {args.lr_floor} of peak"),
            "base_lrs": base_lrs,
            "decisions_per_batch": args.decisions_per_batch,
            "eval_batch_size": args.eval_batch_size,
            "budget": args.budget,
            "patience": args.patience,
            "loss": ("listwise cross-entropy over marker logits"
                     + (f" + {args.pairwise_weight} * pairwise logistic, "
                        "gold against each negative, mean per decision"
                        if args.pairwise_weight else "")),
            "pairwise_weight": args.pairwise_weight,
            "candidate_shuffle": "every pass; canonical text order in eval",
        },
        "retained": {
            "forced": best["forced"],
            "decisions_consumed": best["decisions_consumed"],
            "steps": best["steps"],
            "tag": best["tag"],
            "rule": "the checkpoint on disk is the best eval of this run, "
                    "not the last one",
        },
        "stopped_early": stopped_early,
        "length_report": lengths,
        "episodes": {k: v for k, v in loaded.items() if k != "episodes"},
        "split": split,
        "consumed": CF.consumption_manifest(episodes),
        "consumed_ids": {
            "file": "consumed_ids.json",
            "n": len(consumed_ids),
            "n_unique": len(set(consumed_ids)),
            "sha256": hashlib.sha256(
                "\n".join(consumed_ids).encode()).hexdigest(),
            "max_exposures_in_run": max(consumed_counts.values(), default=0),
        },
        "holdout": (CF.consumption_manifest(holdout_eps)
                    if holdout_eps else None),
        "final": {
            "forced": final["figures"]["forced"]["accuracy"],
            "counterfactual_joint":
                final["figures"]["counterfactual_joint"]["accuracy"],
        },
    }
    _dump(out / "consumed_ids.json", {"ids": consumed_ids})
    _dump(out / "train_manifest.json", manifest)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--episodes", required=True)
    train_parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    train_parser.add_argument("--init", default="none")
    train_parser.add_argument("--out", required=True)
    train_parser.add_argument("--device", default="mps")
    train_parser.add_argument("--seed", type=int, default=20260929)
    train_parser.add_argument("--epochs", type=int, default=4)
    train_parser.add_argument("--lr-encoder", type=float, default=2e-5)
    train_parser.add_argument("--lr-head", type=float, default=1e-4)
    train_parser.add_argument("--decisions-per-batch", type=int, default=1)
    train_parser.add_argument("--eval-batch-size", type=int, default=2)
    train_parser.add_argument("--eval-every", type=int, default=2500)
    train_parser.add_argument("--budget", type=int, default=5000)
    train_parser.add_argument("--patience", type=int, default=3,
                              help="stop after N evals with no new best")
    train_parser.add_argument("--warmup", type=float, default=0.1)
    train_parser.add_argument("--lr-floor", type=float, default=0.1)
    train_parser.add_argument("--pairwise-weight", type=float, default=0.0,
                              help="weight of the auxiliary pairwise loss")
    train_parser.add_argument("--battery", default=str(DEFAULT_BATTERY))
    train_parser.add_argument("--holdout", type=float, default=0.2)
    train_parser.add_argument("--split-seed", type=int, default=20260927)

    eval_parser = commands.add_parser("eval")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    eval_parser.add_argument("--device", default="mps")
    eval_parser.add_argument("--battery", default=str(DEFAULT_BATTERY))
    eval_parser.add_argument("--out", required=True)
    eval_parser.add_argument("--decisions-per-batch", type=int, default=2)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        train(args)
    else:
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
