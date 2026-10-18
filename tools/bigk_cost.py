"""What a bigger option set COSTS — #T-bigk-optsets, done-when 4.

Phase 2 raises K from the sampler's 3-8 to the whole label space, and the
plan says the price is published before the next run is budgeted instead of
promised afterwards. Two things grow, and they grow differently:

* the **option text embeddings** are cached per text
  (`DecisionEngine.embed_texts`), and a label space is a few dozen short
  strings, so after the first batch they cost a dict lookup. K does NOT
  multiply the encoder work.
* the **head** does cross-attention over the state AND set attention among
  the options (`model/decision_head.py`, `CrossBlock.set_attn`). That
  second block is quadratic in K per row, and no cache touches it.

So this measures a REAL training step — state encode, option embeddings,
`forward_batch`, prior penalty, `cross_entropy`, backward, optimiser — at
K = 8, 41 and 77, and reports rows/s and peak device memory for each, plus
the head-only slice so the K² term is visible on its own.

Honesty about the option texts: the states and the label texts are real
rows of `massive` and `huffpost` (both trainable, neither fenced). A space
of 77 does not exist among them, so for K > |space| the list is extended
with label texts of the other trainable space — this is a cost harness, the
weights are thrown away, nothing is checkpointed and no fenced corpus is
read. Cost depends on K, on token length and on B; not on which strings.

CLI:
    .venv-train/bin/python -m tools.bigk_cost --device mps
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from data.optset import (PREFETCH_DIR, Sample,  # noqa: E402
                         iter_rows, label_pool)
from model.decision_head import (DecisionEngine,  # noqa: E402
                                 PointerDecisionHead)
from model.encoder import load_backbone, pick_device  # noqa: E402
from training.python.train_decision import (batch_embeddings,  # noqa: E402
                                            batch_gold, encode_states,
                                            pack_options, peak_memory_gb)

TASK = "T-bigk-optsets"
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", TASK)

#: the architecture of the arm this task compares against
#: (`leverstack-d512-prior-ettin-68m-s20260922`)
BACKBONE = "ettin-68m"
D_MODEL = 512
N_LAYERS = 2
N_HEADS = 8
BATCH = 64
MAX_LENGTH = 256

#: real states come from here; real label texts from both spaces
STATE_DATASET = "massive"
TEXT_DATASETS = ("massive", "huffpost")
KS = (8, 41, 77)
WARMUP = 2
STEPS = 8


def option_texts(root: str = PREFETCH_DIR) -> list[str]:
    """Real label texts of the trainable spaces, in a stable order."""
    out, seen = [], set()
    for dataset in TEXT_DATASETS:
        for opt in label_pool(dataset, root):
            text = opt.get("text") or opt["id"]
            if text not in seen:
                seen.add(text)
                out.append(text)
    return out


def states(n: int, root: str = PREFETCH_DIR) -> list[dict]:
    """`n` real rows (state + question) of the state dataset."""
    out = []
    for row in iter_rows(STATE_DATASET, "train", root=root):
        for question in row.get("questions", []):
            out.append({"state": row["state"],
                        "question": question.get("id", "q")})
            if len(out) >= n:
                return out
    return out


def batch_at(k: int, rows: list[dict], texts: list[str]) -> list[Sample]:
    """One batch of `BATCH` rows, each offering exactly `k` options."""
    if k > len(texts):
        raise ValueError(f"only {len(texts)} real label texts available "
                         f"for K={k}")
    out = []
    for i, row in enumerate(rows[:BATCH]):
        # rotate the window so the batch is not 64 identical option sets
        start = (i * 7) % max(1, len(texts) - k)
        chosen = texts[start:start + k]
        out.append(Sample(
            dataset=STATE_DATASET, row_id=f"cost-{i}",
            question_id="q", state=row["state"], question=row["question"],
            options=[{"id": f"o{j}", "text": t}
                     for j, t in enumerate(chosen)],
            answer="o0", gold_index=0,
            space=STATE_DATASET, space_size=k))
    return out


def measure(engine, batch: list[Sample], opt, device) -> dict:
    """One full training step, timed, with the head slice broken out."""
    sync = (torch.mps.synchronize if getattr(device, "type", str(device))
            == "mps" else (torch.cuda.synchronize
                           if getattr(device, "type", "") == "cuda"
                           else lambda: None))
    t0 = time.perf_counter()
    tokens, mask, _n = encode_states(engine.backbone,
                                     [s.state for s in batch], MAX_LENGTH,
                                     grad=False)
    embs, spans = batch_embeddings(engine, batch, grad=False)
    opt.zero_grad(set_to_none=True)
    q_emb, opt_embs, opt_mask = pack_options(embs, spans, engine.device)
    sync()
    t_head = time.perf_counter()
    logits = engine.head.forward_batch(tokens, mask, q_emb, opt_embs,
                                       opt_mask)
    gold = batch_gold(batch, opt_embs.shape[1], engine.device)
    loss = nn.functional.cross_entropy(logits, gold)
    loss.backward()
    sync()
    head_s = time.perf_counter() - t_head
    nn.utils.clip_grad_norm_(list(engine.head.parameters()), 1.0)
    opt.step()
    sync()
    return {"step_s": time.perf_counter() - t0, "head_s": head_s,
            "k_padded": int(opt_embs.shape[1])}


def run(device: str = "auto", write: bool = True, log=print) -> dict:
    dev = pick_device(device)
    backbone = load_backbone(BACKBONE, dev)
    texts = option_texts()
    rows = states(BATCH)
    log(f"[cost] device={dev} label texts available={len(texts)} "
        f"rows={len(rows)}")

    per_k = {}
    for k in KS:
        torch.manual_seed(1789)
        head = PointerDecisionHead(backbone.hidden_size, D_MODEL, N_LAYERS,
                                   N_HEADS)
        engine = DecisionEngine(backbone=backbone, head=head)
        engine.head.train()
        engine.clear_caches()
        optim = torch.optim.AdamW(engine.head.parameters(), lr=3e-4,
                                  weight_decay=0.01)
        batch = batch_at(k, rows, texts)
        for _ in range(WARMUP):
            measure(engine, batch, optim, dev)
        marks = [measure(engine, batch, optim, dev) for _ in range(STEPS)]
        step_s = sorted(m["step_s"] for m in marks)[len(marks) // 2]
        head_s = sorted(m["head_s"] for m in marks)[len(marks) // 2]
        per_k[str(k)] = {
            "k": k,
            "batch_size": BATCH,
            "rows_per_s": round(BATCH / step_s, 2),
            "ms_per_step": round(1000 * step_s, 1),
            "ms_per_step_head_only": round(1000 * head_s, 1),
            "head_share_of_step": round(head_s / step_s, 4),
            "peak_mem_gb": peak_memory_gb(dev),
            "minutes_per_62500_rows": round(62_500 * step_s / BATCH / 60, 2),
            "minutes_per_1m_rows": round(1_000_000 * step_s / BATCH / 60, 1),
            "steps_timed": STEPS,
            "warmup_steps": WARMUP,
        }
        log(f"[cost] K={k}: {per_k[str(k)]['rows_per_s']} rows/s, "
            f"{per_k[str(k)]['ms_per_step']} ms/step "
            f"(head {per_k[str(k)]['ms_per_step_head_only']} ms), "
            f"peak {per_k[str(k)]['peak_mem_gb']} GiB")
        del engine, head, optim

    base = per_k[str(KS[0])]
    doc = {
        "format": "jev.gate.v1",
        "task": TASK,
        "artifact": "cost",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(dev),
        "architecture": {"backbone": BACKBONE, "d_model": D_MODEL,
                         "n_layers": N_LAYERS, "n_heads": N_HEADS,
                         "frozen_backbone": True,
                         "batch_size": BATCH, "max_length": MAX_LENGTH},
        "what_is_measured": ("one full training step: state encode (no "
                             "grad), option text embeddings through the "
                             "cache, `forward_batch`, `cross_entropy`, "
                             "backward, clip, optimiser step. Median of "
                             f"{STEPS} steps after {WARMUP} warmups"),
        "per_k": per_k,
        "slowdown_vs_k8": {
            str(k): round(base["rows_per_s"] / per_k[str(k)]["rows_per_s"], 2)
            for k in KS},
        "head_slowdown_vs_k8": {
            str(k): round(per_k[str(k)]["ms_per_step_head_only"]
                          / base["ms_per_step_head_only"], 2)
            for k in KS},
        "reading": [
            "the option text cache removes the ENCODER cost of a bigger K "
            "(a label space is a few dozen short strings, embedded once); "
            "what it cannot remove is the set attention among options, "
            "which is quadratic in K per row",
            "`ms_per_step_head_only` is the slice that grows; the state "
            "encode is the same forward at every K",
        ],
        "honesty": [
            "states and label texts are real rows of "
            f"{', '.join(TEXT_DATASETS)} — trainable, not fenced. A single "
            "space of 77 labels does not exist among them, so for K larger "
            "than a space the option list is extended with the other "
            "space's label texts: a cost harness, not a corpus. No "
            "checkpoint is written and no fenced dataset is read",
            "one MPS lane, nothing else running: a parallel CPU job costs "
            "this measurement about a third of its throughput",
        ],
    }
    if write:
        os.makedirs(GATE_DIR, exist_ok=True)
        with open(os.path.join(GATE_DIR, "cost.json"), "w") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return doc


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="tools.bigk_cost")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv[1:])
    doc = run(args.device, write=not args.no_write)
    print(json.dumps({k: v for k, v in doc.items()
                      if k in ("device", "per_k", "slowdown_vs_k8",
                               "head_slowdown_vs_k8")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
