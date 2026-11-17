"""Turn episodes into tensors, and score them."""
from __future__ import annotations

import random

import torch

from .backbone import DecisionModel


def group_by_options(rows: list[dict], size: int,
                     seed: int | None = None) -> list[list[dict]]:
    """Batches whose rows share an option count, so no padding is needed.

    With a seed the batches are shuffled; without one the order is stable,
    which keeps evaluation deterministic.
    """
    pools: dict[int, list[dict]] = {}
    for row in rows:
        pools.setdefault(len(row["candidates"]), []).append(row)
    rng = random.Random(seed) if seed is not None else None
    chunks = []
    for width in sorted(pools):
        pool = pools[width]
        if rng:
            rng.shuffle(pool)
        chunks.extend(pool[start:start + size]
                      for start in range(0, len(pool), size))
    if rng:
        rng.shuffle(chunks)
    return chunks


def batch_tensors(model: DecisionModel, rows: list[dict],
                  device: torch.device) -> tuple[tuple[torch.Tensor, ...],
                                                 torch.Tensor]:
    """Model inputs and gold indices for rows with the same option count.

    A row with no `answer` gets gold -1; such rows can be scored but not
    trained on.
    """
    width = len(rows[0]["candidates"])
    if any(len(row["candidates"]) != width for row in rows):
        raise ValueError("a batch must share its option count")
    limit = model.config.max_length
    tokenizer = model.tokenizer
    state = tokenizer.batch([row["state"] for row in rows], limit, device)
    question = tokenizer.batch([row["question"] for row in rows], limit, device)
    flat = [c["text"] for row in rows for c in row["candidates"]]
    option_ids, option_tokens = tokenizer.batch(flat, limit, device)
    batch = len(rows)
    golds = []
    for row in rows:
        ids = [c["id"] for c in row["candidates"]]
        answer = row.get("answer")
        golds.append(width if answer == "unknown"
                     else ids.index(answer) if answer in ids else -1)
    inputs = (
        *state,
        *question,
        option_ids.view(batch, width, -1),
        option_tokens.view(batch, width, -1),
        torch.ones((batch, width), dtype=torch.bool, device=device),
    )
    return inputs, torch.tensor(golds, device=device)


@torch.inference_mode()
def predict(model: DecisionModel, rows: list[dict], device: torch.device,
            batch_size: int = 32) -> list[dict]:
    """Per row: the forced choice and a probability per option id.

    Forced means the unknown column is dropped and the softmax runs over
    the offered options only. Results come back in input order.
    """
    model.eval()
    index = {id(row): position for position, row in enumerate(rows)}
    out: list[dict | None] = [None] * len(rows)
    for chunk in group_by_options(rows, batch_size):
        inputs, _ = batch_tensors(model, chunk, device)
        logits = model(*inputs)[:, :-1].float()
        probs = torch.softmax(logits, dim=-1).cpu()
        for row, values in zip(chunk, probs):
            ids = [c["id"] for c in row["candidates"]]
            best = int(values.argmax())
            out[index[id(row)]] = {
                "choice": ids[best],
                "probabilities": {i: round(float(p), 6)
                                  for i, p in zip(ids, values)},
            }
    return out
