"""Real neural state encoder (#T-torch-stack).

`encode_state` is the entry point the plan's decision head (#T-pointer-head)
will sit on top of: a V1 schema row's ``state`` goes in, a real backbone
forward pass comes out — token states plus a mask-aware mean pooling —
on MPS when the Mac has it, on CPU otherwise.

Nothing here is a proxy: the weights are the SHA-256-verified bytes under
``artifacts/weights/<id>/`` (see `model.weights`), and loading is refused
outright if a hash does not match.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer

from .weights import BACKBONES, require_verified

DEFAULT_BACKBONE = "ettin-68m"
DEFAULT_MAX_LENGTH = 512


def pick_device(prefer: str = "auto") -> torch.device:
    """`auto` → MPS when this Mac has it, else CPU. Explicit wins."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_report() -> dict:
    return {
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "selected": str(pick_device()),
    }


@dataclass
class Backbone:
    """A loaded encoder pinned to one device."""

    id: str
    device: torch.device
    tokenizer: object
    model: object
    hidden_size: int
    params: int


def load_backbone(backbone_id: str = DEFAULT_BACKBONE,
                  device: str | torch.device = "auto") -> Backbone:
    """Load a hash-verified backbone onto `device` in eval mode."""
    if backbone_id not in BACKBONES:
        raise KeyError(f"unknown backbone {backbone_id!r}; "
                       f"known: {sorted(BACKBONES)}")
    path = require_verified(backbone_id)
    dev = pick_device(device) if isinstance(device, str) else device
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, dtype=torch.float32)
    model.eval().to(dev)
    return Backbone(
        id=backbone_id, device=dev, tokenizer=tokenizer, model=model,
        hidden_size=int(model.config.hidden_size),
        params=sum(p.numel() for p in model.parameters()),
    )


def state_text(row: dict) -> str:
    """The ``state`` field of a V1 schema row, validated as non-empty."""
    state = row.get("state")
    if not isinstance(state, str) or not state.strip():
        raise ValueError("row has no usable V1 `state` field")
    return state


@torch.inference_mode()
def encode_state(backbone: Backbone, state: str,
                 max_length: int = DEFAULT_MAX_LENGTH) -> dict:
    """One real forward pass over a state string.

    Returns ``tokens`` [1, T, H], ``pooled`` [1, H] (mask-aware mean) and
    the wall time of the forward in milliseconds.
    """
    if not isinstance(state, str) or not state.strip():
        raise ValueError("state must be a non-empty string")
    enc = backbone.tokenizer(state, return_tensors="pt", truncation=True,
                             max_length=max_length)
    enc = {k: v.to(backbone.device) for k, v in enc.items()}
    t0 = time.perf_counter()
    out = backbone.model(**enc)
    tokens = out.last_hidden_state
    if backbone.device.type == "mps":
        torch.mps.synchronize()  # the forward is async; time it honestly
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    mask = enc["attention_mask"].unsqueeze(-1).to(tokens.dtype)
    pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    return {"tokens": tokens, "pooled": pooled,
            "n_tokens": int(enc["attention_mask"].sum().item()),
            "ms": elapsed_ms}


def first_row(path: str) -> dict:
    """First JSON line of a converted V1 dataset."""
    import json
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"{path} has no rows")


def prefetch_path(dataset: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "artifacts", "data-prefetch",
                        f"{dataset}.jsonl")
