"""Read and write a checkpoint folder: model.safetensors plus config.json."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .backbone import DecisionModel, ModelConfig

WEIGHTS = "model.safetensors"
CONFIG = "config.json"


def load_config(folder: Path) -> ModelConfig:
    return ModelConfig(**json.loads((Path(folder) / CONFIG).read_text(
        encoding="utf-8")))


def load_weights(model: DecisionModel, path: Path) -> None:
    """Load with strict key matching; a shorter position table may grow.

    Growing the context keeps every trained position and leaves the new
    ones at their initialization, so a model trained at 1024 bytes can
    continue at 1536.
    """
    saved = load_file(str(path))
    current = model.state_dict()
    key = "position_embed.weight"
    if key in saved and saved[key].shape != current[key].shape:
        old, new = saved[key], current[key]
        if old.shape[1] != new.shape[1]:
            raise RuntimeError("position width cannot change")
        if old.shape[0] > new.shape[0]:
            raise RuntimeError("checkpoint context exceeds the new model")
        grown = new.clone()
        grown[:old.shape[0]] = old
        saved[key] = grown
    model.load_state_dict(saved, strict=True)


def load_model(folder: Path, device: str | torch.device = "cpu",
               max_length: int | None = None) -> DecisionModel:
    """An eval-mode model from a checkpoint folder, dropout off."""
    config = replace(load_config(folder), dropout=0.0)
    if max_length and max_length > config.max_length:
        config = replace(config, max_length=max_length)
    model = DecisionModel(config)
    load_weights(model, Path(folder) / WEIGHTS)
    return model.to(device).eval()


def save(model: DecisionModel, folder: Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tensors = {name: value.detach().cpu().contiguous()
               for name, value in model.state_dict().items()}
    save_file(tensors, str(folder / WEIGHTS))
    (folder / CONFIG).write_text(
        json.dumps(asdict(model.config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
