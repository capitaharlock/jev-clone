"""Decision encoder whose published weights start from random initialization.

External models may label training rows, but no external checkpoint enters
this module. The byte vocabulary also avoids a borrowed tokenizer model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .decision_head import PointerDecisionHead


@dataclass(frozen=True)
class OwnModelConfig:
    vocab_size: int = 260
    max_length: int = 1024
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    pointer_layers: int = 2
    dropout: float = 0.1


class ByteTokenizer:
    """Deterministic UTF-8 tokenizer with no fitted or downloaded weights."""

    PAD = 256
    BOS = 257
    EOS = 258
    SEP = 259
    VOCAB_SIZE = 260

    def encode(self, text: str, max_length: int) -> list[int]:
        if max_length < 2:
            raise ValueError("max_length must leave room for BOS and EOS")
        body = list(text.encode("utf-8"))[:max_length - 2]
        return [self.BOS, *body, self.EOS]

    def batch(self, texts: list[str], max_length: int,
              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [self.encode(text, max_length) for text in texts]
        width = max(len(row) for row in rows)
        ids = torch.full((len(rows), width), self.PAD, dtype=torch.long,
                         device=device)
        mask = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
        for index, row in enumerate(rows):
            ids[index, :len(row)] = torch.tensor(row, device=device)
            mask[index, :len(row)] = True
        return ids, mask


class OwnDecisionModel(nn.Module):
    """Shared byte encoder plus a content pointer over dynamic options."""

    format = "asimovia-decision-v1"

    def __init__(self, config: OwnModelConfig = OwnModelConfig()):
        super().__init__()
        if config.vocab_size != ByteTokenizer.VOCAB_SIZE:
            raise ValueError("vocab_size must match the byte tokenizer")
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.config = config
        self.tokenizer = ByteTokenizer()
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embed = nn.Embedding(config.max_length, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.n_layers,
            enable_nested_tensor=False,
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.pointer = PointerDecisionHead(
            d_state=config.d_model,
            d_model=config.d_model,
            n_layers=config.pointer_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every publishable parameter without a checkpoint."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, ids: torch.Tensor,
               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if ids.shape != mask.shape:
            raise ValueError("ids and mask must have the same shape")
        if ids.shape[1] > self.config.max_length:
            raise ValueError("sequence exceeds max_length")
        positions = torch.arange(ids.shape[1], device=ids.device)
        hidden = self.token_embed(ids) + self.position_embed(positions)[None]
        hidden = self.encoder(hidden, src_key_padding_mask=~mask.bool())
        hidden = self.encoder_norm(hidden)
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)
        return hidden, pooled

    def forward(self, state_ids: torch.Tensor, state_mask: torch.Tensor,
                question_ids: torch.Tensor, question_mask: torch.Tensor,
                option_ids: torch.Tensor, option_token_mask: torch.Tensor,
                option_mask: torch.Tensor) -> torch.Tensor:
        """Return `[B, K + 1]` logits; the final column is unknown."""
        if option_ids.dim() != 3:
            raise ValueError("option_ids must be [B, K, T]")
        batch, options, width = option_ids.shape
        if option_mask.shape != (batch, options):
            raise ValueError("option_mask must be [B, K]")
        memory, _ = self.encode(state_ids, state_mask)
        _, questions = self.encode(question_ids, question_mask)
        flat_ids = option_ids.reshape(batch * options, width)
        flat_mask = option_token_mask.reshape(batch * options, width)
        _, flat_options = self.encode(flat_ids, flat_mask)
        option_vectors = flat_options.reshape(batch, options, -1)
        return self.pointer.forward_batch(
            memory, state_mask, questions, option_vectors, option_mask)

    def lineage(self) -> dict:
        return {
            "format": self.format,
            "config": asdict(self.config),
            "initialization": "random",
            "external_checkpoint": False,
            "tokenizer": "utf-8-bytes",
            "parameters": sum(p.numel() for p in self.parameters()),
        }
