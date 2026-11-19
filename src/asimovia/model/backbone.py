"""Byte-level encoder plus pointer head, trained from random initialization.

A decision is one packed sequence:

    state  SEP  question  SEP option_1  SEP option_2  ...  SEP option_K

The encoder reads the whole sequence once, so every option is seen next to
the question and the state. The head then scores each option span against
the rest. No pretrained checkpoint and no fitted tokenizer enter the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .head import PointerDecisionHead


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 260
    max_length: int = 1536
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    pointer_layers: int = 2
    dropout: float = 0.1
    question_slot: int = 108
    option_slot: int = 191


class ByteTokenizer:
    """UTF-8 bytes 0..255 plus four control ids. Nothing to fit or download."""

    PAD = 256
    BOS = 257
    EOS = 258
    SEP = 259
    VOCAB_SIZE = 260

    def encode(self, text: str, max_length: int) -> list[int]:
        if max_length < 2:
            raise ValueError("max_length must leave room for BOS and EOS")
        body = list(text.encode("utf-8"))[: max_length - 2]
        return [self.BOS, *body, self.EOS]

    def batch(
        self, texts: list[str], max_length: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [self.encode(text, max_length) for text in texts]
        width = max(len(row) for row in rows)
        ids = torch.full((len(rows), width), self.PAD, dtype=torch.long, device=device)
        mask = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = torch.tensor(row, device=device)
            mask[index, : len(row)] = True
        return ids, mask


class DecisionModel(nn.Module):
    """Shared byte encoder plus a content pointer over dynamic options."""

    format = "asimovia-decision-v1"

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        config = config or ModelConfig()
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
            layer, num_layers=config.n_layers, enable_nested_tensor=False
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
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def budgets(limit: int, options: int) -> tuple[int, int, int]:
        """Byte budgets for question, state and each option.

        The question can carry the constraint of the decision, so its cap
        holds a full sentence. Options get everything left over: they are
        what the pointer compares, and a product sheet is long.
        """
        question_limit = min(160, max(8, limit // 8))
        separators = 1 + max(options, 1)
        state_limit = min(256, max(16, limit // 8))
        option_room = limit - separators - question_limit - state_limit
        if option_room < 4 * max(options, 1):
            state_limit = max(8, limit - separators - question_limit - 4 * max(options, 1))
            option_room = limit - separators - question_limit - state_limit
        option_limit = max(1, option_room // max(options, 1))
        return question_limit, state_limit, option_limit

    def _pack(
        self,
        state_ids: torch.Tensor,
        state_mask: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        option_ids: torch.Tensor,
        option_token_mask: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Lay every decision out in fixed-width slots.

        The state slot is always `state_limit` wide. The question slot and
        each option slot are `question_slot` and `option_slot` wide, and
        grow only when this row's own text needs more room, up to its
        budget. Unused slot positions are masked out. Where an option
        starts therefore depends on the row alone, never on the rest of
        the batch, so one request and the same request inside a batch get
        the same logits.
        """
        batch, options, _ = option_ids.shape
        question_limit, state_limit, option_limit = self.budgets(self.config.max_length, options)
        device = state_ids.device
        rows = []
        for b in range(batch):
            # The state keeps its tail: in a running dialogue the newest
            # turn is the one the decision is about.
            state = state_ids[b, state_mask[b]][-state_limit:]
            question = question_ids[b, question_mask[b]][:question_limit]
            texts = [
                option_ids[b, k, option_token_mask[b, k]][:option_limit]
                if option_mask[b, k]
                else option_ids[b, k, :0]
                for k in range(options)
            ]
            question_width = min(question_limit, max(self.config.question_slot, question.numel()))
            option_width = min(
                option_limit, max([self.config.option_slot] + [t.numel() for t in texts])
            )
            pieces = [
                ("state", state, state_limit),
                ("sep", None, 1),
                ("question", question, question_width),
            ]
            for k, text in enumerate(texts):
                pieces += [("sep", None, 1), (k, text, option_width)]
            rows.append(pieces)
        width = max(sum(w for _, _, w in pieces) for pieces in rows)
        ids = torch.full((batch, width), self.tokenizer.PAD, dtype=torch.long, device=device)
        mask = torch.zeros((batch, width), dtype=torch.bool, device=device)
        state_span = torch.zeros_like(mask)
        question_span = torch.zeros_like(mask)
        option_span = torch.zeros((batch, options, width), dtype=torch.bool, device=device)
        for b, pieces in enumerate(rows):
            cursor = 0
            for name, tokens, slot in pieces:
                if name == "sep":
                    ids[b, cursor] = self.tokenizer.SEP
                    mask[b, cursor] = True
                else:
                    end = cursor + tokens.numel()
                    ids[b, cursor:end] = tokens
                    mask[b, cursor:end] = True
                    span = (
                        state_span[b]
                        if name == "state"
                        else question_span[b]
                        if name == "question"
                        else option_span[b, name]
                    )
                    span[cursor:end] = True
                cursor += slot
        return ids, mask, state_span, question_span, option_span

    @staticmethod
    def _pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)

    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if ids.shape != mask.shape:
            raise ValueError("ids and mask must have the same shape")
        if ids.shape[1] > self.config.max_length:
            raise ValueError("sequence exceeds max_length")
        positions = torch.arange(ids.shape[1], device=ids.device)
        hidden = self.token_embed(ids) + self.position_embed(positions)[None]
        hidden = self.encoder(hidden, src_key_padding_mask=~mask.bool())
        return self.encoder_norm(hidden)

    def forward(
        self,
        state_ids: torch.Tensor,
        state_mask: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        option_ids: torch.Tensor,
        option_token_mask: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return `[B, K + 1]` logits; the final column is unknown."""
        if option_ids.dim() != 3:
            raise ValueError("option_ids must be [B, K, T]")
        batch, options, _width = option_ids.shape
        if option_mask.shape != (batch, options):
            raise ValueError("option_mask must be [B, K]")
        ids, mask, state_span, question_span, option_span = self._pack(
            state_ids,
            state_mask,
            question_ids,
            question_mask,
            option_ids,
            option_token_mask,
            option_mask,
        )
        hidden = self.encode(ids, mask)
        questions = self._pool(hidden, question_span)
        weights = option_span.unsqueeze(-1).to(hidden.dtype)
        option_vectors = (hidden[:, None] * weights).sum(2) / weights.sum(2).clamp(min=1)
        return self.pointer.forward_batch(
            hidden, state_span, questions, option_vectors, option_mask
        )

    def lineage(self) -> dict:
        return {
            "format": self.format,
            "config": asdict(self.config),
            "initialization": "random",
            "external_checkpoint": False,
            "tokenizer": "utf-8-bytes",
            "parameters": sum(p.numel() for p in self.parameters()),
        }
