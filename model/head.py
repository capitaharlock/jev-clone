"""Pointer head: one logit per option, plus a learned "unknown" logit.

The output space of a row is that row's own option set. No weight is shaped
by a label count or by K, so a row may carry any number of options and an
option text never seen in training is scored like any other. The head adds
no signal along the option axis; any order effect comes from the encoder's
positions, and training shuffles the options to wash it out.
"""
from __future__ import annotations

import torch
from torch import nn


class CrossBlock(nn.Module):
    """Options read the state tokens, then read each other (pre-norm)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln_cross = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.ln_set = nn.LayerNorm(d_model)
        self.set_attn = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout,
                                              batch_first=True)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                memory_pad: torch.Tensor,
                option_pad: torch.Tensor) -> torch.Tensor:
        h = self.ln_cross(x)
        attended, _ = self.cross(h, memory, memory,
                                 key_padding_mask=memory_pad,
                                 need_weights=False)
        x = x + attended
        h = self.ln_set(x)
        # A real option must never read a padding slot, or a batch with
        # mixed option counts would disagree with scoring row by row.
        mixed, _ = self.set_attn(h, h, h, key_padding_mask=option_pad,
                                 need_weights=False)
        x = x + mixed
        return x + self.ff(self.ln_ff(x))


class PointerDecisionHead(nn.Module):
    """Scores a variable-length option set against encoded state tokens."""

    def __init__(self, d_state: int, d_model: int, n_layers: int = 2,
                 n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        if not 1 <= n_layers <= 3:
            raise ValueError("n_layers must be between 1 and 3")
        self.state_proj = nn.Linear(d_state, d_model)
        self.state_ln = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_state, d_model)
        self.opt_proj = nn.Linear(d_state, d_model)
        self.blocks = nn.ModuleList(
            [CrossBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_model)
        # The key comes from the option's own text and the query from the
        # option in context, so the score is content against context.
        self.ptr_query = nn.Linear(d_model, d_model)
        self.ptr_key = nn.Linear(d_model, d_model, bias=False)
        self.ptr_bias = nn.Parameter(torch.zeros(1))
        self.unknown_ln = nn.LayerNorm(d_model)
        self.unknown = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.scale = d_model ** -0.5

    def forward_batch(self, memory: torch.Tensor, memory_mask: torch.Tensor,
                      question_emb: torch.Tensor, option_embs: torch.Tensor,
                      option_mask: torch.Tensor) -> torch.Tensor:
        """Return `[B, K + 1]` logits; the last column is unknown.

        memory        [B, T, d_state]  encoded state tokens
        memory_mask   [B, T]           True for a real token
        question_emb  [B, d_state]     pooled question span
        option_embs   [B, K, d_state]  pooled option spans
        option_mask   [B, K]           True for a real option

        Padded option columns come back as -inf, so a softmax gives them
        zero mass and a cross-entropy on a real gold index ignores them.
        """
        if option_embs.dim() != 3:
            raise ValueError("option_embs must be [B, K, d_state]")
        if option_mask.shape != option_embs.shape[:2]:
            raise ValueError("option_mask must be [B, K]")
        if int(option_mask.sum(1).min()) == 0:
            raise ValueError("every row needs at least one option")
        mem = self.state_ln(self.state_proj(memory))
        memory_pad = ~memory_mask.bool()
        option_pad = ~option_mask.bool()
        opt = self.opt_proj(option_embs)
        q = self.q_proj(question_emb).unsqueeze(1)
        x = opt + q
        for block in self.blocks:
            x = block(x, mem, memory_pad, option_pad)
        ctx = self.ln_out(x)
        scores = (self.ptr_query(ctx) * self.ptr_key(opt)).sum(-1) * self.scale
        scores = scores + self.ptr_bias
        m = memory_mask.unsqueeze(-1).to(mem.dtype)
        mem_pooled = (mem * m).sum(1) / m.sum(1).clamp(min=1e-6)
        om = option_mask.unsqueeze(-1).to(ctx.dtype)
        ctx_mean = (ctx * om).sum(1) / om.sum(1).clamp(min=1e-6)
        summary = mem_pooled + ctx_mean + q.squeeze(1)
        unknown = self.unknown(self.unknown_ln(summary))
        scores = scores.masked_fill(option_pad, float("-inf"))
        return torch.cat([scores, unknown], dim=1)
