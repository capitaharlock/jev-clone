"""Pointer decision head (#T-pointer-head).

The core of the 2026-09-21 audit correction. `train_baseline.py::load()`
learned `state -> clf.classes_`: a fixed, global label space in which the
option set of a row never entered the model. That is why ReClor scored
0.254 (exact chance with 4 options) — its options are positional ids
whose *text* the model never saw.

Here the output space of a row IS that row's option set:

* `DecisionEngine.encode_state` runs the `model.encoder` backbone over the
  state ONCE per row; every question and every option of that row reads
  the same memory `H_s`.
* 1-3 cross-attention blocks let each (question, option) query attend over
  `H_s` and over the other options of the same question (set attention,
  no positional signal — permuting options permutes the outputs and
  nothing else).
* The pointer scorer produces ONE logit per option as a content-based
  dot product between a context query and a key derived from the
  **option's text embedding**. There is no `num_labels x d` matrix
  anywhere: `label_free_report()` asserts it structurally.
* `unknown` is an explicit learned logit computed from a
  permutation-invariant summary of the row, not a post-hoc threshold on
  the max probability.

Consequences: K varies per row with no padding to a fixed maximum, and an
option whose text never appeared in training is scored exactly like any
other — there is no `clf.classes_` left to look a label up in.

CLI:
    .venv-train/bin/python -m model.decision_head gate
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass

import torch
from torch import nn

from .encoder import (DEFAULT_BACKBONE, DEFAULT_MAX_LENGTH, Backbone,
                      encode_state, load_backbone)

UNKNOWN_ID = "unknown"
DEFAULT_D_MODEL = 256
DEFAULT_LAYERS = 2
DEFAULT_HEADS = 8
DEFAULT_SEED = 1789

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", "T-pointer-head")


@dataclass
class StateMemory:
    """`H_s` — one state encoded once, reused by every question/option."""

    text: str
    tokens: torch.Tensor      # [1, T, H_backbone]
    mask: torch.Tensor        # [1, T] bool, True = real token
    pooled: torch.Tensor      # [1, H_backbone]
    n_tokens: int
    ms: float


class CrossBlock(nn.Module):
    """One cross-attention layer: options read `H_s`, then read each other.

    Pre-norm residual. The option axis carries NO positional encoding, so
    the block is permutation-equivariant over options by construction.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 set_attention: bool = True):
        super().__init__()
        self.ln_cross = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout,
                                           batch_first=True)
        # `set_attention=False` is the INDEPENDENT-SCORING ablation
        # (#T-fullspace-objective): no attention among candidates, so a
        # logit depends only on (state, question, this option's text) and
        # the block stops being quadratic in K. It is a CHANGE OF
        # ARCHITECTURE, not the same head with another denominator, and
        # nothing on the default path touches it: True reproduces the
        # published head parameter for parameter.
        self.set_attention = set_attention
        self.ln_set = nn.LayerNorm(d_model) if set_attention else None
        self.set_attn = (nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout,
                                               batch_first=True)
                         if set_attention else None)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                pad_mask: torch.Tensor | None,
                set_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln_cross(x)
        attended, _ = self.cross(h, memory, memory,
                                 key_padding_mask=pad_mask,
                                 need_weights=False)
        x = x + attended
        if self.set_attention:
            h = self.ln_set(x)
            # `set_pad_mask` only exists in the batched path, where rows
            # with fewer options are padded up to K_max: a real option must
            # not read a padding slot, or the batched forward would stop
            # agreeing with the per-row one.
            mixed, _ = self.set_attn(h, h, h, key_padding_mask=set_pad_mask,
                                     need_weights=False)
            x = x + mixed
        return x + self.ff(self.ln_ff(x))


class PointerDecisionHead(nn.Module):
    """Scores a variable-length option set against a state memory.

    Every parameter is shaped by `d_state`, `d_model` or 1 — never by a
    label count and never by K. The forward takes the K option text
    embeddings as they come and returns K+1 logits (K options + one
    `unknown`).
    """

    def __init__(self, d_state: int, d_model: int = DEFAULT_D_MODEL,
                 n_layers: int = DEFAULT_LAYERS,
                 n_heads: int = DEFAULT_HEADS, dropout: float = 0.0,
                 set_attention: bool = True):
        super().__init__()
        if not 1 <= n_layers <= 3:
            raise ValueError("n_layers must be 1..3 (plan: 1-3 layers)")
        self.d_state = d_state
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.set_attention = set_attention
        self.state_proj = nn.Linear(d_state, d_model)
        self.state_ln = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_state, d_model)
        self.opt_proj = nn.Linear(d_state, d_model)
        self.blocks = nn.ModuleList(
            [CrossBlock(d_model, n_heads, dropout, set_attention)
             for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_model)
        # Pointer: query from the contextualised option, key from the
        # option's OWN text embedding. Output width is 1 per option.
        self.ptr_query = nn.Linear(d_model, d_model)
        self.ptr_key = nn.Linear(d_model, d_model, bias=False)
        self.ptr_bias = nn.Parameter(torch.zeros(1))
        # `unknown`: a learned logit over a permutation-invariant summary.
        self.unknown_ln = nn.LayerNorm(d_model)
        self.unknown = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.scale = d_model ** -0.5

    def forward(self, memory: torch.Tensor, memory_mask: torch.Tensor,
                question_emb: torch.Tensor,
                option_embs: torch.Tensor) -> torch.Tensor:
        """`[K + 1]` logits; the last entry is `unknown`.

        memory        [1, T, d_state]  state token states (encoded once)
        memory_mask   [1, T] bool      True = real token
        question_emb  [d_state]        pooled question text
        option_embs   [K, d_state]     pooled option TEXTS, K arbitrary
        """
        if option_embs.dim() != 2:
            raise ValueError("option_embs must be [K, d_state]")
        k = option_embs.shape[0]
        if k == 0:
            raise ValueError("a question needs at least one option")
        mem = self.state_ln(self.state_proj(memory))          # [1, T, d]
        pad = ~memory_mask.bool()                             # True = pad
        opt = self.opt_proj(option_embs)                      # [K, d]
        q = self.q_proj(question_emb).view(1, 1, -1)          # [1, 1, d]
        x = opt.unsqueeze(0) + q                              # [1, K, d]
        for block in self.blocks:
            x = block(x, mem, pad)
        ctx = self.ln_out(x)[0]                               # [K, d]
        keys = self.ptr_key(opt)                              # [K, d] text
        scores = (self.ptr_query(ctx) * keys).sum(-1) * self.scale
        scores = scores + self.ptr_bias
        # Permutation-invariant row summary: masked mean over H_s, mean
        # over option contexts, plus the question.
        m = memory_mask.unsqueeze(-1).to(mem.dtype)
        mem_pooled = (mem * m).sum(1) / m.sum(1).clamp(min=1e-6)
        summary = mem_pooled[0] + ctx.mean(0) + q.view(-1)
        unk = self.unknown(self.unknown_ln(summary))          # [1]
        return torch.cat([scores, unk.view(1)], dim=0)        # [K + 1]

    def forward_batch(self, memory: torch.Tensor,
                      memory_mask: torch.Tensor,
                      question_emb: torch.Tensor,
                      option_embs: torch.Tensor,
                      option_mask: torch.Tensor) -> torch.Tensor:
        """`[B, K_max + 1]` logits for a whole batch in ONE kernel chain.

        Same maths as `forward`, one row at a time replaced by a padded
        option axis. `test_decision_head` asserts the two agree row by
        row to float tolerance; this one exists purely because MPS pays a
        dispatch per launch, so scoring 64 rows in a Python loop spends
        most of a training step queueing tiny kernels instead of
        computing (#T-metal-throughput).

        memory        [B, T, d_state]  state token states
        memory_mask   [B, T] bool      True = real token
        question_emb  [B, d_state]     pooled question texts
        option_embs   [B, K_max, d_state]  pooled option TEXTS, zero-padded
        option_mask   [B, K_max] bool  True = real option

        Column `K_max` is `unknown`; the padded option columns come back
        as `-inf`, so a softmax gives them exactly zero mass and a
        cross-entropy against a real gold index is unaffected.
        """
        if option_embs.dim() != 3:
            raise ValueError("option_embs must be [B, K_max, d_state]")
        if option_mask.shape != option_embs.shape[:2]:
            raise ValueError("option_mask must be [B, K_max]")
        counts = option_mask.sum(1)
        if int(counts.min()) == 0:
            raise ValueError("a question needs at least one option")
        mem = self.state_ln(self.state_proj(memory))           # [B, T, d]
        pad = ~memory_mask.bool()
        opt_pad = ~option_mask.bool()
        opt = self.opt_proj(option_embs)                       # [B, K, d]
        q = self.q_proj(question_emb).unsqueeze(1)             # [B, 1, d]
        x = opt + q
        for block in self.blocks:
            x = block(x, mem, pad, opt_pad)
        ctx = self.ln_out(x)                                   # [B, K, d]
        keys = self.ptr_key(opt)                               # [B, K, d]
        scores = (self.ptr_query(ctx) * keys).sum(-1) * self.scale
        scores = scores + self.ptr_bias
        m = memory_mask.unsqueeze(-1).to(mem.dtype)
        mem_pooled = (mem * m).sum(1) / m.sum(1).clamp(min=1e-6)
        om = option_mask.unsqueeze(-1).to(ctx.dtype)
        ctx_mean = (ctx * om).sum(1) / om.sum(1).clamp(min=1e-6)
        summary = mem_pooled + ctx_mean + q.squeeze(1)         # [B, d]
        unk = self.unknown(self.unknown_ln(summary))           # [B, 1]
        scores = scores.masked_fill(opt_pad, float("-inf"))
        return torch.cat([scores, unk], dim=1)                 # [B, K + 1]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def label_free_report(self) -> dict:
        """Structural proof that no parameter lives in a label space.

        Every parameter dimension must be `d_state`, `d_model`, a fixed
        multiple of `d_model` (the FFN), or 1. Anything else would mean a
        `num_labels`-shaped weight sneaked in.
        """
        allowed = {1, self.d_state, self.d_model, 2 * self.d_model,
                   3 * self.d_model, 4 * self.d_model}
        offenders = {name: list(p.shape)
                     for name, p in self.named_parameters()
                     if any(d not in allowed for d in p.shape)}
        return {"label_free": not offenders, "offenders": offenders,
                "allowed_dims": sorted(allowed),
                "n_params": self.n_params()}


class DecisionEngine:
    """Backbone + head, with encode-once caching for states and texts."""

    def __init__(self, backbone: Backbone | None = None,
                 head: PointerDecisionHead | None = None,
                 backbone_id: str = DEFAULT_BACKBONE,
                 device: str | torch.device = "cpu",
                 d_model: int = DEFAULT_D_MODEL,
                 n_layers: int = DEFAULT_LAYERS,
                 n_heads: int = DEFAULT_HEADS,
                 seed: int = DEFAULT_SEED,
                 max_length: int = DEFAULT_MAX_LENGTH):
        self.backbone = backbone or load_backbone(backbone_id, device)
        self.device = self.backbone.device
        self.max_length = max_length
        if head is None:
            torch.manual_seed(seed)
            head = PointerDecisionHead(self.backbone.hidden_size, d_model,
                                       n_layers, n_heads)
        self.head = head.to(self.device)
        self._states: dict[str, StateMemory] = {}
        self._texts: dict[str, torch.Tensor] = {}

    # -- encoding -------------------------------------------------------
    def encode_state(self, state: str) -> StateMemory:
        """One real forward over the state, memoised per row."""
        cached = self._states.get(state)
        if cached is not None:
            return cached
        # Tokenised here only for the attention mask; the forward pass
        # itself is `model.encoder.encode_state` — one code path.
        enc = self.backbone.tokenizer(state, return_tensors="pt",
                                      truncation=True,
                                      max_length=self.max_length)
        out = encode_state(self.backbone, state, self.max_length)
        # `encoder.encode_state` runs under inference_mode; clone out of
        # it so the head can be trained on top of a frozen backbone.
        mem = StateMemory(text=state, tokens=out["tokens"].clone(),
                          mask=enc["attention_mask"].to(self.device).bool(),
                          pooled=out["pooled"].clone(),
                          n_tokens=out["n_tokens"], ms=out["ms"])
        self._states[state] = mem
        return mem

    @torch.inference_mode()
    def _embed_batch(self, texts: list[str]) -> torch.Tensor:
        enc = self.backbone.tokenizer(texts, return_tensors="pt",
                                      padding=True, truncation=True,
                                      max_length=self.max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.backbone.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

    def _embed_batch_grad(self, texts: list[str]) -> torch.Tensor:
        """`_embed_batch` with the graph kept (#T-unfreeze-backbone).

        The same pooled forward, outside `inference_mode`, so a backbone
        whose parameters are being trained receives gradient from the
        option texts too — not only from the state tokens.
        """
        enc = self.backbone.tokenizer(texts, return_tensors="pt",
                                      padding=True, truncation=True,
                                      max_length=self.max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.backbone.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

    def embed_texts(self, texts: list[str],
                    grad: bool = False) -> torch.Tensor:
        """Pooled text embeddings [n, H], memoised per exact string.

        The cache is what makes option scoring cheap in production (label
        texts repeat across rows) and it is also why permuting an option
        list re-uses bit-identical inputs.

        `grad=True` bypasses the cache entirely: with a trainable backbone
        a memoised embedding is both stale (the encoder moved since it was
        stored) and detached (it carries no graph). Duplicates are still
        encoded once per call and gathered, so the saving that matters —
        the repeated label texts inside one batch — survives.
        """
        if grad:
            uniq = list(dict.fromkeys(texts))
            embs = self._embed_batch_grad(uniq)
            at = {t: i for i, t in enumerate(uniq)}
            return torch.stack([embs[at[t]] for t in texts])
        missing = [t for t in dict.fromkeys(texts) if t not in self._texts]
        if missing:
            embs = self._embed_batch(missing)
            for text, emb in zip(missing, embs):
                self._texts[text] = emb.detach().clone()
        return torch.stack([self._texts[t] for t in texts])

    def clear_caches(self) -> None:
        """Drop the state/text memos — mandatory once the backbone moves."""
        self._states.clear()
        self._texts.clear()

    # -- scoring --------------------------------------------------------
    def score(self, state: str | StateMemory, question: str,
              options: list[dict] | list[str]) -> dict:
        """Distribution over THIS row's options plus `unknown`.

        `options` is a list of `{id, text}` dicts (V1 schema) or plain
        strings. K is whatever the list holds — nothing is padded.
        """
        mem = state if isinstance(state, StateMemory) \
            else self.encode_state(state)
        ids, texts = _option_fields(options)
        q_emb = self.embed_texts([question])[0]
        opt_embs = self.embed_texts(texts)
        with torch.no_grad():  # the differentiable path is `.logits()`
            logits = self.head(mem.tokens, mem.mask, q_emb, opt_embs)
            probs = torch.softmax(logits, dim=-1)
        order = sorted(range(len(ids)), key=lambda i: -float(probs[i]))
        best = int(torch.argmax(probs).item())
        return {
            "option_ids": ids,
            "logits": [float(x) for x in logits[:-1]],
            "probs": [float(x) for x in probs[:-1]],
            "unknown_logit": float(logits[-1]),
            "unknown_prob": float(probs[-1]),
            "argmax_id": UNKNOWN_ID if best == len(ids) else ids[best],
            "ranking": [ids[i] for i in order],
            "k": len(ids),
        }

    def logits(self, mem: StateMemory, question: str,
               options: list[dict] | list[str]) -> torch.Tensor:
        """Differentiable path (head only; embeddings come cached)."""
        _, texts = _option_fields(options)
        q_emb = self.embed_texts([question])[0]
        opt_embs = self.embed_texts(texts)
        return self.head(mem.tokens, mem.mask, q_emb, opt_embs)

    def predict_row(self, row: dict) -> list[dict]:
        """A whole V1 row: the state is encoded once for all questions."""
        mem = self.encode_state(row["state"])
        return [self.score(mem, q.get("text") or q.get("id", ""),
                           q["options"]) for q in row.get("questions", [])]


def _option_fields(options: list[dict] | list[str]) -> tuple[list, list]:
    ids, texts = [], []
    for i, opt in enumerate(options):
        if isinstance(opt, str):
            ids.append(f"o{i}")
            texts.append(opt)
        else:
            texts.append(opt["text"])
            ids.append(opt.get("id", f"o{i}"))
    if not texts:
        raise ValueError("a question needs at least one option")
    return ids, texts


# -- checks the gate and the tests share ---------------------------------

def kl_divergence(p: list[float], q: list[float]) -> float:
    return sum(a * math.log(max(a, 1e-12) / max(b, 1e-12))
               for a, b in zip(p, q))


def full_distribution(result: dict) -> list[float]:
    return list(result["probs"]) + [result["unknown_prob"]]


def check_order_invariance(engine: DecisionEngine, state: str,
                           question: str, options: list[dict],
                           n_perms: int = 20,
                           seed: int = DEFAULT_SEED) -> dict:
    """20 permutations -> same argmax id, KL below tolerance."""
    import random

    mem = engine.encode_state(state)
    base = engine.score(mem, question, options)
    base_dist = full_distribution(base)
    base_by_id = dict(zip(base["option_ids"], base["probs"]))
    rng = random.Random(seed)
    worst_kl, argmax_ok = 0.0, True
    for _ in range(n_perms):
        shuffled = options[:]
        rng.shuffle(shuffled)
        got = engine.score(mem, question, shuffled)
        argmax_ok = argmax_ok and got["argmax_id"] == base["argmax_id"]
        # Re-align the shuffled distribution onto the base ordering.
        by_id = dict(zip(got["option_ids"], got["probs"]))
        aligned = [by_id[i] for i in base["option_ids"]]
        aligned.append(got["unknown_prob"])
        worst_kl = max(worst_kl, abs(kl_divergence(base_dist, aligned)))
    return {"n_perms": n_perms, "argmax_stable": argmax_ok,
            "max_kl": worst_kl, "base_argmax": base["argmax_id"],
            "base_probs": base_by_id}


def check_variable_k(engine: DecisionEngine, state: str, question: str,
                     pool: list[dict], ks: tuple[int, ...] = (2, 4, 9)
                     ) -> dict:
    """The same row at several K, no recompile, no padding."""
    mem = engine.encode_state(state)
    per_k = {}
    for k in ks:
        if k > len(pool):
            raise ValueError(f"pool has {len(pool)} options, need {k}")
        res = engine.score(mem, question, pool[:k])
        dist = full_distribution(res)
        per_k[k] = {
            "k": res["k"],
            "n_logits": len(res["probs"]) + 1,
            "sums_to_one": abs(sum(dist) - 1.0) < 1e-5,
            "finite": all(math.isfinite(x) for x in dist),
            "argmax_id": res["argmax_id"],
        }
    ok = all(v["k"] == k and v["n_logits"] == k + 1 and v["sums_to_one"]
             and v["finite"] for k, v in per_k.items())
    return {"ok": ok, "per_k": per_k}


def check_unseen_label(engine: DecisionEngine, state: str, question: str,
                       options: list[dict], unseen_text: str) -> dict:
    """An option text absent from training is finite and orderable."""
    mem = engine.encode_state(state)
    unseen = {"id": "unseen", "text": unseen_text}
    res = engine.score(mem, question, options + [unseen])
    idx = res["option_ids"].index("unseen")
    logit = res["logits"][idx]
    distinct = len({round(x, 9) for x in res["logits"]}) == len(res["logits"])
    return {"ok": math.isfinite(logit) and distinct
            and "unseen" in res["ranking"],
            "unseen_logit": logit, "unseen_prob": res["probs"][idx],
            "rank": res["ranking"].index("unseen"),
            "orderable": distinct, "k": res["k"]}


def toy_unknown_fit(engine: DecisionEngine, steps: int = 150,
                    lr: float = 1e-3, seed: int = DEFAULT_SEED) -> dict:
    """Head-only fit proving `unknown` is a LEARNED logit.

    Eight toy rows: four whose state contains the answer, four whose
    state does not and whose gold is `unknown`. The backbone stays
    frozen; only the head moves. This demonstrates the MECHANISM —
    gradient reaches the `unknown` parameters and the logit can win a
    row — it does NOT measure abstention quality on real data. That is
    #T-train-real's job (see the model card).
    """
    rows = _toy_rows()
    for row in rows:  # warm the caches once, outside the loop
        engine.encode_state(row["state"])
        engine.embed_texts([row["question"]] +
                           [o["text"] for o in row["options"]])
    torch.manual_seed(seed)
    opt = torch.optim.Adam(engine.head.parameters(), lr=lr)
    engine.head.train()
    first_loss, last_loss, first_grad = None, None, 0.0
    for _ in range(steps):
        opt.zero_grad()
        total = torch.zeros((), device=engine.device)
        for row in rows:
            mem = engine.encode_state(row["state"])
            logits = engine.logits(mem, row["question"], row["options"])
            gold = (len(row["options"]) if row["gold"] == UNKNOWN_ID
                    else [o["id"] for o in row["options"]].index(row["gold"]))
            total = total + nn.functional.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([gold], device=engine.device))
        loss = total / len(rows)
        loss.backward()
        grad = engine.head.unknown[0].weight.grad
        unknown_grad = float(grad.abs().sum()) if grad is not None else 0.0
        opt.step()
        if first_loss is None:
            first_loss, first_grad = float(loss.detach()), unknown_grad
        last_loss = float(loss.detach())
    engine.head.eval()
    with torch.no_grad():
        verdicts = []
        for row in rows:
            mem = engine.encode_state(row["state"])
            res = engine.score(mem, row["question"], row["options"])
            verdicts.append({"id": row["id"], "gold": row["gold"],
                             "argmax": res["argmax_id"],
                             "unknown_prob": res["unknown_prob"]})
    unanswerable = [v for v in verdicts if v["gold"] == UNKNOWN_ID]
    answerable = [v for v in verdicts if v["gold"] != UNKNOWN_ID]
    return {
        "steps": steps, "first_loss": first_loss, "last_loss": last_loss,
        "loss_decreased": last_loss < first_loss,
        "unknown_grad_step0": first_grad,
        "unknown_wins_unanswerable": all(v["argmax"] == UNKNOWN_ID
                                         for v in unanswerable),
        "answerable_not_abstained": all(v["argmax"] == v["gold"]
                                        for v in answerable),
        "verdicts": verdicts,
        "scope": ("toy head-only fit: mechanism only, abstention quality "
                  "is measured by #T-train-real"),
    }


def _toy_rows() -> list[dict]:
    """Four answerable rows + four whose state omits the answer."""
    palette = [("the harbor gate", "open", "closed"),
               ("the night ledger", "balanced", "short"),
               ("the orbit path", "stable", "drifting"),
               ("the signal lamp", "green", "red")]
    rows = []
    for i, (subject, yes, no) in enumerate(palette):
        options = [{"id": "a", "text": f"{subject} is {yes}"},
                   {"id": "b", "text": f"{subject} is {no}"},
                   {"id": "c", "text": f"{subject} was never inspected"}]
        rows.append({"id": f"ans-{i}", "gold": "a",
                     "state": f"inspection notice: {subject} is {yes} "
                              f"as of this morning.",
                     "question": f"what is the state of {subject}?",
                     "options": options})
        rows.append({"id": f"unk-{i}", "gold": UNKNOWN_ID,
                     "state": "inspection notice: the canteen menu "
                              "changed and the lift is serviced monthly.",
                     "question": f"what is the state of {subject}?",
                     "options": options})
    return rows


def measure_latency(engine: DecisionEngine, n: int = 30, k: int = 4,
                    warmup: int = 3) -> dict:
    """Measured p50/p95 per row at K=4 — cold state, cold option cache."""
    def one(tag: str) -> float:
        state = (f"shift log {tag}: the harbor gate is open, the ledger "
                 f"is balanced and the lamp reads green at berth {tag}.")
        question = f"what does the log say about berth {tag}?"
        options = [{"id": f"o{j}", "text": f"berth {tag} note {j}: "
                    f"the gate is {'open' if j == 0 else 'closed'}"}
                   for j in range(k)]
        t0 = time.perf_counter()
        mem = engine.encode_state(state)
        engine.score(mem, question, options)
        if engine.device.type == "mps":
            torch.mps.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0
        # Never let the caches make the next run a lie.
        engine._states.clear()
        engine._texts.clear()
        return elapsed

    for w in range(warmup):
        one(f"w{w}")
    samples = sorted(one(f"r{i}") for i in range(n))
    return {"runs": n, "k": k, "device": str(engine.device),
            "p50_ms": round(samples[len(samples) // 2], 3),
            "p95_ms": round(samples[max(0, int(len(samples) * 0.95) - 1)], 3),
            "mean_ms": round(sum(samples) / len(samples), 3),
            "cache": "cleared between runs (cold state + cold options)"}


def run_gate(device: str = "cpu", write: bool = True) -> dict:
    """Run the four contract checks + latency and write gate.json."""
    engine = DecisionEngine(device=device)
    state = ("shift log: the harbor gate is open, the night ledger is "
             "balanced, and the signal lamp reads green.")
    question = "what is the state of the harbor gate?"
    pool = [{"id": f"o{i}", "text": t} for i, t in enumerate([
        "the harbor gate is open", "the harbor gate is closed",
        "the harbor gate is jammed", "the night ledger is short",
        "the signal lamp reads red", "the orbit path is drifting",
        "the meadow path is quiet", "the engine runs loud",
        "the harbor gate was never inspected"])]

    invariance = check_order_invariance(engine, state, question, pool[:4])
    variable_k = check_variable_k(engine, state, question, pool)
    unseen = check_unseen_label(
        engine, state, question, pool[:4],
        "the bathyscaphe manifest was filed under seal in Reykjavik")
    unknown = toy_unknown_fit(engine)

    fresh = DecisionEngine(device=device)  # untrained head for latency
    latency = measure_latency(fresh, k=4)
    report = fresh.head.label_free_report()

    checks = {
        "order_invariance": {
            "pass": invariance["argmax_stable"] and invariance["max_kl"] < 1e-4,
            "max_kl": invariance["max_kl"], "tolerance": 1e-4,
            "n_perms": invariance["n_perms"],
            "argmax_stable": invariance["argmax_stable"]},
        "variable_k": {"pass": variable_k["ok"], "ks": [2, 4, 9],
                       "detail": variable_k["per_k"]},
        "unseen_label": {"pass": unseen["ok"],
                         "unseen_logit": unseen["unseen_logit"],
                         "rank": unseen["rank"]},
        "unknown_logit": {
            "pass": unknown["loss_decreased"]
                    and unknown["unknown_grad_step0"] > 0.0
                    and unknown["unknown_wins_unanswerable"],
            "first_loss": unknown["first_loss"],
            "last_loss": unknown["last_loss"],
            "unknown_grad_step0": unknown["unknown_grad_step0"],
            "unknown_wins_unanswerable":
                unknown["unknown_wins_unanswerable"],
            "answerable_not_abstained":
                unknown["answerable_not_abstained"],
            "limitation": unknown["scope"]},
        "label_free": {"pass": report["label_free"],
                       "offenders": report["offenders"]},
    }
    gate = {
        "task": "T-pointer-head",
        "pass": all(c["pass"] for c in checks.values()),
        "head_params": report["n_params"],
        "backbone": {"id": fresh.backbone.id,
                     "hidden_size": fresh.backbone.hidden_size,
                     "params": fresh.backbone.params},
        "architecture": {"d_model": fresh.head.d_model,
                         "n_layers": fresh.head.n_layers,
                         "n_heads": fresh.head.n_heads,
                         "scorer": "pointer dot-product over option text "
                                   "embeddings; no num_labels matrix",
                         "unknown": "learned logit over a "
                                    "permutation-invariant row summary"},
        "latency_k4": latency,
        "checks": checks,
        "torch": torch.__version__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if write:
        os.makedirs(GATE_DIR, exist_ok=True)
        with open(os.path.join(GATE_DIR, "gate.json"), "w") as fh:
            json.dump(gate, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return gate


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "gate"
    if cmd != "gate":
        print(f"unknown command {cmd!r}; use gate", file=sys.stderr)
        return 2
    device = argv[2] if len(argv) > 2 else "cpu"
    gate = run_gate(device=device)
    print(json.dumps({k: v for k, v in gate.items() if k != "checks"},
                     indent=2, sort_keys=True))
    for name, check in gate["checks"].items():
        print(f"[gate] {name}: {'PASS' if check['pass'] else 'FAIL'}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
