"""One-sequence decision scorer with one marker per option.

Format:
    [CLS] premise [SEP] [MASK] option_1 [MASK] option_2 ... [SEP]

The encoder sees every option in one pass. A shared scalar head reads each
marker state, producing one logit per option. Evaluation sorts option text
before encoding and maps logits back to the caller order, so candidate order
cannot carry signal.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from model import ce_scorer as CE

FORMAT = "jev.listwise.v1"
MAX_LENGTH = 512
HEAD_FILE = "listwise_head.safetensors"
META_FILE = "listwise.json"


@dataclass
class EncodedDecision:
    input_ids: list[int]
    marker_positions: list[int]
    encoded_to_input: list[int]
    truncated: bool


def _special_id(tokenizer, primary: str, fallback: str) -> int:
    value = getattr(tokenizer, primary, None)
    if value is None:
        value = getattr(tokenizer, fallback, None)
    if value is None:
        raise ValueError(f"tokenizer has neither {primary} nor {fallback}")
    return int(value)


def encode_texts(tokenizer, question: str, options: list[str], *,
                 state: str = "",
                 max_length: int = MAX_LENGTH,
                 canonical_order: bool = False) -> EncodedDecision:
    """Encode one decision and preserve a map back to its input option order."""
    if len(options) < 2:
        raise ValueError("a listwise decision needs at least two options")
    cls_id = _special_id(tokenizer, "cls_token_id", "bos_token_id")
    sep_id = _special_id(tokenizer, "sep_token_id", "eos_token_id")
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        raise ValueError("the listwise scorer needs a mask token marker")

    order = list(range(len(options)))
    if canonical_order:
        order.sort(key=lambda i: options[i])
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    option_ids = [
        tokenizer.encode(" " + options[i], add_special_tokens=False)
        for i in order
    ]
    state_ids = tokenizer.encode(state, add_special_tokens=False)
    option_tokens = sum(1 + len(tokens) for tokens in option_ids)
    fixed = 1 + len(question_ids) + 1 + option_tokens + 1 + 1
    if fixed > max_length:
        raise ValueError(
            f"question and complete options need {fixed} tokens; "
            f"limit is {max_length}")
    state_budget = max_length - fixed
    state_was_cut = len(state_ids) > state_budget
    state_ids = state_ids[:state_budget]

    ids = [cls_id] + question_ids + [sep_id]
    markers = []
    for encoded in option_ids:
        markers.append(len(ids))
        ids.append(int(mask_id))
        ids.extend(encoded)
    ids.extend([sep_id] + state_ids + [sep_id])
    return EncodedDecision(ids, markers, order, state_was_cut)


def encode_decision(tokenizer, decision: CE.Decision, *,
                    max_length: int = MAX_LENGTH,
                    canonical_order: bool = False) -> EncodedDecision:
    return encode_texts(
        tokenizer,
        decision.question,
        [candidate.text for candidate in decision.candidates],
        state=decision.state,
        max_length=max_length,
        canonical_order=canonical_order,
    )


class ListwiseModel:
    """Thin owner for a transformer encoder and a scalar marker head."""

    def __init__(self, encoder, hidden_size: int):
        import torch

        self.encoder = encoder
        self.head = torch.nn.Linear(int(hidden_size), 1)

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.head.parameters())

    def named_parameters(self):
        for name, value in self.encoder.named_parameters():
            yield f"encoder.{name}", value
        for name, value in self.head.named_parameters():
            yield f"head.{name}", value

    def train(self):
        self.encoder.train()
        self.head.train()

    def eval(self):
        self.encoder.eval()
        self.head.eval()

    @property
    def training(self) -> bool:
        return bool(self.encoder.training)

    def to(self, device):
        self.encoder.to(device)
        self.head.to(device)
        return self


def load(weights: str, device: str, checkpoint: str | None = None):
    """Load the pinned encoder or a listwise checkpoint."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    if checkpoint and checkpoint not in ("", "none"):
        root = Path(checkpoint)
        if (root / "model").exists():
            root = root / "model"
        tokenizer = AutoTokenizer.from_pretrained(root / "encoder")
        encoder = AutoModel.from_pretrained(root / "encoder",
                                            dtype=torch.float32)
        meta = json.loads((root / META_FILE).read_text())
        model = ListwiseModel(encoder, int(meta["hidden_size"]))
        model.head.load_state_dict(load_file(root / HEAD_FILE))
        source = {"kind": "checkpoint", "dir": str(root),
                  "format": meta["format"]}
    else:
        from model.weights import require_verified, spec_for

        root = Path(require_verified(weights))
        tokenizer = AutoTokenizer.from_pretrained(root)
        encoder = AutoModel.from_pretrained(root, dtype=torch.float32)
        hidden = int(encoder.config.hidden_size)
        model = ListwiseModel(encoder, hidden)
        spec = spec_for(weights)
        source = {"kind": "weights", "id": weights,
                  "revision": spec["revision"], "format": FORMAT}
    model.to(torch.device(device))
    return tokenizer, model, source


def save(model: ListwiseModel, tokenizer, path: str | Path,
         weights: str) -> None:
    from safetensors.torch import save_file

    root = Path(path)
    encoder_dir = root / "encoder"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(encoder_dir, safe_serialization=True)
    tokenizer.save_pretrained(encoder_dir)
    save_file(model.head.state_dict(), root / HEAD_FILE)
    (root / META_FILE).write_text(json.dumps({
        "format": FORMAT,
        "weights": weights,
        "hidden_size": int(model.encoder.config.hidden_size),
        "marker": tokenizer.mask_token,
        "max_length": MAX_LENGTH,
    }, indent=2) + "\n")


def _padded(tokenizer, encoded: list[EncodedDecision], device):
    import torch

    width = max(len(item.input_ids) for item in encoded)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = _special_id(tokenizer, "sep_token_id", "eos_token_id")
    ids = torch.full((len(encoded), width), int(pad_id), dtype=torch.long)
    attention = torch.zeros((len(encoded), width), dtype=torch.long)
    for row, item in enumerate(encoded):
        n = len(item.input_ids)
        ids[row, :n] = torch.tensor(item.input_ids)
        attention[row, :n] = 1
    return {"input_ids": ids.to(device),
            "attention_mask": attention.to(device)}


def forward_decisions(model: ListwiseModel, tokenizer,
                      decisions: list[CE.Decision], device, *,
                      canonical_order: bool = False,
                      max_length: int = MAX_LENGTH):
    """Return `[B,Kmax]` logits and candidate-order maps."""
    encoded = [
        encode_decision(tokenizer, decision, max_length=max_length,
                        canonical_order=canonical_order)
        for decision in decisions
    ]
    return forward_encoded(model, tokenizer, encoded, device,
                           canonical_order=canonical_order)


def forward_encoded(model: ListwiseModel, tokenizer,
                    encoded: list[EncodedDecision], device, *,
                    canonical_order: bool = False):
    """Run already-rendered listwise sequences through the shared model."""
    import torch

    batch = _padded(tokenizer, encoded, device)
    hidden = model.encoder(**batch).last_hidden_state
    kmax = max(len(item.marker_positions) for item in encoded)
    scores = hidden.new_full((len(encoded), kmax), float("-inf"))
    for row, item in enumerate(encoded):
        positions = torch.tensor(item.marker_positions, device=device)
        states = hidden[row].index_select(0, positions)
        logits = model.head(states).squeeze(-1)
        if canonical_order:
            for encoded_idx, input_idx in enumerate(item.encoded_to_input):
                scores[row, input_idx] = logits[encoded_idx]
        else:
            scores[row, :len(logits)] = logits
    return scores, encoded


def length_report(tokenizer, decisions: list[CE.Decision]) -> dict:
    lengths = []
    state_cuts = 0
    for decision in decisions:
        item = encode_decision(tokenizer, decision)
        lengths.append(len(item.input_ids))
        state_cuts += int(item.truncated)
    return {
        "n_decisions": len(decisions),
        "max_tokens": max(lengths, default=0),
        "max_length": MAX_LENGTH,
        "state_truncated": state_cuts,
        "options_truncated": 0,
        "pass": True,
    }


class ListwisePairScorer(CE.PairScorer):
    """Adapter for the existing evaluation suite's flat pair interface."""

    def __init__(self, tokenizer, model: ListwiseModel, device,
                 model_version: str, batch_size: int = 4):
        import torch

        self.id = model_version
        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.decisions_scored = 0

    @staticmethod
    def _groups(pairs: list[tuple[str, str]]):
        groups: list[tuple[str, list[str]]] = []
        for premise, hypothesis in pairs:
            if not groups or groups[-1][0] != premise:
                groups.append((premise, [hypothesis]))
            else:
                groups[-1][1].append(hypothesis)
        return groups

    @staticmethod
    def _parts(premise: str, hypotheses: list[str]):
        if premise.startswith("State: ") and "\nQuestion: " in premise:
            state, question = premise[len("State: "):].split(
                "\nQuestion: ", 1)
            context = "\n" + CE.CONTEXT_HEADER["en"] + "\n"
            question = question.split(context, 1)[0]
            lang = "en"
        elif premise.startswith("Estado: ") and "\nPregunta: " in premise:
            state, question = premise[len("Estado: "):].split(
                "\nPregunta: ", 1)
            context = "\n" + CE.CONTEXT_HEADER["es"] + "\n"
            question = question.split(context, 1)[0]
            lang = "es"
        else:
            raise ValueError("unknown pair premise format")
        prefix, suffix = CE.HYPOTHESIS[lang].split("{option}")
        options = []
        for hypothesis in hypotheses:
            if not hypothesis.startswith(prefix) or not hypothesis.endswith(suffix):
                raise ValueError("unknown pair hypothesis format")
            options.append(hypothesis[len(prefix):len(hypothesis) - len(suffix)])
        return state, question, options

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        import torch

        groups = self._groups(pairs)
        out: list[float] = []
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(groups), self.batch_size):
                chunk = groups[start:start + self.batch_size]
                encoded = []
                for premise, hypotheses in chunk:
                    state, question, options = self._parts(
                        premise, hypotheses)
                    encoded.append(encode_texts(
                        self.tokenizer, question, options, state=state,
                        canonical_order=True))
                scores, _ = forward_encoded(
                    self.model, self.tokenizer, encoded, self.device,
                    canonical_order=True)
                for row, (_, options) in enumerate(chunk):
                    out.extend(float(x) for x in scores[row, :len(options)])
        if was_training:
            self.model.train()
        self.decisions_scored += len(groups)
        return out

    def describe(self) -> dict:
        return {
            "id": self.id,
            "format": FORMAT,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "max_length": MAX_LENGTH,
            "marker": self.tokenizer.mask_token,
        }


def probabilities(scores, k: int) -> list[float]:
    values = [float(x) for x in scores[:k]]
    hi = max(values)
    exp = [math.exp(x - hi) for x in values]
    total = sum(exp)
    return [x / total for x in exp]
