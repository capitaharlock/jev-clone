"""Tests de la MECÁNICA que el sobreajuste tiene que validar (#T-ce-mechanics).

La cifra de sobreajuste no vale nada si la tubería tiene uno de estos
cuatro fallos, y ninguno de los cuatro da error por su cuenta: el modelo
simplemente aprende otra cosa, o aprende a medias, y la pérdida baja igual.
Así que se miden aquí, con un modelo de JUGUETE de dos capas y un
tokenizador de juguete —nada de pesos reales, nada de GPU— porque lo que
se prueba es el cableado, no el checkpoint:

* **alineación del gold** — el gold se resuelve por id del candidato
  contra la misma tupla que se acaba de renderizar, así que permutar los
  candidatos no puede moverlo; y un gold leído por POSICIÓN se movería
  (el test lo comprueba, para que no pase por falta de teeth);
* **truncado del estado** — un par que no cabe en la ventana para el run
  en vez de recortar el estado en silencio, y cuando se recorta a
  propósito el recorte se come la PREMISA, nunca la hipótesis;
* **máscara de candidatos** — una fila de K=3 puntúa igual sola que
  mezclada con filas de K=4, y el relleno se queda a probabilidad
  exactamente 0;
* **formato entreno/eval** — los strings que tokeniza el entreno son los
  MISMOS bytes que los que construye `eval/ce_nograd.py` para las mismas
  decisiones, medidos desde los dos caminos de verdad, no por inspección.

Y dos más que el gate pide: que el formato de hipótesis esté congelado en
un solo sitio importable, y que el artefacto del árbol no lleve cifras
que nadie ha medido.
"""
from __future__ import annotations

import dataclasses
import math
import random
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers

from eval import ce_nograd as NG
from model import ce_scorer as CE
from model import weights as W
from training.python import ce_overfit as OF

ROOT = Path(__file__).resolve().parents[2]

#: vocabulario del tokenizador de juguete; 0 es el relleno
VOCAB = 1024
CLS, SEP, PAD = 1, 2, 0
TOL = 1e-9


class ToyTokenizer:
    """La superficie del tokenizador de HF que el árbol usa, y nada más.

    Parte las palabras y las mapea a ids por CRC32 —estable entre
    procesos, al contrario que `hash()`— y reproduce lo que
    `model.ce_scorer.encode_pairs` necesita: llamada por pares,
    `truncation`, `max_length`, `padding` y `return_tensors="pt"`. Registra
    cada llamada, que es lo que permite comparar QUÉ strings y con QUÉ
    recorte ha tokenizado cada camino.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def _ids(self, text: str) -> list[int]:
        return [3 + zlib.crc32(w.encode("utf-8")) % (VOCAB - 3)
                for w in text.split()]

    def __call__(self, premises, hypotheses=None, return_tensors=None,
                 padding=False, truncation=False, max_length=None):
        premises = [premises] if isinstance(premises, str) else list(premises)
        if hypotheses is None:
            hypotheses = [""] * len(premises)
        hypotheses = ([hypotheses] if isinstance(hypotheses, str)
                      else list(hypotheses))
        rows = []
        for premise, hypothesis in zip(premises, hypotheses):
            a, b = self._ids(premise), self._ids(hypothesis)
            if truncation and max_length:
                room = max_length - 3 - len(b)
                if truncation == "only_first":
                    a = a[:max(room, 0)]          # la hipótesis no se toca
                else:
                    while len(a) + len(b) + 3 > max_length and (a or b):
                        (a if len(a) >= len(b) else b).pop()
            rows.append([CLS] + a + [SEP] + b + [SEP])
        self.calls.append({
            "premises": premises, "hypotheses": hypotheses,
            "truncation": truncation, "max_length": max_length,
            "padding": padding, "return_tensors": return_tensors})
        if return_tensors != "pt":
            return {"input_ids": rows}
        width = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), width), PAD, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
            attention[i, :len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention}

    def pairs_tokenised(self) -> list[tuple[str, str]]:
        """Los pares que de verdad han entrado al modelo, en orden.

        Sólo las llamadas con `return_tensors="pt"`: las otras son las
        mediciones de longitud de `length_report`, que no tokenizan para
        puntuar.
        """
        out = []
        for call in self.calls:
            if call["return_tensors"] != "pt":
                continue
            out.extend(zip(call["premises"], call["hypotheses"]))
        return out


class ToyNli(torch.nn.Module):
    """Dos capas y tres clases: lo mínimo para que un gradiente llegue.

    El pooling es la media sobre los tokens PRESENTES, así que el relleno
    de un lote no puede mover la puntuación de una fila corta: si el test
    de la máscara falla, es de la máscara y no del modelo.
    """

    def __init__(self, seed: int = 7, dim: int = 16):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = torch.nn.Embedding(VOCAB, dim, padding_idx=PAD)
        self.hidden = torch.nn.Linear(dim, dim)
        self.out = torch.nn.Linear(dim, 3)
        self.config = SimpleNamespace(
            num_labels=3,
            id2label={0: "entailment", 1: "neutral", 2: "contradiction"})

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        mask = (attention_mask if attention_mask is not None
                else torch.ones_like(input_ids)).unsqueeze(-1).to(
                    self.emb.weight.dtype)
        pooled = ((self.emb(input_ids) * mask).sum(1)
                  / mask.sum(1).clamp(min=1.0))
        return SimpleNamespace(logits=self.out(torch.tanh(self.hidden(pooled))))


@pytest.fixture
def toy():
    """`(modelo, tokenizador, índice de entailment)` sobre CPU."""
    model = ToyNli()
    return model, ToyTokenizer(), CE._entailment_index(model.config)


@pytest.fixture
def decisions():
    return OF.overfit_decisions()


def _sample(decisions, n_k3=3, n_k4=2):
    k3 = [d for d in decisions if d.k == 3][:n_k3]
    k4 = [d for d in decisions if d.k == 4][:n_k4]
    return k3, k4


# -- el conjunto y su gold ----------------------------------------------

def test_the_set_is_in_range_and_its_golds_come_from_the_rule(decisions):
    info = OF.validate_set(decisions)
    assert 32 <= info["n"] <= 64
    assert info["golds_recomputed_from_text"] == 36
    assert info["golds_hand_written"] == 12
    assert (info["golds_recomputed_from_text"]
            + info["golds_hand_written"] == info["n"])
    assert info["langs"] == ["en", "es"]
    assert info["ks"] == [3, 4]


def test_a_gold_that_contradicts_its_rule_is_refused(decisions):
    """El fallo que este paso busca: un id tecleado que no cumple la regla."""
    victim = next(i for i, d in enumerate(decisions)
                  if d.meta["rule"] == "unique minimum price")
    wrong = next(c.id for c in decisions[victim].candidates
                 if c.id != decisions[victim].gold)
    broken = list(decisions)
    broken[victim] = dataclasses.replace(decisions[victim], gold=wrong)
    with pytest.raises(ValueError, match="unique minimum price"):
        OF.validate_set(broken)


def test_the_priority_rule_breaks_ties_by_price(decisions):
    """La mitad de las prioridades empata en duración: manda el precio."""
    tied = [d for d in decisions
            if d.family == CE.PRIORITY and "-tie-" in d.meta["id"] + "-"]
    assert tied, "the priority family lost its tie-breaking cases"
    for dec in tied:
        rows = [OF.numbers_from_text(c.text) for c in dec.candidates]
        best = max(y for _, y in rows)
        winners = [i for i, (_, y) in enumerate(rows) if y == best]
        assert len(winners) == 2, dec.meta["id"]
        cheapest = min(winners, key=lambda i: rows[i][0])
        assert dec.gold == dec.candidates[cheapest].id


def test_the_set_fingerprint_covers_the_rendered_strings(decisions):
    """Cambiar un texto mueve la huella: dos runs comparables o ninguno."""
    before = OF.set_fingerprint(decisions)
    assert before == OF.set_fingerprint(OF.overfit_decisions())
    touched = list(decisions)
    first = touched[0]
    touched[0] = dataclasses.replace(
        first, candidates=tuple(
            [CE.Candidate(first.candidates[0].id,
                          first.candidates[0].text + " (new)")]
            + list(first.candidates[1:])))
    assert OF.set_fingerprint(touched) != before


# -- (1) alineación del gold --------------------------------------------

def test_the_gold_index_follows_the_candidate_id_not_the_position(decisions):
    rng = random.Random(3)
    dec = decisions[0]
    ids = [c.id for c in dec.candidates]
    base = OF.gold_indices([dec])[0]
    moved = 0
    for _ in range(12):
        shuffled = OF.shuffled_candidates(dec, rng)
        index = OF.gold_indices([shuffled])[0]
        assert shuffled.candidates[index].id == dec.gold
        moved += int(index != base)
    assert moved, "the permutation never moved the gold: the test has no teeth"
    assert sorted(c.id for c in shuffled.candidates) == sorted(ids)


def test_a_gold_outside_the_candidates_is_refused(decisions):
    faked = dataclasses.replace(decisions[0], gold=None)
    object.__setattr__(faked, "gold", "c99")     # salta el __post_init__
    with pytest.raises(ValueError, match="not among"):
        OF.gold_indices([faked])


def test_permuting_the_candidates_does_not_move_the_gold_probability(
        toy, decisions):
    model, tokenizer, entail = toy
    rng = random.Random(11)
    sample = decisions[:6]
    with torch.no_grad():
        base, golds, _ = OF.forward_logits(model, tokenizer, sample,
                                           torch.device("cpu"), entail)
        p_base = torch.softmax(base.float(), -1).gather(
            1, golds.view(-1, 1)).squeeze(1)
        permuted = [OF.shuffled_candidates(d, rng) for d in sample]
        moved, perm_golds, _ = OF.forward_logits(model, tokenizer, permuted,
                                                 torch.device("cpu"), entail)
        p_perm = torch.softmax(moved.float(), -1).gather(
            1, perm_golds.view(-1, 1)).squeeze(1)
    assert float((p_base - p_perm).abs().max()) < 1e-6


def test_reading_the_gold_by_position_would_break_under_permutation(
        toy, decisions):
    """Teeth del test anterior: el fallo que describe SÍ se vería."""
    model, tokenizer, entail = toy
    rng = random.Random(11)
    sample = decisions[:6]
    with torch.no_grad():
        base, golds, _ = OF.forward_logits(model, tokenizer, sample,
                                           torch.device("cpu"), entail)
        p_base = torch.softmax(base.float(), -1).gather(
            1, golds.view(-1, 1)).squeeze(1)
        permuted = [OF.shuffled_candidates(d, rng) for d in sample]
        moved, _, _ = OF.forward_logits(model, tokenizer, permuted,
                                        torch.device("cpu"), entail)
        # el gold leído por POSICIÓN: el mismo índice que antes de permutar
        p_positional = torch.softmax(moved.float(), -1).gather(
            1, golds.view(-1, 1)).squeeze(1)
    assert float((p_base - p_positional).abs().max()) > 1e-3


# -- (2) truncado del estado --------------------------------------------

def test_a_state_that_does_not_fit_stops_the_run(toy):
    model, tokenizer, _ = toy
    pairs = [("State: " + "word " * 200, "The answer is: yes.")]
    with pytest.raises(CE.ScorerContractError, match="cut in silence"):
        CE.encode_pairs(tokenizer, pairs, None, max_length=32, strict=True)


def test_the_length_report_counts_the_pairs_that_do_not_fit(toy):
    model, tokenizer, _ = toy
    short = ("State: a b c", "The answer is: yes.")
    long = ("State: " + "word " * 100, "The answer is: yes.")
    report = CE.length_report(tokenizer, [short, long, long], max_length=32)
    assert report["n_truncated"] == 2
    assert report["truncated_indices"] == [1, 2]
    assert report["max_tokens"] > 32
    assert report["pass"] is False
    assert CE.length_report(tokenizer, [short], max_length=32)["pass"] is True


def test_when_it_does_truncate_it_eats_the_premise_and_not_the_option(toy):
    """`only_first`: el texto de la OPCIÓN que se juzga llega entero."""
    assert CE.TRUNCATION_STRATEGY == "only_first"
    model, tokenizer, _ = toy
    hypothesis = CE.HYPOTHESIS["en"].format(option="Kettle Flint: 22 euros")
    pairs = [("State: " + "word " * 200, hypothesis)]
    enc, report = CE.encode_pairs(tokenizer, pairs, None, max_length=32,
                                  strict=False)
    assert report["n_truncated"] == 1
    ids = enc["input_ids"][0].tolist()
    wanted = tokenizer._ids(hypothesis)
    assert ids[-len(wanted) - 1:-1] == wanted
    assert len(ids) <= 32


def test_the_trainer_refuses_to_start_on_a_truncated_state(toy, decisions):
    model, tokenizer, entail = toy
    with pytest.raises(CE.ScorerContractError):
        OF.forward_logits(model, tokenizer, decisions[:2],
                          torch.device("cpu"), entail, max_length=24)


def test_the_real_set_fits_the_window(toy, decisions):
    model, tokenizer, _ = toy
    pairs, _ = OF.flatten(decisions)
    report = CE.length_report(tokenizer, pairs)
    assert report["n_truncated"] == 0
    assert report["max_tokens"] < CE.DEFAULT_MAX_LENGTH


# -- (3) máscara de candidatos ------------------------------------------

def test_the_padding_columns_hold_exactly_no_probability():
    z = torch.arange(10, dtype=torch.float32)
    spans = [(0, 3), (3, 7), (7, 10)]
    scores = OF.stack_scores(z, spans)
    assert scores.shape == (3, 4)
    probs = torch.softmax(scores, dim=-1)
    assert float(probs[0, 3]) == 0.0
    assert float(probs[2, 3]) == 0.0
    for row, (lo, hi) in enumerate(spans):
        assert abs(float(probs[row, :hi - lo].sum()) - 1.0) < 1e-6


def test_a_k3_row_scores_the_same_alone_as_batched_with_k4(toy, decisions):
    model, tokenizer, entail = toy
    k3, k4 = _sample(decisions)
    device = torch.device("cpu")
    with torch.no_grad():
        alone, _, _ = OF.forward_logits(model, tokenizer, k3, device, entail)
        mixed, _, _ = OF.forward_logits(model, tokenizer, k3 + k4, device,
                                        entail)
    assert alone.shape[1] == 3 and mixed.shape[1] == 4
    p_alone = torch.softmax(alone.float(), -1)
    p_mixed = torch.softmax(mixed.float(), -1)[:len(k3)]
    assert float((p_alone - p_mixed[:, :3]).abs().max()) < TOL
    assert float(p_mixed[:, 3].sum()) == 0.0


def test_the_mask_lets_no_gradient_reach_a_candidate_that_does_not_exist(
        toy, decisions):
    model, tokenizer, entail = toy
    k3, k4 = _sample(decisions, n_k3=2, n_k4=1)
    scores, golds, _ = OF.forward_logits(model, tokenizer, k3 + k4,
                                         torch.device("cpu"), entail)
    kept = scores.clone().detach().requires_grad_(True)
    torch.nn.functional.cross_entropy(kept, golds).backward()
    pad = kept.grad[:len(k3), 3]
    assert float(pad.abs().max()) == 0.0
    assert float(kept.grad[:, :3].abs().max()) > 0.0


def test_evaluate_reports_no_probability_leaking_to_the_padding(toy,
                                                               decisions):
    model, tokenizer, entail = toy
    sample = decisions[:8]
    got = OF.evaluate(model, tokenizer, sample, torch.device("cpu"), entail,
                      batch=4)
    assert got["n"] == len(sample)
    assert got["max_pad_leak"] == 0.0
    assert 0.0 <= got["accuracy"] <= 1.0
    assert set(got["by_lang"]) <= {"en", "es"}
    assert got["mean_nll"] > 0.0


# -- (4) el gradiente llega ---------------------------------------------

def test_a_handful_of_steps_lowers_the_listwise_loss(toy, decisions):
    """No es aprendizaje: es que el gradiente llega y el paso se aplica."""
    model, tokenizer, entail = toy
    device = torch.device("cpu")
    sample = decisions[:6]
    opt = torch.optim.AdamW(model.parameters(), lr=5e-2)

    def step() -> float:
        scores, golds, _ = OF.forward_logits(model, tokenizer, sample, device,
                                             entail)
        loss = torch.nn.functional.cross_entropy(scores, golds)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and max(float(g.abs().max()) for g in grads) > 0.0
        opt.step()
        opt.zero_grad(set_to_none=True)
        return float(loss.detach())

    first = step()
    for _ in range(6):
        last = step()
    assert last < first
    assert math.isfinite(last)


# -- (5) el formato, el mismo en entreno y en evaluación ----------------

def _toy_checkpoint(monkeypatch, tokenizer, model):
    """Pone el juguete donde el árbol carga pesos de verdad."""
    monkeypatch.setattr(W, "require_verified", lambda wid: "/toy-weights")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: tokenizer))
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification,
                        "from_pretrained",
                        staticmethod(lambda *a, **k: model))


def test_train_and_eval_render_the_same_strings(monkeypatch, decisions):
    """El bug que la task quiere cazar, medido desde los dos caminos.

    El de entreno es `ce_overfit.forward_logits`; el de evaluación es
    `eval.ce_nograd.measure_one`, entero, con su `NliPairScorer`. Los dos
    corren sobre las MISMAS decisiones con el mismo juguete y lo que se
    compara son los pares `(premisa, hipótesis)` que cada uno ha
    tokenizado de verdad, más el recorte con el que los ha tokenizado.
    """
    sample = decisions[:5]

    train_tok = ToyTokenizer()
    with torch.no_grad():
        OF.forward_logits(ToyNli(), train_tok, sample, torch.device("cpu"),
                          0)
    train_pairs = train_tok.pairs_tokenised()

    eval_tok = ToyTokenizer()
    _toy_checkpoint(monkeypatch, eval_tok, ToyNli())
    monkeypatch.setattr(NG, "decisions", lambda: list(sample))
    monkeypatch.setattr(NG, "load_manifest", lambda wid: {"toy": True})
    NG.measure_one(OF.DEFAULT_WEIGHTS, device="cpu")
    eval_pairs = eval_tok.pairs_tokenised()

    assert len(train_pairs) == sum(d.k for d in sample)
    assert eval_pairs[:len(train_pairs)] == train_pairs

    train_call = [c for c in train_tok.calls if c["return_tensors"] == "pt"][0]
    eval_call = [c for c in eval_tok.calls if c["return_tensors"] == "pt"][0]
    assert train_call["truncation"] == eval_call["truncation"]
    assert train_call["max_length"] == eval_call["max_length"]


def test_a_second_template_in_the_trainer_would_be_caught(decisions):
    """Si el entreno tuviera su propia plantilla, los strings diferirían."""
    dec = decisions[0]
    train_pairs = [p for p in CE.render_pairs(dec)]
    local_template = "Answer: {option}"      # la plantilla que NO debe existir
    rogue = [(premise, local_template.format(option=c.text))
             for (premise, _), c in zip(train_pairs, dec.candidates)]
    assert rogue != train_pairs


def test_there_is_a_single_flatten_and_encode_for_both_paths():
    assert OF.flatten is CE.flatten_pairs
    assert OF.encode is CE.encode_pairs
    assert NG.CE is CE


def test_the_hypothesis_format_is_frozen():
    assert CE.format_fingerprint() == CE.HYPOTHESIS_FORMAT_ID


def test_touching_the_format_moves_the_fingerprint(monkeypatch):
    monkeypatch.setitem(CE.HYPOTHESIS, "en", "The answer is: {option}")
    assert CE.format_fingerprint() != CE.HYPOTHESIS_FORMAT_ID


def test_no_second_hypothesis_template_lives_in_the_tree():
    """El formato se congela en un sitio importable, y sólo en uno."""
    needle = CE.HYPOTHESIS["en"].split("{")[0].strip()
    hits = sorted(
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*.py")
        if ".venv" not in path.parts and "__pycache__" not in path.parts
        and needle in path.read_text(encoding="utf-8", errors="ignore"))
    assert hits == ["model/ce_scorer.py"]


# -- el gate: umbrales escritos antes de medir --------------------------

def _result(fitted: float, base: float, checks_pass: bool = True) -> dict:
    checks = {"gold_alignment": {"pass": checks_pass},
              "candidate_mask": {"pass": True},
              "train_eval_format": {"pass": True},
              "state_truncation": {"pass": True}}
    return {"fitted": {"accuracy": fitted},
            "baseline_untrained": {"accuracy": base},
            "pipeline_checks": checks}


def test_the_gate_passes_only_with_overfit_margin_and_pipeline():
    assert OF.gate(_result(1.0, 0.25))["pass"] is True


def test_the_gate_fails_when_the_overfit_misses_the_target():
    got = OF.gate(_result(0.90, 0.25))
    assert got["pass"] is False
    assert got["overfit_above_target"]["pass"] is False


def test_the_gate_fails_when_the_bare_checkpoint_already_solves_the_set():
    got = OF.gate(_result(1.0, 0.75))
    assert got["pass"] is False
    assert got["set_is_valid_evidence"]["pass"] is False


def test_the_gate_fails_on_a_thin_margin():
    got = OF.gate(_result(0.96, 0.70))
    assert got["set_is_valid_evidence"]["pass"] is False
    assert got["margin"]["pass"] is False
    assert got["pass"] is False


def test_the_gate_fails_when_a_pipeline_check_fails():
    got = OF.gate(_result(1.0, 0.25, checks_pass=False))
    assert got["pass"] is False
    assert got["pipeline_checks"]["failed"] == ["gold_alignment"]


def test_the_command_records_everything_that_moves_the_figure():
    args = SimpleNamespace(device="cpu", seed=OF.DEFAULT_SEED, epochs=60,
                           lr=2e-5, decisions_per_batch=8, eval_every=5,
                           weights=OF.DEFAULT_WEIGHTS)
    line = OF.command_line(args)
    for chunk in ("--device cpu", f"--seed {OF.DEFAULT_SEED}", "--epochs 60",
                  "--lr 2e-05", "--decisions-per-batch 8", "--eval-every 5",
                  f"--weights {OF.DEFAULT_WEIGHTS}",
                  "training.python.ce_overfit overfit"):
        assert chunk in line


def test_the_gate_artifact_carries_no_figure_nobody_measured():
    """Sin cómputo no hay número: el hueco se declara, no se rellena."""
    import json

    path = ROOT / "artifacts" / "gates" / OF.TASK / "overfit.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["task"] == OF.TASK
    assert "not_generalization" in doc
    assert doc["hypothesis_format"]["frozen_id"] == CE.HYPOTHESIS_FORMAT_ID
    if doc["status"] == "awaiting-operator-compute":
        assert doc["gate"]["pass"] is None
        assert doc["fitted"] is None
        assert doc["baseline_untrained"] is None
        assert doc["set"]["fingerprint"] == OF.set_fingerprint(
            OF.overfit_decisions())
        assert doc["command"]
    else:
        assert doc["status"] == "measured"
        assert isinstance(doc["fitted"]["accuracy"], float)
        assert isinstance(doc["gate"]["pass"], bool)
