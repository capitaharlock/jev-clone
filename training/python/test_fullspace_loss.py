"""Tests de la pérdida sobre el espacio entero (#T-fullspace-objective).

Lo que estos tests fijan, en el orden en que la task lo pide:

* **modo exacto** — igualdad de VALOR y de GRADIENTE contra un softmax
  completo escrito a mano, y el puente `π ≡ 1` entre los dos modos;
* **modo muestreado** — las propiedades de su estimador por ENUMERACIÓN
  exacta, no por Monte Carlo con una tolerancia inventada: `E[R̂] = R` bajo
  los dos esquemas, la no-intercambiabilidad de `m·Q` con `π` medida como
  sesgo, y el sesgo de Jensen de la pérdida en la dirección correcta;
* **colisiones y falsos negativos** entre espacios;
* **`prior_penalty` no es una corrección log-Q**, medido.
"""
from __future__ import annotations

import itertools
import math

import pytest
import torch

from data.optset import UNKNOWN_ID
from model.decision_head import PointerDecisionHead
from training.python.fullspace_loss import (EXACT, INCLUSION, PROPOSAL,
                                            CandidateSet, LogQError,
                                            exact_loss, filter_pool,
                                            inclusion_from_with_replacement,
                                            inclusion_log_q,
                                            negative_mass_estimate,
                                            normalise_key, proposal_log_q,
                                            sampled_loss, sampled_loss_batch,
                                            spec)
from training.python.train_decision import prior_penalty_for

#: un espacio diminuto pero ENTERO: con 3 negativos, cada esperanza de este
#: fichero se enumera exactamente y ningún test depende de una semilla.
Z_GOLD, Z_UNK = 0.7, -0.4
Z_NEG = [1.3, -0.2, 0.5]
POOL = ["neg_a", "neg_b", "neg_c"]
TOL = 1e-12
#: la enumeración es exacta; el que la aritmética no lo sea sería un
#: artefacto del test, así que todo este fichero mide en float64.
F64 = torch.float64


def _R() -> float:
    """La masa negativa verdadera, `R = Σ_{c ∈ S\\{y}} e^{z_c}`."""
    return sum(math.exp(z) for z in Z_NEG)


def _drawn_set(drawn: list[int], log_q: list[float]) -> tuple:
    """Columnas `[oro, c_1, …, c_m, unknown]` para los candidatos extraídos."""
    logits = torch.tensor([Z_GOLD] + [Z_NEG[i] for i in drawn] + [Z_UNK],
                          dtype=F64)
    cands = CandidateSet(keys=["gold"] + [POOL[i] for i in drawn], gold=0,
                         scheme=PROPOSAL, log_q=[0.0] + log_q,
                         m=max(1, len(drawn)))
    return logits, cands


# -- modo exacto -----------------------------------------------------------

def test_exact_matches_full_softmax_value():
    """(2) con R entera ES el softmax completo sobre esas columnas."""
    z = torch.tensor([Z_GOLD] + Z_NEG + [Z_UNK], dtype=F64)
    hand = -z[0] + torch.logsumexp(z, dim=0)
    assert exact_loss(z, 0).item() == pytest.approx(hand.item(), abs=TOL)


def test_exact_matches_full_softmax_gradient():
    """Y el gradiente, que es lo que entra de verdad en el optimizador."""
    z1 = torch.tensor([Z_GOLD] + Z_NEG + [Z_UNK], dtype=F64,
                      requires_grad=True)
    z2 = z1.detach().clone().requires_grad_(True)
    exact_loss(z1, 0).backward()
    (-z2[0] + torch.logsumexp(z2, dim=0)).backward()
    assert torch.allclose(z1.grad, z2.grad, atol=TOL)


def test_exact_mode_refuses_a_correction():
    """`log Q = 0` es estructural en el modo exacto, no un valor a pasar."""
    with pytest.raises(LogQError, match="exact mode has no correction"):
        CandidateSet(keys=POOL, gold=0, scheme=EXACT, log_q=[0.0, 0.0, 0.0])


def test_unknown_row_puts_the_gold_on_the_last_column():
    """Una fila `unknown` compite contra el espacio entero: R no pierde nada."""
    cands = CandidateSet(keys=POOL, gold=len(POOL), scheme=EXACT)
    assert cands.is_unknown
    z = torch.tensor(Z_NEG + [Z_UNK], dtype=F64)
    got = negative_mass_estimate(z, cands).item()
    assert got == pytest.approx(_R(), abs=1e-9)


# -- el puente entre los dos modos -----------------------------------------

def test_inclusion_with_pi_one_is_exact():
    """`π ≡ 1` ⇒ desplazamiento cero ⇒ la muestreada ES la exacta."""
    z = torch.tensor([Z_GOLD] + Z_NEG + [Z_UNK], dtype=F64)
    cands = CandidateSet(keys=["gold"] + POOL, gold=0, scheme=INCLUSION,
                         log_q=inclusion_log_q([1.0] * 4))
    assert sampled_loss(z, cands).item() == pytest.approx(
        exact_loss(z, 0).item(), abs=TOL)


def test_inclusion_with_pi_one_matches_the_exact_gradient():
    z1 = torch.tensor([Z_GOLD] + Z_NEG + [Z_UNK], dtype=F64,
                      requires_grad=True)
    z2 = z1.detach().clone().requires_grad_(True)
    cands = CandidateSet(keys=["gold"] + POOL, gold=0, scheme=INCLUSION,
                         log_q=inclusion_log_q([1.0] * 4))
    sampled_loss(z1, cands).backward()
    exact_loss(z2, 0).backward()
    assert torch.allclose(z1.grad, z2.grad, atol=TOL)


def test_the_gold_and_unknown_columns_never_carry_a_correction():
    cands = CandidateSet(keys=["gold"] + POOL, gold=0, scheme=INCLUSION,
                         log_q=inclusion_log_q([0.3, 0.5, 0.25, 0.75]))
    shift = cands.shift()
    assert shift[0].item() == 0.0      # el oro
    assert shift[-1].item() == 0.0     # `unknown`
    assert shift[1].item() == pytest.approx(math.log(0.5))


# -- modo muestreado: el estimador, por enumeración ------------------------

def test_proposal_estimator_is_unbiased():
    """`E[R̂] = R` con m extracciones i.i.d. CON reemplazo, enumeradas."""
    q = [0.5, 0.3, 0.2]
    m = 2
    total = 0.0
    for drawn in itertools.product(range(3), repeat=m):
        prob = math.prod(q[i] for i in drawn)
        logits, cands = _drawn_set(list(drawn),
                                   [proposal_log_q(q, m)[i] for i in drawn])
        total += prob * negative_mass_estimate(logits, cands).item()
    assert total == pytest.approx(_R(), abs=1e-9)


def test_proposal_keeps_drawn_duplicates():
    """Quitar un duplicado YA extraído rompería el insesgamiento."""
    q = [0.5, 0.3, 0.2]
    lq = proposal_log_q(q, 2)
    logits, cands = _drawn_set([0, 0], [lq[0], lq[0]])
    assert cands.k == 3          # oro + las dos extracciones del mismo
    assert negative_mass_estimate(logits, cands).item() == pytest.approx(
        2 * math.exp(Z_NEG[0]) / (2 * q[0]), abs=1e-9)


def test_inclusion_estimator_is_unbiased():
    """Horvitz-Thompson con inclusión Bernoulli independiente, enumerado."""
    pi = [0.6, 0.35, 0.8]
    total = 0.0
    for mask in itertools.product((0, 1), repeat=3):
        prob = math.prod(pi[i] if m else 1.0 - pi[i]
                         for i, m in enumerate(mask))
        idx = [i for i, m in enumerate(mask) if m]
        logits = torch.tensor([Z_GOLD] + [Z_NEG[i] for i in idx] + [Z_UNK],
                              dtype=F64)
        cands = CandidateSet(keys=["gold"] + [POOL[i] for i in idx], gold=0,
                             scheme=INCLUSION,
                             log_q=[0.0] + [math.log(pi[i]) for i in idx])
        total += prob * negative_mass_estimate(logits, cands).item()
    assert total == pytest.approx(_R(), abs=1e-9)


def _dedup_expectation(q: list[float], m: int, log_q_of) -> float:
    """`E[R̂]` bajo «m extracciones con reemplazo, luego únicas», enumerado.

    Éste es el diseño de los negativos in-batch: cada fila aporta su
    etiqueta y las repeticiones colapsan en el conjunto de únicas.
    """
    total = 0.0
    for drawn in itertools.product(range(len(q)), repeat=m):
        prob = math.prod(q[i] for i in drawn)
        idx = sorted(set(drawn))
        logits = torch.tensor([Z_GOLD] + [Z_NEG[i] for i in idx] + [Z_UNK],
                              dtype=F64)
        cands = CandidateSet(keys=["gold"] + [POOL[i] for i in idx], gold=0,
                             scheme=INCLUSION,
                             log_q=[0.0] + [log_q_of[i] for i in idx])
        total += prob * negative_mass_estimate(logits, cands).item()
    return total


def test_dedup_design_is_unbiased_with_the_inclusion_correction():
    """La corrección correcta del caso in-batch: `π = 1 - (1 - Q)^m`."""
    q, m = [0.5, 0.3, 0.2], 3
    pi = inclusion_from_with_replacement(q, m)
    got = _dedup_expectation(q, m, inclusion_log_q(pi))
    assert got == pytest.approx(_R(), abs=1e-9)


def test_proposal_correction_on_a_deduped_set_is_biased():
    """`m·Q` sobre un conjunto DEDUPLICADO sesga — medido, no afirmado.

    Es el error que el módulo existe para impedir: la distribución de
    propuesta y la de inclusión no son intercambiables.
    """
    q, m = [0.5, 0.3, 0.2], 3
    pi = inclusion_from_with_replacement(q, m)
    assert all(abs(p - m * qi) > 1e-3 for p, qi in zip(pi, q))
    wrong = _dedup_expectation(q, m, proposal_log_q(q, m))
    assert abs(wrong - _R()) > 0.1 * _R()


# -- la pérdida: sesgada, y consistente ------------------------------------

def _dedup_loss_expectation(q: list[float], m: int) -> float:
    total = 0.0
    pi = inclusion_from_with_replacement(q, m)
    lq = inclusion_log_q(pi)
    for drawn in itertools.product(range(len(q)), repeat=m):
        prob = math.prod(q[i] for i in drawn)
        idx = sorted(set(drawn))
        logits = torch.tensor([Z_GOLD] + [Z_NEG[i] for i in idx] + [Z_UNK],
                              dtype=F64)
        cands = CandidateSet(keys=["gold"] + [POOL[i] for i in idx], gold=0,
                             scheme=INCLUSION, log_q=[0.0] + [lq[i] for i in idx])
        total += prob * sampled_loss(logits, cands).item()
    return total


def test_the_loss_estimator_underestimates():
    """Jensen: `D̂` es insesgada y `log` cóncava ⇒ `E[L̂] ≤ L`.

    El objetivo muestreado es OPTIMISTA: reporta menos pérdida de la que la
    fila tiene sobre el espacio entero. El signo se mide aquí; asumirlo al
    revés es el error que este test cazó cuando se escribió.
    """
    q = [0.5, 0.3, 0.2]
    exact = exact_loss(torch.tensor([Z_GOLD] + Z_NEG + [Z_UNK], dtype=F64), 0).item()
    for m in (1, 2, 3):
        assert _dedup_loss_expectation(q, m) < exact


def test_the_loss_estimator_is_consistent():
    """Al crecer la muestra hacia el pool entero, `E[L̂] → L`."""
    q = [0.5, 0.3, 0.2]
    exact = exact_loss(torch.tensor([Z_GOLD] + Z_NEG + [Z_UNK], dtype=F64), 0).item()
    gaps = [exact - _dedup_loss_expectation(q, m) for m in (1, 3, 6)]
    assert gaps[0] > gaps[1] > gaps[2] > 0.0
    assert gaps[-1] < 0.25 * gaps[0]


# -- colisiones y falsos negativos entre espacios --------------------------

def test_filter_pool_drops_the_gold_by_normalised_key():
    """Una etiqueta de otro espacio con el texto del oro NO es un negativo."""
    keys, q, stats = filter_pool(
        "Card arrival", ["lost card", "card_arrival", "top up", "CARD ARRIVAL!"],
        q=[0.4, 0.3, 0.2, 0.1])
    assert keys == ["lost card", "top up"]
    assert stats["gold_collisions"] == 2
    assert q == pytest.approx([2 / 3, 1 / 3])


def test_filter_pool_drops_declared_positives_and_duplicates():
    keys, _, stats = filter_pool(
        "gold", ["alpha", "Alpha", "beta", "also right"],
        positives=["also-right"])
    assert keys == ["alpha", "beta"]
    assert stats["duplicate_keys"] == 1
    assert stats["declared_positive_collisions"] == 1


def test_filter_pool_declares_what_it_cannot_detect():
    """El falso negativo SEMÁNTICO se declara; no se esconde."""
    _, _, stats = filter_pool("gold", ["alpha"])
    assert "SEMANTIC false negatives" in stats["residual_false_negatives"]
    assert "before sampling" in stats["filtered_where"]


def test_filtering_before_sampling_keeps_the_estimator_unbiased():
    """Filtrar el pool y renormalizar `Q` deja `E[R̂] = R` sobre lo que queda."""
    pool = POOL + ["NEG_A"]                      # un duplicado de clave
    keys, q, _ = filter_pool("gold", pool, q=[0.4, 0.3, 0.2, 0.1])
    assert keys == POOL and sum(q) == pytest.approx(1.0)
    total = 0.0
    for drawn in itertools.product(range(3), repeat=2):
        prob = math.prod(q[i] for i in drawn)
        logits, cands = _drawn_set(list(drawn),
                                   [proposal_log_q(q, 2)[i] for i in drawn])
        total += prob * negative_mass_estimate(logits, cands).item()
    assert total == pytest.approx(_R(), abs=1e-9)


def test_an_empty_pool_after_filtering_is_an_error_not_a_zero():
    with pytest.raises(LogQError, match="renormalised over an empty pool"):
        filter_pool("gold", ["GOLD", "gold"], q=[0.5, 0.5])


def test_normalise_key_collapses_punctuation_and_case():
    assert normalise_key("Card_arrival!") == normalise_key("CARD ARRIVAL")


# -- `prior_penalty` no es log-Q -------------------------------------------

def test_prior_penalty_is_not_a_log_q_correction():
    """Medido: la penalización de prior deja el estimador sesgado.

    `prior_penalty_for` resta `log(cuenta del oro en train)`. Si se usara
    como si fuese `log Q`, `E[R̂] ≠ R`: no es la probabilidad de muestreo de
    ese negativo para esta fila y no corrige ningún denominador.
    """
    q, m = [0.5, 0.3, 0.2], 2
    counts = {("massive", POOL[0]): 900, ("massive", POOL[1]): 40,
              ("massive", POOL[2]): 7}
    prior = [prior_penalty_for("massive", o, counts) for o in POOL]
    assert prior != pytest.approx(proposal_log_q(q, m))
    total = 0.0
    for drawn in itertools.product(range(3), repeat=m):
        prob = math.prod(q[i] for i in drawn)
        logits, cands = _drawn_set(list(drawn), [prior[i] for i in drawn])
        total += prob * negative_mass_estimate(logits, cands).item()
    assert abs(total - _R()) > 0.5 * _R()


def test_prior_penalty_also_hits_the_gold_column_which_log_q_never_does():
    """La otra diferencia estructural, no de valor."""
    cands = CandidateSet(keys=["gold"] + POOL, gold=0, scheme=PROPOSAL,
                         log_q=[5.0, 1.0, 1.0, 1.0], m=3)
    assert cands.shift()[0].item() == 0.0


# -- omisión declarada -----------------------------------------------------

def test_a_sampled_set_without_log_q_must_declare_why():
    with pytest.raises(LogQError, match="must declare WHY"):
        CandidateSet(keys=POOL, gold=0, scheme=INCLUSION)


def test_the_declared_omission_travels_into_the_artifact():
    cands = CandidateSet(keys=POOL, gold=0, scheme=INCLUSION,
                         omission="in_set_ranking")
    decl = cands.declare()
    assert decl["log_q_applied"] is False
    assert "ranking INSIDE the offered set" in decl["log_q_omitted_because"]
    assert decl["scheme"] == INCLUSION


def test_a_zero_probability_candidate_is_refused():
    with pytest.raises(LogQError, match="cannot have been drawn"):
        proposal_log_q([0.5, 0.0, 0.5], 1)


# -- la ruta batched del trainer -------------------------------------------

def test_batch_loss_matches_the_per_row_loss():
    """El desplazamiento (3) sobre el eje acolchado `[B, K_max + 1]`."""
    logits = torch.tensor([[Z_GOLD] + Z_NEG + [Z_UNK],
                           [0.1, -0.3, 0.9, 0.2, -0.8]], dtype=F64)
    shift = torch.tensor([[0.0, -0.7, -1.2, -0.4, 0.0],
                          [0.0, -0.2, -0.5, -0.9, 0.0]], dtype=F64)
    gold = torch.tensor([0, 2])
    batched = sampled_loss_batch(logits, shift, gold)
    rows = [exact_loss(logits[i] - shift[i], int(gold[i])) for i in range(2)]
    assert batched.item() == pytest.approx(
        float(torch.stack(rows).mean()), abs=TOL)


def test_a_padded_column_must_carry_a_zero_shift_not_minus_inf():
    """`-inf - inf` sería NaN: el relleno lleva 0 y se queda en `-inf`."""
    logits = torch.tensor([[0.3, 0.1, float("-inf"), -0.2]])
    shift = torch.tensor([[0.0, -0.5, float("-inf"), 0.0]])
    with pytest.raises(LogQError, match="non-finite correction"):
        sampled_loss_batch(logits, shift, torch.tensor([0]))
    ok = sampled_loss_batch(logits, torch.tensor([[0.0, -0.5, 0.0, 0.0]]),
                            torch.tensor([0]))
    assert math.isfinite(ok.item())


# -- por qué los conjuntos no se unen en un solo softmax -------------------

def test_logits_depend_on_the_candidate_set():
    """`set_attn` ⇒ el logit de una opción cambia con sus rivales.

    La consecuencia que la spec declara: logits calculados sobre conjuntos
    distintos NO se pueden juntar en un softmax. El negativo in-batch exige
    pasar el conjunto ampliado por la cabeza, no reciclar logits viejos.
    """
    torch.manual_seed(20260924)
    head = PointerDecisionHead(d_state=32, d_model=32, n_layers=1, n_heads=4)
    mem = torch.randn(1, 6, 32)
    mask = torch.ones(1, 6, dtype=torch.bool)
    q = torch.randn(32)
    opts = torch.randn(5, 32)
    small = head(mem, mask, q, opts[:2])[0]
    big = head(mem, mask, q, opts)[0]
    assert abs(small.item() - big.item()) > 1e-4


def test_spec_names_both_modes_and_the_omission_rules():
    s = spec()
    assert set(s["modes"]) == {EXACT, PROPOSAL, INCLUSION}
    assert set(s["log_q_omission"]) == {"set_is_space", "in_set_ranking",
                                        "declared_uniform"}
    assert "UNDERESTIMATES" in s["unbiasedness"]["loss"]
    assert UNKNOWN_ID == "unknown"
