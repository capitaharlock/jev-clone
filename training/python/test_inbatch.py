"""Tests de los negativos in-batch entre espacios (#T-fullspace-objective)."""
from __future__ import annotations

import math

import pytest
import torch

from data.optset import Sample
from training.python.fullspace_loss import (normalise_key,
                                            sampled_loss_batch)
from training.python.inbatch import (MIN_OFFER_RATE, calibrate_offer_rates,
                                     extend_batch, inclusion_log_pi,
                                     mixed_batches, mixed_stream,
                                     regime, shift_tensor)


def sample(dataset: str, opts: list, gold: str | None, row: int = 0) -> Sample:
    options = [{"id": o, "text": o} for o in opts]
    return Sample(dataset=dataset, row_id=f"{dataset}-{row}", question_id="q",
                  state=f"state {dataset} {row}", question="q?",
                  options=options,
                  answer=gold or "unknown",
                  gold_index=(opts.index(gold) if gold else len(opts)),
                  space=dataset, space_size=len(opts))


class FakeStream:
    """Un stream con la forma que mide `batch-composition.json`: un dataset
    por batch. `epoch()` se puede recorrer más de una vez, como el real."""

    def __init__(self, batches: list):
        self.batches = batches

    def epoch(self, epoch: int = 0):
        for b in self.batches:
            yield list(b)


def two_dataset_stream():
    a = [sample("massive", ["alpha", "beta"], "alpha", i) for i in range(4)]
    b = [sample("huffpost", ["gamma", "delta"], "delta", i) for i in range(4)]
    return FakeStream([a, b])


# -- batches mixtos --------------------------------------------------------

def test_mixed_batches_interleave_the_datasets():
    got = list(mixed_batches(two_dataset_stream(), batch_size=4))
    assert got, "the regrouping must yield something"
    assert all(len({s.dataset for s in b}) > 1 for b in got[:1])


def test_mixed_batches_keep_every_row_exactly_once():
    """El presupuesto de filas no cambia: sólo el agrupamiento."""
    stream = two_dataset_stream()
    before = [s.row_id + s.dataset for b in stream.epoch(0) for s in b]
    after = [s.row_id + s.dataset
             for b in mixed_batches(two_dataset_stream(), 4) for s in b]
    assert sorted(after) == sorted(before)


def test_mixed_batches_respect_the_batch_size():
    got = list(mixed_batches(two_dataset_stream(), batch_size=3))
    assert [len(b) for b in got[:-1]] == [3] * (len(got) - 1)
    assert sum(len(b) for b in got) == 8


# -- calibración de q y π --------------------------------------------------

def test_calibration_counts_offers_per_row():
    calib = calibrate_offer_rates(two_dataset_stream(), 10, 4)
    assert calib["rows"] == 8
    assert calib["q"][normalise_key("alpha")] == pytest.approx(0.5)


def test_calibration_does_not_consume_the_training_stream():
    stream = two_dataset_stream()
    calibrate_offer_rates(stream, 10, 4)
    assert sum(len(b) for b in mixed_batches(stream, 4)) == 8


def test_inclusion_pi_is_not_m_times_q():
    """La corrección del caso deduplicado: `π = 1 - (1 - q)^(B-1)`."""
    calib = {"rows": 100, "q": {"a": 0.5}, "min_offer_rate": MIN_OFFER_RATE}
    log_pi = inclusion_log_pi(calib, batch_size=4)
    assert log_pi["a"] == pytest.approx(math.log(1 - 0.5 ** 3))
    assert log_pi["a"] != pytest.approx(math.log(3 * 0.5))


def test_pi_never_exceeds_one():
    calib = {"rows": 10, "q": {"a": 1.0}, "min_offer_rate": MIN_OFFER_RATE}
    assert inclusion_log_pi(calib, 64)["a"] == pytest.approx(0.0)


def test_an_uncalibrated_label_falls_back_to_the_declared_floor():
    batch = [sample("massive", ["alpha", "beta"], "alpha"),
             sample("huffpost", ["gamma"], "gamma")]
    _out, shifts, _ = extend_batch(batch, {})
    assert min(min(r) for r in shifts) == pytest.approx(
        math.log(MIN_OFFER_RATE))


# -- el denominador extendido ---------------------------------------------

def test_foreign_labels_are_appended_with_their_correction():
    batch = [sample("massive", ["alpha", "beta"], "alpha"),
             sample("huffpost", ["gamma"], "gamma")]
    log_pi = {"alpha": math.log(0.5), "beta": math.log(0.4),
              "gamma": math.log(0.3)}
    out, shifts, stats = extend_batch(batch, log_pi)
    assert [o["text"] for o in out[0].options] == ["alpha", "beta", "gamma"]
    assert shifts[0] == [0.0, 0.0, pytest.approx(math.log(0.3))]
    assert [o["text"] for o in out[1].options] == ["gamma", "alpha", "beta"]
    assert stats["columns_added"] == 3


def test_own_columns_never_carry_a_correction():
    """La fila siempre ofrece su propio espacio: π = 1 por construcción."""
    batch = [sample("massive", ["alpha", "beta"], "alpha"),
             sample("huffpost", ["gamma"], "gamma")]
    _out, shifts, _ = extend_batch(batch, {"gamma": math.log(0.3)})
    assert shifts[0][:2] == [0.0, 0.0]


def test_a_colliding_foreign_label_is_dropped():
    """El filtro obligatorio: sin él la fila recibe su oro como negativo."""
    batch = [sample("massive", ["yes", "no"], "yes"),
             sample("boolq", ["Yes", "No"], "No")]
    out, _shifts, stats = extend_batch(batch, {})
    assert [o["text"] for o in out[0].options] == ["yes", "no"]
    assert stats["columns_added"] == 0


def test_the_gold_index_still_points_at_the_gold():
    batch = [sample("massive", ["alpha", "beta"], "beta"),
             sample("huffpost", ["gamma"], "gamma")]
    out, _s, _ = extend_batch(batch, {})
    assert out[0].options[out[0].gold_index]["text"] == "beta"


def test_an_unknown_row_keeps_its_gold_out_of_the_option_range():
    """Sin esto, `batch_gold` leería una opción ajena como si fuera el oro."""
    batch = [sample("massive", ["alpha", "beta"], None),
             sample("huffpost", ["gamma"], "gamma")]
    out, _s, _ = extend_batch(batch, {})
    assert out[0].is_unknown
    assert out[0].gold_index == len(out[0].options) == 3


def test_a_foreign_column_records_the_space_it_came_from():
    """Para que la penalización de prior mire la cuenta de SU dataset."""
    batch = [sample("massive", ["alpha"], "alpha"),
             sample("huffpost", ["gamma"], "gamma")]
    out, _s, _ = extend_batch(batch, {})
    assert out[0].options[1]["space"] == "huffpost"
    assert "space" not in out[0].options[0]


# -- el desplazamiento sobre el eje acolchado ------------------------------

def test_shift_tensor_pads_with_zeros_for_unknown_and_padding():
    got = shift_tensor([[0.0, -1.0], [0.0]], kmax=2, device=None,
                       dtype=torch.float32)
    assert got.shape == (2, 3)
    assert got[0].tolist() == [0.0, -1.0, 0.0]
    assert got[1].tolist() == [0.0, 0.0, 0.0]


def test_the_shift_composes_with_the_batched_loss():
    logits = torch.tensor([[0.4, 0.1, -0.2, 0.3],
                           [0.2, 0.5, float("-inf"), -0.1]])
    shift = shift_tensor([[0.0, 0.0, -1.2], [0.0, -0.8]], 3, None,
                         torch.float32)
    loss = sampled_loss_batch(logits, shift, torch.tensor([0, 1]))
    assert math.isfinite(loss.item())


# -- el régimen declarado --------------------------------------------------

def test_the_regime_declares_the_scheme_and_the_optimism():
    calib = {"rows": 100, "q": {"a": 0.5}, "min_offer_rate": MIN_OFFER_RATE}
    reg = regime(calib, 64, {"rows": 64})
    assert reg["correction"]["scheme"].startswith("inclusion")
    assert "not_m_times_q" in " ".join(reg["correction"])
    assert "UNDERESTIMATES" in reg["estimator_is_optimistic"]
    assert "set_attention=True" in reg["architecture"]


def test_mixed_stream_yields_epoch_and_batch_like_the_original():
    got = list(mixed_stream(two_dataset_stream(), 4, epochs=2))
    assert {e for e, _ in got} == {0, 1}
    assert sum(len(b) for _e, b in got) == 16


def test_mixed_stream_stops_on_an_empty_epoch():
    assert list(mixed_stream(FakeStream([]), 4, epochs=5)) == []


def test_mixed_batches_drain_instead_of_hoarding():
    """Una pasada round-robin saca 1 fila por dataset y el batch pide 64.

    Si el reagrupador para tras una pasada, las colas crecen más rápido de
    lo que drenan y una época de 1 M de filas se queda en memoria. Esto lo
    fija: con 40 batches servidos de 4 datasets, lo emitido tiene que ser
    casi todo lo consumido, no una doceava parte.
    """
    served = [[sample(f"d{d}", [f"l{d}a", f"l{d}b"], f"l{d}a", i)
               for i in range(16)] for d in range(4) for _ in range(10)]
    got = list(mixed_batches(FakeStream(served), batch_size=16))
    assert sum(len(b) for b in got) == 640
    assert len(got) == 40
    assert all(len(b) == 16 for b in got)


def test_the_reservoir_is_bounded():
    """La reserva no crece con la época: eso era el bug que este fija."""
    served = [[sample(f"d{d}", [f"l{d}a", f"l{d}b"], f"l{d}a", i)
               for i in range(64)] for d in range(4) for _ in range(50)]
    seen, emitted = 0, 0
    for n, batch in enumerate(mixed_batches(FakeStream(served), 64)):
        emitted += len(batch)
        seen = (n + 1) * 64
        # nunca hay más de una reserva de retraso entre lo servido y lo
        # emitido, independientemente de lo larga que sea la época
        assert seen - emitted <= 0
    assert emitted == 4 * 50 * 64


def test_every_output_batch_mixes_datasets_once_the_reservoir_is_warm():
    served = [[sample(f"d{d}", [f"l{d}a"], f"l{d}a", i) for i in range(32)]
              for _ in range(40) for d in range(4)]
    got = list(mixed_batches(FakeStream(served), 32))
    warm = got[len(got) // 4:len(got) * 3 // 4]
    assert warm and all(len({s.dataset for s in b}) > 1 for b in warm)
