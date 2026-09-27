"""Tests del trainer de `#T-ce-finetune` — sin GPU, sin pesos reales.

Lo que la task exige que se demuestre y no se suponga:

* un episodio inválido se rechaza con su motivo, y un id duplicado también;
* el manifest de consumo cuadra con lo leído (y se mueve si lo leído cambia);
* `eval none` reproduce el listón de `#T-preflight-refs` (0,6225 forzada,
  0,343 conjunto, mismo `rows_sha256`) — dos veces: con las puntuaciones de
  la columna medida inyectadas por el mismo aplanado, y leyendo el artefacto
  que el comando real dejó en el árbol;
* el gate da NO-GO cuando una familia cae aunque la media suba, y «sin
  diferencia» con dos checkpoints iguales; filas distintas se rechazan;
* la evaluación intermedia corre en los presupuestos predeclarados y escribe
  las cuatro cifras, no sólo la media (entreno de juguete de punta a punta).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from data import episode_contract as EC
from eval import preflight_refs as PR
from model import ce_scorer as CE
from training.python import ce_finetune as FT
from training.python.test_ce_overfit import PAD, VOCAB, ToyTokenizer

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data" / "episode_fixture.jsonl"
PREFLIGHT = ROOT / "artifacts" / "gates" / "T-preflight-refs"
GATE_DIR = ROOT / "artifacts" / "gates" / "T-ce-finetune"


class ToyWithHead(torch.nn.Module):
    """El juguete de `test_ce_overfit`, con la cabeza llamada `classifier`
    para que las dos LR del trainer tengan a quién repartirse."""

    def __init__(self, seed: int = 7, dim: int = 16):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = torch.nn.Embedding(VOCAB, dim, padding_idx=PAD)
        self.hidden = torch.nn.Linear(dim, dim)
        self.classifier = torch.nn.Linear(dim, 3)
        self.config = SimpleNamespace(
            num_labels=3,
            id2label={0: "entailment", 1: "neutral", 2: "contradiction"})

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        mask = (attention_mask if attention_mask is not None
                else torch.ones_like(input_ids)).unsqueeze(-1).to(
                    self.emb.weight.dtype)
        pooled = ((self.emb(input_ids) * mask).sum(1)
                  / mask.sum(1).clamp(min=1.0))
        return SimpleNamespace(
            logits=self.classifier(torch.tanh(self.hidden(pooled))))


def _toy_loader(init, weights, device):
    model = ToyWithHead()
    return (ToyTokenizer(), model, CE._entailment_index(model.config),
            {"kind": "toy", "id": "toy", "model_version": "toy@test"})


def _fixture() -> list[dict]:
    return [json.loads(line) for line in
            FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(tmp_path: Path, episodes: list[dict], name="episodes.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(ep, ensure_ascii=False)
                              for ep in episodes) + "\n")
    return path


# -- episodios: rechazo y manifest --------------------------------------

def test_an_invalid_episode_is_rejected_with_its_reason(tmp_path):
    eps = _fixture()
    bad = dict(eps[0], id="bad-1", question="banking77-intent-8412")
    loaded = FT.load_episodes(_write(tmp_path, eps + [bad]))
    assert loaded["n_read"] == len(eps) + 1
    assert loaded["n_valid"] == len(eps)
    assert [r["id"] for r in loaded["rejected"]] == ["bad-1"]
    assert any("dataset id" in reason for reason in
               loaded["rejected"][0]["reasons"])


def test_a_duplicate_episode_id_is_rejected(tmp_path):
    eps = _fixture()
    loaded = FT.load_episodes(_write(tmp_path, eps + [dict(eps[0])]))
    assert loaded["n_valid"] == len(eps)
    assert loaded["rejected"][0]["reasons"] == [
        f"duplicate episode id {eps[0]['id']!r}"]


def test_a_directory_or_a_manifest_resolves_to_its_episodes(tmp_path):
    _write(tmp_path, _fixture())
    (tmp_path / "manifest.json").write_text("{}")
    assert FT.episode_file(tmp_path) == tmp_path / "episodes.jsonl"
    assert FT.episode_file(tmp_path / "manifest.json") == \
        tmp_path / "episodes.jsonl"
    assert FT.load_episodes(tmp_path)["n_valid"] == 24


def test_the_consumption_manifest_matches_what_was_read(tmp_path):
    loaded = FT.load_episodes(_write(tmp_path, _fixture()))
    manifest = FT.consumption_manifest(loaded["episodes"])
    assert manifest["n"] == loaded["n_valid"]
    assert sum(v["n"] for v in manifest["by_family_lang"].values()) == \
        manifest["n"]
    again = FT.consumption_manifest(FT.load_episodes(tmp_path)["episodes"])
    assert again == manifest
    moved = [dict(ep) for ep in loaded["episodes"]]
    moved[3]["state"] = moved[3]["state"] + " (una palabra más)"
    other = FT.consumption_manifest(moved)
    key = f"{moved[3]['family']}/{moved[3]['lang']}"
    assert other["by_family_lang"][key]["sha256"] != \
        manifest["by_family_lang"][key]["sha256"]
    assert other["sha256"] != manifest["sha256"]


def test_positives_cover_the_three_answer_shapes():
    eps = _fixture()
    shapes = {next(k for k in ("answer", "acceptable_answers", "preference")
                   if ep.get(k) is not None): ep for ep in eps}
    assert set(shapes) == {"answer", "acceptable_answers", "preference"}
    assert FT.positives_of(shapes["answer"]) == [shapes["answer"]["answer"]]
    assert FT.positives_of(shapes["acceptable_answers"]) == \
        shapes["acceptable_answers"]["acceptable_answers"]
    assert FT.positives_of(shapes["preference"]) == \
        [shapes["preference"]["preference"]["order"][0]]
    for ep in eps:
        dec = FT.decision_of(ep)
        assert dec.k == len(ep["candidates"])
        assert dec.gold == dec.meta["positives"][0]


# -- el split provisional y el guardia de dev -----------------------------

def test_the_provisional_split_never_splits_a_group():
    eps = _fixture()
    split = FT.provisional_split(eps, seed=FT.SMOKE_SPLIT_SEED)
    parts = split["parts"]
    assert sum(len(p) for p in parts.values()) == len(eps)
    seen = {}
    for name, part in parts.items():
        for ep in part:
            assert seen.setdefault(ep["variant_group"], name) == name
    assert split["counts"] == {n: len(p) for n, p in parts.items()}


def test_training_refuses_a_dev_or_sealed_group():
    eps = _fixture()
    FT.guard_training_cut(eps)
    with pytest.raises(PR.SealedCutTouched):
        FT.guard_training_cut([dict(eps[0], variant_group="dev-ext-es-t01")])
    with pytest.raises(PR.SealedCutTouched):
        FT.guard_training_cut([dict(eps[0], variant_group="sealed-x")])


# -- la pérdida y las dos LR ----------------------------------------------

def test_listwise_loss_equals_cross_entropy_with_one_positive():
    scores = torch.tensor([[0.3, -1.2, 2.0, FT.NEG_INF],
                           [1.5, 0.1, FT.NEG_INF, FT.NEG_INF]])
    golds = torch.tensor([2, 0])
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[0, 2] = mask[1, 0] = True
    want = torch.nn.functional.cross_entropy(scores, golds)
    assert math.isclose(float(FT.listwise_loss(scores, mask)), float(want),
                        rel_tol=1e-6)
    mask[0, 0] = True  # un segundo positivo sólo puede bajar la pérdida
    assert float(FT.listwise_loss(scores, mask)) < float(want)


def test_the_positive_mask_follows_the_candidate_id():
    dec = FT.decision_of(_fixture()[0])
    order = list(dec.candidates)[::-1]
    import dataclasses
    flipped = dataclasses.replace(dec, candidates=tuple(order))
    m0 = FT.positive_mask([dec], dec.k)
    m1 = FT.positive_mask([flipped], dec.k)
    assert [c.id for c in dec.candidates if m0[0][[c.id for c in
            dec.candidates].index(c.id)]] == dec.meta["positives"]
    assert int(m0.sum()) == int(m1.sum()) == len(dec.meta["positives"])
    assert m0[0].tolist() == m1[0].tolist()[::-1]


def test_two_learning_rates_encoder_and_head():
    model = ToyWithHead()
    groups = FT.param_groups(model, 2e-5, 1e-4)
    assert [g["name"] for g in groups] == ["encoder", "head"]
    assert groups[0]["lr"] == 2e-5 and groups[1]["lr"] == 1e-4
    head = {id(p) for p in model.classifier.parameters()}
    assert {id(p) for p in groups[1]["params"]} == head
    assert not head & {id(p) for p in groups[0]["params"]}
    assert len(groups[0]["params"]) + len(groups[1]["params"]) == \
        len(list(model.parameters()))


# -- `eval none` reproduce el listón --------------------------------------

class ColumnScorer(CE.PairScorer):
    """Devuelve, por par, el `log p` de la columna medida de preflight —
    por el MISMO aplanado— y un hash para cualquier otro par."""

    id = "preflight-column"

    def __init__(self, episodes, column_rows):
        decisions = PR.decisions_of(episodes)
        pairs, spans = CE.flatten_pairs(decisions)
        self.z = {}
        for row, (lo, hi) in zip(column_rows, spans):
            for pair, p in zip(pairs[lo:hi], row["probs"]):
                self.z[pair] = math.log(max(p, 1e-12))
        self.fallback = CE.HashPairScorer()

    def score_pairs(self, pairs):
        return [self.z[p] if p in self.z else self.fallback.score_pairs([p])[0]
                for p in pairs]


@pytest.fixture
def preflight_column():
    path = PREFLIGHT / "columns" / "nli-nograd.json"
    if not path.exists():
        pytest.skip("preflight column not in this checkout")
    return json.loads(path.read_text())


@pytest.fixture
def preflight_report():
    path = PREFLIGHT / "refs-nli-nograd.json"
    if not path.exists():
        pytest.skip("preflight report not in this checkout")
    return json.loads(path.read_text())


def test_eval_none_reproduces_the_preflight_column_by_the_same_flatten(
        preflight_column, preflight_report):
    episodes = PR.load_cut()
    scorer = ColumnScorer(episodes, preflight_column["rows"])
    doc = FT.evaluate_battery(scorer, checkpoint={"kind": "test",
                                                  "id": "column"},
                              device="cpu")
    f = doc["figures"]
    assert doc["battery"]["rows_sha256"] == preflight_column["rows_sha256"]
    assert f["forced"]["hits"] == preflight_report["ranking"]["hits"]
    assert f["forced"]["accuracy"] == preflight_report["ranking"]["accuracy"]
    assert f["forced"]["accuracy"] == 0.6225
    assert f["counterfactual_joint"]["accuracy"] == \
        preflight_report["counterfactual"]["accuracy"]
    assert f["macro_by_family"]["accuracy_ranking"] == \
        preflight_report["macro"]["accuracy_ranking"]
    for fam, node in preflight_report["by_family"].items():
        assert f["macro_by_family"]["by_family"][fam]["hits"] == \
            node["ranking"]["hits"]
    assert set(f["by_lang"]) == {"en", "es"}
    assert set(f["by_k"]) == {"2", "3", "8"}
    for fig in (f["forced"], f["abstention"], f["counterfactual_joint"],
                *f["by_lang"].values(), *f["by_k"].values()):
        assert fig["n"] and fig["chance"] is not None
        assert len(fig["accuracy_ci95"]) == 2
    assert f["abstention"]["threshold"] == FT.ABSTAIN_THRESHOLD
    assert doc["hypothesis_format"]["same"]


def test_the_tree_artifact_of_eval_none_reproduces_the_listón(
        preflight_column, preflight_report):
    path = GATE_DIR / "eval-none.json"
    if not path.exists():
        pytest.skip("eval-none.json not measured in this checkout")
    doc = json.loads(path.read_text())
    assert doc["format"] == FT.EVAL_FORMAT
    assert doc["checkpoint"]["kind"] == "weights"
    assert doc["checkpoint"]["model_version"] == \
        preflight_report["model_version"]
    assert doc["battery"]["rows_sha256"] == preflight_column["rows_sha256"]
    assert doc["figures"]["forced"]["hits"] == \
        preflight_report["ranking"]["hits"]
    assert doc["figures"]["forced"]["accuracy"] == 0.6225
    assert doc["figures"]["counterfactual_joint"]["hits"] == \
        preflight_report["counterfactual"]["hits"]
    assert doc["hypothesis_format"]["frozen_id"] == CE.HYPOTHESIS_FORMAT_ID
    assert "--checkpoint none" in doc["command"]
    assert doc["command"].split(" -m ")[1].startswith(
        "training.python.ce_finetune eval")
    got = {r["row_id"]: r["pred"] for r in doc["rows"]}
    want = {r["row_id"]: r["pred"] for r in preflight_column["rows"]}
    assert got == want


# -- el gate --------------------------------------------------------------

def _eval_doc(episodes, right) -> dict:
    """Un documento de evaluación sintético: `right(i, ep) -> bool`."""
    rows = []
    for i, ep in enumerate(episodes):
        k = len(ep["candidates"])
        gold = PR.gold_index(ep)
        pred = gold if right(i, ep) else (gold + 1) % k
        probs = [0.1 / (k - 1)] * k
        probs[pred] = 0.9
        row = PR.row_of(ep, pred, probs)
        row["lang"] = ep["lang"]
        rows.append(row)
    return {"rows": rows, "battery": {"rows_sha256": PR.rows_sha(episodes)},
            "figures": {}, "hypothesis_format": {
                "fingerprint": CE.HYPOTHESIS_FORMAT_ID}}


@pytest.fixture
def battery():
    return PR.load_cut()


def test_the_gate_says_no_go_when_a_family_falls_even_if_the_mean_rises(
        battery):
    control = _eval_doc(battery, lambda i, ep: i % 2 == 0)
    tuned = _eval_doc(battery, lambda i, ep: (
        i % 4 == 0 if ep["family"] == CE.COMPARISON else True))
    got = FT.compare(control, tuned, reps=200)
    assert got["forced"]["tuned"] > got["forced"]["control"]
    assert got["forced"]["ci95"][0] > 0
    assert got["improves"]
    assert got["drops"] == [f"family:{CE.COMPARISON}"]
    assert got["verdict"] == "NO-GO" and got["pass"] is False
    assert "limitation" in got["reason"]


def test_the_gate_says_no_go_when_a_language_falls(battery):
    control = _eval_doc(battery, lambda i, ep: i % 2 == 0)
    tuned = _eval_doc(battery, lambda i, ep: (
        i % 4 == 0 if ep["lang"] == "es" else True))
    got = FT.compare(control, tuned, reps=200)
    assert got["drops"] == ["lang:es"]
    assert got["verdict"] == "NO-GO"


def test_two_identical_checkpoints_give_no_difference(battery):
    control = _eval_doc(battery, lambda i, ep: i % 3 == 0)
    tuned = _eval_doc(battery, lambda i, ep: i % 3 == 0)
    got = FT.compare(control, tuned, reps=200)
    assert got["identical_picks"]
    assert got["forced"]["point"] == 0.0
    assert got["forced"]["ci95"] == [0.0, 0.0]
    assert got["verdict"] == "NO-DIFFERENCE" and got["pass"] is False


def test_a_clear_improvement_with_no_drop_is_go(battery):
    control = _eval_doc(battery, lambda i, ep: i % 2 == 0)
    tuned = _eval_doc(battery, lambda i, ep: i % 5 != 0)
    got = FT.compare(control, tuned, reps=200)
    assert got["drops"] == []
    assert got["counterfactual_joint"]["ci95"][0] > 0
    assert got["verdict"] == "GO" and got["pass"] is True


def test_no_improvement_is_no_go_not_go(battery):
    control = _eval_doc(battery, lambda i, ep: i % 2 == 0)
    tuned = _eval_doc(battery, lambda i, ep: i % 2 == 1)
    got = FT.compare(control, tuned, reps=200)
    assert got["forced"]["point"] == 0.0
    assert not got["identical_picks"]
    assert got["verdict"] == "NO-GO"


def test_the_gate_refuses_two_different_row_sets(battery):
    control = _eval_doc(battery, lambda i, ep: True)
    tuned = _eval_doc(battery[:-1], lambda i, ep: True)
    with pytest.raises(PR.ProtocolMismatch):
        FT.compare(control, tuned, reps=50)


def test_the_paired_bootstrap_is_deterministic():
    a = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0] * 10
    b = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1] * 10
    x = FT._paired_bootstrap(a, b, seed=3, reps=300)
    y = FT._paired_bootstrap(a, b, seed=3, reps=300)
    assert x == y
    assert x["point"] == 0.3 and x["ci95"][0] > 0
    assert x["bootstrap"]["unit"] == "paired"


# -- entreno de juguete de punta a punta ----------------------------------

def test_intermediate_evaluations_run_at_the_declared_budgets_with_four_figures(
        tmp_path):
    episodes_dir = tmp_path / "episodes"
    episodes_dir.mkdir()
    _write(episodes_dir, _fixture())
    out = tmp_path / "run"
    args = FT.build_parser().parse_args([
        "train", "--episodes", str(episodes_dir), "--out", str(out),
        "--device", "cpu", "--budget", "12", "--eval-every", "4",
        "--decisions-per-batch", "4", "--epochs", "5", "--holdout", "0.25",
        "--lr-encoder", "1e-2", "--lr-head", "5e-2"])
    manifest = FT.train(args, load=_toy_loader)

    curve = [json.loads(line) for line in
             (out / "curve.jsonl").read_text().splitlines()]
    assert [row["decisions_consumed"] for row in curve] == [4, 8, 12]
    assert curve[-1]["tag"] == "final"
    for row in curve:
        for key in ("forced", "forced_ci95", "abstention", "macro_by_family",
                    "counterfactual_joint", "counterfactual_ci95",
                    "by_family", "by_lang", "holdout_forced"):
            assert key in row, key
        assert set(row["by_family"]) == set(CE.FAMILIES)
    assert manifest["decisions_consumed"] == 12
    assert manifest["steps"] == 3
    assert manifest["split"]["counts"]["train"] + \
        manifest["split"]["counts"]["holdout"] == 24
    assert manifest["consumed"]["n"] == manifest["split"]["counts"]["train"]
    assert manifest["holdout"]["n"] == manifest["split"]["counts"]["holdout"]
    assert manifest["hyperparams"]["lr_encoder"] == 1e-2
    assert manifest["hyperparams"]["lr_head"] == 5e-2
    assert manifest["hypothesis_format"]["fingerprint"] == \
        CE.HYPOTHESIS_FORMAT_ID
    assert len(manifest["rendered_examples"]) == FT.RENDERED_EXAMPLES
    assert all(any(p["gold"] for p in ex["pairs"])
               for ex in manifest["rendered_examples"])
    assert "chosen by dev" in manifest["selection"]
    assert (out / "model" / "state_dict.pt").exists()

    dev = json.loads((out / "eval-dev.json").read_text())
    assert dev["format"] == FT.EVAL_FORMAT
    assert dev["checkpoint"]["decisions_consumed"] == 12
    assert dev["figures"]["forced"]["n"] == 400
    # the consumption manifest is auditable against what was read
    loaded = FT.load_episodes(episodes_dir)
    split = FT.provisional_split(loaded["episodes"], FT.SMOKE_SPLIT_SEED,
                                 (("train", 0.75), ("holdout", 0.25)))
    assert FT.consumption_manifest(split["parts"]["train"]) == \
        manifest["consumed"]


def test_a_handful_of_toy_steps_lowers_the_listwise_loss(tmp_path):
    episodes_dir = tmp_path / "episodes"
    episodes_dir.mkdir()
    _write(episodes_dir, _fixture())
    out = tmp_path / "run"
    args = FT.build_parser().parse_args([
        "train", "--episodes", str(episodes_dir), "--out", str(out),
        "--device", "cpu", "--budget", "96", "--eval-every", "48",
        "--decisions-per-batch", "8", "--epochs", "4",
        "--lr-encoder", "5e-2", "--lr-head", "5e-2"])
    FT.train(args, load=_toy_loader)
    curve = [json.loads(line) for line in
             (out / "curve.jsonl").read_text().splitlines()]
    assert curve[-1]["train_loss"] < curve[0]["train_loss"]


def test_the_gate_reads_a_run_and_keeps_the_written_prediction(tmp_path,
                                                                battery):
    run = tmp_path / "run"
    run.mkdir()
    control = _eval_doc(battery, lambda i, ep: i % 2 == 0)
    tuned = _eval_doc(battery, lambda i, ep: i % 5 != 0)
    for doc in (control, tuned):
        doc["figures"] = {"forced": {"accuracy": None}}
    (tmp_path / "control.json").write_text(json.dumps(control))
    (run / "eval-dev.json").write_text(json.dumps(tuned))
    (run / "train_manifest.json").write_text(json.dumps(
        {"command": "x", "decisions_consumed": 1, "train_seconds": 1.0}))
    out = tmp_path / "gate.json"
    out.write_text(json.dumps({"prediction": {"written": "before"}}))
    args = FT.build_parser().parse_args([
        "gate", "--run", str(run), "--control", str(tmp_path / "control.json"),
        "--out", str(out), "--bootstrap", "100"])
    doc = FT.run_gate(args)
    assert doc["prediction"] == {"written": "before"}
    assert doc["verdict"] == "GO" and doc["pass"] is True
    assert doc["rule"] == FT.GATE_RULE
    assert json.loads(out.read_text())["verdict"] == "GO"


def test_the_command_records_what_moves_the_figure():
    args = FT.build_parser().parse_args([
        "train", "--episodes", "x", "--out", "y", "--budget", "1000",
        "--lr-encoder", "1e-5", "--holdout", "0.2"])
    cmd = FT.command_line(args)
    for piece in ("train", "--budget 1000", "--lr-encoder 1e-05",
                  "--holdout 0.2", "--seed", "--eval-every", "--lr-head"):
        assert piece in cmd
    assert "--func" not in cmd


def test_the_family_enumeration_matches_the_contract():
    assert tuple(EC.FAMILIES) == tuple(CE.FAMILIES)



def test_the_suite_report_is_written_as_its_own_document(tmp_path):
    """C7: the suite's report lands as its own file next to the gate."""
    out = tmp_path / "smoke.json"
    path = FT.write_suite_report(
        out, "tuned", {"report": {"format": "jev.metrics.v1", "x": 1}})
    assert path.name == "smoke.tuned.report.json"
    assert json.loads(path.read_text())["format"] == "jev.metrics.v1"


def test_no_suite_report_no_file(tmp_path):
    assert FT.write_suite_report(tmp_path / "smoke.json", "tuned", {}) is None
    assert list(tmp_path.iterdir()) == []
