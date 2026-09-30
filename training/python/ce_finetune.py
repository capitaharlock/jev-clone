"""Aprender de verdad — el trainer del scorer compartido (#T-ce-finetune).

Paso 4 del piloto y el experimento que decide si la técnica aprende: el
mismo scorer de `model/ce_scorer.py` que `#T-ce-mechanics` sobreajustó en
48 casos, ajustado ahora sobre episodios `episode-v1` verificados y medido
en la batería de desarrollo CONTRA EL MISMO CHECKPOINT SIN AJUSTAR.

Tres subcomandos, un solo camino de renderizado y tokenizado:

* `train`   — lee episodios (`data/episode_contract.py` los valida y lo
  inválido se rechaza con motivo), construye una `Decision` con TODOS los
  candidatos, y minimiza la CE listwise sobre los K logits enmascarados
  —la misma pérdida de `ce_overfit.py`—. Ajusta el ENCODER con dos LR
  (encoder y cabeza, como Laya). Evalúa en desarrollo en presupuestos
  predeclarados (`--eval-every`, en decisiones consumidas) y escribe una
  fila por corte en `<out>/curve.jsonl`. Guarda el checkpoint FINAL y
  `train_manifest.json` con el sha de lo consumido por familia/idioma.
  Nunca elige el checkpoint con dev: la curva se publica para leerla, no
  para escoger.
* `eval`    — las CUATRO cifras sobre la batería: forzada, con abstención
  (umbral fijo `ABSTAIN_THRESHOLD`, declarado aquí), macro por familia y
  éxito conjunto contrafactual; por idioma y por K; n, azar e IC95 %
  (Wilson) en cada una; `rows_sha256` de las filas. Con `--checkpoint
  none` reproduce la columna `nli-nograd` de `#T-preflight-refs`: es el
  test de que el formato es EL MISMO (misma `eval.preflight_refs.
  nli_column`, mismo `CE.flatten_pairs`, mismo `CE.encode_pairs`).
* `gate`    — compara un run contra el control `none` sobre las MISMAS
  filas (sha, `eval.preflight_refs.same_rows`) con la regla escrita abajo
  (`GATE_RULE`) y un bootstrap pareado determinista para el IC95 % de la
  diferencia. Ninguna cifra se escribe a mano.

Lo que NO hace: abrir el sellado (`PR.load_cut` rechaza cualquier corte
que no sea `dev-`), entrenar con dev (`guard_training_cut`), o ajustar un
umbral después de ver una cifra (el umbral de abstención es una constante
con nombre).

CLI (el job canónico `trainer`; `PYTHONPATH=. .venv-train/bin/python`):

    python -m training.python.ce_finetune eval --checkpoint none \\
        --battery data/battery_dev.jsonl --device mps \\
        --out artifacts/gates/T-ce-finetune/eval-none.json
    python -m training.python.ce_finetune train \\
        --episodes artifacts/episodes-qwen/pilot-2k --init none \\
        --out artifacts/checkpoints/ce/smoke-1k --device mps \\
        --budget 1000 --eval-every 250 --holdout 0.2
    python -m training.python.ce_finetune gate \\
        --run artifacts/checkpoints/ce/smoke-1k \\
        --control artifacts/gates/T-ce-finetune/eval-none.json \\
        --out artifacts/gates/T-ce-finetune/smoke.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import battery_dev as BD  # noqa: E402
from data import episode_contract as EC  # noqa: E402
from eval import metrics_suite as MS  # noqa: E402
from eval import preflight_refs as PR  # noqa: E402
from model import ce_scorer as CE  # noqa: E402
from training.python import ce_overfit as OF  # noqa: E402

TASK = "T-ce-finetune"
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
CHECKPOINT_ROOT = ROOT / "artifacts" / "checkpoints" / "ce"
EVAL_FORMAT = "jev.ce-finetune.eval.v1"
CONTROL = "none"

DEFAULT_WEIGHTS = OF.DEFAULT_WEIGHTS
DEFAULT_SEED = 20260926
DEFAULT_EPOCHS = 4
DEFAULT_LR_ENCODER = 2e-5
DEFAULT_LR_HEAD = 1e-4
DEFAULT_DECISIONS_PER_BATCH = OF.DEFAULT_DECISIONS_PER_BATCH
DEFAULT_EVAL_EVERY = 250
DEFAULT_BUDGET = 1000
#: `warmup-linear`: sube el LR en el primer 6 % de los pasos y lo baja
#: linealmente a 0. `constant` es lo que usaron el smoke y #T-numeric-gen.
WARMUP_FRACTION = 0.06
DEFAULT_BATTERY = Path(BD.BATTERY)
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

#: El umbral de abstención, FIJO y declarado antes de medir: una fila se
#: abstiene cuando el peso máximo del softmax sobre sus K candidatos no
#: llega aquí. No se ajusta con dev; la calibración de producto es
#: `#T-battery-calib`.
ABSTAIN_THRESHOLD = 0.5

#: El split provisional del smoke (sección B de la task): por
#: `variant_group`, semilla fija, 80/20. Sólo para el smoke; la cifra
#: oficial usa los splits de `#T-episode-splits`.
SMOKE_SPLIT_SEED = 20260927
SMOKE_CUTS = (("train", 0.8), ("holdout", 0.2))

#: El IC95 % de una DIFERENCIA pareada sale de un bootstrap determinista:
#: mismas filas remuestreadas para las dos columnas, B réplicas, semilla
#: fija, percentiles 2,5/97,5. Para el conjunto contrafactual la unidad de
#: remuestreo es el grupo, no la fila.
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260927

#: La regla del gate, escrita ANTES de medir (task, sección C).
GATE_RULE = {
    "control": "the same untuned checkpoint (`--checkpoint none`) on the "
               "same rows, checked by rows_sha256 and by row identity "
               "(eval.preflight_refs.same_rows)",
    "improves_if": "the paired difference in forced accuracy AND in joint "
                   "counterfactual success both have a bootstrap CI95 whose "
                   "lower bound is above 0",
    "go_if": "it improves AND no family and no language has a forced "
             "accuracy below its untuned figure",
    "no_go_if": "it does not improve — or it improves on average while a "
                "family or a language falls below its untuned figure: that "
                "is not scaled, it is recorded as a limitation",
    "no_difference_if": "every forced pick of every row is the same in "
                        "both columns",
    "why_paired": "the two columns score the SAME rows, so the interval is "
                  "on the difference row by row, not on two independent "
                  "proportions",
}

NEG_INF = float("-inf")
HEAD_MARKER = "classifier"

#: Cuántos ejemplos renderizados se publican en el manifest (task, lista
#: de revisión punto 2: «el gold está donde el trainer cree»).
RENDERED_EXAMPLES = 5


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _dump(path, doc: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=2, ensure_ascii=False,
                                     sort_keys=True) + "\n")


# ------------------------------------------------------ los episodios

def episode_file(path) -> Path:
    """`episodes.jsonl` de un directorio, de un manifest o el fichero."""
    p = Path(path)
    if p.is_dir():
        return p / "episodes.jsonl"
    if p.suffix == ".json":  # el manifest vive junto a los episodios
        return p.parent / "episodes.jsonl"
    return p


def load_episodes(path) -> dict:
    """Lee y VALIDA por el contrato: lo inválido se rechaza con motivo."""
    src = episode_file(path)
    if not src.exists():
        raise FileNotFoundError(f"no episodes at {src}")
    raw = [json.loads(line) for line in
           src.read_text(encoding="utf-8").splitlines() if line.strip()]
    verdict = EC.batch_validate(raw)
    valid, rejected, seen = [], [], set()
    for i, (ep, reasons) in enumerate(zip(raw, verdict["per_episode"])):
        eid = ep.get("id") if isinstance(ep, dict) else None
        if reasons:
            rejected.append({"index": i, "id": eid, "reasons": reasons})
        elif eid in seen:  # la primera copia entra; las demás, fuera
            rejected.append({"index": i, "id": eid,
                             "reasons": [f"duplicate episode id {eid!r}"]})
        else:
            seen.add(eid)
            valid.append(ep)
    return {"source": _rel(src), "sha256": sha256_file(src),
            "n_read": len(raw), "n_valid": len(valid),
            "n_rejected": len(rejected), "rejected": rejected,
            "episodes": valid}


def positives_of(ep: dict) -> list[str]:
    """Los ids que cuentan como acierto, en las tres formas del contrato."""
    if ep.get("answer") is not None:
        return [ep["answer"]]
    if ep.get("acceptable_answers"):
        return list(ep["acceptable_answers"])
    return [ep["preference"]["order"][0]]


def decision_of(ep: dict) -> CE.Decision:
    """Un episodio como `Decision`, con TODOS sus candidatos y el gold por id."""
    positives = positives_of(ep)
    return CE.Decision(
        state=ep["state"], question=ep["question"],
        candidates=tuple(CE.Candidate(c["id"], c["text"])
                         for c in ep["candidates"]),
        family=ep["family"], lang=ep["lang"], gold=positives[0],
        meta={"id": ep["id"], "group": ep["variant_group"],
              "positives": positives})


def guard_training_cut(episodes: list[dict]) -> None:
    """Nada de dev ni de sellado entra en train (regla de oro 5)."""
    bad = [ep["id"] for ep in episodes
           if PR.DEV_GROUP.match(str(ep.get("variant_group", "")))
           or PR.SEALED_GROUP.match(str(ep.get("variant_group", "")))]
    if bad:
        raise PR.SealedCutTouched(
            f"{len(bad)} training episodes carry a dev-/sealed- variant_group "
            f"({bad[:4]}): the development and sealed cuts never enter train")


def provisional_split(episodes: list[dict], seed: int = SMOKE_SPLIT_SEED,
                      cuts=SMOKE_CUTS) -> dict:
    """El split del smoke: por grupo, semilla fija, auditado."""
    parts = EC.split_by_group(episodes, seed, cuts)
    tagged = [dict(ep, split=name) for name, eps in parts.items()
              for ep in eps]
    errors = EC.check_episodes_no_group_split(tagged)
    if errors:
        raise ValueError(f"group split across cuts: {errors[:3]}")
    return {"seed": seed, "cuts": [list(c) for c in cuts],
            "by": "variant_group (data.episode_contract.assign_split)",
            "counts": {name: len(eps) for name, eps in parts.items()},
            "groups": {name: len({ep["variant_group"] for ep in eps})
                       for name, eps in parts.items()},
            "parts": parts}


def consumption_manifest(episodes: list[dict]) -> dict:
    """Sha de lo consumido, por familia/idioma: auditable contra lo leído."""
    buckets: dict[str, list[dict]] = {}
    for ep in episodes:
        buckets.setdefault(f"{ep['family']}/{ep['lang']}", []).append(ep)
    out = {}
    for key in sorted(buckets):
        eps = sorted(buckets[key], key=lambda e: e["id"])
        payload = "\n".join(json.dumps(e, sort_keys=True, ensure_ascii=False)
                            for e in eps)
        out[key] = {"n": len(eps),
                    "groups": len({e["variant_group"] for e in eps}),
                    "sha256": hashlib.sha256(
                        payload.encode("utf-8")).hexdigest()}
    total = hashlib.sha256("\n".join(
        v["sha256"] for v in out.values()).encode("utf-8")).hexdigest()
    return {"n": len(episodes),
            "groups": len({e["variant_group"] for e in episodes}),
            "by_family_lang": out, "sha256": total,
            "how": "per family/lang bucket, episodes sorted by id and dumped "
                   "with sort_keys; the total is the sha of the bucket shas"}


# ---------------------------------------------------- el modelo cargado

def load_model(init: str | None, weights: str, device: str):
    """`(tokenizer, model, entail_index, source)` desde pesos o checkpoint."""
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    from model.weights import require_verified, spec_for

    if init in (None, "", CONTROL):
        path = require_verified(weights)
        spec = spec_for(weights)
        source = {"kind": "weights", "id": weights, "repo": spec["repo"],
                  "revision": spec["revision"],
                  "model_version": f"{weights}@{spec['revision']}"}
    else:
        run = Path(init)
        path = str(run / "model" if (run / "model").exists() else run)
        manifest = run / "train_manifest.json"
        doc = json.loads(manifest.read_text()) if manifest.exists() else {}
        source = {"kind": "checkpoint", "dir": _rel(path),
                  "run": doc.get("run"), "weights": doc.get("weights"),
                  "decisions_consumed": doc.get("decisions_consumed"),
                  "model_version": doc.get("model_version") or _rel(path)}
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, dtype=torch.float32)
    model.to(torch.device(device))
    return tokenizer, model, CE._entailment_index(model.config), source


class LoadedPairScorer(CE.PairScorer):
    """El scorer de evaluación sobre un modelo ya cargado.

    El MISMO camino que `CE.NliPairScorer.class_logits`: lotes de
    `batch_size`, `CE.encode_pairs` estricto, una columna del cabezal
    reducida por `CE.reduce_logits` con el modo por defecto. Control y
    checkpoint ajustado pasan por esta misma clase, así que la única
    diferencia entre las dos columnas del gate son los pesos.
    """

    def __init__(self, tokenizer, model, entail_index: int, device,
                 model_version: str,
                 batch_size: int = CE.DEFAULT_BATCH_SIZE,
                 max_length: int = CE.DEFAULT_MAX_LENGTH):
        import torch
        self.id = model_version
        self.tokenizer, self.model = tokenizer, model
        self.entail_index = int(entail_index)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.pairs_scored = 0
        self.seconds = 0.0

    def class_logits(self, pairs: list[tuple[str, str]]):
        import torch
        was_training = self.model.training
        self.model.eval()
        chunks = []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start:start + self.batch_size]
                enc, _ = CE.encode_pairs(self.tokenizer, batch, self.device,
                                         self.max_length, strict=True)
                chunks.append(self.model(**enc).logits.float().cpu())
        if self.device.type == "mps":
            torch.mps.synchronize()
        if was_training:
            self.model.train()
        self.seconds += time.perf_counter() - t0
        self.pairs_scored += len(pairs)
        return torch.cat(chunks, dim=0) if chunks else torch.zeros(0, 1)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        return CE.reduce_logits(self.class_logits(pairs),
                                CE.DEFAULT_SCORE_MODE, self.entail_index)

    def describe(self) -> dict:
        return {"id": self.id, "entail_index": self.entail_index,
                "mode": CE.DEFAULT_SCORE_MODE, "device": str(self.device),
                "batch_size": self.batch_size, "max_length": self.max_length,
                "truncation": CE.TRUNCATION_STRATEGY,
                "hypothesis_format_id": CE.HYPOTHESIS_FORMAT_ID,
                "n_classes": int(self.model.config.num_labels)}


# ----------------------------------------------------- las cuatro cifras

def _fig(node: dict) -> dict:
    return {k: node.get(k) for k in ("n", "hits", "cardinality", "chance",
                                     "accuracy", "accuracy_ci95",
                                     "beats_chance")}


def _bucket_figure(rows: list[dict], what: str) -> dict:
    parsed = [MS._row(r) for r in rows]
    hits = sum(1 for r in parsed if MS._forced_right(r))
    return MS.figure(hits, len(parsed), MS._chance_of(parsed),
                     MS._cardinality(parsed), what)


def abstention_at_threshold(rows: list[dict],
                            threshold: float = ABSTAIN_THRESHOLD) -> dict:
    """Con abstención a umbral FIJO: una fila que no llega, abstiene.

    La columna sin ajustar no trae `unknown`, así que su cifra de
    abstención de la suite es la forzada por construcción. Ésta es la que
    la task pide: umbral declarado, abstención contada como fallo, y
    cobertura al lado para que no se lea sola.
    """
    parsed = [MS._row(r) for r in rows]
    answered = [r for r in parsed if max(r["probs"][:r["k"]]) >= threshold]
    hits = sum(1 for r in answered if MS._forced_right(r))
    fig = MS.figure(hits, len(parsed), MS._chance_of(parsed),
                    MS._cardinality(parsed),
                    "accuracy with abstention at a fixed threshold on the "
                    "max softmax weight; an abstention counts as wrong")
    fig["threshold"] = threshold
    fig["rule"] = "abstain when max(softmax over K) < threshold"
    fig["coverage"] = MS._round(len(answered) / len(parsed)) if parsed else None
    fig["answered"] = len(answered)
    return fig


def four_figures(report: dict, rows: list[dict]) -> dict:
    """Forzada, con abstención, macro por familia, contrafactual conjunto;
    y por idioma y por K. Todo con n, azar e IC95 %."""
    by_lang, by_k = {}, {}
    for lang in sorted({r["lang"] for r in rows}):
        by_lang[lang] = _bucket_figure([r for r in rows if r["lang"] == lang],
                                       f"forced accuracy, lang {lang}")
    for k in sorted({r["k"] for r in rows}):
        by_k[str(k)] = _bucket_figure([r for r in rows if r["k"] == k],
                                      f"forced accuracy, K={k}")
    macro = report["macro"]
    return {
        "forced": _fig(report["ranking"]),
        "abstention": abstention_at_threshold(rows),
        "macro_by_family": {
            "accuracy_ranking": macro.get("accuracy_ranking"),
            "chance_ranking": macro.get("chance_ranking"),
            "worst_family": macro.get("worst_family"),
            "by_family": {fam: _fig(node["ranking"]) for fam, node
                          in sorted(report["by_family"].items())},
        },
        "counterfactual_joint": _fig(report["counterfactual"]),
        "by_lang": by_lang,
        "by_k": by_k,
    }


def evaluate_battery(scorer: CE.PairScorer, battery=DEFAULT_BATTERY, *,
                     checkpoint: dict, device: str) -> dict:
    """Las cuatro cifras sobre el corte de desarrollo, por la suite.

    El camino es EL DE PREFLIGHT: `PR.load_cut` (sólo dev), `PR.nli_column`
    (`CE.flatten_pairs`, softmax por fila) y `PR.report_for`
    (`eval.metrics_suite.report`). Con el scorer sin ajustar reproduce la
    columna `nli-nograd` fila a fila.
    """
    t0 = time.perf_counter()
    episodes = PR.load_cut(Path(battery))
    rows, trace = PR.nli_column(episodes, score=scorer.score_pairs)
    for row, ep in zip(rows, episodes):
        row["lang"] = ep["lang"]
    perm = {"measured": True,
            **CE.check_permutation_invariance(CE.CrossEncoderScorer(scorer),
                                              CE.CONTRACT_CASE),
            "source": f"{TASK} — model.ce_scorer.check_permutation_invariance "
                      "on the contract case, same weights, same run"}
    model_version = checkpoint.get("model_version") or scorer.id
    report = PR.report_for(
        PR.REF_NLI, rows, episodes=episodes, model_version=model_version,
        permutation=perm,
        notes=[f"{TASK}: {checkpoint.get('kind')} "
               f"{checkpoint.get('id') or checkpoint.get('dir')}",
               f"abstention at fixed threshold {ABSTAIN_THRESHOLD} is "
               "published in `figures.abstention`; the suite's abstention "
               "section is the forced figure because the K weights carry "
               "no `unknown` column"])
    report["column"] = f"{TASK}:{model_version}"
    return {
        "task": TASK,
        "format": EVAL_FORMAT,
        "generated_utc": utcnow(),
        "checkpoint": {**checkpoint, "scorer": scorer.describe(),
                       "device": device},
        "battery": {"file": _rel(battery), "n": len(episodes),
                    "sha256": BD.battery_sha(episodes),
                    "rows_sha256": PR.rows_sha(episodes),
                    "sealed_cut_read": False},
        "hypothesis_format": {"frozen_id": CE.HYPOTHESIS_FORMAT_ID,
                              "fingerprint": CE.format_fingerprint(),
                              "same": (CE.format_fingerprint()
                                       == CE.HYPOTHESIS_FORMAT_ID)},
        "score_mode": CE.DEFAULT_SCORE_MODE,
        "abstain_threshold": ABSTAIN_THRESHOLD,
        "figures": four_figures(report, rows),
        "pairs_scored": trace["pairs"],
        "seconds": round(time.perf_counter() - t0, 2),
        "report": report,
        "rows": rows,
    }


def figure_line(doc: dict) -> str:
    f = doc["figures"]
    fc, cf = f["forced"], f["counterfactual_joint"]
    return (f"forced {fc['hits']}/{fc['n']}={fc['accuracy']:.4f} "
            f"ci{fc['accuracy_ci95']} | abst@{ABSTAIN_THRESHOLD} "
            f"{f['abstention']['accuracy']:.4f} cov {f['abstention']['coverage']}"
            f" | macro {f['macro_by_family']['accuracy_ranking']:.4f} "
            f"| joint {cf['hits']}/{cf['n']}={cf['accuracy']:.4f} "
            f"ci{cf['accuracy_ci95']}")


# ------------------------------------------------------------ la pérdida

def positive_mask(decisions: list[CE.Decision], kmax: int):
    """`[B, Kmax]` bool: qué columnas son aciertos, EN EL ORDEN RENDERIZADO."""
    import torch
    mask = torch.zeros((len(decisions), kmax), dtype=torch.bool)
    for row, dec in enumerate(decisions):
        ids = [c.id for c in dec.candidates]
        positives = dec.meta.get("positives") or [dec.gold]
        for pid in positives:
            if pid not in ids:
                raise ValueError(f"positive {pid!r} not among {ids}")
            mask[row, ids.index(pid)] = True
    return mask


def listwise_loss(scores, positives):
    """CE listwise sobre los K logits enmascarados: `-log sum_pos p`.

    Con un solo positivo es exactamente `cross_entropy(scores, gold)` de
    `ce_overfit.py` (test); con varios (`acceptable_answers`) reparte la
    masa entre ellos sin inventar un gold único.
    """
    import torch
    pos = scores.masked_fill(~positives.to(scores.device), NEG_INF)
    return (torch.logsumexp(scores, dim=-1)
            - torch.logsumexp(pos, dim=-1)).mean()


def param_groups(model, lr_encoder: float, lr_head: float) -> list[dict]:
    """Dos LR: encoder y cabeza (`classifier.*`), como Laya."""
    head = [p for n, p in model.named_parameters() if HEAD_MARKER in n]
    encoder = [p for n, p in model.named_parameters() if HEAD_MARKER not in n]
    if not head or not encoder:
        raise ValueError("expected both encoder and classifier parameters")
    return [{"params": encoder, "lr": lr_encoder, "name": "encoder"},
            {"params": head, "lr": lr_head, "name": "head"}]


def rendered_examples(decisions: list[CE.Decision],
                      n: int = RENDERED_EXAMPLES) -> list[dict]:
    """Los primeros pares renderizados, con el gold marcado: para LEERLOS."""
    out = []
    for dec in decisions[:n]:
        pairs = CE.render_pairs(dec)
        out.append({"id": dec.meta["id"], "family": dec.family,
                    "lang": dec.lang, "gold": dec.gold,
                    "pairs": [{"candidate": c.id, "gold": c.id == dec.gold,
                               "premise": p, "hypothesis": h}
                              for c, (p, h) in zip(dec.candidates, pairs)]})
    return out


def save_checkpoint(model, tokenizer, path: Path) -> None:
    """HF `save_pretrained` cuando existe; un `state_dict` si es un juguete."""
    import torch
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
    else:
        torch.save(model.state_dict(), path / "state_dict.pt")


# --------------------------------------------------------------- train

def train(args, load=load_model) -> dict:
    """El ajuste. `load` es inyectable para los tests (modelo de juguete)."""
    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    loaded = load_episodes(args.episodes)
    split = None
    train_eps = loaded["episodes"]
    holdout_eps: list[dict] = []
    if args.holdout > 0:
        split = provisional_split(
            train_eps, args.split_seed,
            (("train", 1.0 - args.holdout), ("holdout", args.holdout)))
        train_eps, holdout_eps = split["parts"]["train"], split["parts"]["holdout"]
        del split["parts"]
    guard_training_cut(train_eps)
    if not train_eps:
        raise ValueError("no valid training episodes")
    decisions = [decision_of(ep) for ep in train_eps]
    holdout = [decision_of(ep) for ep in holdout_eps]

    tokenizer, model, entail_index, source = load(args.init, args.weights,
                                                  args.device)
    device = torch.device(args.device)
    all_pairs, _ = OF.flatten(decisions)
    lengths = CE.length_report(tokenizer, all_pairs)
    if lengths["n_truncated"]:
        raise CE.ScorerContractError(
            f"{lengths['n_truncated']} training pairs would be truncated "
            f"(max {lengths['max_tokens']} > {lengths['max_length']}); the "
            "run refuses to start on a cut state")
    examples = rendered_examples(decisions)
    for ex in examples:
        gold = next(p for p in ex["pairs"] if p["gold"])
        print(f"[train] example {ex['id']} ({ex['family']}/{ex['lang']}) "
              f"gold={ex['gold']}: {gold['hypothesis']!r}", flush=True)

    groups = param_groups(model, args.lr_encoder, args.lr_head)
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    sched = None
    schedule = getattr(args, "schedule", "constant")
    if schedule == "warmup-linear":
        total = max(1, math.ceil(min(args.budget, len(decisions) * args.epochs)
                                 / args.decisions_per_batch))
        warm = max(1, int(total * WARMUP_FRACTION))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / warm,
                               max(0.0, (total - s) / max(1, total - warm))))
    model_version = f"{TASK}:{out.name}"
    scorer = LoadedPairScorer(tokenizer, model, entail_index, device,
                              model_version, batch_size=args.batch_size)
    checkpoint_desc = {"kind": "checkpoint", "dir": _rel(out / "model"),
                       "run": out.name, "init": source,
                       "model_version": model_version}
    curve_path = out / "curve.jsonl"
    curve_path.write_text("")
    rng = random.Random(args.seed)
    consumed = steps = epoch = 0
    loss_sum, loss_steps = 0.0, 0
    next_eval = args.eval_every
    last_eval: dict | None = None
    curve: list[dict] = []
    t_train = time.perf_counter()
    eval_seconds = 0.0

    def do_eval(tag: str) -> dict:
        nonlocal loss_sum, loss_steps, eval_seconds
        t0 = time.perf_counter()
        doc = evaluate_battery(scorer, args.battery,
                               checkpoint=dict(checkpoint_desc,
                                               decisions_consumed=consumed),
                               device=args.device)
        row = {"tag": tag, "decisions_consumed": consumed, "steps": steps,
               "epoch": epoch,
               "train_loss": (round(loss_sum / loss_steps, 6)
                              if loss_steps else None),
               "forced": doc["figures"]["forced"]["accuracy"],
               "forced_ci95": doc["figures"]["forced"]["accuracy_ci95"],
               "abstention": doc["figures"]["abstention"]["accuracy"],
               "macro_by_family": doc["figures"]["macro_by_family"][
                   "accuracy_ranking"],
               "counterfactual_joint": doc["figures"]["counterfactual_joint"][
                   "accuracy"],
               "counterfactual_ci95": doc["figures"]["counterfactual_joint"][
                   "accuracy_ci95"],
               "by_family": {f: v["accuracy"] for f, v in doc["figures"][
                   "macro_by_family"]["by_family"].items()},
               "by_lang": {ln: v["accuracy"] for ln, v in
                           doc["figures"]["by_lang"].items()},
               "train_seconds": round(time.perf_counter() - t_train
                                      - eval_seconds, 2)}
        if holdout:
            h = OF.evaluate(model, tokenizer, holdout, device, entail_index,
                            args.decisions_per_batch)
            row["holdout_forced"] = round(h["accuracy"], 6)
            row["holdout_ci95"] = [round(x, 6) for x in h["ci95"]]
            row["holdout_n"] = h["n"]
        loss_sum, loss_steps = 0.0, 0
        with open(curve_path, "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        curve.append(row)
        eval_seconds += time.perf_counter() - t0
        print(f"[train] eval@{consumed} ({tag}): {figure_line(doc)}"
              + (f" | holdout {row['holdout_forced']:.4f}"
                 if holdout else ""), flush=True)
        return doc

    done = consumed >= args.budget
    while not done and epoch < args.epochs:
        epoch += 1
        model.train()
        order = list(range(len(decisions)))
        rng.shuffle(order)
        for start in range(0, len(order), args.decisions_per_batch):
            if consumed >= args.budget:
                break
            take = order[start:start + args.decisions_per_batch]
            take = take[:args.budget - consumed]
            chunk = [OF.shuffled_candidates(decisions[i], rng) for i in take]
            scores, _, _ = OF.forward_logits(model, tokenizer, chunk, device,
                                             entail_index)
            loss = listwise_loss(scores, positive_mask(chunk, scores.shape[1]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            if sched is not None:
                sched.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
            consumed += len(chunk)
            loss_sum += float(loss.detach())
            loss_steps += 1
            if steps % 10 == 0:
                print(f"[train] step {steps} consumed {consumed} "
                      f"loss {loss_sum / loss_steps:.4f} "
                      f"({time.perf_counter() - t_train:.0f}s)", flush=True)
            if consumed >= next_eval and consumed < args.budget:
                last_eval = do_eval("scheduled")
                next_eval += args.eval_every
                model.train()
        done = consumed >= args.budget
    if steps == 0:
        raise ValueError("nothing was trained: budget or epochs is 0")
    if last_eval is None or curve[-1]["decisions_consumed"] != consumed:
        last_eval = do_eval("final")
    train_seconds = time.perf_counter() - t_train - eval_seconds

    save_checkpoint(model, tokenizer, out / "model")
    _dump(out / "eval-dev.json", last_eval)
    manifest = {
        "task": TASK,
        "run": out.name,
        "generated_utc": utcnow(),
        "command": command_line(args),
        "status": "measured",
        "weights": args.weights,
        "init": source,
        "model_version": model_version,
        "device": args.device,
        "seed": args.seed,
        "hyperparams": {
            "lr_encoder": args.lr_encoder, "lr_head": args.lr_head,
            "decisions_per_batch": args.decisions_per_batch,
            "budget_decisions": args.budget, "epochs_max": args.epochs,
            "eval_every_decisions": args.eval_every,
            "schedule": schedule,
            "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "optimizer": "AdamW (two groups: encoder, classifier head)",
            "max_length": CE.DEFAULT_MAX_LENGTH,
            "score_mode": CE.DEFAULT_SCORE_MODE,
            "loss": "listwise cross-entropy over the row's K masked "
                    "candidate scores (-log sum of positive weights)",
            "candidate_shuffle": "every pass, per decision"},
        "hypothesis_format": {"frozen_id": CE.HYPOTHESIS_FORMAT_ID,
                              "fingerprint": CE.format_fingerprint()},
        "episodes": {k: v for k, v in loaded.items() if k != "episodes"},
        "split": split,
        "training_cut_guard": "no dev-/sealed- variant_group in train "
                              "(guard_training_cut)",
        "consumed": consumption_manifest(train_eps),
        "holdout": (consumption_manifest(holdout_eps) if holdout_eps
                    else None),
        "length_report": lengths,
        "rendered_examples": examples,
        "steps": steps,
        "epochs_run": epoch,
        "decisions_consumed": consumed,
        "train_seconds": round(train_seconds, 2),
        "eval_seconds": round(eval_seconds, 2),
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "curve": _rel(curve_path),
        "eval_dev": _rel(out / "eval-dev.json"),
        "selection": "the FINAL checkpoint at the budget is what is saved "
                     "and gated; the dev curve is published for reading, "
                     "nothing was chosen by dev",
        "final_figures": last_eval["figures"],
    }
    _dump(out / "train_manifest.json", manifest)
    print(f"[train] done: {steps} steps, {consumed} decisions, "
          f"{train_seconds:.1f}s train + {eval_seconds:.1f}s eval; "
          f"wrote {_rel(out / 'train_manifest.json')}", flush=True)
    return manifest


# ---------------------------------------------------------------- eval

def suite_report_path(out: Path, tag: str) -> Path:
    """Where a `jev.metrics.v1` report is written next to a gate file.

    C7 (`eval/gate_rules.py`) wants the suite's own report as a document of
    its own in the gate directory, not only nested inside ours.
    """
    return out.with_name(f"{out.stem}.{tag}.report.json")


def write_suite_report(out: Path, tag: str, doc: dict) -> Path | None:
    report = doc.get("report")
    if not isinstance(report, dict) or report.get("format") != "jev.metrics.v1":
        return None
    path = suite_report_path(out, tag)
    _dump(path, report)
    return path


def run_eval(args, load=load_model) -> dict:
    checkpoint = args.checkpoint
    tokenizer, model, entail_index, source = load(checkpoint, args.weights,
                                                  args.device)
    scorer = LoadedPairScorer(tokenizer, model, entail_index, args.device,
                              source["model_version"],
                              batch_size=args.batch_size)
    doc = evaluate_battery(scorer, args.battery, checkpoint=source,
                           device=args.device)
    doc["command"] = command_line(args)
    out = Path(args.out) if args.out else (
        GATE_DIR / f"eval-{CONTROL}.json" if checkpoint in (None, CONTROL)
        else Path(checkpoint) / "eval-dev.json")
    _dump(out, doc)
    write_suite_report(out, "suite", doc)
    print(f"[eval] {source['model_version']}: {figure_line(doc)}")
    print(f"[eval] rows_sha256={doc['battery']['rows_sha256']} "
          f"({doc['seconds']}s) wrote {_rel(out)}")
    return doc


# ---------------------------------------------------------------- gate

def _joint_by_group(rows: list[dict]) -> dict[str, bool]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["variant_group"], []).append(MS._row(r))
    return {g: all(MS._forced_right(r) for r in rs)
            for g, rs in groups.items() if len(rs) >= 2}


def _paired_bootstrap(a: list[int], b: list[int], seed: int,
                      reps: int) -> dict:
    """IC95 % de `mean(b) - mean(a)` remuestreando las MISMAS unidades."""
    n = len(a)
    if n == 0:
        return {"point": None, "ci95": [None, None], "n": 0}
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(b, a)]
    point = sum(diffs) / n
    reps_out = []
    for _ in range(reps):
        total = 0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        reps_out.append(total / n)
    reps_out.sort()
    lo = reps_out[int(math.floor(0.025 * (reps - 1)))]
    hi = reps_out[int(math.ceil(0.975 * (reps - 1)))]
    return {"point": round(point, 6), "ci95": [round(lo, 6), round(hi, 6)],
            "n": n, "bootstrap": {"B": reps, "seed": seed,
                                  "unit": "paired", "ci": "percentile"}}


def compare(control: dict, tuned: dict, seed: int = BOOTSTRAP_SEED,
            reps: int = BOOTSTRAP_B) -> dict:
    """La regla del gate sobre dos evaluaciones de las MISMAS filas."""
    same = PR.same_rows({"control": control["rows"], "tuned": tuned["rows"]})
    if control["battery"]["rows_sha256"] != tuned["battery"]["rows_sha256"]:
        raise PR.ProtocolMismatch("the two evaluations carry different "
                                  "rows_sha256: not the same cut")
    c_rows, t_rows = control["rows"], tuned["rows"]
    c_right = [int(MS._forced_right(MS._row(r))) for r in c_rows]
    t_right = [int(MS._forced_right(MS._row(r))) for r in t_rows]
    forced = _paired_bootstrap(c_right, t_right, seed, reps)
    forced["control"] = round(sum(c_right) / len(c_right), 6)
    forced["tuned"] = round(sum(t_right) / len(t_right), 6)
    cj, tj = _joint_by_group(c_rows), _joint_by_group(t_rows)
    keys = sorted(cj)
    joint = _paired_bootstrap([int(cj[g]) for g in keys],
                              [int(tj[g]) for g in keys], seed + 1, reps)
    joint["control"] = round(sum(cj.values()) / len(keys), 6) if keys else None
    joint["tuned"] = round(sum(tj.values()) / len(keys), 6) if keys else None

    def buckets(key: str) -> dict:
        out = {}
        for name in sorted({r[key] for r in c_rows}):
            idx = [i for i, r in enumerate(c_rows) if r[key] == name]
            c = sum(c_right[i] for i in idx) / len(idx)
            t = sum(t_right[i] for i in idx) / len(idx)
            out[str(name)] = {"n": len(idx), "control": round(c, 6),
                              "tuned": round(t, 6), "delta": round(t - c, 6),
                              "below_control": t < c}
        return out

    by_family, by_lang = buckets("family"), buckets("lang")
    drops = ([f"family:{f}" for f, v in by_family.items() if v["below_control"]]
             + [f"lang:{ln}" for ln, v in by_lang.items() if v["below_control"]])
    identical = all(MS._forced(MS._row(a)) == MS._forced(MS._row(b))
                    for a, b in zip(c_rows, t_rows))
    improves = bool(forced["ci95"][0] is not None and forced["ci95"][0] > 0
                    and joint["ci95"][0] is not None and joint["ci95"][0] > 0)
    if identical:
        verdict, reason = "NO-DIFFERENCE", GATE_RULE["no_difference_if"]
    elif improves and not drops:
        verdict, reason = "GO", GATE_RULE["go_if"]
    elif improves:
        verdict = "NO-GO"
        reason = ("improves on average but falls below the untuned figure "
                  f"in {drops}: not scaled, recorded as a limitation")
    else:
        verdict = "NO-GO"
        reason = ("no improvement with a CI95 clear of 0 (forced "
                  f"{forced['ci95']}, joint {joint['ci95']})")
    return {
        "same_rows": same,
        "rows_sha256": control["battery"]["rows_sha256"],
        "n": len(c_rows),
        "forced": forced,
        "counterfactual_joint": joint,
        "by_family": by_family,
        "by_lang": by_lang,
        "drops": drops,
        "identical_picks": identical,
        "improves": improves,
        "verdict": verdict,
        "pass": verdict == "GO",
        "reason": reason,
    }


def run_gate(args) -> dict:
    run = Path(args.run)
    tuned = json.loads((run / "eval-dev.json").read_text())
    control_path = Path(args.control)
    control = json.loads(control_path.read_text())
    manifest_path = run / "train_manifest.json"
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else {})
    verdict = compare(control, tuned, args.bootstrap_seed, args.bootstrap)
    out = Path(args.out) if args.out else GATE_DIR / "gate.json"
    previous = json.loads(out.read_text()) if out.exists() else {}
    doc = {
        "task": TASK,
        "artifact": _rel(out),
        "generated_utc": utcnow(),
        "status": "measured",
        "command": command_line(args),
        "rule": GATE_RULE,
        "prediction": previous.get("prediction"),
        "control": {"file": _rel(control_path),
                    "checkpoint": control.get("checkpoint"),
                    "command": control.get("command"),
                    "figures": control["figures"]},
        "run": {"dir": _rel(run), "manifest": _rel(manifest_path),
                "command": manifest.get("command"),
                "decisions_consumed": manifest.get("decisions_consumed"),
                "train_seconds": manifest.get("train_seconds"),
                "wall_seconds": manifest.get("wall_seconds"),
                "hyperparams": manifest.get("hyperparams"),
                "consumed": manifest.get("consumed"),
                "split": manifest.get("split"),
                "curve": manifest.get("curve"),
                "figures": tuned["figures"]},
        "hypothesis_format": {
            "control": control.get("hypothesis_format"),
            "tuned": tuned.get("hypothesis_format"),
            "same": (control.get("hypothesis_format", {}).get("fingerprint")
                     == tuned.get("hypothesis_format", {}).get("fingerprint")
                     == CE.HYPOTHESIS_FORMAT_ID)},
        "comparison": verdict,
        "verdict": verdict["verdict"],
        "pass": verdict["pass"],
        "reason": verdict["reason"],
        "honesty": [
            "every figure here is copied from the two evaluation documents "
            "or computed by compare() over their rows; nothing was typed",
            "the control is the same untuned checkpoint on the same rows "
            "(rows_sha256 and row identity checked before comparing)",
            "the checkpoint gated is the final one at the budget; the dev "
            "curve was not used to pick it",
        ],
    }
    if previous.get("attempts"):
        doc["attempts"] = previous["attempts"]
    reports = {tag: write_suite_report(out, tag, d)
               for tag, d in (("control", control), ("tuned", tuned))}
    doc["suite_reports"] = {tag: _rel(p) if p else None
                            for tag, p in reports.items()}
    _dump(out, doc)
    f, j = verdict["forced"], verdict["counterfactual_joint"]
    print(f"[gate] forced {f['control']} -> {f['tuned']} "
          f"delta {f['point']} ci{f['ci95']}")
    print(f"[gate] joint  {j['control']} -> {j['tuned']} "
          f"delta {j['point']} ci{j['ci95']}")
    print(f"[gate] drops {verdict['drops']}")
    print(f"[gate] {verdict['verdict']}: {verdict['reason']}")
    print(f"[gate] wrote {_rel(out)}")
    return doc


# ----------------------------------------------------------------- CLI

def command_line(args) -> str:
    """El comando EXACTO que produjo el artefacto."""
    parts = ["PYTHONPATH=. .venv-train/bin/python -m "
             "training.python.ce_finetune", args.command]
    skip = {"command", "func"}
    for key, value in sorted(vars(args).items()):
        if key in skip or value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        parts.append(flag if value is True else f"{flag} {shlex.quote(str(value))}")
    return " ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="training.python.ce_finetune",
                                 description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="fine-tune on episode-v1 episodes")
    t.add_argument("--episodes", required=True,
                   help="dir with episodes.jsonl, the manifest, or the jsonl")
    t.add_argument("--weights", default=DEFAULT_WEIGHTS)
    t.add_argument("--init", default=CONTROL,
                   help="checkpoint dir to continue from, or `none`")
    t.add_argument("--out", required=True)
    t.add_argument("--device", default="mps")
    t.add_argument("--seed", type=int, default=DEFAULT_SEED)
    t.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help="max passes over the training cut (budget rules)")
    t.add_argument("--lr-encoder", type=float, default=DEFAULT_LR_ENCODER)
    t.add_argument("--lr-head", type=float, default=DEFAULT_LR_HEAD)
    t.add_argument("--decisions-per-batch", type=int,
                   default=DEFAULT_DECISIONS_PER_BATCH)
    t.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY,
                   help="dev evaluation every N decisions consumed")
    t.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                   help="decisions to consume, then stop")
    t.add_argument("--schedule", choices=["constant", "warmup-linear"],
                   default="constant")
    t.add_argument("--battery", default=str(DEFAULT_BATTERY))
    t.add_argument("--holdout", type=float, default=0.0,
                   help="provisional by-group holdout share (smoke only)")
    t.add_argument("--split-seed", type=int, default=SMOKE_SPLIT_SEED)
    t.add_argument("--batch-size", type=int, default=CE.DEFAULT_BATCH_SIZE,
                   help="pairs per forward pass in evaluation")

    e = sub.add_parser("eval", help="the four figures on the battery")
    e.add_argument("--checkpoint", default=CONTROL,
                   help="run dir (with model/) or `none` for the untuned "
                        "weights")
    e.add_argument("--weights", default=DEFAULT_WEIGHTS)
    e.add_argument("--battery", default=str(DEFAULT_BATTERY))
    e.add_argument("--device", default="mps")
    e.add_argument("--batch-size", type=int, default=CE.DEFAULT_BATCH_SIZE)
    e.add_argument("--out", default=None)

    g = sub.add_parser("gate", help="run vs control on the same rows")
    g.add_argument("--run", required=True)
    g.add_argument("--control", default=str(GATE_DIR / f"eval-{CONTROL}.json"))
    g.add_argument("--out", default=None)
    g.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    g.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        train(args)
        return 0
    if args.command == "eval":
        run_eval(args)
        return 0
    doc = run_gate(args)
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
