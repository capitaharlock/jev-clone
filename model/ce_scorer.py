"""El scorer compartido — un solo modelo lee estado, pregunta y opción (#T-ce-scorer).

La primera pieza del piloto de recuperación
(`.meshkore/docs/plan-recuperacion-2026-09-24.md` §5). Un único scorer
preentrenado para relacionar textos puntúa cada candidato leyendo los tres
textos JUNTOS, y la decisión es el softmax sobre esas K puntuaciones:

    z_i = scorer_compartido(ESTADO, PREGUNTA, RESPUESTA_i)
    p   = softmax(z_1 … z_K)
    L   = -log p_correcta

Lo que este módulo garantiza estructuralmente, y `scalar_output_report()`
comprueba en vez de afirmarlo:

* **Una salida escalar.** El checkpoint es un cabezal NLI de 2 o 3 clases
  (`entailment` / …) y de esa salida se lee UN número por par. El número
  de clases del cabezal no depende del catálogo de etiquetas: no hay una
  neurona por etiqueta ni una matriz `num_labels x d` que crezca con el
  espacio, exactamente igual que en `model/decision_head.py`.
* **Todos los candidatos pasan por el mismo modelo.** Los K pares de una
  pregunta comparten pesos, formato y tokenizador; la única diferencia
  entre dos pasadas es el texto de la opción (y, cuando el contrato lo
  pide, nada más — véase abajo).
* **La interacción ocurre dentro del encoder**, no en una cabeza sobre
  representaciones congeladas: estado, pregunta y opción entran en la
  misma secuencia de tokens.

El formato de hipótesis, FIJADO
-------------------------------

Medir «sin entrenar» sólo significa algo si el formato está escrito y no
se toca después de ver la cifra. El de esta task es `HYPOTHESIS`, una
plantilla por idioma:

    premisa   = "Estado: {state}\\nPregunta: {question}"
    hipótesis = "La respuesta a esta pregunta es: {option}."

Cambiar cualquiera de las dos invalida las cifras publicadas en
`artifacts/gates/T-ce-scorer/`: se republica con el formato nuevo o no se
compara. `SCORE_MODES` fija la misma disciplina sobre CÓMO se reduce la
salida del cabezal a `z`: `entail_logit` (el logit crudo de la clase
`entailment`, que es la receta zero-shot publicada de estos checkpoints)
es el modo por defecto; `entail_logprob` (log-softmax sobre las clases
NLI) se publica al lado porque los dos checkpoints no tienen el mismo
número de clases —3 y 2— y el logit crudo no es directamente comparable
entre ellos.

El contrato de contexto comparativo
-----------------------------------

De `#T-episode-contract`, y la razón por la que este módulo no es un
bucle de K llamadas independientes. Cuando la información decisiva vive
DENTRO de las opciones («¿cuál dura más?» con los atributos en cada
candidato), puntuar una opción sin ver las demás pierde el significado de
«mejor»: el scorer no puede saber si 5 años es mucho sin saber contra qué.
`COMPARATIVE_CONTEXT` declara, POR FAMILIA, si la premisa lleva un bloque
con todos los candidatos.

Ese bloque se renderiza ordenado por TEXTO, no por id ni por posición, y
por eso es literalmente el mismo string en las K pasadas de una pregunta.
De ahí salen dos propiedades que los tests exigen en vez de suponer:
permutar los candidatos no mueve nada, e intercambiar los textos de dos
candidatos conservando sus ids intercambia sus puntuaciones — el scorer
lee texto, no índice.

Con el contexto DESACTIVADO la premisa no menciona a los demás
candidatos, así que cambiar un candidato ajeno no puede mover la
puntuación del evaluado. Las dos direcciones están probadas
(`check_comparative_context`).

Alcance: esta task es la pieza y su medición SIN entrenar. El ajuste es
`#T-ce-finetune`; el coste de las K pasadas (y su recuperación por
destilación) es `#shared-state-distill`, y aquí no se da por resuelto.

CLI:
    PYTHONPATH=. .venv-train/bin/python -m model.ce_scorer contract
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = "T-ce-scorer"
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", TASK)

LANGS = ("en", "es")

#: La plantilla de hipótesis, por idioma. FIJADA: cambiarla obliga a
#: republicar toda cifra medida con ella.
HYPOTHESIS = {
    "en": "The answer to this question is: {option}.",
    "es": "La respuesta a esta pregunta es: {option}.",
}

#: La premisa sin contexto comparativo.
PREMISE = {
    "en": "State: {state}\nQuestion: {question}",
    "es": "Estado: {state}\nPregunta: {question}",
}

#: El bloque de contexto comparativo, cuando la familia lo pide.
CONTEXT_HEADER = {
    "en": "Options under comparison:",
    "es": "Opciones en comparación:",
}

#: Las cinco familias del plan §6. `#T-episode-contract` es quien fija la
#: enumeración canónica del episodio; estos nombres son los mismos y hay
#: un test que falla si aquí aparece una sexta a espaldas de aquel.
EXTRACTION = "extraction_paraphrase"
COMPARISON = "attribute_comparison"
DESCRIPTION = "description_classification"
INFERENCE = "textual_inference_negation"
PRIORITY = "priority_decision"
FAMILIES = (EXTRACTION, COMPARISON, DESCRIPTION, INFERENCE, PRIORITY)

#: El contrato de contexto comparativo, familia por familia y explícito.
#: True = la premisa lleva a TODOS los candidatos; la información decisiva
#: vive dentro de las opciones y «mejor» no significa nada sin las demás.
COMPARATIVE_CONTEXT = {
    EXTRACTION: False,    # el hecho está en el estado; las opciones no aportan
    COMPARISON: True,     # «¿cuál es más barato?» exige ver los dos precios
    DESCRIPTION: False,   # la definición viaja con cada opción, se juzga sola
    INFERENCE: False,     # entailment del estado contra una sola hipótesis
    PRIORITY: True,       # el criterio ordena candidatos entre sí, y desempata
}

#: Reducciones admitidas de la salida del cabezal NLI a un escalar.
ENTAIL_LOGIT = "entail_logit"
ENTAIL_LOGPROB = "entail_logprob"
SCORE_MODES = (ENTAIL_LOGIT, ENTAIL_LOGPROB)
DEFAULT_SCORE_MODE = ENTAIL_LOGIT

#: Tolerancia numérica de los tests de invariancia, sobre PROBABILIDADES
#: realineadas. No es 0 porque la inferencia por lotes reduce en un orden
#: que depende de con quién comparte batch un par; con el mismo string en
#: el mismo lote la diferencia real observada es 0.
PERMUTATION_TOL = 1e-5

DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 512

#: Cómo se recorta un par que no cabe. `only_first` recorta SÓLO la
#: premisa: la hipótesis —el texto de la opción que se está juzgando— no
#: se toca nunca. La estrategia por defecto de HF es `longest_first`, que
#: con premisas largas recorta el ESTADO en silencio; ese es exactamente
#: el fallo de tubería que `#T-ce-mechanics` tenía que descartar, y por
#: eso aquí es explícita y además va vigilada (`length_report`). Cambiarla
#: no altera ninguna cifra ya publicada: el par más largo de
#: `eval/ce_nograd.py` mide 104 tokens contra una ventana de 512.
TRUNCATION_STRATEGY = "only_first"


def format_fingerprint() -> str:
    """La huella del FORMATO, para congelarlo por igualdad y no por prosa.

    Cubre todo lo que decide qué texto ve el modelo: las dos plantillas,
    la cabecera del bloque comparativo, el contrato por familia y la
    reducción por defecto a escalar. Cualquier cambio mueve la huella, y
    `HYPOTHESIS_FORMAT_ID` deja de cuadrar: el test falla y la cifra
    publicada con la huella vieja queda marcada como no comparable.
    """
    payload = json.dumps({
        "hypothesis": HYPOTHESIS,
        "premise": PREMISE,
        "context_header": CONTEXT_HEADER,
        "comparative_context": COMPARATIVE_CONTEXT,
        "default_score_mode": DEFAULT_SCORE_MODE,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


#: El formato CONGELADO para el resto del piloto (`#T-ce-mechanics`, punto
#: 4 del gate). No es un comentario: es el valor que `format_fingerprint()`
#: tiene que devolver, comprobado por `test_hypothesis_format_is_frozen`.
#: Para cambiar el formato hay que (a) cambiarlo, (b) actualizar esta
#: constante y (c) republicar toda cifra medida con la anterior —
#: `artifacts/gates/T-ce-scorer/` y `artifacts/gates/T-ce-mechanics/`
#: llevan la huella dentro para que se sepa cuál es cuál.
HYPOTHESIS_FORMAT_ID = "b215e3003cc60c0f"


class ScorerContractError(ValueError):
    """El contrato de contexto comparativo o el formato no cuadran."""


@dataclass(frozen=True)
class Candidate:
    """Un candidato: id OPACO (nunca se puntúa) y el texto que sí se lee."""

    id: str
    text: str

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("a candidate needs a non-empty id")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(f"candidate {self.id!r} has no text to read")


@dataclass(frozen=True)
class Decision:
    """Una pregunta con sus K candidatos: la unidad que el softmax cubre."""

    state: str
    question: str
    candidates: tuple[Candidate, ...]
    family: str = EXTRACTION
    lang: str = "en"
    gold: str | None = None
    #: sobreescribe el contrato de la familia — sólo para los tests que
    #: miden las DOS direcciones sobre el mismo caso
    comparative_context: bool | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.state.strip():
            raise ValueError("a decision needs a non-empty state")
        if not self.question.strip():
            raise ValueError("a decision needs a non-empty question")
        if len(self.candidates) < 2:
            raise ValueError("a decision needs at least 2 candidates (K>=2)")
        ids = [c.id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate candidate ids: {ids}")
        if self.family not in FAMILIES:
            raise ScorerContractError(
                f"unknown family {self.family!r}; {FAMILIES}")
        if self.lang not in LANGS:
            raise ValueError(f"unsupported lang {self.lang!r}; {LANGS}")
        if self.gold is not None and self.gold not in ids:
            raise ValueError(f"gold {self.gold!r} is not a candidate id")

    @property
    def k(self) -> int:
        return len(self.candidates)

    def wants_context(self) -> bool:
        """El contrato de la familia, salvo override explícito."""
        if self.comparative_context is not None:
            return bool(self.comparative_context)
        return COMPARATIVE_CONTEXT[self.family]


def context_block(candidates: tuple[Candidate, ...] | list[Candidate],
                  lang: str) -> str:
    """El bloque de candidatos, IDÉNTICO en las K pasadas de la pregunta.

    Ordenado por texto —no por id, no por posición— para que sea el mismo
    string tanto si se permutan los candidatos como si se intercambian sus
    textos entre ids. Los ids no entran: son opacos y el scorer no debe
    poder leerlos.
    """
    lines = sorted(c.text for c in candidates)
    return CONTEXT_HEADER[lang] + "\n" + "\n".join(f"- {t}" for t in lines)


def render_pair(decision: Decision, candidate: Candidate) -> tuple[str, str]:
    """(premisa, hipótesis) de UN candidato, con el formato fijado."""
    premise = PREMISE[decision.lang].format(state=decision.state.strip(),
                                            question=decision.question.strip())
    if decision.wants_context():
        premise = premise + "\n" + context_block(decision.candidates,
                                                 decision.lang)
    hypothesis = HYPOTHESIS[decision.lang].format(option=candidate.text.strip())
    return premise, hypothesis


def render_pairs(decision: Decision) -> list[tuple[str, str]]:
    """Los K pares de una decisión, en el orden en que llegaron."""
    return [render_pair(decision, c) for c in decision.candidates]


def flatten_pairs(decisions: list[Decision] | tuple[Decision, ...]
                  ) -> tuple[list[tuple[str, str]], list[tuple[int, int]]]:
    """Los pares de VARIAS decisiones y el tramo `[lo, hi)` de cada una.

    La ÚNICA función que aplana en el árbol: la usan el scorer de
    evaluación (`CrossEncoderScorer.score_many`, `eval/ce_nograd.py`) y el
    entreno (`training/python/ce_overfit.py`). Dos aplanados distintos son
    la forma más fácil de que entreno y evaluación acaben viendo strings
    distintos sin que nada falle, y es el fallo que `#T-ce-mechanics`
    tenía que descartar: aquí no hay dos.
    """
    pairs: list[tuple[str, str]] = []
    spans: list[tuple[int, int]] = []
    for decision in decisions:
        rendered = render_pairs(decision)
        spans.append((len(pairs), len(pairs) + len(rendered)))
        pairs.extend(rendered)
    return pairs, spans


def length_report(tokenizer, pairs: list[tuple[str, str]],
                  max_length: int = DEFAULT_MAX_LENGTH) -> dict:
    """¿Cabe el ESTADO en la ventana, o se está recortando en silencio?

    El truncado silencioso es un fallo de tubería que no da error y no se
    ve en la pérdida: el modelo simplemente deja de leer el final del
    estado, que es donde suele estar el hecho decisivo. Esto lo cuenta
    ANTES de tokenizar de verdad, para que un run pueda negarse a
    empezar (`strict_length`) en vez de publicar una cifra medida sobre
    estados a medias.
    """
    lengths = [len(ids) for ids in
               tokenizer([p for p, _ in pairs], [h for _, h in pairs],
                         truncation=False)["input_ids"]] if pairs else []
    over = [i for i, n in enumerate(lengths) if n > max_length]
    return {"n_pairs": len(pairs), "max_tokens": max(lengths, default=0),
            "max_length": int(max_length), "truncation": TRUNCATION_STRATEGY,
            "n_truncated": len(over), "truncated_indices": over[:16],
            "pass": not over}


def encode_pairs(tokenizer, pairs: list[tuple[str, str]], device=None,
                 max_length: int = DEFAULT_MAX_LENGTH, strict: bool = True):
    """Tokeniza los pares con la estrategia FIJADA, o se niega a hacerlo.

    El otro sitio donde entreno y evaluación pueden divergir sin que salte
    nada: mismas plantillas pero `truncation=True` en un lado y
    `only_first` en el otro, o ventanas distintas, y el modelo lee textos
    distintos. Esta función es la única que llama al tokenizador para
    puntuar pares, así que la divergencia no cabe: `NliPairScorer` (eval) y
    `training.python.ce_overfit` (entreno) pasan por aquí.

    Devuelve `(enc, report)`; con `strict` un par que no cabe es un error,
    no un aviso — un estado recortado en silencio invalida la cifra.
    """
    report = length_report(tokenizer, pairs, max_length)
    if report["n_truncated"] and strict:
        raise ScorerContractError(
            f"{report['n_truncated']}/{report['n_pairs']} pairs need "
            f"{report['max_tokens']} tokens and the window holds "
            f"{report['max_length']}: the STATE would be cut in silence. "
            f"Shorten the state or pass strict=False and publish the "
            f"truncation count next to the figure.")
    enc = tokenizer([p for p, _ in pairs], [h for _, h in pairs],
                    return_tensors="pt", padding=True,
                    truncation=TRUNCATION_STRATEGY, max_length=max_length)
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    else:
        enc = dict(enc)
    return enc, report


def softmax(values: list[float]) -> list[float]:
    hi = max(values)
    exps = [math.exp(v - hi) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


class PairScorer:
    """Puntúa pares (premisa, hipótesis) y devuelve UN escalar por par.

    Es la única frontera entre el contrato de este módulo y el checkpoint
    que lo implementa: los tests del contrato corren contra
    `HashPairScorer` sin descargar un gigabyte, y la medición sin entrenar
    corre contra `NliPairScorer` con los pesos verificados por sha.
    """

    id = "abstract"

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"id": self.id}


class HashPairScorer(PairScorer):
    """Scorer de juguete, determinista y sin pesos.

    `z` es una función hash del par COMPLETO. No mide nada y no aparece en
    ninguna cifra publicada: existe para que los tests del contrato
    —permutación, seguimiento del texto, contexto comparativo— se ejecuten
    en un checkout limpio. Y es exactamente el banco de pruebas correcto
    para ese contrato, porque lo que el contrato promete es DE QUÉ depende
    `z`, y un hash del par entero no deja ninguna dependencia escondida.
    """

    id = "hash-toy"

    def __init__(self, scale: float = 4.0):
        self.scale = float(scale)
        self.calls = 0

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls += 1
        out = []
        for premise, hypothesis in pairs:
            digest = hashlib.sha256(
                (premise + "\x00" + hypothesis).encode("utf-8")).digest()
            unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
            out.append(self.scale * (2.0 * unit - 1.0))
        return out

    def describe(self) -> dict:
        return {"id": self.id, "kind": "deterministic toy, no weights",
                "scale": self.scale,
                "warning": "measures nothing; contract tests only"}


def reduce_logits(class_logits, mode: str, entail_index: int) -> list[float]:
    """De `[N, C]` logits del cabezal NLI a un escalar por par."""
    if mode not in SCORE_MODES:
        raise ValueError(f"unknown score mode {mode!r}; {SCORE_MODES}")
    import torch
    if mode == ENTAIL_LOGIT:
        col = class_logits[:, entail_index]
    else:
        col = torch.log_softmax(class_logits, dim=-1)[:, entail_index]
    return [float(x) for x in col]


class NliPairScorer(PairScorer):
    """Un checkpoint NLI real, cargado de `artifacts/weights/<id>/`.

    Los pesos se verifican por sha-256 antes de cargarse
    (`model.weights.require_verified`); un hash que no cuadra es un
    rechazo, no un aviso. El índice de la clase `entailment` se LEE del
    `config.json` del checkpoint — nunca se asume que sea 0.
    """

    def __init__(self, weight_id: str, device: str = "cpu",
                 mode: str = DEFAULT_SCORE_MODE,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 max_length: int = DEFAULT_MAX_LENGTH,
                 strict_length: bool = True):
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)

        from .weights import require_verified, spec_for

        if mode not in SCORE_MODES:
            raise ValueError(f"unknown score mode {mode!r}; {SCORE_MODES}")
        self.id = weight_id
        self.spec = spec_for(weight_id)
        self.mode = mode
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.strict_length = bool(strict_length)
        path = require_verified(weight_id)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, dtype=torch.float32)
        self.model.eval().to(self.device)
        self.entail_index = _entailment_index(self.model.config)
        self.n_classes = int(self.model.config.num_labels)
        self.params = sum(p.numel() for p in self.model.parameters())
        self.pairs_scored = 0
        self.seconds = 0.0
        self.truncated_pairs = 0

    def class_logits(self, pairs: list[tuple[str, str]]):
        """`[N, C]` sobre CPU, por lotes de `batch_size` (medio gas)."""
        import torch
        chunks = []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start:start + self.batch_size]
                enc, report = encode_pairs(
                    self.tokenizer, batch, self.device, self.max_length,
                    strict=self.strict_length)
                self.truncated_pairs += report["n_truncated"]
                chunks.append(self.model(**enc).logits.float().cpu())
        if self.device.type == "mps":
            torch.mps.synchronize()
        self.seconds += time.perf_counter() - t0
        self.pairs_scored += len(pairs)
        return torch.cat(chunks, dim=0) if chunks else torch.zeros(0, 1)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        return reduce_logits(self.class_logits(pairs), self.mode,
                             self.entail_index)

    def describe(self) -> dict:
        return {
            "id": self.id,
            "repo": self.spec["repo"],
            "revision": self.spec["revision"],
            "license": self.spec["license"],
            "params": self.params,
            "n_classes": self.n_classes,
            "entail_index": self.entail_index,
            "mode": self.mode,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "truncation": TRUNCATION_STRATEGY,
            "strict_length": self.strict_length,
            "truncated_pairs": self.truncated_pairs,
            "hypothesis_format_id": HYPOTHESIS_FORMAT_ID,
            "contaminated_benchmarks": list(
                self.spec.get("contaminated_benchmarks", ())),
        }


def _entailment_index(config) -> int:
    """El índice de la clase `entailment`, leído del checkpoint."""
    id2label = getattr(config, "id2label", None) or {}
    for key, label in id2label.items():
        if str(label).strip().lower() == "entailment":
            return int(key)
    raise ScorerContractError(
        f"checkpoint has no `entailment` class; id2label={id2label}")


class CrossEncoderScorer:
    """El scorer compartido: K pasadas por el mismo modelo, un softmax."""

    def __init__(self, pair_scorer: PairScorer):
        self.pair_scorer = pair_scorer

    def score(self, decision: Decision) -> dict:
        return self.score_many([decision])[0]

    def score_many(self, decisions: list[Decision]) -> list[dict]:
        """Todas las decisiones en UNA tanda de pares.

        El agrupado es lo que hace que esto corra en el Mac: las K pasadas
        de cada pregunta y las de todas las preguntas entran en el mismo
        lote, y el softmax se aplica después, por pregunta.
        """
        flat, spans = flatten_pairs(decisions)
        scores = self.pair_scorer.score_pairs(flat)
        if len(scores) != len(flat):
            raise ScorerContractError(
                f"pair scorer returned {len(scores)} scores for "
                f"{len(flat)} pairs")

        out = []
        for decision, (lo, hi) in zip(decisions, spans):
            z = scores[lo:hi]
            probs = softmax(z)
            ids = [c.id for c in decision.candidates]
            order = sorted(range(len(ids)), key=lambda i: -z[i])
            argmax_id = ids[order[0]]
            out.append({
                "candidate_ids": ids,
                "z": z,
                "probs": probs,
                "argmax_id": argmax_id,
                "ranking": [ids[i] for i in order],
                "k": decision.k,
                "family": decision.family,
                "lang": decision.lang,
                "comparative_context": decision.wants_context(),
                "gold": decision.gold,
                "correct": (None if decision.gold is None
                            else argmax_id == decision.gold),
                "nll": (None if decision.gold is None
                        else -math.log(max(probs[ids.index(decision.gold)],
                                           1e-12))),
            })
        return out


def scalar_output_report(scorer: PairScorer) -> dict:
    """¿Es de verdad una salida escalar sin neurona por etiqueta?

    Lo comprueba estructuralmente, como `label_free_report()` en
    `model/decision_head.py`: recorre los parámetros del checkpoint y
    exige que ninguna dimensión dependa del catálogo de candidatos. Lo
    único que sale del cabezal es una columna, la de `entailment`.
    """
    report = {"scorer": scorer.describe(), "scalar_output": True,
              "label_free": True, "offenders": []}
    model = getattr(scorer, "model", None)
    if model is None:  # el juguete no tiene pesos que auditar
        report["audited_params"] = 0
        return report
    n_classes = int(model.config.num_labels)
    report["n_classes"] = n_classes
    report["reduction"] = (
        f"one column ({getattr(scorer, 'entail_index', '?')}) of a "
        f"{n_classes}-class NLI head; the class count is a property of the "
        f"checkpoint, not of the candidate catalogue")
    # Una neurona por etiqueta sería una dimensión de salida que crece con
    # el catálogo. En el cabezal sólo se admiten anchos del propio
    # checkpoint (el modelo, su capa intermedia) y `n_classes`, que es una
    # propiedad del checkpoint y no del catálogo: 3 para MNLI/XNLI, 2 para
    # los zero-shot binarios, iguales con 3 candidatos que con 300.
    hidden = int(getattr(model.config, "hidden_size", 0) or 0)
    allowed = {n_classes, 1, hidden, 4 * hidden,
               int(getattr(model.config, "intermediate_size", 0) or 0)}
    report["allowed_head_dims"] = sorted(d for d in allowed if d)
    n = 0
    for name, param in model.named_parameters():
        n += 1
        if "classifier" in name or name.startswith("score"):
            if param.dim() >= 1 and int(param.shape[0]) not in allowed:
                report["label_free"] = False
                report["offenders"].append(
                    {"param": name, "shape": list(param.shape)})
    report["audited_params"] = n
    return report


# -- las tres comprobaciones que el gate y los tests comparten -----------

def check_permutation_invariance(scorer: CrossEncoderScorer,
                                 decision: Decision,
                                 max_perms: int = 24) -> dict:
    """Permutar los candidatos no mueve las probabilidades realineadas."""
    import itertools
    base = scorer.score(decision)
    at = dict(zip(base["candidate_ids"], base["probs"]))
    worst, n = 0.0, 0
    for perm in itertools.permutations(decision.candidates):
        if n >= max_perms:
            break
        n += 1
        got = scorer.score(_replace_candidates(decision, list(perm)))
        for cid, p in zip(got["candidate_ids"], got["probs"]):
            worst = max(worst, abs(at[cid] - p))
    return {"max_abs_prob_delta": worst, "tolerance": PERMUTATION_TOL,
            "n_perms": n, "pass": worst <= PERMUTATION_TOL}


def check_text_follows_id(scorer: CrossEncoderScorer,
                          decision: Decision) -> dict:
    """Intercambiar dos textos conservando los ids intercambia sus `z`.

    Es la comprobación de que el scorer lee TEXTO y no índice: si los
    pares se construyeran con la posición, las puntuaciones se quedarían
    donde estaban.
    """
    cands = list(decision.candidates)
    a, b = cands[0], cands[1]
    base = scorer.score(decision)
    zb = dict(zip(base["candidate_ids"], base["z"]))
    swapped = [Candidate(a.id, b.text), Candidate(b.id, a.text)] + cands[2:]
    got = scorer.score(_replace_candidates(decision, swapped))
    zs = dict(zip(got["candidate_ids"], got["z"]))
    delta = max(abs(zs[a.id] - zb[b.id]), abs(zs[b.id] - zb[a.id]))
    # y que de verdad se han movido: si `z_a == z_b` de partida el test no
    # distinguiría «sigue al texto» de «no ha pasado nada».
    separation = abs(zb[a.id] - zb[b.id])
    return {"swapped_ids": [a.id, b.id], "max_abs_z_delta": delta,
            "base_separation": separation, "tolerance": PERMUTATION_TOL,
            "pass": delta <= PERMUTATION_TOL and separation > PERMUTATION_TOL}


def check_comparative_context(scorer: CrossEncoderScorer, decision: Decision,
                              foreign_id: str, foreign_text: str) -> dict:
    """Las DOS direcciones del contrato, sobre el MISMO caso.

    Cambia el texto de un candidato AJENO y mira la puntuación del
    evaluado: con el contexto activo tiene que moverse (el estado de los
    demás es parte de la premisa), y con el contexto desactivado no puede
    moverse en absoluto.
    """
    target_id = next(c.id for c in decision.candidates if c.id != foreign_id)
    out = {"target_id": target_id, "foreign_id": foreign_id,
           "tolerance": PERMUTATION_TOL}
    for on in (True, False):
        base_dec = _with_context(decision, on)
        altered = _replace_candidates(base_dec, [
            Candidate(c.id, foreign_text) if c.id == foreign_id else c
            for c in base_dec.candidates])
        z0 = dict(zip(*_zs(scorer, base_dec)))[target_id]
        z1 = dict(zip(*_zs(scorer, altered)))[target_id]
        out["on" if on else "off"] = {"z_before": z0, "z_after": z1,
                                      "abs_delta": abs(z1 - z0)}
    out["pass"] = (out["on"]["abs_delta"] > PERMUTATION_TOL
                   and out["off"]["abs_delta"] <= PERMUTATION_TOL)
    return out


def _zs(scorer: CrossEncoderScorer, decision: Decision):
    got = scorer.score(decision)
    return got["candidate_ids"], got["z"]


def _replace_candidates(decision: Decision,
                        candidates: list[Candidate]) -> Decision:
    import dataclasses
    return dataclasses.replace(decision, candidates=tuple(candidates))


def _with_context(decision: Decision, on: bool) -> Decision:
    import dataclasses
    return dataclasses.replace(decision, comparative_context=on)


#: El caso que el CLI y los tests usan para las tres comprobaciones: la
#: familia de comparación, donde la información decisiva vive DENTRO de
#: los candidatos y el contexto comparativo es obligatorio.
CONTRACT_CASE = Decision(
    state=("Catalogue: two lamps are on offer this week and the shop will "
           "ship either of them tomorrow."),
    question="Which lamp lasts longer?",
    candidates=(
        Candidate("c1", "Lamp Aurora: 20 euros, lasts 2 years"),
        Candidate("c2", "Lamp Borealis: 50 euros, lasts 5 years"),
        Candidate("c3", "Lamp Caldera: 35 euros, lasts 3 years"),
    ),
    family=COMPARISON,
    lang="en",
    gold="c2",
)
CONTRACT_FOREIGN_ID = "c3"
CONTRACT_FOREIGN_TEXT = "Lamp Caldera: 35 euros, lasts 9 years"


def run_contract(write: bool = True,
                 pair_scorer: PairScorer | None = None) -> dict:
    """Las tres comprobaciones del `Verification gate`, sin pesos."""
    ps = pair_scorer or HashPairScorer()
    scorer = CrossEncoderScorer(ps)
    checks = {
        "permutation_invariance": check_permutation_invariance(
            scorer, CONTRACT_CASE),
        "text_follows_id": check_text_follows_id(scorer, CONTRACT_CASE),
        "comparative_context": check_comparative_context(
            scorer, CONTRACT_CASE, CONTRACT_FOREIGN_ID, CONTRACT_FOREIGN_TEXT),
    }
    report = {
        "task": TASK,
        "pass": all(c["pass"] for c in checks.values()),
        "scope": ("the CONTRACT only — rendering, permutation, text "
                  "following and comparative context. Measures no quality: "
                  "the untrained figure is eval/ce_nograd.py"),
        "pair_scorer": ps.describe(),
        "scalar_output": scalar_output_report(ps),
        "hypothesis_format": HYPOTHESIS,
        "premise_format": PREMISE,
        "comparative_context_by_family": COMPARATIVE_CONTEXT,
        "checks": checks,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if write:
        os.makedirs(GATE_DIR, exist_ok=True)
        with open(os.path.join(GATE_DIR, "contract.json"), "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
    return report


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "contract"
    if cmd != "contract":
        print(f"unknown command {cmd!r}; use contract", file=sys.stderr)
        return 2
    report = run_contract()
    for name, check in report["checks"].items():
        print(f"[contract] {name}: {'PASS' if check['pass'] else 'FAIL'}")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"},
                     indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
