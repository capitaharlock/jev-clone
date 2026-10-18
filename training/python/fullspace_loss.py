"""La pérdida sobre el espacio entero — exacta y muestreada (#T-fullspace-objective).

`train_decision.py` entrega hoy `cross_entropy(logits, gold)` sobre las
`[K + 1]` columnas que la cabeza devolvió para esa fila. Eso **ya** es un
softmax normalizado sobre lo que se le dé: cuando las columnas son el
espacio entero de la fila, la vía exacta no necesita ninguna fórmula nueva
(es lo que midió `#T-bigk-optsets`). Este módulo escribe lo que la vía
exacta no puede hacer — normalizar sobre un espacio que no se enumera — y
lo escribe con su ecuación, su estimador y sus tests.

La ecuación
-----------

Para una fila `i` con estado `s`, pregunta `q`, espacio de etiquetas `S`
y oro `y ∈ S ∪ {unknown}`, la cabeza produce un logit por columna,
`z_c = head(s, q, texto(c) | C)` — condicionado al CONJUNTO `C` que se le
pasó, porque `model/decision_head.py:105-110` hace atención entre opciones.
La pérdida objetivo es el softmax sobre el espacio COMPLETO:

    L = - z_y + log( e^{z_unk} + Σ_{c ∈ S} e^{z_c} )                    (1)

Separando el oro del resto, con `R = Σ_{c ∈ S \\ {y}} e^{z_c}` la *masa
negativa*:

    L = - z_y + log( e^{z_y} + e^{z_unk} + R )                          (2)

**Modo exacto** (`S` enumerable y cabe): `R` se calcula entero, (2) es
idéntica a `cross_entropy` sobre esas columnas, y `test_fullspace_loss`
lo verifica en VALOR y en GRADIENTE contra un softmax completo escrito a
mano. No hay estimador y no hay corrección: `log Q ≡ 0`.

**Modo muestreado** (`S` no enumerable, o no cabe): se sustituye `R` por
un estimador `R̂` construido sobre un subconjunto de candidatos, y la
corrección `log Q` es lo que endereza ese estimador **bajo la condición
que la sección «Qué se demuestra y sobre qué» escribe entera**. La
corrección se aplica como un desplazamiento por columna sobre los logits,

    z'_c = z_c - logQ_c ,   logQ_y = 0 ,   logQ_unk = 0                 (3)

y entonces `cross_entropy(z', y)` es exactamente (2) con `R̂` en lugar de
`R`. Esa es toda la implementación: `sampled_loss` = `exact_loss` con un
vector de desplazamiento.

Dos esquemas que NO son intercambiables
---------------------------------------

`logQ_c` es una cosa distinta según cómo se haya construido el conjunto, y
el error que este módulo existe para impedir es usar la fórmula de uno con
el diseño muestral del otro.

* `PROPOSAL` — `m` extracciones i.i.d. **con reemplazo** de una
  distribución de propuesta `Q` sobre el pool de negativos. Entonces

      R̂ = Σ_{j=1..m} e^{z_{c_j}} / (m · Q(c_j))   ⇒   logQ_c = log(m·Q(c))

  y `E[R̂] = R` para valores `e^{z_c}` fijos respecto al muestreo. Los
  duplicados extraídos **cuentan**: quitarlos rompe esa identidad (por eso
  el filtrado de colisiones se hace sobre el POOL antes de muestrear,
  nunca sobre la muestra).

* `INCLUSION` — un subconjunto `N` donde cada candidato entra con
  probabilidad de inclusión `π(c)` (Horvitz-Thompson). Entonces

      R̂ = Σ_{c ∈ N} e^{z_c} / π(c)               ⇒   logQ_c = log π(c)

  y `E[R̂] = R`, otra vez para valores fijos respecto al muestreo. Éste es
  el caso de los **negativos in-batch**:
  el conjunto de etiquetas únicas de un batch es una muestra
  DEDUPLICADA, y para el diseño «`m` extracciones con reemplazo, luego
  únicas» la inclusión es `π(c) = 1 - (1 - Q(c))^m`, que **no** es
  `m · Q(c)`. Usar `m·Q` sobre un conjunto deduplicado da un estimador
  sesgado; `test_fullspace_loss` mide ese sesgo en vez de afirmarlo.

Qué se demuestra y sobre qué
----------------------------

**La condición, primero, porque es la que este head NO cumple.** Las dos
identidades de arriba —y la de Horvitz–Thompson en particular— exigen que
los valores sumados sean FIJOS respecto al muestreo: `e^{z_c}` tiene que
ser el mismo número entre en el conjunto quien entre. Aquí no lo es. La
ecuación (1) ya lo dice: `z_c = head(s, q, texto(c) | C)` está
condicionado al conjunto, porque `model/decision_head.py:105-110` hace
atención entre opciones. Cambiar `C` cambia todos los logits, el del oro y
el de `unknown` incluidos, y `test_logits_depend_on_the_candidate_set` lo
mide sobre la cabeza real en vez de suponerlo.

De ahí sale lo que este módulo puede y no puede afirmar:

* `E[R̂] = R` está demostrado para logits **independientes del conjunto**.
  Los tests de `test_fullspace_loss` enumeran el diseño muestral con
  logits FIJOS: verifican el álgebra del estimador y la distinción entre
  los dos esquemas, que es para lo que existen. **No dicen nada** sobre
  los logits que produce `PointerDecisionHead`, y citarlos como si lo
  dijeran es exactamente el error que esta sección existe para impedir.
* Con `set_attention=True` —la cabeza publicada— `R̂` NO es un estimador
  insesgado de la masa negativa calculada sobre el espacio completo, y
  este módulo no lo afirma. Lo que queda es **otro objetivo**: una
  pérdida sobre logits condicionados al conjunto muestreado, sin garantía
  de insesgamiento heredada del diseño muestral. Es utilizable y es
  medible; no es una solución matemáticamente acreditada del problema del
  denominador, y no puede presentarse como tal.
* La ablación `set_attention=False` (logits independientes del conjunto,
  `CrossBlock` sin atención entre opciones) SÍ cumple la condición. Es un
  cambio de arquitectura, no la misma cabeza con otro denominador: quien
  quiera el argumento de insesgamiento tiene que pagar ese precio y
  medirlo.

**La PÉRDIDA no es insesgada en ningún caso, y subestima**: el denominador
entra dentro de un `log`, que es cóncavo, así que por Jensen

    E[L̂] = -z_y + E[log D̂] ≤ -z_y + log E[D̂] = L                      (4)

con igualdad sólo cuando `D̂` es determinista (el caso `π ≡ 1`, que es el
modo exacto). Es decir: el objetivo muestreado es **optimista** — reporta
menos pérdida de la que la fila tiene sobre el espacio entero, y la brecha
se cierra al crecer la muestra (consistencia). El argumento de Jensen
tampoco se hereda gratis: presupone `E[D̂] = D`, que es la misma condición
de arriba. Prometer insesgamiento de la pérdida sería falso, y decir que
la sobreestima también: el test mide el signo, no lo asume.

Y el aviso de alcance: el piloto de recuperación (`#cross-encoder-pilot`)
NO necesita esta pérdida. Su objetivo es CE sobre los candidatos
ofrecidos, donde el denominador es el conjunto y no hay nada que
estimar. Este módulo es la especificación de la vía muestreada de
`#T-fullspace-objective`, que cerró en NO-GO; no está en el camino del
piloto y nada de lo de aquí debe entrar allí por inercia.

Cuándo se omite `log Q`, y por qué
----------------------------------

Sólo tres casos, y cada uno se declara en el artefacto (`OMISSIONS`):

1. **El conjunto ES el espacio** (`π ≡ 1`): `log π = 0`. No es una omisión,
   es el valor correcto — el régimen de `#T-bigk-optsets`.
2. **La tarea medida es el ranking DENTRO del conjunto ofrecido** y no
   sobre el espacio: entonces (1) no es el objetivo y no hay nada que
   corregir. Es el régimen de fase 1 (K ∈ [3,8]) y es exactamente la razón
   por la que sus cifras no son un normalizador sobre el espacio.
3. **`Q` uniforme sobre un pool de tamaño fijo Y esquema `PROPOSAL`**: la
   corrección es la misma constante en todas las columnas de candidato. NO
   se cancela (el oro no lleva corrección), así que omitirla SÍ cambia la
   pérdida; se declara como omisión deliberada con su motivo o no se omite.

`prior_penalty` NO es una corrección log-Q
------------------------------------------

`train_decision.py:741-784` resta `w · log(cuenta empírica de la etiqueta
como respuesta en el train)`. Difiere de (3) en las tres cosas que
importan: depende de la frecuencia de la etiqueta **como oro en el
corpus**, no de su probabilidad de haber sido **muestreada como negativo
para esta fila**; lleva un peso libre `w`; y se aplica también a la columna
del oro. No hace insesgado ningún estimador de `R` y `test_fullspace_loss`
lo mide.

CLI:
    PYTHONPATH=. .venv-train/bin/python -m training.python.fullspace_loss spec
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK = "T-fullspace-objective"
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", TASK)

#: esquemas de corrección. El nombre viaja en el artefacto: un conjunto de
#: candidatos sin esquema declarado no se puede auditar después.
PROPOSAL = "proposal"
INCLUSION = "inclusion"
EXACT = "exact"
SCHEMES = (EXACT, PROPOSAL, INCLUSION)

#: los tres motivos por los que `log Q` puede faltar, y ninguno más
OMISSIONS = {
    "set_is_space": (
        "the candidate set IS the row's whole space, so every inclusion "
        "probability is 1 and log Q = 0. Not an omission: the correct value"),
    "in_set_ranking": (
        "the measured task is the ranking INSIDE the offered set, not over "
        "the space; equation (1) is not the objective, so there is nothing "
        "to correct. This is the phase-1 regime (K in [3, 8]) and it is why "
        "its numbers are not a normaliser over the space"),
    "declared_uniform": (
        "Q is uniform over a fixed pool under the PROPOSAL scheme, so the "
        "correction is one constant on every candidate column. It does NOT "
        "cancel (the gold column carries none), so dropping it changes the "
        "loss: it is a deliberate, declared omission or it is not made"),
}


class LogQError(ValueError):
    """La corrección no cuadra con el esquema declarado."""


@dataclass(frozen=True)
class CandidateSet:
    """Las columnas que se le entregan a la pérdida para UNA fila.

    `keys` identifica cada columna de opción (el id de etiqueta, o el texto
    normalizado cuando el pool cruza espacios); `gold` indexa `keys`, o vale
    `len(keys)` para una fila `unknown`. `log_q` es el vector de la ecuación
    (3) **sólo sobre las columnas de opción** — esta clase le añade el cero
    del `unknown` y fuerza el cero del oro.
    """

    keys: list[str]
    gold: int
    scheme: str = EXACT
    log_q: list[float] | None = None
    #: extracciones del esquema PROPOSAL (m de la ecuación); 1 en los demás
    m: int = 1
    #: motivo, de `OMISSIONS`, cuando `log_q is None` y el esquema no es exacto
    omission: str = ""
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        k = len(self.keys)
        if k == 0:
            raise ValueError("a row needs at least one option column")
        if not 0 <= self.gold <= k:
            raise ValueError(f"gold {self.gold} outside [0, {k}]")
        if self.scheme not in SCHEMES:
            raise LogQError(f"unknown scheme {self.scheme!r}; {SCHEMES}")
        if self.m < 1:
            raise ValueError("m (proposal draws) must be >= 1")
        if self.scheme == EXACT:
            if self.log_q is not None:
                raise LogQError(
                    "the exact mode has no correction: log Q = 0 by "
                    "construction. Pass log_q=None")
        elif self.log_q is None:
            if self.omission not in OMISSIONS:
                raise LogQError(
                    f"a sampled set without log_q must declare WHY, one of "
                    f"{sorted(OMISSIONS)}; got {self.omission!r}")
        elif len(self.log_q) != k:
            raise LogQError(
                f"log_q has {len(self.log_q)} entries for {k} option columns")

    @property
    def k(self) -> int:
        return len(self.keys)

    @property
    def is_unknown(self) -> bool:
        """¿El oro es la columna `unknown`?"""
        return self.gold == self.k

    def shift(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """El vector `[K + 1]` de la ecuación (3), con sus ceros forzados.

        El oro y el `unknown` nunca llevan corrección: el oro no se muestreó
        (esquema PROPOSAL) o se incluyó con probabilidad 1 (INCLUSION), y el
        `unknown` es una columna estructural de la cabeza, no un candidato.
        """
        if self.log_q is None:
            return torch.zeros(self.k + 1, device=device, dtype=dtype)
        out = torch.tensor(list(self.log_q) + [0.0], device=device,
                           dtype=dtype)
        if not self.is_unknown:
            out[self.gold] = 0.0
        return out

    def declare(self) -> dict:
        """Lo que va al artefacto: el régimen de esta pérdida, no un número."""
        return {
            "scheme": self.scheme,
            "k_columns": self.k,
            "gold_is_unknown": self.is_unknown,
            "log_q_applied": self.log_q is not None,
            "log_q_omitted_because": (
                "" if self.log_q is not None
                else OMISSIONS.get(self.omission, self.omission)),
            "proposal_draws_m": self.m if self.scheme == PROPOSAL else None,
            "estimator": ESTIMATORS[self.scheme],
            "collisions": dict(self.stats),
        }


ESTIMATORS = {
    EXACT: ("R is summed over the whole space: no estimator, no correction, "
            "and equation (2) is `cross_entropy` over those columns"),
    PROPOSAL: ("R_hat = sum_j exp(z_j) / (m * Q(c_j)) over m i.i.d. draws "
               "WITH replacement; E[R_hat] = R FOR VALUES FIXED WITH "
               "RESPECT TO THE SAMPLING (see `unbiasedness.precondition`). "
               "Duplicates count — removing a drawn duplicate breaks it"),
    INCLUSION: ("R_hat = sum_{c in N} exp(z_c) / pi(c) (Horvitz-Thompson); "
                "E[R_hat] = R under the same precondition. For `m draws "
                "with replacement, then unique` — the in-batch case — "
                "pi(c) = 1 - (1 - Q(c))^m, which is NOT m * Q(c)"),
}


# -- las dos correcciones, cada una con su diseño muestral ------------------

def proposal_log_q(q: list[float], m: int) -> list[float]:
    """`log(m · Q(c))` — esquema PROPOSAL, `m` extracciones con reemplazo."""
    if m < 1:
        raise ValueError("m must be >= 1")
    if any(p <= 0.0 for p in q):
        raise LogQError("a candidate with Q(c) = 0 cannot have been drawn; "
                        "its correction is -inf and the estimator is undefined")
    return [math.log(m * p) for p in q]


def inclusion_log_q(pi: list[float]) -> list[float]:
    """`log π(c)` — esquema INCLUSION (Horvitz-Thompson)."""
    if any(not 0.0 < p <= 1.0 for p in pi):
        raise LogQError("an inclusion probability must be in (0, 1]")
    return [math.log(p) for p in pi]


def inclusion_from_with_replacement(q: list[float], m: int) -> list[float]:
    """`π(c) = 1 - (1 - Q(c))^m` — el diseño de los negativos in-batch.

    Las etiquetas únicas de un batch son una muestra CON REEMPLAZO
    deduplicada: cada fila aporta su oro (y sus opciones) y las repeticiones
    colapsan. Ésta es la inclusión de ese diseño, y es la que
    `inclusion_log_q` necesita. `m · Q(c)` sería la de PROPOSAL sin
    deduplicar, y sobre un conjunto deduplicado sesga — que es lo que
    `test_fullspace_loss::test_proposal_correction_on_deduped_set_is_biased`
    mide.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    return [1.0 - (1.0 - p) ** m for p in q]


# -- colisiones y falsos negativos -----------------------------------------

def normalise_key(text: str) -> str:
    """Clave de identidad de una etiqueta: minúsculas, sin puntuación blanda."""
    out = "".join(c if c.isalnum() else " " for c in text.lower())
    return " ".join(out.split())


def filter_pool(gold_key: str, pool: list[str], q: list[float] | None = None,
                positives=(), normalise=normalise_key) -> tuple:
    """Limpia el POOL de negativos ANTES de muestrear, y renormaliza `Q`.

    Tres cosas se van, y las tres se cuentan:

    * el **oro**, por clave normalizada — una etiqueta de otro espacio con
      el mismo texto que el oro de la fila no es un negativo, es el oro;
    * los **positivos declarados** de la fila (multi-etiqueta, o un espacio
      que ofrece sinónimos): falsos negativos DETECTABLES;
    * los **duplicados** de clave dentro del pool, que en un pool cruzado
      entre espacios aparecen solos (`card_arrival` en dos taxonomías).

    Se hace aquí, sobre el pool, y NO sobre la muestra extraída: bajo
    PROPOSAL quitar un duplicado ya extraído rompe el insesgamiento, porque
    el estimador cuenta las `m` extracciones tal cual salieron.

    Lo que este filtro NO puede hacer se declara en vez de esconderse: un
    falso negativo **semántico** entre espacios (una etiqueta ajena que sí
    sería correcta para la fila pero se escribe distinto) no es detectable
    por identidad de texto y queda como sesgo declarado del estimador.

    Devuelve `(keys, q_renormalizada_o_None, stats)`.
    """
    if q is not None and len(q) != len(pool):
        raise ValueError("q must have one entry per pool candidate")
    gold_n = normalise(gold_key)
    pos_n = {normalise(p) for p in positives}
    keys, kept_q, seen = [], [], set()
    gold_hits = pos_hits = dups = 0
    for i, key in enumerate(pool):
        n = normalise(key)
        if n == gold_n:
            gold_hits += 1
            continue
        if n in pos_n:
            pos_hits += 1
            continue
        if n in seen:
            dups += 1
            continue
        seen.add(n)
        keys.append(key)
        if q is not None:
            kept_q.append(q[i])
    stats = {
        "pool": len(pool),
        "kept": len(keys),
        "gold_collisions": gold_hits,
        "declared_positive_collisions": pos_hits,
        "duplicate_keys": dups,
        "filtered_where": "the pool, before sampling — never the drawn sample",
        "residual_false_negatives": (
            "SEMANTIC false negatives across spaces are not detectable by "
            "text identity and remain a declared bias of the estimator"),
    }
    if q is None:
        return keys, None, stats
    total = sum(kept_q)
    if total <= 0.0:
        raise LogQError("every candidate was filtered out: Q cannot be "
                        "renormalised over an empty pool")
    return keys, [p / total for p in kept_q], stats


# -- la pérdida ------------------------------------------------------------

def exact_loss(logits: torch.Tensor, gold: int) -> torch.Tensor:
    """Ecuación (2) con `R` sumada entera. `log Q = 0` por construcción.

    `logits` es `[K + 1]`, la última columna es `unknown`, y las `K`
    primeras SON el espacio de la fila. Idéntica a `cross_entropy` — eso es
    el punto, y `test_fullspace_loss` lo fija en valor y en gradiente contra
    un softmax completo escrito a mano.
    """
    if logits.dim() != 1:
        raise ValueError("exact_loss takes one row: logits is [K + 1]")
    return nn.functional.cross_entropy(
        logits.unsqueeze(0),
        torch.tensor([gold], device=logits.device))


def sampled_loss(logits: torch.Tensor, cands: CandidateSet) -> torch.Tensor:
    """Ecuación (2) con `R̂` en lugar de `R`: (3) y después `cross_entropy`.

    Con `π ≡ 1` sobre el pool entero el desplazamiento es cero y esto
    devuelve `exact_loss` bit a bit — el puente entre los dos modos que
    `test_fullspace_loss::test_inclusion_with_pi_one_is_exact` verifica en
    valor y en gradiente.
    """
    if logits.dim() != 1:
        raise ValueError("sampled_loss takes one row: logits is [K + 1]")
    if logits.shape[0] != cands.k + 1:
        raise ValueError(
            f"logits has {logits.shape[0]} columns for {cands.k} candidates "
            "plus `unknown`")
    shift = cands.shift(device=logits.device, dtype=logits.dtype)
    return exact_loss(logits - shift, cands.gold)


def sampled_loss_batch(logits: torch.Tensor, shift: torch.Tensor,
                       gold: torch.Tensor) -> torch.Tensor:
    """La misma (3) sobre el eje acolchado `[B, K_max + 1]` del trainer.

    `shift` trae ya sus ceros en el oro y en el `unknown`; las columnas de
    relleno llegan a `forward_batch` como `-inf` y su desplazamiento debe
    ser 0 para que `-inf - 0` siga siendo `-inf` (restar `inf` daría NaN).
    """
    if logits.shape != shift.shape:
        raise ValueError("shift must match the logits block")
    if torch.isinf(shift).any() or torch.isnan(shift).any():
        raise LogQError("a non-finite correction: a padded column must "
                        "carry 0, not -inf")
    return nn.functional.cross_entropy(logits - shift, gold)


def negative_mass_estimate(logits: torch.Tensor,
                           cands: CandidateSet) -> torch.Tensor:
    """`R̂` sola — la cantidad sobre la que se enuncia `E[R̂] = R`.

    Es la pérdida menos el oro y el `unknown`: aislarla es lo que permite
    testear `E[R̂] = R` exactamente por enumeración, en vez de por Monte
    Carlo con una tolerancia inventada. Esa enumeración se hace con
    logits FIJOS y demuestra el álgebra del estimador, no una propiedad
    de los logits que devuelve `PointerDecisionHead`: con atención entre
    opciones no son fijos respecto al muestreo (docstring del módulo,
    «Qué se demuestra y sobre qué»).
    """
    shift = cands.shift(device=logits.device, dtype=logits.dtype)
    z = (logits - shift)[:cands.k]
    if not cands.is_unknown:
        keep = [i for i in range(cands.k) if i != cands.gold]
        z = z[torch.tensor(keep, device=logits.device, dtype=torch.long)] \
            if keep else z[:0]
    if z.numel() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return z.exp().sum()


def spec() -> dict:
    """La especificación escrita, para el artefacto del gate."""
    return {
        "task": TASK,
        "artifact": "loss-spec",
        "format": "jev.gate.v1",
        "equation": {
            "objective": "L = -z_y + log( exp(z_unk) + sum_{c in S} exp(z_c) )",
            "split": "L = -z_y + log( exp(z_y) + exp(z_unk) + R ), "
                     "R = sum_{c in S \\ {y}} exp(z_c)",
            "correction": "z'_c = z_c - logQ_c, with logQ_y = 0 and "
                          "logQ_unk = 0; then L_hat = cross_entropy(z', y)",
            "conditioning": (
                "z_c = head(state, question, text(c) | C) is conditioned on "
                "the SET C handed to the head: `model/decision_head.py:105-110` "
                "attends among options. Logits computed over different sets do "
                "not join into one softmax"),
        },
        "modes": {
            EXACT: {
                "when": "the space is enumerable and fits",
                "estimator": ESTIMATORS[EXACT],
                "verified": "value AND gradient against a hand-written full "
                            "softmax (test_exact_matches_full_softmax*)",
            },
            PROPOSAL: {
                "when": "m i.i.d. draws WITH replacement from a proposal Q",
                "log_q": "log(m * Q(c))",
                "estimator": ESTIMATORS[PROPOSAL],
                "verified": "E[R_hat] = R by exact enumeration over all "
                            "m-draw outcomes, not Monte Carlo — with FIXED "
                            "logits (see unbiasedness.precondition)",
            },
            INCLUSION: {
                "when": "a deduplicated subset — the in-batch negatives case",
                "log_q": "log pi(c); for `m draws with replacement, then "
                         "unique`, pi(c) = 1 - (1 - Q(c))^m",
                "estimator": ESTIMATORS[INCLUSION],
                "verified": "E[R_hat] = R by exact enumeration over subsets "
                            "with FIXED logits (see "
                            "unbiasedness.precondition); and the PROPOSAL "
                            "formula on the same deduplicated design is "
                            "measured to be biased",
            },
        },
        "unbiasedness": {
            "precondition": (
                "E[R_hat] = R holds for values exp(z_c) FIXED with respect "
                "to the sampling. The published head violates it: "
                "z_c = head(s, q, text(c) | C) is conditioned on the set "
                "because CrossBlock attends among options, so changing C "
                "changes every logit (measured by "
                "test_logits_depend_on_the_candidate_set). The "
                "set_attention=False ablation satisfies it, at the price of "
                "being another architecture"),
            "R_hat": ("E[R_hat] = R under both schemes ONLY under that "
                      "precondition; the enumeration tests use FIXED logits "
                      "and prove the estimator algebra, not a property of "
                      "the real head's logits. With set attention on, this "
                      "is another objective, without an inherited "
                      "unbiasedness guarantee"),
            "loss": ("NOT unbiased, and it UNDERESTIMATES: the denominator "
                     "sits inside a log, which is concave, so by Jensen "
                     "E[L_hat] <= L, with equality only when the denominator "
                     "estimate is deterministic (pi == 1, the exact mode). "
                     "The sampled objective is optimistic; the gap closes as "
                     "the sample grows (consistency). Jensen is not free "
                     "either: it assumes E[D_hat] = D, the same "
                     "precondition. The test measures the sign rather than "
                     "assuming it"),
        },
        "collisions": {
            "filtered": ["the gold by normalised key",
                         "declared positives of the row",
                         "duplicate keys inside the pool"],
            "where": "on the POOL before sampling — removing a drawn "
                     "duplicate under PROPOSAL breaks unbiasedness",
            "q_renormalised": True,
            "residual": ("semantic false negatives across spaces are not "
                         "detectable by text identity and remain a declared "
                         "bias of the estimator"),
        },
        "log_q_omission": OMISSIONS,
        "prior_penalty_is_not_log_q": {
            "what_it_is": "train_decision.py:741-784 subtracts w * log(count "
                          "of the label as a gold ANSWER in train)",
            "why_not": ["it depends on the label's frequency as a gold in the "
                        "corpus, not on its probability of having been SAMPLED "
                        "as a negative for this row",
                        "it carries a free weight w",
                        "it is applied to the gold column too",
                        "it makes no estimator of R unbiased"],
            "measured": "test_prior_penalty_is_not_a_log_q_correction",
        },
    }


def main(argv: list) -> int:
    if len(argv) > 1 and argv[1] == "spec":
        os.makedirs(GATE_DIR, exist_ok=True)
        path = os.path.join(GATE_DIR, "loss-spec.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(spec(), fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        print(path)
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
