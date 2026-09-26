"""Contrato de episodio versionado (#T-episode-contract).

Un episodio es la unidad auditable de la iniciativa `#episodic-data`: estado
completo, pregunta en prosa natural, candidatos con ID opaco + texto,
respuesta válida, evidencia, familia, idioma, origen, semilla/versión del
generador y grupo de variantes. Todo lo demás —`#T-episode-gen`,
`#T-battery-dev`, `#T-battery-sealed`— consume este contrato.

Lo que este módulo fija, y los tests exigen en vez de suponer:

* **Esquema versionado** (`SCHEMA_VERSION`): los campos obligatorios están en
  `REQUIRED_FIELDS` y `schema_sha()` resume el contrato entero en un hash
  que cada manifest de generación debe registrar (`manifest_record()`).
* **Cinco familias como enumeración cerrada** (`FAMILIES`): los mismos
  valores que `model.ce_scorer.FAMILIES`, con un test que falla si una
  sexta aparece en un lado y no en el otro.
* **Contrato de contexto comparativo por familia** (`COMPARATIVE_CONTEXT`):
  si la información decisiva vive dentro de las opciones, puntuar una sin
  ver las demás pierde el significado de «mejor». `True` = la premisa lleva
  a TODOS los candidatos (bloque idéntico en cada pasada, ordenado por
  texto); `False` = el estado lleva los hechos y cada pasada juzga un solo
  candidato con un contexto de candidatos idéntico (vacío).
* **`canonical_question()` no vale como pregunta**: el corpus convertido
  usa IDs de dataset como pregunta (`training.python.train_decision` los
  reduce a un token estable por dataset). Un episodio con eso en
  `question` es INVÁLIDO, no degradado (`is_dataset_id_question()`).
* **El `variant_group` agrupa un caso con sus contrafactuales**:
  `assign_split()` reparte POR GRUPO (hash estable del grupo + semilla),
  así que ninguna variante del mismo caso cae en dos cortes distintos;
  `check_no_group_split()` lo verifica sobre un reparto ya hecho.

CPU puro, sin torch, sin Qwen, sin red. Validar la fixture de 24
episodios tarda milisegundos.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = "T-episode-contract"

#: Versión del esquema. Cambiar cualquier regla, campo o familia obliga a
#: bump: los episodios de una versión vieja son otro corpus con el mismo
#: nombre, igual que en `data.episodic.EPISODIC_VERSION`.
SCHEMA_VERSION = "episode-v1"

#: Las cinco familias del plan §6 — enumeración cerrada, no texto libre.
#: Canónicas en `model.ce_scorer`; aquí se repiten (data/ no importa model/)
#: y `test_families_match_ce_scorer` impide que diverjan.
EXTRACTION = "extraction_paraphrase"
COMPARISON = "attribute_comparison"
DESCRIPTION = "description_classification"
INFERENCE = "textual_inference_negation"
PRIORITY = "priority_decision"
FAMILIES = (EXTRACTION, COMPARISON, DESCRIPTION, INFERENCE, PRIORITY)

#: Contrato de contexto comparativo, familia por familia y explícito.
#: True = la información decisiva vive DENTRO de las opciones («¿cuál es
#: más barato?» exige ver los dos precios): la premisa del scorer lleva el
#: bloque con TODOS los candidatos. False = los hechos están en el estado
#: y cada pasada juzga un solo candidato con un contexto idéntico.
COMPARATIVE_CONTEXT = {
    EXTRACTION: False,  # el hecho está en el estado; las opciones no aportan
    COMPARISON: True,  # «¿cuál dura más?» no significa nada viendo solo una
    DESCRIPTION: False,  # la definición viaja con cada opción, se juzga sola
    INFERENCE: False,  # entailment del estado contra una sola hipótesis
    PRIORITY: True,  # el criterio ordena candidatos entre sí, y desempata
}

#: Por qué, en una frase por familia — la parte que el generador y el
#: scorer leen para saber qué poner en el estado y qué en el bloque.
COMPARATIVE_WHY = {
    EXTRACTION: "state-carries: the supporting fact is in the state",
    COMPARISON: "context-required: 'best' needs all option attributes",
    DESCRIPTION: "state-carries: each option carries its own definition",
    INFERENCE: "state-carries: one hypothesis judged against the state",
    PRIORITY: "context-required: the criterion ranks candidates jointly",
}

LANGS = ("es", "en")

#: Campos obligatorios por episodio (la task los lista; ni uno más se
#: exige, ni uno menos se acepta). `answer` admite tres formas
#: excluyentes —ver `_check_answer`— así que aquí figura el grupo.
REQUIRED_FIELDS = (
    "state",
    "question",
    "candidates",
    "evidence",
    "family",
    "lang",
    "origin",
    "generator_seed",
    "generator_version",
    "variant_group",
)

#: La misma cola que `training.python.train_decision._QID_TAIL`: el corpus
#: convertido cuelga el índice de fila del ID (`banco-8412`,
#: `epdiv-00001-3`). Una pregunta con esa forma es un ID, no prosa.
_QID_TAIL = re.compile(r"-\d+$")

#: Prefijos de dataset que aparecen como pregunta en el corpus convertido.
_DATASET_PREFIXES = (
    "banking77",
    "massive",
    "huffpost",
    "logiqa",
    "reclor",
    "boolq",
    "dbpedia",
    "clinc",
    "oos",
    "civil",
    "helpsteer",
    "swag",
    "hellaswag",
    "anli",
    "snli",
    "mnli",
    "xnli",
    "epdiv",
    "qwen",
)

_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:~-]{0,63}$")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def is_dataset_id_question(question: Any) -> bool:
    """True si `question` es un ID de dataset, no prosa natural.

    Cubre las dos formas que deja el corpus convertido: el ID de fila con
    cola numérica (`epdiv-00001-3`, `banking77-intent-8412`) y su forma
    canónica tras `canonical_question()` (`banking77-intent`, `massive`),
    además de cualquier token único sin espacios. Una pregunta válida es
    prosa con criterio explícito; esto nunca lo es.
    """
    if not isinstance(question, str):
        return True
    q = question.strip()
    if not q or not re.search(r"\s", q):
        return True
    low = q.lower()
    if _QID_TAIL.search(q):
        return True
    first = re.split(r"[\s:]+", low, maxsplit=1)[0].strip("¿?¡!\"'()[]")
    if first in _DATASET_PREFIXES:
        return True
    if re.fullmatch(r"[a-z0-9_.-]+", first) and "-" in first:
        return True
    return False


def is_opaque_id(candidate_id: Any, text: Any) -> bool:
    """True si el ID del candidato es opaco: no porta su contenido.

    No opaco = el ID ES el texto (`{"id": "Madrid", "text": "Madrid"}`),
    lo contiene entero (un ID que ya dice la respuesta), o ni siquiera
    tiene forma de identificador. El scorer lee texto, nunca IDs.
    """
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        return False
    cid = candidate_id.strip()
    if not _OPAQUE_ID.match(cid):
        return False
    if not isinstance(text, str) or not text.strip():
        return False
    n_id, n_text = _norm(cid), _norm(text)
    if n_id == n_text:
        return False
    if len(n_text) >= 4 and n_text in n_id:
        return False
    return True


def validate(ep: Any) -> list[str]:
    """Valida un episodio. Vacío = válido; si no, un motivo por fallo."""
    reasons: list[str] = []
    if not isinstance(ep, dict):
        return ["episode is not an object"]
    for field in REQUIRED_FIELDS:
        if field not in ep:
            reasons.append(f"missing required field: {field}")
    if reasons:
        return reasons
    if ep.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        reasons.append(
            f"unsupported schema_version: {ep.get('schema_version')!r} "
            f"(contract is {SCHEMA_VERSION})"
        )
    state = ep["state"]
    if not isinstance(state, str) or not state.strip():
        reasons.append("state must be non-empty prose")
    question = ep["question"]
    if not isinstance(question, str) or not question.strip():
        reasons.append("question must be natural prose (>=12 chars)")
    elif is_dataset_id_question(question):
        reasons.append(
            f"question is a dataset id, not prose: {question.strip()!r} "
            "(invalid, not degraded)"
        )
    elif len(question.strip()) < 12:
        reasons.append("question must be natural prose (>=12 chars)")
    elif not question.strip().rstrip("\"')]}").endswith("?"):
        reasons.append("question must be a question (end with '?')")
    evidence = ep["evidence"]
    if not isinstance(evidence, str) or not evidence.strip():
        reasons.append("missing evidence: no justifying fragment given")
    elif isinstance(state, str) and state.strip():
        if _norm(evidence) not in _norm(state):
            reasons.append("evidence is not a fragment of the state")
    group = ep["variant_group"]
    if not isinstance(group, str) or not group.strip():
        reasons.append("missing variant_group: case has no counterfactual key")
    cands = ep["candidates"]
    if not isinstance(cands, list) or len(cands) < 2:
        reasons.append("candidates must be a list of at least 2 (K>=2)")
    else:
        ids: list[str] = []
        for i, c in enumerate(cands):
            if not isinstance(c, dict):
                reasons.append(f"candidates[{i}] is not an object")
                continue
            ids.append(c.get("id"))
            if not is_opaque_id(c.get("id"), c.get("text")):
                reasons.append(f"candidates[{i}].id is not opaque: {c.get('id')!r}")
            elif not isinstance(c.get("text"), str) or not c["text"].strip():
                reasons.append(f"candidates[{i}] has no text to read")
        if len(set(map(str, ids))) != len(ids):
            reasons.append(f"duplicate candidate ids: {ids}")
    reasons.extend(_check_answer(ep, cands))
    if ep["family"] not in FAMILIES:
        reasons.append(f"unknown family {ep['family']!r}; closed: {FAMILIES}")
    if ep["lang"] not in LANGS:
        reasons.append(f"unsupported lang {ep['lang']!r}; {LANGS}")
    if not isinstance(ep["origin"], str) or not ep["origin"].strip():
        reasons.append("origin must be a non-empty string")
    if isinstance(ep["generator_seed"], bool) or not isinstance(
        ep["generator_seed"], int
    ):
        reasons.append("generator_seed must be an int")
    if (
        not isinstance(ep["generator_version"], str)
        or not ep["generator_version"].strip()
    ):
        reasons.append("generator_version must be a non-empty string")
    return reasons


def _check_answer(ep: dict, cands: Any) -> list[str]:
    """Exactamente una de las tres formas de respuesta, con IDs válidos."""
    ids = [c.get("id") for c in cands] if isinstance(cands, list) else []
    shapes = [
        k
        for k in ("answer", "acceptable_answers", "preference")
        if ep.get(k) is not None
    ]
    if len(shapes) != 1:
        return [
            "answer shape must be exactly one of: answer, "
            f"acceptable_answers, preference (found {shapes})"
        ]
    shape = shapes[0]
    if shape == "answer":
        if ep["answer"] not in ids:
            return [f"answer {ep['answer']!r} is not a candidate id"]
        return []
    if shape == "acceptable_answers":
        acc = ep["acceptable_answers"]
        if not isinstance(acc, list) or not acc or not all(a in ids for a in acc):
            return [f"acceptable_answers must be a non-empty subset of {ids}"]
        return []
    pref = ep["preference"]
    if not isinstance(pref, dict):
        return ["preference must be {order: [...], criterion: str}"]
    order, criterion = pref.get("order"), pref.get("criterion")
    if not isinstance(order, list) or sorted(map(str, order)) != sorted(map(str, ids)):
        return [f"preference.order must rank all candidate ids {ids}"]
    if not isinstance(criterion, str) or not criterion.strip():
        return ["preference.criterion must state the explicit criterion"]
    return []


def is_valid(ep: Any) -> bool:
    """Azúcar sobre `validate`: True si no hay ningún motivo de rechazo."""
    return validate(ep) == []


def batch_validate(episodes: list[dict]) -> dict:
    """Valida un lote: por episodio + unicidad de IDs de episodio."""
    seen: dict[str, int] = {}
    per_episode = [validate(ep) for ep in episodes]
    for i, ep in enumerate(episodes):
        eid = ep.get("id") if isinstance(ep, dict) else None
        if isinstance(eid, str) and eid.strip():
            seen.setdefault(eid.strip(), []).append(i)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    n_valid = sum(1 for r in per_episode if not r)
    return {
        "n": len(episodes),
        "n_valid": n_valid,
        "n_invalid": len(episodes) - n_valid,
        "per_episode": per_episode,
        "duplicate_ids": sorted(dupes),
    }


def context_block(candidates: list[dict], lang: str) -> str:
    """El bloque con TODOS los candidatos, idéntico en las K pasadas.

    Ordenado por texto —no por id ni por posición— para que sea el mismo
    string se publiquen los candidatos en el orden que se publiquen. Los
    IDs no entran: son opacos y el scorer no debe poder leerlos. Las
    familias con `COMPARATIVE_CONTEXT[family] is True` lo llevan en la
    premisa; las demás no.
    """
    header = "Opciones en comparación:" if lang == "es" else "Options under comparison:"
    lines = sorted(c["text"].strip() for c in candidates)
    return header + "\n" + "\n".join(f"- {t}" for t in lines)


def wants_context(family: str) -> bool:
    """El contrato de la familia: ¿la premisa lleva a todos los candidatos?"""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; closed: {FAMILIES}")
    return COMPARATIVE_CONTEXT[family]


def assign_split(
    variant_group: str,
    seed: int,
    cuts: tuple[tuple[str, float], ...],
) -> str:
    """Reparte POR GRUPO DE VARIANTES: hash estable grupo+semilla → corte.

    `cuts` son pares (nombre, peso) en orden; los pesos se normalizan. El
    mismo grupo con la misma semilla cae siempre en el mismo corte, y dos
    grupos distintos se reparten ~proporcional a los pesos. Los episodios
    de un grupo NUNCA se separan porque el reparto no ve episodios, solo
    grupos — esa es toda la garantía, y `check_no_group_split` la audita.
    """
    if not isinstance(variant_group, str) or not variant_group.strip():
        raise ValueError("assign_split needs a non-empty variant_group")
    names = [n for n, _ in cuts]
    weights = [float(w) for _, w in cuts]
    if len(names) < 2 or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(f"cuts must be >=2 positive weights: {cuts!r}")
    total = sum(weights)
    digest = hashlib.sha256(
        f"{seed}\x00{variant_group.strip()}".encode("utf-8")
    ).hexdigest()
    point = int(digest, 16) / 16 ** len(digest)
    acc = 0.0
    for (name, _), w in zip(cuts, weights):
        acc += w / total
        if point < acc:
            return name
    return names[-1]


def split_by_group(
    episodes: list[dict],
    seed: int,
    cuts: tuple[tuple[str, float], ...],
) -> dict[str, list[dict]]:
    """Aplica `assign_split` al grupo de cada episodio y agrupa por corte."""
    out: dict[str, list[dict]] = {name: [] for name, _ in cuts}
    for ep in episodes:
        out[assign_split(ep["variant_group"], seed, cuts)].append(ep)
    return out


def check_no_group_split(assignment: dict[str, str]) -> list[str]:
    """Audita un reparto {episode_id: cut}: un grupo en dos cortes = fallo.

    El grupo se lee del ID como prefijo hasta el último `-` cuando el
    episodio no viaja con su grupo; `check_episodes_no_group_split`
    prefiere el campo `variant_group` real.
    """
    return check_episodes_no_group_split(
        [
            {"id": eid, "variant_group": eid.rsplit("-", 1)[0], "split": cut}
            for eid, cut in assignment.items()
        ]
    )


def check_episodes_no_group_split(episodes: list[dict]) -> list[str]:
    """Cada `variant_group` vive en un solo corte; si no, un motivo por grupo."""
    cuts_of: dict[str, set] = {}
    for ep in episodes:
        group = ep.get("variant_group")
        cut = ep.get("split", ep.get("cut"))
        if group is None or cut is None:
            continue
        cuts_of.setdefault(str(group), set()).add(str(cut))
    return [
        f"variant_group {g!r} split across cuts: {sorted(c)}"
        for g, c in sorted(cuts_of.items())
        if len(c) > 1
    ]


def schema_sha() -> str:
    """Huella del contrato: lo que cada manifest de generación registra."""
    payload = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "required_fields": list(REQUIRED_FIELDS),
            "answer_shapes": ["answer", "acceptable_answers", "preference"],
            "families": list(FAMILIES),
            "comparative_context": dict(COMPARATIVE_CONTEXT),
            "langs": list(LANGS),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_record(
    n_episodes: int,
    seed: int,
    generator_version: str,
    splits: dict[str, int] | None = None,
) -> dict:
    """El registro que cada manifest de generación incrusta (task: sha en
    cada manifest). Reproducible por semilla; sin cómputo pesado."""
    return {
        "contract_task": TASK,
        "schema_version": SCHEMA_VERSION,
        "schema_sha": schema_sha(),
        "n_episodes": n_episodes,
        "seed": seed,
        "generator_version": generator_version,
        "splits": dict(splits or {}),
    }
