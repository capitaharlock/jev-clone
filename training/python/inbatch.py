"""Negativos in-batch entre espacios, con corrección de inclusión.

La vía muestreada de `#T-fullspace-objective`, montada sobre la
especificación de `training/python/fullspace_loss.py` y sobre lo que
`artifacts/gates/T-fullspace-objective/batch-composition.json` midió del
cargador real. Dos piezas, en este orden:

**Batches mixtos.** `MixtureStream.epoch()` sirve cada batch de un solo
dataset — medido: 100 % de los batches, un dataset, mediana de 5 etiquetas
ofrecidas únicas. Con ese cargador los negativos in-batch caen DENTRO del
espacio de la fila, y el brazo sería el brazo exacto de `#T-bigk-optsets`
con ruido. `mixed_batches()` reagrupa el mismo presupuesto de filas
intercalando datasets, que es lo que sube el pool a ~72 etiquetas ajenas
por fila y lo convierte en una muestra de un espacio más grande que el
propio.

**El denominador extendido.** A las opciones propias de cada fila se le
añaden las etiquetas ofrecidas por las OTRAS filas del batch que no
colisionan por clave normalizada con ninguna suya — el filtro obligatorio:
sin él, el 97,7 % de las filas recibiría su propio oro otra vez como
columna negativa (`false_negatives_if_mixed`) y el gradiente empujaría
hacia abajo la respuesta correcta.

Cada columna ajena lleva su `log π(c)`, con

    π(c) = 1 - (1 - q(c))^(B-1)

— la inclusión del diseño «B-1 filas más, cada una ofrece sus etiquetas,
las repeticiones colapsan», que es el caso INCLUSION de la spec y **no**
`m·q(c)`. `q(c)` se calibra una vez al arrancar contando ofertas por fila
sobre un paso en seco del stream, y la calibración viaja en `run.json`.

La cabeza se conserva: `set_attention=True`. Este módulo cambia el
DENOMINADOR, no la arquitectura — con `set_attention=False` sería otra cosa
y se etiquetaría como cambio de arquitectura (R4, R9).
"""
from __future__ import annotations

import math
from dataclasses import replace

import torch

from training.python.fullspace_loss import (inclusion_from_with_replacement,
                                            normalise_key)

#: suelo de `q(c)`: una etiqueta que la calibración nunca vio tiene una
#: inclusión positiva pero desconocida, y `log 0` no es un número. El suelo
#: se declara en el artefacto en vez de esconderse en un `clamp`.
MIN_OFFER_RATE = 1e-6

#: batches de reserva que el reagrupador acumula antes de drenar. Con la
#: mezcla real (12 datasets, round-robin) una reserva de 16 batches basta
#: para que cada batch de salida lleve filas de todos los datasets vivos, y
#: son 16 x B filas en memoria: nada.
LOOKAHEAD_BATCHES = 16


def mixed_batches(stream, batch_size: int, epoch: int = 0,
                  lookahead: int = LOOKAHEAD_BATCHES):
    """Los mismos batches del stream, reagrupados entre datasets.

    Consume `stream.epoch(epoch)` tal cual y mantiene una cola por dataset.
    Dos cosas que no son negociables y que un reagrupador ingenuo se salta:

    * **se drena de verdad.** Una pasada round-robin saca UNA fila por
      dataset —12 de las 64 que entran— así que parar tras una pasada deja
      las colas creciendo más rápido de lo que drenan, y una época de 1 M
      de filas acaba entera en memoria.
    * **se espera a tener con qué mezclar.** Drenar en cuanto hay filas
      devuelve el primer batch entero del primer dataset, que es justo lo
      que este reagrupador existe para evitar. Se acumula una reserva de
      `lookahead` batches y se drena contra ella, así que cada batch de
      salida lleva filas de tantos datasets como haya vivos.

    El presupuesto de filas, el orden de consumo de cada sampler y la
    mezcla realizada son los mismos: lo único que cambia es el AGRUPAMIENTO.
    """
    queues: dict = {}
    current: list = []
    backlog = batch_size * max(lookahead, 1)

    def drain(floor: int):
        nonlocal current
        while sum(len(q) for q in queues.values()) >= floor:
            progressed = False
            for dataset in sorted(queues):
                q = queues[dataset]
                if not q:
                    continue
                current.append(q.pop(0))
                progressed = True
                if len(current) == batch_size:
                    out, current = current, []
                    yield out
                    break
            if not progressed:
                return

    for batch in stream.epoch(epoch):
        for sample in batch:
            queues.setdefault(sample.dataset, []).append(sample)
        yield from drain(backlog)
    yield from drain(1)
    if current:
        yield current


def calibrate_offer_rates(stream, batches: int, batch_size: int,
                          epoch: int = 0) -> dict:
    """`q(c)` — con qué frecuencia una fila ofrece la etiqueta `c`.

    Un paso en seco sobre `stream.epoch(epoch)`, que crea generadores
    nuevos y NO consume el stream de entreno. Devuelve el mapa
    `clave normalizada -> q` más el recuento que lo respalda, para que el
    artefacto pueda decir sobre cuántas filas se calibró.
    """
    counts: dict = {}
    rows = 0
    for n, batch in enumerate(mixed_batches(stream, batch_size, epoch)):
        if n >= batches:
            break
        for sample in batch:
            rows += 1
            for key in {normalise_key(o["text"]) for o in sample.options}:
                counts[key] = counts.get(key, 0) + 1
    if not rows:
        raise ValueError("the calibration pass saw no rows")
    return {"rows": rows,
            "q": {k: v / rows for k, v in counts.items()},
            "min_offer_rate": MIN_OFFER_RATE}


def inclusion_log_pi(calib: dict, batch_size: int) -> dict:
    """`log π(c)` por clave, para un batch de `batch_size` filas."""
    keys = sorted(calib["q"])
    q = [max(calib["q"][k], MIN_OFFER_RATE) for k in keys]
    pi = inclusion_from_with_replacement(q, max(batch_size - 1, 1))
    return {k: math.log(min(p, 1.0)) for k, p in zip(keys, pi)}


def extend_batch(batch: list, log_pi: dict) -> tuple:
    """Añade a cada fila las etiquetas ajenas del batch, con su corrección.

    Devuelve `(batch_extendido, filas_de_desplazamiento)`. Cada fila de
    desplazamiento tiene una entrada por columna de opción: 0 en las
    propias (se incluyen con π = 1 por construcción) y `log π(c)` en las
    ajenas. `fullspace_loss.sampled_loss_batch` le añade el 0 del
    `unknown` y el de las columnas de relleno.

    Una fila `unknown` mantiene su oro FUERA del rango de opciones: al
    crecer la lista, `gold_index` se recoloca al nuevo final, o
    `batch_gold` la leería como si apuntase a una opción ajena.
    """
    offered = {}
    for i, sample in enumerate(batch):
        for opt in sample.options:
            offered.setdefault(normalise_key(opt["text"]), []).append(
                (i, sample.dataset, opt))
    out, shifts, added = [], [], 0
    for i, sample in enumerate(batch):
        own = {normalise_key(o["text"]) for o in sample.options}
        options = [dict(o) for o in sample.options]
        shift = [0.0] * len(options)
        for key in sorted(offered):
            if key in own:
                continue                      # colisión: el filtro obligatorio
            j, dataset, opt = offered[key][0]
            options.append({"id": opt["id"], "text": opt["text"],
                            "space": dataset})
            shift.append(log_pi.get(key, math.log(MIN_OFFER_RATE)))
            added += 1
        gold = (len(options) if sample.is_unknown else sample.gold_index)
        out.append(replace(sample, options=options, gold_index=gold))
        shifts.append(shift)
    return out, shifts, {"rows": len(batch), "columns_added": added,
                         "mean_added_per_row": round(
                             added / max(len(batch), 1), 3)}


def shift_tensor(shifts: list, kmax: int, device, dtype) -> torch.Tensor:
    """`[B, K_max + 1]`: el desplazamiento, con 0 en relleno y `unknown`.

    Las columnas de relleno llegan a la cabeza como `-inf`; su
    desplazamiento DEBE ser 0, porque `-inf - (-inf)` es NaN.
    """
    rows = [row + [0.0] * (kmax + 1 - len(row)) for row in shifts]
    return torch.tensor(rows, device=device, dtype=dtype)


def regime(calib: dict, batch_size: int, stats: dict) -> dict:
    """El régimen del denominador, declarado — R1/R4."""
    return {
        "mode": "full space + corrected cross-space in-batch negatives",
        "loss_normalised_over": (
            "the row's own label space plus every label text offered by the "
            "OTHER rows of the batch that does not collide by normalised "
            "key with one of its own, plus the `unknown` logit"),
        "correction": {
            "scheme": "inclusion (Horvitz-Thompson)",
            "pi": "pi(c) = 1 - (1 - q(c))^(B-1)",
            "why_not_m_times_q": (
                "the in-batch set is a DEDUPLICATED sample; m * q(c) is the "
                "proposal correction for draws with replacement and biases "
                "here (training/python/test_fullspace_loss.py::"
                "test_proposal_correction_on_a_deduped_set_is_biased)"),
            "own_columns": "log pi = 0 — a row always offers its own space",
            "calibration_rows": calib["rows"],
            "calibrated_labels": len(calib["q"]),
            "min_offer_rate": MIN_OFFER_RATE,
        },
        "collision_filter": (
            "a foreign label whose normalised key matches one of the row's "
            "own columns is dropped. Without it 97.7 % of rows would be "
            "handed their own gold again as a negative column "
            "(artifacts/gates/T-fullspace-objective/batch-composition.json)"),
        "loader": "mixed batches — the same rows, regrouped across datasets",
        "batch_size": batch_size,
        "architecture": "set_attention=True — the head is NOT changed; this "
                        "arm moves the denominator only",
        "estimator_is_optimistic": (
            "the sampled loss UNDERESTIMATES the full-space loss (Jensen): a "
            "lower train loss than the exact arm is the estimator, not an "
            "improvement. Only the full-cardinality primary decides"),
        "measured": stats,
    }


def mixed_stream(stream, batch_size: int, epochs: int = 1_000_000):
    """`(epoch, batch)` como `MixtureStream.stream()`, pero reagrupado.

    El bucle de entreno consume esto en lugar de `stream.stream()` cuando
    el brazo pide batches mixtos; se para igual que el original cuando una
    época no rinde nada.
    """
    for e in range(epochs):
        yielded = False
        for batch in mixed_batches(stream, batch_size, e):
            yielded = True
            yield e, batch
        if not yielded:
            return
