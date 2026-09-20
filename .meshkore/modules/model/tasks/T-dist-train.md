---
id: T-dist-train
title: Optional three-device experiment acceleration
status: backlog
priority: low
owner: unassigned
category: model
initiative: decision-model
depends_on:
  - T-curriculum
created: 2026-09-19
updated: 2026-09-19
---

# Optional three-device experiment acceleration

OPCIONAL y solo si el hierro local no basta: hasta 2 dispositivos extra
toman jobs independientes de la cola de experimentos (p. ej. Mac A familia A,
Mac B familia B, CUDA familia C). Esto puede acercarse a 3× en throughput de
investigación, pero cada run sigue siendo single-machine y reproducible.
Incluye claim/lease idempotente, sincronización de manifests/resultados y
recuperación de jobs huérfanos.

No se usa DDP/Gloo para sincronizar gradientes entre MPS y CUDA: el overhead,
las diferencias de backend y la falta de una ruta GPU heterogénea fiable lo
hacen peor que repartir experimentos. DDP solo se ensaya como gate separado
si aparecen 2–3 nodos CUDA homogéneos y un run individual domina el calendario.

Fuente: plan §§123–125 (hardware doméstico, no DDP entre Macs, scheduler
simple) y stack §34.

## Verification gate

- Rama opcional: requiere PASS de `T-curriculum`, pero ninguna task obligatoria
  depende de ella.
- Antes de usar otros equipos, tres workers simulados localmente prueban claim,
  lease, heartbeat, expiración, reintento, idempotencia y hash de artifacts.
- Con dispositivos reales, un A/B compara throughput y paridad contra un único
  worker; fallar el umbral desactiva la distribución sin afectar el roadmap.
- DDP sólo puede habilitarse mediante gate separado en CUDA homogéneo; MPS↔CUDA
  queda rechazado por diseño.

## Done when

- Tres workers pueden reclamar jobs sin duplicarlos y devolver artifacts
  verificables al mismo results registry.
- El throughput agregado se compara con una máquina; no se promete 3× sin
  medición y se conserva paridad por seed/manifest.
- Si se prueba DDP, queda limitado a CUDA homogéneo y solo se adopta si mejora
  tiempo/coste frente al scheduler por el umbral escrito.
