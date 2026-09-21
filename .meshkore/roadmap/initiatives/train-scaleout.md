---
id: train-scaleout
title: Entrenamiento en 3 dispositivos (futuro)
status: backlog
owner: architect-master
modules:
  - model
created: 2026-09-20
updated: 2026-09-20
---

# Entrenamiento en 3 dispositivos (futuro)

Work-stream futuro, en `backlog`: escalar el entrenamiento a los tres
dispositivos disponibles (Mac M4 Max 48 GB, Mac M5 48 GB, Windows NVIDIA
CUDA) cuando el volumen de datos de entrenamiento y verificación lo exija.
Estrategia fijada por las fuentes: **paralelizar experimentos, no
gradientes** (plan §§123–125) — Mac A → familia A, Mac B → familia B,
CUDA → familia C — con una cola simple (`experiments/*.yaml` →
`results/<run_id>/`), sin Kubernetes.

Fuera de alcance por diseño: DDP/Gloo entre MPS y CUDA (overhead y
backends heterogéneos lo hacen peor que repartir experimentos), prometer
3× sin medir, y cualquier dependencia desde tasks obligatorias — ninguna
task requerida depende de esta iniciativa. DDP solo se ensaya como gate
separado si aparecen 2–3 nodos CUDA homogéneos y un run individual domina
el calendario.

Fuentes: plan §§123–126 (training doméstico, no-distribuido inicial,
scheduler simple), stack §§8–9 y §34 (matriz de máquinas).

Tasks (secuenciales): `#T-scaleout-gate` → `#T-dist-train` →
`#T-scaleout-data`.

## Done when

- El criterio de activación y el inventario de las 3 máquinas están
  escritos y medidos antes de encender ningún dispositivo extra.
- Tres workers reclaman jobs sin duplicarlos y devuelven artifacts
  verificables al mismo results registry, con paridad seed/manifest.
- El throughput agregado se compara contra una máquina sin prometer 3×
  sin medición; fallar el umbral desactiva la distribución sin afectar
  al roadmap.
