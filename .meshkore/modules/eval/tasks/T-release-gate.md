---
id: T-release-gate
title: Criterio de release escrito antes de medir, y gate que lo aplica
status: blocked
blocked_reason: espera la firma del operador en .meshkore/docs/release-criteria.md (signed_by)
priority: high
owner: unassigned
category: eval
initiative: honest-eval
depends_on:
  - T-unseen-labels
created: 2026-09-21
updated: 2026-09-21
---

# Criterio de release escrito antes de medir, y gate que lo aplica

Hallazgo F, cierre: **no hay criterio explícito y verificable de "ya es
suficientemente capaz"**, y el gate actual publica un verde que no lo es.
`artifacts/gates/T-release/release.json` da `exact_agree_rate: 0.852` junto a
`cohen_kappa: 0.0` — acuerdo con el profesor indistinguible del azar,
enmascarado por desbalanceo de clases. Un kappa 0 no puede convivir con un
PASS.

Trabajo:

1. Escribir `.meshkore/docs/release-criteria.md` **antes** de la medición que
   lo evalúa, fechado y versionado en git, con umbrales numéricos para:
   accuracy unseen mínima por corte, ECE máxima, kappa mínimo contra el
   profesor, cobertura mínima a riesgo fijo, latencia p95, y el
   comportamiento exigido en `unknown`. Cada umbral con una línea de
   justificación: por qué ese número y no otro.
2. Implementar el gate que lo aplica mecánicamente: lee el criterio, lee los
   artefactos de `#T-unseen-labels`, y emite GO/NO-GO. Sin juicio humano en
   el medio.
3. **Reglas de coherencia** que hacen fallar cualquier gate del repo, no solo
   este: `pass: true` con `cohen_kappa <= 0.1`, o con una métrica cuyo
   `model_version` falte, o con un split no sellado → error, no warning.
4. Aplicar las reglas retroactivamente a los gates ya publicados y marcar los
   que no las cumplen.

El criterio es del operador: esta task lo redacta como propuesta y no se
cierra hasta que el operador lo firma.

## Verification gate

- Test: un artefacto sintético con `pass: true` y `cohen_kappa: 0.0` hace
  fallar el runner de gates.
- Test: el gate de release recalcula GO/NO-GO desde
  `release-criteria.md` y no desde constantes en el código.
- El gate escribe `artifacts/gates/T-release-gate/gate.json` con el veredicto
  y el sha del criterio que aplicó.

## Done when

- `.meshkore/docs/release-criteria.md` existe, está firmado por el operador y
  su sha queda registrado en cada veredicto.
- Ningún gate del repo puede publicar verde con kappa ≈ 0 o sin
  `model_version`.
- Los gates históricos que no cumplen están marcados, no borrados.
