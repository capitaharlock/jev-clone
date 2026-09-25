---
id: T-battery-calib
title: Calibración y abstención sobre la batería — medidas aparte del ranking
status: next
priority: medium
owner: unassigned
category: eval
initiative: honest-eval
depends_on:
  - T-battery-metrics
  - T-ce-finetune
created: 2026-09-25
updated: 2026-09-25
---

# Calibración y abstención sobre la batería — medidas aparte del ranking

Sustituye a `#T-data-eval`, que estaba escrita contra la mezcla de 5 M y el clean
room de Jevals: esa base ya no es la del entreno. Lo que sobrevive es lo que hace
creíble un número, rehecho sobre la batería privada.

- **Temperatura y umbral se calibran en desarrollo y se verifican en test**,
  nunca al revés.
- **Curva risk–coverage** con abstención real, comparada contra el azar de cada K.
- **`unknown` / answerability se conserva**, pero se evalúa **aparte del
  ranking**: la lección de `#T-fullspace-objective` es que 87,7 % de abstención
  con 0/1000 acierto no es prudencia, y que calibrar el umbral no arregla un
  ranking indistinguible del azar.
- **ECE, NLL y Brier** por familia e idioma, con n. Una calibración buena de
  media puede esconder una familia descalibrada.
- Reportes reproducibles por semilla y manifest, publicados junto al veredicto y
  no en un mensaje de chat.

`data/firewall.py` y `data/leakage.py` se reutilizan tal cual sobre las mezclas de
`#T-episode-splits`.

## Verification gate

- Test: la calibración ajustada en desarrollo se aplica al test sin reajustar; el
  runner falla si detecta un fit sobre el corte de test.
- Test: la curva risk–coverage se publica con el azar de su K al lado y la
  abstención en bloque separado del ranking.
- Test: un modelo que abstiene mucho y acierta poco **no** puede producir un
  veredicto favorable.

## Done when

- Calibración por familia e idioma publicada con ECE, NLL, Brier, n e IC95 %.
- Curva risk–coverage con abstención real, mejor que azar, o el NO-GO escrito.
- Los reportes son reproducibles desde semilla + manifest por alguien que no los
  generó.
