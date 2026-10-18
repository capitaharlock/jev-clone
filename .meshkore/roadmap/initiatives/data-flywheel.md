---
id: data-flywheel
title: El volante de datos — absorber todo lo que enseña a decidir
status: active
owner: architect-master
modules:
  - data
created: 2026-09-27
updated: 2026-09-27
---

# El volante de datos — absorber todo lo que enseña a decidir

Decisión del operador (2026-09-27): el único objetivo es un modelo que aprende y
mejora cada día. Eso exige **volumen**, y el volumen no va a salir sólo de Qwen
a 13 s por episodio. Esta iniciativa abre las otras tres fuentes y las mete
todas en el **mismo contrato** (`episode-v1`, `data/episode_contract.py`) para
que el trainer no distinga de dónde vino una decisión:

1. **Laya** (`github.com/NandhaKishorM/laya`, Apache-2.0): su dataset de ajuste
   es público, `LocalLLaMA/typed-decisions` en Hugging Face — 1 600 estados × 4
   flujos × ~5 preguntas tipadas (`choice` / `score` / `noul`) ≈ 8 000
   decisiones, EN, con split train/test publicado. Es **el corte donde Jev
   publica 0,727 y Laya 0,766**: convertirlo nos da la comparación directa que
   hoy no tenemos. Su mezcla base (AG News, BoolQ, spam, phishing, relevancia
   RAG, triage, MASSIVE, XNLI) son datasets públicos que también convertimos.
2. **Datasets públicos de decisión** (`source-register.md` ya tiene P0/P1 con
   licencia; se amplía): cada uno se convierte a episodios con ID opaco de
   candidato, texto de opción, y su corte de test publicado **fuera del train**.
3. **Qwen** como generador y verificador (`#episodic-data`): sigue siendo la
   única fuente de episodios **con contrafactuales**, que es lo que el modelo
   falla. Aquí no se sustituye; se escala en `#T-episode-scale`.
4. **Jev como profesor** (`#teacher-distill`): sus probabilidades sobre
   nuestros episodios como objetivos suaves y como recompensa. Depende de una
   credencial válida que hoy no existe.

## Principios

- **Un solo contrato.** Todo entra como `episode-v1` y pasa el validador; lo
  que no cabe en el contrato se adapta el adaptador, no el contrato.
- **Procedencia y licencia en cada fila** (`origin`, `source`, `license`), y en
  el manifest de cada mezcla la lista de fuentes con su sha.
- **Los cortes de test publicados nunca entran en train.** typed-decisions
  test, BANKING77 test, y cualquier corte que otro sistema haya publicado como
  su benchmark, se registran como `eval-only`.
- **Sin contrafactuales no hay sensibilidad:** el dato externo aporta volumen y
  cobertura de dominio; los episodios de Qwen aportan la clase de ejemplo que
  enseña a mirar el hecho decisivo. La mezcla declara el peso de cada uno y el
  gate del trainer (`#T-ce-finetune`) mide el contrafactual conjunto aparte.

## Done when

- typed-decisions está convertido a `episode-v1` (train y test por separado),
  con gate que publica n por flujo y tipo, y el test está marcado `eval-only`.
- Existe un adaptador genérico «dataset de clasificación/QA → episodio» con al
  menos ocho fuentes públicas convertidas, licencias declaradas, y una cifra
  de volumen total consumible por el trainer (no «disponible»: consumible,
  validado).
- Cada mezcla que consuma un entreno lleva manifest con fuentes, pesos, shas y
  el detector de fuga corrido (`data/leakage.py`).
- Cuando exista credencial de Jev: los objetivos suaves de Jev sobre una
  muestra de train están cacheados, con coste publicado.
