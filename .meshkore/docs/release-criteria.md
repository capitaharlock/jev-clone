# Criterio de release — JEv decision model

    criteria_version: 1
    criteria_date:    2026-09-21
    author:           #T-release-gate (init #honest-eval)
    status:           PROPUESTA — sin firmar

Este documento se escribe **antes** de la medición que lo evalúa. Ese orden es
el punto entero: un umbral elegido después de ver el número no es un umbral,
es una justificación. El fichero queda versionado en git y cada veredicto
publica el `sha256` del fichero exacto que aplicó (`criteria_sha`), de forma
que un umbral movido a posteriori es detectable: cambia el sha y el veredicto
viejo deja de corresponder a este texto.

Hallazgo F de la auditoría 2026-09-21: *la evaluación no mide la propiedad que
define el producto*, y el gate publicaba un verde que no lo era
(`artifacts/gates/T-release/release.json`: `exact_agree_rate 0.852` junto a
`cohen_kappa 0.0`). Un acuerdo indistinguible del azar no puede convivir con
un PASS.

El criterio es del **operador**. Esta task lo redacta; no queda cerrado hasta
que el operador lo firma (bloque `signature:` al final). Mientras la firma
esté pendiente el gate **se ejecuta igual** y emite veredicto, marcado
`criteria_signed: false` — la falta de firma no es una excusa para no medir.

---

## 1. Qué se está decidiendo

El producto promete **apuntar a la opción correcta de un conjunto de opciones
que nunca vio en entrenamiento**. Por tanto el release se decide sobre el
corte *unseen*, nunca sobre el corte *seen*, y siempre con los dos publicados
al lado (`#T-unseen-labels`). Un número *seen* solo no decide nada aquí: mide
memorización de un espacio de etiquetas fijo, que no es lo que el producto
promete (`#decision-rebuild`, auditoría 2026-09-21).

Candidato a release = el `model_version` que publica el artefacto de
evidencia. No hay release de "el modelo" en abstracto: se libera un
checkpoint con nombre.

## 2. Evidencia admisible

| qué | de dónde |
|---|---|
| accuracy / ECE / riesgo-cobertura / abstención, por corte | `artifacts/gates/T-unseen-labels/gate.json` |
| latencia p95 de la cabeza de decisión | `artifacts/gates/T-pointer-head/gate.json` → `latency_k4.p95_ms` |
| kappa contra el profesor | *no existe todavía ningún artefacto que lo mida* |

Reglas sobre la evidencia, sin excepciones:

- **Evidencia ausente = criterio FALLADO**, nunca criterio omitido. Un umbral
  sin medición no se salta: se cuenta como no cumplido. Es la única forma de
  que "todavía no lo medimos" no se lea igual que "lo medimos y salió bien".
- Todo número que decide viene del **checkpoint entrenado**, identificado por
  `model_version`. Un número sin `model_version` no es atribuible a nada y no
  puede sostener un verde (regla de coherencia C2, §5).
- Los cortes exentos los declara el propio artefacto de evidencia (`boolq`:
  un pool de dos etiquetas *es* la pregunta, no se puede retener ninguna sin
  borrar la tarea). El gate lee la exención del artefacto; no la lleva en el
  código.

## 3. Umbrales

Cada número con la línea de por qué ese número y no otro.

### 3.1 Accuracy en etiquetas no vistas

- **`unseen_accuracy_all_min = 0.50`** — corte `ALL`, sin calibrar.
  *Por qué 0,50:* con `mean_k ≈ 5,5` el azar está en 0,166. 0,50 es tres veces
  el azar y, en producto, es la frontera en que el bucle de decisión ahorra
  trabajo: por debajo de la mitad de aciertos el operador revisa más
  respuestas de las que acepta, y automatizar sale más caro que no hacerlo.
  No se pone en 0,80 porque a K variable y con distractores duros
  (`#T-optset-sampler`) ese número no lo alcanza hoy ningún modelo de 68M
  parámetros, y un umbral inalcanzable no informa: nunca cambia de estado.
- **`unseen_accuracy_cut_min = 0.35`** — cada corte no exento por separado.
  *Por qué 0,35:* poco más del doble del azar del peor corte (0,166). Existe
  para que la media no tape un dominio muerto: un `ALL` decente compuesto de
  un corte excelente y uno en el azar no es un producto, es una media.
- **`unseen_above_chance_every_cut = true`** — el límite inferior del IC95 de
  cada corte no exento debe superar su tasa de azar.
  *Por qué:* un punto por encima del azar sin IC no distingue capacidad de
  ruido muestral. Es el mínimo que hace la afirmación falsable.
- **`seen_unseen_accuracy_drop_max = 0.10`** — `accuracy_seen − accuracy_unseen`.
  *Por qué 0,10:* los dos cortes no son igual de difíciles (distinto K, 
  distinto azar), así que una caída pequeña es esperable y no se castiga. Por
  encima de 10 puntos lo que se está vendiendo es generalización y lo que se
  entrega es memorización, que es literalmente el hallazgo F.

### 3.2 Calibración

- **`unseen_ece_max = 0.10`** — ECE **calibrada** del corte unseen.
  *Por qué 0,10 y no el 0,05 de `#T-calib`:* la temperatura se ajusta sobre
  etiquetas *vistas* (una etiqueta retenida no puede ajustar ni siquiera una
  temperatura), así que sobre el corte unseen se aplica extrapolada. Se
  concede el doble de holgura que en el corte donde sí se ajustó, y no más:
  10 puntos de error de calibración ya mueven el punto de operación de §3.4
  más que la banda de cobertura que se tolera.
- **`seen_ece_max = 0.05`** — ECE calibrada del corte seen.
  *Por qué 0,05:* es el `ece_target` que este repo ya fijó en `#T-calib`. En
  el corte donde la temperatura sí se ajustó no hay motivo para relajarlo.
- La calibración debe haberse ajustado en un corte que **excluye** las
  etiquetas retenidas (`fit_cut`). Una temperatura ajustada sobre la etiqueta
  que se va a evaluar convierte el número unseen en un número seen.

### 3.3 Acuerdo con el profesor

- **`teacher_cohen_kappa_min = 0.60`**, medido sobre el corte unseen, contra
  las etiquetas del profesor de destilación.
  *Por qué 0,60:* es la frontera Landis-Koch entre acuerdo *moderado* y
  *sustancial*. Por debajo, el desacuerdo con el profesor es lo bastante
  grande como para que destilar de él sea añadir ruido. **Se usa kappa y no
  `exact_agree_rate` precisamente por el caso de `T-release`**: allí un 0,852
  de acuerdo exacto convivía con kappa 0,0 — todo el acuerdo era el que
  produce el desbalanceo de clases, cero por encima del azar.
- Hoy **no existe** ningún artefacto que mida el acuerdo del checkpoint
  entrenado contra el profesor. Por la regla de §2, este criterio sale
  FALLADO por evidencia ausente. No se rebaja ni se declara N/A.

### 3.4 Cobertura a riesgo fijo

- **`coverage_min = 0.50` a `risk_max = 0.20`**, corte unseen, estrategia de
  abstención `unknown`.
  *Por qué:* el producto siempre tiene salida de escape (abstenerse y
  escalar). Lo que se libera no es "acierta el 50%", es "sabe cuándo calla".
  Riesgo 0,20 = 80% de precisión en lo que **sí** contesta, que es el mínimo
  con el que un humano deja de revisar cada respuesta. Cobertura 0,50 = si a
  esa precisión no puede contestar ni la mitad del tráfico, la automatización
  no ahorra nada y el release no tiene caso de uso.

### 3.5 Latencia

- **`latency_p95_max_ms = 150.0`** — p95 de la cabeza de decisión, K=4, CPU,
  estado y opciones en frío.
  *Por qué 150:* el presupuesto de un salto de decisión en el bucle JEv es
  200 ms de extremo a extremo (por debajo de eso la interacción se percibe
  instantánea); 150 ms para el modelo dejan 50 ms para transporte y
  serialización. Se mide en CPU en frío a propósito: es el peor caso servible,
  no el mejor caso con caché caliente.
- La medición de latencia también tiene que llevar `model_version`: un p95 sin
  saber de qué modelo es no sostiene un release (regla C2).

### 3.6 Comportamiento exigido en `unknown`

La opción `unknown` no es un extra: es la mitad del contrato. Tres
condiciones, las tres obligatorias:

- **`unknown_abstain_band = [0.05, 0.80]`** sobre el corte unseen.
  *Por qué esa banda:* fuera de ella la abstención es degenerada y el número
  de accuracy deja de significar nada. Por debajo de 0,05 el modelo nunca
  calla y el `unknown` es decorativo; por encima de 0,80 calla casi siempre y
  cualquier precisión selectiva es trivial de alcanzar sin capacidad alguna.
- **`unknown_ranks_best = true`** — el `avg_selective_risk` de la estrategia
  `unknown` debe ser ≤ el de `maxprob` y el de `margin`.
  *Por qué:* si ordenar por `maxprob` descarta mejor que el logit `unknown`
  aprendido, entonces la cabeza de abstención no aprendió a abstenerse: el
  producto estaría vendiendo un mecanismo que la confianza genérica ya da
  gratis.
- **`unknown_beats_random = true`** — la curva riesgo-cobertura de `unknown`
  debe batir a la selección aleatoria.
  *Por qué:* es el suelo absoluto. Una cabeza de abstención que no bate al
  azar está eligiendo a qué callarse al azar.

## 4. Veredicto

**GO** si y solo si **todos** los criterios de §3 pasan. Cualquier criterio
fallado o sin evidencia ⇒ **NO-GO**, y el veredicto nombra cuáles.

No hay juicio humano intermedio, no hay "fallo menor", no hay criterio
ponderado. El gate es mecánico: lee este fichero, lee los artefactos, y
publica `artifacts/gates/T-release-gate/gate.json` con el veredicto y el
`criteria_sha` del fichero que aplicó. Un NO-GO honesto es un resultado
correcto del gate, no un fallo del gate.

## 5. Reglas de coherencia — aplican a CUALQUIER gate del repo

Estas tres no son umbrales de release: son condiciones de publicación. Valen
para todo artefacto bajo `artifacts/gates/**`, no solo para este. Su
incumplimiento es **ERROR duro**, nunca warning: el runner de gates falla y el
artefacto queda marcado como inválido (marcado, no borrado).

Un artefacto **reclama verde** cuando su `pass` de primer nivel es `true`, o
—si no tiene `pass`— cuando publica algún `*verdict*` con valor `GO` / `PASS`
/ `OK`. Las reglas se evalúan por **directorio de gate**: `artifacts/gates/<task>/`
es *una* publicación, y un `gate.json` verde responde de los números que
publican sus ficheros hermanos.

- **C1 · kappa ≈ azar no puede convivir con un verde.**
  `cohen_kappa ≤ 0.10` (o `kappa`) en un gate que reclama verde ⇒ ERROR.
  *Por qué 0,10:* kappa corrige por azar; 0,10 es acuerdo despreciable bajo
  cualquier convención (Landis-Koch: *slight* empieza en 0,01). El caso que
  origina la regla es `T-release`, con kappa 0,0 publicado junto a un
  `exact_agree_rate` de 0,852 y un `cold_verdict: GO`.
- **C2 · una métrica sin `model_version` no sostiene un verde.**
  Un gate que reclama verde y publica métricas (accuracy, ECE, Brier, NLL,
  kappa, tasas de acuerdo/detección, percentiles de latencia) sin nombrar el
  `model_version` que las produjo ⇒ ERROR.
  *Por qué:* un número no atribuible no es reproducible ni comparable entre
  releases; y es exactamente el hueco por el que un número de otro predictor
  (el `cosine-char3-softmax` sin parámetros) se publicó como si fuera el del
  producto.
- **C3 · un split sin sellar no sostiene un verde.**
  Un gate que reclama verde y publica métricas **de calidad** sin declarar el
  sello del split sobre el que se midieron ⇒ ERROR. Un split está *sellado*
  cuando el artefacto nombra un digest de 64 hex en `split_sha256` /
  `manifest_sha256`, o un bloque `splits` cuyas entradas lo llevan, o un check
  `group_split_sealed` que pasó. Si el artefacto referencia un
  `artifacts/splits/<name>/manifest.json` existente, el runner **re-hashea** el
  fichero y exige que el digest coincida.
  *Por qué una `seed` sola no basta:* una semilla no prueba qué filas se
  puntuaron de verdad, solo cómo se pretendían repartir. El hallazgo C
  (`i % 10`) era reproducible con semilla y aun así medía contaminación.

Los gates históricos que incumplen quedan marcados con un
`artifacts/gates/<task>/coherence-invalid.json` junto al artefacto — **no se
borran**: un número malo publicado es parte del historial y borrarlo es otra
forma de mentir. El inventario completo vive en
`artifacts/gates/T-release-gate/coherence-audit.json`.

---

<!-- criteria-machine-block -->
```json
{
  "criteria_version": 1,
  "criteria_date": "2026-09-21",
  "evidence": {
    "unseen": "artifacts/gates/T-unseen-labels/gate.json",
    "latency": {
      "artifact": "artifacts/gates/T-pointer-head/gate.json",
      "pointer": "latency_k4.p95_ms"
    },
    "teacher_agreement": null
  },
  "missing_evidence_is": "fail",
  "thresholds": {
    "unseen_accuracy_all_min": {
      "value": 0.5,
      "why": "3x el azar (mean_k~5.5 => chance 0.166); por debajo el operador revisa mas de lo que acepta"
    },
    "unseen_accuracy_cut_min": {
      "value": 0.35,
      "why": "algo mas del doble del azar del peor corte; impide que la media tape un dominio muerto"
    },
    "unseen_above_chance_every_cut": {
      "value": true,
      "why": "sin IC95 por encima del azar la afirmacion no es falsable"
    },
    "seen_unseen_accuracy_drop_max": {
      "value": 0.1,
      "why": "los cortes no son igual de dificiles; mas de 10 puntos es vender generalizacion y entregar memorizacion"
    },
    "unseen_ece_max": {
      "value": 0.1,
      "why": "doble holgura que el corte donde si se ajusto la temperatura, porque sobre unseen se extrapola"
    },
    "seen_ece_max": {
      "value": 0.05,
      "why": "es el ece_target que el repo ya fijo en #T-calib, en el corte donde la temperatura si se ajusto"
    },
    "teacher_cohen_kappa_min": {
      "value": 0.6,
      "why": "frontera Landis-Koch moderado/sustancial; por debajo destilar del profesor anade ruido"
    },
    "coverage_min_at_risk": {
      "value": 0.5,
      "risk_max": 0.2,
      "strategy": "unknown",
      "why": "riesgo 0.20 = 80% de precision en lo contestado (minimo para dejar de revisar); cubrir menos de la mitad del trafico no ahorra nada"
    },
    "latency_p95_max_ms": {
      "value": 150.0,
      "why": "presupuesto de 200 ms por salto de decision; 150 para el modelo y 50 para transporte y serializacion"
    },
    "unknown_abstain_band": {
      "value": [0.05, 0.8],
      "why": "fuera de la banda la abstencion es degenerada y la accuracy deja de significar nada"
    },
    "unknown_ranks_best": {
      "value": true,
      "why": "si maxprob descarta mejor que el logit unknown, la cabeza de abstencion no aprendio a abstenerse"
    },
    "unknown_beats_random": {
      "value": true,
      "why": "suelo absoluto: una cabeza de abstencion que no bate al azar elige al azar a que callarse"
    }
  },
  "coherence": {
    "cohen_kappa_floor": {
      "value": 0.1,
      "why": "kappa corrige por azar; 0.10 es acuerdo despreciable bajo cualquier convencion"
    },
    "require_model_version": {
      "value": true,
      "why": "un numero no atribuible no es reproducible ni comparable entre releases"
    },
    "require_sealed_split": {
      "value": true,
      "why": "una seed sola no prueba que filas se puntuaron; el hallazgo C era reproducible y aun asi medía contaminacion"
    }
  }
}
```

<!-- criteria-signature -->
```yaml
signature:
  signed_by: pending-operator
  signed_utc: null
  criteria_version: 1
```

Mientras `signed_by` sea `pending-operator` el gate publica
`criteria_signed: false` y **sigue emitiendo veredicto**. Firmar es fijar el
criterio, no habilitarlo: el gate nunca se salta por falta de firma.
