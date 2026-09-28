---
id: T-teacher-probe
title: Cablear el profesor y medir la distancia real, mismo corte y mismas filas
status: blocked
blocked_reason: el proxy de esta máquina bloquea api.typesafe.ai; ejecución remota pendiente
priority: high
owner: unassigned
category: eval
initiative: teacher-distill
created: 2026-09-23
updated: 2026-09-28
---

> **2026-09-28, ejecución remota.** Esta máquina no alcanza
> `api.typesafe.ai` por el proxy corporativo. El operador ejecutará la batería
> desde otro ordenador y devolverá los artefactos a esta misma task. La
> credencial y la caché nunca viajan por Git.
>
> **Bloqueo actual:** falta
> `artifacts/gates/T-teacher-probe/battery.json`, generado por el procedimiento
> de abajo. Cuando llegue, esta task pasa a `done` si valida las 400 filas.

> **2026-09-27, alcance ampliado por el objetivo único.** Además del corte `unseen`, el profesor se mide sobre (a) la batería de desarrollo de `#honest-eval` (`data/battery_dev.jsonl`, 400 filas, mismo formato forzado que `eval/preflight_refs.py` usa para Qwen) y (b) el test de typed-decisions convertido por `#T-ingest-laya` (400 estados / 2 000 decisiones), donde Jev publica 0,727: si nuestra medición de Jev no reproduce ≈0,73 ahí, el protocolo no es comparable y hay que arreglarlo antes de citar ninguna distancia. Depende de `#T-teacher-auth`.

# Cablear el profesor y medir la distancia real

Primera task de `#teacher-distill` y prerrequisito de las otras dos. Hace dos
cosas y ninguna toca pesos.

**1. El cliente.** Un módulo `eval/teacher.py` con: la key leída de entorno o de
un fichero fuera de git (nunca en el repo, nunca en un JSON de gate), caché en
disco por hash de `(modelo, prompt, opciones)` para que re-ejecutar un
experimento cueste 0, reintento con backoff, y un contador de tokens y coste que
cada gate que lo use tiene que publicar. El endpoint y el modelo están fijados
en el procedimiento remoto. La credencial se aporta solo en el ordenador que
ejecuta la prueba.

**2. La medición que contesta la pregunta del operador.** Se le pasa al profesor
el mismo corte `unseen` de `#T-unseen-labels`: las mismas filas, el mismo
`STATE`, el mismo conjunto de opciones y el mismo formato de respuesta forzada.
Se publica su accuracy al lado de la nuestra, por corte (banking77, huffpost,
massive, ...). Es la única forma honesta de decir "estamos a N puntos": el
umbral de release lo escribimos nosotros, el profesor no.

Muestreo: no se puntúan las 5 624 filas del corte completo de entrada. Se
empieza por una submuestra estratificada por corte con IC95 declarado (≈ 400
filas), se publica el coste, y sólo se amplía si el operador lo aprueba viendo
el gasto por punto de precisión.

## Estado 2026-09-23

**Hecho.** `eval/teacher.py` (cliente: caché en disco por hash de
`(modelo, state, pregunta, opciones)`, presupuesto duro en llamadas y en
dólares, reintento con backoff, contador de tokens/coste, clave fuera de git
y redactada de todo lo que sale del módulo) y `eval/teacher_probe.py` (corte
`unseen`, submuestra estratificada y sembrada, nuestro head re-puntuado sobre
ESAS filas, tabla de distancia por corte). 29 tests verdes contra un servidor
HTTP local que habla el formato System-One; nada stubbed en producción.
Submuestra versionada: 399 filas, coste estimado **$0,0018**. El corte
completo de 5 624 filas costaría ~$0,03, así que el muestreo deja de ser una
restricción de presupuesto en cuanto haya endpoint.

**Bloqueado.** La key anterior devuelve
401 en los seis endpoints públicos de Jev que existen: `api.typesafe.ai/v1/
systemone`, `tokenra.io/v1/decisions`, `jev-ai.pro/api/v1/systemone`,
`thejevai.com/v1/systemone`, `api.venice.ai/api/v1/decisions` y
`jevai.org/api/v1/decisions`. Falta que el operador diga **de qué proveedor
es la key**; con eso son dos variables de entorno y un `gate`.

**Medido mientras tanto** (`fullspace.json`, `eval/fullspace.py`): nuestro
head sobre las 3 080 filas de test de BANKING77 con las **77 etiquetas** como
opciones, el régimen en el que Jev publica 0,924, da **0,0123, por debajo
del azar (0,0130)**. El barrido de cardinalidad sobre las mismas filas
(K=5/8/20/40/77) muestra que la ventaja sobre azar aguanta hasta K=40 y
desaparece en K=77: lo que el head tiene con pocas opciones es una
preferencia débil, no un ranking del espacio real de etiquetas. Eso abre
`#T-bigk-optsets`.

## Done when

- `eval/teacher.py` existe con caché en disco, presupuesto y contador de coste, y
  su suite verde; la key no aparece en ningún fichero versionado.
- `artifacts/gates/T-teacher-probe/gate.json` publica accuracy del profesor por
  corte unseen, con n, IC95, coste en tokens y en dinero, y el `model_version`
  del checkpoint nuestro contra el que se compara.
- El gate incluye la línea de distancia: profesor − nosotros, por corte, y el
  corte donde la diferencia es mayor.
- La submuestra y su semilla quedan versionadas para que la medición sea
  repetible sin volver a gastar saldo.

## Procedimiento para el ordenador con acceso a TypeSafe

### Preparar

El repositorio contiene la batería, el cliente y el runner. No hay que copiar
ningún dataset por fuera de Git.

```bash
git pull --ff-only
cd jev-clone
mkdir -p .meshkore/credentials
chmod 700 .meshkore/credentials
printf '%s\n' 'PEGAR_LA_KEY_AQUI' > .meshkore/credentials/jev-api-key
chmod 600 .meshkore/credentials/jev-api-key
cat > .meshkore/credentials/jev.env <<'EOF'
JEV_TEACHER_ENDPOINT=https://api.typesafe.ai/v1/systemone
JEV_TEACHER_MODEL=jev-latest
JEV_TEACHER_KEY_FILE=.meshkore/credentials/jev-api-key
EOF
```

`.meshkore/credentials/`, `.secrets/` y `artifacts/cache/` están ignorados por
Git. No añadirlos con `git add -f`, no pegar la key en una incidencia y no
publicar la salida de entorno.

### Comprobar y ejecutar

El runner usa solo la biblioteca estándar. Primero se hace una llamada:

```bash
PYTHONPATH=. python3 -m eval.jev_battery ping
```

Si devuelve una elección, ejecutar las 400 filas con un límite duro de 400
llamadas y 0,25 USD:

```bash
PYTHONPATH=. python3 -m eval.jev_battery run \
  --budget-usd 0.25 \
  --max-calls 400
```

Se puede reanudar el mismo comando: las respuestas ya obtenidas se leen de
`artifacts/gates/T-teacher-probe/battery.picks.jsonl` y la caché local evita
pagar dos veces por el mismo prompt.

### Validar antes de devolver

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("artifacts/gates/T-teacher-probe/battery.json")
d = json.loads(p.read_text())
assert d["n_cut"] == 400, d["n_cut"]
assert d["n_scored"] == 400, d["n_scored"]
assert d["rows_sha256"] == \
    "8e8ccea5a9916eeb31d63aa8f79c247a02d5e3cbbce23b21c75dacc18535c3fb"
print("forced", d["report"]["ranking"]["accuracy"])
print("ci95", d["report"]["ranking"]["accuracy_ci95"])
print("budget", d["budget"])
PY
```

No aceptar un corte parcial como cifra del dashboard. Si una respuesta no se
puede parsear, el gate conserva `parse_errors`; no se corrige a mano.

### Devolver por Git

Los únicos resultados del benchmark que deben volver son:

```text
artifacts/gates/T-teacher-probe/battery.json
artifacts/gates/T-teacher-probe/battery.picks.jsonl
```

Actualizar también esta task con fecha, acierto, IC95, errores de parseo,
llamadas y coste. Comprobar antes del commit:

```bash
git check-ignore -v .meshkore/credentials/jev-api-key
git diff --check
git add artifacts/gates/T-teacher-probe/battery.json \
        artifacts/gates/T-teacher-probe/battery.picks.jsonl \
        .meshkore/modules/eval/tasks/T-teacher-probe.md
git diff --cached --check
```

No subir `jev-api-key`, `jev.env`, `.secrets/` ni `artifacts/cache/`.
