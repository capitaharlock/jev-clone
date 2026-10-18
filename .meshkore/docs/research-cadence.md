# Calendario de auditoría, investigación y publicación

Task: `#T-evidence-cadence`. Initiative: `#daily-learning-loop`.

El entrenamiento y la generación son continuos. Las decisiones sobre el rumbo
no lo son: necesitan cortes periódicos, comparables y escritos. Este calendario
se ancla el 9 de octubre de 2026, zona `Europe/Madrid`.

## Calendario

| Frecuencia | Momento | Trabajo | Resultado |
|---|---|---|---|
| Cada 24 h | Continuo desde la última ejecución | Salud y descubrimiento externo | Latido, procesos, fuentes externas, checkpoint, velocidad, disco, volumen generado y errores |
| Cada 48 h | Desde 2026-10-09 18:00 | Auditoría interna | Informe con veredicto `continue`, `correct`, `change-arm` o `pause` |
| Cada viernes | 19:00 | Investigación externa | Novedades de los últimos 14 días, fuentes, licencias y decisiones |
| Primer viernes del mes | Después de la investigación | Revisión de publicación | Candidatos para Hugging Face, GitHub o Zenodo, con bloqueos de licencia y privacidad |

Una promoción de checkpoint, un cambio de arquitectura, dos bloqueos del
trainer en 24 horas, una caída de más de dos puntos por familia o un aumento
anormal de rechazos adelantan la siguiente auditoría. No reinician el calendario
semanal.

## Revisión diaria de salud

Esta revisión comprueba operación, no calidad:

- un solo trainer y un solo supervisor;
- estado fresco y checkpoint reanudable;
- pasos, decisiones, pérdida, velocidad y tiempo del último paso;
- productor por reglas, productor Qwen y sus últimos lotes;
- espacio libre, volumen en disco y tasa de rechazos;
- procesos bloqueados, reinicios y tiempo sin progreso.

Un latido verde no demuestra que el modelo esté aprendiendo.

## Auditoría interna cada 48 horas

El informe vive en `.meshkore/docs/audits/YYYY-MM-DD-results.md` y conserva:

1. checkpoint, hash, datos consumidos y rango temporal;
2. batería de desarrollo en orden escrito y al menos cuatro permutaciones;
3. resultado por familia, idioma y `K`, calibración y consistencia
   contrafactual;
4. holdout de reglas, marcado como misma distribución y nunca como veredicto;
5. benchmark externo disponible, separando `measured`, `cited` y `missing`;
6. calidad de datos: procedencia, licencia, duplicados, fugas, balance,
   aceptados y rechazados;
7. coste de cómputo y rendimiento;
8. decisión: continuar, corregir datos o código, cambiar de brazo o pausar.

La batería sellada no se abre. Una mejora en el holdout de reglas no compensa
una regresión en una familia, idioma, cardinalidad o permutación. No se cambia
la composición de un benchmark después de mirar el resultado.

## Investigación externa semanal

Cada viernes se revisan los últimos 14 días. El informe vive en
`.meshkore/docs/research/YYYY-MM-DD-weekly-scan.md`.

Fuentes mínimas:

- publicaciones, documentación y repositorios oficiales de JEv;
- GitHub: modelos de decisión, pointer heads, rerankers, listwise learning,
  verificadores y herramientas de evaluación;
- Hugging Face: modelos y datasets nuevos o actualizados;
- arXiv y Papers with Code: decision making, preference learning, NLI,
  reranking, abstención, calibración y datos sintéticos verificables;
- catálogos de datasets y búsquedas web para ciencia, lógica, libros,
  comparación de productos, opinión y decisiones multicriterio.

Cada hallazgo registra URL, fecha, revisión o commit, licencia, posibilidad de
redistribución, tamaño, idiomas, arena que cubre, riesgo de fuga y coste de
integración. Internet aporta candidatos y evidencia, no gold automático.
Nada entra en entrenamiento hasta pasar el registro de fuentes, el contrato,
deduplicación, firewall de evaluación y una revisión de licencia.

## Automatización

`tools/research_loop.py` mantiene tres cadencias independientes:

- cada 24 horas consulta arXiv, GitHub, Hugging Face y las páginas oficiales
  de JEv;
- cada siete días amplía la muestra y vuelve a ordenar los candidatos;
- el primer viernes del mes prepara la revisión de publicación.

El servicio `com.jevclone.research` queda bajo `launchd`, se reinicia si cae y
escribe el latido en `artifacts/research-loop/state.json`. Cada ejecución
conserva un JSON y un informe Markdown bajo `artifacts/research-loop/runs/` y
`reports/`; `events.jsonl` y `findings.jsonl` son append-only. Un fallo de red
queda registrado por fuente y se reintenta una hora después.

El proceso solo descubre y prioriza evidencia. Nunca descarga pesos, ejecuta
código ajeno, modifica mezclas de entrenamiento ni crea gold. La adopción
exige revisión de licencia, revisión de contaminación y un experimento con
criterio de éxito fijado de antemano.

## Publicación y contribución

La publicación es secundaria al resultado del modelo, pero forma parte del
proyecto. El destino principal para datasets es Hugging Face. GitHub lleva el
contrato, validadores, conversores y benchmark. Zenodo puede fijar versiones
citables.

Se pueden publicar:

- `decision-dataset-v1`, sus validadores y manifests;
- episodios sintéticos cuyo gold tenga regla o verificador;
- herramientas de deduplicación, fuga y evaluación;
- benchmark harness y resultados reproducibles;
- source cards y recetas cuando la licencia permita redistribución.

No se publican la batería sellada, secretos, datos personales, fuentes sin
licencia de redistribución ni copias de datasets ajenos. Cada release necesita
dataset card o model card, hashes, procedencia, licencia por fuente, splits,
limitaciones y una prueba de reproducción desde cero.

## Regla de decisión

El trabajo sigue sin cambios cuando la evidencia mejora o permanece estable y
no aparece una alternativa con mejor relación entre calidad, coste y riesgo.
Se abre una tarea de corrección cuando el problema está localizado. Se cambia
de brazo cuando dos auditorías consecutivas no muestran avance transferible o
la investigación aporta una alternativa reproducible con evidencia superior.
