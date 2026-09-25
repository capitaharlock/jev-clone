# Recuperar el modelo de decisión: diagnóstico y experimento recomendado

24 de septiembre de 2026. Ancla: `full-space-training / T-fullspace-objective`; tareas relacionadas: `T-encoder-finetune`, `T-labelspace-factory`, `T-option-text` y `honest-eval`.

Este documento amplía y actualiza la auditoría inicial. Es una revisión del código, tareas, resultados guardados y cinco pruebas nuevas de inferencia sobre dos checkpoints existentes. No se ha entrenado ningún modelo nuevo ni modificado código del repositorio. Las alternativas propuestas aún no tienen resultados locales: no se presenta una expectativa como una mejora conseguida.

## 1. Decisión recomendada

**Dejar de escalar la receta actual como siguiente apuesta. Probar un scorer semántico preentrenado, ajustarlo con decisiones verificadas que dependan del estado y la pregunta, y medirlo en los usos reales del producto.** Mantener los checkpoints actuales como controles. El objetivo sigue siendo un modelo pequeño, no generativo, que devuelve pesos sobre opciones nuevas.

El último experimento de denominador ampliado ya terminó y no mejora el ranking. No veo fundamento para dedicar el siguiente presupuesto a más filas de la misma mezcla, más taxonomías rellenadas con las mismas plantillas o cuatro variantes de descongelación antes de disponer de una referencia semántica competente.

La hipótesis de recuperación es concreta: partir de representaciones ya entrenadas para relacionar textos y enseñar decisiones con interacciones estado–pregunta–respuesta. Es una hipótesis comprobable, no una promesa de alcanzar el 70% ni una demostración de que el pointer actual sea irrecuperable.

## 2. Qué está demostrado y qué hay que corregir del diagnóstico

| Afirmación | Evidencia y conclusión admisible |
|---|---|
| «El modelo no puede generalizar con esta arquitectura» | Los checkpoints evaluados fallan en BANKING77 a K=77. No demuestra imposibilidad arquitectónica. El código acepta textos de opciones arbitrarios y los usa. |
| «Entrenar con las 77 etiquetas por fila no ayuda» | `T-bigk-optsets` obtiene 9/1000 tanto en brazo como en control. Ofrece el espacio propio de cada dataset de entrenamiento; BANKING77 está excluido. No entrenó cada fila con las 77 etiquetas de BANKING77. No hay evidencia de beneficio al ampliar K en ese presupuesto. |
| «El texto de la opción no se usa; puntúa por posición» | `T-option-text` compara nombres, definiciones y ejemplos; no prueba dependencia posicional. El código y las nuevas pruebas contradicen esa explicación. |
| «Más datos/parámetros nunca lo arreglan» | Las curvas negativas desaconsejan repetir esas configuraciones. No cubren otras distribuciones de datos, objetivos, preentrenamientos ni un encoder completamente ajustado. |
| «El nuevo objetivo sigue entrenando» | Hay gate final de `T-fullspace-objective`, generado a las 15:39:27 UTC: 0/1000 aciertos, 87,7% abstención; forzando elegir, 9/1000. Control: 10/1000 sin abstención. |
| «El encoder no representa las descripciones: causa medida» | Es una hipótesis razonable. No está aislada frente al head, entrenamiento, preguntas poco informativas, datos o incompatibilidad entre distribución de entrenamiento y evaluación. |

Con K=77, azar uniforme = 1/77 = 1,2987%. El 0,9% forzado del nuevo brazo tiene IC95% [0,4742%, 1,7016%]: no acredita mejora sobre azar. El 0% incluyendo abstención tiene IC95% [0%, 0,3827%]. **Calibrar la abstención por sí sola no arreglará un ranking que sigue al azar.**

La narrativa automática también requiere revisión: `verdict.reading` de fullspace dice que el intervalo primario contiene el azar, aunque sus números lo excluyen por debajo. El intervalo que sí lo contiene es el de elección forzada. En bigK el resumen habla de descartar la cardinalidad como causa, pero un resultado negativo con una semilla y presupuesto limitado sólo descarta el beneficio observado de ese brazo. Esas frases no deben alimentar el siguiente plan como hechos causales.

Tampoco daría por demostrado que Jev entrena como DPR/E5/SetFit ni que basta reproducir esa receta. La descripción pública habla de una arquitectura y entrenamiento propios; no publica información suficiente para esa equivalencia. Un resultado externo de BANKING77 tampoco prueba un 99% universal sobre decisiones de cualquier clase. Véase la [descripción oficial de Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

## 3. Prueba nueva: las puntuaciones siguen al texto, pero fallan los cambios de significado

Se ejecutó `probe_mechanism.py` en CPU, con los pesos locales y sin red, sobre el checkpoint leverstack de 1 M y el fullspace de 62.528 muestras. Se recorrieron todas las permutaciones de opciones en cinco casos y se intercambiaron sus textos manteniendo los IDs.

- La mayor diferencia al permutar y realinear probabilidades fue menor de 5 × 10⁻⁷.
- Al intercambiar textos conservando IDs, las puntuaciones siguieron al texto con el mismo margen numérico.
- Ambos modelos eligen «green» cuando el estado dice explícitamente que el color favorito es verde **y también cuando dice rojo**.
- Ambos eligen «amarillo» ante «Mi color favorito es el verde».
- Ambos eligen producto A ante «¿cuál es más barato?» y «¿cuál dura más?», aunque el estado establece que A es más barato y B dura más.

En el checkpoint fullspace la probabilidad de A normalizada sólo entre las dos opciones cambia de 69,709% a 69,747% al pasar de precio a duración: no sigue el cambio de respuesta correcta. Esto es evidencia de un fallo funcional concreto, compatible con preferencias aprendidas sobre textos y escasa sensibilidad útil al contexto. No demuestra que siempre ignore la pregunta o el estado.

**Son cinco pruebas de mecanismo, no una estimación de accuracy ni un benchmark representativo.** La arquitectura `CrossBlock` documenta y aplica atención sin posición en el eje de opciones (`model/decision_head.py:73`); `score()` codifica sus textos (`:372`). Confundir esto con una cabeza de clases fijas conduce a diagnosticar y arreglar otra cosa.

Archivos reproducibles: [script](probe_mechanism.py), [resultados completos](probe_mechanism.json). Desde la raíz del repositorio:

```sh
PYTHONPATH=. .venv-train/bin/python ../TMP/REANALISIS_DECISION_2026-09-24/probe_mechanism.py
```

El script conserva rutas locales y requiere esos checkpoints. Su nueva ejecución sobrescribiría el JSON: preservarlo antes si se desea mantener esta evidencia histórica.

## 4. El problema de datos y preguntas es prioritario

El corpus existente puede seguir aportando ejemplos útiles, pero no asumiría que un millón de filas constituye un millón de lecciones sobre decisiones abiertas.

1. `data/episodic.py` genera 20.000 taxonomías de pseudopalabras con estructuras `campo = valor` y distractores. Es una prueba útil de recuperación literal y opciones nuevas; no enseña por sí sola comparación de productos, negación, preferencias o inferencia. Sus preguntas no llevan prosa natural. `canonical_question()` explica que parte del corpus convertido usa IDs de dataset como pregunta. El entrenamiento necesita muchos ejemplos donde **el mismo estado y las mismas opciones cambien de respuesta al cambiar la pregunta**.
2. `data/labelgen.py:297` recibe taxonomías de Qwen y construye las filas con plantillas deterministas: `Field report: definición del oro`, `Cross-check against: definición hermana`. Diversidad de nombres de categoría no equivale a diversidad de razonamiento. El marcador de la línea facilita atajos.
3. Durante esta revisión `spaces.jsonl` contiene 294 entradas; la cifra 7/2000 del mensaje ya no describe ese archivo. Eso no prueba que las 294 estén publicadas en la mezcla realmente consumida por un entrenamiento. Hay que medir por separado generación, aceptación, publicación y consumo. No usaría la estimación de 90 horas para decidir una paralelización ahora.
4. Mezclar etiquetas de otros espacios como negativos puede introducir alternativas semánticamente válidas. Filtrar duplicados literales no prueba exclusión semántica. Cada pregunta necesita un conjunto de candidatos y una regla de respuesta coherentes.

**No hay que extraer todos los conocimientos del mundo de Qwen para comparar productos.** Los atributos del catálogo deben viajar en el estado o recuperarse de una fuente. El estudiante aprende a aplicar una pregunta o criterio a esos hechos. Los conocimientos no presentes en el estado sí requieren conocimiento previo, recuperación o cascada; un encoder pequeño no garantiza resolver preguntas arbitrarias de cualquier dominio.

## 5. Técnica recomendada: cross-encoder supervisado, después destilación

El primer candidato debe leer conjuntamente los textos relevantes:

```text
entrada por candidato:
  ESTADO: hechos y atributos pertinentes, incluidas alternativas para comparar
  PREGUNTA: criterio explícito
  RESPUESTA CANDIDATA: texto/descripción de la opción i

z_i = scorer_compartido(estado, pregunta, respuesta_i)
p_i = softmax(z_1, ..., z_K)
L = -log p_respuesta_correcta
```

Todos los candidatos pasan por el mismo modelo y la misma salida escalar. No hay una neurona entrenada por cada etiqueta del catálogo. La interacción entre tokens del estado, la pregunta y la opción ocurre dentro del encoder preentrenado. La documentación de [Sentence Transformers sobre cross-encoders](https://sbert.net/docs/cross_encoder/training_overview.html) describe este patrón de puntuación de pares; su [referencia de pérdidas](https://sbert.net/docs/package_reference/cross_encoder/losses.html) incluye objetivos de ranking por listas.

Para preguntas de comparación, el estado debe incluir los datos de los demás productos, o incorporarse un contexto de candidatos idéntico para cada puntuación. Si las opciones contienen información exclusiva y relevante, puntuar cada una sin acceso a las demás puede perder el significado de «mejor». Documentar ese contrato desde el primer piloto.

**La CE sobre las opciones ofrecidas ya es un objetivo válido para el producto descrito.** Empezar con K=2–8, negativos difíciles y variación semántica real es razonable. Incluir después K mayores si se necesitan; medirlos siempre. No es obligatorio normalizar contra un universo global de etiquetas ajenas a la pregunta para aprender a elegir entre tres colores. Cambiar el denominador no garantiza que el modelo use semántica.

### Punto de partida concreto

- Para ES+EN, usar como primera referencia económica un checkpoint ya afinado en inferencia textual, como [multilingual-MiniLMv2-L6-mnli-xnli](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli). Medir primero sin entrenar mediante hipótesis explícitas («La respuesta a esta pregunta es …»), usando entailment; después ajustar el encoder y su scorer a las decisiones. No asumir que un checkpoint NLI resuelve aritmética o preferencias complejas sin adaptación.
- [ModernBERT-base-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0) es una alternativa relevante para inglés y tamaño cercano al objetivo. **Su mezcla publicada incluye BANKING77**: no presentarlo como transferencia a BANKING77 nunca visto ni como comparación limpia con nuestro train cercado. Sí puede medirse sobre la batería privada nueva.
- [GLiClass](https://github.com/knowledgator/gliclass) es una alternativa existente que procesa etiquetas dinámicas con el texto. Su [artículo](https://arxiv.org/abs/2508.07662) permite evaluar una arquitectura ya entrenada para esta función. Lo dejaría como siguiente candidato si el coste de K pasadas es el obstáculo; no abriría diez brazos simultáneos ni asumiría calidad en español sin medirla.

Ajustar el encoder completo o mediante adaptación eficiente; comparar con el checkpoint sin ajustar. Entrenar sólo otra cabeza aleatoria sobre representaciones congeladas repite la limitación que queremos someter a prueba. El anterior NO-GO de dos capas finales no equivale a probar esta receta.

Qwen local es la referencia de capacidad y el profesor para producir/verificar ejemplos; no tiene que ser el modelo servido. Pedir una elección estructurada entre IDs válidos, guardar evidencia y comprobar resultados. Las cifras de confianza escritas por un LLM no son probabilidades calibradas: empezar con gold validado y CE; añadir KL con temperatura sólo si se obtienen distribuciones del profesor reproducibles y útiles. Evitar entrenar contra porcentajes inventados por el generador.

### Coste y contrato de producto

El cross-encoder repite la lectura del estado por opción, aunque pueda agruparlas en batches. **No doy por cumplidos 20–40 ms ni la caché de estado compartida.** Es la referencia de calidad para descubrir si los datos y la tarea permiten aprender. Una vez funcione, destilar sus puntuaciones a un modelo con estado compartido, bi-encoder o interacción tardía y medir cuánto pierde. Si pierde demasiado, mantener la referencia en una cascada para casos difíciles. No elegir la arquitectura definitiva sólo por latencia antes de demostrar competencia.

## 6. Cómo generar datos que sí enseñen la tarea

Producir episodios completos, no sólo taxonomías. Cada episodio debe guardar `state`, pregunta natural, candidatos con ID opaco y texto, respuesta válida, evidencia, familia, idioma, origen, semilla/versión del generador y grupo de variantes. Para casos ambiguos, guardar conjunto de respuestas aceptables o preferencia explícita; no imponer arbitrariamente una única verdad.

Familias iniciales:

- Extracción y paráfrasis: hechos explícitos, reformulaciones sin copiar la frase del candidato.
- Comparación de atributos: precio, duración, calidad definida por una regla, restricciones y desempates.
- Clasificación por descripciones: categorías nuevas cuya definición acompaña a la opción.
- Inferencia textual y negación: evidencia favorable, contraria y datos insuficientes.
- Decisiones con prioridades: «prioriza duración; si empatan, menor precio», no «mejor» sin criterio.

Para cada familia generar contrafactuales: cambiar sólo el hecho decisivo, sólo la pregunta, o sólo una descripción de opción. La respuesta debe cambiar cuando corresponde. Balancear nombres, posiciones, vocabulario y etiquetas ganadoras; incluir distractores plausibles. No exigir cambio ante una paráfrasis equivalente.

Usar hechos estructurados y gold por regla para comparaciones numéricas; Qwen puede redactarlos sin alterar los hechos. En decisiones semánticas, pedir gold y evidencia, contrastar mediante un verificador separado y revisar manualmente una muestra estratificada. Un segundo pase del mismo modelo reduce algunos errores, pero no es una verdad independiente. Descartar desacuerdos o ambigüedades del primer piloto.

Separar train/dev/test por familias de plantillas, entidades, espacios y grupos contrafactuales; no repartir las variantes del mismo caso entre cortes. Mantener un test privado que no haya producido el profesor ni se le haya mostrado durante generación. Evitar declaraciones absolutas de ausencia de contaminación del preentrenamiento.

El ejemplo «le gusta la fruta verde → color favorito verde» es una inferencia plausible, no una consecuencia necesaria. Puede entrenarse como elección de la alternativa más compatible, pero no etiquetarse como certeza factual. En los tests básicos utilizar primero hechos inequívocos como «su color favorito es verde».

## 7. Piloto acotado y criterios de decisión

Las cantidades siguientes son una propuesta de presupuesto, no resultados medidos ni estimaciones de tiempo garantizadas.

| Paso | Trabajo | Criterio para continuar |
|---|---|---|
| 1. Contrato y batería | 400 casos de desarrollo + 600 finales sellados, revisados, con las cinco familias, ES/EN y K=2/3/8; conjunto diagnóstico separado a K=20/77. Definir mezcla y pesos antes de evaluar. | Gold justificable, variantes agrupadas, preguntas completas; desacuerdos resueltos. |
| 2. Referencias | Ejecutar Qwen local, NLI sin ajustar y checkpoint actual sobre desarrollo. Un único formato por modelo elegido sin consultar el test final. | Si ni Qwen responde bien, revisar tarea/datos/formato antes de destilar. Si NLI funciona, conservarlo como punto de partida. |
| 3. Mecánica | Ajustar 32–64 ejemplos de entrenamiento inequívocos y comprobar que el modelo los aprende. | >95% en este pequeño conjunto; sólo valida mecánica, nunca generalización. Si falla, arreglar pipeline. |
| 4. Aprendizaje | Generar 5.000 decisiones verificadas; ampliar hasta 20.000 si mejora desarrollo. CE por pregunta y encoder ajustable; evaluación intermedia a presupuestos predefinidos. | Mejorar respecto al mismo modelo sin ajustar en desarrollo, sin destruir contrafactuales ni un idioma. Si empeora, revisar datos/objetivo y volver al checkpoint base. |
| 5. Confirmación | Repetir el candidato seleccionado con otra semilla; elegir usando desarrollo. Abrir una vez el test final y publicar todos los cortes. | Meta del usuario: ≥70% macro entre familias en preguntas respondibles de la batería representativa; informar IC95%, K e idiomas. Para afirmar «al menos 70%» con respaldo estadístico, exigir también límite inferior ≥70%. |
| 6. Optimización | Sólo tras mejora confirmada: 100k ejemplos dirigidos a errores, destilación a estado compartido, cuantización y paridad Rust/Python. | Calidad y calibración retenidas; medir latencia y memoria en hardware real. |

Cualquier familia o idioma con bajo rendimiento sigue siendo una limitación aunque la media pase. El tamaño propuesto no permite estimaciones precisas en cada cruce familia × idioma × K: publicar n y ampliar los cortes críticos antes de release. No reutilizar repetidamente el test final para decidir el siguiente brazo; tras usarlo pasa a evidencia histórica y se prepara otro para la próxima decisión final.

Métricas mínimas: accuracy forzada; accuracy incluyendo abstención; cobertura y precisión entre respondidas; macro por familia; azar por K; éxito conjunto en pares contrafactuales; invariancia a permutaciones; NLL/Brier y calibración. Calibrar temperatura/umbral en desarrollo y verificar en test. Los pesos softmax son relativos a los candidatos disponibles; no equivalen automáticamente a probabilidad absoluta de verdad. Conservar `unknown`/answerability, pero evaluar el ranking y la abstención por separado para no ocultar el fallo de uno con el otro.

## 8. Cambios propuestos a las tareas pendientes

- **`T-fullspace-objective`:** registrar el NO-GO y corregir el texto del gate. No atribuirlo automáticamente a una imposibilidad del encoder. No repetir el mismo run como siguiente apuesta.
- **`T-encoder-finetune`:** reformular el primer entregable: comparación sin entrenar de un scorer semántico preentrenado y un piloto de ajuste sobre decisiones verificadas. El head actual con encoder ajustado puede ser un control posterior, no cuatro brazos obligatorios antes de tener referencia. Su requisito de usar «el objetivo ganador» carece de un ganador de calidad en los gates actuales.
- **`T-labelspace-factory`:** priorizar generación de episodios y contrafactuales auditables; medir el consumo efectivo. No paralelizar la plantilla actual sólo para llegar antes a 2.000 espacios.
- **`T-option-text`:** incorporar el control de permutación y seguimiento de textos aquí reproducido, y distinguirlo del seguimiento del estado/pregunta. Retirar la conclusión posicional si figura en la explicación operativa.
- **`honest-eval`:** incorporar la batería de producto, protocolos equivalentes de información y la regla de test sellado. BANKING77 se conserva como transferencia difícil; no es la única definición de éxito para decisiones binarias y catálogos.
- **`teacher-distill`:** Qwen local como primera referencia y fuente verificable; Jev como comparación externa cuando exista acceso válido y protocolo comparable. No bloquear la recuperación por esa credencial.

Estas son propuestas para el responsable del roadmap. No se han modificado estados de tareas ni detenido trabajos existentes durante esta revisión.

## 9. Corrección matemática adicional del objetivo muestreado

`training/python/fullspace_loss.py` define logits dependientes del conjunto, `z_c(C)`, y después afirma que `sum(exp(z_c)/pi_c)` estima sin sesgo la masa del espacio completo. La identidad de Horvitz–Thompson requiere valores poblacionales fijos respecto al muestreo, o una justificación adicional. Aquí la atención entre opciones cambia los logits al cambiar C, incluido oro/unknown. Por tanto, los tests con logits fijos **no demuestran** que ese estimador sea insesgado respecto a los logits calculados con el conjunto completo. Tampoco se traslada automáticamente el argumento de Jensen a esa pérdida completa.

Se puede justificar la fórmula para logits independientes del conjunto bajo el diseño muestral adecuado, o describir la pérdida actual como otro objetivo sin esa garantía. No afirmo que esto explique causalmente el 0,9%. Sí impide presentar el muestreo como solución matemáticamente acreditada para este head. Para el piloto propuesto basta CE sobre los candidatos explícitos: no introducir este problema si el producto no lo requiere.

## 10. Estado global y trazabilidad

El proyecto conserva valor: contrato de decisiones, infraestructura de datos, checkpoints comparables, evaluación que ahora expone el fallo, componentes de inferencia y posibilidad de generación local. La principal deuda es demostrar una cadena completa **caso bien definido → datos válidos → modelo competente → puntuaciones calibradas → mismo comportamiento en producción**.

La auditoría inicial recoge además integración del motor en el servidor, carga del encoder afinado y claves de caché V1. No los doy por resueltos por el mero hecho de que el runtime sea rápido. No explican estos fallos medidos en Python, pero deben verificarse sobre el checkpoint finalmente elegido antes de release. Esta revisión prioriza entrenamiento; no acredita una nueva auditoría completa de todos los cambios concurrentes de runtime.

La evidencia procede de un árbol con cambios concurrentes. `evidence_manifest.json` fija HEAD, hashes y copias de los archivos pequeños relevantes, sin copiar corpus, secretos ni pesos. Los números citados corresponden a esas lecturas; la fábrica y los jobs pueden avanzar después. La carpeta de entrega está físicamente fuera del repo, en `../TMP`; `tmp` es un enlace ignorado por Git.

**Conclusión:** el fracaso actual está medido; la explicación de «no usa texto, sólo posición» no lo está y la prueba nueva la contradice. La siguiente inversión debe demostrar que un modelo ya semántico aprende decisiones auténticas con una evaluación representativa. Calidad primero, después diversidad a escala y optimización del estado compartido.
