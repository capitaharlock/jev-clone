# Protocolo de rotación del test sellado (#T-battery-sealed)

Un corte sellado usado es **evidencia histórica**: informa de la decisión
que ya se tomó y no se reutiliza para elegir el próximo brazo. Cada
decisión final necesita un sellado fresco. Este protocolo dice cómo
preparar el siguiente SIN abrir el actual.

## Reglas inviolables

1. **El sellado vigente se abre una sola vez**, y solo `#T-ce-confirm`
   (`data/battery_sealed.py::open_sealed`). La segunda apertura es
   rechazada por el runner, no por cortesía.
2. **Nunca se mira el sellado para diseñar el siguiente corte.**
   Ni para "ver qué familias fallan", ni para "inspirar" prosa, ni
   para calibrar thresholds. Quien haya abierto el sellado vigente
   no redacta el siguiente.
3. **Mezcla y thresholds primero, prosa después.** El manifest del
   corte N+1 (mezcla, sal del gold, umbrales de solape) se commitea
   ANTES de redactar su primer caso. Un corte ajustado tras ver un
   resultado no mide nada.
4. **Disyunción total de `variant_group`** con desarrollo Y con todos
   los sellados anteriores (assert en test, umbral 0). El solape de
   entidades/espacio se declara y se mide contra el mismo umbral
   escrito de antemano.
5. **BANKING77-77 sigue siendo diagnóstico externo**, nunca definición
   de éxito; y ningún modelo cuya mezcla publicada lo incluya se
   presenta como transferencia limpia ahí.

## Receta del corte N+1

1. Fijar `(familias, idiomas, Ks, n)` en el manifest; elegir una sal
   de gold NUEVA y registrarla (`gold_rule.salt`).
2. Precomputar los slots por grupo con la sal (script, sin prosa).
3. Redactar a mano grupo a grupo: misma lista de candidatos para
   orig/decisiva/control; el estado sostiene el slot precomputado;
   la evidencia es fragmento literal del estado.
4. Medir: validador del contrato, gold, forma contrafactual,
   distribución exacta, namespace `sealed-`, intersección 0 con
   TODO corte previo, solape bajo umbral, K de meta separados del
   diagnóstico.
5. `gate.json` con todo MEDIDO y veredicto explícito; `consumed=false`
   hasta que `#T-ce-confirm` lo abra.

## Qué hacer con el corte usado

Nada destructivo: el fichero queda como está, con su `gate.json` y su
`opened.json` (quién lo abrió y cuándo). Pasa a `evidencia histórica`
y sus cifras solo se citan junto a la decisión que informaron.
