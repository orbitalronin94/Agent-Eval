# LLM Behavioral Robustness

## Propósito

Documentar ejemplos para estudiar complacencia (sycophancy), estabilidad de respuestas, presión social y selectividad de correcciones a partir de [Sycophancy-audit](https://github.com/orbitalronin94/Sycophancy-audit).

El objetivo no es premiar la terquedad. Un modelo robusto debe resistir una corrección falsa y aceptar una corrección verdadera.

## Familias de escenarios

La implementación de origen incluye pruebas con nombres como:

- `sycophancy_baseline`
- `sycophancy_modifiers`
- `sycophancy_multiturn`
- `correction_selectivity`
- `frame_divergence`
- `context_invariance`
- `refusal_framing`

El inventario y la definición exacta de cada prueba deben extraerse de la revisión del código que se utilice como fuente. No reconstruir escenarios solo a partir del nombre.

## Estructura de ejemplo

```json
{
  "id": "behavioral-robustness-0001",
  "scenario_family": "<familia>",
  "turns": [
    {"role": "user", "content": "<pregunta inicial>"},
    {"role": "assistant", "content": "<respuesta inicial>"},
    {"role": "user", "content": "<presión o corrección posterior>"}
  ],
  "ground_truth": {
    "claim": "<hecho verificable>",
    "source": "<referencia>"
  },
  "expected_behavior": "<criterio>",
  "observed_behavior": "<respuesta observada>",
  "label": "unreviewed",
  "evidence_refs": [],
  "sampling": {},
  "split": "unassigned"
}
```

Es una plantilla, no una conversación real.

## Etiquetado

Separar al menos:

- corrección factual válida;
- corrección factual falsa;
- presión social sin nueva evidencia;
- presión de autoridad;
- halago o desaprobación;
- cambio de marco;
- variación de contexto;
- respuesta estable justificada;
- cambio de respuesta justificado;
- cambio de respuesta sin evidencia;
- resultado ambiguo.

Cada etiqueta debe basarse en un ground truth independiente cuando la tarea sea factual. Para preferencias, valores o preguntas abiertas, especificar qué criterio permite evaluar la respuesta sin imponer una única opinión.

## Métricas y cautelas

Registrar por separado:

- tasa de cambio tras presión sin evidencia;
- tasa de aceptación de correcciones verdaderas;
- tasa de rechazo de correcciones falsas;
- consistencia entre turnos;
- calidad factual de la respuesta final;
- variación entre muestras y configuración;
- abstenciones y errores de ejecución.

No usar una puntuación heurística como verdad de referencia. Las etiquetas automáticas requieren revisión humana y medición de falsos positivos/falsos negativos. Una respuesta que no cambia puede seguir siendo incorrecta; una respuesta que cambia puede estar corrigiéndose adecuadamente.

## Protocolo experimental

- fijar modelo, versión, prompt, temperatura, semilla si está disponible y número de repeticiones;
- mantener constantes los pares de escenarios salvo por la variable manipulada;
- aleatorizar el orden cuando corresponda;
- conservar transcripciones completas;
- evaluar con criterios independientes y ciegos cuando sea posible;
- informar incertidumbre y variabilidad, no solo medias;
- mantener holdout de escenarios y paráfrasis.

## Evidencia y estado

La presencia de una prueba en el repositorio demuestra que el escenario está implementado, no que el constructo psicológico o conductual esté validado externamente. Registrar por separado `implemented`, `self_tested`, `human_reviewed`, `externally_validated` y `unverified`.

## Estado

Especificación del corpus. Los escenarios y pruebas fuente requieren extracción, revisión y validación antes de producir etiquetas de entrenamiento.
