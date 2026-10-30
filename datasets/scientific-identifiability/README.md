# Scientific Reasoning & Identifiability

## Propósito

Crear ejemplos trazables de razonamiento científico y diagnóstico de identificabilidad a partir de los documentos de [Papers](https://github.com/orbitalronin94/Papers).

El foco inicial es la identificabilidad estructural y práctica, la degeneración paramétrica, la matriz de información de Fisher (FIM), SVD, sensibilidad y protocolos de diagnóstico en modelos como Hill y PBPK.

## Temas candidatos

- Diferenciar identificabilidad estructural de identificabilidad práctica.
- Reconocer parámetros no identificables en un régimen de observación concreto.
- Diagnosticar degeneraciones y correlaciones paramétricas.
- Interpretar FIM, valores singulares y condicionamiento.
- Identificar qué mediciones adicionales podrían romper una degeneración.
- Distinguir falta de información experimental de un fallo del optimizador.
- Diseñar protocolos de diagnóstico antes del ajuste.
- Explicar los límites de inferencia y las condiciones bajo las que una conclusión es válida.

## Unidad de datos

Un registro debe representar un problema y su cadena de evidencia, no solo un párrafo extraído:

```json
{
  "id": "identifiability-case-0001",
  "model_family": "Hill",
  "observation_regime": "<régimen>",
  "question": "<pregunta de diagnóstico>",
  "observations": {},
  "analysis": {
    "method": [],
    "diagnosis": "<conclusión limitada al dominio>"
  },
  "evidence_refs": [],
  "counterexamples": [],
  "epistemic_status": "pending_verification",
  "limitations": [],
  "split": "unassigned"
}
```

Este objeto es una plantilla; no es un resultado científico ni contiene valores medidos.

## Estado epistemológico obligatorio

Los documentos de origen son manuscritos/autopublicaciones del repositorio y no deben etiquetarse automáticamente como resultados revisados por pares. Distinguir:

- `mathematically_derived`: derivación matemática reproducible, con hipótesis explícitas.
- `synthetic_tested`: probado con datos sintéticos y condiciones registradas.
- `externally_validated`: validación con datos externos documentados.
- `peer_reviewed`: revisión por pares comprobada.
- `proposed_protocol`: protocolo propuesto, aún no validado.
- `open_question`: cuestión pendiente.
- `contradicted`: resultado refutado o inconsistente.
- `unverified`: no hay evidencia suficiente.

No inferir validación externa ni revisión por pares a partir de una publicación en GitHub.

## Límites y reservas que no deben perderse

La propia documentación identifica asuntos pendientes, como la validación externa con logs reales y la falta de un modo de perturbación `trend`. También advierte que ciertos tests de fontanería pueden alcanzar el 100 % por construcción, lo que no equivale a validar una hipótesis científica.

Por tanto:

- conservar el diseño y los supuestos del experimento;
- diferenciar tests de infraestructura de pruebas de hipótesis;
- guardar datos, código, semillas y parámetros cuando estén disponibles;
- indicar explícitamente el dominio de validez;
- incluir contraejemplos y resultados negativos;
- no presentar umbrales o conclusiones particulares como leyes universales.

## Tareas de entrenamiento/evaluación

1. Clasificar si un problema es de identificabilidad estructural o práctica.
2. Detectar afirmaciones que exceden la información disponible.
3. Proponer un diagnóstico y especificar qué evidencia falta.
4. Interpretar resultados de FIM/SVD sin ocultar supuestos.
5. Diseñar una medición o perturbación que discrimine hipótesis.
6. Rechazar conclusiones no identificables y explicar por qué.

## Procedencia y derechos

Guardar ruta, commit, título y fragmento de origen. Revisar licencia y derechos antes de redistribuir texto completo. Si no hay licencia clara, no copiar el manuscrito al dataset; utilizar referencias, metadatos y ejemplos nuevos que no reproduzcan material protegido.

## Estado

Especificación de curación. Los documentos son fuentes de hipótesis y protocolos; cada afirmación científica requiere validación independiente apropiada a su alcance.
