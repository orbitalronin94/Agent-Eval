# Multi-Agent Systems & RAG Dynamics

## Propósito

Documentar una línea experimental para estudiar dinámica de sistemas multiagente, asignación de recursos, contradicciones, drift y comportamiento de pipelines RAG a partir de materiales de [Portfolio](https://github.com/orbitalronin94/Portfolio).

El portfolio presenta conceptos y prototipos relacionados con dinámica de agentes y auditoría de sistemas. Esta línea debe comenzar como inventario de hipótesis y experimentos, no como un conjunto de hechos científicos ya establecidos.

## Temas candidatos

- dinámica y estabilidad de poblaciones de agentes;
- asignación y reasignación de recursos;
- interacciones entre agentes y efectos emergentes;
- acumulación y propagación de contradicciones en RAG;
- drift temporal o entre versiones;
- olvido, retención y recuperación de información;
- auditoría de consistencia de sistemas completos;
- sensibilidad a parámetros, condiciones iniciales y topología;
- diferencias entre simulación, observación y evidencia externa.

## Unidad de datos

Cada registro debe describir un experimento reproducible:

```json
{
  "id": "multi-agent-experiment-0001",
  "system": {
    "agents": [],
    "topology": {},
    "environment": {}
  },
  "hypothesis": "<hipótesis>",
  "protocol": {
    "parameters": {},
    "seed": null,
    "repetitions": null
  },
  "observations": [],
  "metrics": {},
  "analysis": "<interpretación limitada>",
  "counterexamples": [],
  "epistemic_status": "proposed",
  "evidence_refs": [],
  "split": "unassigned"
}
```

Es un esquema ilustrativo, no datos de un experimento ejecutado.

## Clasificación de evidencia

- `implemented`: existe código para el mecanismo.
- `unit_tested`: hay tests de componentes.
- `simulation_observed`: resultado de una simulación reproducible.
- `ablated`: existe una ablación con configuración y resultados registrados.
- `externally_validated`: contrastado con datos externos documentados.
- `mathematically_derived`: derivación formal con supuestos explícitos.
- `hypothesis`: hipótesis aún no contrastada.
- `proposed`: diseño o protocolo no ejecutado.
- `unverified`: evidencia insuficiente.

No inferir que una teoría está validada porque exista un simulador o un gráfico.

## Requisitos de extracción

Antes de incluir cualquier afirmación del portfolio:

1. localizar el archivo o artefacto exacto;
2. comprobar commit y estado del código;
3. localizar pruebas, fixtures y workflows;
4. reproducir resultados cuando sea viable;
5. registrar parámetros, semillas, repeticiones y métricas;
6. distinguir resultados sintéticos de datos reales;
7. guardar limitaciones y resultados negativos;
8. comprobar si el resultado se deriva de la construcción del test.

Las afirmaciones sobre equilibrio, estabilidad, colapso, drift o mejora deben incluir las condiciones y el dominio en que se observaron.

## Métricas candidatas

Dependiendo del experimento: consistencia entre agentes, contradicciones no resueltas, retención/recuperación, estabilidad temporal, distribución de recursos, tasa de fallos, latencia, coste y sensibilidad a perturbaciones. No asumir que todas son aplicables a todos los sistemas.

## Prevención de contaminación

Separar por sistema, familia de escenarios y semilla. Mantener configuraciones y escenarios reservados para holdout. Evitar que trazas casi idénticas o ejecuciones del mismo escenario aparezcan en train y holdout.

## Estado y reservas

La línea está propuesta. Parte del material del portfolio no se ha verificado todavía a nivel de artefactos, tests y ejecuciones. Hasta completar esa revisión, los registros derivados deben permanecer como `proposed` o `unverified`, no como conocimiento validado.
