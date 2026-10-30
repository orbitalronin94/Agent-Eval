# Catálogo de líneas de investigación ML

Este directorio reúne líneas de conocimiento derivadas de proyectos públicos del autor. Cada línea mantiene su propia procedencia, esquema, criterios de evidencia y riesgos de contaminación. No se presupone que los contenidos estén listos para entrenamiento: son especificaciones de curación y extracción.

## Líneas documentadas

| Dataset | Fuente principal | Objetivo |
|---|---|---|
| [Cobol2py](cobol2py/README.md) | [cobol2py](https://github.com/orbitalronin94/cobol2py) | Semántica COBOL, reglas de traducción y equivalencia observable |
| [RONIN Knowledge & Semantics](ronin-knowledge/README.md) | [ronin-lang](https://github.com/orbitalronin94/ronin-lang) | Lenguaje formal, runtime, reglas semánticas y simulación |
| [Scientific Reasoning & Identifiability](scientific-identifiability/README.md) | [Papers](https://github.com/orbitalronin94/Papers) | Identificabilidad, degeneración paramétrica y diagnóstico científico |
| [LLM Behavioral Robustness](llm-behavioral-robustness/README.md) | [Sycophancy-audit](https://github.com/orbitalronin94/Sycophancy-audit) | Complacencia, consistencia multturno y selectividad de correcciones |
| [Adversarial Security & Prompt Injection](adversarial-security/README.md) | [Local-LLM-Red-Team](https://github.com/orbitalronin94/Local-LLM-Red-Team) | Ataques adversariales y evaluación de resistencia de modelos |
| [Multi-Agent Systems & RAG Dynamics](multi-agent-rag/README.md) | [Portfolio](https://github.com/orbitalronin94/Portfolio) | Dinámica de agentes, contradicciones, drift y sistemas RAG |

## Principios compartidos

1. **Procedencia:** cada ejemplo debe apuntar al repositorio, archivo, revisión y fragmento de origen.
2. **Estado epistemológico:** separar implementación, prueba automatizada, demostración, resultado experimental, hipótesis, propuesta y trabajo pendiente.
3. **Evidencia:** conservar tests, resultados, configuraciones y contraejemplos; no convertir una descripción en README en evidencia experimental.
4. **Licencia:** revisar la licencia del origen y de cada dependencia o corpus de terceros antes de redistribuir. La licencia MIT de este repositorio no relicencia automáticamente los materiales fuente.
5. **Separación:** agrupar variantes derivadas de un mismo caso en una partición para evitar fuga entre train, dev y holdout.
6. **Holdout protegido:** no utilizar holdouts para entrenar, ajustar prompts, seleccionar reglas ni tomar decisiones de desarrollo.
7. **Reproducibilidad:** versionar manifests, hashes, herramientas, parámetros y procedimientos de generación.
8. **No inventar etiquetas:** si una afirmación no está demostrada, su estado debe reflejarlo.

## Esquema mínimo común

Cada línea puede ampliar este esquema, pero no debería perder estos campos:

- `id`: identificador estable del ejemplo.
- `source_repo`, `source_path`, `source_revision`: procedencia.
- `source_license`: licencia verificada o `unknown`.
- `knowledge_type`: tipo de conocimiento.
- `input`, `target`: entrada y salida/afirmación objetivo, según la tarea.
- `evidence_refs`: pruebas, fragmentos o resultados que respaldan la etiqueta.
- `epistemic_status`: estado de evidencia.
- `limitations`: límites conocidos.
- `counterexamples`: contraejemplos o condiciones de fallo.
- `split`: `train`, `dev`, `holdout` o `unassigned`.

## Relación con Agent-Eval

[Agent-Eval](../README.md) proporciona un arnés general para evaluar agentes y pipelines RAG. Sus métricas basadas en LLM-juez son señales de evaluación, no ground truth infalible. Para datasets que requieran exactitud formal o equivalencia observable, deben prevalecer los oráculos y criterios específicos de cada dominio.

## Estado actual

Estos README describen objetivos, esquemas y protocolos de curación. **No afirman que los datasets estén ya extraídos, curados, licenciados para redistribución, balanceados o listos para fine-tuning.** La extracción debe hacerse línea por línea y con revisión de evidencia.
