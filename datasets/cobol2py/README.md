# Línea de investigación ML: conocimiento COBOL para Cobol2py

Esta carpeta documenta una línea de trabajo para convertir el conocimiento aprendido durante el desarrollo de [Cobol2py](https://github.com/orbitalronin94/cobol2py) en recursos reutilizables para modelos de ML, sin convertir al modelo generativo en el traductor de producción.

La hipótesis de arquitectura es separar **descubrimiento**, **ejecución** y **verificación**. Cada componente tiene una responsabilidad distinta y una fuente de verdad explícita.

## Arquitectura propuesta

```text
Corpus de conocimiento COBOL ──► Modelo / pesos
             │                       │
             │                       └─ descubre patrones,
             │                          formula hipótesis y
             │                          propone reglas
             │
             ▼
Casos, contraejemplos y evidencias
                                     │
                                     ▼
                         Reglas revisadas y aceptadas
                                     │
                                     ▼
                           Traductor determinista
                                     │
                                     ▼
                         Python generado y ejecutado
                                     │
                                     ▼
                         Benchmark diferencial
                         (GnuCOBOL como oráculo)
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
               Aciertos                         Contraejemplos
                    │                                 │
                    └──────────► nuevo conocimiento ◄┘
```

### 1. Pesos del modelo: descubrir conocimiento

El modelo aporta capacidades de alto nivel:

- conocimiento lingüístico de COBOL y de sus variantes;
- razonamiento sobre construcciones y sus interacciones;
- reconocimiento de patrones en programas y fallos;
- agrupación de discrepancias por posibles causas raíz;
- formulación de hipótesis semánticas;
- propuesta de reglas, ejemplos y contraejemplos para revisión.

El objetivo de esta línea es estudiar cómo representar ese conocimiento en datasets y, cuando exista evidencia suficiente, en adaptaciones o pesos de modelos. **No se afirma que ya exista un modelo entrenado ni que los pesos contengan actualmente el traductor.**

Los pesos no deben tratarse como una fuente de verdad ejecutable: sus propuestas necesitan convertirse en reglas explícitas y superar pruebas.

### 2. Script traductor: ejecutar reglas deterministas

El traductor de [Cobol2py](https://github.com/orbitalronin94/cobol2py) es el componente que realiza la traducción efectiva.

- reglas explícitas, versionadas y revisables;
- misma entrada y misma versión producen una traducción reproducible;
- sin decisiones probabilísticas del modelo durante la traducción;
- pruebas de regresión por regla y por interacción;
- trazabilidad entre regla, caso de prueba y cambio de código.

El modelo puede proponer una regla o un parche, pero no se incorpora automáticamente: debe revisarse, implementarse y verificarse.

### 3. Benchmark: árbitro independiente

El benchmark determina si una regla mejora realmente la equivalencia.

Para Cobol2py, la referencia de comportamiento es GnuCOBOL. El criterio estricto compara la salida estándar byte a byte y el código de retorno, no solo que el código generado compile o se ejecute.

Principios:

- corpus y particiones identificados por versión y hash;
- conjuntos de desarrollo separados de los holdouts;
- mismas muestras y condiciones al comparar candidatos;
- registro de SHA, configuración, versiones y resultados;
- fallos conservados como contraejemplos;
- ninguna mejora se acepta por intuición, por compilación aislada o por una puntuación no comparable.

El benchmark debe permanecer independiente del agente que propone las mejoras. El modelo no puede modificar el oráculo, el criterio de aceptación ni los resultados de evaluación.

## Qué datasets podemos aportar

La contribución más útil no es almacenar programas sin contexto, sino construir ejemplos trazables que enseñen y permitan comprobar semántica.

| Recurso | Contenido | Uso |
|---|---|---|
| Ejemplos COBOL anotados | Fragmentos o programas, construcción implicada, dialecto y explicación | Aprender vocabulario, sintaxis y patrones |
| Pares de comportamiento | Entrada, salida de GnuCOBOL, código de retorno y entorno | Enseñar semántica observable |
| Casos mínimos | Ejemplo reducido que aísla una regla | Diagnóstico y pruebas de regresión |
| Contraejemplos | Programa que rompe una hipótesis o implementación | Evitar reglas incorrectas y sobreajuste |
| Catálogo de reglas | Regla propuesta, estado, evidencia, implementación y tests | Conectar conocimiento con código |
| Informes de benchmark | Versión del corpus, SHA, partición, aciertos y fallos | Medir progreso sin perder trazabilidad |

Cada registro debería incluir, cuando aplique: identificador estable, procedencia, licencia, dialecto/compilador, fuente COBOL, condiciones de ejecución, salida esperada, retorno esperado, etiquetas semánticas, regla relacionada y partición del dataset.

No deben incorporarse datos sin procedencia o licencia clara. Los programas completos de terceros requieren revisar sus condiciones de redistribución; cuando no se puedan redistribuir, se conservarán metadatos o ejemplos mínimos propios que no reproduzcan material protegido.

## Separación de datos y prevención de fugas

La separación entre entrenamiento, desarrollo y evaluación es esencial:

- **Train:** ejemplos que pueden utilizarse para adaptar el modelo.
- **Dev:** ejemplos para formular y comparar hipótesis durante el desarrollo.
- **Holdout:** casos reservados para medir generalización; no deben utilizarse para entrenar, ajustar prompts ni seleccionar reglas.
- **Benchmark externo:** evaluación independiente y versionada, con su criterio de aceptación congelado durante cada comparación.

Los casos derivados de un mismo programa o familia de programas deben permanecer en la misma partición cuando exista riesgo de que versiones casi idénticas aparezcan en train y holdout.

Un aumento en métricas de entrenamiento no equivale a una mejora de equivalencia. La evidencia decisiva procede del benchmark reservado.

## Flujo de contribución

1. El benchmark detecta discrepancias y conserva los casos reproducibles.
2. El agente agrupa fallos y propone una explicación semántica.
3. Se crea un caso mínimo y se contrasta la hipótesis con GnuCOBOL.
4. La regla se revisa y se implementa en el traductor determinista.
5. Se añaden pruebas de regresión y se ejecuta el benchmark comparable.
6. Se registra si la hipótesis fue confirmada, refutada o quedó inconclusa.
7. Solo entonces el ejemplo validado puede incorporarse al corpus de conocimiento, respetando su partición y procedencia.

## Estructura prevista

Esta línea puede crecer gradualmente, sin añadir pesos binarios ni datasets grandes al historial Git:

```text
datasets/cobol2py/
├── README.md                  # Este documento
├── schemas/                   # Esquemas versionados de los registros
├── manifests/                 # Inventario, procedencia y hashes de datasets
├── examples/                  # Ejemplos mínimos redistribuibles y anotaciones
├── rules/                     # Catálogo de reglas semánticas y evidencias
└── reports/                   # Índice de informes y resultados publicados
```

Los pesos, checkpoints y artefactos grandes se alojarán fuera de Git (por ejemplo, en almacenamiento de modelos/datasets apropiado), referenciados mediante identificadores inmutables, hashes, licencia y versión. Esta carpeta no presupone que esos artefactos existan ya.

## Relación con los proyectos

- [Cobol2py — traductor y benchmark diferencial](https://github.com/orbitalronin94/cobol2py)
- [Agent-Eval — arnés de evaluación y persistencia de resultados](https://github.com/orbitalronin94/Agent-Eval)

Agent-Eval puede ayudar a organizar evaluaciones de modelos o agentes, pero no sustituye al benchmark diferencial de Cobol2py: la equivalencia COBOL/Python necesita su propio oráculo y su propio criterio exacto.

## Estado

**Propuesta inicial de arquitectura y organización.** Aún no declara datasets COBOL preparados para entrenamiento, pesos adaptados ni resultados de fine-tuning. Esos hitos se añadirán únicamente cuando existan artefactos, procedencia y evaluaciones reproducibles.

La regla metodológica central es sencilla:

> El modelo descubre y propone; el traductor ejecuta; el benchmark decide.
