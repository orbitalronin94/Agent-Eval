# `agent-eval.py`

> **Arnés de evaluación autocontenido para agentes y pipelines RAG. Un archivo. Sin framework obligatorio. Métricas, persistencia y reproducibilidad.**
>
> [![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
> [![CI](https://github.com/orbitalronin94/Agent-Eval/actions/workflows/ci.yml/badge.svg)](https://github.com/orbitalronin94/Agent-Eval/actions/workflows/ci.yml)
> [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
> [![Runtime dependencies](https://img.shields.io/badge/runtime-stdlib%20%2B%20PyYAML-orange.svg)](#configuración)
> [![Version](https://img.shields.io/badge/version-0.2.2-orange.svg)](#)

---

## 📌 El problema

La mayoría de los portfolios de IA son cajas negras: notebooks, demos y sistemas difíciles de reproducir.

Una demo puede demostrar que un agente responde. No demuestra necesariamente que:

* responde correctamente;
* utiliza el contexto recuperado;
* mantiene una calidad estable;
* tiene una latencia aceptable;
* cuesta lo esperado;
* puede compararse contra otra configuración;
* deja un histórico auditable de las evaluaciones.

Un pipeline de IA sin métricas cuantificadas es una prueba de concepto, no una evaluación reproducible.

## 💡 La solución

**Agent-Eval** es un arnés de evaluación pequeño y autocontenido para agentes y pipelines RAG.

El flujo es deliberadamente simple:

```text
Dataset JSONL
     │
     ▼
   Agente
     │
     ▼
Respuesta + contextos
     │
     ▼
   LLM-juez
     │
     ▼
Métricas + latencia + coste
     │
     ├──────────────► SQLite
     │
     └──────────────► Markdown report
```

El proyecto permite:

* cargar datasets JSONL;
* ejecutar agentes HTTP;
* evaluar respuestas con un LLM-juez;
* medir faithfulness, relevance, precision y recall;
* registrar latencia y coste;
* conservar las ejecuciones en SQLite;
* generar informes Markdown;
* identificar ejecuciones mediante hashes;
* repetir evaluaciones de forma comparable;
* ejecutar self-tests sin API keys ni red.

El núcleo está concentrado en **un único archivo Python**.

No requiere Docker ni un framework de evaluación específico.

---

## 🚀 Quick start

### 1. Self-tests

Los self-tests embebidos permiten comprobar el núcleo del sistema sin red, API keys ni servicios externos:

```bash
python agent-eval.py --selftest
```

### 2. Demo end-to-end

```bash
python agent-eval.py --demo
```

La demo ejecuta el pipeline completo y genera los artefactos de evaluación:

```text
eval-report.md
eval-results.db
```

### 3. Evaluar un agente HTTP

```bash
export OPENAI_API_KEY=sk-...
python agent-eval.py \
  --dataset data.jsonl \
  --agent-url http://localhost:8000/chat
```

También puedes utilizar el `Makefile`:

```bash
make selftest
make demo
make test
make cov
```

Para preparar un entorno de desarrollo:

```bash
make install
```

---

## 📦 Dataset

El dataset utiliza JSONL.

Ejemplo:

```jsonl
{"id": "q1", "question": "¿Qué es MCP?", "ground_truth": "Un estándar para conectar modelos con herramientas y fuentes externas."}
{"id": "q2", "question": "¿Qué métricas debe tener un RAG?", "ground_truth": "Faithfulness, relevance, precision, recall, latencia y coste."}
```

Cada item puede incluir:

* `id`
* `question`
* `ground_truth`

El agente debe devolver una respuesta que permita extraer al menos:

```json
{
  "answer": "...",
  "contexts": [
    "...",
    "..."
  ]
}
```

Las rutas de extracción pueden configurarse para adaptarse al formato concreto del agente.

---

## 📊 Métricas

| Métrica             | Qué evalúa                                                 |
| ------------------- | ---------------------------------------------------------- |
| `faithfulness`      | Si la respuesta está respaldada por el contexto recuperado |
| `answer_relevance`  | Si la respuesta responde a la pregunta planteada           |
| `context_precision` | Si los fragmentos recuperados son relevantes               |
| `context_recall`    | Si se recuperó la información necesaria                    |
| `latency_ms`        | Latencia de ejecución del agente                           |
| `cost_usd`          | Coste estimado de la evaluación                            |

Las métricas evaluadas mediante LLM utilizan una escala de `0.0` a `1.0`.

El coste mantiene separados:

* tokens de entrada del agente;
* tokens de salida del agente;
* tokens de entrada del juez;
* tokens de salida del juez.

Cuando el proveedor proporciona información de `usage`, se utilizan esos valores.

Cuando no existe información de uso, el sistema utiliza una estimación basada en el texto disponible.

---

## 🛡️ Resiliencia

El sistema incorpora varios mecanismos para que una evaluación completa no dependa de que todos los items sean perfectos:

* **Retry con backoff exponencial** en errores transitorios.
* Los errores HTTP `4xx` no se reintentan.
* Clasificación de errores del agente:

  * `timeout`
  * `http`
  * `parse`
  * `other`
* Validación del dataset con información de línea y campos.
* Transacciones SQLite con rollback.
* Parsing tolerante de la respuesta JSON del juez.
* Concurrencia controlada mediante `asyncio.Semaphore`.
* Los errores de un item quedan registrados sin descartar automáticamente el resto de la evaluación.
* Si falla el juez después de ejecutar el agente, se conserva la información disponible del agente.

El objetivo es que los fallos sean **datos de la evaluación**, no simplemente crashes silenciosos.

---

## 🧪 Verificación

El repositorio incluye tanto self-tests embebidos como una suite externa de `pytest`.

### Self-tests

```bash
make selftest
```

o:

```bash
python agent-eval.py --selftest
```

Los self-tests están diseñados para ejecutarse sin dependencias externas, red ni API keys.

### Demo

```bash
make demo
```

### Tests

```bash
make test
```

Equivalentemente:

```bash
python -m pytest -q --strict-markers
```

### Cobertura

```bash
make cov
```

La cobertura se genera con `pytest-cov` e incluye un informe HTML.

### Lint

```bash
make lint
```

El proyecto utiliza Ruff para comprobaciones de lint y formato.

### CI

GitHub Actions ejecuta la suite sobre:

```text
Python 3.10
Python 3.11
Python 3.12
```

El workflow también ejecuta:

1. self-tests;
2. demo end-to-end;
3. comprobación de artefactos;
4. tests con cobertura;
5. subida del informe de demo;
6. subida de cobertura cuando corresponde.

---

## 📄 Ejemplo de informe

El sistema genera un informe Markdown similar a:

```markdown
# Informe de evaluación — `gpt-4o-mini`

**Run:** `20261003-...`
**Hash del dataset:** `a3f9c1e8b742`
**Hash de configuración:** `71b82d94ce11`
**Tiempo:** 4.82 s

## Resumen ejecutivo

**Veredicto:** ✅ APTO

**Aprobados:** 3/3 (100%)

**Coste estimado:** $0.0031

## Métricas agregadas

| Métrica | Media | P50 | P95 | Mín | Máx |
|---|---:|---:|---:|---:|---:|
| `faithfulness` | 0.933 | 0.950 | 1.000 | 0.850 | 1.000 |
| `answer_relevance` | 0.900 | 0.900 | 0.950 | 0.850 | 0.950 |
| `latency_ms` | 182.4 | 175.0 | 210.5 | 160.0 | 220.0 |
| `cost_usd` | 0.0010 | 0.0009 | 0.0013 | 0.0007 | 0.0015 |
```

Cada ejecución queda asociada a:

* `run_id`;
* hash del dataset;
* hash de configuración;
* timestamp;
* resultados individuales;
* métricas agregadas.

Los fallos pueden analizarse individualmente junto con el razonamiento del juez y el tipo de error registrado.

---

## 🗄️ Persistencia e histórico

Las ejecuciones se almacenan en SQLite.

Cada ejecución tiene un `run_id` independiente.

Se conserva información como:

* configuración de la ejecución;
* hash del dataset;
* hash de configuración;
* timestamp;
* número de items;
* respuestas del agente;
* contextos;
* latencia;
* tokens de entrada y salida;
* coste;
* métricas del juez;
* razonamiento del juez;
* errores y clasificación del error.

Las ejecuciones anteriores no se sobrescriben.

Esto permite comparar diferentes ejecuciones del mismo dataset y detectar cambios en:

* calidad;
* latencia;
* coste;
* comportamiento del agente.

SQLite forma parte de la librería estándar de Python, por lo que no requiere una base de datos externa.

---

## 🔁 Reproducibilidad

Cada evaluación genera identificadores que permiten saber exactamente qué configuración produjo los resultados:

```text
Run ID:       20261003-...
Dataset hash: a3f9c1e8b742
Config hash:  71b82d94ce11
```

El hash del dataset permite detectar cambios en:

* preguntas;
* respuestas esperadas;
* datos de evaluación.

El hash de configuración permite detectar cambios en:

* modelo;
* endpoint;
* prompts;
* métricas;
* threshold;
* concurrencia;
* precios;
* configuración del agente.

Esto evita comparar como equivalentes dos ejecuciones que realmente proceden de configuraciones diferentes.

---

## ⚙️ Configuración

La configuración puede definirse mediante flags CLI y archivos JSON o YAML.

Ejemplo:

```yaml
judge:
  provider: openai-compatible
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
  model: gpt-4o-mini
  timeout_s: 60
  retries: 2
  temperature: 0.0
  max_tokens: 800
  prices:
    input_per_1m: 0.15
    output_per_1m: 0.60

agent:
  type: http
  url: http://localhost:8000/chat
  timeout_s: 60
  prices:
    input_per_1m: 0.50
    output_per_1m: 1.50
  request_template:
    question: "{question}"
  answer_path: answer
  contexts_path: contexts

dataset:
  path: dataset.jsonl
  encoding: utf-8

eval:
  metrics:
    - faithfulness
    - answer_relevance
    - context_precision
    - context_recall
  concurrency: 4
  pass_threshold: 0.7

output:
  sqlite: eval-results.db
  report: eval-report.md
```

Los precios se expresan por **1 millón de tokens** y se mantienen separados para agente y juez.

Si los precios están a `0.0`, el sistema continúa funcionando, pero el coste calculado será `0.0`.

### Dependencia YAML

El núcleo está diseñado alrededor de la librería estándar de Python.

La lectura de configuración YAML requiere:

```bash
pip install pyyaml
```

`PyYAML` está declarada como dependencia del proyecto en `pyproject.toml`.

JSON y la configuración por defecto no requieren una dependencia YAML adicional.

### Proveedores compatibles

El juez utiliza un protocolo compatible con OpenAI (`/v1/chat/completions`).

Esto permite trabajar con distintos proveedores o servidores que expongan una API compatible, por ejemplo:

* OpenAI;
* Groq;
* Together;
* Ollama;
* vLLM;
* llama.cpp.

La compatibilidad concreta depende del grado de compatibilidad del endpoint utilizado.

---

## 🧠 Decisiones de diseño

| Decisión                           | Alternativa                     | Por qué                                                |
| ---------------------------------- | ------------------------------- | ------------------------------------------------------ |
| Un solo archivo                    | Proyecto multi-módulo           | Copiar, ejecutar y auditar con mínima fricción         |
| Stdlib-first                       | `requests`, `httpx`, `tiktoken` | Reduce dependencias obligatorias y superficie de fallo |
| `asyncio` + semáforo               | `ThreadPoolExecutor`            | El cuello principal es I/O                             |
| Retry solo en errores transitorios | Reintentar todo                 | Evita repetir errores permanentes como autenticación   |
| SQLite en disco                    | Postgres, DuckDB                | Cero infraestructura externa para el caso de uso       |
| Histórico de ejecuciones           | Sobrescribir resultados         | Permite comparar evaluaciones                          |
| Hash de dataset + configuración    | —                               | Permite identificar exactamente qué se evaluó          |
| LLM-juez agnóstico                 | Framework específico            | Control del prompt y menor lock-in                     |
| Precios de juez/agente separados   | Una única tabla                 | Permite ver el coste real de cada componente           |
| Usage real cuando existe           | Estimar siempre                 | Aprovecha información proporcionada por el proveedor   |
| Estimación como fallback           | Tokenizer obligatorio           | Mantiene el diseño stdlib-first                        |
| Self-tests embebidos               | Solo pytest                     | Permite verificar el núcleo sin instalar tooling       |

La filosofía general es mantener el sistema pequeño sin sacrificar las propiedades que importan para una evaluación técnica: **trazabilidad, reproducibilidad y observabilidad**.

---

## 🔬 Casos de uso

### 1. Evaluar regresiones

Ejecutar el mismo dataset contra una nueva versión del agente y comparar:

* calidad;
* latencia;
* coste.

### 2. A/B testing de modelos

Mantener constante el dataset y cambiar:

* modelo;
* proveedor;
* prompt;
* configuración.

Los hashes permiten distinguir las ejecuciones.

### 3. Auditar agentes de terceros

Utilizar un conjunto de preguntas representativas y generar un informe reproducible de resultados.

### 4. Debug local

Utilizar servidores locales compatibles con OpenAI, como Ollama o vLLM, sin necesidad de utilizar APIs externas.

### 5. Control de costes

Separar el coste del agente y del juez para detectar evaluaciones que se vuelven progresivamente más caras.

### 6. Integración en CI

Ejecutar self-tests, tests y evaluaciones automatizadas como parte del ciclo de desarrollo.

---

## 🏗️ Filosofía

> **El mejor código no es el más listo. Es el que se puede leer, auditar y ejecutar sin fricción.**

Agent-Eval no intenta convertirse en otro framework de evaluación.

La propuesta es deliberadamente más pequeña:

```text
un archivo
+
un dataset
+
un agente
+
un juez
+
un informe
```

El objetivo es tener una herramienta suficientemente pequeña para entenderla completa y suficientemente seria para utilizarla como pieza de evaluación reproducible.

La evaluación debe dejar evidencia:

* qué se ejecutó;
* con qué configuración;
* sobre qué dataset;
* qué respondió el agente;
* cuánto tardó;
* cuánto costó;
* qué puntuó el juez.

---

## 👤 Autor

**David Ferrandez Canalis** — AI Engineer

Especializado en sistemas de IA en producción: RAG, agentes, evaluación y MLOps.

[LinkedIn](https://www.linkedin.com/in/david-ferrandez-canalis-48ab99229)

---

## 📜 Licencia

MIT.

Úsalo, modifícalo, intégralo o haz fork del proyecto.

Si encuentras un problema o tienes una mejora, abre un issue o pull request.

---

<div align="center">

**Si este proyecto te resulta útil, deja una ⭐.**

*Un archivo. Métricas reales. Persistencia. Reproducibilidad.*

</div>
