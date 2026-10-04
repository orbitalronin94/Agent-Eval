# `agent-eval.py`

> **Arnés de evaluación autocontenido para agentes y pipelines RAG. Un archivo. Cero dependencias. Métricas reales. Persistencia y reproducibilidad.**
> [![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
> [![CI](https://img.shields.io/badge/CI-passing-brightgreen.svg)](#)
> [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
> [![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](#)
> [![Version](https://img.shields.io/badge/version-0.2.2-orange.svg)](#)

---

## 📌 El problema

La mayoría de los portfolios de IA son cajas negras: notebooks, demos, "funciona en mi máquina". Nadie sabe si el agente alucina, si el RAG recupera basura, cuánto cuesta cada consulta o cuántos timeouts silenciosos traga el sistema.

Un pipeline de IA sin métricas cuantificadas es una prueba de concepto, no un producto.

## 💡 La solución

Un **único archivo Python** que carga un dataset JSONL, ejecuta tu agente, lo evalúa con un LLM-juez, calcula métricas, persiste en SQLite y genera un informe Markdown con tablas, percentiles y diagramas Mermaid.

Además, cada ejecución queda identificada mediante un `run_id`, hash del dataset y hash de configuración, permitiendo conservar y comparar evaluaciones históricas.

El cálculo de coste separa el consumo del **agente** y del **juez**, utilizando usage real cuando está disponible y estimaciones de tokens cuando el proveedor no lo expone.

**Sin frameworks. Sin Docker. Sin dependencias obligatorias.** Solo Python 3.10+ y la librería estándar.

---

## 🚀 Quick start

```bash
# Self-tests internos (0 deps, sin red, sin API keys, <1s)
python agent-eval.py --selftest

# Demo end-to-end (0 deps)
python agent-eval.py --demo

# Evaluar tu agente HTTP
export OPENAI_API_KEY=sk-...
python agent-eval.py --dataset data.jsonl --agent-url http://localhost:8000/chat
```

También puedes utilizar el `Makefile` incluido:

```bash
make selftest
make demo
make test
make cov
```

### Dataset (`data.jsonl`)

```jsonl
{"id": "q1", "question": "¿Qué es MCP?", "ground_truth": "Estándar de Anthropic para conectar LLMs con herramientas."}
{"id": "q2", "question": "¿Qué métricas debe tener un RAG?", "ground_truth": "Precision, recall, faithfulness, relevance, latencia y coste."}
```

Tu agente debe responder `{"answer": "...", "contexts": ["...", "..."]}`. Se pueden configurar rutas anidadas para extraer respuesta y contextos.

---

## 📊 Métricas

| Métrica             | Qué evalúa                                                       |
| ------------------- | ---------------------------------------------------------------- |
| `faithfulness`      | ¿La respuesta está anclada al contexto?                          |
| `answer_relevance`  | ¿Contesta a la pregunta?                                         |
| `context_precision` | ¿Los fragmentos recuperados eran relevantes?                     |
| `context_recall`    | ¿Se recuperó la información necesaria? (requiere `ground_truth`) |
| `latency_ms`        | Latencia end-to-end (P50, P95, mín, máx)                         |
| `cost_usd`          | Coste estimado por tokens del agente + juez                      |

Las métricas LLM se puntúan de 0.0 a 1.0 con `temperature=0` y salida JSON estricta.

El coste separa internamente tokens de entrada y salida del **agente** y del **juez**, aplicando precios independientes por millón de tokens.

Cuando el proveedor devuelve usage real, se utilizan esos valores. Cuando no están disponibles, los tokens se estiman para el cálculo de coste.

---

## 🛡️ Resiliencia

* **Retry con backoff exponencial** (`1s → 2s`) solo en errores 5xx y de red. Los 4xx no se reintentan.
* **Clasificación de errores** del agente: `timeout`, `http`, `parse`, `other`.
* **Validación de esquema** en carga del dataset, con número de línea y campos encontrados.
* **Transacciones SQLite** con rollback automático.
* **Parsing tolerante** del JSON del juez.
* **Concurrencia controlada** mediante `asyncio.Semaphore`.
* Los errores de un item se registran sin perder el resto de la evaluación.
* Si falla un juez después de haber ejecutado el agente, se conserva la respuesta y el coste ya calculado.

---

## 🧪 Verificación

El repositorio incluye un `Makefile` para ejecutar las comprobaciones habituales:

```bash
make selftest   # self-tests embebidos, <1s, 0 deps
make demo       # pipeline completo end-to-end
make test       # tests con pytest
make cov        # tests + cobertura
```

Los self-tests no requieren dependencias adicionales.

`make test` y `make cov` requieren `pytest` y `pytest-cov` instalados en el entorno de desarrollo.

El proyecto también incluye una suite de tests automatizados en `tests/` y configuración de pytest/cobertura en `pyproject.toml`.

---

## 📄 Ejemplo de informe

```markdown
# Informe de evaluación — `gpt-4o-mini`
**Run:** `20261003-...`
**Hash del dataset:** `a3f9c1e8b742`
**Hash de configuración:** `71b82d94ce11`
**Tiempo:** 4.82 s

## Resumen ejecutivo
**Veredicto:** ✅ APTO — **Aprobados:** 3/3 (100%) — **Coste estimado:** $0.0031

## Métricas agregadas
| Métrica | Media | P50 | P95 | Mín | Máx |
|---|---:|---:|---:|---:|---:|
| `faithfulness` | 0.933 | 0.950 | 1.000 | 0.850 | 1.000 |
| `answer_relevance` | 0.900 | 0.900 | 0.950 | 0.850 | 0.950 |
| `latency_ms` | 182.4 | 175.0 | 210.5 | 160.0 | 220.0 |
| `cost_usd` | 0.0010 | 0.0009 | 0.0013 | 0.0007 | 0.0015 |
```

Cada ejecución queda asociada a un `run_id`, un hash del dataset y un hash de configuración para poder identificar exactamente qué se evaluó.

Los fallos se analizan individualmente con razonamiento del juez y tipo de error.

---

## 🗄️ Persistencia e histórico

Cada ejecución se almacena en SQLite con un `run_id` único.

Se conservan:

* configuración de la ejecución;
* hash del dataset;
* hash de la configuración;
* timestamp;
* número de items;
* respuestas del agente;
* contextos;
* latencia;
* tokens de entrada y salida;
* coste;
* métricas del juez;
* razonamiento del juez;
* errores y tipo de error.

Las ejecuciones anteriores **no se sobrescriben**.

Esto permite comparar diferentes ejecuciones del mismo dataset y detectar cambios en resultados, latencia o coste.

SQLite forma parte de la librería estándar de Python, por lo que no requiere infraestructura externa.

---

## 🔁 Reproducibilidad

Cada ejecución utiliza identificadores que permiten reconstruir qué se evaluó:

```text
Run ID:       20261003-...
Dataset hash: a3f9c1e8b742
Config hash:  71b82d94ce11
```

El hash del dataset permite detectar cambios en las preguntas, respuestas esperadas o datos de evaluación.

El hash de configuración permite detectar cambios en:

* modelo;
* endpoint;
* prompts;
* métricas;
* threshold;
* concurrencia;
* precios;
* configuración del agente.

Esto evita comparar resultados como si fueran equivalentes cuando proceden de configuraciones diferentes.

---

## ⚙️ Configuración

Todo sobrescribible vía flags CLI, JSON o YAML (con `pyyaml` opcional).

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
  metrics: [faithfulness, answer_relevance, context_precision, context_recall]
  concurrency: 4
  pass_threshold: 0.7

output:
  sqlite: eval-results.db
  report: eval-report.md
```

Los precios se expresan por **1 millón de tokens** y se mantienen separados para agente y juez.

Si los precios están a `0.0`, el sistema sigue funcionando pero el coste calculado será `0.0`.

Cuando el proveedor del juez devuelve usage real, se utiliza directamente. En ausencia de usage, se aplica una estimación basada en el texto disponible.

Para el agente HTTP, el coste también puede ser estimado cuando no existe información de tokens proporcionada por el propio agente.

La configuración YAML es opcional y requiere `PyYAML`. JSON y la configuración por defecto no requieren dependencias externas.

**Jueces soportados:** cualquier endpoint compatible con OpenAI (`/v1/chat/completions`), incluyendo proveedores y servidores compatibles como OpenAI, Groq, Together, Ollama, vLLM o llama.cpp.

---

## 🧠 Decisiones de diseño

| Decisión                         | Alternativa                     | Por qué                                                                             |
| -------------------------------- | ------------------------------- | ----------------------------------------------------------------------------------- |
| Un solo archivo                  | Proyecto multi-módulo           | Fricción cero: copiar, ejecutar, auditar en 30s                                     |
| Cero dependencias runtime        | `requests`, `httpx`, `tiktoken` | Cada dep es un vector de fallo. `urllib` + `asyncio` cubren el caso                 |
| Asyncio + semáforo               | `ThreadPoolExecutor`            | El cuello es I/O, no CPU. Concurrencia controlada sin ahogar al proveedor           |
| Retry solo en 5xx/red            | Reintentar todo                 | Reintentar un 401 es tirar dinero                                                   |
| SQLite en disco                  | Postgres, DuckDB                | El informe es para humanos; SQLite para máquinas. Cero infraestructura              |
| Histórico de ejecuciones         | Sobrescribir resultados         | Permite conservar y comparar evaluaciones anteriores                                |
| Hash del dataset + configuración | —                               | Reproducibilidad: permite saber exactamente qué se evaluó                           |
| LLM-juez agnóstico               | `ragas`, `deepeval`             | Control absoluto del prompt. Sin lock-in                                            |
| Precios juez/agente separados    | Una tabla única                 | El juez suele ser barato; el agente puede ser caro. Mezclarlos oculta el coste real |
| Usage real cuando existe         | Estimar siempre                 | Aprovecha los datos proporcionados por el proveedor                                 |
| Coste estimado como fallback     | Tokenizer obligatorio           | Mantiene el proyecto stdlib-first                                                   |
| Self-tests embebidos             | Solo pytest                     | Verificable sin instalar nada en 1 segundo                                          |

---

## 🔬 Casos de uso

1. **Bloquear merges regresivos en CI** — si una métrica baja del threshold configurado, la evaluación puede marcarse como no apta.
2. **A/B testing de modelos** — mismo dataset, configuraciones diferentes y resultados comparables.
3. **Auditar agentes de terceros** — preguntas representativas, métricas y un informe reproducible.
4. **Debug local con Ollama** — sin gastar un céntimo en APIs.
5. **Controlar costes de evaluación** — separar el coste del agente del coste del juez y detectar ejecuciones que se encarecen.

---

## 🧪 Filosofía

> **El mejor código no es el más listo. Es el que se puede leer, auditar y ejecutar sin fricción.**

Un archivo. Una cosa bien hecha. Sin ecosistema de dependencias obligatorio. Eso es lo que un equipo contrata: no un framework, sino **criterio**.

La evaluación debe ser además reproducible: mismo dataset, misma configuración, resultados almacenados y coste visible.

---

## 👤 Autor

**David Ferrandez Canalis** — AI Engineer
Especializado en sistemas de IA en producción: RAG, agentes, evaluación y MLOps.

[LinkedIn](https://www.linkedin.com/in/david-ferrandez-canalis-48ab99229)

---

## 📜 Licencia

MIT. Úsalo, fork éalo, modifícalo, véndelo. Si lo mejoras, abre un PR.

<div align="center">

**Si este archivo te ha sido útil, deja una ⭐.**

*Un archivo. Cero dependencias. Métricas reales. Persistencia y reproducibilidad.*

</div>
