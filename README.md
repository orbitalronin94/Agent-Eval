# `agent-eval.py`

> **Arnés de evaluación autocontenido para agentes y pipelines RAG. Un archivo. Cero dependencias. Métricas reales.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](#)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](#)

---

## 📌 El problema

La mayoría de los proyectos de IA que se ven en portfolios son **cajas negras**: un notebook, una demo, un "funciona en mi máquina". Nadie sabe si el agente alucina, si el RAG recupera basura, cuánto cuesta cada consulta o cuántos timeouts silenciosos está tragando el sistema.

Los informes de mercado de 2026 son tajantes: **un pipeline de IA sin métricas de calidad cuantificadas se considera una prueba de concepto, no un producto**. Las empresas que pagan 90.000–150.000 € por un AI Engineer no buscan a alguien que sepa llamar a la API de OpenAI; buscan a alguien que sepa **medir, comparar y mejorar** sistemas de IA en producción.

`agent-eval.py` resuelve exactamente eso.

---

## 💡 La solución

Un **único archivo Python** que:

1. Carga un dataset de evaluación en JSONL con validación de esquema.
2. Ejecuta tu agente o pipeline RAG (HTTP, builtin o callable).
3. Evalúa cada respuesta con un **LLM-juez** (cualquier endpoint compatible con OpenAI) con **retry y backoff exponencial** ante errores 5xx o de red.
4. Calcula métricas de recuperación, generación y operativas.
5. Clasifica los errores del agente por tipo (`timeout`, `http`, `parse`, `other`).
6. Persiste los resultados en SQLite con transacciones.
7. Genera un **informe Markdown** con tablas, percentiles, veredicto y diagramas Mermaid.

**Sin frameworks. Sin Docker. Sin dependencias obligatorias. Sin API keys en el código.** Solo Python 3.10+ y la librería estándar.

---

## 🏗️ Arquitectura

```mermaid
flowchart LR
    D[Dataset JSONL<br/>validado] --> R[Runner async<br/>semáforo N]
    R --> A[Agente bajo prueba<br/>HTTP / builtin]
    A -->|answer + contexts<br/>+ error_type| J[LLM-juez<br/>retry exponencial]
    J --> M[Métricas<br/>faithfulness · relevance<br/>precision · recall]
    M --> S[(SQLite<br/>eval-results.db)]
    M --> MD[Informe Markdown<br/>eval-report.md]
    M --> V{Veredicto<br/>PASS / FAIL}
    style J fill:#f9f,stroke:#333
    style M fill:#bbf,stroke:#333
    style MD fill:#bfb,stroke:#333
```

---

## 🚀 Quick start

### Demo autocontenida (sin API key, sin servicios externos)

```bash
python agent-eval.py --demo
```

Genera `eval-report.md` y `eval-results.db` evaluando un agente léxico local contra un corpus embebido. Las métricas LLM quedarán marcadas como error si no hay `OPENAI_API_KEY`, pero el pipeline completo se ejecuta y demuestra la arquitectura.

### Evaluar tu agente HTTP

```bash
export OPENAI_API_KEY=sk-...
python agent-eval.py \
  --dataset data.jsonl \
  --agent-url http://localhost:8000/ask \
  --judge-model gpt-4o-mini \
  --output informe.md \
  --verbose
```

### Formato del dataset (`data.jsonl`)

Una línea por ítem, JSON puro. El campo `question` es **obligatorio**; si falta, el arnés aborta con un mensaje claro indicando la línea y los campos encontrados.

```jsonl
{"id": "q1", "question": "¿Qué es el protocolo MCP?", "ground_truth": "Es un estándar abierto de Anthropic para conectar LLMs con herramientas externas mediante JSON-RPC."}
{"id": "q2", "question": "¿Cómo funciona LangGraph?", "ground_truth": "Modela agentes como máquinas de estado con nodos, aristas y un checkpointer para persistir el estado."}
{"id": "q3", "question": "¿Qué métricas debe tener un pipeline RAG?", "ground_truth": "Precisión y recall de recuperación, fidelidad y relevancia de la respuesta, además de latencia y coste."}
```

Tu agente debe responder con:

```json
{
  "answer": "El protocolo MCP es...",
  "contexts": ["Fragmento 1...", "Fragmento 2..."]
}
```

Si tu esquema es distinto, cambia `response_answer_path` y `response_contexts_path` en la configuración (soporta rutas anidadas tipo `data.answer`).

---

## 📊 Qué mide

| Métrica | Qué evalúa | Por qué importa |
|---|---|---|
| **`faithfulness`** | ¿La respuesta está anclada al contexto recuperado? | Detecta alucinaciones. Sin esto, no hay confianza. |
| **`answer_relevance`** | ¿La respuesta contesta a la pregunta? | Evita respuestas evasivas o fuera de tema. |
| **`context_precision`** | ¿Los fragmentos recuperados eran relevantes? | Un buen RAG empieza por una buena recuperación. |
| **`context_recall`** | ¿Se recuperó toda la información necesaria? | Requiere `ground_truth` en el dataset. |
| **`latency_ms`** | Latencia de extremo a extremo. | P50, P95, mín, máx. La experiencia de usuario es medible. |
| **`cost_usd`** | Coste estimado por tokens (juez + agente, precios separados). | La sostenibilidad del producto depende de esto. |

Todas las métricas LLM se puntúan de **0.0 a 1.0** usando un juez con `temperature=0` y salida JSON estricta.

### Contabilidad de coste honesta

El coste del juez y el del agente se calculan con **tablas de precios independientes**:

- El juez usa `MODEL_PRICES` (editable al inicio del archivo).
- El agente usa `agent.prices` en la config (`{"in": 0.0, "out": 0.0}` por defecto).
- Si no configuras los precios del agente, se asume coste cero **pero se sigue midiendo el coste del juez**, que es real.
- El coste total solo se calcula si `cost_usd` está en `eval.metrics`. No se factura trabajo que no se pide.

La estimación de tokens usa `TOKEN_CHAR_RATIO = 3.5`, ajustado para español (el estándar `4.0` subestima sistemáticamente en idiomas con palabras más largas).

---

## 🛡️ Resiliencia

El arnés está diseñado para no caerse cuando la red falla o el juez devuelve basura.

### Reintentos con backoff exponencial

`LLMClient.complete` reintenta hasta `MAX_JUDGE_RETRIES = 2` veces ante:

- Errores HTTP 5xx del juez.
- Errores de red (`URLError`).
- Timeouts.

El backoff es exponencial: `1s → 2s`. Los errores 4xx (401, 429, 400) **no se reintentan**: son fallos del cliente y reintentar solo empeora la factura.

### Clasificación de errores del agente

Cada respuesta fallida del agente se etiqueta con `error_type`:

| Tipo | Causa | Aparece en el informe como |
|---|---|---|
| `timeout` | Timeout de red | Error del agente (timeout) |
| `http` | Error HTTP del agente | Error del agente (http) |
| `parse` | Respuesta no es JSON válido | Error del agente (parse) |
| `other` | Cualquier otro fallo | Error del agente (desconocido) |

Esto permite distinguir en el informe final si un fallo fue culpa del agente, de la red, o de un contrato de API roto.

### Validación de esquema en carga

`load_dataset` valida que cada línea tenga el campo `question`. Si falta, aborta con un mensaje que incluye el número de línea y los campos encontrados, en lugar de lanzar un `KeyError` genérico.

### Transacciones SQLite

`persist_results` envuelve toda la escritura en una transacción. Si falla a mitad, hace rollback y no deja la base de datos en estado inconsistente.

---

## 📄 Ejemplo de informe generado

El archivo `eval-report.md` que produce el arnés tiene esta pinta:

````markdown
# Informe de evaluación — `gpt-4o-mini`

**Generado:** 2026-10-03T14:22:07+00:00
**Versión del arnés:** `agent-eval 0.1.0`
**Hash del dataset:** `a3f9c1e8b742`
**Tiempo total:** 4.82 s

## 1. Resumen ejecutivo

**Veredicto:** ✅ APTO
**Ítems evaluados:** 3
**Aprobados (umbral 0.75):** 3/3 (100.0%)
**Coste total:** $0.0031
**Errores del agente:** 0

## 4. Métricas agregadas

| Métrica | Media | P50 | P95 | Mín | Máx |
|---|---:|---:|---:|---:|---:|
| `faithfulness` | 0.933 | 0.950 | 1.000 | 0.850 | 1.000 |
| `answer_relevance` | 0.900 | 0.900 | 0.950 | 0.850 | 0.950 |
| `context_precision` | 0.867 | 0.850 | 0.950 | 0.800 | 0.950 |
| `latency_ms` | 182.4 | 175.0 | 210.5 | 160.0 | 220.0 |
| `cost_usd` | 0.0010 | 0.0009 | 0.0013 | 0.0007 | 0.0015 |
````

Cada ítem se detalla con ID, estado, latencia, coste y puntuación por métrica. Los fallos se analizan individualmente con el razonamiento del juez y el tipo de error.

---

## ⚙️ Configuración

Por defecto, `agent-eval.py` funciona sin configuración. Pero todo es sobrescribible desde:

1. **Flags de CLI** (`--dataset`, `--agent-url`, `--judge-model`, `--verbose`, etc.)
2. **Archivo JSON** (`--config eval.json`)
3. **Archivo YAML** (`--config eval.yaml`, requiere `pyyaml` opcional)

### Ejemplo de `eval.yaml`

```yaml
judge:
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 512

agent:
  type: http
  url: http://localhost:8000/ask
  method: POST
  request_template:
    question: "{question}"
    top_k: 5
  response_answer_path: "data.answer"
  response_contexts_path: "data.retrieved_contexts"
  timeout: 60.0
  prices:
    in: 0.0005    # USD por 1K tokens de entrada del agente
    out: 0.0015   # USD por 1K tokens de salida del agente

dataset:
  path: data/golden_set.jsonl
  id_field: id
  question_field: question
  ground_truth_field: ground_truth

eval:
  metrics: [faithfulness, answer_relevance, context_precision, latency_ms, cost_usd]
  concurrency: 4
  pass_threshold: 0.75
  save_sqlite: true
  token_char_ratio: 3.5

output:
  report_path: reports/2026-10-03.md
  include_per_question: true
  include_mermaid: true
  include_raw_judge: false
```

### Modelos de juez soportados

Cualquier endpoint compatible con OpenAI `/v1/chat/completions`:

| Proveedor | `base_url` | Modelo ejemplo |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b` |
| Together | `https://api.together.xyz/v1` | `qwen-2.5-72b` |
| Ollama (local) | `http://localhost:11434/v1` | `llama3.3` |
| vLLM / llama.cpp | `http://localhost:8080/v1` | tu modelo |

La tabla de precios (`MODEL_PRICES`) es editable al inicio del archivo. Si el modelo no está listado, se asume coste 0 **para el juez**, pero se advierte en logs si el coste es una métrica solicitada.

---

## 🧠 Decisiones de diseño

Cada decisión de este archivo tiene un porqué. Estas son las que importan:

### 1. Un solo archivo

**Alternativa considerada:** proyecto multi-módulo con `pyproject.toml`, `src/`, tests separados.

**Por qué no:** la fricción de instalación mata la adopción. Un archivo se copia, se ejecuta y se audita en 30 segundos. El objetivo es que un CTO pueda leer todo el código en una sentada y confiar en él. La complejidad se concentra, no se dispersa.

### 2. Cero dependencias obligatorias

**Alternativa considerada:** usar `requests`, `httpx`, `tiktoken`, `openai`, `pydantic`.

**Por qué no:** cada dependencia es un vector de fallo, una versión que rompe, una incompatibilidad de Python. `urllib.request` + `asyncio.to_thread` cubren el 100% del caso de uso. La estimación de tokens por longitud de caracteres (`len/3.5`) es suficiente para medir coste con un error del ±15%, y elimina la necesidad de tokenizers específicos por modelo.

### 3. Asincronía con semáforo

**Alternativa considerada:** `ThreadPoolExecutor` o `multiprocessing`.

**Por qué no:** el cuello de botella es I/O de red (llamadas HTTP al juez y al agente), no CPU. `asyncio` es el modelo natural. El semáforo limita la concurrencia sin ahogar al proveedor del LLM ni disparar la factura por rate limits.

### 4. Retry solo en errores recuperables

**Alternativa considerada:** reintentar todo, o no reintentar nada.

**Por qué no:** reintentar un 401 o un 400 es tirar dinero. Reintentar un 500 o un timeout es lo que salva una evaluación de 200 ítems cuando el proveedor tiene un hipo. `MAX_JUDGE_RETRIES = 2` con backoff `1s → 2s` es el punto dulce: cubre el 95% de los fallos transitorios sin bloquear el pipeline más de 3 segundos por ítem.

### 5. SQLite en disco (no Postgres, no DuckDB)

**Alternativa considerada:** guardar solo el informe Markdown, o usar una base de datos externa.

**Por qué no:** el informe es para humanos; SQLite es para máquinas. Permite hacer consultas posteriores ("¿qué ítems fallaron más de 3 veces esta semana?") sin montar infraestructura. Un solo archivo, cero servidores. La escritura va en transacción para no dejar estados corruptos.

### 6. LLM-juez agnóstico

**Alternativa considerada:** usar `ragas`, `deepeval` o `promptfoo` como dependencia.

**Por qué no:** cada framework de evaluación impone su propio paradigma, sus propias métricas y su propio formato de dataset. `agent-eval` implementa las cuatro métricas canónicas del RAG con prompts explícitos y transparentes. Si quieres cambiar el prompt del juez, lo editas y ya está. Control absoluto.

### 7. Parsing tolerante del JSON del juez

Los LLMs no siempre devuelven JSON limpio. `_extract_json` acepta bloques con ` ```json ` fences, texto antes y después, y objetos anidados. Cuando falla, se registra en logs con la respuesta cruda truncada a 200 caracteres para depurar sin llenar el disco.

### 8. Reproducibilidad por hash

Cada informe incluye el `sha256` corto del dataset serializado. Si cambias una coma de una pregunta, el hash cambia y sabes que no puedes comparar el informe nuevo con el viejo. Esto es lo que separa una evaluación de una demo.

### 9. Logging estructurado en lugar de `print`

**Alternativa considerada:** `print()` a stderr.

**Por qué no:** el logging con timestamps y niveles permite filtrar con `grep`, redirigir a un archivo con `tee`, y activar modo debug con `--verbose`. Un pipeline de CI/CD necesita logs parseables, no líneas sueltas.

### 10. Precios separados para juez y agente

**Alternativa considerada:** una sola tabla de precios.

**Por qué no:** el juez suele ser un modelo pequeño y barato (`gpt-4o-mini`); el agente puede ser un modelo grande y caro. Mezclarlos produce un coste total que no refleja ninguna realidad. Separarlos permite ver exactamente cuánto cuesta evaluar (juez) y cuánto cuesta operar (agente).

---

## 🔬 Casos de uso reales

### 1. Evaluar un cambio de prompt en producción

Antes de desplegar un cambio en el system prompt de tu agente, corres `agent-eval` contra el golden set. Si `faithfulness` baja de 0.85, el pipeline de CI/CD bloquea el merge.

```bash
python agent-eval.py --dataset golden_set.jsonl --output reports/pr-1234.md
```

### 2. Comparar dos modelos (A/B testing)

Corres el mismo dataset contra dos modelos y comparas los informes:

```bash
python agent-eval.py --dataset gs.jsonl --judge-model gpt-4o-mini --output report-gpt.md
python agent-eval.py --dataset gs.jsonl --judge-model claude-3-5-sonnet --output report-claude.md
```

### 3. Detectar regresiones en un RAG

Integrado en un workflow de GitHub Actions, `agent-eval` corre en cada PR que toque el pipeline RAG. Si la precisión de recuperación baja más del 5%, el PR falla. El `error_type` en el informe ayuda a distinguir regresiones reales de fallos transitorios de red.

### 4. Auditar un agente de terceros

Le das la URL de cualquier agente HTTP y un dataset de 50 preguntas representativas. En 5 minutos tienes un informe de si alucina, cuánto cuesta, si es rápido, y qué porcentaje de sus fallos son timeouts frente a errores reales.

### 5. Debug local con Ollama

Sin gastar un céntimo en APIs:

```bash
ollama serve &
python agent-eval.py \
  --dataset data.jsonl \
  --agent-url http://localhost:8000/ask \
  --config config/ollama.yaml  # judge.base_url = http://localhost:11434/v1
```

---

## 📈 Roadmap

- [ ] **v0.2** — Soporte para métricas personalizadas vía función Python externa (`--metric-plugin path.py`).
- [ ] **v0.3** — Exportación a CSV y JSONL además de Markdown y SQLite.
- [ ] **v0.4** — Modo `--compare` para diff entre dos informes.
- [ ] **v0.5** — Integración nativa con GitHub Actions como reusable workflow.
- [ ] **v0.6** — Soporte de `--retries` y `--retry-delay` configurables por CLI.
- [ ] **v1.0** — Tests unitarios con cobertura >90% y publicación en PyPI (manteniendo el archivo único).

Las contribuciones son bienvenidas, especialmente forks que demuestren métricas adicionales (toxicity, bias, PII leakage) manteniendo la filosofía de un solo archivo.

---

## 🧪 Filosofía del proyecto

> **El mejor código no es el más listo. Es el que se puede leer, auditar y ejecutar sin fricción.**

Este archivo es una respuesta a una pregunta concreta: **¿cómo demuestras que un agente de IA es fiable sin montar un servicio?**

La respuesta es un archivo. Un archivo que hace una cosa y la hace bien. Un archivo que cualquier ingeniero puede leer entero en una sentada y confiar en él. Un archivo que se puede copiar en cualquier proyecto sin arrastrar un ecosistema de dependencias.

Eso es lo que un equipo contrata: no un framework, sino **criterio**.

---

## 👤 Autor

**David Ferrandez Canalis** — AI Engineer
Especializado en sistemas de IA en producción: RAG, agentes, evaluación y MLOps.

- LinkedIn: [david-ferrandez-canalis](https://www.linkedin.com/in/david-ferrandez-canalis-48ab99229)



---

## 📜 Licencia

MIT. Úsalo, fork éalo, modifícalo, véndelo. Solo pido que si lo mejoras, abras un PR.

---

<div align="center">

**Si este archivo te ha sido útil, deja una ⭐.**

*Un archivo. Cero dependencias. Métricas reales.*

</div>
