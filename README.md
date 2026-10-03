# `agent-eval.py`

> **Arnés de evaluación autocontenido para agentes y pipelines RAG. Un archivo. Cero dependencias. Métricas reales.**

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](#)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](#)

---

## 📌 El problema

La mayoría de los portfolios de IA son cajas negras: notebooks, demos, "funciona en mi máquina". Nadie sabe si el agente alucina, si el RAG recupera basura, cuánto cuesta cada consulta o cuántos timeouts silenciosos traga el sistema.

Un pipeline de IA sin métricas cuantificadas es una prueba de concepto, no un producto.

## 💡 La solución

Un **único archivo Python** que carga un dataset JSONL, ejecuta tu agente, lo evalúa con un LLM-juez, calcula métricas, persiste en SQLite y genera un informe Markdown con tablas, percentiles y diagramas Mermaid.

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
python agent-eval.py --dataset data.jsonl --agent-url http://localhost:8000/ask
```

### Dataset (`data.jsonl`)

```jsonl
{"id": "q1", "question": "¿Qué es MCP?", "ground_truth": "Estándar de Anthropic para conectar LLMs con herramientas."}
{"id": "q2", "question": "¿Qué métricas debe tener un RAG?", "ground_truth": "Precision, recall, faithfulness, relevance, latencia y coste."}
```

Tu agente debe responder `{"answer": "...", "contexts": ["...", "..."]}`. Rutas anidadas soportadas (`data.answer`).

---

## 📊 Métricas

| Métrica | Qué evalúa |
|---|---|
| `faithfulness` | ¿La respuesta está anclada al contexto? |
| `answer_relevance` | ¿Contesta a la pregunta? |
| `context_precision` | ¿Los fragmentos recuperados eran relevantes? |
| `context_recall` | ¿Se recuperó la información necesaria? (requiere `ground_truth`) |
| `latency_ms` | Latencia end-to-end (P50, P95, mín, máx) |
| `cost_usd` | Coste por tokens (juez + agente con precios separados) |

Todas las métricas LLM se puntúan de 0.0 a 1.0 con `temperature=0` y salida JSON estricta.

---

## 🛡️ Resiliencia

- **Retry con backoff exponencial** (`1s → 2s`) solo en errores 5xx y de red. Los 4xx no se reintentan.
- **Clasificación de errores** del agente: `timeout`, `http`, `parse`, `other`.
- **Validación de esquema** en carga del dataset, con número de línea y campos encontrados.
- **Transacciones SQLite** con rollback automático.
- **Parsing tolerante** del JSON del juez (fences, texto alrededor, anidamiento).

---

## 🧪 Verificación

```bash
make selftest   # 40+ checks embebidos, <1s, 0 deps
make demo       # pipeline completo end-to-end
make test       # ~90 tests con pytest
make cov        # cobertura
```

**CI:** GitHub Actions corre self-tests + demo + pytest con cobertura en Python 3.10, 3.11 y 3.12.

---

## 📄 Ejemplo de informe

````markdown
# Informe de evaluación — `gpt-4o-mini`
**Hash del dataset:** `a3f9c1e8b742` | **Tiempo:** 4.82 s

## Resumen ejecutivo
**Veredicto:** ✅ APTO — **Aprobados:** 3/3 (100%) — **Coste:** $0.0031

## Métricas agregadas
| Métrica | Media | P50 | P95 | Mín | Máx |
|---|---:|---:|---:|---:|---:|
| `faithfulness` | 0.933 | 0.950 | 1.000 | 0.850 | 1.000 |
| `answer_relevance` | 0.900 | 0.900 | 0.950 | 0.850 | 0.950 |
| `latency_ms` | 182.4 | 175.0 | 210.5 | 160.0 | 220.0 |
| `cost_usd` | 0.0010 | 0.0009 | 0.0013 | 0.0007 | 0.0015 |
````

Los fallos se analizan individualmente con razonamiento del juez y tipo de error.

---

## ⚙️ Configuración

Todo sobrescribible vía flags CLI, JSON o YAML (con `pyyaml` opcional).

```yaml
judge:
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
  model: gpt-4o-mini

agent:
  type: http
  url: http://localhost:8000/ask
  response_answer_path: "data.answer"
  response_contexts_path: "data.retrieved_contexts"
  prices: { in: 0.0005, out: 0.0015 }

eval:
  metrics: [faithfulness, answer_relevance, context_precision, latency_ms, cost_usd]
  concurrency: 4
  pass_threshold: 0.75
  token_char_ratio: 3.5   # ajustado para español

output:
  report_path: reports/2026-10-03.md
```

**Jueces soportados:** cualquier endpoint compatible con OpenAI (`/v1/chat/completions`) — OpenAI, Groq, Together, Ollama, vLLM, llama.cpp.

---

## 🧠 Decisiones de diseño

| Decisión | Alternativa | Por qué |
|---|---|---|
| Un solo archivo | Proyecto multi-módulo | Fricción cero: copiar, ejecutar, auditar en 30s |
| Cero dependencias | `requests`, `httpx`, `tiktoken` | Cada dep es un vector de fallo. `urllib` + `asyncio.to_thread` cubren el caso |
| Asyncio + semáforo | `ThreadPoolExecutor` | El cuello es I/O, no CPU. Concurrencia controlada sin ahogar al proveedor |
| Retry solo en 5xx/red | Reintentar todo | Reintentar un 401 es tirar dinero |
| SQLite en disco | Postgres, DuckDB | El informe es para humanos; SQLite para máquinas. Cero infraestructura |
| LLM-juez agnóstico | `ragas`, `deepeval` | Control absoluto del prompt. Sin lock-in |
| `TOKEN_CHAR_RATIO = 3.5` | 4.0 (estándar inglés) | El español tiene palabras más largas; 4.0 subestima |
| Precios juez/agente separados | Una tabla única | El juez suele ser barato; el agente puede ser caro. Mezclarlos oculta el coste real |
| Hash del dataset | — | Reproducibilidad: cambias una coma y el informe queda invalidado |
| Self-tests embebidos | Solo pytest | Verificable sin instalar nada en 1 segundo |

---

## 🔬 Casos de uso

1. **Bloquear merges regresivos en CI** — si `faithfulness` baja de 0.85, el PR falla.
2. **A/B testing de modelos** — mismo dataset, dos jueces, dos informes.
3. **Auditar agentes de terceros** — 50 preguntas representativas, informe en 5 minutos.
4. **Debug local con Ollama** — sin gastar un céntimo en APIs.

---

## 🧪 Filosofía

> **El mejor código no es el más listo. Es el que se puede leer, auditar y ejecutar sin fricción.**

Un archivo. Una cosa bien hecha. Sin ecosistema de dependencias. Eso es lo que un equipo contrata: no un framework, sino **criterio**.

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

*Un archivo. Cero dependencias. Métricas reales.*

</div>
