agent-eval.py

Arnés de evaluación autocontenido para agentes y pipelines RAG. Un archivo. Cero dependencias obligatorias. Métricas, persistencia y reproducibilidad.

￼
￼
￼
￼
￼

📌 El problema

La mayoría de los portfolios de IA son cajas negras: notebooks, demos, "funciona en mi máquina". Nadie sabe si el agente alucina, si el RAG recupera basura, cuánto cuesta cada consulta, qué latencia tiene o si una nueva versión ha empeorado respecto a una evaluación anterior.

Un pipeline de IA sin métricas cuantificadas es una prueba de concepto, no un producto.

💡 La solución

agent-eval.py es un arnés de evaluación autocontenido que:

carga un dataset JSONL;

ejecuta tu agente;

evalúa las respuestas con un LLM-juez;

calcula métricas de calidad, latencia y coste;

controla la concurrencia;

persiste los resultados en SQLite;

conserva el histórico de ejecuciones;

identifica cada dataset y configuración mediante hashes;

genera un informe Markdown reproducible;

funciona con Python 3.10+ y la librería estándar.

Sin frameworks. Sin Docker. Sin dependencias obligatorias.

🚀 Quick start

# Self-tests internos: sin red, sin API keys python agent-eval.py --selftest # Demo end-to-end python agent-eval.py --demo # Evaluar tu agente HTTP export OPENAI_API_KEY=sk-... python agent-eval.py --dataset data.jsonl --agent-url http://localhost:8000/ask 

Dataset (data.jsonl)

{"id": "q1", "question": "¿Qué es MCP?", "ground_truth": "Estándar de Anthropic para conectar LLMs con herramientas."} {"id": "q2", "question": "¿Qué métricas debe tener un RAG?", "ground_truth": "Precision, recall, faithfulness, relevance, latencia y coste."} 

Cada item necesita como mínimo:

{ "id": "q1", "question": "¿Qué es MCP?" } 

ground_truth es opcional salvo para métricas que necesiten referencia.

El agente HTTP debe devolver una estructura equivalente a:

{ "answer": "...", "contexts": ["...", "..."] } 

También se soportan rutas anidadas mediante configuración, por ejemplo:

data.answer data.retrieved_contexts 

📊 Métricas

MétricaQué evalúafaithfulness¿La respuesta está respaldada por el contexto recuperado?answer_relevance¿La respuesta contesta realmente a la pregunta?context_precision¿Los fragmentos recuperados son relevantes?context_recall¿Se recuperó la información necesaria? Requiere ground_truth.latency_msLatencia end-to-end de la ejecución del agente.cost_usdCoste estimado de tokens del agente + juez.

Las métricas LLM se puntúan de 0.0 a 1.0.

El LLM-juez utiliza temperature=0 y solicita salida JSON estructurada para reducir variabilidad y facilitar el parsing.

Coste por tokens

El coste se calcula separando:

coste del agente: tokens de entrada + salida del agente;

coste del juez: tokens de entrada + salida de cada evaluación del juez;

coste total: agente + juez.

Los precios se expresan por millón de tokens:

prices: input_per_1m: 0.50 output_per_1m: 1.50 

Cuando el proveedor del LLM devuelve información de uso, se utiliza ese dato. Cuando no está disponible, el sistema utiliza una estimación basada en caracteres y token_char_ratio.

Por ello, el coste mostrado debe interpretarse como estimación, especialmente para agentes HTTP que no exponen usage/token counts.

🗄️ Persistencia e histórico

Cada ejecución genera un run_id único y queda registrada en SQLite.

La base de datos conserva:

configuración de la ejecución;

hash del dataset;

hash de la configuración;

timestamp;

número de items;

resultados individuales;

respuestas del agente;

contextos;

latencia;

tokens de entrada y salida;

coste;

métricas del juez;

razonamiento del juez;

errores y tipo de error.

Las ejecuciones anteriores no se sobrescriben.

Esto permite comparar evaluaciones realizadas con diferentes versiones del agente, datasets o configuraciones.

La persistencia utiliza únicamente SQLite incluido en Python.

🔁 Reproducibilidad

Cada ejecución identifica el estado de entrada mediante hashes:

Dataset hash: a3f9c1e8b742 Config hash: 71b82d94ce11 Run ID: 20261004-... 

El hash del dataset permite detectar cambios en las preguntas, respuestas esperadas o contextos.

El hash de configuración permite detectar cambios en:

modelo;

endpoint;

prompts;

métricas;

threshold;

concurrencia;

precios;

configuración del agente.

Esto evita comparar resultados como si fueran equivalentes cuando realmente proceden de configuraciones distintas.

📄 Ejemplo de informe

# Informe de evaluación — `gpt-4o-mini` **Run:** `20261004-...` **Hash del dataset:** `a3f9c1e8b742` **Hash de configuración:** `71b82d94ce11` ## Resumen ejecutivo **Veredicto:** ✅ APTO **Aprobados:** 3/3 (100%) **Coste estimado:** $0.0031 ## Métricas agregadas | Métrica | Media | P50 | P95 | Mín | Máx | |---|---:|---:|---:|---:|---:| | `faithfulness` | 0.933 | 0.950 | 1.000 | 0.850 | 1.000 | | `answer_relevance` | 0.900 | 0.900 | 0.950 | 0.850 | 0.950 | | `latency_ms` | 182.4 | 175.0 | 210.5 | 160.0 | 220.0 | | `cost_usd` | 0.0010 | 0.0009 | 0.0013 | 0.0007 | 0.0015 | 

El informe también incluye los fallos individuales, el razonamiento del juez y la clasificación de errores.

⚙️ Configuración

La configuración puede proporcionarse mediante los parámetros disponibles en CLI.

Ejemplo conceptual:

judge: base_url: https://api.openai.com/v1 api_key_env: OPENAI_API_KEY model: gpt-4o-mini prices: input_per_1m: 0.15 output_per_1m: 0.60 agent: type: http url: http://localhost:8000/ask response_answer_path: "data.answer" response_contexts_path: "data.retrieved_contexts" prices: input_per_1m: 0.50 output_per_1m: 1.50 eval: metrics: - faithfulness - answer_relevance - context_precision - latency_ms - cost_usd concurrency: 4 pass_threshold: 0.75 token_char_ratio: 3.5 

Precios

Los precios del agente y del juez se mantienen separados deliberadamente:

agent.prices.input_per_1m agent.prices.output_per_1m judge.prices.input_per_1m judge.prices.output_per_1m 

Si los precios permanecen en 0.0, la evaluación continúa funcionando pero el coste calculado será 0.0.

Esto permite ejecutar el arnés con proveedores gratuitos, modelos locales o cuando el coste no sea relevante.

🤖 Agentes

El arnés puede trabajar con agentes HTTP y agentes internos de demostración.

Para un agente HTTP, la respuesta puede mapearse mediante rutas configurables:

{ "data": { "answer": "...", "retrieved_contexts": ["...", "..."] } } 

Configuración:

agent: response_answer_path: "data.answer" response_contexts_path: "data.retrieved_contexts" 

El diseño mantiene desacoplado el arnés del framework utilizado para construir el agente.

Puedes evaluar un agente construido con FastAPI, Flask, Starlette, un servidor propio o cualquier otro sistema que exponga HTTP.

🛡️ Resiliencia

El sistema incorpora varias defensas para que una evaluación no falle de forma silenciosa:

Retry con backoff exponencial para errores 5xx y errores de red.

Los errores 4xx no se reintentan.

Timeouts explícitos para llamadas HTTP.

Clasificación de errores (timeout, http, parse, other).

Validación del dataset con número de línea y campos encontrados.

Transacciones SQLite con rollback en caso de error.

Parsing tolerante de JSON del juez.

Soporte para JSON rodeado de texto o Markdown fences.

Concurrencia controlada mediante asyncio.Semaphore.

Los errores de un item no obligan a perder el resto de la evaluación.

🧪 Self-tests

El archivo incluye un conjunto de tests internos que pueden ejecutarse sin instalar dependencias:

python agent-eval.py --selftest 

Los tests cubren, entre otras cosas:

estimación de tokens;

cálculo de precios;

hashing determinista;

percentiles;

extracción de JSON;

configuración;

agente builtin;

agregación de resultados;

cálculo de costes;

persistencia SQLite;

múltiples ejecuciones históricas;

generación del informe.

La intención es que el propio archivo pueda verificar su funcionamiento básico incluso en una máquina limpia.

💰 Costes y precisión

El cálculo de coste está diseñado para funcionar incluso cuando el proveedor no expone usage.

Cuando existe usage real

El LLM-juez puede devolver:

{ "usage": { "prompt_tokens": 120, "completion_tokens": 85 } } 

Esos valores se utilizan directamente.

Cuando no existe usage

Se estima el número de tokens mediante:

tokens ≈ caracteres / token_char_ratio 

Por defecto:

token_char_ratio = 3.5 

La estimación es deliberadamente simple para mantener el proyecto stdlib-only.

No pretende sustituir a un tokenizer específico del modelo.

Por tanto:

Los costes son exactos solo cuando el proveedor proporciona usage fiable; en el resto de casos son estimaciones.

🧠 Decisiones de diseño

DecisiónAlternativaPor quéUn solo archivoProyecto multi-móduloFricción cero: copiar, ejecutar y auditar rápidamenteCero dependencias obligatoriasrequests, httpx, tiktokenLa librería estándar cubre HTTP, async y persistenciaasyncio + semáforoThreadPoolExecutorEl cuello de botella es I/O; permite concurrencia controladaRetry solo en 5xx/redReintentar todoReintentar un 401 es tirar dineroSQLite en discoPostgres, DuckDBPersistencia local sin infraestructuraLLM-juez agnósticoragas, deepevalControl del prompt y sin lock-inTOKEN_CHAR_RATIO configurableTokenizer obligatorioMantiene cero dependenciasPrecios juez/agente separadosUna tabla únicaPermite conocer cuánto cuesta realmente evaluar frente a ejecutarCoste basado en usage cuando existeEstimación siempreAprovecha datos reales del proveedorHash del dataset—Detecta cambios en el conjunto evaluadoHash de configuración—Permite distinguir configuraciones aparentemente igualesHistórico SQLiteSobrescribir resultadosPermite comparar ejecucionesSelf-tests embebidosSolo pytestEl proyecto puede verificarse sin instalar nada

🔬 Casos de uso

1. Bloquear regresiones en CI

Evaluar el mismo dataset después de cada cambio y utilizar el resultado como criterio de aceptación.

Por ejemplo:

faithfulness >= 0.85 

2. A/B testing de modelos

Ejecutar el mismo dataset contra:

modelo A → run A modelo B → run B 

y comparar métricas, latencia y coste.

3. Auditar agentes de terceros

Enviar un conjunto representativo de preguntas y obtener:

calidad;

recuperación;

latencia;

errores;

tokens;

coste estimado.

4. Evaluación local

Usar un endpoint compatible con el formato esperado y ejecutar la evaluación sin necesidad de incorporar un framework de evaluación completo.

5. Control de costes

Separar:

coste del agente + coste del juez = coste total de evaluación 

Esto evita que el coste del propio proceso de evaluación quede oculto.

🧪 Filosofía

El mejor código no es el más listo. Es el que se puede leer, auditar y ejecutar sin fricción.

Un archivo. Una cosa bien hecha.

La prioridad del proyecto es que alguien pueda clonar el repositorio, leer el código completo, ejecutar los self-tests y entender de dónde sale cada métrica sin tener que aprender un framework.

No intenta competir con plataformas completas de observabilidad o evaluación.

Intenta resolver bien una pregunta concreta:

"¿Mi agente ha mejorado o empeorado, cuánto ha tardado y cuánto ha costado?"

👤 Autor

David Ferrandez Canalis — AI Engineer

Especializado en sistemas de IA en producción: RAG, agentes, evaluación y MLOps.

LinkedIn

📜 Licencia

MIT. Úsalo, forkéalo, modifícalo, véndelo.

Si lo mejoras, abre un PR.

Si este archivo te ha sido útil, deja una ⭐.

Un archivo. Cero dependencias. Métricas, persistencia y reproducibilidad.

