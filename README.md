# Agent-Eval: Un Arnés de Evaluación Autocontenido para Pipelines de IA

## 🎯 El Problema
La mayoría de los proyectos de IA son cajas negras sin métricas de calidad. Esto los hace inútiles para producción.

## 💡 La Solución
`agent-eval.py` es un único archivo que mide la fiabilidad de cualquier sistema RAG o agente.

## 🚀 Cómo Empezar
1.  Copia el archivo.
2.  Configura tu agente y tu dataset en el bloque `CONFIG`.
3.  Ejecuta: `python agent-eval.py --report mi_informe.md`

## 📊 Qué Mide
| Métrica | Descripción | ¿Por qué importa? |
| :--- | :--- | :--- |
| **Fidelidad (Faithfulness)** | ¿La respuesta está basada en el contexto recuperado? | Evita alucinaciones. |
| **Precisión de Recuperación** | ¿Los fragmentos recuperados son relevantes? | La base de un buen RAG. |
| **Latencia (P95)** | Tiempo de respuesta en el percentil 95. | Experiencia de usuario. |
| **Coste por 1k consultas** | Gasto medio en tokens. | Sostenibilidad del producto. |

## 🧠 Decisiones de Arquitectura (El porqué del monolito)
- **Un solo archivo:** Para eliminar la fricción de la instalación y demostrar que la complejidad no requiere infraestructura.
- **SQLite en memoria:** Para persistir los resultados de las evaluaciones sin necesidad de un servidor de base de datos.

## 📈 Resultados de Ejemplo
*(Aquí incluirías una tabla con los resultados reales de evaluar un agente de ejemplo. Ej: "Agente de Atención al Cliente: Fidelidad 0.92, Precisión 0.88, Latencia 1.2s")*

## 🛠️ Código Clave
*(Aquí incluirías el fragmento de código que calcula la métrica de fidelidad, por ejemplo, usando un LLM como juez)*
