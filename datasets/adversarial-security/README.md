# Adversarial Security & Prompt Injection

## Propósito

Crear un corpus de escenarios adversariales para modelos locales a partir de [Local-LLM-Red-Team](https://github.com/orbitalronin94/Local-LLM-Red-Team).

El proyecto fuente contiene un arnés de evaluación con ataques estáticos y adaptativos, objetivos locales o compatibles con API, resultados estructurados y un detector heurístico. El corpus debe preservar la diferencia entre ataque, respuesta, detección y vulnerabilidad confirmada.

## Familias iniciales observadas en el código

La versión examinada declara ocho ataques iniciales:

- `jailbreak-roleplay`
- `jailbreak-authority`
- `instruction-override`
- `system-prompt-extraction`
- `system-prompt-summary`
- `prompt-injection-document`
- `prompt-injection-tool`
- `persistence`

Sus categorías incluyen jailbreak, instruction override, system extraction, prompt injection y persistence. Las versiones de los ataques deben conservarse como parte del identificador de experimento.

## Esquema de registro

```json
{
  "id": "security-case-0001",
  "attack_id": "<id del manifiesto>",
  "attack_version": "1",
  "category": "<categoría>",
  "mode": "static",
  "target": {
    "provider": "<proveedor>",
    "model": "<modelo>",
    "version": "<versión>"
  },
  "conversation": [],
  "response": "<respuesta>",
  "detector": {
    "name": "<detector>",
    "status": "unreviewed",
    "evidence": []
  },
  "human_assessment": "pending",
  "run_manifest_hash": "<hash>",
  "split": "unassigned"
}
```

Plantilla de esquema; no es un resultado observado.

## Separación de conceptos

No colapsar estos estados:

- `PASS`: el detector encontró señales de rechazo según sus reglas.
- `FAIL`: el detector encontró patrones de aceptación del ataque.
- `NOT_DETECTED`: no encontró patrones conocidos; no significa que el modelo sea seguro.
- `ERROR`: la ejecución no produjo una evaluación válida.
- `EMPTY`: respuesta vacía.

La detección heurística del proyecto se apoya en patrones de texto y puede producir falsos positivos y falsos negativos. Sus etiquetas son resultados del detector, no ground truth de seguridad. Añadir evaluación humana o un protocolo independiente antes de usar ejemplos como etiquetas definitivas.

## Métricas recomendadas

- tasa de éxito del ataque con definición operacional explícita;
- tasa de rechazo y tasa de abstención;
- falsos positivos y falsos negativos del detector;
- cobertura por familia de ataque;
- resultados por turno en modo adaptativo;
- latencia y errores de infraestructura;
- variación entre repeticiones y configuraciones.

No mezclar errores de transporte con resistencia del modelo. Informar el denominador de cada tasa.

## Protocolo

1. Congelar el manifiesto de ataques y su hash.
2. Registrar modelo, versión, proveedor, parámetros y modo.
3. Guardar la conversación completa, no solo la respuesta final.
4. Ejecutar repeticiones controladas.
5. Evaluar detecciones con revisión independiente.
6. Mantener ataques y paráfrasis de holdout fuera del ajuste.
7. Versionar detectores por separado de los ataques.

## Seguridad y publicación

Los ejemplos deben centrarse en evaluar defensas y preservar el contexto de seguridad. No incorporar secretos reales, prompts privados, credenciales ni datos personales. Revisar cuidadosamente los ejemplos antes de publicarlos para evitar convertir el corpus en una colección de instrucciones operativas dañinas.

## Licencia y estado

El repositorio fuente declara MIT. Aun así, conservar atribución, commit de origen y licencia en cada registro. Esta carpeta define el corpus; no afirma que los resultados del detector heurístico estén validados ni que exista ya un dataset curado.
