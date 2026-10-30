# RONIN Knowledge & Semantics

## Propósito

Documentar una línea de conocimiento para modelos sobre el lenguaje RONIN, su especificación y su runtime de referencia en Python. La fuente principal es [ronin-lang](https://github.com/orbitalronin94/ronin-lang).

El objetivo no es enseñar únicamente ejemplos de sintaxis. Se busca representar la relación entre código, AST, reglas semánticas, validación, ejecución, simulación, resultados esperados y límites conocidos.

## Áreas candidatas

- Léxico, gramática, parser y formas válidas/ inválidas.
- AST esperado para fragmentos representativos.
- Reglas de validación semántica y mensajes de error.
- Tipos, rangos, unidades y restricciones entre variables.
- Fórmulas y modelos documentados, incluidos PUSFRE, CES, Hill y CES-Hill cuando la implementación y la especificación lo respalden.
- Simulación, asignación de recursos y comportamiento temporal.
- Diagnóstico de identificabilidad, degeneración y sensibilidad.
- Casos límite numéricos y condiciones de convergencia.
- Diferencias entre especificación, documentación y comportamiento real del runtime.

## Formatos de ejemplo

### Sintaxis y AST

```json
{
  "id": "ronin-parse-example-0001",
  "knowledge_type": "syntax_ast",
  "input": "<fragmento RONIN>",
  "target": {"ast": "<AST esperado>"},
  "epistemic_status": "pending_verification",
  "evidence_refs": [],
  "split": "unassigned"
}
```

El marcador es un esquema ilustrativo, no un ejemplo real ni una afirmación sobre la gramática. Los ejemplos reales deben derivarse de fixtures y pruebas del repositorio.

### Regla semántica

```json
{
  "id": "ronin-semantic-rule-0001",
  "knowledge_type": "semantic_rule",
  "claim": "<regla concreta>",
  "preconditions": [],
  "expected_result": "<resultado>",
  "counterexamples": [],
  "epistemic_status": "pending_verification",
  "evidence_refs": [],
  "split": "unassigned"
}
```

## Etiquetas epistemológicas

Usar etiquetas diferenciadas:

- `implemented`: existe implementación identificable.
- `test_verified`: una prueba automatizada cubre el comportamiento descrito.
- `experimentally_observed`: observado en una ejecución registrada.
- `specified_only`: aparece en la especificación, sin implementación verificada.
- `proposed`: propuesta aún no aceptada.
- `known_gap`: limitación o funcionalidad ausente reconocida.
- `contradiction`: especificación, implementación o pruebas no concuerdan.
- `unverified`: evidencia insuficiente.

No elevar una regla a `test_verified` por el mero hecho de que exista código relacionado. Registrar la prueba exacta y su alcance.

## Discrepancias y límites que deben conservarse

La documentación de la fuente identifica huecos y discrepancias concretos, entre ellos:

- `debt()` descrita como pendiente o con retorno placeholder de `0.0`.
- Intervalos bootstrap no implementados.
- Rechazo de memoria temporal superior a orden 1.
- Diferencia documentada entre el límite CES cuando λ tiende a cero y una prueba que lo compara con PUSFRE.

Estos puntos son excelentes ejemplos para un dataset de detección de discrepancias entre especificación, tests e implementación. Antes de convertirlos en registros, comprobar la revisión exacta del repositorio y preservar el contexto de cada afirmación.

## Evidencia requerida

Cada ejemplo curado debe incluir, cuando exista:

- ruta y commit del archivo fuente;
- fragmento de especificación relevante;
- test que lo cubre y resultado;
- versión/configuración del runtime;
- entrada y salida observadas;
- límites del dominio probado;
- contraejemplo mínimo;
- licencia y procedencia.

## Particiones y prevención de fugas

Agrupar por familia semántica y por origen de los fixtures. No repartir versiones casi idénticas del mismo caso entre train y holdout. Reservar programas y combinaciones de reglas para evaluación fuera de muestra.

## Estado

Especificación de dataset. No contiene todavía un corpus extraído ni afirma que todas las reglas RONIN estén formalmente validadas.
