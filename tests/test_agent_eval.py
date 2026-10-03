"""
Tests para agent-eval.py usando pytest.

El monolito se carga dinámicamente porque su nombre contiene un guión
y no es importable directamente con `import`. Todos los tests son
deterministas, sin red y sin API keys.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sqlite3
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Carga dinámica del monolito
# ---------------------------------------------------------------------------

def _load_monolith():
    here = Path(__file__).resolve().parent.parent
    for name in ("agent-eval.py", "agent_eval.py"):
        p = here / name
        if p.exists():
            spec = importlib.util.spec_from_file_location("agent_eval", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["agent_eval"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError("No se encontró el monolito agent-eval.py")


ae = _load_monolith()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def corpus():
    return [
        "Python es un lenguaje interpretado de alto nivel.",
        "Rust es un lenguaje compilado con ownership.",
        "SQLite es una base de datos embebida sin servidor.",
    ]


@pytest.fixture
def builtin_agent(corpus):
    return ae.BuiltinAgent(corpus, k=2)


@pytest.fixture
def sample_item():
    return ae.EvalItem(
        id="q1",
        question="¿Qué es Python?",
        ground_truth="Un lenguaje interpretado.",
    )


@pytest.fixture
def sample_response():
    return ae.AgentResponse(
        answer="Python es un lenguaje interpretado.",
        contexts=["Python es un lenguaje interpretado de alto nivel."],
        latency_ms=42.0,
    )


class FakeResponse:
    """Sustituto de la respuesta de urllib.request.urlopen."""

    def __init__(self, body: str, status: int = 200):
        self._body = body.encode("utf-8")
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


# ===========================================================================
# 1. Funciones puras
# ===========================================================================

class TestEstimateTokens:
    def test_vacio(self):
        assert ae.estimate_tokens("") == 0

    def test_texto_corto(self):
        assert ae.estimate_tokens("hola") > 0

    def test_ratio_afecta_resultado(self):
        a = ae.estimate_tokens("a" * 100, ratio=4.0)
        b = ae.estimate_tokens("a" * 100, ratio=2.0)
        assert a < b

    def test_monotonia(self):
        assert ae.estimate_tokens("a" * 100) <= ae.estimate_tokens("a" * 200)


class TestSha256Short:
    def test_determinista(self):
        assert ae.sha256_short("hola") == ae.sha256_short("hola")

    def test_distinto_input(self):
        assert ae.sha256_short("hola") != ae.sha256_short("hola ")

    def test_longitud_por_defecto(self):
        assert len(ae.sha256_short("hola")) == 12

    def test_longitud_personalizada(self):
        assert len(ae.sha256_short("hola", n=8)) == 8


class TestPercentile:
    def test_vacio(self):
        assert ae.percentile([], 0.5) == 0.0

    def test_p50(self):
        assert abs(ae.percentile([1, 2, 3, 4, 5], 0.5) - 3.0) < 1e-9

    def test_p0(self):
        assert ae.percentile([1, 2, 3, 4, 5], 0.0) == 1.0

    def test_p100(self):
        assert ae.percentile([1, 2, 3, 4, 5], 1.0) == 5.0

    def test_un_solo_valor(self):
        assert ae.percentile([42.0], 0.5) == 42.0


class TestSafeMean:
    def test_vacio(self):
        assert ae.safe_mean([]) == 0.0

    def test_media_simple(self):
        assert abs(ae.safe_mean([1, 2, 3, 4]) - 2.5) < 1e-9

    def test_ignora_none(self):
        assert abs(ae.safe_mean([1, None, 3]) - 2.0) < 1e-9


class TestPriceFor:
    def test_modelo_exacto(self):
        assert ae.price_for("gpt-4o-mini")["in"] == 0.00015

    def test_prefijo(self):
        assert ae.price_for("gpt-4o-mini-2024")["in"] == 0.00015

    def test_desconocido(self):
        assert ae.price_for("modelo-inexistente")["in"] == 0.0


class TestFmt:
    def test_tres_decimales(self):
        assert ae._fmt(1.23456) == "1.235"

    def test_un_decimal(self):
        assert ae._fmt(1.23456, 1) == "1.2"


# ===========================================================================
# 2. Utilidades recursivas
# ===========================================================================

class TestDeepGet:
    def test_un_nivel(self):
        assert ae._deep_get({"a": 1}, "a") == 1

    def test_dos_niveles(self):
        assert ae._deep_get({"a": {"b": 2}}, "a.b") == 2

    def test_falta(self):
        assert ae._deep_get({"a": 1}, "a.b") is None

    def test_valor_cero(self):
        assert ae._deep_get({"a": 0}, "a") == 0


class TestDeepFormat:
    def test_string_plano(self):
        assert ae._deep_format("hola {x}", {"x": "mundo"}) == "hola mundo"

    def test_sin_placeholders(self):
        assert ae._deep_format("hola", {"x": "mundo"}) == "hola"

    def test_dict_anidado(self):
        assert ae._deep_format({"q": "{x}"}, {"x": "y"}) == {"q": "y"}

    def test_lista(self):
        assert ae._deep_format([{"q": "{x}"}], {"x": "y"}) == [{"q": "y"}]


class TestMergeConfig:
    def test_override_plano(self):
        assert ae.merge_config({"a": 1}, {"a": 2}) == {"a": 2}

    def test_merge_profundo(self):
        assert ae.merge_config({"a": {"b": 1}}, {"a": {"c": 2}}) == {
            "a": {"b": 1, "c": 2}
        }

    def test_no_muta_base(self):
        base = {"a": 1}
        ae.merge_config(base, {"a": 2})
        assert base["a"] == 1


# ===========================================================================
# 3. Parsing de JSON del juez
# ===========================================================================

class TestExtractJson:
    def test_limpio(self):
        assert ae._extract_json('{"score": 0.8}') == {"score": 0.8}

    def test_con_fence(self):
        assert ae._extract_json('```json\n{"score": 0.5}\n```') == {"score": 0.5}

    def test_con_fence_sin_etiqueta(self):
        assert ae._extract_json('```\n{"score": 0.5}\n```') == {"score": 0.5}

    def test_con_basura(self):
        assert ae._extract_json('bla {"score": 0.3} bla') == {"score": 0.3}

    def test_invalido(self):
        assert ae._extract_json("no json aquí") is None

    def test_anidado(self):
        assert ae._extract_json('{"a": {"b": 1}}') == {"a": {"b": 1}}

    def test_array_no_dict(self):
        assert ae._extract_json("[1, 2, 3]") is None


# ===========================================================================
# 4. BuiltinAgent
# ===========================================================================

class TestBuiltinAgent:
    def test_devuelve_respuesta(self, builtin_agent):
        resp = asyncio.run(builtin_agent.ask("Python lenguaje"))
        assert resp.answer

    def test_devuelve_contextos(self, builtin_agent):
        resp = asyncio.run(builtin_agent.ask("Python lenguaje"))
        assert len(resp.contexts) > 0

    def test_sin_error(self, builtin_agent):
        resp = asyncio.run(builtin_agent.ask("Python lenguaje"))
        assert resp.error is None

    def test_contexto_correcto(self, builtin_agent):
        resp = asyncio.run(builtin_agent.ask("Python lenguaje"))
        assert "Python" in resp.contexts[0]

    def test_query_sin_match(self, builtin_agent):
        resp = asyncio.run(builtin_agent.ask("xyz qwerty"))
        assert resp.answer

    def test_latencia_no_negativa(self, builtin_agent):
        resp = asyncio.run(builtin_agent.ask("Python"))
        assert resp.latency_ms >= 0.0


# ===========================================================================
# 5. HTTPAgent con urlopen mockeado
# ===========================================================================

class TestHTTPAgent:
    def _cfg(self):
        return {
            "type": "http",
            "url": "http://fake/ask",
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "request_template": {"question": "{question}"},
            "response_answer_path": "answer",
            "response_contexts_path": "contexts",
            "timeout": 5.0,
        }

    def test_respuesta_valida(self):
        body = json.dumps({"answer": "42", "contexts": ["c1", "c2"]})
        agent = ae.HTTPAgent(self._cfg())
        with patch("urllib.request.urlopen",
                   return_value=FakeResponse(body)):
            resp = asyncio.run(agent.ask("¿?"))
        assert resp.answer == "42"
        assert resp.contexts == ["c1", "c2"]
        assert resp.error is None

    def test_contexto_string_se_convierte_a_lista(self):
        body = json.dumps({"answer": "a", "contexts": "solo uno"})
        agent = ae.HTTPAgent(self._cfg())
        with patch("urllib.request.urlopen",
                   return_value=FakeResponse(body)):
            resp = asyncio.run(agent.ask("¿?"))
        assert resp.contexts == ["solo uno"]

    def test_json_invalido(self):
        agent = ae.HTTPAgent(self._cfg())
        with patch("urllib.request.urlopen",
                   return_value=FakeResponse("no es json")):
            resp = asyncio.run(agent.ask("¿?"))
        assert resp.error is not None
        assert resp.error_type == "parse"

    def test_error_de_red(self):
        agent = ae.HTTPAgent(self._cfg())
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("boom")):
            resp = asyncio.run(agent.ask("¿?"))
        assert resp.error is not None
        assert resp.error_type in ("http", "timeout")

    def test_timeout(self):
        agent = ae.HTTPAgent(self._cfg())
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("timed out")):
            resp = asyncio.run(agent.ask("¿?"))
        assert resp.error_type == "timeout"

    def test_auth_env(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "secreto")
        cfg = self._cfg()
        cfg["auth_env"] = "MY_TOKEN"
        agent = ae.HTTPAgent(cfg)
        assert agent.auth == "secreto"


# ===========================================================================
# 6. LLMClient (con urlopen mockeado)
# ===========================================================================

class TestLLMClient:
    def _client(self):
        return ae.LLMClient(
            base_url="http://fake/v1",
            model="gpt-4o-mini",
            api_key="fake",
        )

    def test_complete_ok(self):
        body = json.dumps({
            "choices": [{"message": {"content": '{"score": 0.9}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })
        client = self._client()
        with patch("urllib.request.urlopen",
                   return_value=FakeResponse(body)):
            text, t_in, t_out = asyncio.run(client.complete("sys", "user"))
        assert text == '{"score": 0.9}'
        assert t_in == 10
        assert t_out == 5

    def test_sin_usage_estima(self):
        body = json.dumps({
            "choices": [{"message": {"content": "hola"}}],
        })
        client = self._client()
        with patch("urllib.request.urlopen",
                   return_value=FakeResponse(body)):
            text, t_in, t_out = asyncio.run(client.complete("sys", "user"))
        assert t_in > 0
        assert t_out > 0

    def test_http_500_reintenta_y_falla(self):
        client = self._client()
        err = urllib.error.HTTPError(
            "http://fake", 500, "boom", {}, None,
        )
        with patch("urllib.request.urlopen", side_effect=err), \
             patch.object(ae, "JUDGE_RETRY_DELAY", 0.0):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                asyncio.run(client.complete("sys", "user"))

    def test_http_401_no_reintenta(self):
        client = self._client()
        err = urllib.error.HTTPError(
            "http://fake", 401, "no auth", {}, None,
        )
        with patch("urllib.request.urlopen", side_effect=err) as m:
            with pytest.raises(RuntimeError, match="HTTP 401"):
                asyncio.run(client.complete("sys", "user"))
        assert m.call_count == 1


# ===========================================================================
# 7. Métricas individuales (juez mockeado)
# ===========================================================================

class _FakeJudge:
    """Juez determinista que devuelve siempre el mismo score."""

    def __init__(self, score: float, reasoning: str = "ok"):
        self.score = score
        self.reasoning = reasoning
        self.model = "fake"

    async def complete(self, system: str, user: str):
        payload = json.dumps({"score": self.score, "reasoning": self.reasoning})
        return payload, 10, 5


class TestMetrics:
    def test_faithfulness_sin_contextos(self, sample_item):
        resp = ae.AgentResponse(answer="x", contexts=[])
        judgment, _, _ = asyncio.run(
            ae.metric_faithfulness(_FakeJudge(0.9), sample_item, resp)
        )
        assert judgment.score == 0.0

    def test_faithfulness_con_contextos(self, sample_item, sample_response):
        judgment, t_in, t_out = asyncio.run(
            ae.metric_faithfulness(
                _FakeJudge(0.85), sample_item, sample_response
            )
        )
        assert judgment.score == 0.85
        assert t_in > 0 and t_out > 0

    def test_answer_relevance(self, sample_item, sample_response):
        judgment, _, _ = asyncio.run(
            ae.metric_answer_relevance(
                _FakeJudge(0.7), sample_item, sample_response
            )
        )
        assert judgment.score == 0.7

    def test_context_precision_sin_contextos(self, sample_item):
        resp = ae.AgentResponse(answer="x", contexts=[])
        judgment, _, _ = asyncio.run(
            ae.metric_context_precision(_FakeJudge(0.9), sample_item, resp)
        )
        assert judgment.score == 0.0

    def test_context_recall_sin_ground_truth(self):
        item = ae.EvalItem(id="x", question="q", ground_truth=None)
        resp = ae.AgentResponse(answer="a", contexts=["c"])
        judgment, _, _ = asyncio.run(
            ae.metric_context_recall(_FakeJudge(0.9), item, resp)
        )
        assert judgment.score == 0.0

    def test_score_fuera_de_rango_se_recorta(self, sample_item, sample_response):
        judgment, _, _ = asyncio.run(
            ae.metric_faithfulness(
                _FakeJudge(5.0), sample_item, sample_response
            )
        )
        assert judgment.score == 1.0


# ===========================================================================
# 8. Evaluator
# ===========================================================================

class TestEvaluator:
    def _cfg(self):
        cfg = json.loads(json.dumps(ae.CONFIG))
        cfg["eval"]["metrics"] = ["faithfulness", "latency_ms"]
        cfg["eval"]["concurrency"] = 2
        cfg["eval"]["pass_threshold"] = 0.5
        return cfg

    def test_evaluate_item_ok(self, builtin_agent, sample_item):
        cfg = self._cfg()
        evaluator = ae.Evaluator(cfg, builtin_agent, _FakeJudge(0.9))
        result = asyncio.run(evaluator.evaluate_item(sample_item))
        assert result.passed is True
        assert result.metrics["faithfulness"] == 0.9
        assert "latency_ms" in result.metrics

    def test_evaluate_item_falla_umbral(self, builtin_agent, sample_item):
        cfg = self._cfg()
        cfg["eval"]["pass_threshold"] = 0.95
        evaluator = ae.Evaluator(cfg, builtin_agent, _FakeJudge(0.5))
        result = asyncio.run(evaluator.evaluate_item(sample_item))
        assert result.passed is False

    def test_metricas_invalidas_se_ignoran(self, builtin_agent):
        cfg = self._cfg()
        cfg["eval"]["metrics"] = ["faithfulness", "metrica_inventada"]
        evaluator = ae.Evaluator(cfg, builtin_agent, _FakeJudge(0.9))
        result = asyncio.run(
            evaluator.evaluate_item(ae.EvalItem(id="x", question="q"))
        )
        assert "metrica_inventada" not in result.metrics
        assert "faithfulness" in result.metrics


# ===========================================================================
# 9. Persistencia SQLite
# ===========================================================================

class TestPersistResults:
    def test_crea_db_y_tablas(self, tmp_path):
        results = [
            ae.ItemResult(
                item=ae.EvalItem(id="q1", question="¿?"),
                response=ae.AgentResponse(
                    answer="a", contexts=["c"], latency_ms=10.0,
                ),
                judgments={
                    "faithfulness": ae.Judgment(score=0.9, reasoning="ok"),
                },
                metrics={"faithfulness": 0.9},
                passed=True,
            ),
        ]
        db = tmp_path / "out.db"
        ae.persist_results(results, str(db))
        assert db.exists()

        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM evaluations")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM metrics")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT passed FROM evaluations WHERE id='q1'")
        assert cur.fetchone()[0] == 1
        conn.close()

    def test_sobrescribe_db_existente(self, tmp_path):
        db = tmp_path / "out.db"
        db.write_text("basura", encoding="utf-8")
        ae.persist_results([], str(db))
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM evaluations")
        assert cur.fetchone()[0] == 0
        conn.close()


# ===========================================================================
# 10. load_dataset
# ===========================================================================

class TestLoadDataset:
    def _cfg(self, path):
        cfg = json.loads(json.dumps(ae.CONFIG))
        cfg["dataset"]["path"] = str(path)
        return cfg

    def test_dataset_valido(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text(
            '{"id": "a", "question": "q1"}\n'
            '{"id": "b", "question": "q2", "ground_truth": "gt"}\n',
            encoding="utf-8",
        )
        items = ae.load_dataset(self._cfg(p))
        assert len(items) == 2
        assert items[0].id == "a"
        assert items[1].ground_truth == "gt"

    def test_ignora_lineas_vacias_y_comentarios(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text(
            '# comentario\n{"id": "a", "question": "q"}\n\n',
            encoding="utf-8",
        )
        items = ae.load_dataset(self._cfg(p))
        assert len(items) == 1

    def test_falta_question(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text('{"id": "a"}\n', encoding="utf-8")
        with pytest.raises(SystemExit, match="falta campo requerido"):
            ae.load_dataset(self._cfg(p))

    def test_json_invalido(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text("no json\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="JSON inválido"):
            ae.load_dataset(self._cfg(p))

    def test_id_por_defecto_es_linea(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text('{"question": "q"}\n', encoding="utf-8")
        items = ae.load_dataset(self._cfg(p))
        assert items[0].id == "1"


# ===========================================================================
# 11. render_report
# ===========================================================================

class TestRenderReport:
    def _cfg(self):
        cfg = json.loads(json.dumps(ae.CONFIG))
        cfg["eval"]["metrics"] = ["faithfulness"]
        return cfg

    def _result(self, score: float, passed: bool):
        return ae.ItemResult(
            item=ae.EvalItem(id="q1", question="¿?"),
            response=ae.AgentResponse(
                answer="a", contexts=["c"], latency_ms=10.0,
            ),
            judgments={"faithfulness": ae.Judgment(score=score)},
            metrics={"faithfulness": score, "latency_ms": 10.0},
            passed=passed,
        )

    def test_incluye_secciones(self):
        cfg = self._cfg()
        report = ae.render_report(
            cfg,
            [self._result(0.9, True), self._result(0.4, False)],
            "deadbeef",
            1.23,
        )
        assert "# Informe de evaluación" in report
        assert "deadbeef" in report
        assert "## 1. Resumen ejecutivo" in report
        assert "## 4. Métricas agregadas" in report
        assert "## 6. Análisis de fallos" in report
        assert "```mermaid" in report

    def test_veredicto_apto(self):
        cfg = self._cfg()
        report = ae.render_report(
            cfg, [self._result(0.9, True)], "h", 1.0,
        )
        assert "✅ APTO" in report

    def test_veredicto_revisar_por_fallos(self):
        cfg = self._cfg()
        report = ae.render_report(
            cfg, [self._result(0.4, False)], "h", 1.0,
        )
        assert "⚠️ REVISAR" in report

    def test_sin_mermaid_si_desactivado(self):
        cfg = self._cfg()
        cfg["output"]["include_mermaid"] = False
        report = ae.render_report(
            cfg, [self._result(0.9, True)], "h", 1.0,
        )
        assert "```mermaid" not in report


# ===========================================================================
# 12. Self-tests embebidos del monolito
# ===========================================================================

class TestSelfTests:
    def test_selftests_pasan(self):
        rc = ae.run_selftests()
        assert rc == 0


# ===========================================================================
# 13. Demo end-to-end
# ===========================================================================

class TestDemo:
    def test_demo_config(self):
        cfg = ae.demo_config()
        assert cfg["agent"]["type"] == "builtin"
        assert cfg["dataset"]["path"] == "(demo)"

    def test_demo_dataset(self):
        items = ae.load_demo_dataset()
        assert len(items) == 3
        assert all(i.question for i in items)
