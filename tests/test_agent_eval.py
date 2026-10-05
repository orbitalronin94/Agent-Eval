"""
Tests para Agent-Eval 0.2.2.

El monolito se carga dinámicamente porque su nombre contiene un guión.
Todos los tests son deterministas, sin red real y sin API keys.
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
    root = Path(__file__).resolve().parent.parent
    path = root / "agent-eval.py"

    if not path.exists():
        raise RuntimeError(f"No se encontró el monolito: {path}")

    spec = importlib.util.spec_from_file_location("agent_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "No se pudo crear el import spec para agent-eval.py"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_eval"] = module
    spec.loader.exec_module(module)
    return module


ae = _load_monolith()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeHTTPResponse:
    """Respuesta mínima compatible con urllib.request.urlopen."""

    def __init__(self, body: str):
        self.body = body.encode("utf-8")

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeJudgeClient:
    """Cliente LLM determinista para tests del Judge."""

    def __init__(
        self,
        score: float = 0.9,
        reasoning: str = "ok",
        usage: tuple[int, int] = (10, 5),
    ):
        self.score = score
        self.reasoning = reasoning
        self.usage = usage

    async def complete(self, messages):
        payload = json.dumps(
            {
                "score": self.score,
                "reasoning": self.reasoning,
            }
        )
        return payload, {
            "prompt_tokens": self.usage[0],
            "completion_tokens": self.usage[1],
        }


@pytest.fixture
def sample_item():
    return ae.EvalItem(
        id="q1",
        question="What is Python?",
        ground_truth="A programming language.",
        contexts=[
            "Python is a high-level programming language."
        ],
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_response():
    return ae.AgentResponse(
        answer="Python is a programming language.",
        contexts=[
            "Python is a high-level programming language."
        ],
        latency_ms=42.0,
        tokens_in=10,
        tokens_out=8,
    )


@pytest.fixture
def base_config():
    return json.loads(json.dumps(ae.CONFIG))


# ===========================================================================
# 1. Dataclasses
# ===========================================================================

class TestDataclasses:
    def test_eval_item_defaults(self):
        item = ae.EvalItem(
            id="1",
            question="q",
        )

        assert item.ground_truth is None
        assert item.contexts == []
        assert item.metadata == {}

    def test_agent_response_defaults(self):
        response = ae.AgentResponse(answer="a")

        assert response.contexts == []
        assert response.latency_ms == 0.0
        assert response.tokens_in == 0
        assert response.tokens_out == 0
        assert response.raw is None

    def test_judgment_defaults(self):
        judgment = ae.Judgment(
            metric="faithfulness",
            score=0.8,
        )

        assert judgment.reasoning == ""
        assert judgment.raw is None
        assert judgment.tokens_in == 0
        assert judgment.tokens_out == 0

    def test_item_result_defaults(self):
        item = ae.EvalItem(
            id="1",
            question="q",
        )
        result = ae.ItemResult(
            item=item,
            response=None,
        )

        assert result.judgments == []
        assert result.passed is False
        assert result.error is None
        assert result.error_type is None
        assert result.cost_usd == 0.0


# ===========================================================================
# 2. Funciones puras
# ===========================================================================

class TestEstimateTokens:
    def test_empty(self):
        assert ae.estimate_tokens("") == 0

    def test_non_empty(self):
        assert ae.estimate_tokens("hello") >= 1

    def test_monotonic(self):
        assert (
            ae.estimate_tokens("a" * 100)
            <= ae.estimate_tokens("a" * 200)
        )

    def test_long_text(self):
        assert ae.estimate_tokens("a" * 1000) == 250


class TestPriceAndCost:
    def test_price_for(self):
        assert ae.price_for(1_000_000, 2.0) == pytest.approx(2.0)

    def test_price_for_zero(self):
        assert ae.price_for(0, 10.0) == 0.0

    def test_token_cost(self):
        value = ae.token_cost(
            1_000_000,
            500_000,
            1.0,
            2.0,
        )

        assert value == pytest.approx(2.0)

    def test_token_cost_zero(self):
        assert ae.token_cost(0, 0, 1.0, 2.0) == 0.0


class TestHashes:
    def test_sha256_short_default_length(self):
        assert len(ae.sha256_short("hello")) == 12

    def test_sha256_short_custom_length(self):
        assert len(ae.sha256_short("hello", 8)) == 8

    def test_sha256_short_deterministic(self):
        assert ae.sha256_short("hello") == ae.sha256_short("hello")

    def test_sha256_short_changes_with_input(self):
        assert ae.sha256_short("hello") != ae.sha256_short("hello!")

    def test_canonical_hash_order_independent(self):
        a = ae.canonical_hash({"b": 2, "a": 1})
        b = ae.canonical_hash({"a": 1, "b": 2})

        assert a == b

    def test_canonical_hash_custom_length(self):
        value = ae.canonical_hash(
            {"a": 1},
            length=8,
        )

        assert len(value) == 8


class TestPercentile:
    def test_empty(self):
        assert ae.percentile([], 50) == 0.0

    def test_single_value(self):
        assert ae.percentile([42.0], 50) == 42.0

    def test_zero(self):
        assert ae.percentile([1, 2, 3], 0) == 1

    def test_hundred(self):
        assert ae.percentile([1, 2, 3], 100) == 3

    def test_interpolation(self):
        assert ae.percentile(
            [1, 2, 3, 4, 5],
            25,
        ) == pytest.approx(2.0)

    def test_p50(self):
        assert ae.percentile(
            [1, 2, 3, 4, 5],
            50,
        ) == pytest.approx(3.0)


class TestSafeMean:
    def test_empty(self):
        assert ae.safe_mean([]) == 0.0

    def test_mean(self):
        assert ae.safe_mean([1, 2, 3, 4]) == pytest.approx(2.5)


class TestDeepMerge:
    def test_flat_override(self):
        result = ae.deep_merge(
            {"a": 1, "b": 2},
            {"a": 3},
        )

        assert result == {
            "a": 3,
            "b": 2,
        }

    def test_nested_merge(self):
        result = ae.deep_merge(
            {"a": {"b": 1, "c": 2}},
            {"a": {"b": 3}},
        )

        assert result == {
            "a": {
                "b": 3,
                "c": 2,
            }
        }

    def test_does_not_mutate_base(self):
        base = {
            "a": {
                "b": 1,
            }
        }

        ae.deep_merge(
            base,
            {
                "a": {
                    "b": 2,
                }
            },
        )

        assert base["a"]["b"] == 1

    def test_override_dict_with_scalar(self):
        result = ae.deep_merge(
            {"a": {"b": 1}},
            {"a": 5},
        )

        assert result["a"] == 5


# ===========================================================================
# 3. Utilidades recursivas
# ===========================================================================

class TestDeepFormat:
    def test_string(self):
        assert ae._deep_format(
            "hello {name}",
            {"name": "world"},
        ) == "hello world"

    def test_missing_key_is_left_unchanged(self):
        value = ae._deep_format(
            "hello {missing}",
            {},
        )

        assert value == "hello {missing}"

    def test_invalid_format_is_left_unchanged(self):
        value = ae._deep_format(
            "{",
            {},
        )

        assert value == "{"

    def test_nested_dict(self):
        value = ae._deep_format(
            {
                "question": "{question}",
                "nested": {
                    "id": "{id}",
                },
            },
            {
                "question": "hello",
                "id": "123",
            },
        )

        assert value == {
            "question": "hello",
            "nested": {
                "id": "123",
            },
        }

    def test_nested_list(self):
        value = ae._deep_format(
            [
                "{a}",
                {
                    "b": "{b}",
                },
            ],
            {
                "a": "A",
                "b": "B",
            },
        )

        assert value == [
            "A",
            {
                "b": "B",
            },
        ]

    def test_non_string_value(self):
        assert ae._deep_format(123, {}) == 123


class TestGetPath:
    def test_none_path_returns_value(self):
        value = {"a": 1}

        assert ae._get_path(value, None) == value

    def test_empty_path_returns_value(self):
        value = {"a": 1}

        assert ae._get_path(value, "") == value

    def test_dict_path(self):
        value = {
            "a": {
                "b": 2,
            }
        }

        assert ae._get_path(value, "a.b") == 2

    def test_list_path(self):
        value = {
            "items": [
                {"name": "first"},
                {"name": "second"},
            ]
        }

        assert ae._get_path(
            value,
            "items.1.name",
        ) == "second"

    def test_out_of_range(self):
        assert ae._get_path(
            {"items": []},
            "items.0",
        ) is None

    def test_invalid_path(self):
        assert ae._get_path(
            {"a": 1},
            "a.b",
        ) is None


# ===========================================================================
# 4. JSON parsing
# ===========================================================================

class TestExtractJson:
    def test_clean_object(self):
        assert ae._extract_json(
            '{"score": 0.8}'
        ) == {"score": 0.8}

    def test_clean_array(self):
        assert ae._extract_json(
            "[1, 2, 3]"
        ) == [1, 2, 3]

    def test_prefixed_object(self):
        assert ae._extract_json(
            'Here is the result: {"score": 0.8}'
        ) == {"score": 0.8}

    def test_prefixed_array(self):
        assert ae._extract_json(
            "Result: [1, 2, 3]"
        ) == [1, 2, 3]

    def test_multiple_fragments_returns_first_valid(self):
        value = ae._extract_json(
            'text {"a": 1} {"b": 2}'
        )

        assert value == {"a": 1}

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="No valid JSON"):
            ae._extract_json("not json")

    def test_nested_json(self):
        assert ae._extract_json(
            '{"a": {"b": 1}}'
        ) == {
            "a": {
                "b": 1,
            }
        }


# ===========================================================================
# 5. Configuración
# ===========================================================================

class TestConfig:
    def test_load_json_config(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"agent": {"type": "builtin"}}),
            encoding="utf-8",
        )

        value = ae.load_config_file(path)

        assert value == {
            "agent": {
                "type": "builtin",
            }
        }

    def test_load_yaml_config(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(
            "agent:\n"
            "  type: builtin\n",
            encoding="utf-8",
        )

        try:
            value = ae.load_config_file(path)
        except RuntimeError as exc:
            pytest.skip(str(exc))

        assert value["agent"]["type"] == "builtin"

    def test_missing_config(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ae.load_config_file(tmp_path / "missing.json")

    def test_invalid_json_config(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("not json", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            ae.load_config_file(path)

    def test_json_config_must_be_object(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("[]", encoding="utf-8")

        with pytest.raises(
            ValueError,
            match="must contain an object",
        ):
            ae.load_config_file(path)

    def test_unsupported_config_extension(self, tmp_path):
        path = tmp_path / "config.txt"
        path.write_text("x", encoding="utf-8")

        with pytest.raises(
            ValueError,
            match="Unsupported config format",
        ):
            ae.load_config_file(path)


class TestResolveApiKey:
    def test_without_environment_setting(self):
        assert ae.resolve_api_key({}) is None

    def test_environment_key(self, monkeypatch):
        monkeypatch.setenv(
            "TEST_AGENT_EVAL_KEY",
            "secret",
        )

        assert ae.resolve_api_key(
            {
                "api_key_env": "TEST_AGENT_EVAL_KEY"
            }
        ) == "secret"

    def test_missing_environment_key(self, monkeypatch):
        monkeypatch.delenv(
            "TEST_AGENT_EVAL_KEY",
            raising=False,
        )

        assert ae.resolve_api_key(
            {
                "api_key_env": "TEST_AGENT_EVAL_KEY"
            }
        ) is None


# ===========================================================================
# 6. Dataset
# ===========================================================================

class TestDataset:
    def test_load_valid_dataset(self, tmp_path):
        path = tmp_path / "dataset.jsonl"
        path.write_text(
            '{"id":"a","question":"q1"}\n'
            '{"id":"b","question":"q2",'
            '"ground_truth":"gt",'
            '"contexts":["c1"],'
            '"metadata":{"x":1}}\n',
            encoding="utf-8",
        )

        items = ae.load_dataset(path)

        assert len(items) == 2
        assert items[0].id == "a"
        assert items[0].ground_truth is None
        assert items[1].ground_truth == "gt"
        assert items[1].contexts == ["c1"]
        assert items[1].metadata == {"x": 1}

    def test_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / "dataset.jsonl"
        path.write_text(
            "\n"
            '{"question":"q"}\n'
            "\n",
            encoding="utf-8",
        )

        items = ae.load_dataset(path)

        assert len(items) == 1
        assert items[0].id == "2"

    def test_default_id_uses_line_number(self, tmp_path):
        path = tmp_path / "dataset.jsonl"
        path.write_text(
            '{"question":"q"}\n',
            encoding="utf-8",
        )

        items = ae.load_dataset(path)

        assert items[0].id == "1"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ae.load_dataset(tmp_path / "missing.jsonl")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "dataset.jsonl"
        path.write_text("not json\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid JSON"):
            ae.load_dataset(path)

    def test_non_object_row(self, tmp_path):
        path = tmp_path / "dataset.jsonl"
        path.write_text("[]\n", encoding="utf-8")

        with pytest.raises(
            ValueError,
            match="must be a JSON object",
        ):
            ae.load_dataset(path)

    def test_empty_question(self, tmp_path):
        path = tmp_path / "dataset.jsonl"
        path.write_text(
            '{"question":"   "}\n',
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError,
            match="empty question",
        ):
            ae.load_dataset(path)

    def test_contexts_must_be_list(self, tmp_path):
        path = tmp_path / "dataset.jsonl"
        path.write_text(
            '{"question":"q","contexts":"not-list"}\n',
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError,
            match="contexts must be a list",
        ):
            ae.load_dataset(path)

    def test_empty_dataset(self, tmp_path):
        path = tmp_path / "dataset.jsonl"
        path.write_text("\n", encoding="utf-8")

        with pytest.raises(
            ValueError,
            match="Dataset is empty",
        ):
            ae.load_dataset(path)

    def test_dataset_hash_is_stable(self):
        items = [
            ae.EvalItem(
                id="1",
                question="q",
            )
        ]

        assert ae.dataset_hash(items) == ae.dataset_hash(items)


# ===========================================================================
# 7. BuiltinAgent
# ===========================================================================

class TestBuiltinAgent:
    def test_selects_relevant_context(self):
        agent = ae.BuiltinAgent()

        item = ae.EvalItem(
            id="1",
            question="What is Python?",
            contexts=[
                "Rust uses ownership.",
                "Python is a programming language.",
            ],
        )

        response = asyncio.run(agent.run(item))

        assert (
            response.answer
            == "Python is a programming language."
        )
        assert response.contexts == item.contexts
        assert response.latency_ms >= 0.0
        assert response.tokens_in > 0
        assert response.tokens_out > 0

    def test_falls_back_to_first_context(self):
        agent = ae.BuiltinAgent()

        item = ae.EvalItem(
            id="1",
            question="xyz qwerty",
            contexts=[
                "first context",
                "second context",
            ],
        )

        response = asyncio.run(agent.run(item))

        assert response.answer == "first context"

    def test_without_contexts(self):
        agent = ae.BuiltinAgent()

        item = ae.EvalItem(
            id="1",
            question="xyz qwerty",
            contexts=[],
        )

        response = asyncio.run(agent.run(item))

        assert response.answer == "No relevant context found."
        assert response.contexts == []


# ===========================================================================
# 8. HTTPAgent
# ===========================================================================

class TestHTTPAgent:
    def _cfg(self):
        return {
            "type": "http",
            "url": "http://fake.test/chat",
            "timeout_s": 5,
            "headers": {
                "X-Test": "1",
            },
            "request_template": {
                "question": "{question}",
                "id": "{id}",
                "contexts": "{contexts}",
            },
            "answer_path": "answer",
            "contexts_path": "contexts",
            "prices": {
                "input_per_1m": 1.0,
                "output_per_1m": 2.0,
            },
        }

    def test_success(self, sample_item):
        body = json.dumps(
            {
                "answer": "Python",
                "contexts": ["context"],
            }
        )

        agent = ae.HTTPAgent(self._cfg())

        with patch.object(
            ae,
            "urlopen",
            return_value=FakeHTTPResponse(body),
        ):
            response = asyncio.run(agent.run(sample_item))

        assert response.answer == "Python"
        assert response.contexts == ["context"]
        assert response.latency_ms >= 0.0
        assert response.tokens_in > 0
        assert response.tokens_out > 0
        assert response.raw == {
            "answer": "Python",
            "contexts": ["context"],
        }

    def test_request_template_is_formatted(self, sample_item):
        body = json.dumps(
            {
                "answer": "ok",
                "contexts": [],
            }
        )
        agent = ae.HTTPAgent(self._cfg())

        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeHTTPResponse(body)

        with patch.object(
            ae,
            "urlopen",
            side_effect=fake_urlopen,
        ):
            asyncio.run(agent.run(sample_item))

        payload = json.loads(
            captured["request"].data.decode("utf-8")
        )

        assert payload["question"] == sample_item.question
        assert payload["id"] == sample_item.id
        assert captured["timeout"] == 5.0

    def test_missing_answer(self, sample_item):
        body = json.dumps(
            {
                "contexts": [],
            }
        )

        agent = ae.HTTPAgent(self._cfg())

        with patch.object(
            ae,
            "urlopen",
            return_value=FakeHTTPResponse(body),
        ):
            with pytest.raises(
                ValueError,
                match="missing answer",
            ):
                asyncio.run(agent.run(sample_item))

    def test_contexts_must_be_list(self, sample_item):
        body = json.dumps(
            {
                "answer": "ok",
                "contexts": "invalid",
            }
        )

        agent = ae.HTTPAgent(self._cfg())

        with patch.object(
            ae,
            "urlopen",
            return_value=FakeHTTPResponse(body),
        ):
            with pytest.raises(
                ValueError,
                match="must be a list",
            ):
                asyncio.run(agent.run(sample_item))

    def test_missing_contexts_defaults_to_empty(self, sample_item):
        body = json.dumps(
            {
                "answer": "ok",
            }
        )

        agent = ae.HTTPAgent(self._cfg())

        with patch.object(
            ae,
            "urlopen",
            return_value=FakeHTTPResponse(body),
        ):
            response = asyncio.run(agent.run(sample_item))

        assert response.contexts == []

    def test_invalid_json(self, sample_item):
        agent = ae.HTTPAgent(self._cfg())

        with patch.object(
            ae,
            "urlopen",
            return_value=FakeHTTPResponse("not-json"),
        ):
            with pytest.raises(json.JSONDecodeError):
                asyncio.run(agent.run(sample_item))

    def test_http_error(self, sample_item):
        error = urllib.error.HTTPError(
            "http://fake.test/chat",
            500,
            "server error",
            {},
            None,
        )

        agent = ae.HTTPAgent(self._cfg())

        with patch.object(
            ae,
            "urlopen",
            side_effect=error,
        ):
            with pytest.raises(
                RuntimeError,
                match="Agent HTTP 500",
            ):
                asyncio.run(agent.run(sample_item))

    def test_url_error(self, sample_item):
        error = urllib.error.URLError("connection refused")
        agent = ae.HTTPAgent(self._cfg())

        with patch.object(
            ae,
            "urlopen",
            side_effect=error,
        ):
            with pytest.raises(
                RuntimeError,
                match="Agent connection error",
            ):
                asyncio.run(agent.run(sample_item))


# ===========================================================================
# 9. LLMClient
# ===========================================================================

class TestLLMClient:
    def _cfg(self):
        return {
            "base_url": "http://fake.test/v1",
            "model": "test-model",
            "api_key_env": "TEST_LLM_KEY",
            "timeout_s": 5,
            "retries": 0,
            "temperature": 0.0,
            "max_tokens": 100,
        }

    def test_init_reads_config(self, monkeypatch):
        monkeypatch.setenv(
            "TEST_LLM_KEY",
            "secret",
        )

        client = ae.LLMClient(self._cfg())

        assert client.base_url == "http://fake.test/v1"
        assert client.model == "test-model"
        assert client.timeout_s == 5.0
        assert client.retries == 0
        assert client.temperature == 0.0
        assert client.max_tokens == 100
        assert client.api_key == "secret"

    def test_complete_with_provider_usage(self, monkeypatch):
        monkeypatch.setenv(
            "TEST_LLM_KEY",
            "secret",
        )

        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"score": 0.9}'
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                },
            }
        )

        client = ae.LLMClient(self._cfg())

        with patch.object(
            ae,
            "urlopen",
            return_value=FakeHTTPResponse(body),
        ):
            text, usage = asyncio.run(
                client.complete(
                    [
                        {
                            "role": "user",
                            "content": "hello",
                        }
                    ]
                )
            )

        assert text == '{"score": 0.9}'
        assert usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }

    def test_complete_estimates_missing_usage(self):
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "hello"
                        }
                    }
                ]
            }
        )

        client = ae.LLMClient(self._cfg())

        with patch.object(
            ae,
            "urlopen",
            return_value=FakeHTTPResponse(body),
        ):
            text, usage = asyncio.run(
                client.complete(
                    [
                        {
                            "role": "user",
                            "content": "hello",
                        }
                    ]
                )
            )

        assert text == "hello"
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0

    def test_complete_requires_choices(self):
        body = json.dumps(
            {
                "choices": [],
            }
        )

        client = ae.LLMClient(self._cfg())

        with patch.object(
            ae,
            "urlopen",
            return_value=FakeHTTPResponse(body),
        ):
            with pytest.raises(
                RuntimeError,
                match="no choices",
            ):
                asyncio.run(client.complete([]))

    def test_complete_requires_content(self):
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {},
                    }
                ]
            }
        )

        client = ae.LLMClient(self._cfg())

        with patch.object(
            ae,
            "urlopen",
            return_value=FakeHTTPResponse(body),
        ):
            with pytest.raises(
                RuntimeError,
                match="no message content",
            ):
                asyncio.run(client.complete([]))

    def test_http_error_is_wrapped(self):
        error = urllib.error.HTTPError(
            "http://fake.test/v1/chat/completions",
            500,
            "server error",
            {},
            FakeHTTPResponse('{"error":"boom"}'),
        )

        client = ae.LLMClient(self._cfg())

        with patch.object(
            ae,
            "urlopen",
            side_effect=error,
        ):
            with pytest.raises(
                RuntimeError,
                match="LLM HTTP 500",
            ):
                asyncio.run(client.complete([]))

    def test_url_error_is_wrapped(self):
        client = ae.LLMClient(self._cfg())

        with patch.object(
            ae,
            "urlopen",
            side_effect=urllib.error.URLError(
                "connection refused"
            ),
        ):
            with pytest.raises(
                RuntimeError,
                match="LLM connection error",
            ):
                asyncio.run(client.complete([]))

    def test_retries_then_fails(self):
        cfg = self._cfg()
        cfg["retries"] = 2

        client = ae.LLMClient(cfg)

        async def no_sleep(_):
            return None

        with patch.object(
            ae,
            "urlopen",
            side_effect=urllib.error.URLError(
                "temporary failure"
            ),
        ), patch.object(
            ae.asyncio,
            "sleep",
            new=no_sleep,
        ):
            with pytest.raises(
                RuntimeError,
                match="LLM connection error",
            ):
                asyncio.run(client.complete([]))


# ===========================================================================
# 10. Judge
# ===========================================================================

class TestJudge:
    def test_unknown_metric(self, sample_item, sample_response):
        judge = ae.Judge(FakeJudgeClient())

        with pytest.raises(
            ValueError,
            match="Unknown metric",
        ):
            asyncio.run(
                judge.evaluate(
                    "unknown",
                    sample_item,
                    sample_response,
                )
            )

    def test_context_recall_without_ground_truth(
        self,
        sample_response,
    ):
        item = ae.EvalItem(
            id="1",
            question="q",
            ground_truth=None,
        )
        judge = ae.Judge(FakeJudgeClient(0.9))

        result = asyncio.run(
            judge.evaluate(
                "context_recall",
                item,
                sample_response,
            )
        )

        assert result.score == 0.0
        assert "no ground truth" in result.reasoning

    def test_faithfulness(self, sample_item, sample_response):
        judge = ae.Judge(
            FakeJudgeClient(
                score=0.85,
                reasoning="supported",
            )
        )

        result = asyncio.run(
            judge.evaluate(
                "faithfulness",
                sample_item,
                sample_response,
            )
        )

        assert result.metric == "faithfulness"
        assert result.score == pytest.approx(0.85)
        assert result.reasoning == "supported"
        assert result.tokens_in == 10
        assert result.tokens_out == 5
        assert result.raw["score"] == 0.85

    def test_score_is_clamped_high(
        self,
        sample_item,
        sample_response,
    ):
        judge = ae.Judge(FakeJudgeClient(score=2.0))

        result = asyncio.run(
            judge.evaluate(
                "answer_relevance",
                sample_item,
                sample_response,
            )
        )

        assert result.score == 1.0

    def test_score_is_clamped_low(
        self,
        sample_item,
        sample_response,
    ):
        judge = ae.Judge(FakeJudgeClient(score=-1.0))

        result = asyncio.run(
            judge.evaluate(
                "answer_relevance",
                sample_item,
                sample_response,
            )
        )

        assert result.score == 0.0

    def test_non_object_json(self, sample_item, sample_response):
        class ArrayClient:
            async def complete(self, messages):
                return "[1, 2, 3]", {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                }

        judge = ae.Judge(ArrayClient())

        with pytest.raises(
            ValueError,
            match="JSON object",
        ):
            asyncio.run(
                judge.evaluate(
                    "faithfulness",
                    sample_item,
                    sample_response,
                )
            )

    def test_missing_score_defaults_to_zero(
        self,
        sample_item,
        sample_response,
    ):
        class NoScoreClient:
            async def complete(self, messages):
                return '{"reasoning":"missing"}', {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                }

        judge = ae.Judge(NoScoreClient())

        result = asyncio.run(
            judge.evaluate(
                "faithfulness",
                sample_item,
                sample_response,
            )
        )

        assert result.score == 0.0


# ===========================================================================
# 11. Evaluator
# ===========================================================================

class TestEvaluator:
    def _config(self, base_config):
        cfg = json.loads(json.dumps(base_config))
        cfg["eval"]["metrics"] = [
            "faithfulness",
            "answer_relevance",
        ]
        cfg["eval"]["concurrency"] = 2
        cfg["eval"]["pass_threshold"] = 0.7
        return cfg

    def test_success_with_judge(
        self,
        base_config,
        sample_item,
    ):
        cfg = self._config(base_config)

        evaluator = ae.Evaluator(
            agent=ae.BuiltinAgent(),
            judge=ae.Judge(FakeJudgeClient(0.9)),
            cfg=cfg,
        )

        result = asyncio.run(
            evaluator.evaluate_item(sample_item)
        )

        assert result.response is not None
        assert result.passed is True
        assert len(result.judgments) == 2
        assert result.error is None

    def test_fails_threshold(
        self,
        base_config,
        sample_item,
    ):
        cfg = self._config(base_config)
        cfg["eval"]["pass_threshold"] = 0.95

        evaluator = ae.Evaluator(
            agent=ae.BuiltinAgent(),
            judge=ae.Judge(FakeJudgeClient(0.9)),
            cfg=cfg,
        )

        result = asyncio.run(
            evaluator.evaluate_item(sample_item)
        )

        assert result.passed is False

    def test_no_metrics_does_not_require_judge(
        self,
        base_config,
        sample_item,
    ):
        cfg = self._config(base_config)
        cfg["eval"]["metrics"] = []

        evaluator = ae.Evaluator(
            agent=ae.BuiltinAgent(),
            judge=None,
            cfg=cfg,
        )

        result = asyncio.run(
            evaluator.evaluate_item(sample_item)
        )

        assert result.response is not None
        assert result.judgments == []

    def test_metrics_require_judge(
        self,
        base_config,
        sample_item,
    ):
        cfg = self._config(base_config)

        evaluator = ae.Evaluator(
            agent=ae.BuiltinAgent(),
            judge=None,
            cfg=cfg,
        )

        result = asyncio.run(
            evaluator.evaluate_item(sample_item)
        )

        assert result.response is None
        assert result.error_type == "RuntimeError"
        assert "LLM judge is required" in result.error

    def test_agent_failure_is_captured(
        self,
        base_config,
        sample_item,
    ):
        class BrokenAgent:
            async def run(self, item):
                raise RuntimeError("agent exploded")

        cfg = self._config(base_config)

        evaluator = ae.Evaluator(
            agent=BrokenAgent(),
            judge=None,
            cfg=cfg,
        )

        result = asyncio.run(
            evaluator.evaluate_item(sample_item)
        )

        assert result.response is None
        assert result.passed is False
        assert result.error == "agent exploded"
        assert result.error_type == "RuntimeError"

    def test_run_multiple_items(self, base_config):
        cfg = self._config(base_config)

        evaluator = ae.Evaluator(
            agent=ae.BuiltinAgent(),
            judge=ae.Judge(FakeJudgeClient(0.9)),
            cfg=cfg,
        )

        items = [
            ae.EvalItem(
                id="1",
                question="What is Python?",
                contexts=["Python is a language."],
            ),
            ae.EvalItem(
                id="2",
                question="What is SQLite?",
                contexts=["SQLite is a database."],
            ),
        ]

        results = asyncio.run(evaluator.run(items))

        assert len(results) == 2
        assert [r.item.id for r in results] == ["1", "2"]

    def test_calculate_cost(self, base_config):
        cfg = json.loads(json.dumps(base_config))
        cfg["agent"]["prices"] = {
            "input_per_1m": 1.0,
            "output_per_1m": 2.0,
        }
        cfg["judge"]["prices"] = {
            "input_per_1m": 3.0,
            "output_per_1m": 4.0,
        }

        evaluator = ae.Evaluator(
            agent=ae.BuiltinAgent(),
            judge=None,
            cfg=cfg,
        )

        response = ae.AgentResponse(
            answer="a",
            tokens_in=1_000_000,
            tokens_out=500_000,
        )
        judgment = ae.Judgment(
            metric="faithfulness",
            score=1.0,
            tokens_in=2_000_000,
            tokens_out=250_000,
        )

        cost = evaluator._calculate_cost(
            response,
            [judgment],
        )

        expected = 1.0 + 1.0 + 6.0 + 1.0

        assert cost == pytest.approx(expected)

    def test_calculate_cost_without_response(self, base_config):
        evaluator = ae.Evaluator(
            agent=ae.BuiltinAgent(),
            judge=None,
            cfg=base_config,
        )

        assert (
            evaluator._calculate_cost(
                None,
                [],
            )
            == 0.0
        )


# ===========================================================================
# 12. Aggregation
# ===========================================================================

class TestAggregateResults:
    def test_empty(self):
        summary = ae.aggregate_results([])

        assert summary["total"] == 0
        assert summary["passed"] == 0
        assert summary["failed"] == 0
        assert summary["errors"] == 0
        assert summary["pass_rate"] == 0.0
        assert summary["cost_usd"] == 0.0

    def test_complete_summary(self):
        item = ae.EvalItem(
            id="1",
            question="q",
        )

        result = ae.ItemResult(
            item=item,
            response=ae.AgentResponse(
                answer="a",
                latency_ms=10,
                tokens_in=100,
                tokens_out=50,
            ),
            judgments=[
                ae.Judgment(
                    metric="faithfulness",
                    score=1.0,
                    tokens_in=20,
                    tokens_out=5,
                )
            ],
            passed=True,
            cost_usd=0.5,
        )

        summary = ae.aggregate_results([result])

        assert summary["total"] == 1
        assert summary["passed"] == 1
        assert summary["failed"] == 0
        assert summary["errors"] == 0
        assert summary["pass_rate"] == 1.0
        assert summary["latency_ms"]["mean"] == 10
        assert summary["latency_ms"]["p50"] == 10
        assert summary["latency_ms"]["p95"] == 10
        assert summary["tokens"]["agent_in"] == 100
        assert summary["tokens"]["agent_out"] == 50
        assert summary["tokens"]["judge_in"] == 20
        assert summary["tokens"]["judge_out"] == 5
        assert summary["tokens"]["total"] == 175
        assert summary["metrics"]["faithfulness"] == 1.0
        assert summary["cost_usd"] == 0.5

    def test_failed_result_without_response(self):
        result = ae.ItemResult(
            item=ae.EvalItem(
                id="1",
                question="q",
            ),
            response=None,
            passed=False,
            error="boom",
            error_type="RuntimeError",
        )

        summary = ae.aggregate_results([result])

        assert summary["total"] == 1
        assert summary["passed"] == 0
        assert summary["failed"] == 1
        assert summary["errors"] == 1
        assert summary["pass_rate"] == 0.0


# ===========================================================================
# 13. SQLite persistence
# ===========================================================================

def _persist_kwargs(cfg):
    return {
        "run_id": "run-1",
        "dataset_hash_value": "dataset-hash",
        "config_hash_value": "config-hash",
        "cfg": cfg,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "elapsed_s": 1.0,
    }


class TestSQLite:
    def test_persist_results(self, tmp_path, base_config):
        db = tmp_path / "results.db"

        result = ae.ItemResult(
            item=ae.EvalItem(
                id="q1",
                question="Question",
                ground_truth="Answer",
                contexts=["gold"],
            ),
            response=ae.AgentResponse(
                answer="Answer",
                contexts=["ctx"],
                latency_ms=10.0,
                tokens_in=10,
                tokens_out=5,
            ),
            judgments=[
                ae.Judgment(
                    metric="faithfulness",
                    score=0.9,
                    reasoning="supported",
                )
            ],
            passed=True,
            cost_usd=0.001,
        )

        ae.persist_results(
            [result],
            db,
            **_persist_kwargs(base_config),
        )

        conn = sqlite3.connect(db)

        try:
            assert conn.execute(
                "SELECT COUNT(*) FROM runs"
            ).fetchone()[0] == 1

            assert conn.execute(
                "SELECT COUNT(*) FROM evaluations"
            ).fetchone()[0] == 1

            assert conn.execute(
                "SELECT COUNT(*) FROM metrics"
            ).fetchone()[0] == 1

            row = conn.execute(
                """
                SELECT id, answer, passed, cost_usd
                FROM evaluations
                """
            ).fetchone()

            assert row == (
                "q1",
                "Answer",
                1,
                0.001,
            )
        finally:
            conn.close()

    def test_historical_runs_are_preserved(
        self,
        tmp_path,
        base_config,
    ):
        db = tmp_path / "results.db"

        result = ae.ItemResult(
            item=ae.EvalItem(
                id="q1",
                question="q",
            ),
            response=ae.AgentResponse(answer="a"),
        )

        ae.persist_results(
            [result],
            db,
            **_persist_kwargs(base_config),
        )

        second_kwargs = _persist_kwargs(base_config)
        second_kwargs["run_id"] = "run-2"

        ae.persist_results(
            [result],
            db,
            **second_kwargs,
        )

        conn = sqlite3.connect(db)

        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM runs"
            ).fetchone()[0]
            assert count == 2
        finally:
            conn.close()


# ===========================================================================
# 14. Reporting
# ===========================================================================

class TestRenderReport:
    def _result(self, score, passed, error=None):
        return ae.ItemResult(
            item=ae.EvalItem(
                id="q1",
                question="Question?",
            ),
            response=ae.AgentResponse(
                answer="Answer",
                contexts=["Context"],
                latency_ms=12.5,
                tokens_in=10,
                tokens_out=5,
            ),
            judgments=[
                ae.Judgment(
                    metric="faithfulness",
                    score=score,
                    reasoning="reason",
                )
            ],
            passed=passed,
            error=error,
            error_type="RuntimeError" if error else None,
            cost_usd=0.001,
        )

    def test_report_contains_summary(self, base_config):
        report = ae.render_report(
            [
                self._result(0.9, True),
            ],
            base_config,
            dataset_hash_value="dataset",
            config_hash_value="config",
            run_id="run",
            elapsed_s=1.23,
        )

        assert "# Agent-Eval Report" in report
        assert "dataset" in report
        assert "config" in report
        assert "run" in report
        assert "faithfulness" in report

    def test_report_contains_configuration(self, base_config):
        report = ae.render_report(
            [],
            base_config,
            dataset_hash_value="d",
            config_hash_value="c",
            run_id="r",
            elapsed_s=0.1,
        )

        assert "## Configuration" in report
        assert "```json" in report
        assert '"judge"' in report

    def test_report_contains_results_table(self, base_config):
        report = ae.render_report(
            [
                self._result(0.9, True),
            ],
            base_config,
            dataset_hash_value="d",
            config_hash_value="c",
            run_id="r",
            elapsed_s=0.1,
        )

        assert "## Results" in report
        assert "q1" in report
        assert "12.5" in report

    def test_report_contains_failures(self, base_config):
        report = ae.render_report(
            [
                self._result(
                    0.2,
                    False,
                    error="boom",
                ),
            ],
            base_config,
            dataset_hash_value="d",
            config_hash_value="c",
            run_id="r",
            elapsed_s=0.1,
        )

        assert "## Failures" in report
        assert "RuntimeError" in report
        assert "boom" in report
        assert "Answer" in report

    def test_report_without_metrics(self, base_config):
        cfg = json.loads(json.dumps(base_config))
        cfg["eval"]["metrics"] = []

        report = ae.render_report(
            [],
            cfg,
            dataset_hash_value="d",
            config_hash_value="c",
            run_id="r",
            elapsed_s=0.1,
        )

        assert "No LLM metrics were configured." in report


# ===========================================================================
# 15. Demo
# ===========================================================================

class TestDemo:
    def test_demo_dataset(self):
        items = ae.demo_dataset()

        assert len(items) == 3
        assert all(
            isinstance(item, ae.EvalItem)
            for item in items
        )
        assert all(
            item.question
            for item in items
        )

    def test_demo_dataset_hash(self):
        items = ae.demo_dataset()

        assert ae.dataset_hash(items)

    def test_build_builtin_agent(self, base_config):
        cfg = json.loads(json.dumps(base_config))
        cfg["agent"]["type"] = "builtin"

        agent = ae.build_agent(cfg)

        assert isinstance(agent, ae.BuiltinAgent)

    def test_build_http_agent(self, base_config):
        cfg = json.loads(json.dumps(base_config))
        cfg["agent"]["type"] = "http"

        agent = ae.build_agent(cfg)

        assert isinstance(agent, ae.HTTPAgent)

    def test_build_unknown_agent(self, base_config):
        cfg = json.loads(json.dumps(base_config))
        cfg["agent"]["type"] = "unknown"

        with pytest.raises(
            ValueError,
            match="Unsupported agent type",
        ):
            ae.build_agent(cfg)

    def test_build_judge_without_metrics(self, base_config):
        cfg = json.loads(json.dumps(base_config))
        cfg["eval"]["metrics"] = []

        assert ae.build_judge(cfg) is None

    def test_build_judge_with_metrics(self, base_config):
        cfg = json.loads(json.dumps(base_config))
        cfg["eval"]["metrics"] = ["faithfulness"]

        judge = ae.build_judge(cfg)

        assert isinstance(judge, ae.Judge)


# ===========================================================================
# 16. Self-test embebido
# ===========================================================================

class TestEmbeddedSelftest:
    def test_selftest_passes(self):
        ae.selftest()


# ===========================================================================
# 17. CLI
# ===========================================================================

class TestCLI:
    def test_parse_args_defaults(self):
        with patch.object(
            sys,
            "argv",
            ["agent-eval.py"],
        ):
            args = ae.parse_args()

        assert args.config is None
        assert args.dataset is None
        assert args.agent_url is None
        assert args.judge_model is None
        assert args.output is None
        assert args.concurrency is None
        assert args.demo is False
        assert args.selftest is False
        assert args.no_sqlite is False
        assert args.verbose is False

    def test_parse_args_flags(self):
        with patch.object(
            sys,
            "argv",
            [
                "agent-eval.py",
                "--demo",
                "--no-sqlite",
                "--verbose",
                "--concurrency",
                "2",
                "--output",
                "report.md",
            ],
        ):
            args = ae.parse_args()

        assert args.demo is True
        assert args.no_sqlite is True
        assert args.verbose is True
        assert args.concurrency == 2
        assert args.output == "report.md"
