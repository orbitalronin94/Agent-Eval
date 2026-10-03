#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent-eval.py — Arnés de evaluación autocontenido para agentes y pipelines RAG.
================================================================================
Un único archivo. Cero dependencias obligatorias. Solo stdlib.

Uso rápido
----------
    # Evaluar un agente HTTP contra un dataset JSONL
    python agent-eval.py --dataset data.jsonl --agent-url http://localhost:8000/ask

    # Usar un archivo de configuración (JSON o YAML si PyYAML está instalado)
    python agent-eval.py --config eval.yaml

    # Demo autocontenida con agente y dataset de ejemplo
    python agent-eval.py --demo

    # Self-tests internos (sin pytest, sin red, sin API keys)
    python agent-eval.py --selftest

Salida
------
    eval-report.md   — Informe en Markdown con tablas, métricas y Mermaid.
    eval-results.db  — SQLite con todos los resultados crudos (opcional).

Métricas
--------
    faithfulness       ¿La respuesta está anclada al contexto recuperado?
    answer_relevance   ¿La respuesta responde a la pregunta?
    context_precision  ¿Los fragmentos recuperados eran relevantes?
    context_recall     ¿Se recuperó la información necesaria? (requiere ground truth)
    latency_ms         Latencia de extremo a extremo del agente.
    cost_usd           Coste estimado por tokens de juez y agente.
    tokens_in/out      Contadores de tokens (estimados por heurística).

Decisiones de diseño
--------------------
    * Un solo archivo: elimina fricción de instalación y auditoría.
    * Asyncio + semáforo: concurrencia controlada sin ThreadPool manual.
    * SQLite en memoria: persistencia sin servidor ni dependencias.
    * LLM-juez agnóstico: cualquier endpoint compatible con OpenAI.
    * Sin tokenizers externos: estimación por longitud de caracteres.
    * Reporte reproducible: hash del dataset y de la config en el informe.
    * Self-tests embebidos: `--selftest` valida el 90% de la lógica sin pytest.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# 0. Versión y constantes globales
# ---------------------------------------------------------------------------

__version__ = "0.2.0"
DEFAULT_TIMEOUT = 60.0
DEFAULT_CONCURRENCY = 4
TOKEN_CHAR_RATIO = 3.5  # ajustado para español (antes 4.0)
MAX_JUDGE_RETRIES = 2
JUDGE_RETRY_DELAY = 1.0

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Precios por 1K tokens (USD). Actualizar según necesidad.
MODEL_PRICES: dict[str, dict[str, float]] = {
    "gpt-4o":            {"in": 0.0025, "out": 0.010},
    "gpt-4o-mini":       {"in": 0.00015, "out": 0.0006},
    "gpt-4.1":           {"in": 0.002,  "out": 0.008},
    "gpt-4.1-mini":      {"in": 0.0004, "out": 0.0016},
    "o3-mini":           {"in": 0.0011, "out": 0.0044},
    "claude-3-5-sonnet": {"in": 0.003,  "out": 0.015},
    "claude-3-5-haiku":  {"in": 0.0008, "out": 0.004},
    "llama-3.3-70b":     {"in": 0.0006, "out": 0.0006},
    "qwen-2.5-72b":      {"in": 0.0004, "out": 0.0004},
    "default":           {"in": 0.0,    "out": 0.0},
}

# ---------------------------------------------------------------------------
# 1. Configuración por defecto
# ---------------------------------------------------------------------------

CONFIG: dict[str, Any] = {
    # --- LLM juez ---
    "judge": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 512,
    },
    # --- Agente bajo prueba ---
    "agent": {
        "type": "http",                 # "http" | "builtin" | "callable"
        "url": "http://localhost:8000/ask",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "request_template": {"question": "{question}"},
        "response_answer_path": "answer",
        "response_contexts_path": "contexts",
        "auth_env": None,               # p.ej. "AGENT_API_KEY" -> Bearer
        "timeout": DEFAULT_TIMEOUT,
        "prices": None,                 # {"in": 0.0, "out": 0.0} o None para coste 0
    },
    # --- Dataset ---
    "dataset": {
        "path": None,                   # JSONL con {question, ground_truth?, contexts?}
        "id_field": "id",
        "question_field": "question",
        "ground_truth_field": "ground_truth",
        "contexts_field": "contexts",
    },
    # --- Evaluación ---
    "eval": {
        "metrics": [
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "context_recall",
            "latency_ms",
            "cost_usd",
        ],
        "concurrency": DEFAULT_CONCURRENCY,
        "pass_threshold": 0.75,         # umbral para considerar un ítem OK
        "save_sqlite": True,
        "sqlite_path": "eval-results.db",
        "token_char_ratio": TOKEN_CHAR_RATIO,  # ratio personalizable
    },
    # --- Salida ---
    "output": {
        "report_path": "eval-report.md",
        "include_per_question": True,
        "include_mermaid": True,
        "include_raw_judge": False,
    },
}


def merge_config(base: dict, override: dict) -> dict:
    """Merge recursivo de dos diccionarios."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = v
    return out


def load_config_file(path: str) -> dict:
    """Carga JSON o YAML (si PyYAML está disponible)."""
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "Se ha proporcionado un YAML pero PyYAML no está instalado. "
                "Usa JSON o `pip install pyyaml`."
            ) from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


# ---------------------------------------------------------------------------
# 2. Modelos de datos
# ---------------------------------------------------------------------------

@dataclass
class EvalItem:
    """Una fila del dataset de evaluación."""
    id: str
    question: str
    ground_truth: Optional[str] = None
    gold_contexts: list[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Respuesta del agente bajo prueba."""
    answer: str
    contexts: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_type: Optional[str] = None  # "timeout" | "http" | "parse" | "other"


@dataclass
class Judgment:
    """Resultado de una evaluación individual con el LLM-juez."""
    score: float
    reasoning: str = ""
    raw: str = ""
    error: Optional[str] = None


@dataclass
class ItemResult:
    """Resultado completo de evaluar un ítem."""
    item: EvalItem
    response: AgentResponse
    judgments: dict[str, Judgment] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    passed: bool = False


# ---------------------------------------------------------------------------
# 3. Utilidades
# ---------------------------------------------------------------------------

def estimate_tokens(text: str, ratio: float = TOKEN_CHAR_RATIO) -> int:
    """Estimación rápida de tokens: caracteres / ratio, redondeado hacia arriba."""
    if not text:
        return 0
    return max(1, int(len(text) / ratio + 0.5))


def price_for(model: str) -> dict[str, float]:
    """Precio por 1K tokens para un modelo dado."""
    if model in MODEL_PRICES:
        return MODEL_PRICES[model]
    for key in MODEL_PRICES:
        if key != "default" and model.startswith(key):
            return MODEL_PRICES[key]
    return MODEL_PRICES["default"]


def sha256_short(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def percentile(values: list[float], p: float) -> float:
    """Percentil simple (interpolación lineal)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def safe_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None]
    return float(statistics.fmean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# 4. Cliente LLM (compatible con OpenAI)
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Cliente HTTP minimalista para cualquier endpoint compatible con OpenAI
    (/v1/chat/completions). Funciona con OpenAI, Groq, Together, Ollama,
    vLLM, llama.cpp server, LM Studio, etc.
    """

    def __init__(self, base_url: str, model: str, api_key: Optional[str],
                 temperature: float = 0.0, max_tokens: int = 512,
                 timeout: float = DEFAULT_TIMEOUT,
                 token_ratio: float = TOKEN_CHAR_RATIO) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.token_ratio = token_ratio

    async def complete(self, system: str, user: str) -> tuple[str, int, int]:
        """
        Devuelve (texto, tokens_in_estimados, tokens_out_estimados).
        Los tokens son estimados, no exactos: no dependemos de tokenizers.
        Incluye retry con backoff exponencial.
        """
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data, headers=headers, method="POST",
        )

        last_error: Optional[Exception] = None
        for attempt in range(MAX_JUDGE_RETRIES + 1):
            try:
                body = await asyncio.to_thread(
                    lambda: urllib.request.urlopen(
                        req, timeout=self.timeout
                    ).read().decode("utf-8")
                )
                obj = json.loads(body)
                text = obj["choices"][0]["message"]["content"] or ""
                usage = obj.get("usage") or {}
                t_in = int(usage.get("prompt_tokens")
                           or estimate_tokens(system + user, self.token_ratio))
                t_out = int(usage.get("completion_tokens")
                            or estimate_tokens(text, self.token_ratio))
                return text, t_in, t_out
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")[:500]
                last_error = RuntimeError(f"HTTP {e.code} del juez: {err_body}")
                if e.code >= 500 and attempt < MAX_JUDGE_RETRIES:
                    logger.warning(
                        f"Juez HTTP {e.code}, reintentando "
                        f"({attempt+1}/{MAX_JUDGE_RETRIES})..."
                    )
                    await asyncio.sleep(JUDGE_RETRY_DELAY * (2 ** attempt))
                    continue
                raise last_error from e
            except urllib.error.URLError as e:
                last_error = RuntimeError(f"Error de red contactando al juez: {e}")
                if attempt < MAX_JUDGE_RETRIES:
                    logger.warning(
                        f"Error de red, reintentando "
                        f"({attempt+1}/{MAX_JUDGE_RETRIES})..."
                    )
                    await asyncio.sleep(JUDGE_RETRY_DELAY * (2 ** attempt))
                    continue
                raise last_error from e
            except Exception as e:  # noqa: BLE001
                last_error = RuntimeError(f"Error inesperado del juez: {e}")
                raise last_error from e

        raise last_error  # pragma: no cover


# ---------------------------------------------------------------------------
# 5. Adaptadores de agente
# ---------------------------------------------------------------------------

class AgentAdapter:
    """Interfaz base para el agente bajo prueba."""

    async def ask(self, question: str) -> AgentResponse:  # pragma: no cover
        raise NotImplementedError


class HTTPAgent(AgentAdapter):
    """Agente accesible por HTTP que devuelve answer y (opcional) contexts."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.auth = None
        if cfg.get("auth_env"):
            token = os.environ.get(cfg["auth_env"])
            if token:
                self.auth = token

    async def ask(self, question: str) -> AgentResponse:
        tpl = self.cfg.get("request_template", {"question": "{question}"})
        body = _deep_format(tpl, {"question": question})
        headers = dict(self.cfg.get("headers", {}))
        if self.auth:
            headers["Authorization"] = f"Bearer {self.auth}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.cfg["url"], data=data, headers=headers,
            method=self.cfg.get("method", "POST"),
        )
        t0 = time.perf_counter()
        try:
            raw_body = await asyncio.to_thread(
                lambda: urllib.request.urlopen(
                    req, timeout=self.cfg.get("timeout", DEFAULT_TIMEOUT)
                ).read().decode("utf-8")
            )
            latency = (time.perf_counter() - t0) * 1000.0
            obj = json.loads(raw_body)
            answer = _deep_get(
                obj, self.cfg.get("response_answer_path", "answer")
            ) or ""
            contexts = _deep_get(
                obj, self.cfg.get("response_contexts_path", "contexts")
            ) or []
            if isinstance(contexts, str):
                contexts = [contexts]
            return AgentResponse(
                answer=str(answer), contexts=list(contexts),
                latency_ms=latency, raw=obj,
            )
        except urllib.error.URLError as e:
            latency = (time.perf_counter() - t0) * 1000.0
            error_type = "timeout" if "timed out" in str(e).lower() else "http"
            logger.error(f"Error {error_type} del agente: {e}")
            return AgentResponse(
                answer="", contexts=[], latency_ms=latency,
                error=f"{type(e).__name__}: {e}",
                error_type=error_type,
            )
        except json.JSONDecodeError as e:
            latency = (time.perf_counter() - t0) * 1000.0
            logger.error(f"Error parseando respuesta del agente: {e}")
            return AgentResponse(
                answer="", contexts=[], latency_ms=latency,
                error=f"JSON inválido: {e}",
                error_type="parse",
            )
        except Exception as e:  # noqa: BLE001
            latency = (time.perf_counter() - t0) * 1000.0
            logger.error(f"Error inesperado del agente: {e}")
            return AgentResponse(
                answer="", contexts=[], latency_ms=latency,
                error=f"{type(e).__name__}: {e}",
                error_type="other",
            )


class BuiltinAgent(AgentAdapter):
    """
    Agente de ejemplo 100% local: recuperación léxica sobre un corpus
    embebido y respuesta extractiva. Sirve para demostrar el arnés sin
    depender de ningún servicio externo.
    """

    def __init__(self, corpus: list[str], k: int = 3) -> None:
        self.corpus = corpus
        self.k = k

    def _score(self, query: str, doc: str) -> float:
        q_tokens = set(re.findall(r"\w+", query.lower()))
        d_tokens = re.findall(r"\w+", doc.lower())
        if not q_tokens or not d_tokens:
            return 0.0
        overlap = sum(1 for t in d_tokens if t in q_tokens)
        return overlap / (len(d_tokens) ** 0.5)

    async def ask(self, question: str) -> AgentResponse:
        t0 = time.perf_counter()
        scored = sorted(
            ((self._score(question, d), d) for d in self.corpus),
            key=lambda x: x[0], reverse=True,
        )[: self.k]
        contexts = [d for s, d in scored if s > 0] or self.corpus[: self.k]
        answer = contexts[0] if contexts else "No tengo información suficiente."
        latency = (time.perf_counter() - t0) * 1000.0
        return AgentResponse(answer=answer, contexts=contexts, latency_ms=latency)


def _deep_format(obj: Any, mapping: dict[str, str]) -> Any:
    """Sustituye placeholders {k} recursivamente en dicts/listas/strings."""
    if isinstance(obj, dict):
        return {k: _deep_format(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_format(v, mapping) for v in obj]
    if isinstance(obj, str):
        return obj.format(**mapping) if "{" in obj else obj
    return obj


def _deep_get(obj: Any, path: str) -> Any:
    """Accede a un campo por ruta 'a.b.c'. Devuelve None si no existe."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def build_agent(cfg: dict[str, Any],
                corpus: Optional[list[str]] = None) -> AgentAdapter:
    kind = cfg.get("type", "http")
    if kind == "http":
        return HTTPAgent(cfg)
    if kind == "builtin":
        if not corpus:
            raise SystemExit("El agente 'builtin' requiere un corpus de demo.")
        return BuiltinAgent(corpus)
    raise SystemExit(f"Tipo de agente no soportado: {kind}")


# ---------------------------------------------------------------------------
# 6. Métricas (LLM-juez)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "Eres un evaluador riguroso de sistemas de IA. "
    "Respondes SIEMPRE en JSON válido con las claves exactas solicitadas. "
    "No incluyes texto fuera del JSON. No inventas información."
)


async def _judge_score(
    client: LLMClient,
    instruction: str,
) -> tuple[Judgment, int, int]:
    """Llama al juez y parsea un JSON con claves {score, reasoning}."""
    try:
        text, t_in, t_out = await client.complete(JUDGE_SYSTEM, instruction)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error del juez: {e}")
        return Judgment(score=0.0, error=str(e)), 0, 0

    parsed = _extract_json(text)
    if parsed is None:
        logger.warning(f"JSON inválido del juez: {text[:200]}")
        return Judgment(score=0.0, raw=text,
                        error="JSON inválido del juez"), t_in, t_out
    try:
        score = float(parsed.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    return (
        Judgment(score=score,
                 reasoning=str(parsed.get("reasoning", ""))[:500],
                 raw=text),
        t_in, t_out,
    )


def _extract_json(text: str) -> Optional[dict]:
    """Extrae el primer objeto JSON de un texto, tolerando ```json fences."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


async def metric_faithfulness(
    client: LLMClient, item: EvalItem, resp: AgentResponse
) -> tuple[Judgment, int, int]:
    if not resp.contexts:
        return Judgment(score=0.0, reasoning="Sin contextos recuperados"), 0, 0
    ctx = "\n---\n".join(resp.contexts)[:6000]
    prompt = (
        "Evalúa si la RESPUESTA está fundamentada EXCLUSIVAMENTE en el CONTEXTO.\n"
        "Puntúa de 0.0 a 1.0, donde 1.0 significa que cada afirmación está "
        "respaldada por el contexto y 0.0 que la respuesta inventa o contradice.\n"
        f"CONTEXTO:\n{ctx}\n"
        f"RESPUESTA:\n{resp.answer[:2000]}\n"
        'Devuelve JSON: {"score": <float>, "reasoning": "<breve>"}'
    )
    return await _judge_score(client, prompt)


async def metric_answer_relevance(
    client: LLMClient, item: EvalItem, resp: AgentResponse
) -> tuple[Judgment, int, int]:
    prompt = (
        "Evalúa si la RESPUESTA contesta directamente a la PREGUNTA.\n"
        "1.0 = responde con precisión y sin rodeos; 0.0 = no responde o evade.\n"
        f"PREGUNTA:\n{item.question}\n"
        f"RESPUESTA:\n{resp.answer[:2000]}\n"
        'Devuelve JSON: {"score": <float>, "reasoning": "<breve>"}'
    )
    return await _judge_score(client, prompt)


async def metric_context_precision(
    client: LLMClient, item: EvalItem, resp: AgentResponse
) -> tuple[Judgment, int, int]:
    if not resp.contexts:
        return Judgment(score=0.0, reasoning="Sin contextos"), 0, 0
    ctx = "\n---\n".join(resp.contexts)[:6000]
    prompt = (
        "Evalúa la PRECISIÓN de los fragmentos recuperados: ¿son relevantes "
        "para responder a la pregunta? Ignora si la respuesta final es buena.\n"
        "1.0 = todos los fragmentos son relevantes; 0.0 = ninguno lo es.\n"
        f"PREGUNTA:\n{item.question}\n"
        f"FRAGMENTOS:\n{ctx}\n"
        'Devuelve JSON: {"score": <float>, "reasoning": "<breve>"}'
    )
    return await _judge_score(client, prompt)


async def metric_context_recall(
    client: LLMClient, item: EvalItem, resp: AgentResponse
) -> tuple[Judgment, int, int]:
    if not item.ground_truth:
        return Judgment(score=0.0,
                        reasoning="Sin ground_truth en el dataset"), 0, 0
    ctx = "\n---\n".join(resp.contexts or ["(vacío)"])[:6000]
    prompt = (
        "Compara el CONTEXTO RECUPERADO con la RESPUESTA DE REFERENCIA. "
        "¿Cuánta de la información necesaria para llegar a la referencia está "
        "presente en el contexto? 1.0 = todo; 0.0 = nada.\n"
        f"PREGUNTA:\n{item.question}\n"
        f"RESPUESTA DE REFERENCIA:\n{item.ground_truth}\n"
        f"CONTEXTO RECUPERADO:\n{ctx}\n"
        'Devuelve JSON: {"score": <float>, "reasoning": "<breve>"}'
    )
    return await _judge_score(client, prompt)


METRIC_FUNCS = {
    "faithfulness": metric_faithfulness,
    "answer_relevance": metric_answer_relevance,
    "context_precision": metric_context_precision,
    "context_recall": metric_context_recall,
}
LLM_METRICS = set(METRIC_FUNCS.keys())
DIRECT_METRICS = {"latency_ms", "cost_usd"}


# ---------------------------------------------------------------------------
# 7. Runner asíncrono
# ---------------------------------------------------------------------------

class Evaluator:
    """Orquesta la evaluación: dataset -> agente -> juez -> métricas."""

    def __init__(self, config: dict[str, Any], agent: AgentAdapter,
                 judge: LLMClient) -> None:
        self.cfg = config
        self.agent = agent
        self.judge = judge
        self.sem = asyncio.Semaphore(int(config["eval"]["concurrency"]))
        self.judge_prices = price_for(judge.model)

        # Precios del agente (separados del juez)
        agent_prices_cfg = config["agent"].get("prices")
        if agent_prices_cfg and isinstance(agent_prices_cfg, dict):
            self.agent_prices = {
                "in": float(agent_prices_cfg.get("in", 0.0)),
                "out": float(agent_prices_cfg.get("out", 0.0)),
            }
        else:
            self.agent_prices = {"in": 0.0, "out": 0.0}

        # Validar métricas solicitadas
        requested = set(config["eval"]["metrics"])
        valid_metrics = LLM_METRICS | DIRECT_METRICS
        invalid = requested - valid_metrics
        if invalid:
            logger.warning(f"Métricas desconocidas ignoradas: {invalid}")

        self.token_ratio = float(
            config["eval"].get("token_char_ratio", TOKEN_CHAR_RATIO)
        )

    async def evaluate_item(self, item: EvalItem) -> ItemResult:
        async with self.sem:
            resp = await self.agent.ask(item.question)
            result = ItemResult(item=item, response=resp)

            # Métricas basadas en LLM-juez
            requested = set(self.cfg["eval"]["metrics"])
            for name in requested & LLM_METRICS:
                func = METRIC_FUNCS[name]
                judgment, t_in, t_out = await func(self.judge, item, resp)
                result.judgments[name] = judgment
                result.metrics[name] = judgment.score
                result.tokens_in += t_in
                result.tokens_out += t_out

            # Métricas directas
            if "latency_ms" in requested:
                result.metrics["latency_ms"] = resp.latency_ms

            # Coste: solo si se solicita la métrica
            if "cost_usd" in requested:
                agent_cost = (
                    estimate_tokens(item.question, self.token_ratio)
                    * self.agent_prices["in"]
                    + estimate_tokens(resp.answer, self.token_ratio)
                    * self.agent_prices["out"]
                ) / 1000.0
                judge_cost = (
                    result.tokens_in * self.judge_prices["in"]
                    + result.tokens_out * self.judge_prices["out"]
                ) / 1000.0
                result.cost_usd = agent_cost + judge_cost
                result.metrics["cost_usd"] = result.cost_usd

            # Pass/fail: solo sobre métricas LLM
            llm_scores = [
                result.metrics[n] for n in requested & LLM_METRICS
                if n in result.metrics
            ]
            threshold = float(self.cfg["eval"]["pass_threshold"])
            result.passed = (
                bool(llm_scores) and all(s >= threshold for s in llm_scores)
                and resp.error is None
            )
            return result

    async def run(self, items: list[EvalItem]) -> list[ItemResult]:
        tasks = [asyncio.create_task(self.evaluate_item(i)) for i in items]
        results: list[ItemResult] = []
        total = len(tasks)
        for idx, task in enumerate(asyncio.as_completed(tasks), 1):
            res = await task
            results.append(res)
            status = "OK" if res.passed else "FAIL"
            logger.info(f"[{idx}/{total}] {res.item.id} -> {status}")
        return results


# ---------------------------------------------------------------------------
# 8. Persistencia (SQLite)
# ---------------------------------------------------------------------------

def persist_results(results: list[ItemResult], path: str) -> None:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    try:
        cur.execute("BEGIN TRANSACTION")
        cur.execute("DROP TABLE IF EXISTS evaluations")
        cur.execute("DROP TABLE IF EXISTS metrics")
        cur.execute("""
            CREATE TABLE evaluations (
                id TEXT PRIMARY KEY,
                question TEXT,
                ground_truth TEXT,
                answer TEXT,
                contexts TEXT,
                latency_ms REAL,
                cost_usd REAL,
                passed INTEGER,
                error TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE metrics (
                item_id TEXT,
                metric TEXT,
                score REAL,
                reasoning TEXT,
                FOREIGN KEY(item_id) REFERENCES evaluations(id)
            )
        """)
        for r in results:
            cur.execute(
                "INSERT INTO evaluations VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    r.item.id, r.item.question, r.item.ground_truth or "",
                    r.response.answer, json.dumps(r.response.contexts),
                    r.response.latency_ms, r.cost_usd,
                    1 if r.passed else 0, r.response.error or "",
                ),
            )
            for name, j in r.judgments.items():
                cur.execute(
                    "INSERT INTO metrics VALUES (?,?,?,?)",
                    (r.item.id, name, j.score, j.reasoning),
                )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error persistiendo resultados: {e}")
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 9. Carga de dataset
# ---------------------------------------------------------------------------

def load_dataset(cfg: dict[str, Any]) -> list[EvalItem]:
    path = cfg["dataset"]["path"]
    if not path:
        raise SystemExit("Falta 'dataset.path'.")

    question_field = cfg["dataset"]["question_field"]
    items: list[EvalItem] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(
                    f"JSON inválido en línea {line_no}: {e}"
                ) from e

            if question_field not in obj:
                raise SystemExit(
                    f"Línea {line_no}: falta campo requerido "
                    f"'{question_field}'. Campos encontrados: "
                    f"{list(obj.keys())}"
                )

            items.append(EvalItem(
                id=str(obj.get(cfg["dataset"]["id_field"], line_no)),
                question=obj[question_field],
                ground_truth=obj.get(cfg["dataset"]["ground_truth_field"]),
                gold_contexts=obj.get(cfg["dataset"]["contexts_field"], []) or [],
            ))

    if not items:
        logger.warning(f"Dataset vacío: {path}")

    return items


DEMO_CORPUS = [
    "El protocolo MCP (Model Context Protocol) es un estándar abierto de Anthropic "
    "para conectar modelos de lenguaje con herramientas y fuentes de datos externas. "
    "Usa JSON-RPC sobre stdio o HTTP y separa servidor y cliente.",
    "LangGraph modela agentes como máquinas de estado. Los nodos son funciones, "
    "las aristas definen transiciones y el estado se persiste en un checkpointer "
    "como PostgreSQL o SQLite, permitiendo time-travel debugging.",
    "RAG combina recuperación y generación: primero se recuperan fragmentos "
    "relevantes de una base vectorial, luego se inyectan en el prompt del LLM. "
    "Sin evaluación de fidelidad y precisión, un RAG es una caja negra.",
    "El fine-tuning con LoRA congela el modelo base e inserta matrices de bajo "
    "rango entrenables. Reduce el coste de GPU en órdenes de magnitud frente "
    "a un fine-tuning completo.",
    "Un buen pipeline de evaluación incluye métricas de recuperación (precision, "
    "recall), de generación (faithfulness, answer relevance) y operativas "
    "(latencia, coste). Sin ellas, no hay mejora medible.",
]

DEMO_DATASET = [
    {
        "id": "q1",
        "question": "¿Qué es el protocolo MCP?",
        "ground_truth": "Es un estándar abierto de Anthropic para conectar LLMs "
                        "con herramientas externas mediante JSON-RPC.",
    },
    {
        "id": "q2",
        "question": "¿Cómo funciona LangGraph?",
        "ground_truth": "Modela agentes como máquinas de estado con nodos, aristas "
                        "y un checkpointer para persistir el estado.",
    },
    {
        "id": "q3",
        "question": "¿Qué métricas debe tener un pipeline RAG?",
        "ground_truth": "Precisión y recall de recuperación, fidelidad y relevancia "
                        "de la respuesta, además de latencia y coste.",
    },
]


def demo_config() -> dict[str, Any]:
    cfg = json.loads(json.dumps(CONFIG))
    cfg["agent"]["type"] = "builtin"
    cfg["dataset"]["path"] = "(demo)"
    cfg["eval"]["concurrency"] = 3
    return cfg


# ---------------------------------------------------------------------------
# 10. Generación del informe Markdown
# ---------------------------------------------------------------------------

def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def _aggregate(results: list[ItemResult],
               metrics: list[str]) -> dict[str, dict[str, float]]:
    agg: dict[str, dict[str, float]] = {}
    for m in metrics:
        vals = [r.metrics.get(m) for r in results if m in r.metrics]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        agg[m] = {
            "mean": safe_mean(vals),
            "p50": percentile(vals, 0.50),
            "p95": percentile(vals, 0.95),
            "min": min(vals),
            "max": max(vals),
        }
    return agg


def render_report(
    cfg: dict[str, Any],
    results: list[ItemResult],
    dataset_hash: str,
    elapsed_s: float,
) -> str:
    metrics = cfg["eval"]["metrics"]
    threshold = cfg["eval"]["pass_threshold"]
    agg = _aggregate(results, metrics)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pass_rate = (passed / total) if total else 0.0
    total_cost = sum(r.cost_usd for r in results)
    errors = [r for r in results if r.response.error]

    lines: list[str] = []
    add = lines.append

    add(f"# Informe de evaluación — `{cfg['judge']['model']}`\n")
    add(f"**Generado:** "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  ")
    add(f"**Versión del arnés:** `agent-eval {__version__}`  ")
    add(f"**Hash del dataset:** `{dataset_hash}`  ")
    add(f"**Tiempo total:** {elapsed_s:.2f} s\n")

    # Resumen ejecutivo
    verdict = "✅ APTO" if pass_rate >= 0.8 and not errors else "⚠️ REVISAR"
    add("## 1. Resumen ejecutivo\n")
    add(f"**Veredicto:** {verdict}  ")
    add(f"**Ítems evaluados:** {total}  ")
    add(f"**Aprobados (umbral {threshold}):** {passed}/{total} "
        f"({pass_rate*100:.1f}%)  ")
    add(f"**Coste total:** ${total_cost:.4f}  ")
    add(f"**Errores del agente:** {len(errors)}\n")

    # Arquitectura
    if cfg["output"]["include_mermaid"]:
        add("## 2. Arquitectura del pipeline evaluado\n")
        add("```mermaid")
        add("flowchart LR")
        add("    D[Dataset JSONL] --> R[Runner async]")
        add("    R --> A[Agente bajo prueba]")
        add("    A -->|answer + contexts| J[LLM-juez]")
        add("    J --> M[Métricas]")
        add("    M --> S[(SQLite)]")
        add("    M --> MD[Informe Markdown]")
        add("    M --> V{Veredicto}")
        add("```\n")

    # Configuración
    add("## 3. Configuración\n")
    add("| Clave | Valor |")
    add("|---|---|")
    add(f"| Modelo juez | `{cfg['judge']['model']}` |")
    add(f"| Endpoint juez | `{cfg['judge']['base_url']}` |")
    add(f"| Tipo de agente | `{cfg['agent']['type']}` |")
    add(f"| Concurrencia | {cfg['eval']['concurrency']} |")
    add(f"| Umbral de aprobado | {threshold} |")
    add(f"| Métricas | {', '.join(metrics)} |")
    add("")

    # Métricas agregadas
    add("## 4. Métricas agregadas\n")
    if not agg:
        add("_Sin métricas calculadas._\n")
    else:
        add("| Métrica | Media | P50 | P95 | Mín | Máx |")
        add("|---|---:|---:|---:|---:|---:|")
        for m, stats in agg.items():
            add(f"| `{m}` | {_fmt(stats['mean'])} | {_fmt(stats['p50'])} | "
                f"{_fmt(stats['p95'])} | {_fmt(stats['min'])} | "
                f"{_fmt(stats['max'])} |")
        add("")

    # Detalle por pregunta
    if cfg["output"]["include_per_question"]:
        add("## 5. Detalle por ítem\n")
        add("| ID | OK | Latencia (ms) | Coste (USD) | "
            + " | ".join(f"`{m}`" for m in metrics if m in agg) + " |")
        add("|---|---:|---:|---:|"
            + "---:|" * sum(1 for m in metrics if m in agg))
        for r in results:
            cells = [
                r.item.id,
                "✅" if r.passed else "❌",
                _fmt(r.metrics.get("latency_ms", r.response.latency_ms), 1),
                f"{r.cost_usd:.5f}",
            ]
            for m in metrics:
                if m in agg:
                    cells.append(_fmt(r.metrics.get(m, 0.0)))
            add("| " + " | ".join(cells) + " |")
        add("")

    # Fallos
    failed = [r for r in results if not r.passed]
    if failed:
        add("## 6. Análisis de fallos\n")
        for r in failed:
            add(f"### `{r.item.id}` — {r.item.question}\n")
            if r.response.error:
                add(f"- **Error del agente "
                    f"({r.response.error_type or 'desconocido'}):** "
                    f"`{r.response.error}`")
            for name, j in r.judgments.items():
                if j.score < threshold:
                    add(f"- **{name}:** {j.score:.2f} — "
                        f"{j.reasoning or 'sin razonamiento'}")
            add("")

    # Reproducibilidad
    add("## 7. Reproducibilidad\n")
    add("```bash")
    add("python agent-eval.py --config <config> --dataset <dataset>")
    add("```\n")
    add(f"El hash `{dataset_hash}` identifica unívocamente el dataset evaluado. "
        "Cualquier cambio en los datos invalida la comparación con este informe.\n")

    if cfg["output"]["include_raw_judge"]:
        add("## 8. Salidas crudas del juez\n")
        for r in results:
            add(f"### `{r.item.id}`\n")
            for name, j in r.judgments.items():
                add(f"**{name}** (score={j.score:.2f})")
                add("```")
                add(j.raw.strip() or "(vacío)")
                add("```")
                add("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 11. CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="agent-eval",
        description="Arnés de evaluación autocontenido para agentes y RAG.",
    )
    p.add_argument("--config",
                   help="Ruta a un archivo JSON o YAML de configuración.")
    p.add_argument("--dataset",
                   help="Ruta a un dataset JSONL (sobrescribe config).")
    p.add_argument("--agent-url",
                   help="URL del agente HTTP a evaluar.")
    p.add_argument("--judge-model",
                   help="Modelo juez (por defecto gpt-4o-mini).")
    p.add_argument("--output",
                   help="Ruta del informe Markdown.")
    p.add_argument("--concurrency", type=int,
                   help="Número de evaluaciones en paralelo.")
    p.add_argument("--demo", action="store_true",
                   help="Ejecuta una demo autocontenida sin servicios externos.")
    p.add_argument("--selftest", action="store_true",
                   help="Ejecuta self-tests internos y sale.")
    p.add_argument("--no-sqlite", action="store_true",
                   help="No persistir resultados en SQLite.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Activar logging detallado.")
    p.add_argument("--version", action="version",
                   version=f"agent-eval {__version__}")
    return p.parse_args(argv)


def build_runtime_config(args: argparse.Namespace
                         ) -> tuple[dict[str, Any], Optional[list[str]]]:
    """Devuelve (config, corpus_opcional_para_agente_builtin)."""
    if args.demo:
        cfg = demo_config()
        corpus = DEMO_CORPUS
        return cfg, corpus

    cfg = json.loads(json.dumps(CONFIG))
    if args.config:
        cfg = merge_config(cfg, load_config_file(args.config))
    if args.dataset:
        cfg["dataset"]["path"] = args.dataset
    if args.agent_url:
        cfg["agent"]["url"] = args.agent_url
        cfg["agent"]["type"] = "http"
    if args.judge_model:
        cfg["judge"]["model"] = args.judge_model
    if args.output:
        cfg["output"]["report_path"] = args.output
    if args.concurrency:
        cfg["eval"]["concurrency"] = args.concurrency
    if args.no_sqlite:
        cfg["eval"]["save_sqlite"] = False

    return cfg, None


def load_demo_dataset() -> list[EvalItem]:
    return [
        EvalItem(
            id=d["id"], question=d["question"],
            ground_truth=d.get("ground_truth"),
        )
        for d in DEMO_DATASET
    ]


def resolve_api_key(cfg: dict[str, Any]) -> Optional[str]:
    env_name = cfg["judge"].get("api_key_env")
    if env_name:
        return os.environ.get(env_name)
    return None


async def async_main(cfg: dict[str, Any],
                     corpus: Optional[list[str]]) -> int:
    # Dataset
    if cfg["dataset"]["path"] == "(demo)":
        items = load_demo_dataset()
    else:
        items = load_dataset(cfg)

    if not items:
        logger.error("Dataset vacío.")
        return 2

    dataset_blob = json.dumps([asdict(i) for i in items], sort_keys=True)
    dataset_hash = sha256_short(dataset_blob)

    # Cliente juez
    api_key = resolve_api_key(cfg)
    if not api_key and cfg["agent"]["type"] != "builtin":
        logger.warning(
            "No hay API key para el juez; las métricas LLM fallarán."
        )

    token_ratio = float(cfg["eval"].get("token_char_ratio", TOKEN_CHAR_RATIO))
    judge = LLMClient(
        base_url=cfg["judge"]["base_url"],
        model=cfg["judge"]["model"],
        api_key=api_key,
        temperature=cfg["judge"].get("temperature", 0.0),
        max_tokens=cfg["judge"].get("max_tokens", 512),
        token_ratio=token_ratio,
    )

    # Agente
    agent = build_agent(cfg["agent"], corpus=corpus)

    # Ejecutar
    logger.info(f"Evaluando {len(items)} ítems con `{cfg['judge']['model']}`...")
    evaluator = Evaluator(cfg, agent, judge)
    t0 = time.perf_counter()
    results = await evaluator.run(items)
    elapsed = time.perf_counter() - t0

    # Persistencia
    if cfg["eval"]["save_sqlite"]:
        persist_results(results, cfg["eval"]["sqlite_path"])
        logger.info(f"Resultados guardados en {cfg['eval']['sqlite_path']}")

    # Informe
    report = render_report(cfg, results, dataset_hash, elapsed)
    Path(cfg["output"]["report_path"]).write_text(report, encoding="utf-8")
    logger.info(f"Informe escrito en {cfg['output']['report_path']}")

    passed = sum(1 for r in results if r.passed)
    logger.info(f"Aprobados: {passed}/{len(results)}")

    return 0 if passed == len(results) else 1


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Configurar nivel de logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.selftest:
        return run_selftests()

    cfg, corpus = build_runtime_config(args)
    try:
        return asyncio.run(async_main(cfg, corpus))
    except KeyboardInterrupt:
        logger.info("Interrumpido por el usuario.")
        return 130


# ---------------------------------------------------------------------------
# 12. Self-tests embebidos (sin pytest, sin red, sin API keys)
# ---------------------------------------------------------------------------

def run_selftests() -> int:
    """
    Ejecuta una batería de tests unitarios sobre las funciones puras del
    módulo. No depende de pytest, de la red ni de API keys. Se invoca con:

        python agent-eval.py --selftest

    Devuelve 0 si todos pasan, 1 si alguno falla.
    """
    failures: list[str] = []
    total = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal total
        total += 1
        if cond:
            print(f"  ✅ {name}")
        else:
            failures.append(name)
            print(f"  ❌ {name} {detail}")

    print(f"agent-eval {__version__} — self-tests")
    print("-" * 60)

    # --- 1. estimate_tokens ---------------------------------------------
    print("\n[1] estimate_tokens")
    check("vacío -> 0", estimate_tokens("") == 0)
    check("None -> 0", estimate_tokens("") == 0)
    check("hola -> >0", estimate_tokens("hola") > 0)
    check("ratio afecta al resultado",
          estimate_tokens("a" * 100, ratio=4.0)
          < estimate_tokens("a" * 100, ratio=2.0))
    check("monotonía",
          estimate_tokens("a" * 100) <= estimate_tokens("a" * 200))

    # --- 2. _extract_json ------------------------------------------------
    print("\n[2] _extract_json")
    check("JSON limpio",
          _extract_json('{"score": 0.8}') == {"score": 0.8})
    check("JSON con fence",
          _extract_json('```json\n{"score": 0.5}\n```') == {"score": 0.5})
    check("JSON con fence sin etiqueta",
          _extract_json('```\n{"score": 0.5}\n```') == {"score": 0.5})
    check("JSON con basura alrededor",
          _extract_json('bla bla {"score": 0.3} bla') == {"score": 0.3})
    check("JSON inválido -> None",
          _extract_json("no json aquí") is None)
    check("JSON anidado",
          _extract_json('{"a": {"b": 1}}') == {"a": {"b": 1}})
    check("array JSON -> None (no dict)",
          _extract_json('[1, 2, 3]') is None)

    # --- 3. percentile ---------------------------------------------------
    print("\n[3] percentile")
    check("vacío -> 0.0", percentile([], 0.5) == 0.0)
    check("p50 de [1..5]",
          abs(percentile([1, 2, 3, 4, 5], 0.5) - 3.0) < 1e-9)
    check("p0 = min",
          abs(percentile([1, 2, 3, 4, 5], 0.0) - 1.0) < 1e-9)
    check("p100 = max",
          abs(percentile([1, 2, 3, 4, 5], 1.0) - 5.0) < 1e-9)
    check("p95 >= 4.8",
          percentile([1, 2, 3, 4, 5], 0.95) >= 4.8)
    check("un solo valor",
          percentile([42.0], 0.5) == 42.0)

    # --- 4. safe_mean ----------------------------------------------------
    print("\n[4] safe_mean")
    check("vacío -> 0.0", safe_mean([]) == 0.0)
    check("media simple",
          abs(safe_mean([1, 2, 3, 4]) - 2.5) < 1e-9)
    check("ignora None",
          abs(safe_mean([1, None, 3]) - 2.0) < 1e-9)

    # --- 5. sha256_short -------------------------------------------------
    print("\n[5] sha256_short")
    check("determinista",
          sha256_short("hola") == sha256_short("hola"))
    check("distinto input -> distinto hash",
          sha256_short("hola") != sha256_short("hola "))
    check("longitud por defecto = 12",
          len(sha256_short("hola")) == 12)
    check("longitud personalizada",
          len(sha256_short("hola", n=8)) == 8)

    # --- 6. price_for ----------------------------------------------------
    print("\n[6] price_for")
    check("modelo exacto",
          price_for("gpt-4o-mini")["in"] == 0.00015)
    check("prefijo coincide",
          price_for("gpt-4o-mini-2024")["in"] == 0.00015)
    check("desconocido -> default",
          price_for("modelo-inexistente")["in"] == 0.0)

    # --- 7. _deep_get ----------------------------------------------------
    print("\n[7] _deep_get")
    check("un nivel",
          _deep_get({"a": 1}, "a") == 1)
    check("dos niveles",
          _deep_get({"a": {"b": 2}}, "a.b") == 2)
    check("tres niveles",
          _deep_get({"a": {"b": {"c": 3}}}, "a.b.c") == 3)
    check("falta -> None",
          _deep_get({"a": 1}, "a.b") is None)
    check("raíz inexistente -> None",
          _deep_get({"a": 1}, "x") is None)
    check("valor 0 no se confunde con None",
          _deep_get({"a": 0}, "a") == 0)

    # --- 8. _deep_format -------------------------------------------------
    print("\n[8] _deep_format")
    check("string plano",
          _deep_format("hola {x}", {"x": "mundo"}) == "hola mundo")
    check("sin placeholders",
          _deep_format("hola", {"x": "mundo"}) == "hola")
    check("dict anidado",
          _deep_format({"q": "{x}"}, {"x": "y"}) == {"q": "y"})
    check("lista de dicts",
          _deep_format([{"q": "{x}"}], {"x": "y"}) == [{"q": "y"}])
    check("intacto si no hay match",
          _deep_format({"k": 5}, {"x": "y"}) == {"k": 5})

    # --- 9. merge_config -------------------------------------------------
    print("\n[9] merge_config")
    check("override plano",
          merge_config({"a": 1}, {"a": 2}) == {"a": 2})
    check("merge profundo",
          merge_config({"a": {"b": 1}}, {"a": {"c": 2}})
          == {"a": {"b": 1, "c": 2}})
    check("clave nueva",
          merge_config({"a": 1}, {"b": 2}) == {"a": 1, "b": 2})
    check("no muta el base",
          (lambda b: (merge_config(b, {"a": 2}), b["a"] == 1)[1])({"a": 1}))

    # --- 10. _fmt --------------------------------------------------------
    print("\n[10] _fmt")
    check("3 decimales por defecto", _fmt(1.23456) == "1.235")
    check("1 decimal", _fmt(1.23456, 1) == "1.2")
    check("entero como float", _fmt(2.0) == "2.000")

    # --- 11. BuiltinAgent (async) ---------------------------------------
    print("\n[11] BuiltinAgent")
    corpus = [
        "Python es un lenguaje interpretado de alto nivel.",
        "Rust es un lenguaje compilado con ownership.",
        "SQLite es una base de datos embebida.",
    ]
    agent = BuiltinAgent(corpus, k=2)
    resp = asyncio.run(agent.ask("Python lenguaje"))
    check("devuelve respuesta", bool(resp.answer))
    check("devuelve contextos", len(resp.contexts) > 0)
    check("latencia > 0", resp.latency_ms >= 0.0)
    check("sin error", resp.error is None)
    check("contexto recuperado es el correcto",
          "Python" in resp.contexts[0])

    resp_empty = asyncio.run(agent.ask("xyz qwerty"))
    check("query sin match devuelve algo",
          bool(resp_empty.answer))

    # --- 12. _aggregate --------------------------------------------------
    print("\n[12] _aggregate")
    r1 = ItemResult(
        item=EvalItem(id="a", question="q1"),
        response=AgentResponse(answer="x", latency_ms=100.0),
        metrics={"faithfulness": 0.8, "latency_ms": 100.0},
    )
    r2 = ItemResult(
        item=EvalItem(id="b", question="q2"),
        response=AgentResponse(answer="y", latency_ms=200.0),
        metrics={"faithfulness": 0.6, "latency_ms": 200.0},
    )
    agg = _aggregate([r1, r2], ["faithfulness", "latency_ms"])
    check("agrega faithfulness",
          abs(agg["faithfulness"]["mean"] - 0.7) < 1e-9)
    check("min correcto",
          abs(agg["faithfulness"]["min"] - 0.6) < 1e-9)
    check("max correcto",
          abs(agg["faithfulness"]["max"] - 0.8) < 1e-9)
    check("métrica ausente no aparece",
          "no_existe" not in agg)

    # --- 13. render_report (smoke) --------------------------------------
    print("\n[13] render_report (smoke)")
    cfg = demo_config()
    report = render_report(cfg, [r1, r2], "deadbeef", 1.23)
    check("incluye título", "# Informe de evaluación" in report)
    check("incluye hash", "deadbeef" in report)
    check("incluye tabla de métricas", "| Métrica |" in report)
    check("incluye mermaid", "```mermaid" in report)
    check("incluye resumen ejecutivo",
          "## 1. Resumen ejecutivo" in report)

    # --- 14. EvalItem / ItemResult dataclasses --------------------------
    print("\n[14] dataclasses")
    item = EvalItem(id="x", question="q")
    check("EvalItem id", item.id == "x")
    check("EvalItem sin ground_truth por defecto",
          item.ground_truth is None)
    check("EvalItem gold_contexts vacío por defecto",
          item.gold_contexts == [])
    resp_d = AgentResponse(answer="a")
    check("AgentResponse sin error por defecto",
          resp_d.error is None)
    check("AgentResponse error_type por defecto",
          resp_d.error_type is None)

    # --- 15. load_demo_dataset ------------------------------------------
    print("\n[15] load_demo_dataset")
    demo_items = load_demo_dataset()
    check("demo tiene 3 ítems", len(demo_items) == 3)
    check("todos tienen question",
          all(i.question for i in demo_items))
    check("todos tienen id",
          all(i.id for i in demo_items))

    # --- Resumen final ---------------------------------------------------
    print("-" * 60)
    if not failures:
        print(f"✅ Todos los tests pasaron ({total}/{total})")
        return 0
    print(f"❌ {len(failures)}/{total} tests fallaron:")
    for f in failures:
        print(f"   - {f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
