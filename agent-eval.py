````python
#!/usr/bin/env python3
"""
Agent-Eval: lightweight evaluation harness for agents/RAG systems.

Features:
- JSONL datasets
- HTTP and built-in agents
- LLM-as-a-judge metrics
- Concurrent evaluation
- SQLite persistence with historical runs
- Markdown reports
- Reproducibility hashes
- Token-based cost calculation
- Built-in demo and self-tests

Stdlib-first: no mandatory third-party dependencies.

Version: 0.2.2
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sqlite3
import statistics
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


__version__ = "0.2.2"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG: dict[str, Any] = {
    "judge": {
        "provider": "openai-compatible",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "timeout_s": 60,
        "retries": 2,
        "temperature": 0.0,
        "max_tokens": 800,
        "prices": {
            "input_per_1m": 0.0,
            "output_per_1m": 0.0,
        },
    },
    "agent": {
        "type": "http",
        "url": "http://localhost:8000/chat",
        "timeout_s": 60,
        "headers": {},
        "prices": {
            "input_per_1m": 0.0,
            "output_per_1m": 0.0,
        },
        "request_template": {
            "question": "{question}",
        },
        "answer_path": "answer",
        "contexts_path": "contexts",
    },
    "dataset": {
        "path": "dataset.jsonl",
        "encoding": "utf-8",
    },
    "eval": {
        "concurrency": 4,
        "metrics": [
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "context_recall",
        ],
        "pass_threshold": 0.7,
    },
    "output": {
        "sqlite": "eval-results.db",
        "report": "eval-report.md",
        "include_raw_judge": False,
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvalItem:
    id: str
    question: str
    ground_truth: str | None = None
    contexts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentResponse:
    answer: str
    contexts: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    raw: Any = None


@dataclass(slots=True)
class Judgment:
    metric: str
    score: float
    reasoning: str = ""
    raw: Any = None
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass(slots=True)
class ItemResult:
    item: EvalItem
    response: AgentResponse | None
    judgments: list[Judgment] = field(default_factory=list)
    passed: bool = False
    error: str | None = None
    error_type: str | None = None
    cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token estimate suitable for cost estimation only."""
    if not text:
        return 0

    return max(1, round(len(text) / 4))


def price_for(
    tokens: int,
    usd_per_million: float,
) -> float:
    """Convert a token count into USD at a per-million-token price."""
    return (tokens / 1_000_000) * usd_per_million


def token_cost(
    tokens_in: int,
    tokens_out: int,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> float:
    """
    Calculate cost using separate input/output token prices.

    Prices are expected to be USD per 1M tokens.
    """
    return (
        price_for(tokens_in, input_price_per_1m)
        + price_for(tokens_out, output_price_per_1m)
    )


def sha256_short(
    value: str,
    length: int = 12,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:length]


def canonical_hash(
    value: Any,
    length: int = 12,
) -> str:
    """Stable hash for JSON-compatible configuration/data."""
    blob = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    return sha256_short(
        blob,
        length,
    )


def percentile(
    values: Iterable[float],
    p: float,
) -> float:
    values = sorted(values)

    if not values:
        return 0.0

    if p <= 0:
        return values[0]

    if p >= 100:
        return values[-1]

    rank = (len(values) - 1) * (p / 100)
    lower = int(rank)
    upper = min(lower + 1, len(values))
    weight = rank - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def safe_mean(
    values: Iterable[float],
) -> float:
    values = list(values)

    return (
        statistics.mean(values)
        if values
        else 0.0
    )


def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    result = dict(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(
                result[key],
                value,
            )
        else:
            result[key] = value

    return result


def _deep_format(
    value: Any,
    mapping: dict[str, Any],
) -> Any:
    if isinstance(value, str):
        try:
            return value.format(**mapping)
        except (KeyError, ValueError):
            return value

    if isinstance(value, list):
        return [
            _deep_format(item, mapping)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: _deep_format(item, mapping)
            for key, item in value.items()
        }

    return value


def _get_path(
    value: Any,
    path: str | None,
) -> Any:
    if not path:
        return value

    current = value

    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)

        elif (
            isinstance(current, list)
            and part.isdigit()
        ):
            index = int(part)

            if index >= len(current):
                return None

            current = current[index]

        else:
            return None

    return current


def _extract_json(text: str) -> Any:
    """
    Extract the first valid JSON object/array from a model response.

    Uses JSONDecoder.raw_decode instead of a greedy regex so multiple JSON
    fragments do not get incorrectly merged.
    """
    decoder = json.JSONDecoder()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for index, char in enumerate(text):
        if char not in "{[":
            continue

        try:
            value, _ = decoder.raw_decode(
                text[index:]
            )
            return value
        except json.JSONDecodeError:
            continue

    raise ValueError(
        "No valid JSON found in judge response"
    )


# ---------------------------------------------------------------------------
# Config / dataset
# ---------------------------------------------------------------------------


def load_config_file(
    path: str | Path,
) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    text = path.read_text(
        encoding="utf-8"
    )

    if path.suffix.lower() == ".json":
        value = json.loads(text)

        if not isinstance(value, dict):
            raise ValueError(
                "Config JSON must contain an object"
            )

        return value

    if path.suffix.lower() in {
        ".yaml",
        ".yml",
    }:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requires PyYAML. "
                "Use JSON or install PyYAML."
            ) from exc

        value = yaml.safe_load(text) or {}

        if not isinstance(value, dict):
            raise ValueError(
                "Config YAML must contain a mapping"
            )

        return value

    raise ValueError(
        f"Unsupported config format: {path.suffix}. "
        "Use .json, .yaml or .yml."
    )


def load_dataset(
    path: str | Path,
    encoding: str = "utf-8",
) -> list[EvalItem]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    items: list[EvalItem] = []

    with path.open(
        "r",
        encoding=encoding,
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Dataset line {line_number} "
                    "must be a JSON object"
                )

            item_id = str(
                row.get("id") or line_number
            )

            question = str(
                row.get("question", "")
            ).strip()

            if not question:
                raise ValueError(
                    f"Dataset line {line_number} "
                    "has an empty question"
                )

            contexts = row.get("contexts") or []

            if not isinstance(contexts, list):
                raise ValueError(
                    f"Dataset line {line_number}: "
                    "contexts must be a list"
                )

            items.append(
                EvalItem(
                    id=item_id,
                    question=question,
                    ground_truth=(
                        None
                        if row.get("ground_truth") is None
                        else str(row["ground_truth"])
                    ),
                    contexts=[
                        str(context)
                        for context in contexts
                    ],
                    metadata=(
                        row.get("metadata") or {}
                    ),
                )
            )

    if not items:
        raise ValueError(
            f"Dataset is empty: {path}"
        )

    return items


def dataset_hash(
    items: list[EvalItem],
) -> str:
    return canonical_hash(
        [
            asdict(item)
            for item in items
        ]
    )


def resolve_api_key(
    cfg: dict[str, Any],
) -> str | None:
    env_name = cfg.get("api_key_env")

    if not env_name:
        return None

    return os.environ.get(
        str(env_name)
    )


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


class LLMClient:
    def __init__(
        self,
        cfg: dict[str, Any],
    ):
        self.cfg = cfg
        self.base_url = str(
            cfg["base_url"]
        ).rstrip("/")
        self.model = str(
            cfg["model"]
        )
        self.timeout_s = float(
            cfg.get("timeout_s", 60)
        )
        self.retries = int(
            cfg.get("retries", 2)
        )
        self.temperature = float(
            cfg.get("temperature", 0.0)
        )
        self.max_tokens = int(
            cfg.get("max_tokens", 800)
        )
        self.api_key = resolve_api_key(
            cfg
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, dict[str, int]]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        body = json.dumps(
            payload
        ).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )

        last_error: Exception | None = None

        for attempt in range(
            self.retries + 1
        ):
            try:
                response_body = (
                    await asyncio.to_thread(
                        self._request,
                        request,
                    )
                )

                data = json.loads(
                    response_body
                )

                choices = (
                    data.get("choices")
                    or []
                )

                if not choices:
                    raise RuntimeError(
                        "LLM response contains "
                        "no choices"
                    )

                message = (
                    choices[0].get("message")
                    or {}
                )

                content = message.get(
                    "content"
                )

                if content is None:
                    raise RuntimeError(
                        "LLM response contains "
                        "no message content"
                    )

                usage = (
                    data.get("usage")
                    or {}
                )

                prompt_tokens = usage.get(
                    "prompt_tokens"
                )

                completion_tokens = usage.get(
                    "completion_tokens"
                )

                # Prefer provider-reported usage.
                # Only estimate when usage is unavailable.
                if prompt_tokens is None:
                    prompt_tokens = estimate_tokens(
                        json.dumps(
                            messages,
                            ensure_ascii=False,
                        )
                    )

                if completion_tokens is None:
                    completion_tokens = estimate_tokens(
                        str(content)
                    )

                return str(content), {
                    "prompt_tokens": int(
                        prompt_tokens
                    ),
                    "completion_tokens": int(
                        completion_tokens
                    ),
                }

            except Exception as exc:
                last_error = exc

                if attempt >= self.retries:
                    break

                await asyncio.sleep(
                    0.5 * (2**attempt)
                )

        assert last_error is not None
        raise last_error

    def _request(
        self,
        request: Request,
    ) -> str:
        try:
            with urlopen(
                request,
                timeout=self.timeout_s,
            ) as response:
                return response.read().decode(
                    "utf-8"
                )

        except HTTPError as exc:
            body = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(
                f"LLM HTTP {exc.code}: "
                f"{body[:500]}"
            ) from exc

        except URLError as exc:
            raise RuntimeError(
                f"LLM connection error: "
                f"{exc.reason}"
            ) from exc


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class HTTPAgent:
    def __init__(
        self,
        cfg: dict[str, Any],
    ):
        self.cfg = cfg
        self.url = str(
            cfg["url"]
        )
        self.timeout_s = float(
            cfg.get("timeout_s", 60)
        )
        self.headers = dict(
            cfg.get("headers") or {}
        )

        self.request_template = (
            cfg.get("request_template")
            or {
                "question": "{question}"
            }
        )

        self.answer_path = cfg.get(
            "answer_path",
            "answer",
        )

        self.contexts_path = cfg.get(
            "contexts_path",
            "contexts",
        )

        prices = cfg.get("prices") or {}

        self.input_price_per_1m = float(
            prices.get(
                "input_per_1m",
                0.0,
            )
        )

        self.output_price_per_1m = float(
            prices.get(
                "output_per_1m",
                0.0,
            )
        )

    async def run(
        self,
        item: EvalItem,
    ) -> AgentResponse:
        payload = _deep_format(
            self.request_template,
            {
                "question": item.question,
                "id": item.id,
                "contexts": item.contexts,
                "ground_truth": (
                    item.ground_truth or ""
                ),
            },
        )

        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            **self.headers,
        }

        request = Request(
            self.url,
            data=body,
            headers=headers,
            method="POST",
        )

        started = time.perf_counter()

        try:
            raw_body = await asyncio.to_thread(
                self._request,
                request,
            )

            latency_ms = (
                time.perf_counter()
                - started
            ) * 1000

            data = json.loads(
                raw_body
            )

            answer = _get_path(
                data,
                self.answer_path,
            )

            contexts = _get_path(
                data,
                self.contexts_path,
            )

            if answer is None:
                raise ValueError(
                    "Agent response missing "
                    f"answer at path "
                    f"'{self.answer_path}'"
                )

            if contexts is None:
                contexts = []

            if not isinstance(
                contexts,
                list,
            ):
                raise ValueError(
                    "Agent contexts at "
                    f"'{self.contexts_path}' "
                    "must be a list"
                )

            answer_text = str(answer)

            context_texts = [
                str(context)
                for context in contexts
            ]

            # This is an estimate because a generic HTTP agent does not
            # necessarily expose its actual provider token usage.
            tokens_in = estimate_tokens(
                item.question
                + "\n"
                + "\n".join(
                    item.contexts
                )
            )

            tokens_out = estimate_tokens(
                answer_text
            )

            return AgentResponse(
                answer=answer_text,
                contexts=context_texts,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                raw=data,
            )

        except HTTPError as exc:
            body_text = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(
                f"Agent HTTP {exc.code}: "
                f"{body_text[:500]}"
            ) from exc

    def _request(
        self,
        request: Request,
    ) -> str:
        try:
            with urlopen(
                request,
                timeout=self.timeout_s,
            ) as response:
                return response.read().decode(
                    "utf-8"
                )

        except HTTPError:
            raise

        except URLError as exc:
            raise RuntimeError(
                f"Agent connection error: "
                f"{exc.reason}"
            ) from exc


class BuiltinAgent:
    """
    Minimal deterministic agent useful for demos and self-tests.
    """

    async def run(
        self,
        item: EvalItem,
    ) -> AgentResponse:
        started = time.perf_counter()

        words = {
            word.lower()
            for word in re.findall(
                r"\w+",
                item.question,
            )
            if len(word) > 3
        }

        selected = None

        for context in item.contexts:
            context_words = {
                word.lower()
                for word in re.findall(
                    r"\w+",
                    context,
                )
            }

            if words & context_words:
                selected = context
                break

        if (
            selected is None
            and item.contexts
        ):
            selected = item.contexts[0]

        answer = (
            selected
            or "No relevant context found."
        )

        return AgentResponse(
            answer=answer,
            contexts=item.contexts,
            latency_ms=(
                time.perf_counter()
                - started
            ) * 1000,
            tokens_in=estimate_tokens(
                item.question
            ),
            tokens_out=estimate_tokens(
                answer
            ),
        )


# ---------------------------------------------------------------------------
# Judge metrics
# ---------------------------------------------------------------------------


JUDGE_PROMPTS: dict[str, str] = {
    "faithfulness": """
Evaluate whether the answer is fully supported by the supplied contexts.

Question:
{question}

Answer:
{answer}

Contexts:
{contexts}

Return JSON only:
{
  "score": 0.0,
  "reasoning": "brief explanation"
}

Score from 0.0 to 1.0.
""",
    "answer_relevance": """
Evaluate how directly and completely the answer addresses the question.

Question:
{question}

Answer:
{answer}

Return JSON only:
{
  "score": 0.0,
  "reasoning": "brief explanation"
}

Score from 0.0 to 1.0.
""",
    "context_precision": """
Evaluate whether the retrieved contexts are relevant to answering the question.

Question:
{question}

Contexts:
{contexts}

Return JSON only:
{
  "score": 0.0,
  "reasoning": "brief explanation"
}

Score from 0.0 to 1.0.
""",
    "context_recall": """
Evaluate whether the supplied contexts contain the information needed to
support the reference answer.

Question:
{question}

Reference answer:
{ground_truth}

Contexts:
{contexts}

Return JSON only:
{
  "score": 0.0,
  "reasoning": "brief explanation"
}

Score from 0.0 to 1.0.
""",
}


class Judge:
    def __init__(
        self,
        client: LLMClient,
    ):
        self.client = client

    async def evaluate(
        self,
        metric: str,
        item: EvalItem,
        response: AgentResponse,
    ) -> Judgment:
        if metric not in JUDGE_PROMPTS:
            raise ValueError(
                f"Unknown metric: {metric}"
            )

        if (
            metric == "context_recall"
            and not item.ground_truth
        ):
            return Judgment(
                metric=metric,
                score=0.0,
                reasoning=(
                    "Metric unavailable: "
                    "item has no ground truth."
                ),
            )

        prompt = JUDGE_PROMPTS[
            metric
        ].format(
            question=item.question,
            answer=response.answer,
            contexts="\n\n".join(
                response.contexts
            ),
            ground_truth=(
                item.ground_truth or ""
            ),
        )

        content, usage = (
            await self.client.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict "
                            "evaluation judge. "
                            "Return valid JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
            )
        )

        parsed = _extract_json(
            content
        )

        if not isinstance(
            parsed,
            dict,
        ):
            raise ValueError(
                "Judge response must be "
                "a JSON object"
            )

        score = float(
            parsed.get("score", 0.0)
        )

        score = max(
            0.0,
            min(1.0, score),
        )

        return Judgment(
            metric=metric,
            score=score,
            reasoning=str(
                parsed.get(
                    "reasoning",
                    "",
                )
            ),
            raw=parsed,
            tokens_in=usage[
                "prompt_tokens"
            ],
            tokens_out=usage[
                "completion_tokens"
            ],
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class Evaluator:
    def __init__(
        self,
        agent: Any,
        judge: Judge | None,
        cfg: dict[str, Any],
    ):
        self.agent = agent
        self.judge = judge
        self.cfg = cfg

        eval_cfg = (
            cfg.get("eval") or {}
        )

        self.metrics = list(
            eval_cfg.get(
                "metrics",
                [],
            )
        )

        self.pass_threshold = float(
            eval_cfg.get(
                "pass_threshold",
                0.7,
            )
        )

        self.semaphore = asyncio.Semaphore(
            max(
                1,
                int(
                    eval_cfg.get(
                        "concurrency",
                        4,
                    )
                ),
            )
        )

        agent_cfg = (
            cfg.get("agent") or {}
        )

        agent_prices = (
            agent_cfg.get("prices")
            or {}
        )

        self.agent_input_price_per_1m = (
            float(
                agent_prices.get(
                    "input_per_1m",
                    0.0,
                )
            )
        )

        self.agent_output_price_per_1m = (
            float(
                agent_prices.get(
                    "output_per_1m",
                    0.0,
                )
            )
        )

        judge_cfg = (
            cfg.get("judge") or {}
        )

        judge_prices = (
            judge_cfg.get("prices")
            or {}
        )

        self.judge_input_price_per_1m = (
            float(
                judge_prices.get(
                    "input_per_1m",
                    0.0,
                )
            )
        )

        self.judge_output_price_per_1m = (
            float(
                judge_prices.get(
                    "output_per_1m",
                    0.0,
                )
            )
        )

    def _calculate_cost(
        self,
        response: AgentResponse | None,
        judgments: list[Judgment],
    ) -> float:
        """
        Calculate the complete cost for one evaluated item.

        Total =
            agent input cost
          + agent output cost
          + judge input cost
          + judge output cost

        Agent token counts are estimated unless the agent itself provides
        actual usage. Judge token counts come from the LLM provider whenever
        available, with estimation as a fallback.
        """
        if response is None:
            return 0.0

        agent_cost = token_cost(
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            input_price_per_1m=(
                self.agent_input_price_per_1m
            ),
            output_price_per_1m=(
                self.agent_output_price_per_1m
            ),
        )

        judge_tokens_in = sum(
            judgment.tokens_in
            for judgment in judgments
        )

        judge_tokens_out = sum(
            judgment.tokens_out
            for judgment in judgments
        )

        judge_cost = token_cost(
            tokens_in=judge_tokens_in,
            tokens_out=judge_tokens_out,
            input_price_per_1m=(
                self.judge_input_price_per_1m
            ),
            output_price_per_1m=(
                self.judge_output_price_per_1m
            ),
        )

        return agent_cost + judge_cost

    async def evaluate_item(
        self,
        item: EvalItem,
    ) -> ItemResult:
        async with self.semaphore:
            try:
                response = await self.agent.run(
                    item
                )

                if not self.metrics:
                    return ItemResult(
                        item=item,
                        response=response,
                        passed=True,
                        cost_usd=self._calculate_cost(
                            response,
                            [],
                        ),
                    )

                if self.judge is None:
                    raise RuntimeError(
                        "LLM judge is required "
                        "for configured metrics"
                    )

                judgments: list[Judgment] = []

                for metric in self.metrics:
                    judgment = (
                        await self.judge.evaluate(
                            metric,
                            item,
                            response,
                        )
                    )

                    judgments.append(
                        judgment
                    )

                scores = [
                    judgment.score
                    for judgment in judgments
                ]

                passed = (
                    bool(scores)
                    and all(
                        score
                        >= self.pass_threshold
                        for score in scores
                    )
                )

                return ItemResult(
                    item=item,
                    response=response,
                    judgments=judgments,
                    passed=passed,
                    cost_usd=self._calculate_cost(
                        response,
                        judgments,
                    ),
                )

            except Exception as exc:
                return ItemResult(
                    item=item,
                    response=None,
                    passed=False,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

    async def run(
        self,
        items: list[EvalItem],
    ) -> list[ItemResult]:
        tasks = [
            asyncio.create_task(
                self.evaluate_item(item)
            )
            for item in items
        ]

        results = await asyncio.gather(
            *tasks
        )

        return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_results(
    results: list[ItemResult],
) -> dict[str, Any]:
    latency_values = [
        result.response.latency_ms
        for result in results
        if result.response is not None
    ]

    costs = [
        result.cost_usd
        for result in results
        if result.response is not None
    ]

    agent_tokens_in = sum(
        result.response.tokens_in
        for result in results
        if result.response is not None
    )

    agent_tokens_out = sum(
        result.response.tokens_out
        for result in results
        if result.response is not None
    )

    judge_tokens_in = sum(
        judgment.tokens_in
        for result in results
        for judgment in result.judgments
    )

    judge_tokens_out = sum(
        judgment.tokens_out
        for result in results
        for judgment in result.judgments
    )

    metric_scores: dict[
        str,
        list[float],
    ] = {}

    for result in results:
        for judgment in result.judgments:
            metric_scores.setdefault(
                judgment.metric,
                [],
            ).append(
                judgment.score
            )

    return {
        "total": len(results),
        "passed": sum(
            result.passed
            for result in results
        ),
        "failed": sum(
            not result.passed
            for result in results
        ),
        "errors": sum(
            result.error is not None
            for result in results
        ),
        "pass_rate": (
            sum(
                result.passed
                for result in results
            )
            / len(results)
            if results
            else 0.0
        ),
        "latency_ms": {
            "mean": safe_mean(
                latency_values
            ),
            "p50": percentile(
                latency_values,
                50,
            ),
            "p95": percentile(
                latency_values,
                95,
            ),
            "max": (
                max(latency_values)
                if latency_values
                else 0.0
            ),
        },
        "tokens": {
            "agent_in": agent_tokens_in,
            "agent_out": agent_tokens_out,
            "judge_in": judge_tokens_in,
            "judge_out": judge_tokens_out,
            "total": (
                agent_tokens_in
                + agent_tokens_out
                + judge_tokens_in
                + judge_tokens_out
            ),
        },
        "cost_usd": sum(costs),
        "metrics": {
            metric: safe_mean(scores)
            for metric, scores
            in metric_scores.items()
        },
    }


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------


def _table_exists(
    conn: sqlite3.Connection,
    table: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table,),
    ).fetchone()

    return row is not None


def _table_columns(
    conn: sqlite3.Connection,
    table: str,
) -> set[str]:
    if not _table_exists(
        conn,
        table,
    ):
        return set()

    rows = conn.execute(
        f'PRAGMA table_info("{table}")'
    ).fetchall()

    return {
        str(row[1])
        for row in rows
    }


def _rename_legacy_table(
    conn: sqlite3.Connection,
    table: str,
) -> None:
    """
    Preserve pre-0.2.1 tables instead of silently overwriting them.
    """
    if not _table_exists(
        conn,
        table,
    ):
        return

    columns = _table_columns(
        conn,
        table,
    )

    if "run_id" in columns:
        return

    legacy_name = (
        f"{table}_legacy"
    )

    if _table_exists(
        conn,
        legacy_name,
    ):
        legacy_name = (
            f"{table}_legacy_"
            f"{uuid.uuid4().hex[:8]}"
        )

    conn.execute(
        f'ALTER TABLE "{table}" '
        f'RENAME TO "{legacy_name}"'
    )


def _ensure_sqlite_schema(
    conn: sqlite3.Connection,
) -> None:
    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    _rename_legacy_table(
        conn,
        "metrics",
    )

    _rename_legacy_table(
        conn,
        "evaluations",
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            elapsed_s REAL NOT NULL,
            dataset_hash TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            harness_version TEXT NOT NULL,
            judge_model TEXT,
            agent_type TEXT,
            total_items INTEGER NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluations (
            run_id TEXT NOT NULL,
            id TEXT NOT NULL,
            question TEXT NOT NULL,
            ground_truth TEXT,
            answer TEXT,
            contexts TEXT NOT NULL,
            gold_contexts TEXT NOT NULL,
            latency_ms REAL,
            tokens_in INTEGER NOT NULL DEFAULT 0,
            tokens_out INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL NOT NULL DEFAULT 0,
            passed INTEGER NOT NULL,
            error TEXT,
            error_type TEXT,
            PRIMARY KEY (run_id, id),
            FOREIGN KEY (run_id)
                REFERENCES runs(run_id)
                ON DELETE CASCADE
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            run_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            metric TEXT NOT NULL,
            score REAL NOT NULL,
            reasoning TEXT,
            PRIMARY KEY (run_id, item_id, metric),
            FOREIGN KEY (run_id, item_id)
                REFERENCES evaluations(run_id, id)
                ON DELETE CASCADE
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_evaluations_id
        ON evaluations(id)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_metrics_metric
        ON metrics(metric)
        """
    )


def persist_results(
    results: list[ItemResult],
    sqlite_path: str | Path,
    *,
    run_id: str,
    dataset_hash_value: str,
    config_hash_value: str,
    cfg: dict[str, Any],
    started_at: str,
    finished_at: str,
    elapsed_s: float,
) -> None:
    sqlite_path = Path(
        sqlite_path
    )

    if str(sqlite_path) != ":memory:":
        sqlite_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    conn = sqlite3.connect(
        sqlite_path
    )

    try:
        _ensure_sqlite_schema(
            conn
        )

        judge_cfg = (
            cfg.get("judge") or {}
        )

        agent_cfg = (
            cfg.get("agent") or {}
        )

        conn.execute("BEGIN")

        conn.execute(
            """
            INSERT INTO runs (
                run_id,
                started_at,
                finished_at,
                elapsed_s,
                dataset_hash,
                config_hash,
                harness_version,
                judge_model,
                agent_type,
                total_items
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                started_at,
                finished_at,
                float(elapsed_s),
                dataset_hash_value,
                config_hash_value,
                __version__,
                str(
                    judge_cfg.get(
                        "model",
                        "",
                    )
                ),
                str(
                    agent_cfg.get(
                        "type",
                        "",
                    )
                ),
                len(results),
            ),
        )

        for result in results:
            response = result.response

            conn.execute(
                """
                INSERT INTO evaluations (
                    run_id,
                    id,
                    question,
                    ground_truth,
                    answer,
                    contexts,
                    gold_contexts,
                    latency_ms,
                    tokens_in,
                    tokens_out,
                    cost_usd,
                    passed,
                    error,
                    error_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    result.item.id,
                    result.item.question,
                    result.item.ground_truth,
                    (
                        response.answer
                        if response
                        else None
                    ),
                    json.dumps(
                        (
                            response.contexts
                            if response
                            else []
                        ),
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        result.item.contexts,
                        ensure_ascii=False,
                    ),
                    (
                        response.latency_ms
                        if response
                        else None
                    ),
                    (
                        response.tokens_in
                        if response
                        else 0
                    ),
                    (
                        response.tokens_out
                        if response
                        else 0
                    ),
                    result.cost_usd,
                    int(result.passed),
                    result.error,
                    result.error_type,
                ),
            )

            for judgment in result.judgments:
                conn.execute(
                    """
                    INSERT INTO metrics (
                        run_id,
                        item_id,
                        metric,
                        score,
                        reasoning
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        result.item.id,
                        judgment.metric,
                        judgment.score,
                        judgment.reasoning,
                    ),
                )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_report(
    results: list[ItemResult],
    cfg: dict[str, Any],
    *,
    dataset_hash_value: str,
    config_hash_value: str,
    run_id: str,
    elapsed_s: float,
) -> str:
    summary = aggregate_results(
        results
    )

    lines: list[str] = []

    lines.append(
        "# Agent-Eval Report"
    )
    lines.append("")

    lines.append(
        f"- Run ID: `{run_id}`"
    )
    lines.append(
        f"- Harness version: `{__version__}`"
    )
    lines.append(
        f"- Dataset hash: `{dataset_hash_value}`"
    )
    lines.append(
        f"- Config hash: `{config_hash_value}`"
    )
    lines.append(
        f"- Duration: `{elapsed_s:.2f}s`"
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")

    lines.append(
        f"- Items: **{summary['total']}**"
    )
    lines.append(
        f"- Passed: **{summary['passed']}**"
    )
    lines.append(
        f"- Failed: **{summary['failed']}**"
    )
    lines.append(
        f"- Errors: **{summary['errors']}**"
    )
    lines.append(
        f"- Pass rate: **{summary['pass_rate']:.1%}**"
    )

    lines.append(
        f"- Mean latency: "
        f"**{summary['latency_ms']['mean']:.1f} ms**"
    )

    lines.append(
        f"- P95 latency: "
        f"**{summary['latency_ms']['p95']:.1f} ms**"
    )

    lines.append(
        f"- Agent input tokens: "
        f"**{summary['tokens']['agent_in']:,}**"
    )

    lines.append(
        f"- Agent output tokens: "
        f"**{summary['tokens']['agent_out']:,}**"
    )

    lines.append(
        f"- Judge input tokens: "
        f"**{summary['tokens']['judge_in']:,}**"
    )

    lines.append(
        f"- Judge output tokens: "
        f"**{summary['tokens']['judge_out']:,}**"
    )

    lines.append(
        f"- Total tokens: "
        f"**{summary['tokens']['total']:,}**"
    )

    lines.append(
        f"- Estimated cost: "
        f"**${summary['cost_usd']:.6f}**"
    )

    lines.append("")

    lines.append("## Metrics")
    lines.append("")

    if summary["metrics"]:
        lines.append(
            "| Metric | Mean score |"
        )
        lines.append(
            "|---|---:|"
        )

        for metric, score in sorted(
            summary["metrics"].items()
        ):
            lines.append(
                f"| {metric} | {score:.3f} |"
            )

    else:
        lines.append(
            "No LLM metrics were configured."
        )

    lines.append("")

    lines.append(
        "## Configuration"
    )
    lines.append("")
    lines.append("```json")

    lines.append(
        json.dumps(
            cfg,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    lines.append("```")
    lines.append("")

    lines.append("## Results")
    lines.append("")

    metrics = list(
        summary["metrics"].keys()
    )

    header = [
        "ID",
        "Passed",
        "Latency ms",
        *metrics,
        "Cost USD",
        "Error",
    ]

    lines.append(
        "| "
        + " | ".join(header)
        + " |"
    )

    lines.append(
        "| "
        + " | ".join(
            "---"
            if index in {
                0,
                len(header) - 1,
            }
            else "---:"
            for index in range(
                len(header)
            )
        )
        + " |"
    )

    for result in results:
        score_map = {
            judgment.metric: judgment.score
            for judgment in result.judgments
        }

        values = [
            result.item.id,
            (
                "yes"
                if result.passed
                else "no"
            ),
            (
                f"{result.response.latency_ms:.1f}"
                if result.response
                else "-"
            ),
        ]

        values.extend(
            (
                f"{score_map[metric]:.3f}"
                if metric in score_map
                else "-"
            )
            for metric in metrics
        )

        values.append(
            f"{result.cost_usd:.6f}"
        )

        values.append(
            (result.error or "")
            .replace("|", "\\|")
            .replace("\n", " ")
        )

        lines.append(
            "| "
            + " | ".join(values)
            + " |"
        )

    lines.append("")

    failures = [
        result
        for result in results
        if not result.passed
    ]

    if failures:
        lines.append(
            "## Failures"
        )
        lines.append("")

        for result in failures:
            lines.append(
                f"### {result.item.id}"
            )
            lines.append("")

            lines.append(
                f"**Question:** "
                f"{result.item.question}"
            )
            lines.append("")

            if result.error:
                lines.append(
                    f"**Error:** "
                    f"`{result.error_type or 'Error'}` "
                    f"— {result.error}"
                )
                lines.append("")

            if result.response:
                lines.append(
                    f"**Answer:** "
                    f"{result.response.answer}"
                )
                lines.append("")

            for judgment in result.judgments:
                lines.append(
                    f"- **{judgment.metric}:** "
                    f"{judgment.score:.3f} — "
                    f"{judgment.reasoning}"
                )

            lines.append("")

    lines.append(
        "## Reproducibility"
    )
    lines.append("")

    lines.append(
        f"- Dataset SHA-256: "
        f"`{dataset_hash_value}`"
    )

    lines.append(
        f"- Config SHA-256: "
        f"`{config_hash_value}`"
    )

    lines.append(
        f"- Run ID: `{run_id}`"
    )

    lines.append(
        f"- Harness version: "
        f"`{__version__}`"
    )

    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo_dataset() -> list[EvalItem]:
    return [
        EvalItem(
            id="demo-1",
            question=(
                "What is the capital of France?"
            ),
            ground_truth="Paris",
            contexts=[
                (
                    "Paris is the capital and "
                    "largest city of France."
                )
            ],
        ),
        EvalItem(
            id="demo-2",
            question=(
                "What language is primarily "
                "spoken in Brazil?"
            ),
            ground_truth="Portuguese",
            contexts=[
                (
                    "Portuguese is the official "
                    "and most widely spoken "
                    "language of Brazil."
                )
            ],
        ),
        EvalItem(
            id="demo-3",
            question=(
                "What is the largest planet "
                "in the Solar System?"
            ),
            ground_truth="Jupiter",
            contexts=[
                (
                    "Jupiter is the largest planet "
                    "in the Solar System."
                )
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lightweight evaluation harness "
            "for agents/RAG systems."
        )
    )

    parser.add_argument(
        "--config",
        help=(
            "Path to JSON/YAML "
            "configuration file."
        ),
    )

    parser.add_argument(
        "--dataset",
        help="Override dataset path.",
    )

    parser.add_argument(
        "--agent-url",
        help="Override HTTP agent URL.",
    )

    parser.add_argument(
        "--judge-model",
        help="Override judge model.",
    )

    parser.add_argument(
        "--output",
        help="Override Markdown report path.",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        help=(
            "Override evaluation "
            "concurrency."
        ),
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run the built-in "
            "deterministic demo."
        ),
    )

    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run built-in self-tests.",
    )

    parser.add_argument(
        "--no-sqlite",
        action="store_true",
        help="Disable SQLite persistence.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )

    return parser.parse_args()


def build_agent(
    cfg: dict[str, Any],
) -> Any:
    agent_cfg = (
        cfg.get("agent") or {}
    )

    agent_type = str(
        agent_cfg.get(
            "type",
            "http",
        )
    ).lower()

    if agent_type == "http":
        return HTTPAgent(
            agent_cfg
        )

    if agent_type == "builtin":
        return BuiltinAgent()

    raise ValueError(
        f"Unsupported agent type: "
        f"{agent_type}"
    )


def build_judge(
    cfg: dict[str, Any],
) -> Judge | None:
    metrics = list(
        (
            cfg.get("eval") or {}
        ).get(
            "metrics",
            [],
        )
    )

    if not metrics:
        return None

    judge_cfg = (
        cfg.get("judge") or {}
    )

    return Judge(
        LLMClient(
            judge_cfg
        )
    )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def selftest() -> None:
    checks: list[
        tuple[str, bool]
    ] = []

    checks.append(
        (
            "estimate_tokens empty",
            estimate_tokens("") == 0,
        )
    )

    checks.append(
        (
            "estimate_tokens non-empty",
            estimate_tokens("hello") >= 1,
        )
    )

    checks.append(
        (
            "price_for",
            abs(
                price_for(
                    1_000_000,
                    2.0,
                )
                - 2.0
            )
            < 1e-9,
        )
    )

    checks.append(
        (
            "token_cost",
            abs(
                token_cost(
                    1_000_000,
                    500_000,
                    1.0,
                    2.0,
                )
                - 2.0
            )
            < 1e-9,
        )
    )

    checks.append(
        (
            "sha256_short",
            len(
                sha256_short("hello")
            )
            == 12,
        )
    )

    checks.append(
        (
            "canonical_hash stable",
            canonical_hash(
                {
                    "b": 2,
                    "a": 1,
                }
            )
            == canonical_hash(
                {
                    "a": 1,
                    "b": 2,
                }
            ),
        )
    )

    checks.append(
        (
            "percentile",
            percentile(
                [1, 2, 3, 4, 5],
                50,
            )
            == 3,
        )
    )

    checks.append(
        (
            "safe_mean empty",
            safe_mean([]) == 0.0,
        )
    )

    parsed = _extract_json(
        'Here is the result: '
        '{"score": 0.8, "reasoning": "ok"}'
    )

    checks.append(
        (
            "extract_json",
            isinstance(
                parsed,
                dict,
            )
            and parsed["score"] == 0.8,
        )
    )

    merged = deep_merge(
        {
            "a": {
                "b": 1,
                "c": 2,
            }
        },
        {
            "a": {
                "b": 3,
            }
        },
    )

    checks.append(
        (
            "deep_merge",
            merged
            == {
                "a": {
                    "b": 3,
                    "c": 2,
                }
            },
        )
    )

    async def test_builtin() -> None:
        agent = BuiltinAgent()

        response = await agent.run(
            EvalItem(
                id="test",
                question=(
                    "What is Python?"
                ),
                contexts=[
                    (
                        "Python is a "
                        "programming language."
                    )
                ],
            )
        )

        checks.append(
            (
                "builtin_agent",
                response.answer
                == (
                    "Python is a "
                    "programming language."
                ),
            )
        )

    asyncio.run(
        test_builtin()
    )

    result = ItemResult(
        item=EvalItem(
            id="test",
            question="Question",
        ),
        response=AgentResponse(
            answer="Answer",
            latency_ms=10,
            tokens_in=100,
            tokens_out=50,
        ),
        judgments=[
            Judgment(
                metric="faithfulness",
                score=1.0,
                tokens_in=200,
                tokens_out=20,
            )
        ],
        passed=True,
        cost_usd=0.002,
    )

    aggregate = aggregate_results(
        [result]
    )

    checks.append(
        (
            "aggregate",
            aggregate["total"] == 1
            and aggregate["passed"] == 1
            and aggregate["metrics"][
                "faithfulness"
            ]
            == 1.0
            and aggregate["tokens"][
                "agent_in"
            ]
            == 100
            and aggregate["tokens"][
                "judge_in"
            ]
            == 200
            and aggregate["cost_usd"]
            == 0.002,
        )
    )

    # Verify that the evaluator calculates agent + judge costs independently.
    cost_cfg = json.loads(
        json.dumps(CONFIG)
    )

    cost_cfg["agent"]["prices"] = {
        "input_per_1m": 1.0,
        "output_per_1m": 2.0,
    }

    cost_cfg["judge"]["prices"] = {
        "input_per_1m": 3.0,
        "output_per_1m": 4.0,
    }

    evaluator = Evaluator(
        agent=BuiltinAgent(),
        judge=None,
        cfg=cost_cfg,
    )

    cost_response = AgentResponse(
        answer="answer",
        tokens_in=1_000_000,
        tokens_out=500_000,
    )

    cost_judgment = Judgment(
        metric="faithfulness",
        score=1.0,
        tokens_in=2_000_000,
        tokens_out=250_000,
    )

    expected_cost = (
        price_for(
            1_000_000,
            1.0,
        )
        + price_for(
            500_000,
            2.0,
        )
        + price_for(
            2_000_000,
            3.0,
        )
        + price_for(
            250_000,
            4.0,
        )
    )

    actual_cost = evaluator._calculate_cost(
        cost_response,
        [cost_judgment],
    )

    checks.append(
        (
            "agent + judge token cost",
            abs(
                actual_cost
                - expected_cost
            )
            < 1e-9,
        )
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = (
            Path(temp_dir)
            / "test.db"
        )

        cfg = json.loads(
            json.dumps(CONFIG)
        )

        persist_results(
            [result],
            db_path,
            run_id="run-1",
            dataset_hash_value=(
                "dataset-1"
            ),
            config_hash_value=(
                "config-1"
            ),
            cfg=cfg,
            started_at=(
                "2026-01-01T00:00:00+00:00"
            ),
            finished_at=(
                "2026-01-01T00:00:01+00:00"
            ),
            elapsed_s=1.0,
        )

        result_2 = ItemResult(
            item=EvalItem(
                id="test",
                question="Question 2",
            ),
            response=AgentResponse(
                answer="Answer 2",
                latency_ms=20,
            ),
            judgments=[
                Judgment(
                    metric="faithfulness",
                    score=0.5,
                )
            ],
            passed=False,
        )

        persist_results(
            [result_2],
            db_path,
            run_id="run-2",
            dataset_hash_value=(
                "dataset-2"
            ),
            config_hash_value=(
                "config-2"
            ),
            cfg=cfg,
            started_at=(
                "2026-01-02T00:00:00+00:00"
            ),
            finished_at=(
                "2026-01-02T00:00:01+00:00"
            ),
            elapsed_s=1.0,
        )

        conn = sqlite3.connect(
            db_path
        )

        try:
            runs_count = conn.execute(
                "SELECT COUNT(*) FROM runs"
            ).fetchone()[0]

            evaluations_count = (
                conn.execute(
                    "SELECT COUNT(*) "
                    "FROM evaluations"
                )
                .fetchone()[0]
            )

            metrics_count = conn.execute(
                "SELECT COUNT(*) "
                "FROM metrics"
            ).fetchone()[0]

        finally:
            conn.close()

        checks.extend(
            [
                (
                    "sqlite keeps historical runs",
                    runs_count == 2,
                ),
                (
                    "sqlite keeps evaluations",
                    evaluations_count == 2,
                ),
                (
                    "sqlite keeps metrics",
                    metrics_count == 2,
                ),
            ]
        )

    report = render_report(
        [result],
        CONFIG,
        dataset_hash_value="dataset",
        config_hash_value="config",
        run_id="run",
        elapsed_s=0.1,
    )

    checks.append(
        (
            "report smoke",
            "# Agent-Eval Report"
            in report
            and "dataset"
            in report
            and "config"
            in report
            and "Estimated cost"
            in report,
        )
    )

    failures = [
        name
        for name, passed
        in checks
        if not passed
    ]

    if failures:
        raise AssertionError(
            "Self-tests failed: "
            + ", ".join(failures)
        )

    print(
        f"Self-tests passed: "
        f"{len(checks)} checks."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main(
    args: argparse.Namespace,
) -> int:
    cfg = json.loads(
        json.dumps(CONFIG)
    )

    if args.config:
        cfg = deep_merge(
            cfg,
            load_config_file(
                args.config
            ),
        )

    if args.dataset:
        cfg["dataset"]["path"] = (
            args.dataset
        )

    if args.agent_url:
        cfg["agent"]["url"] = (
            args.agent_url
        )

    if args.judge_model:
        cfg["judge"]["model"] = (
            args.judge_model
        )

    if args.output:
        cfg["output"]["report"] = (
            args.output
        )

    if args.concurrency is not None:
        cfg["eval"]["concurrency"] = (
            args.concurrency
        )

    if args.demo:
        items = demo_dataset()

        cfg["agent"]["type"] = (
            "builtin"
        )

        cfg["eval"]["metrics"] = []

    else:
        items = load_dataset(
            cfg["dataset"]["path"],
            cfg["dataset"].get(
                "encoding",
                "utf-8",
            ),
        )

    dataset_hash_value = dataset_hash(
        items
    )

    config_hash_value = canonical_hash(
        cfg
    )

    run_id = uuid.uuid4().hex[:12]

    started = time.perf_counter()

    started_at = (
        datetime.now(
            timezone.utc
        ).isoformat(
            timespec="seconds"
        )
    )

    agent = build_agent(
        cfg
    )

    judge = build_judge(
        cfg
    )

    evaluator = Evaluator(
        agent=agent,
        judge=judge,
        cfg=cfg,
    )

    results = await evaluator.run(
        items
    )

    elapsed_s = (
        time.perf_counter()
        - started
    )

    finished_at = (
        datetime.now(
            timezone.utc
        ).isoformat(
            timespec="seconds"
        )
    )

    report = render_report(
        results,
        cfg,
        dataset_hash_value=(
            dataset_hash_value
        ),
        config_hash_value=(
            config_hash_value
        ),
        run_id=run_id,
        elapsed_s=elapsed_s,
    )

    report_path = Path(
        cfg["output"]["report"]
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    if not args.no_sqlite:
        persist_results(
            results,
            cfg["output"]["sqlite"],
            run_id=run_id,
            dataset_hash_value=(
                dataset_hash_value
            ),
            config_hash_value=(
                config_hash_value
            ),
            cfg=cfg,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_s=elapsed_s,
        )

    if args.verbose:
        summary = aggregate_results(
            results
        )

        print(
            f"Run ID: {run_id}"
        )

        print(
            f"Dataset hash: "
            f"{dataset_hash_value}"
        )

        print(
            f"Config hash: "
            f"{config_hash_value}"
        )

        print(
            f"Items: "
            f"{summary['total']}"
        )

        print(
            f"Passed: "
            f"{summary['passed']}"
        )

        print(
            f"Failed: "
            f"{summary['failed']}"
        )

        print(
            f"Agent input tokens: "
            f"{summary['tokens']['agent_in']:,}"
        )

        print(
            f"Agent output tokens: "
            f"{summary['tokens']['agent_out']:,}"
        )

        print(
            f"Judge input tokens: "
            f"{summary['tokens']['judge_in']:,}"
        )

        print(
            f"Judge output tokens: "
            f"{summary['tokens']['judge_out']:,}"
        )

        print(
            f"Total cost: "
            f"${summary['cost_usd']:.6f}"
        )

        print(
            f"Duration: "
            f"{elapsed_s:.2f}s"
        )

        print(
            f"Report: "
            f"{report_path}"
        )

    return (
        0
        if all(
            result.passed
            for result in results
        )
        else 1
    )


def main() -> int:
    args = parse_args()

    if args.selftest:
        selftest()
        return 0

    try:
        return asyncio.run(
            async_main(args)
        )

    except KeyboardInterrupt:
        print(
            "Interrupted."
        )
        return 130

    except Exception as exc:
        print(
            f"Error: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=os.sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
````
