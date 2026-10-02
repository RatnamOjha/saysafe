"""The only LLM entry point: OpenAI-compatible complete() that returns None on any failure.

LLMs are optional helpers here. Every caller must have a non-LLM fallback, so this
never raises and never retries: it returns a result or None and logs why.
Message content is never logged, since it can hold codes or balances.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from saysafe.config import env

log = logging.getLogger(__name__)

ModelTier = Literal["fast", "smart"]

_DEFAULT_MODELS = {"fast": "openai/gpt-oss-20b", "smart": "openai/gpt-oss-120b"}


@dataclass
class LLMResult:
    model: str
    latency_ms: float
    text: str | None = None
    parsed: BaseModel | None = None
    tool_calls: list[Any] | None = None


def model_name(tier: ModelTier) -> str:
    return env(f"LLM_MODEL_{tier.upper()}", _DEFAULT_MODELS[tier])


def _client(timeout_s: float):
    """Built per call so env changes (and test monkeypatching) take effect."""
    from openai import OpenAI

    return OpenAI(
        base_url=env("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=env("LLM_API_KEY"),
        timeout=timeout_s,
        max_retries=0,
    )


def complete(
    messages: list[dict],
    model: ModelTier = "fast",
    timeout_s: float = 1.5,
    json_schema: type[BaseModel] | None = None,
    tools: list[dict] | None = None,
) -> LLMResult | None:
    """One chat completion. Returns None on missing key, error, timeout or invalid JSON.

    With json_schema, the model is asked for JSON matching that pydantic model and
    the result is validated into LLMResult.parsed.
    """
    if not env("LLM_API_KEY"):
        log.info("llm: skipped, no LLM_API_KEY set")
        return None

    name = model_name(model)
    kwargs: dict[str, Any] = {"model": name, "messages": list(messages)}
    if json_schema is not None:
        schema = json.dumps(json_schema.model_json_schema())
        instruction = f"Reply with only a JSON object matching this schema: {schema}"
        kwargs["messages"] = [
            {"role": "system", "content": instruction},
            *messages,
        ]
        kwargs["response_format"] = {"type": "json_object"}
    if tools:
        kwargs["tools"] = tools

    start = time.perf_counter()
    try:
        response = _client(timeout_s).chat.completions.create(**kwargs)
    except Exception as e:  # any provider, network or timeout error -> fallback path
        log.warning("llm: %s failed after %.0f ms: %s", name, _ms(start), type(e).__name__)
        return None
    latency_ms = _ms(start)

    message = response.choices[0].message
    result = LLMResult(
        model=response.model or name,
        latency_ms=latency_ms,
        text=message.content,
        tool_calls=message.tool_calls or None,
    )
    if json_schema is not None:
        try:
            result.parsed = json_schema.model_validate_json(message.content or "")
        except ValidationError:
            log.warning("llm: %s returned JSON that doesn't match %s", name, json_schema.__name__)
            return None
    return result


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
