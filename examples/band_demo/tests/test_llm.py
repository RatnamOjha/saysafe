from types import SimpleNamespace

import pytest
from band_demo import llm
from pydantic import BaseModel


class Level(BaseModel):
    level: str


def _fake_client(content=None, exc=None):
    def create(**kwargs):
        if exc:
            raise exc
        message = SimpleNamespace(content=content, tool_calls=None)
        return SimpleNamespace(model=kwargs["model"], choices=[SimpleNamespace(message=message)])

    completions = SimpleNamespace(create=create)
    return lambda timeout_s: SimpleNamespace(chat=SimpleNamespace(completions=completions))


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")


def test_no_key_returns_none(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    assert llm.complete([{"role": "user", "content": "hi"}]) is None


def test_error_returns_none(with_key, monkeypatch):
    monkeypatch.setattr(llm, "_client", _fake_client(exc=TimeoutError()))
    assert llm.complete([{"role": "user", "content": "hi"}]) is None


def test_text_result(with_key, monkeypatch):
    monkeypatch.setattr(llm, "_client", _fake_client(content="pong"))
    result = llm.complete([{"role": "user", "content": "hi"}], model="smart")
    assert result.text == "pong"
    assert result.model == llm.model_name("smart")
    assert result.latency_ms >= 0


def test_json_schema_valid(with_key, monkeypatch):
    monkeypatch.setattr(llm, "_client", _fake_client(content='{"level": "secret"}'))
    result = llm.complete([{"role": "user", "content": "x"}], json_schema=Level)
    assert result.parsed == Level(level="secret")


@pytest.mark.parametrize("content", ['{"lvl": "secret"}', "not json", None])
def test_json_schema_invalid_returns_none(with_key, monkeypatch, content):
    monkeypatch.setattr(llm, "_client", _fake_client(content=content))
    assert llm.complete([{"role": "user", "content": "x"}], json_schema=Level) is None
