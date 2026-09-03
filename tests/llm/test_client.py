import json

import httpx
import pytest

from arxiv_digest.llm.client import (
    LLMClient,
    LLMClientError,
    LLMConfigurationError,
    LLMResponseError,
)


MESSAGES = [{"role": "user", "content": "Return JSON"}]


def response(content, status_code=200):
    return httpx.Response(
        status_code,
        json={"choices": [{"message": {"content": content}}]},
    )


def client(handler, **kwargs):
    return LLMClient(
        api_key="test-secret-key",
        base_url="https://llm.example/v1",
        model="test-model",
        transport=httpx.MockTransport(handler),
        sleep=lambda seconds: None,
        **kwargs,
    )


def test_llm_valid_json():
    llm = client(lambda request: response('{"relevance": 9}'))

    assert llm.chat_json(MESSAGES) == {"relevance": 9}


def test_llm_json_inside_markdown_fence():
    llm = client(lambda request: response('```json\n{"relevance": 8}\n```'))

    assert llm.chat_json(MESSAGES) == {"relevance": 8}


def test_llm_malformed_json():
    llm = client(lambda request: response("{not-json}"))

    with pytest.raises(LLMResponseError, match="not valid JSON"):
        llm.chat_json(MESSAGES)


def test_llm_retries_429():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return response("rate limited", status_code=429) if calls == 1 else response("ok")

    llm = client(handler, max_retries=2)

    assert llm.chat(MESSAGES) == "ok"
    assert calls == 2


def test_llm_retries_5xx():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return response("unavailable", status_code=503) if calls < 3 else response("ok")

    llm = client(handler, max_retries=2)

    assert llm.chat(MESSAGES) == "ok"
    assert calls == 3


def test_llm_timeout():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    llm = client(handler, max_retries=1)

    with pytest.raises(LLMClientError, match="timed out"):
        llm.chat(MESSAGES)
    assert calls == 2


def test_llm_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    with pytest.raises(LLMConfigurationError, match="LLM_API_KEY is required"):
        LLMClient.from_env()


def test_llm_uses_prefixed_model_configuration(monkeypatch):
    monkeypatch.setenv("DEEP_LLM_API_KEY", "deep-key")
    monkeypatch.setenv("DEEP_LLM_BASE_URL", "https://deep.example/v1")
    monkeypatch.setenv("DEEP_LLM_MODEL", "strong-model")

    llm = LLMClient.from_env(
        env_prefix="DEEP_LLM",
        transport=httpx.MockTransport(lambda request: response("ok")),
    )
    try:
        assert llm.model == "strong-model"
        assert llm.chat(MESSAGES) == "ok"
    finally:
        llm.close()


def test_llm_prefixed_configuration_falls_back_to_legacy(monkeypatch):
    for name in ("API_KEY", "BASE_URL", "MODEL"):
        monkeypatch.delenv(f"SCREENING_LLM_{name}", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "legacy-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://legacy.example/v1")
    monkeypatch.setenv("LLM_MODEL", "legacy-model")

    llm = LLMClient.from_env(
        env_prefix="SCREENING_LLM",
        fallback_prefix="LLM",
        transport=httpx.MockTransport(lambda request: response("ok")),
    )
    try:
        assert llm.model == "legacy-model"
        assert llm.chat(MESSAGES) == "ok"
    finally:
        llm.close()


def test_llm_does_not_log_api_key(capsys, caplog):
    secret = "a-very-sensitive-test-key"
    llm = LLMClient(
        api_key=secret,
        base_url="https://llm.example/v1",
        model="test-model",
        transport=httpx.MockTransport(lambda request: response("denied", 401)),
        max_retries=0,
    )

    with pytest.raises(LLMClientError) as error:
        llm.chat(MESSAGES)

    output = capsys.readouterr()
    observed = f"{output.out}{output.err}{caplog.text}{error.value}"
    assert secret not in observed


def test_llm_sends_openai_compatible_request():
    def handler(request):
        assert request.url == "https://llm.example/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-secret-key"
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        assert payload["messages"] == MESSAGES
        return response("ok")

    assert client(handler).chat(MESSAGES) == "ok"
