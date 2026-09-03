from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from typing import Any

import httpx


class LLMClientError(RuntimeError):
    """Base error for safe, redacted LLM client failures."""


class LLMConfigurationError(LLMClientError):
    """Required LLM configuration is missing."""


class LLMResponseError(LLMClientError):
    """The provider returned an invalid response."""


class LLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        timeout_seconds: float = 120,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise LLMConfigurationError("LLM_API_KEY is required")
        if not base_url:
            raise LLMConfigurationError("LLM_BASE_URL is required")
        if not model:
            raise LLMConfigurationError("LLM_MODEL is required")
        self.model = model
        self.temperature = temperature
        self.max_retries = max(0, max_retries)
        self._api_key = api_key
        self._endpoint = _chat_endpoint(base_url)
        self._sleep = sleep
        self._http = httpx.Client(timeout=timeout_seconds, transport=transport)

    @classmethod
    def from_env(
        cls,
        config: dict[str, Any] | None = None,
        *,
        env_prefix: str = "LLM",
        fallback_prefix: str | None = None,
        **kwargs: Any,
    ) -> LLMClient:
        settings = config or {}
        def setting(name: str) -> str:
            value = os.environ.get(f"{env_prefix}_{name}", "")
            if not value and fallback_prefix:
                value = os.environ.get(f"{fallback_prefix}_{name}", "")
            return value

        return cls(
            api_key=setting("API_KEY"),
            base_url=setting("BASE_URL"),
            model=setting("MODEL"),
            temperature=settings.get("temperature", 0.2),
            timeout_seconds=settings.get("timeout_seconds", 120),
            max_retries=settings.get("max_retries", 3),
            **kwargs,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        response = self._post_with_retry(payload)
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError):
            raise LLMResponseError("LLM response is missing message content") from None
        if not isinstance(content, str):
            raise LLMResponseError("LLM message content must be text")
        return content

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        content = _strip_json_fence(self.chat(messages, temperature=temperature))
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            raise LLMResponseError("LLM message content is not valid JSON") from None
        if not isinstance(parsed, dict):
            raise LLMResponseError("LLM JSON response must be an object")
        return parsed

    def _post_with_retry(self, payload: dict[str, Any]) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            try:
                response = self._http.post(
                    self._endpoint,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    self._sleep(2**attempt)
                    continue
                raise LLMClientError("LLM request timed out") from None
            except httpx.RequestError:
                raise LLMClientError("LLM request failed") from None

            if response.status_code == 429 or response.status_code >= 500:
                if attempt < self.max_retries:
                    self._sleep(2**attempt)
                    continue
            if response.is_error:
                raise LLMClientError(
                    f"LLM request failed with HTTP {response.status_code}"
                )
            return response
        raise LLMClientError("LLM request failed after retries")


def _chat_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.I)
    return match.group(1).strip() if match else stripped
