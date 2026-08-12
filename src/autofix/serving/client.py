"""Inference against the fine-tuned models via vLLM.

vLLM serves an OpenAI-compatible endpoint, so this client is thin. Two features
matter for this project specifically:

* **n>1 sampling in one request.** Rejection sampling needs k candidates per
  instance; asking vLLM for all k at once lets it batch them against a single
  prefill of the (long) prompt, which is several times faster than k separate
  requests.
* **LoRA adapter selection per request.** The reranker and editor are different
  adapters over different bases, and `model=` picks between served adapters
  without reloading anything.

There is no Anthropic dependency anywhere in this codebase. `LocalLlmClient` is
the only path to a model.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from autofix.logging_conf import get_logger

log = get_logger(__name__)


class InferenceError(RuntimeError):
    pass


class _Retryable(RuntimeError):
    pass


@dataclass(slots=True)
class Generation:
    texts: list[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LocalLlmClient:
    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout: float = 600.0) -> None:
        self._base = base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> LocalLlmClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def health(self) -> bool:
        try:
            resp = await self._http.get(f"{self._base}/models", timeout=10.0)
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    @retry(
        retry=retry_if_exception_type(_Retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    async def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        n: int = 1,
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> Generation:
        payload: dict = {
            "model": model,
            "messages": messages,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop

        try:
            resp = await self._http.post(f"{self._base}/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise _Retryable(str(exc)) from exc

        if resp.status_code >= 500 or resp.status_code == 429:
            raise _Retryable(f"{resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise InferenceError(f"vLLM {resp.status_code}: {resp.text[:400]}")

        data = resp.json()
        usage = data.get("usage", {})
        return Generation(
            texts=[c["message"]["content"] for c in data["choices"]],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
