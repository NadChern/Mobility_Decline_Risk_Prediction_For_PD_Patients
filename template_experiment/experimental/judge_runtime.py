"""Judge-only bounded OpenRouter transport; never substitutes reasoning for an answer.

Use the SDK directly to avoid the installed LangChain wrapper's retry-default bug.
The asynchronous request is cancelled at the deadline; no background worker is left running.
Cancellation cannot guarantee remote cancellation or prevent charges already incurred.
"""

import asyncio
from contextlib import contextmanager
import fcntl
from importlib.metadata import version
from pathlib import Path

import httpx
from openrouter import OpenRouter

from explanation.data_loader import (OPENROUTER_API_KEY, SYNTHESIS_JUDGE_MAX_TOKENS,
                                     SYNTHESIS_JUDGE_REASONING, SYNTHESIS_JUDGE_MODEL_SETTINGS)
from ..run_support import output_lock
from ..synthesis_protocol import PROJECT_ROOT

RUNTIME_VERSION = "openrouter-judge-runtime-v2"
STUDY_OUTPUT = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2"


@contextmanager
def collection_guard(output_dir, execute):
    """Serialize live study collectors across versions; detect still-running legacy jobs."""
    if not execute or not Path(output_dir).resolve().is_relative_to(STUDY_OUTPUT.resolve()):
        yield
        return
    with output_lock(STUDY_OUTPUT / ".judge_runtime_session"):
        for path in STUDY_OUTPUT.rglob(".collection.lock"):
            if not path.parent.name.startswith("judge_"):
                continue
            with path.open("r") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(f"A judge collector is still running in {path.parent}. "
                                       "Stop it before starting a different collection.") from exc
                fcntl.flock(stream, fcntl.LOCK_UN)
        yield


_DEFAULT_REASONING = object()


def runtime_settings(request_timeout=120, *, model=None, max_tokens=None, reasoning=_DEFAULT_REASONING):
    """Resolve model overrides identically for the frozen request and live client.

    Explicit overrides take precedence; omitting model preserves global/retest behavior.
    """
    if request_timeout <= 0:
        raise ValueError("Judge timeout must be positive seconds.")
    overrides = SYNTHESIS_JUDGE_MODEL_SETTINGS.get(model, {})
    tokens = overrides.get("max_tokens", SYNTHESIS_JUDGE_MAX_TOKENS) if max_tokens is None else max_tokens
    reasoning = overrides.get("reasoning", SYNTHESIS_JUDGE_REASONING) if reasoning is _DEFAULT_REASONING else reasoning
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("Judge max_tokens must be a positive integer.")
    return {"runtime_version": RUNTIME_VERSION, "max_tokens": tokens,
            "reasoning": reasoning, "timeout_seconds": request_timeout,
            "sdk_timeout_ms": round(request_timeout * 1000), "sdk_retries": 0,
            "openrouter_sdk_version": version("openrouter"),
            "httpx_version": version("httpx"), "final_answer_source": "content_only"}


class JudgeClient:
    def __init__(self, model, temperature, settings):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        self.model, self.temperature, self.settings = model, temperature, settings

    async def _request(self, prompt):
        settings = self.settings
        async with httpx.AsyncClient(timeout=settings["timeout_seconds"],
                                     transport=httpx.AsyncHTTPTransport(retries=0)) as client:
            sdk = OpenRouter(api_key=OPENROUTER_API_KEY, async_client=client,
                             timeout_ms=settings["sdk_timeout_ms"], retry_config=None)
            kwargs = {"model": self.model, "temperature": self.temperature,
                      "max_tokens": settings["max_tokens"], "stream": False,
                      "messages": [{"role": "user", "content": prompt}],
                      "timeout_ms": settings["sdk_timeout_ms"], "retries": None}
            if settings["reasoning"] is not None:
                kwargs["reasoning"] = settings["reasoning"]
            response = await sdk.chat.send_async(**kwargs)
            return response.model_dump(mode="json", by_alias=True, exclude_none=True)

    async def _bounded_request(self, prompt):
        return await asyncio.wait_for(self._request(prompt), self.settings["timeout_seconds"])

    def invoke(self, prompt):
        return asyncio.run(self._bounded_request(prompt))


def build_judge_client(model, temperature, request_timeout=120):
    return JudgeClient(model, temperature, runtime_settings(request_timeout, model=model))


def panel_runtime_settings(model, request_timeout=120):
    """Return the exact per-model settings used by normal panel collection."""
    return runtime_settings(request_timeout, model=model)


def extract_final_content(response):
    if isinstance(response, dict):
        choices = response.get("choices") or []
        content = (choices[0].get("message") or {}).get("content") if choices else None
    else:  # Offline test doubles; live responses always retain the SDK envelope.
        content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block["text"] for block in content
                       if isinstance(block, dict) and block.get("type") == "text"
                       and isinstance(block.get("text"), str))
    return ""


def response_details(response):
    if not isinstance(response, dict):
        return {}
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    usage = response.get("usage") or {}
    return {"response_envelope": response, "request_id": response.get("id"),
            "finish_reason": choice.get("finish_reason"),
            "native_finish_reason": choice.get("native_finish_reason"),
            "usage": usage, "reported_cost_usd": usage.get("cost"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}
