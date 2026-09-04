import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from template_experiment.experimental import judge_runtime as runtime
from template_experiment.experimental.judge_collection import collect_judgments
from template_experiment.run_support import read_jsonl
from template_experiment.synthesis_protocol import sha256_text


def requests():
    return [{"audit_id": str(i), "judge_model": model, "judge_prompt": "test",
             "prompt_hash": sha256_text("test")} for i, model in enumerate(["a", "a", "b", "b"])]


def test_sdk_timeout_conversion_and_retries_disabled_at_both_levels(monkeypatch):
    captured = {}
    send = AsyncMock(return_value=SimpleNamespace(model_dump=lambda **kw: {"choices": []}))
    def sdk(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(send_async=send))
    monkeypatch.setattr(runtime, "OPENROUTER_API_KEY", "fake-test-key")
    monkeypatch.setattr(runtime, "OpenRouter", sdk)
    runtime.build_judge_client("test", 0, 90).invoke("prompt")
    assert captured["timeout_ms"] == 90000
    assert "retry_config" in captured and captured["retry_config"] is None
    assert send.call_count == 1
    kwargs = send.call_args.kwargs
    assert kwargs["timeout_ms"] == 90000 and kwargs["retries"] is None
    assert kwargs["max_tokens"] == 2048


def test_full_panel_resolves_model_specific_reasoning():
    glm = runtime.panel_runtime_settings("z-ai/glm-5.3-flash")
    qwen = runtime.panel_runtime_settings("qwen/qwen3.8-flash")
    assert glm["reasoning"] == {"effort": "low"}
    assert qwen["reasoning"] is None
    assert glm["max_tokens"] == qwen["max_tokens"] == 2048


def test_manifest_freezes_resolved_settings_per_model():
    from template_experiment.experimental.judge_collection import request_manifest
    rows = request_manifest(
        requests(), 0, lambda model: runtime.panel_runtime_settings(model)
    )
    # Unknown test-model IDs both correctly inherit the global defaults.
    assert all(row["runtime_config"]["reasoning"] is None for row in rows)


def test_wall_clock_deadline_cancels_request_without_lingering_task(monkeypatch):
    monkeypatch.setattr(runtime, "OPENROUTER_API_KEY", "fake-test-key")
    client = runtime.build_judge_client("test", 0, .02)
    cancelled = []
    async def slow(prompt):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
    monkeypatch.setattr(client, "_request", slow)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        client.invoke("test")
    assert cancelled == [True] and time.monotonic() - started < 1


@pytest.mark.parametrize("finish,content,error", [
    ("stop", None, "empty_final_content"),
    ("length", '{"valid": true}', "truncated_output"),
    ("stop", "not json", "invalid_structured_output"),
    ("tool_calls", "{}", "non_final_response"),
])
def test_response_envelope_saved_and_never_resampled(tmp_path, finish, content, error):
    response = {"id": "request-1", "choices": [{"finish_reason": finish,
        "message": {"content": content, "reasoning": '{"valid": true}'}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 200, "cost": .001,
                  "completion_tokens_details": {"reasoning_tokens": 150}}}
    def parse(text):
        try: return json.loads(text)
        except ValueError: return None
    kwargs = dict(temperature=0, extract=runtime.extract_final_content, parse=parse, label="Test", max_calls=1)
    rows = collect_judgments(requests(), tmp_path,
        client_factory=lambda m: SimpleNamespace(invoke=lambda p: response), **kwargs)
    row = rows[0]
    assert row["error"] == error and row["parsed"] is None
    assert row["response_envelope"] == response
    assert row["reported_cost_usd"] == .001 and row["reasoning_tokens"] == 150
    assert read_jsonl(tmp_path / "raw_judgments.jsonl")[0]["request_id"] == "request-1"
    kwargs["max_calls"] = 0
    collect_judgments(requests(), tmp_path,
        client_factory=lambda m: pytest.fail("must not resample"), **kwargs)


def test_small_cap_covers_both_models_and_settings_are_frozen(tmp_path):
    models = []
    def factory(model):
        models.append(model)
        return SimpleNamespace(invoke=lambda p: SimpleNamespace(content="valid"))
    kwargs = dict(temperature=0, client_factory=factory, extract=runtime.extract_final_content,
                  parse=lambda t: {"valid": True}, label="Test", max_calls=2,
                  runtime_config=runtime.runtime_settings())
    collect_judgments(requests(), tmp_path, **kwargs)
    assert models == ["a", "b"]
    before = (tmp_path / "raw_judgments.jsonl").read_bytes()
    kwargs["runtime_config"] = {**kwargs["runtime_config"], "max_tokens": 4096}
    with pytest.raises(ValueError, match="runtime settings"):
        collect_judgments(requests(), tmp_path, **kwargs)
    assert (tmp_path / "raw_judgments.jsonl").read_bytes() == before


def test_running_legacy_collector_blocks_new_version(tmp_path, monkeypatch):
    from template_experiment.run_support import output_lock
    monkeypatch.setattr(runtime, "STUDY_OUTPUT", tmp_path)
    with output_lock(tmp_path / "judge_part2"):
        with pytest.raises(RuntimeError, match="still running"):
            with runtime.collection_guard(tmp_path / "judge_part2_runtime_v2", True):
                pytest.fail("would permit duplicate billing")
        with runtime.collection_guard(tmp_path / "judge_part2_runtime_v2", False):
            pass  # Offline preparation remains possible.


def test_real_sdk_makes_one_transport_attempt_with_no_retries(monkeypatch):
    import httpx
    attempts = []
    async def handler(request):
        attempts.append(request)
        raise httpx.ReadTimeout("offline test", request=request)
    original_client = httpx.AsyncClient
    monkeypatch.setattr(runtime, "OPENROUTER_API_KEY", "fake-test-key")
    monkeypatch.setattr(runtime.httpx, "AsyncClient", lambda **kw:
        original_client(timeout=kw["timeout"], transport=httpx.MockTransport(handler)))
    with pytest.raises(httpx.ReadTimeout):
        runtime.build_judge_client("test", 0, 1).invoke("test")
    assert len(attempts) == 1
    assert attempts[0].extensions["timeout"]["read"] == 1.0
    payload = json.loads(attempts[0].content)
    assert payload["max_tokens"] == 2048 and payload["stream"] is False
