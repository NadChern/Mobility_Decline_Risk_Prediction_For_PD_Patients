import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from template_experiment import run_judge_retest as retest
from template_experiment.experimental import judge_runtime as runtime
from template_experiment.experimental.llm_judge import RUBRIC


def test_retest_changes_only_reasoning_and_profile_version():
    item, temperature, settings, provenance = retest.prepare()
    old = provenance["original_runtime_config"]
    differences = {k for k in old.keys() | settings.keys() if old.get(k) != settings.get(k)}
    assert differences == {"reasoning", "profile_version"}
    assert settings["reasoning"] == {"effort": "low"}
    assert temperature == provenance["temperature"] and item["prompt_hash"] == provenance["prompt_hash"]


def test_low_effort_reaches_sdk(monkeypatch):
    send = AsyncMock(return_value=SimpleNamespace(model_dump=lambda **kw: {}))
    monkeypatch.setattr(runtime, "OPENROUTER_API_KEY", "test-only-key")
    monkeypatch.setattr(runtime, "OpenRouter", lambda **kw: SimpleNamespace(chat=SimpleNamespace(send_async=send)))
    runtime.JudgeClient("test", 0, runtime.runtime_settings(reasoning={"effort": "low"})).invoke("test")
    assert send.call_args.kwargs["reasoning"] == {"effort": "low"}
    assert runtime.runtime_settings()["reasoning"] is None


def test_retest_one_call_and_preserves_source_and_success(tmp_path):
    before = (retest.SOURCE / "raw_judgments.jsonl").read_bytes()
    result = {"scores_a": {k: 3 for k in RUBRIC}, "scores_b": {k: 3 for k in RUBRIC},
              "preference": "tie", "rationale": "Test response"}
    calls = []
    def factory(model):
        calls.append(model)
        return SimpleNamespace(invoke=lambda p: SimpleNamespace(content=json.dumps(result)))
    row = retest.run(True, output_dir=tmp_path, client_factory=factory)
    assert row["status"] == "complete" and len(calls) == 1
    retest.run(True, output_dir=tmp_path, client_factory=lambda m: pytest.fail("must reuse saved reply"))
    assert before == (retest.SOURCE / "raw_judgments.jsonl").read_bytes()


def test_cannot_overwrite_original():
    with pytest.raises(ValueError, match="must not overwrite"):
        retest.run(output_dir=retest.SOURCE)
