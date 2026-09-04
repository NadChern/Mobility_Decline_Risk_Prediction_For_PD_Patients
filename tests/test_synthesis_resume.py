"""Offline regression tests: never contact an LLM or modify saved study outputs."""

import json
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from template_experiment import generate_dataset_a as dataset_a
from template_experiment.experimental import flexibility_stress_test as dataset_c
from template_experiment.evaluation.reproducibility import deterministic_flex_hash
from template_experiment.run_support import (
    collect_responses, heartbeat, output_lock, read_jsonl, write_jsonl,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        result = next(self.responses)
        if isinstance(result, BaseException):
            raise result
        return SimpleNamespace(content=result)


def forbid_client(*args, **kwargs):
    raise AssertionError("A saved response must not cause another API call")


@pytest.fixture
def small_a(tmp_path, monkeypatch):
    pilot = tmp_path / "pilot.csv"
    pd.read_csv(dataset_a.PILOT_DATASET, keep_default_na=False).head(3).to_csv(pilot, index=False)
    monkeypatch.setattr(dataset_a, "PILOT_DATASET", pilot)
    monkeypatch.setattr(dataset_a, "freeze_protocol", lambda: None)
    output = tmp_path / "a"
    prepared = dataset_a.run(output_dir=output)
    replies = [r["explanation"] for r in prepared if r["method"] == "domain_template"]
    return output, replies


@pytest.mark.parametrize("failure", [TimeoutError("test"), KeyboardInterrupt()])
def test_dataset_a_resumes_only_missing_requests(small_a, monkeypatch, capsys, failure):
    output, replies = small_a
    client = FakeClient([replies[0], failure])
    monkeypatch.setattr(dataset_a, "build_langchain_llm", lambda *a, **kw: client)
    with pytest.raises(type(failure)):
        dataset_a.run(True, output)
    rows = [r for r in read_jsonl(output / "records.jsonl") if r["method"] == "taxonomy_conditioned_llm"]
    assert [r["status"] for r in rows] == ["complete", "request_failed", "pending_api_generation"]
    assert rows[0]["raw_response"] == replies[0]
    assert rows[1]["attempts"][0]["error_type"] == type(failure).__name__
    client2 = FakeClient(replies[1:])
    monkeypatch.setattr(dataset_a, "build_langchain_llm", lambda *a, **kw: client2)
    resumed = dataset_a.run(True, output)
    assert len(client2.prompts) == 2
    assert client2.prompts == [r["prompt"] for r in rows[1:]]
    assert [r["patient_id"] for r in resumed] == [r["patient_id"] for r in read_jsonl(output / "records.jsonl")]
    assert "response received and saved" in capsys.readouterr().out
    monkeypatch.setattr(dataset_a, "build_langchain_llm", forbid_client)
    dataset_a.run(True, output)


def test_contract_failed_is_not_a_failed_api_call(small_a, monkeypatch):
    output, _ = small_a
    client = FakeClient(["Not a compliant report."] * 3)
    monkeypatch.setattr(dataset_a, "build_langchain_llm", lambda *a, **kw: client)
    first = dataset_a.run(True, output)
    assert sum(r["status"] == "contract_failed" for r in first) == 3
    monkeypatch.setattr(dataset_a, "build_langchain_llm", forbid_client)
    second = dataset_a.run(True, output)
    assert first == second
    assert "contract_failed: 3" in (output / "status.md").read_text()


def test_dataset_a_rejects_changed_prompt_without_overwriting(small_a, monkeypatch):
    output, _ = small_a
    original = (output / "records.jsonl").read_bytes()
    build = dataset_a.build_taxonomy_conditioned_prompt
    monkeypatch.setattr(dataset_a, "build_taxonomy_conditioned_prompt", lambda p: build(p) + "changed")
    monkeypatch.setattr(dataset_a, "build_langchain_llm", forbid_client)
    with pytest.raises(ValueError, match="Saved request differs"):
        dataset_a.run(True, output)
    assert original == (output / "records.jsonl").read_bytes()


def test_dataset_c_incremental_resume_and_legacy_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_c, "PROTOCOL_CASES", tmp_path / "protocol.json")
    output = tmp_path / "c"
    client = FakeClient(["first report", TimeoutError("test")])
    monkeypatch.setattr(dataset_c, "build_langchain_llm", lambda *a, **kw: client)
    with pytest.raises(TimeoutError):
        dataset_c.run(output, True)
    rows = read_jsonl(output / "generations.jsonl")
    jobs = [r for r in rows if r["method"] == "map_free_gemini"]
    assert jobs[0]["text"] == "first report"
    assert jobs[1]["status"] == "request_failed"
    assert len(rows) == 18
    client2 = FakeClient(["other report"] * 5)
    monkeypatch.setattr(dataset_c, "build_langchain_llm", lambda *a, **kw: client2)
    result = dataset_c.run(output, True)
    assert len(client2.prompts) == 5
    assert result.status.eq("complete").all()
    legacy = read_jsonl(output / "generations.jsonl")
    for row in legacy:
        if row["method"] == "map_free_gemini":
            for key in ("status", "raw_response", "response_received", "temperature"):
                row.pop(key, None)
    write_jsonl(output / "generations.jsonl", legacy)
    monkeypatch.setattr(dataset_c, "build_langchain_llm", forbid_client)
    assert dataset_c.run(output, True).status.eq("complete").all()


def test_raw_response_survives_local_processing_error(tmp_path):
    jobs = [{"id": "one", "prompt": "prompt", "status": "pending_api_generation"}]
    path = tmp_path / "checkpoint.jsonl"

    def fail(row):
        assert read_jsonl(path)[0]["raw_response"] == "paid output"
        raise ValueError("parser failed")

    kwargs = dict(extract=lambda r: r.content, checkpoint=lambda: write_jsonl(path, jobs),
                  label="Test", identity_field="id")
    with pytest.raises(ValueError):
        collect_responses(jobs, client_factory=lambda: FakeClient(["paid output"]), finalize=fail, **kwargs)
    assert read_jsonl(path)[0]["status"] == "processing_failed"
    collect_responses(jobs, client_factory=forbid_client,
                      finalize=lambda r: r.update(status="complete"), **kwargs)
    assert len(jobs[0]["attempts"]) == 1


def test_empty_received_response_is_not_retried(tmp_path):
    jobs = [{"id": "empty", "prompt": "prompt", "status": "response_received",
             "raw_response": "", "response_received": True}]
    collect_responses(jobs, client_factory=forbid_client, extract=lambda r: r.content,
                      finalize=lambda r: r.update(status="contract_failed"),
                      checkpoint=lambda: write_jsonl(tmp_path / "raw.jsonl", jobs),
                      label="Test", identity_field="id")
    assert jobs[0]["status"] == "contract_failed"


def test_output_lock_prevents_duplicate_collectors(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another collection"):
            with output_lock(tmp_path):
                pass
    with output_lock(tmp_path):
        pass


def test_heartbeat_is_visible_and_stops(capsys):
    with heartbeat("patient=123", interval=0.01):
        time.sleep(0.05)
    assert "patient=123: still waiting for response" in capsys.readouterr().out
    time.sleep(0.03)
    assert capsys.readouterr().out == ""


def test_audit_normalizes_csv_types_but_detects_real_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_c, "PROTOCOL_CASES", tmp_path / "protocol.json")
    original = dataset_c.run(tmp_path / "c")
    serialized = pd.read_csv(tmp_path / "c/cases.csv", keep_default_na=False)
    deterministic = serialized[serialized.method != "map_free_gemini"].copy()
    deterministic["unsupported_content_count"] = 0
    assert deterministic_flex_hash(original) == deterministic_flex_hash(deterministic)
    deterministic.loc[deterministic.index[0], "unsupported_content_count"] = 1
    assert deterministic_flex_hash(original) != deterministic_flex_hash(deterministic)
