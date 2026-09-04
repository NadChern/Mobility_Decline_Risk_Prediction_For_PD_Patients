import json
from types import SimpleNamespace

import pytest

from template_experiment.experimental.judge_collection import collect_judgments
from template_experiment.experimental import llm_judge as paired
from template_experiment.experimental import llm_judge_map_free as map_free
from template_experiment.run_support import read_jsonl
from template_experiment.synthesis_protocol import sha256_text


def items():
    return [{"audit_id": f"call-{i}", "patient_id": i, "judge_model": model,
             "order": "primary", "repetition": 1, "judge_prompt": f"prompt-{i}",
             "prompt_hash": sha256_text(f"prompt-{i}")}
            for i, model in enumerate(["judge-1", "judge-1", "judge-2"])]


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        response = next(self.replies)
        if isinstance(response, BaseException):
            raise response
        return SimpleNamespace(content=response)


def collect(path, factory, requests=None, max_calls=None, parse=None):
    return collect_judgments(items() if requests is None else requests, path, temperature=0,
        client_factory=factory, extract=lambda r: r.content,
        parse=parse or (lambda r: {"valid": True} if r == "valid" else None),
        label="Test judges", max_calls=max_calls)


def forbidden(*args, **kwargs):
    raise AssertionError("No repeated API call allowed")


@pytest.mark.parametrize("failure", [TimeoutError("offline"), KeyboardInterrupt()])
def test_judge_interruption_is_durable_and_resumable(tmp_path, failure):
    first = Client(["valid", failure])
    with pytest.raises(type(failure)):
        collect(tmp_path, lambda model: first)
    rows = read_jsonl(tmp_path / "raw_judgments.jsonl")
    assert [r["status"] for r in rows] == ["complete", "request_failed", "pending_api_generation"]
    second, other = Client(["valid"]), Client(["valid"])
    resumed = collect(tmp_path, lambda m: second if m == "judge-1" else other)
    assert second.calls == ["prompt-1"] and other.calls == ["prompt-2"]
    assert all(r["parsed"] for r in resumed)
    collect(tmp_path, forbidden)


def test_budget_caps_new_calls_and_does_not_exclude_remaining_requests(tmp_path):
    client = Client(["valid"])
    rows = collect(tmp_path, lambda m: client, max_calls=1)
    assert len(client.calls) == 1
    assert sum(r["status"] == "pending_api_generation" for r in rows) == 2
    collect(tmp_path, forbidden, max_calls=0)


def test_invalid_json_is_retained_not_automatically_regenerated(tmp_path):
    rows = collect(tmp_path, lambda m: Client(["invalid", "invalid"]))
    assert all(r["status"] == "contract_failed" for r in rows)
    assert all(r["error"] == "invalid_structured_output" for r in rows)
    assert collect(tmp_path, forbidden) == rows


def test_changed_prompt_is_rejected_before_existing_artifacts_change(tmp_path):
    collect(tmp_path, lambda m: Client(["valid"]), max_calls=1)
    before = (tmp_path / "raw_judgments.jsonl").read_bytes()
    changed = items()
    changed[1]["judge_prompt"] = "changed"
    changed[1]["prompt_hash"] = sha256_text("changed")
    with pytest.raises(ValueError, match="changed"):
        collect(tmp_path, forbidden, changed)
    assert before == (tmp_path / "raw_judgments.jsonl").read_bytes()


def test_local_parse_failure_reuses_paid_raw_response(tmp_path):
    def broken(text):
        assert read_jsonl(tmp_path / "raw_judgments.jsonl")[0]["raw_response"] == "valid"
        raise ValueError("local bug")
    with pytest.raises(ValueError, match="local bug"):
        collect(tmp_path, lambda m: Client(["valid"]), max_calls=1, parse=broken)
    rows = collect(tmp_path, forbidden, max_calls=0)
    assert rows[0]["parsed"] and len(rows[0]["attempts"]) == 1


@pytest.mark.parametrize("payload", ["null", "[]", '"text"', "42"])
def test_judge_parsers_reject_non_objects_without_crashing(payload):
    assert paired.parse_judgment(payload) is None
    assert map_free.parse_response(payload) is None


def test_judge_parsers_reject_boolean_scores():
    payload = {"scores_a": {k: True for k in paired.RUBRIC},
               "scores_b": {k: 3 for k in paired.RUBRIC}, "preference": "A", "rationale": "test"}
    assert paired.parse_judgment(json.dumps(payload)) is None
    payload = {"single_construct_coherence": True, "category_label_fit": 3,
               "possible_over_grouping": 2, "representation": "keep_together", "rationale": "test"}
    assert map_free.parse_response(json.dumps(payload)) is None


def test_map_free_all_invalid_responses_do_not_crash_or_summarize(tmp_path, monkeypatch):
    monkeypatch.setattr(map_free, "prepare_items", lambda path: items())
    monkeypatch.setattr(map_free, "build_judge_client", lambda *a, **kw: Client(["invalid", "invalid"]))
    _, summary = map_free.run(tmp_path, execute=True)
    assert summary is None and not (tmp_path / "summary.csv").exists()
    before = (tmp_path / "report.md").read_bytes()
    map_free.run(tmp_path, execute=False)
    assert before == (tmp_path / "report.md").read_bytes()
