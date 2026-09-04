"""Shared durable judge collection. Public runners hold output_lock around the whole run."""

from collections import Counter
from pathlib import Path

from ..run_support import (atomic_text, collect_responses, indexed, progress, read_jsonl,
                           verify_identity, write_jsonl, _received)
from ..synthesis_protocol import sha256_text
from .judge_runtime import response_details


def request_manifest(items, temperature, runtime_config=None):
    """Accept shared settings or a resolver callable taking the judge model ID."""
    rows = []
    for item in items:
        if item.get("prompt_hash") != sha256_text(item["judge_prompt"]):
            raise ValueError(f"Prompt hash is inconsistent for {item['audit_id']}")
        settings = runtime_config(item["judge_model"]) if callable(runtime_config) else runtime_config
        rows.append({**item, "temperature": temperature,
                     **({"runtime_config": settings} if settings is not None else {})})
    indexed(rows, ("audit_id",))
    return rows


def verify_frozen_requests(items, output_dir, temperature, runtime_config=None):
    """Verify before writing derived artifacts or charging for a changed experiment."""
    expected = request_manifest(items, temperature, runtime_config)
    frozen = Path(output_dir) / "frozen_requests.jsonl"
    if frozen.exists():
        if read_jsonl(frozen) != expected:
            raise ValueError("Judge prompts, models, order, temperature, runtime settings, or cohort changed. "
                             "Use a new output directory; previous judgments are untouched.")
    elif (Path(output_dir) / "raw_judgments.jsonl").exists():
        raise ValueError("Legacy judge responses have no frozen request manifest. Preserve them "
                         "and reconcile their provenance before attempting a resume.")
    return expected


def collect_judgments(items, output_dir, *, temperature, client_factory, extract, parse,
                      label, max_calls=None, runtime_config=None):
    """Checkpoint each paid response before parsing; invalid JSON is retained, not resampled."""
    if max_calls is not None and max_calls < 0:
        raise ValueError("max_calls must be nonnegative.")
    output_dir = Path(output_dir)
    expected = verify_frozen_requests(items, output_dir, temperature, runtime_config)
    if not (output_dir / "frozen_requests.jsonl").exists():
        write_jsonl(output_dir / "frozen_requests.jsonl", expected)
    prior = indexed(read_jsonl(output_dir / "raw_judgments.jsonl"), ("audit_id",))
    jobs = []
    for item in expected:
        row = {key: value for key, value in item.items()
               if key in {"audit_id", "patient_id", "judge_model", "order", "repetition", "prompt_hash", "temperature"}}
        row.update(status="pending_api_generation", raw_response="", parsed=None, error="")
        old = prior.pop((item["audit_id"],), None)
        if old is not None:
            verify_identity(old, row, ("audit_id", "judge_model", "prompt_hash", "temperature", "order", "repetition"))
            row = old
        row["prompt"] = item["judge_prompt"]
        jobs.append(row)
    if prior:
        raise ValueError("Judge checkpoint contains requests outside the frozen manifest.")

    def checkpoint():
        write_jsonl(output_dir / "raw_judgments.jsonl",
                    [{k: v for k, v in row.items() if k != "prompt"} for row in jobs])
        counts = Counter(row["status"] for row in jobs)
        costs = [r["reported_cost_usd"] for r in jobs if isinstance(r.get("reported_cost_usd"), (int, float))]
        atomic_text(output_dir / "collection_status.md",
            f"# {label} collection status\n\nExpected calls: {len(jobs)}.\n\n"
            + "\n".join(f"- {state}: {count}" for state, count in sorted(counts.items()))
            + f"\n\nProvider-reported cost: ${sum(costs):.6f} across {len(costs)} responses with cost metadata. "
            "Missing costs and interrupted requests are unknown, not zero.\n"
            + "\n\nRaw responses are saved before parsing. Invalid structured outputs are retained "
            "as contract_failed and are not automatically regenerated. Retry the same command "
            "to resume missing/failed requests. Attempt history is in raw_judgments.jsonl.\n")

    def finalize(row):
        finish = row.get("finish_reason")
        if finish in {"length", "max_tokens"}:
            row["parsed"], row["error"] = None, "truncated_output"
        elif finish not in {None, "stop"}:
            row["parsed"], row["error"] = None, "non_final_response"
        elif not row["raw_response"].strip():
            row["parsed"], row["error"] = None, "empty_final_content"
        else:
            row["parsed"] = parse(row["raw_response"])
            row["error"] = "" if row["parsed"] is not None else "invalid_structured_output"
        row["status"] = "complete" if row["parsed"] is not None else "contract_failed"
        row["timestamp_utc"] = row.get("generated_utc", "")

    missing = [r for r in jobs if not _received(r)]
    # Round-robin selection covers both models before spending the cap on repetitions.
    queues = [[r for r in missing if r["judge_model"] == model]
              for model in dict.fromkeys(r["judge_model"] for r in missing)]
    balanced = []
    while any(queues):
        for queue in queues:
            if queue:
                balanced.append(queue.pop(0))
    allowed = {r["audit_id"] for r in (missing if max_calls is None else balanced[:max_calls])}
    progress(f"{label}: {len(jobs)} total; {len(missing)} missing responses; "
             f"at most {len(allowed)} new calls this run")
    checkpoint()
    for model in dict.fromkeys(r["judge_model"] for r in jobs):
        selected = [r for r in jobs if r["judge_model"] == model
                    and (_received(r) or r["audit_id"] in allowed)]
        collect_responses(selected, client_factory=lambda: client_factory(model), extract=extract,
                          finalize=finalize, checkpoint=checkpoint,
                          label=f"{label} / {model}", identity_field="audit_id",
                          response_details=response_details)
    return [{k: v for k, v in row.items() if k != "prompt"} for row in jobs]


def collection_complete(rows):
    return all(row.get("status") == "complete" for row in rows)
