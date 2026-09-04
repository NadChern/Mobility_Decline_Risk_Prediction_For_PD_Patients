"""Single-call GLM paired-judge budget retest; preserve the original failed experiment."""

import argparse
import json
from pathlib import Path

from explanation.data_loader import SYNTHESIS_JUDGE_RETEST_PROFILES
from .experimental.judge_collection import collect_judgments, verify_frozen_requests
from .experimental.judge_runtime import (JudgeClient, collection_guard, extract_final_content,
                                        runtime_settings)
from .experimental.llm_judge import parse_judgment
from .run_support import atomic_text, output_lock, read_jsonl
from .synthesis_protocol import PROJECT_ROOT, sha256_file, sha256_text

BASE = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/history/judge_smoke_v2"
SOURCE = BASE / "judge_part1_runtime_v2"
OUTPUT = BASE / "judge_part1_glm_low_v1_retest"
AUDIT_ID = "J-3001-P-1-c81915"
PROFILE = "glm_paired_low_v1"


def prepare(source=SOURCE):
    profile = SYNTHESIS_JUDGE_RETEST_PROFILES[PROFILE]
    originals = [r for r in read_jsonl(source / "frozen_requests.jsonl") if r["audit_id"] == AUDIT_ID]
    outcomes = [r for r in read_jsonl(source / "raw_judgments.jsonl") if r["audit_id"] == AUDIT_ID]
    if len(originals) != 1 or len(outcomes) != 1 or outcomes[0].get("error") != "truncated_output":
        raise ValueError("Retest requires the single preserved truncated GLM paired request.")
    original = originals[0]
    if original["judge_model"] != profile["model"]:
        raise ValueError("Retest model must match the original request.")
    if sha256_text(original["judge_prompt"]) != original["prompt_hash"]:
        raise ValueError("Original prompt hash mismatch.")
    settings = runtime_settings(profile["request_timeout"], max_tokens=profile["max_tokens"],
                                reasoning=profile["reasoning"])
    settings["profile_version"] = PROFILE
    item = {k: v for k, v in original.items() if k not in {"runtime_config", "temperature"}}
    provenance = {"scope": "one-call paired-judge smoke retest, not a full panel",
                  "source": str(source), "audit_id": AUDIT_ID,
                  "source_requests_sha256": sha256_file(source / "frozen_requests.jsonl"),
                  "source_responses_sha256": sha256_file(source / "raw_judgments.jsonl"),
                  "original_runtime_config": original["runtime_config"],
                  "retest_runtime_config": settings, "prompt_hash": item["prompt_hash"],
                  "temperature": original["temperature"]}
    return item, original["temperature"], settings, provenance


def run(execute=False, output_dir=OUTPUT, source=SOURCE, client_factory=None):
    output_dir, source = Path(output_dir), Path(source)
    if output_dir.resolve() == source.resolve():
        raise ValueError("Retest must not overwrite its source collection.")
    item, temperature, settings, provenance = prepare(source)
    with collection_guard(output_dir, execute), output_lock(output_dir):
        verify_frozen_requests([item], output_dir, temperature, settings)
        provenance_path = output_dir / "retest_provenance.json"
        if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Retest provenance changed; preserve this directory and use a new version.")
        atomic_text(provenance_path, json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        if not execute:
            print(f"Prepared one GLM paired retest: {settings}. No API call made.")
            return None
        rows = collect_judgments([item], output_dir, temperature=temperature,
            client_factory=client_factory or (lambda model: JudgeClient(model, temperature, settings)),
            extract=extract_final_content, parse=parse_judgment, label="GLM low-effort paired retest",
            max_calls=1, runtime_config=settings)
        row = rows[0]
        result = {k: row.get(k) for k in ("audit_id", "status", "error", "finish_reason",
                   "request_id", "usage", "reported_cost_usd", "attempts")}
        atomic_text(output_dir / "report.md", "# GLM paired low-effort smoke retest\n\n"
            "Same patient, evidence, prompt, presentation order, temperature, and 2,048-token cap. "
            "Reasoning changed from provider default to low; original responses are preserved.\n\n"
            + "```json\n" + json.dumps(result, indent=2) + "\n```\n\n"
            + "A valid result verifies this one request only. No full panel ran. Do not merge "
            "with the original collection as if both used identical runtime settings.\n")
        return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Allow at most one new paid request.")
    args = parser.parse_args()
    run(args.execute)
