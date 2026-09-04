"""Prepare or resume Dataset A without resampling received Gemini responses."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from explanation.data_loader import GOOGLE_MODEL, LLM_TEMPERATURE
from explanation.explanation_builder import build_patient_explanation_data_full, get_patient_index
from explanation.llm import _build_prompt, _extract_text_content, build_langchain_llm

from .domain_template import GROUPED_METHOD_NAME, render_template_grouped_explanation
from .evaluation.shared import canonical_labels, parse_report
from .evaluation.interpretation_contract import CHECKER_VERSION
from .run_support import (atomic_text, collect_responses, indexed, output_lock, progress,
                          read_jsonl, verify_identity, write_jsonl)
from .synthesis_protocol import (
    PILOT_DATASET, PROJECT_ROOT, PROTOCOL_VERSION, build_taxonomy_conditioned_prompt,
    evidence_packet, freeze_protocol, sha256_text, validate_interpretation_contract,
)

OUTPUT_DIR = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/dataset_a"
LLM_METHOD = "taxonomy-conditioned-gemini-v1"


def _record(packet, method, version, interpretation, explanation, status, **metadata):
    return {
        "dataset": "A", **packet, "method": method, "model_or_template_version": version,
        "interpretation": interpretation, "explanation": explanation, "status": status, **metadata,
    }


def run(generate=False, output_dir=OUTPUT_DIR, request_timeout=120):
    if request_timeout <= 0:
        raise ValueError("Request timeout must be positive.")
    with output_lock(output_dir):
        return _run(generate, Path(output_dir), request_timeout)


def _run(generate, output_dir, request_timeout):
    freeze_protocol()
    prior = indexed(read_jsonl(output_dir / "records.jsonl"), ("patient_id", "method"))
    paired = pd.read_csv(PILOT_DATASET, keep_default_na=False)
    norm_set, norm_to_short = canonical_labels()
    records, patients = [], {}
    old_times = {}
    if (output_dir / "generation_log.csv").exists():
        old_times = pd.read_csv(output_dir / "generation_log.csv", keep_default_na=False).set_index(
            "patient_id")["generated_utc"].to_dict()

    progress("Dataset A: preparing paired prompts and checking saved request identities")
    for source in paired.itertuples(index=False):
        patient_info = build_patient_explanation_data_full(get_patient_index(int(source.patient_id)))
        packet = evidence_packet(patient_info)
        patients[packet["patient_id"]] = patient_info
        template_text = render_template_grouped_explanation(patient_info)
        template_interpretation = parse_report(template_text, norm_set, norm_to_short)["interpretation"]
        template_errors = validate_interpretation_contract(template_interpretation, patient_info)
        if template_errors:
            raise ValueError(f"Domain-template contract failure for {packet['patient_id']}: {template_errors}")
        template = _record(
            packet, "domain_template", GROUPED_METHOD_NAME, template_interpretation,
            template_text, "complete", protocol_version=PROTOCOL_VERSION, contract_errors=[],
        )
        prompt = build_taxonomy_conditioned_prompt(patient_info)
        record = _record(
            packet, "taxonomy_conditioned_llm", GOOGLE_MODEL, "", "", "pending_api_generation",
            protocol_version=PROTOCOL_VERSION, provider="google", temperature=LLM_TEMPERATURE,
            prompt=prompt, prompt_hash=sha256_text(prompt), raw_response="",
            base_map_free_prompt_hash=sha256_text(_build_prompt(patient_info)),
            prompt_difference="taxonomy mapping block + one taxonomy-use instruction",
            contract_errors=[],
        )
        for expected in (template, record):
            previous = prior.pop((packet["patient_id"], expected["method"]), None)
            if previous is not None:
                fields = ("evidence_hash", "model_or_template_version", "protocol_version")
                if expected["method"] == "taxonomy_conditioned_llm":
                    fields += ("prompt", "prompt_hash", "temperature", "provider")
                else:
                    fields += ("explanation",)
                verify_identity(previous, expected, fields)
                expected = previous
            records.append(expected)
    if prior:
        raise ValueError("Checkpoint contains patients/methods outside the current paired cohort; refusing overwrite.")
    jobs = [row for row in records if row["method"] == "taxonomy_conditioned_llm"]
    for row in jobs:
        if old_times.get(row["patient_id"]) and not row.get("generated_utc"):
            row["generated_utc"] = old_times[row["patient_id"]]

    def checkpoint():
        write_jsonl(output_dir / "records.jsonl", records)
        logs = [{
            "patient_id": r["patient_id"], "evidence_hash": r["evidence_hash"],
            "model": r["model_or_template_version"], "temperature": r["temperature"],
            "prompt_hash": r["prompt_hash"], "base_map_free_prompt_hash": r["base_map_free_prompt_hash"],
            "prompt_difference": r["prompt_difference"], "status": r["status"],
            "generated_utc": r.get("generated_utc", ""), "attempts": len(r.get("attempts", [])),
            "error": r.get("error", ""),
        } for r in jobs]
        atomic_text(output_dir / "generation_log.csv", pd.DataFrame(logs).to_csv(index=False))
        counts = pd.Series([r["status"] for r in jobs]).value_counts().to_dict()
        atomic_text(output_dir / "status.md",
            "# Dataset A status\n\n"
            f"Protocol: `{PROTOCOL_VERSION}`. Domain-template records: {len(jobs)} complete.\n\n"
            "Generation-time statuses (preserved; later checker revisions are audited separately):\n\n"
            + "\n".join(f"- {status}: {count}" for status, count in sorted(counts.items()))
            + "\n\nContract-failed responses were received and saved; they are not API failures. "
            "The latest checker-v2.1 assessments are in `../checker_v2_1/report.md`; "
            "rechecking does not rewrite these original generation flags. "
            "Rerunning `python -m template_experiment.run_synthesis_v2 --generate` resumes missing/failed "
            "requests and reprocesses saved raw responses locally if necessary. "
            "Request attempt history is in `records.jsonl`.\n")

    def finalize(row):
        row["explanation"] = row["raw_response"]
        row["interpretation"] = parse_report(row["explanation"], norm_set, norm_to_short)["interpretation"]
        row["contract_errors"] = validate_interpretation_contract(row["interpretation"], patients[row["patient_id"]])
        row["checker_version"] = CHECKER_VERSION
        row["status"] = "contract_failed" if row["contract_errors"] else "complete"
        row["error"] = ""

    checkpoint()
    if generate:
        progress(f"Dataset A: google / {GOOGLE_MODEL}; request timeout {request_timeout}s; SDK retries disabled")
        collect_responses(jobs,
            client_factory=lambda: build_langchain_llm("google", GOOGLE_MODEL, LLM_TEMPERATURE,
                                                       request_timeout=request_timeout, max_retries=0),
            extract=lambda response: _extract_text_content(response).strip(), finalize=finalize,
            checkpoint=checkpoint, label="Dataset A", identity_field="patient_id")
    progress(f"Dataset A: checkpoint at {output_dir / 'records.jsonl'}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--request-timeout", type=int, default=120)
    args = parser.parse_args()
    run(args.generate, args.output_dir, args.request_timeout)


if __name__ == "__main__":
    main()
