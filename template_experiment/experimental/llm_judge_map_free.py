"""Secondary two-model plausibility judging for explicit map-free disagreement claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from explanation.data_loader import (OPENROUTER_API_KEY, SYNTHESIS_JUDGE_MODELS,
                                     SYNTHESIS_JUDGE_REPETITIONS_PER_ORDER,
                                     SYNTHESIS_JUDGE_TEMPERATURE)
from .judge_runtime import (build_judge_client, collection_guard, extract_final_content,
                            panel_runtime_settings)

from ..synthesis_protocol import PROJECT_ROOT, PROTOCOL_DIR, canonical_json
from ..taxonomy import CATEGORY_LABELS
from ..run_support import progress, output_lock
from .judge_collection import collect_judgments, collection_complete, verify_frozen_requests


WORKSHEET = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/checker_v2_1/part2a_map_free/disagreement_worksheet.csv"
EVIDENCE = PROTOCOL_DIR / "dataset_b_evidence_packets.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/checker_v2_1/judge_part2_runtime_v3"


def build_prompt(group_label, members, evidence):
    return f"""Evaluate one anonymous grouping claim from a model explanation. You are not shown a
researcher reference grouping or a method identity. Judge the claim only from the supplied factor
names and their value/scale descriptions. This is model-based plausibility review, not clinical
validation.

CLAIM: {group_label} ({', '.join(members)})
FACTOR EVIDENCE:
{json.dumps(evidence, indent=2, ensure_ascii=False)}

Return only JSON:
{{"single_construct_coherence": 1, "category_label_fit": 1, "possible_over_grouping": 1,
 "representation": "keep_together|split|leave_individual", "rationale": "short rationale"}}

Use integers 1-5. For coherence and fit, 5 is strongest. For possible_over_grouping, 1 means no
sign and 5 means severe over-grouping.
"""


def prepare_items(worksheet_path=WORKSHEET):
    worksheet = pd.read_csv(worksheet_path, keep_default_na=False)
    packets = {item["patient_id"]: item for item in
               (json.loads(line) for line in EVIDENCE.read_text(encoding="utf-8").splitlines())}
    items = []
    for row in worksheet.itertuples(index=False):
        packet = packets[int(row.patient_id)]
        by_factor = {item["factor"]: item for item in packet["selected_factors"]}
        for claim_index, claim in enumerate(json.loads(row.map_free_groups), start=1):
            members = claim["members"]
            evidence = [by_factor[member] for member in members if member in by_factor]
            label = CATEGORY_LABELS.get(claim["group"], claim["group"])
            for judge in SYNTHESIS_JUDGE_MODELS:
                for repetition in range(1, SYNTHESIS_JUDGE_REPETITIONS_PER_ORDER + 1):
                    audit_id = f"MF-{row.patient_id}-{claim_index}-{repetition}-{hashlib.sha1(judge.encode()).hexdigest()[:6]}"
                    prompt = build_prompt(label, members, evidence)
                    items.append({"audit_id": audit_id, "patient_id": int(row.patient_id),
                                  "claim_index": claim_index, "judge_model": judge,
                                  "repetition": repetition, "group_label": label,
                                  "members": members, "judge_prompt": prompt,
                                  "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()})
    return items


def parse_response(text):
    try:
        value = json.loads(str(text).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    if value.get("representation") not in {"keep_together", "split", "leave_individual"}:
        return None
    for key in ("single_construct_coherence", "category_label_fit", "possible_over_grouping"):
        if type(value.get(key)) is not int or not 1 <= value[key] <= 5:
            return None
    return value if isinstance(value.get("rationale"), str) and value["rationale"].strip() else None


def run(output_dir=OUTPUT_DIR, execute=False, worksheet_path=WORKSHEET, request_timeout=120, max_calls=None):
    if request_timeout <= 0 or (max_calls is not None and max_calls < 0):
        raise ValueError("Timeout must be positive and max_calls nonnegative.")
    with collection_guard(output_dir, execute), output_lock(output_dir):
        return _run(Path(output_dir), execute, worksheet_path, request_timeout, max_calls)


def _run(output_dir, execute, worksheet_path, request_timeout, max_calls):
    items = prepare_items(worksheet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_profiles = {model: panel_runtime_settings(model, request_timeout)
                        for model in SYNTHESIS_JUDGE_MODELS}
    verify_frozen_requests(items, output_dir, SYNTHESIS_JUDGE_TEMPERATURE,
                           lambda model: panel_runtime_settings(model, request_timeout))
    (output_dir / "runtime_profiles.json").write_text(
        json.dumps(runtime_profiles, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(items).to_json(output_dir / "blinded_groups.jsonl", orient="records", lines=True,
                                force_ascii=False)
    if not execute and (output_dir / "raw_judgments.jsonl").exists():
        progress("Existing map-free judge collection retained; no API calls made. See collection_status.md.")
        return items, None
    if not execute:
        (output_dir / "report.md").write_text(
            "# Part 2 map-free claim plausibility judge\n\n"
            f"Prepared {len(items)} blinded claim calls. No reference group or method identity is "
            "included. Run with `--execute` after setting `OPENROUTER_API_KEY`. Results remain "
            "separate from Part 1 narrative judging and cannot establish clinical correctness.\n",
            encoding="utf-8")
        return items, None
    raw = collect_judgments(items, output_dir, temperature=SYNTHESIS_JUDGE_TEMPERATURE,
        client_factory=lambda model: build_judge_client(model, SYNTHESIS_JUDGE_TEMPERATURE, request_timeout),
        extract=extract_final_content, parse=parse_response, label="Map-free judges", max_calls=max_calls,
        runtime_config=lambda model: panel_runtime_settings(model, request_timeout))
    if not collection_complete(raw) or not raw:
        (output_dir / "report.md").write_text(
            "# Part 2 map-free claim plausibility judge\n\nNo final summary: collection is empty, "
            "partial, or contains invalid responses. See collection_status.md. "
            "Saved invalid responses require review, not automatic resampling.\n", encoding="utf-8")
        return items, None
    parsed = pd.DataFrame([{**next(i for i in items if i["audit_id"] == row["audit_id"]),
                            **row["parsed"]} for row in raw if row["parsed"]])
    summary = parsed.groupby("group_label")[["single_construct_coherence", "category_label_fit",
                                              "possible_over_grouping"]].agg(["mean", "median"])
    summary.to_csv(output_dir / "summary.csv")
    (output_dir / "report.md").write_text(
        "# Part 2 map-free claim plausibility judge\n\nCompleted secondary model-based ratings. "
        "These results are separate from Part 1 and do not establish clinical correctness.\n",
        encoding="utf-8")
    return items, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worksheet", default=str(WORKSHEET))
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--max-calls", type=int, help="Maximum new API calls in this run.")
    args = parser.parse_args()
    run(args.output_dir, args.execute, args.worksheet, args.request_timeout, args.max_calls)
