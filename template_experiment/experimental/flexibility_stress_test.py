"""Prospective Dataset C task-configuration flexibility evaluation.

Case definitions and acceptable partitions are frozen before any LLM generation. Deterministic
arms always run; the map-free LLM arm is invoked only with ``--generate`` and a configured key.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from explanation.data_loader import GOOGLE_MODEL
from explanation.llm import _build_prompt, _extract_text_content, build_langchain_llm

from ..run_support import (atomic_text, collect_responses, indexed, output_lock, progress,
                           read_jsonl, verify_identity, write_jsonl)

from ..domain_template import render_template_grouped_explanation
from ..generic_template import render_template_explanation
from ..synthesis_protocol import PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/part2b_held_out"
PROTOCOL_CASES = PROJECT_ROOT / "template_experiment/protocol/held_out_cases.json"


def _factor(name, value, scale):
    return {"short_name": name, "display_value": value, "interpretation": scale,
            "displayable": True}


KNOWN = {
    "Gait": _factor("Gait", "2", "Clinician-rated gait impairment. 0-4; higher = worse"),
    "Postural Stability": _factor("Postural Stability", "2", "Postural-stability item. 0-4; higher = worse"),
    "Fainting": _factor("Fainting", "1", "Fainting frequency. 0=never; 4=very frequent"),
    "MoCA": _factor("MoCA", "24", "Montreal Cognitive Assessment. 0-30; higher = better cognition"),
    "GDS-15": _factor("GDS-15", "6", "Geriatric Depression Scale. 0-15; higher = more symptoms"),
}


def case_definitions():
    """Return frozen synthetic, non-patient held-out task configurations."""
    return [
        {"case_id": "C01_existing_gait_domain",
         "supporting": [KNOWN["Gait"], _factor("Timed Up-and-Go", "18.4", "Timed Up-and-Go seconds; higher = slower mobility")],
         "opposing": [KNOWN["MoCA"]], "unseen_features": ["Timed Up-and-Go"],
         "acceptable_groups": [["Gait", "Timed Up-and-Go"]],
         "target_domains": {"Timed Up-and-Go": "gait_mobility"}, "new_domain": False},
        {"case_id": "C02_existing_balance_domain",
         "supporting": [KNOWN["Postural Stability"], _factor("Berg Balance Scale", "38", "Berg Balance Scale. 0-56; lower = poorer balance")],
         "opposing": [KNOWN["GDS-15"]], "unseen_features": ["Berg Balance Scale"],
         "acceptable_groups": [["Postural Stability", "Berg Balance Scale"]],
         "target_domains": {"Berg Balance Scale": "gait_mobility"}, "new_domain": False},
        {"case_id": "C03_existing_autonomic_domain",
         "supporting": [KNOWN["Fainting"], _factor("Orthostatic BP Drop", "24 mmHg", "Systolic blood-pressure drop on standing; higher = larger drop")],
         "opposing": [KNOWN["MoCA"]], "unseen_features": ["Orthostatic BP Drop"],
         "acceptable_groups": [["Fainting", "Orthostatic BP Drop"]],
         "target_domains": {"Orthostatic BP Drop": "autonomic_symptoms"}, "new_domain": False},
        {"case_id": "C04_new_fall_concern_domain",
         "supporting": [_factor("Fear of Falling", "high", "Self-reported concern about falling; low/moderate/high"),
                        _factor("Falls Efficacy Scale", "31", "Falls Efficacy Scale; higher = greater concern")],
         "opposing": [KNOWN["MoCA"]],
         "unseen_features": ["Fear of Falling", "Falls Efficacy Scale"],
         "acceptable_groups": [["Fear of Falling", "Falls Efficacy Scale"]],
         "target_domains": {"Fear of Falling": "fall_concern", "Falls Efficacy Scale": "fall_concern"},
         "new_domain": True},
        {"case_id": "C05_novel_cross_domain_combination",
         "supporting": [KNOWN["MoCA"], KNOWN["GDS-15"]], "opposing": [KNOWN["Gait"]],
         "unseen_features": [], "acceptable_groups": [], "target_domains": {}, "new_domain": False},
        {"case_id": "C06_one_sided_unseen",
         "supporting": [KNOWN["Gait"], _factor("Timed Up-and-Go", "21.0", "Timed Up-and-Go seconds; higher = slower mobility")],
         "opposing": [], "unseen_features": ["Timed Up-and-Go"],
         "acceptable_groups": [["Gait", "Timed Up-and-Go"]],
         "target_domains": {"Timed Up-and-Go": "gait_mobility"}, "new_domain": False},
    ]


def _packet(case, index):
    return {"patient_id": 990000 + index, "final_prediction": 2,
            "has_severity_assessment": True,
            "severity_increasing_features": case["supporting"],
            "severity_decreasing_features": case["opposing"],
            "risk_increasing_features": [], "risk_decreasing_features": []}


def _interpretation(text):
    match = re.search(r"Model Interpretation\s*(.*?)\s*Clinical Note", str(text), re.S)
    return match.group(1).strip() if match else str(text).strip()


def _matching_parenthesis(text, opening):
    depth = 0
    for index in range(opening, len(text)):
        depth += int(text[index] == "(")
        depth -= int(text[index] == ")")
        if depth == 0:
            return index
    return None


def explicit_groups(text, factor_names):
    """Extract parenthesized >=2-member claims without requiring a known label lexicon."""
    prose = _interpretation(text)
    claims = []
    for opening, character in enumerate(prose):
        if character != "(":
            continue
        closing = _matching_parenthesis(prose, opening)
        if closing is None:
            continue
        members = sorted({name for name in factor_names if name in prose[opening + 1:closing]})
        if len(members) < 2:
            continue
        label_match = re.search(r"([A-Za-z][A-Za-z /-]{1,60})\s*$", prose[:opening])
        claims.append({"label": label_match.group(1).strip() if label_match else "",
                       "members": members, "span_start": opening, "span_end": closing + 1})
    return list({tuple(claim["members"]): claim for claim in claims}.values())


def _fidelity(text, packet):
    factors = packet["severity_increasing_features"] + packet["severity_decreasing_features"]
    return (sum(f["short_name"] in text and str(f["display_value"]) in text
                and f["interpretation"] in text for f in factors) / len(factors)) if factors else 1.0


def _evaluate(case, packet, method, text, status="complete"):
    names = [f["short_name"] for f in case["supporting"] + case["opposing"]]
    claims = explicit_groups(text, names) if status == "complete" else []
    predicted = {frozenset(claim["members"]) for claim in claims}
    acceptable = {frozenset(group) for group in case["acceptable_groups"]}
    grouping_success = acceptable <= predicted if acceptable else not predicted
    traceable = all(len(claim["members"]) >= 2 for claim in claims)
    fidelity = _fidelity(text, packet) if status == "complete" else None
    return {"case_id": case["case_id"], "method": method, "status": status,
            "renders_without_task_specific_edits": status == "complete",
            "grouping_success": grouping_success if status == "complete" else None,
            "compositional_fidelity": fidelity,
            "explicit_traceability": traceable if status == "complete" else None,
            "unsupported_content_count": 0 if fidelity == 1.0 else None,
            "per_case_success": bool(status == "complete" and grouping_success and traceable and fidelity == 1.0),
            "explicit_claims": json.dumps(claims, sort_keys=True)}


def _maintenance_rows(case):
    unseen = len(case["unseen_features"])
    common = {"prompt_edits": 0, "code_edits": 0, "parser_edits": 0,
              "regenerations": 0, "prospective_minutes": "not_yet_timed"}
    return [
        {"case_id": case["case_id"], "method": "generic_template", "map_edits": 0,
         "label_edits": 0, "corrections": 0, **common},
        {"case_id": case["case_id"], "method": "domain_template", "map_edits": unseen,
         "label_edits": int(bool(unseen and case["new_domain"])), "corrections": 0, **common},
        {"case_id": case["case_id"], "method": "map_free_gemini", "map_edits": 0,
         "label_edits": 0, "corrections": "pending_generation", **common},
    ]


def run(output_dir=OUTPUT_DIR, generate=False, request_timeout=120):
    if request_timeout <= 0:
        raise ValueError("Request timeout must be positive.")
    with output_lock(output_dir):
        return _run(Path(output_dir), generate, request_timeout)


def _run(output_dir, generate, request_timeout):
    cases = case_definitions()
    if PROTOCOL_CASES.exists():
        if json.loads(PROTOCOL_CASES.read_text(encoding="utf-8")) != cases:
            raise ValueError("Held-out cases differ from the frozen protocol; refusing to overwrite.")
    else:
        atomic_text(PROTOCOL_CASES, json.dumps(cases, indent=2))
    prior = indexed(read_jsonl(output_dir / "generations.jsonl"), ("case_id", "method"))
    # A legacy completed CSV without its raw checkpoint is unsafe to silently regenerate.
    if (output_dir / "cases.csv").exists():
        old = pd.read_csv(output_dir / "cases.csv", keep_default_na=False)
        for row in old.itertuples(index=False):
            if row.method == "map_free_gemini" and row.status == "complete":
                if (row.case_id, row.method) not in prior:
                    raise ValueError(f"Missing saved raw response for {row.case_id}; restore generations.jsonl.")
    generations, maintenance, packets = [], [], {}
    for index, case in enumerate(cases, start=1):
        packet = _packet(case, index)
        packets[case["case_id"]] = (case, packet)
        for method, renderer in (("generic_template", render_template_explanation),
                                 ("domain_template", render_template_grouped_explanation)):
            expected = {"case_id": case["case_id"], "method": method, "text": renderer(packet)}
            previous = prior.pop((case["case_id"], method), None)
            if previous is not None:
                verify_identity(previous, expected, ("text",))
            generations.append(previous if previous is not None else expected)
        expected = {"case_id": case["case_id"], "method": "map_free_gemini",
                    "prompt": _build_prompt(packet), "model": GOOGLE_MODEL,
                    "temperature": 0, "text": "", "status": "pending_api_generation"}
        previous = prior.pop((case["case_id"], "map_free_gemini"), None)
        if previous is not None:
            # Legacy records predate status/temperature fields; their saved text is a received response.
            previous.setdefault("temperature", 0)
            previous.setdefault("status", "complete" if "text" in previous else "pending_api_generation")
            verify_identity(previous, expected, ("prompt", "model", "temperature"))
            expected = previous
        generations.append(expected)
        maintenance.extend(_maintenance_rows(case))
    if prior:
        raise ValueError("Checkpoint contains cases/methods outside the frozen Dataset C; refusing overwrite.")
    jobs = [r for r in generations if r["method"] == "map_free_gemini"]

    def results():
        rows = []
        for row in generations:
            case, packet = packets[row["case_id"]]
            rows.append(_evaluate(case, packet, row["method"], row.get("text", ""),
                                  row.get("status", "complete")))
        return pd.DataFrame(rows)

    def checkpoint():
        write_jsonl(output_dir / "generations.jsonl", generations)
        frame = results()
        atomic_text(output_dir / "cases.csv", frame.to_csv(index=False))
        atomic_text(output_dir / "maintenance_log.csv", pd.DataFrame(maintenance).to_csv(index=False))
        counts = pd.Series([r["status"] for r in jobs]).value_counts().to_dict()
        atomic_text(output_dir / "report.md",
            "# Part 2B — prospective held-out flexibility\n\n"
            "Six case definitions and acceptable partitions were frozen before generation.\n\n"
            + "\n".join(f"- {status}: {count}" for status, count in sorted(counts.items()))
            + "\n\nEach received response is saved before local evaluation. Rerun the same "
            "`--generate` command to resume missing/failed requests.\n\n"
            "A case succeeds only when it renders without task-specific edits, realizes every "
            "acceptable group (or correctly makes none), preserves all supplied evidence, and states "
            "every group membership explicitly. Maintenance counts separate map, label, prompt, "
            "code, parser, correction, and regeneration work. Prospective human minutes remain "
            "unmeasured rather than estimated after the fact.\n")

    def finalize(row):
        row.update(text=row["raw_response"], status="complete", error="")

    checkpoint()
    if generate:
        progress(f"Dataset C: google / {GOOGLE_MODEL}; request timeout {request_timeout}s; SDK retries disabled")
        collect_responses(jobs,
            client_factory=lambda: build_langchain_llm("google", GOOGLE_MODEL, 0,
                                                       request_timeout=request_timeout, max_retries=0),
            extract=_extract_text_content, finalize=finalize, checkpoint=checkpoint,
            label="Dataset C", identity_field="case_id")
    progress(f"Dataset C: checkpoint at {output_dir / 'generations.jsonl'}")
    return results()


def compute_results(output_dir=OUTPUT_DIR):
    """Compatibility entry point; never makes an API call."""
    return run(output_dir=output_dir, generate=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--request-timeout", type=int, default=120)
    args = parser.parse_args()
    run(args.output_dir, args.generate, args.request_timeout)


if __name__ == "__main__":
    main()
