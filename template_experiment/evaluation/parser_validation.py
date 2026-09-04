"""Frozen hand-authored validation corpus for the common group-claim parser."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .group_claims import extract_group_claims
from ..synthesis_protocol import PROJECT_ROOT, PROTOCOL_DIR


OUTPUT_DIR = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/checker_v2_1/parser_validation"
ACCEPTANCE_THRESHOLD = 0.95

# Each expected relation is side/group/sorted-members. Negative and mutation cases are intentional.
GOLD_CASES = (
    ("P01", "moderate", "Within this model, the result was associated with gait and mobility (FOG (Freezing of Gait), Gait).",
     [("supporting", "gait_mobility", ["FOG (Freezing of Gait)", "Gait"])]),
    ("P02", "moderate", "Within this model, the result was associated with motor impairment (UPDRS-III Total, H&Y Stage).",
     [("supporting", "motor_impairment", ["H&Y Stage", "UPDRS-III Total"])]),
    ("P03", "moderate", "Factors in the opposite direction included autonomic symptoms (Fainting, Postural Hypotension); however, these did not outweigh the result.",
     [("opposing", "autonomic_symptoms", ["Fainting", "Postural Hypotension"])]),
    ("P04", "mild", "The evidence included gait-related measures, specifically FOG (Freezing of Gait) and Gait.",
     [("supporting", "gait_mobility", ["FOG (Freezing of Gait)", "Gait"])]),
    ("P05", "moderate", "Motor symptoms were represented by Rigidity at Dx and UPDRS-III Total.",
     [("supporting", "motor_impairment", ["Rigidity at Dx", "UPDRS-III Total"])]),
    ("P06", "moderate", "The result was associated with gait and mobility.", []),
    ("P07", "moderate", "The result was associated with cognition (MoCA).", []),
    ("P08", "moderate", "The result mentioned Gait and Mobility Summary.", []),
    ("P09", "moderate", "Although mood symptoms (GDS-15, GDS-15) were noted, the result remained unchanged.", []),
    ("P10", "no_falls", "Factors toward Fall included motor impairment (UPDRS-III Total, Rigidity at Dx).",
     [("opposing", "motor_impairment", ["Rigidity at Dx", "UPDRS-III Total"])]),
    ("P11", "moderate", "Within this model, gait assessments (Gait, Postural Stability) were important.",
     [("supporting", "gait_mobility", ["Gait", "Postural Stability"])]),
    ("P12", "moderate", "Opposite-direction evidence included sleep/wake symptoms (Daytime Sleepiness, UPDRS-I Total).",
     [("opposing", "sleep_wake_symptoms", ["Daytime Sleepiness", "UPDRS-I Total"])]),
    ("P13", "moderate", "The result was associated with autonomic symptoms, supported by Fainting and Lightheadedness on Standing.",
     [("supporting", "autonomic_symptoms", ["Fainting", "Lightheadedness on Standing"])]),
    ("P14", "moderate", "The result was associated with disease-course characteristics (PD Duration, Age).",
     [("supporting", "disease_course", ["Age", "PD Duration"])]),
    ("P15", "moderate", "Gait and MoCA were listed individually without a category.", []),
    ("P16", "moderate", "The result was associated with gait and mobility (FOG (Freezing of Gait), Gait", []),
)


def _normalized(claims):
    return sorted((claim.side, claim.group, sorted(claim.members)) for claim in claims)


def run(output_dir=OUTPUT_DIR):
    gold_rows, validation = [], []
    for case_id, category, prose, expected in GOLD_CASES:
        claims = extract_group_claims(prose, category)
        observed = _normalized(claims)
        expected = sorted(expected)
        spans_valid = all(0 <= claim.span_start < claim.span_end <= len(prose) for claim in claims)
        gold_rows.append({"case_id": case_id, "category": category, "prose": prose,
                          "expected_relations": json.dumps(expected)})
        validation.append({"case_id": case_id, "exact_relation_match": observed == expected,
                           "source_spans_valid": spans_valid,
                           "expected_relations": json.dumps(expected),
                           "observed_relations": json.dumps(observed)})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gold = pd.DataFrame(gold_rows)
    results = pd.DataFrame(validation)
    gold.to_csv(PROTOCOL_DIR / "parser_gold.csv", index=False)
    results.to_csv(output_dir / "parser_validation.csv", index=False)
    accuracy = float(results.exact_relation_match.mean())
    (output_dir / "parser_validation_report.md").write_text(
        "# Common parser validation\n\n"
        f"Exact relation accuracy: {accuracy:.1%} ({int(results.exact_relation_match.sum())}/{len(results)}). "
        f"Frozen acceptance threshold: {ACCEPTANCE_THRESHOLD:.0%}. "
        f"Status: {'pass' if accuracy >= ACCEPTANCE_THRESHOLD else 'fail; use manual extraction'}.\n",
        encoding="utf-8")
    return results


if __name__ == "__main__":
    run()
