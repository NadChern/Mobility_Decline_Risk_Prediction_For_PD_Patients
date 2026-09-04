"""Self-contained completeness comparison for paired LLM/template explanations."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .shared import (
    METHODS, canonical_labels, expected_display_directions, expected_evidence, expected_outcomes,
    parse_report,
)
from .prose_checks import represented_factors_by_side

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRED = PROJECT_ROOT / "template_experiment/outputs/pilot/pilot_reports.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "template_experiment/outputs/pilot/completeness"

# Dropped vs. the first version: `required_section_coverage` (presence-only, subsumed by the
# nonempty version) and `required_stage_decision_coverage` (numerically identical to
# `classification_field_coverage`). Nothing is lost — those two answered questions another metric
# already answers.
METRICS = [
    "factor_recall", "nonempty_section_coverage",
    "classification_field_coverage", "expected_table_field_completeness",
    "displayed_stage_coverage", "required_stage_evidence_coverage",
    "interpretation_key_factor_coverage", "opposing_evidence_coverage",
    "overall_displayed_contract_complete", "overall_full_pipeline_complete",
]


def _direction_section_present(report, direction, expected_count):
    if direction not in report["directions"]:
        return False, False
    rows = [row for row in report["rows"] if row.direction == direction]
    if expected_count:
        return True, bool(rows)
    # An explicitly empty factor section is complete when its heading exists and the report says so.
    return True, (bool(rows) or "No qualifying factors." in report["text"])


def _required_section_metrics(report, evidence, patient_id):
    """Fraction of required sections that are present AND non-empty, plus the all-filled flag.

    Presence alone (a heading with no content) is not tracked separately — a non-empty section is
    necessarily present, so the presence-only variant was redundant.
    """
    directions = sorted(expected_display_directions(evidence["stage"]))
    expected_counts = {direction: len(evidence["expected_order"].get(direction, []))
                       for direction in directions}
    title_nonempty = any(str(patient_id) in title for title in report["title_lines"])
    nonempty = [title_nonempty, bool(report["prediction_content"].strip())]
    for direction in directions:
        _present, filled = _direction_section_present(report, direction, expected_counts[direction])
        nonempty.append(filled)
    nonempty += [bool(report["interpretation"].strip()), bool(report["clinical_note"].strip())]
    return float(np.mean(nonempty)), all(nonempty)


def _classification_coverage(report, source_row):
    expected = expected_outcomes(source_row)
    present = [bool(report["fall_values"])]
    if expected["severity"] is not None:
        present.append(bool(report["severity_values"]))
    return float(np.mean(present)), all(present)


def _table_field_completeness(report, evidence):
    """Completeness over expected cells; a missing expected row contributes four missing cells."""
    total = 4 * len(evidence["factor_set"])
    if total == 0:
        return 1.0
    complete = 0
    for factor in evidence["factor_set"]:
        rows = [row for row in report["rows"] if row.factor == factor]
        if not rows:
            continue
        # Completeness is presence, not correctness; use the best-filled occurrence.
        complete += max(
            int(bool(row.number.strip()) and row.number.isdigit())
            + int(bool(row.raw_factor.strip()))
            + int(bool(row.value.strip()))
            + int(bool(row.scale.strip()))
            for row in rows
        )
    return complete / total


def _stage_evidence_coverage(report, source_row):
    """Full-pipeline stage-evidence coverage: does the report show factor tables for every stage the
    prediction depends on? A routed faller needs Stage-1 and Stage-2 evidence; showing only Stage-2
    scores 0.5. (The stage *decision* fields are already covered by classification-field coverage.)"""
    stage = int(source_row["stage"])
    stage1_evidence = {"s1_positive", "s1_negative"}.issubset(set(report["directions"]))
    stage2_evidence = {"s2_positive", "s2_negative"}.issubset(set(report["directions"]))
    required_evidence = [stage1_evidence] if stage == 1 else [stage1_evidence, stage2_evidence]
    return float(np.mean(required_evidence))


def _interpretation_coverage(report, evidence, category, method):
    """Coverage of the saved top-5 supporting and top-3 opposing interpretation roles.

    This asks only whether selected factors are represented on the correct side. Whether factors
    were grouped correctly is owned exclusively by prose grouping fidelity.
    """
    represented = represented_factors_by_side(
        report["interpretation"], evidence, category, method)
    required_supporting = list(evidence["supporting_order"])
    required_opposing = list(evidence["opposing_order"])
    supporting_coverage = (len(set(required_supporting) & represented["supporting"])
                           / len(required_supporting)
                           if required_supporting else 1.0)
    opposing_coverage = (len(set(required_opposing) & represented["opposing"])
                         / len(required_opposing)
                         if required_opposing else 1.0)
    mentioned = represented["supporting"] | represented["opposing"]
    return supporting_coverage, opposing_coverage, required_supporting, required_opposing, mentioned


def compute_patient_completeness(source_row, method, model_col, explanation_col,
                                 norm_set=None, norm_to_short=None):
    if norm_set is None or norm_to_short is None:
        norm_set, norm_to_short = canonical_labels()
    report = parse_report(source_row[explanation_col], norm_set, norm_to_short)
    evidence = expected_evidence(source_row, norm_to_short)
    displayed = {row.factor for row in report["rows"] if row.factor is not None}
    expected = evidence["factor_set"]
    factor_recall = len(displayed & expected) / len(expected) if expected else 1.0
    nonempty_coverage, sections_filled = _required_section_metrics(
        report, evidence, int(source_row["patient_id"]))
    classification_coverage, classifications_filled = _classification_coverage(report, source_row)
    table_fields = _table_field_completeness(report, evidence)
    evidence_stage_coverage = _stage_evidence_coverage(report, source_row)
    displayed_stage_coverage = float(expected_display_directions(evidence["stage"]).issubset(
        set(report["directions"])))
    key_coverage, opposing_coverage, required_key, required_opposing, mentioned = (
        _interpretation_coverage(report, evidence, source_row["category"], method))

    displayed_complete = (sections_filled and classifications_filled
                          and np.isclose(factor_recall, 1.0)
                          and np.isclose(table_fields, 1.0)
                          and np.isclose(displayed_stage_coverage, 1.0))
    # `classifications_filled` already requires the stage decision fields, so no separate
    # decision-coverage term is needed here (it would double-count).
    full_pipeline_complete = (displayed_complete
                              and np.isclose(evidence_stage_coverage, 1.0))

    return {
        "patient_id": int(source_row["patient_id"]), "category": source_row["category"],
        "stage": int(source_row["stage"]), "method": method, "model": source_row[model_col],
        "factor_recall": factor_recall,
        "missing_factors": "|".join(sorted(expected - displayed)),
        "nonempty_section_coverage": nonempty_coverage,
        "classification_field_coverage": classification_coverage,
        "expected_table_field_completeness": table_fields,
        "displayed_stage_coverage": displayed_stage_coverage,
        "required_stage_evidence_coverage": evidence_stage_coverage,
        "interpretation_key_factor_coverage": key_coverage,
        "opposing_evidence_coverage": opposing_coverage,
        "required_interpretation_key_factors": "|".join(required_key),
        "required_opposing_factors": "|".join(required_opposing),
        "detected_interpretation_factors": "|".join(sorted(mentioned)),
        "overall_displayed_contract_complete": displayed_complete,
        "overall_full_pipeline_complete": full_pipeline_complete,
    }


def compute_results(paired_df):
    norm_set, norm_to_short = canonical_labels()
    rows = [compute_patient_completeness(source_row, method, model_col, explanation_col,
                                         norm_set, norm_to_short)
            for _, source_row in paired_df.iterrows()
            for method, (model_col, explanation_col) in METHODS.items()]
    return pd.DataFrame(rows).sort_values(["patient_id", "method"]).reset_index(drop=True)


def summarize(patient_results):
    rows = []
    groups = list(patient_results.groupby(["method", "category"]))
    groups += [((method, "overall"), group) for method, group in patient_results.groupby("method")]
    for (method, category), group in groups:
        result = {"method": method, "category": category, "n_patients": len(group)}
        for metric in METRICS:
            result[f"{metric}_mean"] = float(pd.to_numeric(group[metric]).mean())
        rows.append(result)
    return pd.DataFrame(rows).sort_values(["category", "method"]).reset_index(drop=True)


def write_report(summary, output_path):
    overall = summary[summary["category"] == "overall"].set_index("method")
    method_order = [m for m in ("template", "template_grouped", "gemini") if m in overall.index]
    method_labels = {"template": "Generic template", "template_grouped": "Domain template",
                     "gemini": "LLM (Gemini)"}
    labels = {
        "factor_recall": "Supplied-factor recall",
        "nonempty_section_coverage": "Required-section nonempty coverage",
        "classification_field_coverage": "Classification-field coverage",
        "expected_table_field_completeness": "Expected table-cell completeness",
        "displayed_stage_coverage": "Displayed-stage evidence coverage",
        "required_stage_evidence_coverage": "Full-pipeline stage-evidence coverage",
        "interpretation_key_factor_coverage": "Interpretation key-factor coverage",
        "opposing_evidence_coverage": "Opposing-evidence coverage",
        "overall_displayed_contract_complete": "Displayed-contract complete rate",
        "overall_full_pipeline_complete": "Full-pipeline complete rate",
    }
    header = "| Metric | " + " | ".join(method_labels[m] for m in method_order) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(method_order)) + " |"
    lines = [header, sep]
    for metric in METRICS:
        cells = " | ".join(f"{100*overall.loc[m, metric+'_mean']:.2f}%" for m in method_order)
        lines.append(f"| {labels[metric]} | {cells} |")
    output_path.write_text(
        "# Completeness pilot\n\n" + "\n".join(lines)
        + "\n\nDisplayed-contract completeness evaluates the current single-stage tables and requires a "
          "non-empty interpretation, but does not fold the lexical interpretation-coverage screens "
          "into the overall pass. Full-pipeline completeness additionally requires Stage-1 evidence "
          "for routed fallers (so a routed faller caps at 0.5 until both stages are shown). "
          "Interpretation coverage uses the saved top-5/top-3 roles and the correct prose side. "
          "All methods must name represented factors explicitly; hidden group membership receives "
          "no coverage credit. "
          "Grouping correctness is reported only under prose grouping fidelity.\n",
        encoding="utf-8")


def run(paired_path=DEFAULT_PAIRED, output_dir=DEFAULT_OUTPUT_DIR):
    paired = pd.read_csv(paired_path, keep_default_na=False)
    if len(paired) != 30 or paired["patient_id"].nunique() != 30:
        raise ValueError("Expected the 30-patient paired pilot dataset.")
    patient_results = compute_results(paired)
    summary = summarize(patient_results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_results.to_csv(output_dir / "patients.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    write_report(summary, output_dir / "report.md")
    return patient_results, summary


def main():
    parser = argparse.ArgumentParser(description="Run the self-contained completeness comparison.")
    parser.add_argument("--paired", default=str(DEFAULT_PAIRED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    run(args.paired, args.output_dir)


if __name__ == "__main__":
    main()
