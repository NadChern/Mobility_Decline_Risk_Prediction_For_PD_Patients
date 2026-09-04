"""Self-contained fidelity comparison for paired LLM/template explanations.

Fidelity asks whether present content is correct relative to model-derived evidence, *conditional
on that content being shown*. Presence of a supplied factor is a completeness concern; extra or
duplicated factors are an unsupported-information concern; both live in their own modules. Every
fidelity metric here therefore:

  - deduplicates repeated factors by first occurrence (duplicates are counted in the
    unsupported-information module, not penalized a second time here);
  - is evaluated only over factors that are actually displayed and expected; and
  - returns NaN when there is nothing to evaluate, so an empty or near-empty report is recorded as
    "not evaluable" (and reflected in the evaluable denominator) rather than silently scoring 0 or 1.

No metric implementation from the legacy ``explanation.evaluation`` package is reused here. The
current pilot scores the one factor stage displayed by the existing report contract.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .shared import (
    METHODS, canonical_labels, expected_evidence, normalize_space, parse_report,
    strict_outcome_correct,
)
from .prose_checks import PROSE_METRICS, compute_interpretation_fidelity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRED = PROJECT_ROOT / "template_experiment/outputs/pilot/pilot_reports.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "template_experiment/outputs/pilot/fidelity"

# Conditional-only fidelity core for the structured report (tables + label). Recall (missing
# factors) is reported by completeness; precision (extra/duplicate factors) by
# unsupported-information.
TABLE_METRICS = [
    "outcome_correct", "direction_accuracy", "value_fidelity", "scale_fidelity",
    "rank_order_fidelity",
]
# Automatic interpretation-prose fidelity panel (conservative screens; see prose_checks).
METRICS = TABLE_METRICS + PROSE_METRICS


def _first_occurrence(report_rows, expected):
    """Map each expected factor to its first displayed row (dedup-by-first).

    Duplicates are collapsed here so they are not double-counted; they are quantified separately by
    the unsupported-information module.
    """
    first = {}
    for row in report_rows:
        if row.factor in expected and row.factor not in first:
            first[row.factor] = row
    return first


def _content_fidelity(report_rows, evidence, field):
    """Conditional verbatim fidelity for a value or scale cell.

    Denominator = distinct expected factors that are actually displayed (dedup-by-first). Returns
    NaN when none are displayed, so a report that shows nothing is 'not evaluable' rather than
    scored 0 or 1. Match is exact after whitespace normalization; a paraphrase is a mismatch.
    """
    expected = evidence["factor_set"]
    first = _first_occurrence(report_rows, expected)
    if not first:
        return np.nan, []
    correct = 0
    mismatches = []
    for factor, row in first.items():
        expected_cell = normalize_space(evidence["content"].get(factor, {}).get(field, ""))
        observed = normalize_space(getattr(row, field))
        if observed == expected_cell:
            correct += 1
        else:
            mismatches.append(f"{factor}: expected={expected_cell!r}; observed={observed!r}")
    return correct / len(first), mismatches


def _direction_accuracy(report_rows, evidence):
    """Conditional directional fidelity over displayed expected factors (dedup-by-first).

    NaN when no expected factor is displayed (a recall problem, owned by completeness), so a misplaced
    factor is scored, but an absent one never counts as direction-0.
    """
    expected = evidence["factor_set"]
    first = _first_occurrence(report_rows, expected)
    if not first:
        return np.nan
    correct = sum(row.direction == evidence["expected_direction"][factor]
                  for factor, row in first.items())
    return correct / len(first)


def _rank_order_fidelity(report_rows, evidence):
    """Pairwise within-direction concordance: correctly-ordered comparable pairs / comparable pairs.

    Correct order -> 1.0, fully reversed -> 0.0, partial ordering errors in between, fewer than two
    comparable factors -> NaN. A comparable pair is an expected ordered pair whose two factors both
    appear in the report (first occurrence used for position).
    """
    pair_correct = pair_total = 0
    for direction, expected_order in evidence["expected_order"].items():
        displayed = [row.factor for row in report_rows
                     if row.direction == direction and row.factor in expected_order]
        first_position = {}
        for index, factor in enumerate(displayed):
            first_position.setdefault(factor, index)
        for left_index, left in enumerate(expected_order):
            for right in expected_order[left_index + 1:]:
                if left in first_position and right in first_position:
                    pair_total += 1
                    pair_correct += first_position[left] < first_position[right]
    return pair_correct / pair_total if pair_total else np.nan


def compute_patient_fidelity(source_row, method, model_col, explanation_col,
                             norm_set=None, norm_to_short=None):
    if norm_set is None or norm_to_short is None:
        norm_set, norm_to_short = canonical_labels()
    report = parse_report(source_row[explanation_col], norm_set, norm_to_short)
    evidence = expected_evidence(source_row, norm_to_short)

    direction_accuracy = _direction_accuracy(report["rows"], evidence)
    value_fidelity, value_mismatches = _content_fidelity(report["rows"], evidence, "value")
    scale_fidelity, scale_mismatches = _content_fidelity(report["rows"], evidence, "scale")
    rank_fidelity = _rank_order_fidelity(report["rows"], evidence)
    prose = compute_interpretation_fidelity(report, evidence, source_row, method=method)

    return {
        "patient_id": int(source_row["patient_id"]), "category": source_row["category"],
        "stage": int(source_row["stage"]), "method": method, "model": source_row[model_col],
        "outcome_correct": strict_outcome_correct(report, source_row),
        "direction_accuracy": direction_accuracy,
        "value_fidelity": value_fidelity,
        "scale_fidelity": scale_fidelity,
        "rank_order_fidelity": rank_fidelity,
        **prose,
        "value_mismatches": "|".join(value_mismatches),
        "scale_mismatches": "|".join(scale_mismatches),
        "interpretation": report["interpretation"],
    }


def compute_results(paired_df):
    norm_set, norm_to_short = canonical_labels()
    rows = [compute_patient_fidelity(source_row, method, model_col, explanation_col,
                                     norm_set, norm_to_short)
            for _, source_row in paired_df.iterrows()
            for method, (model_col, explanation_col) in METHODS.items()]
    return pd.DataFrame(rows).sort_values(["patient_id", "method"]).reset_index(drop=True)


def summarize(patient_results):
    """Per-(method, category) and overall means, each with its evaluable denominator.

    ``{metric}_n_evaluable`` is the count of non-NaN observations behind ``{metric}_mean``. It must
    be reported alongside the mean: with NaN-on-empty conditional fidelity, a method that emits empty
    output would otherwise appear artificially faithful because its hard cases were silently dropped.
    """
    rows = []
    groups = list(patient_results.groupby(["method", "category"]))
    groups += [((method, "overall"), group) for method, group in patient_results.groupby("method")]
    for (method, category), group in groups:
        result = {"method": method, "category": category, "n_patients": len(group)}
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            result[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            result[f"{metric}_n_evaluable"] = int(len(values))
        rows.append(result)
    return pd.DataFrame(rows).sort_values(["category", "method"]).reset_index(drop=True)


def _percent(value):
    return "NA" if pd.isna(value) else f"{100 * value:.2f}%"


def write_report(summary, output_path):
    overall = summary[summary["category"] == "overall"].set_index("method")
    descriptions = [
        ("Outcome correctness", "Is the reported fall classification correct?",
         "outcome_correct"),
        ("Direction accuracy", "Are displayed factors in the correct supporting or opposing table?",
         "direction_accuracy"),
        ("Value fidelity", "Are displayed patient values copied exactly from the supplied evidence?",
         "value_fidelity"),
        ("Scale fidelity", "Is each displayed scale explanation copied completely and exactly?",
         "scale_fidelity"),
        ("Rank-order fidelity", "Are factors kept in their supplied order within each table?",
         "rank_order_fidelity"),
    ]

    prose_descriptions = [
        ("Prose direction fidelity",
         "Of the factors mentioned, how many are discussed on the correct supporting or opposing side?",
         "prose_direction_fidelity"),
        ("Prose grouping fidelity",
         "For each report with groups, what is the average of the two checks below?",
         "prose_grouping_fidelity"),
        ("— Selected-factor precision",
         "Of the factors placed in groups, how many belong to the required top 5/top 3?",
         "selected_factor_precision"),
        ("— Minimum-size compliance",
         "Of the groups created, how many contain at least two factors?",
         "minimum_group_size_compliance"),
        ("Prose value-descriptor fidelity",
         "Of descriptions with an exact categorical scale label, how many match that label?",
         "prose_value_descriptor_fidelity"),
    ]

    taxonomy_descriptions = [
        ("Taxonomy grouping recall",
         "Of the factor–category links defined by the reference taxonomy, how many did the prose reproduce?",
         "taxonomy_grouping_recall"),
        ("Taxonomy grouping precision",
         "Of the factor–category links made in the prose, how many match the reference taxonomy?",
         "taxonomy_grouping_precision"),
    ]

    arms = [("gemini", "LLM (Gemini)"), ("template", "Generic template"),
            ("template_grouped", "Domain template")]
    arms = [(key, label) for key, label in arms if key in overall.index]

    def cell(method, metric):
        mean = _percent(overall.loc[method, f"{metric}_mean"])
        n_eval = int(overall.loc[method, f"{metric}_n_evaluable"])
        total = int(overall.loc[method, "n_patients"])
        return f"{mean} ({n_eval}/{total})"

    def table(rows):
        header = "| Metric | Definition | " + " | ".join(f"{label} (n eval)" for _, label in arms) + " |"
        out = [header, "| --- | --- " + "| ---: " * len(arms) + "|"]
        for label, definition, metric in rows:
            cells = " | ".join(cell(key, metric) for key, _ in arms)
            out.append(f"| {label} | {definition} | {cells} |")
        return "\n".join(out)

    grouping_note = ""
    if "template_grouped" in overall.index:
        total = int(overall.loc["template_grouped", "n_patients"])
        domain_evaluable = int(
            overall.loc["template_grouped", "selected_factor_precision_n_evaluable"]
        )
        pieces = [
            f"The Domain template is evaluable for {domain_evaluable}/{total} reports; the other "
            f"{total - domain_evaluable} had no category containing at least two selected factors."
        ]
        if "gemini" in overall.index:
            llm_total = int(overall.loc["gemini", "n_patients"])
            llm_evaluable = int(
                overall.loc["gemini", "selected_factor_precision_n_evaluable"]
            )
            pieces.append(
                f"The LLM is evaluable for {llm_evaluable}/{llm_total}; it formed no group in "
                f"{llm_total - llm_evaluable} report."
            )
        grouping_note = "\n\n**Grouping denominators:** " + " ".join(pieces)

    output_path.write_text(
        "# Fidelity pilot\n\n## Tables and label\n\n" + table(descriptions)
        + "\n\n## Interpretation prose (automatic panel)\n\n" + table(prose_descriptions)
        + grouping_note
        + "\n\nThis supports the conclusion:\n\n"
          "> When the LLM formed groups, it used grounded selected factors and followed the "
          "minimum two-factor requirement.\n\n"
          "It does **not** support:\n\n"
          "> The LLM’s clinical categories were definitively correct."
        + "\n\n## Agreement with the reference taxonomy\n\n" + table(taxonomy_descriptions)
        + "\n\nThese are descriptive agreement metrics, not clinical-validity metrics. The taxonomy "
          "is one predefined operational scheme for the rule-based template. The Domain template’s "
          "100% agreement is expected because it uses that same taxonomy; a different LLM grouping "
          "may still be clinically defensible."
        + "\n\nTable/label fidelity is conditional: it scores only content that is actually displayed. "
          "A missing factor is recorded by completeness (recall) and an extra or duplicate factor by "
          "unsupported-information (precision), so neither is double-counted here. Metrics return NaN "
          "when nothing is evaluable; the evaluable denominator `(n eval / n patients)` is shown so "
          "empty output cannot look artificially faithful. Value and scale fidelity are verbatim "
          "(whitespace-normalized) matches, so a paraphrase counts as a mismatch.\n\n"
          "The interpretation-prose panel is a conservative, human-free screen. Prose grouping "
          "fidelity is the unweighted mean of selected-factor precision and minimum-size compliance. "
          "It is conditional on a group being formed. Direction is scored separately, and required "
          "factor representation is assessed by completeness. The generic template has no grouping "
          "mechanism, so grouping metrics are not applicable for that arm. Value-description fidelity "
          "uses only qualitative labels explicitly assigned to the patient’s exact value in the "
          "supplied scale; continuous scores without categorical cutoffs are not evaluated. "
          "The panel does not judge synthesis quality, "
          "and it cannot catch a subtly worded invented interaction. Feature hallucination and causal "
          "language are reported by the unsupported-information module.\n",
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
    parser = argparse.ArgumentParser(description="Run the self-contained fidelity comparison.")
    parser.add_argument("--paired", default=str(DEFAULT_PAIRED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    run(args.paired, args.output_dir)


if __name__ == "__main__":
    main()
