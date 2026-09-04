"""Unsupported-information comparison for paired LLM/template explanations.

Automated metrics cover structured additions/alterations plus frozen-lexicon prose feature mentions
and prohibited causal language. They are screening measures and do not establish general semantic
claim validity.
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from .shared import (
    METHODS, canonical_labels, causal_sentences, detect_prose_features, expected_evidence,
    normalize_space, parse_report,
)
from ..taxonomy import FEATURE_CATEGORY

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRED = PROJECT_ROOT / "template_experiment/outputs/pilot/pilot_reports.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "template_experiment/outputs/pilot/unsupported_information"

# Rate metrics kept in this family. Removed as fidelity mirrors: `factor_precision`
# (= 1 - unsupported_factor_rate), `unsupported_value_rate` (= 1 - fidelity value_fidelity), and
# `unsupported_scale_rate` (= 1 - fidelity scale_fidelity). Value/scale *alterations* are still
# reported here as raw COUNTS (the primary form for this family), just not as duplicated rates.
METRICS = [
    "unsupported_factor_rate", "duplicate_factor_rate", "prose_unsupported_feature_rate",
    "causal_claim_sentence_rate", "automated_unsupported_information_free",
]

# Raw-count diagnostics, summed across patients and reported as the primary evidence.
COUNTS = [
    "unsupported_factor_count", "duplicate_factor_count", "unsupported_value_count",
    "unsupported_scale_count", "prose_unsupported_feature_count", "causal_claim_sentence_count",
]


def compute_patient_unsupported(source_row, method, model_col, explanation_col,
                                norm_set=None, norm_to_short=None):
    if norm_set is None or norm_to_short is None:
        norm_set, norm_to_short = canonical_labels()
    report = parse_report(source_row[explanation_col], norm_set, norm_to_short)
    evidence = expected_evidence(source_row, norm_to_short)
    expected = evidence["factor_set"]
    rows = report["rows"]

    supported_rows = [row for row in rows if row.factor in expected]
    unsupported_rows = [row for row in rows if row.factor not in expected]
    unsupported_factor_rate = len(unsupported_rows) / len(rows) if rows else 0.0

    known_counts = Counter(row.factor for row in rows if row.factor is not None)
    duplicate_count = sum(max(0, count - 1) for count in known_counts.values())
    duplicate_rate = duplicate_count / len(rows) if rows else 0.0

    # Altered value/scale are reported as raw counts; their *rate* is the exact complement of
    # fidelity value/scale fidelity, so it is not duplicated here.
    value_bad = [row for row in supported_rows
                 if normalize_space(row.value)
                 != normalize_space(evidence["content"].get(row.factor, {}).get("value", ""))]
    scale_bad = [row for row in supported_rows
                 if normalize_space(row.scale)
                 != normalize_space(evidence["content"].get(row.factor, {}).get("scale", ""))]

    # Group-aware grounding: a prose mention is unsupported only if neither the feature itself nor
    # its clinical category is present in the evidence. Without this, a group label ("motor
    # impairment") that the lexicon maps to one specific feature (UPDRS-III Total) is falsely flagged
    # when the group actually stands for other present members (e.g. Rigidity / H&Y).
    prose_features = detect_prose_features(report["interpretation"])
    supported_categories = {FEATURE_CATEGORY.get(factor) for factor in expected}
    prose_unsupported = {factor for factor in prose_features
                         if factor not in expected
                         and FEATURE_CATEGORY.get(factor) not in supported_categories}
    prose_rate = len(prose_unsupported) / len(prose_features) if prose_features else 0.0
    sentences, causal = causal_sentences(report["interpretation"])
    causal_rate = len(causal) / len(sentences) if sentences else 0.0

    unsupported_free = not (
        unsupported_rows or duplicate_count or value_bad or scale_bad or prose_unsupported or causal
    )
    return {
        "patient_id": int(source_row["patient_id"]), "category": source_row["category"],
        "stage": int(source_row["stage"]), "method": method, "model": source_row[model_col],
        "unsupported_factor_rate": unsupported_factor_rate,
        "unsupported_factor_count": len(unsupported_rows),
        "unsupported_factors": "|".join(row.raw_factor for row in unsupported_rows),
        "duplicate_factor_rate": duplicate_rate, "duplicate_factor_count": duplicate_count,
        "unsupported_value_count": len(value_bad),
        "unsupported_value_factors": "|".join(row.factor for row in value_bad),
        "unsupported_scale_count": len(scale_bad),
        "unsupported_scale_factors": "|".join(row.factor for row in scale_bad),
        "prose_unsupported_feature_rate": prose_rate,
        "prose_unsupported_feature_count": len(prose_unsupported),
        "prose_unsupported_features": "|".join(sorted(prose_unsupported)),
        "causal_claim_sentence_rate": causal_rate, "causal_claim_sentence_count": len(causal),
        "causal_claim_sentences": " || ".join(causal),
        "automated_unsupported_information_free": unsupported_free,
        "interpretation": report["interpretation"],
        "supplied_factors": "|".join(sorted(expected)),
    }


def compute_results(paired_df):
    norm_set, norm_to_short = canonical_labels()
    rows = [compute_patient_unsupported(source_row, method, model_col, explanation_col,
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
        for count in COUNTS:
            result[f"{count}_total"] = int(group[count].sum())
        rows.append(result)
    return pd.DataFrame(rows).sort_values(["category", "method"]).reset_index(drop=True)


def write_report(summary, output_path):
    overall = summary[summary["category"] == "overall"].set_index("method")
    method_order = [m for m in ("template", "template_grouped", "gemini") if m in overall.index]
    labels = {"template": "Generic template", "template_grouped": "Domain template",
              "gemini": "LLM (Gemini)"}
    head = " | ".join(labels[m] for m in method_order)
    sep = " | ".join(["---:"] * len(method_order))

    count_labels = {
        "unsupported_factor_count": "Extra (unsupported) factors",
        "duplicate_factor_count": "Duplicated factors",
        "unsupported_value_count": "Altered patient values",
        "unsupported_scale_count": "Altered scale texts",
        "prose_unsupported_feature_count": "Unsupported prose features",
        "causal_claim_sentence_count": "Causal-language sentences",
    }
    count_lines = [f"| Raw count (sum over patients) | {head} |", f"| --- | {sep} |"]
    for count in COUNTS:
        cells = " | ".join(str(int(overall.loc[m, f"{count}_total"])) for m in method_order)
        count_lines.append(f"| {count_labels[count]} | {cells} |")

    rate_lines = [f"| Rate | {head} |", f"| --- | {sep} |"]
    for metric in METRICS:
        cells = " | ".join(f"{100*overall.loc[m, metric+'_mean']:.2f}%" for m in method_order)
        rate_lines.append(f"| {metric.replace('_', ' ').title()} | {cells} |")

    output_path.write_text(
        "# Unsupported-information pilot\n\n"
        "**Was anything added, duplicated, or altered relative to the evidence?** Raw counts are the "
        "primary evidence; rates are secondary.\n\n"
        "## Raw counts (primary)\n\n" + "\n".join(count_lines)
        + "\n\n## Rates and all-clear (secondary)\n\n" + "\n".join(rate_lines)
        + "\n\nValue/scale *alteration rates* are intentionally omitted here — they are the exact "
          "complement of fidelity's value/scale fidelity, so counts (above) are reported instead of a "
          "duplicated rate. Automated prose measures are screening only and do not establish the "
          "absence of general semantic or clinical hallucinations.\n",
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
    parser = argparse.ArgumentParser(description="Run unsupported-information comparison.")
    parser.add_argument("--paired", default=str(DEFAULT_PAIRED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    run(args.paired, args.output_dir)


if __name__ == "__main__":
    main()
