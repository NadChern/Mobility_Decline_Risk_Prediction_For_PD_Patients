"""Self-contained structure-compliance comparison for paired explanation reports."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .shared import (
    METHODS, direction_from_heading, expected_display_directions, heading_text,
    parse_report, split_markdown_row,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRED = PROJECT_ROOT / "template_experiment/outputs/pilot/pilot_reports.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "template_experiment/outputs/pilot/structure"

# The eight independent component checks, plus the single all-pass headline. A mean-of-booleans
# "structure_score" was dropped (averaging unlike format checks is not interpretable); the graded
# ordinal `n_components_passed` (0-8) is kept as a diagnostic column instead, not in this table.
COMPONENTS = [
    "title_format_correct", "section_order_correct", "stage_template_correct",
    "table_header_format_correct", "table_row_format", "table_numbering_correct",
    "supporting_table_first", "forbidden_content_clean",
]
METRICS = COMPONENTS + ["overall_structure_compliant"]
EXPECTED_TABLE_HEADER = ["#", "Factor", "Patient Value", "Interpretation / Scale"]
FORBIDDEN_PATTERNS = (r"%", r"\bprobabilit", r"\bpercent", r"\blikelihood")


def _line_positions(text):
    lines = str(text).splitlines()
    title = next((i for i, line in enumerate(lines)
                  if heading_text(line).startswith("Clinical Fall Risk Summary")), None)
    prediction = next((i for i, line in enumerate(lines)
                       if heading_text(line) == "Model Prediction"), None)
    factors = [i for i, line in enumerate(lines) if direction_from_heading(line)]
    interpretation = next((i for i, line in enumerate(lines)
                           if heading_text(line) == "Model Interpretation"), None)
    note = next((i for i, line in enumerate(lines)
                 if heading_text(line) == "Clinical Note"), None)
    return lines, title, prediction, factors, interpretation, note


def _section_order(text):
    """Test the *order* of the sections only. Whether there are exactly two correct factor tables is
    `stage_template_correct`'s job, so it is not re-checked here (avoids double-counting)."""
    _, title, prediction, factors, interpretation, note = _line_positions(text)
    if None in (title, prediction, interpretation, note) or not factors:
        return False
    return title < prediction < factors[0] and factors[-1] < interpretation < note


def _title_format(report, patient_id):
    expected = f"Clinical Fall Risk Summary for Patient ID: {int(patient_id)}"
    return report["title_lines"] == [expected]


def _stage_template(report, stage):
    expected = expected_display_directions(stage)
    return set(report["directions"]) == expected and len(report["directions"]) == 2


def _table_format_metrics(text):
    lines = str(text).splitlines()
    current_direction = None
    header_correct, row_valid, numbering = [], [], {}
    for line in lines:
        direction = direction_from_heading(line)
        if direction:
            current_direction = direction
            numbering.setdefault(direction, [])
            continue
        if current_direction is None or not line.strip().startswith("|"):
            continue
        cells = split_markdown_row(line)
        first = cells[0] if cells else ""
        if first == "#":
            header_correct.append(cells == EXPECTED_TABLE_HEADER)
            continue
        if first and set(first) <= set("-: "):
            continue
        valid = len(cells) == 4 and first.isdigit()
        row_valid.append(valid)
        if valid:
            numbering[current_direction].append(int(first))
    header_ok = len(header_correct) == 2 and all(header_correct)
    row_format = float(np.mean(row_valid)) if row_valid else 1.0
    numbering_ok = all(numbers == list(range(1, len(numbers) + 1))
                       for numbers in numbering.values())
    return header_ok, row_format, numbering_ok


def _supporting_first(report, category):
    if not report["directions"]:
        return False
    expected = {"no_falls": "s1_negative", "mild": "s2_negative",
                "moderate": "s2_positive"}.get(str(category))
    return expected is not None and report["directions"][0] == expected


def _forbidden_clean(text):
    return not any(re.search(pattern, str(text), re.I) for pattern in FORBIDDEN_PATTERNS)


def compute_patient_structure(source_row, method, model_col, explanation_col):
    text = source_row[explanation_col]
    report = parse_report(text)
    header_ok, row_format, numbering_ok = _table_format_metrics(text)
    components = {
        "title_format_correct": _title_format(report, source_row["patient_id"]),
        "section_order_correct": _section_order(text),
        "stage_template_correct": _stage_template(report, source_row["stage"]),
        "table_header_format_correct": header_ok,
        "table_row_format": row_format,
        "table_numbering_correct": numbering_ok,
        "supporting_table_first": _supporting_first(report, source_row["category"]),
        "forbidden_content_clean": _forbidden_clean(text),
    }
    passed = [np.isclose(float(value), 1.0) for value in components.values()]
    return {
        "patient_id": int(source_row["patient_id"]), "category": source_row["category"],
        "stage": int(source_row["stage"]), "method": method, "model": source_row[model_col],
        **components,
        "overall_structure_compliant": all(passed),
        # Interpretable graded summary (0-8), replacing the mean-of-booleans structure_score.
        "n_components_passed": int(sum(passed)),
        "structure_failures": "|".join(name for name, value in components.items()
                                        if not np.isclose(float(value), 1.0)),
    }


def compute_results(paired_df):
    rows = [compute_patient_structure(source_row, method, model_col, explanation_col)
            for _, source_row in paired_df.iterrows()
            for method, (model_col, explanation_col) in METHODS.items()]
    return pd.DataFrame(rows).sort_values(["patient_id", "method"]).reset_index(drop=True)


def summarize(patient_results):
    rows = []
    groups = list(patient_results.groupby(["method", "category"]))
    groups += [((method, "overall"), group) for method, group in patient_results.groupby("method")]
    for (method, category), group in groups:
        result = {"method": method, "category": category, "n_patients": len(group)}
        for metric in METRICS + ["n_components_passed"]:
            result[f"{metric}_mean"] = float(pd.to_numeric(group[metric]).mean())
        rows.append(result)
    return pd.DataFrame(rows).sort_values(["category", "method"]).reset_index(drop=True)


def write_report(summary, output_path):
    overall = summary[summary["category"] == "overall"].set_index("method")
    method_order = [m for m in ("template", "template_grouped", "gemini") if m in overall.index]
    labels = {"template": "Generic template", "template_grouped": "Domain template",
              "gemini": "LLM (Gemini)"}
    header = "| Metric | " + " | ".join(labels[m] for m in method_order) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(method_order)) + " |"
    lines = [header, sep]
    for metric in METRICS:
        cells = " | ".join(f"{100*overall.loc[m, metric+'_mean']:.2f}%" for m in method_order)
        lines.append(f"| {metric.replace('_', ' ').title()} | {cells} |")
    passed = " | ".join(f"{overall.loc[m, 'n_components_passed_mean']:.2f} / 8"
                        for m in method_order)
    lines.append(f"| Components passed (mean) | {passed} |")
    output_path.write_text(
        "# Structure-compliance pilot\n\n" + "\n".join(lines)
        + "\n\nStructure compliance evaluates layout and formatting only; it does not establish "
          "factual correctness, evidence completeness, or absence of unsupported claims. The eight "
          "component checks are independent; `overall_structure_compliant` is their all-pass "
          "conjunction and `components passed` is the interpretable graded summary (0-8).\n",
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
    parser = argparse.ArgumentParser(description="Run paired structure-compliance comparison.")
    parser.add_argument("--paired", default=str(DEFAULT_PAIRED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    run(args.paired, args.output_dir)


if __name__ == "__main__":
    main()
