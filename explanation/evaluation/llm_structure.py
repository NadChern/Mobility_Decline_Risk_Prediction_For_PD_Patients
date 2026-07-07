"""Structure Compliance Evaluation — Section 3 of llm_eval_plan.md.

Standalone, no LLM calls. Verifies the generated document conforms to the required clinical
template (a formatting/compliance check, distinct from stability and faithfulness). Computed
once on the production output (temperature=0, run=1) — one value per patient, then aggregated
patient-wise → per category as rates.

Checks:
  - Section presence:   title, Model Prediction, two factor tables, Model Interpretation, Clinical Note
  - Stage / routing:    no_falls → "Toward Fall/No Fall" tables, no severity line;
                        mild/moderate → "Toward Higher/Lower Severity" tables + severity line
  - Table row format:   data rows match `| # | Factor | Patient Value | Interpretation / Scale |`
  - Table ordering:     the prediction-supporting (drivers) table appears FIRST
  - Forbidden content:  no probabilities / percentages (the prompt bans them)
  - Structure score:    mean of the above components

Usage:
    python -m explanation.evaluation.llm_structure
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .llm_stability import GENERATIONS_PATH, RESULTS_DIR, CATEGORIES, load_generations

PATIENT_RESULTS_PATH = RESULTS_DIR / "llm_eval_structure_patient.csv"
SUMMARY_PATH = RESULTS_DIR / "llm_eval_structure_summary.csv"

TEMP_PRODUCTION = 0.0
RUN_PRODUCTION = 1

# The prediction-supporting (drivers) table that must appear FIRST, per category.
_SUPPORT_FIRST = {"no_falls": "no_fall", "mild": "lower", "moderate": "higher"}
_FORBIDDEN = ["probabilit", "percent", "likelihood"]  # plus a literal "%" check


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------
def table_header_directions(text):
    """Return the table directions in document order: 'fall'/'no_fall'/'higher'/'lower'."""
    dirs = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") or "Factors Pushing the Prediction Toward" not in s:
            continue
        if "No Fall" in s:
            dirs.append("no_fall")
        elif "Higher Severity" in s:
            dirs.append("higher")
        elif "Lower Severity" in s:
            dirs.append("lower")
        elif "Toward Fall" in s:
            dirs.append("fall")
    return dirs


def section_checks(text):
    """Binary presence of each required section."""
    return {
        "title": "Clinical Fall Risk Summary" in text,
        "model_prediction": "Model Prediction" in text,
        "two_tables": len(table_header_directions(text)) >= 2,
        "model_interpretation": "Model Interpretation" in text,
        "clinical_note": "Clinical Note" in text,
    }


def table_row_format(text):
    """Fraction of table data rows that match `| # | Factor | Value | Interpretation |`."""
    mi = text.find("Model Interpretation")
    region = text[:mi] if mi >= 0 else text
    total = wellformed = 0
    for line in region.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        first = cells[0] if cells else ""
        if first == "#":                       # header row
            continue
        if first and set(first) <= set("-: "):  # separator row
            continue
        total += 1                              # candidate data row
        if len(cells) == 4 and first.isdigit():
            wellformed += 1
    return (wellformed / total) if total else np.nan


def stage_compliance(text, category):
    """Tables and severity line match the patient's routing case."""
    dirs = set(table_header_directions(text))
    severity_line = bool(re.search(r"Severity Classification", text, re.I))
    if category == "no_falls":
        return dirs == {"fall", "no_fall"} and not severity_line
    return dirs == {"higher", "lower"} and severity_line


def table_ordering(text, category):
    """The prediction-supporting (drivers) table appears first."""
    dirs = table_header_directions(text)
    return bool(dirs) and dirs[0] == _SUPPORT_FIRST[category]


def forbidden_content_clean(text):
    """No probabilities / percentages (the prompt forbids them)."""
    low = text.lower()
    if "%" in text:
        return False
    return not any(w in low for w in _FORBIDDEN)


# -----------------------------------------------------------------------------
# Per-patient
# -----------------------------------------------------------------------------
def compute_patient_structure(row):
    text, category = row["explanation"], row["category"]
    secs = section_checks(text)
    all_sections = all(secs.values())
    stage_ok = stage_compliance(text, category)
    row_fmt = table_row_format(text)
    order_ok = table_ordering(text, category)
    clean = forbidden_content_clean(text)

    components = [
        float(all_sections), float(stage_ok), float(row_fmt if row_fmt == row_fmt else 0.0),
        float(order_ok), float(clean),
    ]
    structure_score = float(np.mean(components))

    failures = []
    failures += [f"missing_section:{k}" for k, v in secs.items() if not v]
    if not stage_ok:
        failures.append("stage_routing")
    if row_fmt == row_fmt and row_fmt < 1.0:
        failures.append(f"row_format={row_fmt:.2f}")
    if not order_ok:
        failures.append("table_ordering")
    if not clean:
        failures.append("forbidden_content")

    return {
        "patient_id": row["patient_id"],
        "category": category,
        "stage": int(row["stage"]),
        "model": row["model"],
        "section_presence": float(np.mean(list(secs.values()))),
        "all_sections": all_sections,
        "stage_compliance": stage_ok,
        "table_row_format": row_fmt,
        "table_ordering": order_ok,
        "forbidden_content_clean": clean,
        "structure_score": structure_score,
        "failures": "|".join(failures),
    }


# -----------------------------------------------------------------------------
# Aggregation / orchestration
# -----------------------------------------------------------------------------
def aggregate_by_category(patient_df):
    metric_cols = ["section_presence", "all_sections", "stage_compliance", "table_row_format",
                   "table_ordering", "forbidden_content_clean", "structure_score"]
    rows = []
    for (model, category), grp in patient_df.groupby(["model", "category"]):
        row = {"model": model, "category": category, "n_patients": len(grp)}
        for c in metric_cols:
            row[f"{c}_mean"] = float(pd.to_numeric(grp[c], errors="coerce").mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["category"] = pd.Categorical(summary["category"], CATEGORIES, ordered=True)
        summary = summary.sort_values(["model", "category"]).reset_index(drop=True)
    return summary


def run_structure_evaluation(generations_df):
    prod = generations_df[
        np.isclose(generations_df["temperature"].astype(float), TEMP_PRODUCTION)
        & (generations_df["run"] == RUN_PRODUCTION)
    ]
    patient_df = pd.DataFrame([compute_patient_structure(r) for _, r in prod.iterrows()])
    return patient_df, aggregate_by_category(patient_df)


def _print_summary(summary_df):
    cols = ["model", "category", "n_patients", "section_presence_mean", "stage_compliance_mean",
            "table_row_format_mean", "table_ordering_mean",
            "forbidden_content_clean_mean", "structure_score_mean"]
    cols = [c for c in cols if c in summary_df.columns]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("\nPer-category structure summary:\n")
        print(summary_df[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Structure Compliance Evaluation (Section 3).")
    parser.add_argument("--generations", default=str(GENERATIONS_PATH))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    df = load_generations(args.generations)
    patient_df, summary_df = run_structure_evaluation(df)

    patient_path = out_dir / PATIENT_RESULTS_PATH.name
    summary_path = out_dir / SUMMARY_PATH.name
    patient_df.to_csv(patient_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved per-patient structure → {patient_path}  ({len(patient_df)} rows)")
    print(f"Saved per-category summary   → {summary_path}  ({len(summary_df)} rows)")
    _print_summary(summary_df)

    failed = patient_df[patient_df["failures"] != ""]
    if len(failed):
        print("\n⚠ Structure issues (inspect):")
        for _, r in failed.iterrows():
            print(f"  patient {r['patient_id']} ({r['category']}): {r['failures']}")
    else:
        print("\n✓ All patients fully structure-compliant.")


if __name__ == "__main__":
    main()
