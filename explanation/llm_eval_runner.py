"""LLM evaluation runner — generates llm_generations.csv.

Usage:
    # Verification run (1 run at temperature=0) to inspect CSV structure:
    python -m explanation.llm_eval_runner

    # Full evaluation run (5 runs × 2 temperatures):
    python -m explanation.llm_eval_runner --runs 5 --temperature 0 0.3
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from .data_loader import y_pred_final, patient_ids
from .explanation_builder import build_patient_explanation_data_full
from .llm import generate_explanation

RESULTS_DIR = Path(__file__).resolve().parent.parent / "evaluation_results"
GENERATIONS_PATH = RESULTS_DIR / "llm_generations.csv"

_FEATURE_MAP_PATH = Path(__file__).resolve().parent / "feature_map.csv"

_EXPECTED_COLS = [
    "patient_id", "category", "case", "model", "temperature", "run", "explanation",
    "stage1_contributing", "stage1_contributing_shap",
    "stage1_mitigating", "stage1_mitigating_shap",
    "stage2_contributing", "stage2_contributing_shap",
    "stage2_mitigating", "stage2_mitigating_shap",
]


def load_generations(path=GENERATIONS_PATH):
    """Load llm_generations.csv, preserving empty strings for Stage 2 Case 1 columns."""
    return pd.read_csv(path, keep_default_na=False)

_CATEGORY = {0: "no_falls", 1: "mild", 2: "moderate"}
_CASE = {0: 1, 1: 2, 2: 2}

_N_PER_CATEGORY = 10
_PATIENT_SEED = 42


def select_patients():
    """Return array of 30 test-set indices: 10 per final prediction category.

    Selection is fixed by seed=42 for reproducibility across all evaluation runs.
    """
    np.random.seed(_PATIENT_SEED)
    return np.concatenate([
        np.random.choice(np.where(y_pred_final == 0)[0], size=_N_PER_CATEGORY, replace=False),
        np.random.choice(np.where(y_pred_final == 1)[0], size=_N_PER_CATEGORY, replace=False),
        np.random.choice(np.where(y_pred_final == 2)[0], size=_N_PER_CATEGORY, replace=False),
    ])


def _pipe_shap(features):
    """Return (names, shap_magnitudes) as pipe-separated strings.

    Sign is encoded by which column the features appear in
    (contributing vs mitigating), so we store |SHAP| here.
    """
    if not features:
        return "", ""
    names = "|".join(f["feature"] for f in features)
    shap_vals = "|".join(f"{abs(f['shap_contribution']):.6f}" for f in features)
    return names, shap_vals


def _build_row(patient_id, patient_info, temperature, run, model, explanation):
    final_class = patient_info["final_prediction"]

    s1c, s1c_shap = _pipe_shap(patient_info["risk_increasing_features"])
    s1m, s1m_shap = _pipe_shap(patient_info["risk_decreasing_features"])

    if patient_info["has_severity_assessment"]:
        s2c, s2c_shap = _pipe_shap(patient_info["severity_increasing_features"])
        s2m, s2m_shap = _pipe_shap(patient_info["severity_decreasing_features"])
    else:
        s2c = s2c_shap = s2m = s2m_shap = ""

    return {
        "patient_id":               patient_id,
        "category":                 _CATEGORY[final_class],
        "case":                     _CASE[final_class],
        "model":                    model,
        "temperature":              temperature,
        "run":                      run,
        "explanation":              explanation,
        "stage1_contributing":      s1c,
        "stage1_contributing_shap": s1c_shap,
        "stage1_mitigating":        s1m,
        "stage1_mitigating_shap":   s1m_shap,
        "stage2_contributing":      s2c,
        "stage2_contributing_shap": s2c_shap,
        "stage2_mitigating":        s2m,
        "stage2_mitigating_shap":   s2m_shap,
    }


def generate_generations(temperatures, n_runs, output_path=GENERATIONS_PATH):
    """Generate LLM explanations for all selected patients and write to CSV.

    Args:
        temperatures: List of float temperature values.
        n_runs: Number of runs per patient per temperature.
        output_path: Path to write llm_generations.csv.

    Returns:
        DataFrame written to output_path.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    selected_indices = select_patients()
    total = len(selected_indices) * len(temperatures) * n_runs

    print(f"Patients: {len(selected_indices)}  |  Temperatures: {temperatures}  |  Runs per temp: {n_runs}")
    print(f"Total LLM calls: {total}\n")

    rows = []
    call_num = 0

    for patient_idx in selected_indices:
        patient_id = int(patient_ids.iloc[patient_idx]["PATNO"])
        final_class = int(y_pred_final[patient_idx])
        patient_info = build_patient_explanation_data_full(patient_idx)

        for temp in temperatures:
            for run in range(1, n_runs + 1):
                call_num += 1
                tag = (
                    f"[{call_num}/{total}] "
                    f"patient={patient_id} ({_CATEGORY[final_class]}, case={_CASE[final_class]}) "
                    f"temp={temp} run={run}"
                )
                print(tag, end=" ... ", flush=True)
                try:
                    result = generate_explanation(patient_id, temperature=temp)
                    rows.append(_build_row(
                        patient_id=patient_id,
                        patient_info=patient_info,
                        temperature=temp,
                        run=run,
                        model=result["model"],
                        explanation=result["explanation"],
                    ))
                    print("ok")
                except Exception as exc:
                    print(f"FAILED: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows → {output_path}")
    return df


def _print_sample(df):
    """Print column list and a truncated sample row for inspection."""
    print("\nColumns:")
    for col in df.columns:
        print(f"  {col}")

    print("\nSample row (first patient):")
    for col, val in df.iloc[0].items():
        display = str(val)
        if len(display) > 90:
            display = display[:87] + "..."
        print(f"  {col:<30} {display}")


def validate_generations(path=GENERATIONS_PATH):
    """Validate llm_generations.csv has everything needed for metric computation.

    Checks (all are errors → FAIL if triggered):
      1. Required columns present
      2. 30 patients, 10 per category
      3. Case assignment (no_falls→1, others→2)
      4. Empty explanations
      5. Pipe count alignment (feature names ↔ SHAP values)
      6. Stage 2 routing (empty for Case 1, non-empty for Case 2)
      7. Feature names in feature_map.csv (prerequisite for faithfulness evaluation)
      8. Duplicate (patient, temperature, run, model) rows

    Prints a PASS/FAIL report and returns True if no errors.
    """
    df = load_generations(path)
    errors = []

    # 1. Required columns
    missing_cols = [c for c in _EXPECTED_COLS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # 2. Patient counts per category
    for cat, expected_n in [("no_falls", 10), ("mild", 10), ("moderate", 10)]:
        n = df[df["category"] == cat]["patient_id"].nunique()
        if n != expected_n:
            errors.append(f"Expected {expected_n} {cat} patients, got {n}")

    # 3. Case assignment
    wrong = df[
        ((df["category"] == "no_falls") & (df["case"] != 1)) |
        ((df["category"] != "no_falls") & (df["case"] != 2))
    ]
    if len(wrong):
        errors.append(f"{len(wrong)} rows have wrong case assignment")

    # 4. Empty explanations
    empty_exp = (df["explanation"] == "").sum()
    if empty_exp:
        errors.append(f"{empty_exp} rows have empty explanations")

    # 5. Pipe count alignment between feature name and SHAP value columns
    pipe_pairs = [
        ("stage1_contributing", "stage1_contributing_shap"),
        ("stage1_mitigating",   "stage1_mitigating_shap"),
        ("stage2_contributing", "stage2_contributing_shap"),
        ("stage2_mitigating",   "stage2_mitigating_shap"),
    ]
    for nc, sc in pipe_pairs:
        if nc not in df.columns or sc not in df.columns:
            continue
        def _mismatch(row, nc=nc, sc=sc):
            n, s = row[nc], row[sc]
            if n == "" and s == "":
                return False
            return len(str(n).split("|")) != len(str(s).split("|"))
        bad = df.apply(_mismatch, axis=1).sum()
        if bad:
            errors.append(f"Pipe count mismatch {nc} vs {sc}: {bad} rows")

    # 6. Stage 2 routing: empty for Case 1, non-empty for Case 2
    case1 = df[df["case"] == 1]
    case2 = df[df["case"] == 2]
    if not (case1["stage2_contributing"] == "").all():
        errors.append("Case 1 rows have non-empty stage2_contributing")
    if len(case2) and not (case2["stage2_contributing"] != "").all():
        errors.append("Some Case 2 rows have empty stage2_contributing")

    # 7. Feature names in feature_map.csv
    # Guards faithfulness/feature-consistency evaluation: unknown names can't be
    # matched to LLM-mentioned variants and would be misclassified as ghost features.
    known = set(pd.read_csv(_FEATURE_MAP_PATH)["feature_name"])
    unknown = set()
    for col in ("stage1_contributing", "stage2_contributing"):
        if col not in df.columns:
            continue
        for val in df[col]:
            if val:
                for name in val.split("|"):
                    if name and name not in known:
                        unknown.add(name)
    if unknown:
        errors.append(f"Feature names not in feature_map.csv: {sorted(unknown)[:5]}")

    # 8. Duplicate (patient, temperature, run, model) rows
    dupes = df.duplicated(["patient_id", "temperature", "run", "model"]).sum()
    if dupes:
        errors.append(f"{dupes} duplicate (patient, temperature, run, model) rows")

    # Report
    print(f"\nValidating: {path}")
    print(
        f"Shape: {df.shape}  |  Patients: {df['patient_id'].nunique()}  |  "
        f"Models: {list(df['model'].unique())}"
    )
    print(
        f"Temperatures: {sorted(df['temperature'].unique())}  |  "
        f"Runs: {sorted(df['run'].unique())}"
    )
    if not errors:
        print("\n✓ PASS — CSV is ready for full evaluation run")
    else:
        for e in errors:
            print(f"\n✗ ERROR: {e}")
        print(f"\nFAIL — fix {len(errors)} error(s) before full run")

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM explanations for evaluation. "
                    "Default: 1 run at temperature=0 for column verification."
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Runs per patient per temperature (default: 1)",
    )
    parser.add_argument(
        "--temperature", type=float, nargs="+", default=[0.0],
        help="Temperature value(s) (default: 0.0)",
    )
    parser.add_argument(
        "--output", type=str, default=str(GENERATIONS_PATH),
        help="Output CSV path (default: evaluation_results/llm_generations.csv)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate existing llm_generations.csv without generating new data",
    )
    args = parser.parse_args()

    if args.validate:
        validate_generations(Path(args.output))
        return

    df = generate_generations(
        temperatures=args.temperature,
        n_runs=args.runs,
        output_path=Path(args.output),
    )
    _print_sample(df)


if __name__ == "__main__":
    main()
