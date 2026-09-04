"""LLM evaluation runner — generates llm_generations.csv.

Usage:
    # Verification run (1 run at temperature=0) to inspect CSV structure:
    python -m explanation.evaluation.llm_eval_runner

    # Full evaluation run (5 runs × 2 temperatures):
    python -m explanation.evaluation.llm_eval_runner --runs 5 --temperature 0 0.3
"""

import argparse
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

from ..contract import PATIENT_ID_COLUMN
from ..data_loader import y_pred_final, patient_ids
from ..explanation_builder import build_patient_explanation_data_full
from ..llm import generate_explanation, _displayable, _interpretation_factor_names

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "evaluation_results"
GENERATIONS_PATH = RESULTS_DIR / "llm_generations.csv"

_FEATURE_MAP_PATH = Path(__file__).resolve().parent.parent / "feature_map.csv"

_EXPECTED_COLS = [
    "patient_id", "category", "stage", "model", "temperature", "run", "explanation",
    "contributing", "contributing_shap",
    "mitigating", "mitigating_shap",
    "supporting_factors", "opposing_factors",
]


def load_generations(path=GENERATIONS_PATH):
    """Load llm_generations.csv, preserving empty strings in feature columns."""
    return pd.read_csv(path, keep_default_na=False)

_CATEGORY = {0: "no_falls", 1: "mild", 2: "moderate"}

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

    Names use each factor's short_name — the label the prompt shows the LLM and that
    the LLM reproduces verbatim in its output tables — so evaluation compares
    like-for-like with no name mapping needed. Sign is encoded by which column the
    features appear in (contributing vs mitigating), so we store |SHAP| here.
    """
    if not features:
        return "", ""
    names = "|".join(f["short_name"] for f in features)
    shap_vals = "|".join(f"{abs(f['shap_contribution']):.6f}" for f in features)
    return names, shap_vals


def _build_row(patient_id, patient_info, temperature, run, model, explanation):
    """Build one generations row from the deterministic explanation packet.

    Each explanation is single-stage: non-fallers are explained with the Stage 1 tables,
    fallers with the Stage 2 (severity) tables. We store only the factors actually shown
    to the LLM (displayable, for the relevant stage), split by SHAP direction:
      - contributing = SHAP > 0 (toward fall for Stage 1, toward moderate for Stage 2)
      - mitigating   = SHAP < 0 (toward no-fall for Stage 1, toward mild for Stage 2)
    """
    final_class = patient_info["final_prediction"]

    if patient_info["has_severity_assessment"]:
        stage = 2
        contributing = _displayable(patient_info["severity_increasing_features"])
        mitigating = _displayable(patient_info["severity_decreasing_features"])
    else:
        stage = 1
        contributing = _displayable(patient_info["risk_increasing_features"])
        mitigating = _displayable(patient_info["risk_decreasing_features"])

    c, c_shap = _pipe_shap(contributing)
    m, m_shap = _pipe_shap(mitigating)
    supporting_factors, opposing_factors = _interpretation_factor_names(patient_info)

    return {
        "patient_id":        patient_id,
        "category":          _CATEGORY[final_class],
        "stage":             stage,
        "model":             model,
        "temperature":       temperature,
        "run":               run,
        "explanation":       explanation,
        "contributing":      c,
        "contributing_shap": c_shap,
        "mitigating":        m,
        "mitigating_shap":   m_shap,
        "supporting_factors": "|".join(supporting_factors),
        "opposing_factors":   "|".join(opposing_factors),
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
        patient_id = int(patient_ids.iloc[patient_idx][PATIENT_ID_COLUMN])
        final_class = int(y_pred_final[patient_idx])
        patient_info = build_patient_explanation_data_full(patient_idx)

        for temp in temperatures:
            for run in range(1, n_runs + 1):
                call_num += 1
                tag = (
                    f"[{call_num}/{total}] "
                    f"patient={patient_id} ({_CATEGORY[final_class]}) "
                    f"temp={temp} run={run}"
                )
                print(tag, end=" ... ", flush=True)
                try:
                    # Reuse the exact packet already built for this patient. The saved evidence
                    # columns and the prompt now derive from the same in-memory object.
                    result = generate_explanation(
                        patient_id=patient_id,
                        patient_info=patient_info,
                        temperature=temp,
                    )
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


def _print_columns(df):
    """Print the generated columns (compact) as a quick schema confirmation."""
    print(f"\nColumns ({len(df.columns)}): {', '.join(df.columns)}")


def validate_generations(path=GENERATIONS_PATH):
    """Validate llm_generations.csv has everything needed for metric computation.

    Checks (all are errors → FAIL if triggered):
      1. Required columns present
      2. 30 patients, 10 per category
      3. Stage matches category (no_falls→1, mild/moderate→2)
      4. Empty explanations
      5. Pipe count alignment (feature names ↔ SHAP values)
      6. Each row has at least one stored factor
      7. Feature labels in feature_map.csv short_name (prerequisite for faithfulness evaluation)
      8. Duplicate (patient, temperature, run, model) rows
      9. Value domains: temperature ∈ {0, 0.3}, run contiguous from 1
     10. Category labels ∈ {no_falls, mild, moderate}
     11. Completeness: every (patient, temperature, run) present once per model (catches
         rows silently dropped by failed LLM calls)
     12. SHAP magnitude columns parse as non-negative floats
      13. contributing ∩ mitigating disjoint (a feature can't be SHAP>0 and SHAP<0)
      14. Saved supporting/opposing lists exactly match the first-table top 5 / second-table top 3
      15. Each explanation references its own patient_id (row↔explanation alignment)

    Warnings (reported but do NOT fail):
      - Patient set differs from the seed-42 selection

    Prints a PASS/FAIL report and returns True if no errors.
    """
    df = load_generations(path)
    errors = []
    warnings = []

    # Domains observed in the data — reused by the value-domain and completeness checks.
    temps_present = sorted(float(t) for t in df["temperature"].unique())
    runs_present = sorted(int(r) for r in df["run"].unique())

    # 1. Required columns
    missing_cols = [c for c in _EXPECTED_COLS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # 2. Patient counts per category
    for cat, expected_n in [("no_falls", 10), ("mild", 10), ("moderate", 10)]:
        n = df[df["category"] == cat]["patient_id"].nunique()
        if n != expected_n:
            errors.append(f"Expected {expected_n} {cat} patients, got {n}")

    # 3. Stage matches category (no_falls→Stage 1; mild/moderate→Stage 2)
    if "stage" in df.columns:
        wrong = df[
            ((df["category"] == "no_falls") & (df["stage"] != 1)) |
            ((df["category"] != "no_falls") & (df["stage"] != 2))
        ]
        if len(wrong):
            errors.append(f"{len(wrong)} rows have stage not matching category")

    # 4. Empty explanations
    empty_exp = (df["explanation"] == "").sum()
    if empty_exp:
        errors.append(f"{empty_exp} rows have empty explanations")

    # 5. Pipe count alignment between feature name and SHAP value columns
    pipe_pairs = [
        ("contributing", "contributing_shap"),
        ("mitigating",   "mitigating_shap"),
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

    # 6. Each row has at least one stored factor (contributing or mitigating)
    if {"contributing", "mitigating"}.issubset(df.columns):
        both_empty = df[(df["contributing"] == "") & (df["mitigating"] == "")]
        if len(both_empty):
            errors.append(f"{len(both_empty)} rows have no stored factors (both columns empty)")

    # 7. Feature labels in feature_map.csv short_name column.
    # The canonical evaluation vocabulary is short_name — the same label the prompt shows
    # and the LLM reproduces in its tables. Guards faithfulness/feature-consistency:
    # an unknown label can't be matched and would be misclassified as a ghost feature.
    known = set(pd.read_csv(_FEATURE_MAP_PATH)["short_name"])
    unknown = set()
    for col in ("contributing", "mitigating"):
        if col not in df.columns:
            continue
        for val in df[col]:
            if val:
                for name in val.split("|"):
                    if name and name not in known:
                        unknown.add(name)
    if unknown:
        errors.append(f"Feature labels not in feature_map.csv short_name: {sorted(unknown)[:5]}")

    # 8. Duplicate (patient, temperature, run, model) rows
    dupes = df.duplicated(["patient_id", "temperature", "run", "model"]).sum()
    if dupes:
        errors.append(f"{dupes} duplicate (patient, temperature, run, model) rows")

    # 9. Value domains
    bad_temps = [t for t in temps_present if not (np.isclose(t, 0.0) or np.isclose(t, 0.3))]
    if bad_temps:
        errors.append(f"Unexpected temperature values: {bad_temps}")
    if runs_present and runs_present != list(range(1, len(runs_present) + 1)):
        errors.append(f"Run values not contiguous from 1: {runs_present}")

    # 10. Category labels valid
    bad_cats = set(df["category"].unique()) - {"no_falls", "mild", "moderate"}
    if bad_cats:
        errors.append(f"Unexpected category labels: {sorted(bad_cats)}")

    # 11. Completeness — every (patient, temperature, run) combo present once per model.
    # Catches rows silently dropped by failed LLM calls (the duplicate check only finds extras).
    for model, mdf in df.groupby("model"):
        pids = sorted(int(p) for p in mdf["patient_id"].unique())
        have = {
            (int(p), float(t), int(r))
            for p, t, r in mdf[["patient_id", "temperature", "run"]].itertuples(index=False, name=None)
        }
        missing = [c for c in product(pids, temps_present, runs_present) if c not in have]
        if missing:
            expected_n = len(pids) * len(temps_present) * len(runs_present)
            errors.append(
                f"[{model}] {len(missing)} missing (patient,temp,run) rows "
                f"(expected {expected_n}, have {len(mdf)}); e.g. {missing[:3]}"
            )

    # 12. SHAP magnitude columns parse as non-negative floats
    for sc in ("contributing_shap", "mitigating_shap"):
        if sc not in df.columns:
            continue
        bad = 0
        for val in df[sc]:
            if val == "":
                continue
            for tok in str(val).split("|"):
                try:
                    if float(tok) < 0:
                        bad += 1
                except ValueError:
                    bad += 1
        if bad:
            errors.append(f"{sc}: {bad} non-numeric or negative SHAP value(s)")

    # 13. contributing ∩ mitigating disjoint
    if {"contributing", "mitigating"}.issubset(df.columns):
        def _overlap(row):
            c = {x for x in str(row["contributing"]).split("|") if x}
            m = {x for x in str(row["mitigating"]).split("|") if x}
            return bool(c & m)
        n_overlap = df.apply(_overlap, axis=1).sum()
        if n_overlap:
            errors.append(f"{n_overlap} rows have a feature in both contributing and mitigating")

    # 14. Explicit interpretation roles match the actual displayed-table order and prediction.
    role_columns = {"supporting_factors", "opposing_factors", "contributing", "mitigating"}
    if role_columns.issubset(df.columns):
        bad_roles = 0
        for _, row in df.iterrows():
            contributing = [x for x in str(row["contributing"]).split("|") if x]
            mitigating = [x for x in str(row["mitigating"]).split("|") if x]
            if row["category"] == "moderate":
                expected_supporting, expected_opposing = contributing[:5], mitigating[:3]
            else:
                expected_supporting, expected_opposing = mitigating[:5], contributing[:3]
            saved_supporting = [x for x in str(row["supporting_factors"]).split("|") if x]
            saved_opposing = [x for x in str(row["opposing_factors"]).split("|") if x]
            if (saved_supporting != expected_supporting or saved_opposing != expected_opposing):
                bad_roles += 1
        if bad_roles:
            errors.append(
                f"{bad_roles} rows have supporting/opposing lists that do not match the displayed "
                "first-table top 5 / second-table top 3"
            )

    # 15. Each explanation references its own patient_id (row↔explanation alignment)
    misref = sum(
        1 for _, row in df.iterrows()
        if str(row["patient_id"]) not in str(row["explanation"])
    )
    if misref:
        errors.append(f"{misref} explanations do not mention their own patient_id")

    # Warning: patient set matches the seed-42 selection (identity, not just counts)
    try:
        expected_ids = {int(patient_ids.iloc[i][PATIENT_ID_COLUMN]) for i in select_patients()}
        actual_ids = {int(p) for p in df["patient_id"].unique()}
        if actual_ids != expected_ids:
            warnings.append(
                "Patient set differs from the seed-42 selection "
                f"(missing e.g. {sorted(expected_ids - actual_ids)[:3]}, "
                f"extra e.g. {sorted(actual_ids - expected_ids)[:3]})"
            )
    except Exception as exc:
        warnings.append(f"Could not verify patient identity against seed-42 selection: {exc}")

    # Report
    print(f"\nValidating: {path}")
    print(
        f"Shape: {df.shape}  |  Patients: {df['patient_id'].nunique()}  |  "
        f"Models: {list(df['model'].unique())}"
    )
    print(
        f"Temperatures: {sorted(float(t) for t in df['temperature'].unique())}  |  "
        f"Runs: {sorted(int(r) for r in df['run'].unique())}"
    )
    for w in warnings:
        print(f"\n⚠ WARNING: {w}")
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
    _print_columns(df)

    # Always validate what we just wrote, so PASS/FAIL is shown without a separate command.
    validate_generations(Path(args.output))


if __name__ == "__main__":
    main()
