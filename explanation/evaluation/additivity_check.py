"""Verify SHAP additivity for the current exported artifact setup.

Usage:
    python -m explanation.evaluation.additivity_check
    python -m explanation.evaluation.additivity_check --out-dir evaluation_results/additivity_check

What it does:
    1. Rebuilds the current two-stage modelling setup from `Revised_Modelling/Modelling.csv`
       using the same split/parameters documented in the repository.
    2. Recomputes the Stage 1 / Stage 2 explained cohorts in the exact feature order stored in
       `explanation_artifacts/X_test.csv`.
    3. Checks SHAP local accuracy (base value + sum of SHAP values ~= model output) per patient.
    4. Writes per-patient evidence CSVs and a compact summary CSV that can be cited in the report.

This utility verifies additivity independently from the write-time export path. It is therefore
best understood as a reproducibility / numerical-consistency check for the current artifact set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ..contract import PATIENT_ID_COLUMN
from ..export import compute_shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "Revised_Modelling" / "Modelling.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "explanation_artifacts"
DEFAULT_OUT_DIR = PROJECT_ROOT / "evaluation_results" / "additivity_check"
TARGET_COLUMN = "Falls not rel to freezing past 36 months"


def _load_reference_feature_order(artifacts_dir: Path) -> list[str]:
    x_test_path = artifacts_dir / "X_test.csv"
    if not x_test_path.exists():
        raise FileNotFoundError(
            f"Missing artifact file: {x_test_path}\n"
            "Generate explanation artifacts first so the feature order is known."
        )
    return list(pd.read_csv(x_test_path).columns)


def _fit_current_models(data_path: Path, feature_order: list[str]):
    data = pd.read_csv(data_path)
    if TARGET_COLUMN not in data.columns:
        raise KeyError(f"Target column not found in {data_path}: {TARGET_COLUMN}")
    if PATIENT_ID_COLUMN not in data.columns:
        raise KeyError(f"Patient ID column not found in {data_path}: {PATIENT_ID_COLUMN}")

    X = data[feature_order]
    y = data[TARGET_COLUMN]
    patient_ids = data[PATIENT_ID_COLUMN]

    X_train, X_test, y_train, _y_test, id_train, id_test = train_test_split(
        X,
        y,
        patient_ids,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    y_train_stage1 = (y_train != 0).astype(int)
    faller_mask = y_train.isin([1, 2])
    X_train_stage2 = X_train.loc[faller_mask]
    y_train_stage2 = (y_train.loc[faller_mask] == 2).astype(int)

    stage1_model = RandomForestClassifier(
        max_depth=8,
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=5,
        n_estimators=100,
        class_weight={0: 1, 1: 2},
        random_state=42,
    ).fit(X_train, y_train_stage1)

    stage2_model = XGBClassifier(
        colsample_bytree=1.0,
        learning_rate=0.05,
        max_depth=3,
        n_estimators=100,
        subsample=1.0,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    ).fit(X_train_stage2, y_train_stage2)

    # Enforce artifact feature order for the explained test matrix.
    X_test = X_test[feature_order]
    routed_mask = stage1_model.predict(X_test) == 1
    X_routed = X_test.loc[routed_mask]
    routed_ids = id_test.loc[routed_mask]

    return {
        "stage1_model": stage1_model,
        "stage2_model": stage2_model,
        "X_train_stage1": X_train,
        "X_train_stage2": X_train_stage2,
        "X_test": X_test,
        "X_routed": X_routed,
        "patient_ids_test": id_test.reset_index(drop=True),
        "patient_ids_routed": routed_ids.reset_index(drop=True),
    }


def _stage1_additivity(stage1_model, X_train, X_test, patient_ids):
    explainer = shap.Explainer(stage1_model, X_train)
    explanation = explainer(X_test)

    base = np.asarray(explanation.base_values)[:, 1]
    shap_sum = np.asarray(explanation.values)[:, :, 1].sum(axis=1)
    model_output = stage1_model.predict_proba(X_test)[:, 1]
    reconstructed = base + shap_sum
    abs_error = np.abs(reconstructed - model_output)

    return pd.DataFrame(
        {
            "patient_id": patient_ids.to_numpy(),
            "base_value": base,
            "shap_sum": shap_sum,
            "reconstructed_output": reconstructed,
            "model_output": model_output,
            "absolute_error": abs_error,
        }
    )


def _stage2_additivity(stage2_model, X_train_stage2, X_routed, patient_ids_routed):
    if len(X_routed) == 0:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "base_value",
                "shap_sum",
                "reconstructed_output",
                "model_output",
                "absolute_error",
            ]
        )

    explainer = shap.Explainer(stage2_model, X_train_stage2)
    explanation = explainer(X_routed)

    base = np.asarray(explanation.base_values)
    shap_sum = np.asarray(explanation.values).sum(axis=1)
    model_output = stage2_model.predict(X_routed, output_margin=True)
    reconstructed = base + shap_sum
    abs_error = np.abs(reconstructed - model_output)

    return pd.DataFrame(
        {
            "patient_id": patient_ids_routed.to_numpy(),
            "base_value": base,
            "shap_sum": shap_sum,
            "reconstructed_output": reconstructed,
            "model_output": model_output,
            "absolute_error": abs_error,
        }
    )


def _summary_row(stage: str, space: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "stage": stage,
            "space": space,
            "n_patients": 0,
            "max_absolute_error": np.nan,
            "mean_absolute_error": np.nan,
            "median_absolute_error": np.nan,
        }
    return {
        "stage": stage,
        "space": space,
        "n_patients": int(len(df)),
        "max_absolute_error": float(df["absolute_error"].max()),
        "mean_absolute_error": float(df["absolute_error"].mean()),
        "median_absolute_error": float(df["absolute_error"].median()),
    }


def run_additivity_check(
    data_path: Path = DEFAULT_DATA_PATH,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
):
    warnings.filterwarnings("ignore")

    feature_order = _load_reference_feature_order(artifacts_dir)
    fitted = _fit_current_models(data_path, feature_order)

    # Optional cross-check: the recomputed SHAP arrays should match the saved artifacts if the
    # current export setup is unchanged. This is not the additivity test itself, but it helps
    # confirm we are verifying the same artifact-producing setup.
    saved_stage1 = np.load(artifacts_dir / "shap_values_stage1.npy")
    saved_stage2 = np.load(artifacts_dir / "shap_values_stage2.npy")
    recomputed_stage1 = compute_shap(
        fitted["stage1_model"], fitted["X_train_stage1"], fitted["X_test"]
    )
    recomputed_stage2 = compute_shap(
        fitted["stage2_model"], fitted["X_train_stage2"], fitted["X_routed"]
    )

    stage1_df = _stage1_additivity(
        fitted["stage1_model"],
        fitted["X_train_stage1"],
        fitted["X_test"],
        fitted["patient_ids_test"],
    )
    stage2_df = _stage2_additivity(
        fitted["stage2_model"],
        fitted["X_train_stage2"],
        fitted["X_routed"],
        fitted["patient_ids_routed"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = out_dir / "additivity_stage1_patient.csv"
    stage2_path = out_dir / "additivity_stage2_patient.csv"
    summary_path = out_dir / "additivity_summary.csv"

    stage1_df.to_csv(stage1_path, index=False)
    stage2_df.to_csv(stage2_path, index=False)

    summary = pd.DataFrame(
        [
            {
                **_summary_row("stage1", "probability", stage1_df),
                "saved_shap_matches_recomputed": bool(np.allclose(saved_stage1, recomputed_stage1)),
            },
            {
                **_summary_row("stage2", "raw_margin", stage2_df),
                "saved_shap_matches_recomputed": bool(np.allclose(saved_stage2, recomputed_stage2)),
            },
        ]
    )
    summary.to_csv(summary_path, index=False)

    print("Additivity check complete.")
    print(f"  Stage 1 patient-level evidence: {stage1_path}")
    print(f"  Stage 2 patient-level evidence: {stage2_path}")
    print(f"  Summary: {summary_path}")
    print()
    for row in summary.to_dict(orient="records"):
        print(
            f"{row['stage']}: n={row['n_patients']}  "
            f"max_abs_error={row['max_absolute_error']:.6e}  "
            f"mean_abs_error={row['mean_absolute_error']:.6e}  "
            f"saved_shap_matches_recomputed={row['saved_shap_matches_recomputed']}"
        )

    return {
        "stage1_patient": stage1_path,
        "stage2_patient": stage2_path,
        "summary": summary_path,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Verify SHAP additivity for the current setup.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to modelling data CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Path to explanation artifacts directory (default: {DEFAULT_ARTIFACTS_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for evidence CSVs (default: {DEFAULT_OUT_DIR})",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    run_additivity_check(
        data_path=args.data_path,
        artifacts_dir=args.artifacts_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
