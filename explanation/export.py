"""Generate the explanation artifacts from a trained two-stage model.

This is the single source of truth for producing everything the explanation
pipeline consumes. Any modelling notebook (the original or a future revision)
calls :func:`export_explanation_artifacts` instead of hand-writing SHAP/save
logic, so swapping the underlying model never requires editing export code.

The SHAP helper is model-agnostic: it leans on ``shap.Explainer``'s dispatcher
(Tree / Linear / Kernel) and normalises the output to a 2-D
``(n_samples, n_features)`` array for the positive class, so a Random Forest,
XGBoost or linear model all work without special-casing.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from .contract import ARTIFACT_FILENAMES, MANIFEST_FILENAME, PATIENT_ID_COLUMN

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "explanation_artifacts"
DEFAULT_FEATURE_MAP_PATH = BASE_DIR / "feature_map.csv"


def compute_shap(model, X_background, X, positive_class=1):
    """Compute SHAP values for ``X`` and return a 2-D array for one class.

    Works for any estimator ``shap.Explainer`` can dispatch on (tree, linear,
    or a fitted pipeline). Output is always ``(n_samples, n_features)``:

    * binary tree models (e.g. XGBoost) already return ``(N, F)`` -> used as-is
    * classifiers returning ``(N, F, n_classes)`` (e.g. RandomForest) ->
      sliced at ``positive_class``
    * the legacy ``list[array]`` form -> ``[positive_class]`` is taken

    ``shap.Explainer`` aligns ``X`` and ``X_background`` by column position, so a
    mismatched order would silently produce wrong SHAP values. Guard against it.
    """
    if hasattr(X_background, "columns") and hasattr(X, "columns"):
        if list(X_background.columns) != list(X.columns):
            raise ValueError(
                "compute_shap: X_background and X must have identical column order.\n"
                f"  X_background: {list(X_background.columns)}\n"
                f"  X:            {list(X.columns)}"
            )

    explainer = shap.Explainer(model, X_background)
    explanation = explainer(X)
    values = explanation.values

    if isinstance(values, list):
        values = values[positive_class]

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, positive_class]
    elif values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP values shape {values.shape}; expected 2-D or 3-D."
        )
    return values


def _explainer_name(model, X_background):
    """Name of the SHAP explainer the dispatcher picks (for provenance only).

    Constructing the explainer is cheap — it does not compute SHAP values.
    """
    return type(shap.Explainer(model, X_background)).__name__


def _validate(
    X_test,
    feature_names,
    shap_stage1,
    y_pred_proba,
    routing_mask,
    shap_stage2,
    y_pred_s2,
    y_pred_final,
):
    """Fail loudly *before* anything is written if the contract is violated."""
    n = len(X_test)

    # --- Feature alignment (by set; lookups are name-based, order-independent) ---
    cols = set(X_test.columns)
    mapped = set(feature_names)
    missing = cols - mapped
    extra = mapped - cols
    if missing:
        raise ValueError(
            f"Features missing from feature_map.csv (no metadata): {sorted(missing)}"
        )
    if extra:
        raise ValueError(
            f"feature_map.csv has features not present in X_test: {sorted(extra)}"
        )

    # --- Row-count alignment of all (N, *) artifacts ---
    if shap_stage1.shape[0] != n:
        raise ValueError("shap_values_stage1 row count != X_test.")
    if len(y_pred_proba) != n:
        raise ValueError("y_pred_proba length != X_test.")
    if len(routing_mask) != n:
        raise ValueError("routing_mask length != X_test.")
    if len(y_pred_final) != n:
        raise ValueError("y_pred_final length != X_test.")

    # --- Stage 2 shape/consistency ---
    n_routed = int(routing_mask.sum())
    if shap_stage2.shape[0] != n_routed:
        raise ValueError(
            f"shap_values_stage2 rows {shap_stage2.shape[0]} != routed {n_routed}."
        )
    if len(y_pred_s2) != n_routed:
        raise ValueError("y_pred_s2 length != routed patients.")
    if routing_mask.dtype != bool:
        raise ValueError("routing_mask must be boolean.")

    routed_idx = np.where(routing_mask)[0]
    expected_final = np.where(y_pred_s2 == 0, 1, 2)
    if not (y_pred_final[routed_idx].astype(int) == expected_final).all():
        raise ValueError("y_pred_final does not match y_pred_s2 mapping for routed.")
    non_routed_idx = np.where(~routing_mask)[0]
    if not (y_pred_final[non_routed_idx] == 0).all():
        raise ValueError("Non-routed patients must have final prediction 0.")
    if not set(np.unique(y_pred_s2)).issubset({0, 1}):
        raise ValueError("y_pred_s2 must contain only 0 (mild) or 1 (moderate).")
    if not set(np.unique(y_pred_final)).issubset({0, 1, 2}):
        raise ValueError("y_pred_final must contain only 0, 1 or 2.")

    # --- SHAP finiteness / sanity ---
    for name, arr in [("shap_stage1", shap_stage1), ("shap_stage2", shap_stage2)]:
        if arr.size and (np.isnan(arr).any() or np.isinf(arr).any()):
            raise ValueError(f"{name} contains NaN/Inf.")
    if shap_stage1.size and np.all(shap_stage1 == 0):
        raise ValueError("shap_stage1 is all zeros.")
    if n_routed and np.all(shap_stage2 == 0):
        raise ValueError("shap_stage2 is all zeros.")


def export_explanation_artifacts(
    final_s1,
    final_s2,
    X_train,
    X_test,
    X_train_12,
    patient_ids,
    cols,
    out_dir=DEFAULT_ARTIFACTS_DIR,
    feature_map_path=DEFAULT_FEATURE_MAP_PATH,
    positive_class_s1=1,
    positive_class_s2=1,
    dataset_name=None,
    source_notebook=None,
):
    """Compute SHAP, validate, and write all 8 explanation artifacts.

    Parameters
    ----------
    final_s1, final_s2 : fitted estimators
        Stage 1 (any-fall) and Stage 2 (mild-vs-moderate) models.
    X_train, X_test : DataFrames
        Stage 1 background and the test set to explain.
    X_train_12 : DataFrame
        Stage 2 background (class {1,2} training rows).
    patient_ids : Series
        PATNO aligned row-for-row with X_test (e.g. ``id_test`` from the split).
    cols : list[str]
        Feature column order to enforce on the saved X_test.
    dataset_name, source_notebook : str, optional
        Recorded in manifest.json for provenance (the caller knows these; the
        export does not). Purely informational — the consumer never reads them.

    Returns
    -------
    dict
        Summary with artifact shapes and the routed-patient count.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test = X_test[cols].copy()
    feature_names = pd.read_csv(feature_map_path)["feature_name"].tolist()

    # --- Stage 1 ---
    shap_stage1 = compute_shap(final_s1, X_train, X_test, positive_class_s1)
    y_pred_proba = final_s1.predict_proba(X_test)[:, positive_class_s1]

    # --- Routing + Stage 2 ---
    routing_mask = np.asarray(final_s1.predict(X_test) == 1, dtype=bool)
    routed_idx = np.where(routing_mask)[0]
    X_routed = X_test.iloc[routed_idx]

    if len(routed_idx):
        shap_stage2 = compute_shap(final_s2, X_train_12, X_routed, positive_class_s2)
        y_pred_s2 = np.asarray(final_s2.predict(X_routed), dtype=np.int64)
    else:
        shap_stage2 = np.empty((0, len(cols)))
        y_pred_s2 = np.empty((0,), dtype=np.int64)

    y_pred_final = np.zeros(len(X_test), dtype=np.int64)
    y_pred_final[routed_idx] = np.where(y_pred_s2 == 0, 1, 2)

    # --- Validate the full contract before writing anything ---
    _validate(
        X_test, feature_names, shap_stage1, y_pred_proba,
        routing_mask, shap_stage2, y_pred_s2, y_pred_final,
    )

    # --- Determinism: recomputing SHAP must reproduce the same values ---
    recompute_s1 = compute_shap(final_s1, X_train, X_test, positive_class_s1)
    if not np.allclose(shap_stage1, recompute_s1):
        raise ValueError("Stage 1 SHAP values are not deterministic!")
    s1_max_diff = float(np.max(np.abs(shap_stage1 - recompute_s1)))
    s1_mean_diff = float(np.mean(np.abs(shap_stage1 - recompute_s1)))

    if len(routed_idx):
        recompute_s2 = compute_shap(final_s2, X_train_12, X_routed, positive_class_s2)
        if not np.allclose(shap_stage2, recompute_s2):
            raise ValueError("Stage 2 SHAP values are not deterministic!")
        s2_max_diff = float(np.max(np.abs(shap_stage2 - recompute_s2)))
        s2_mean_diff = float(np.mean(np.abs(shap_stage2 - recompute_s2)))
    else:
        s2_max_diff = s2_mean_diff = 0.0

    # --- Write artifacts ---
    f = ARTIFACT_FILENAMES
    np.save(out_dir / f["shap_stage1"], shap_stage1)
    X_test.to_csv(out_dir / f["x_test"], index=False)
    np.save(out_dir / f["y_pred_proba"], y_pred_proba)
    patient_ids.rename(PATIENT_ID_COLUMN).to_csv(out_dir / f["patient_ids"], index=False)
    np.save(out_dir / f["routing_mask"], routing_mask)
    np.save(out_dir / f["shap_stage2"], shap_stage2)
    np.save(out_dir / f["y_pred_s2"], y_pred_s2)
    np.save(out_dir / f["y_pred_final"], y_pred_final)

    final_counts = {
        "no_fall": int((y_pred_final == 0).sum()),
        "mild": int((y_pred_final == 1).sum()),
        "moderate": int((y_pred_final == 2).sum()),
    }

    # --- Write provenance manifest (write-only; consumer never reads it) ---
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "source_notebook": source_notebook,
        "stage1_model": type(final_s1).__name__,
        "stage2_model": type(final_s2).__name__,
        "stage1_shap_explainer": _explainer_name(final_s1, X_train),
        "stage2_shap_explainer": (
            _explainer_name(final_s2, X_train_12) if len(routed_idx) else None
        ),
        "features": list(cols),
        "n_test": len(X_test),
        "n_routed": int(routing_mask.sum()),
        "final_prediction_counts": final_counts,
    }
    with open(out_dir / MANIFEST_FILENAME, "w") as fh:
        json.dump(manifest, fh, indent=2)

    summary = {
        "out_dir": str(out_dir.resolve()),
        "n_test": len(X_test),
        "n_routed": int(routing_mask.sum()),
        "shap_stage1_shape": shap_stage1.shape,
        "shap_stage2_shape": shap_stage2.shape,
        "final_counts": final_counts,
        "manifest": str((out_dir / MANIFEST_FILENAME).resolve()),
    }

    print("Explanation artifacts written to:", summary["out_dir"])
    print(f"  Test patients      : {summary['n_test']}")
    print(f"  Routed to Stage 2  : {summary['n_routed']}")
    print(f"  shap_stage1 shape  : {summary['shap_stage1_shape']}")
    print(f"  shap_stage2 shape  : {summary['shap_stage2_shape']}")
    print(f"  Final prediction   : {summary['final_counts']}")
    print(f"  Manifest           : {summary['manifest']}")
    print("SHAP determinism (recompute vs saved, expected 0):")
    print(f"  Stage 1: max|diff|={s1_max_diff:.2e}  mean|diff|={s1_mean_diff:.2e}")
    print(f"  Stage 2: max|diff|={s2_max_diff:.2e}  mean|diff|={s2_mean_diff:.2e}")
    print("All write-time validation and determinism checks passed.")
    return summary
