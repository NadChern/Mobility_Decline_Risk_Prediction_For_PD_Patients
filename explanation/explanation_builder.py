"""Build patient explanation data from SHAP values and predictions."""
import re

import pandas as pd

from .contract import PATIENT_ID_COLUMN
from .data_loader import (
    COVERAGE_THRESHOLD,
    shap_values,
    X_test,
    y_pred_proba,
    patient_ids,
    feature_map_lookup,
    STAGE2_AVAILABLE,
    routing_mask,
    shap_values_stage2,
    y_pred_final,
    get_stage2_index_map,
)
from .shap_utils import select_features_by_coverage


def get_patient_index(patient_id):
    """Return the aligned row index for a PATNO in saved artifacts."""
    matches = patient_ids.index[patient_ids[PATIENT_ID_COLUMN] == patient_id].tolist()
    if not matches:
        # Stop early if the requested PATNO is not in the saved test-set IDs.
        raise KeyError(f"Patient ID {patient_id} not found in patient_ids_test.csv.")
    if len(matches) > 1:
        # Each patient should map to exactly one saved artifact row.
        raise ValueError(f"Patient ID {patient_id} appears multiple times in patient_ids_test.csv.")
    return matches[0]


def get_stage2_index(patient_idx):
    """Map test set index to Stage 2 artifact index.

    Args:
        patient_idx: Index in the test set (X_test row number)

    Returns:
        Index in Stage 2 artifacts, or None if patient was not routed to Stage 2
    """
    if not STAGE2_AVAILABLE:
        return None
    stage2_index_map = get_stage2_index_map()
    if stage2_index_map is None:
        return None
    return stage2_index_map.get(patient_idx)


def get_final_prediction_label(final_class):
    """Convert 3-class prediction to human-readable label.

    Args:
        final_class: 0, 1, or 2

    Returns:
        Human-readable prediction label
    """
    labels = {
        0: "No falls",
        1: "Rare Fall",
        2: "Recurrent Fall",
    }
    return labels.get(int(final_class), f"Unknown class {final_class}")


def get_feature_metadata(feature_name):
    """Return concise human-readable metadata for a model feature."""
    return feature_map_lookup.get(
        feature_name,
        {
            "description": feature_name,
            "data_type": "unknown",
            "value_encoding_or_scale": "",
        },
    )


# Value labels that mean "not assessed" — such factors are dropped from the explanation.
_MISSING_LABEL_PATTERNS = ("unable to rate", "not rated", "not assessed", "cannot rate")


def _format_display_value(value, metadata):
    """Format a feature value for display based on its data type (ints vs 2-dp)."""
    try:
        float_val = float(value)
    except (ValueError, TypeError):
        return str(value)
    if metadata.get("data_type", "") in (
        "continuous_integer", "ordinal_numeric", "ordinal_categorical", "binary_categorical"
    ):
        return str(int(float_val))
    return str(int(float_val)) if float_val == int(float_val) else f"{float_val:.2f}"


def _value_label(scale_str, value):
    """Return the scale label for an integer value (e.g. '101 = unable to rate' -> 'unable to rate')."""
    try:
        int_val = int(value) if float(value) == int(float(value)) else None
    except (ValueError, TypeError):
        return None
    if int_val is None:
        return None
    match = re.search(rf"\b{int_val}\s*=\s*([^;]+)", scale_str)
    return match.group(1).strip() if match else None


def _is_displayable(value, metadata):
    """False for NaN, or an 'unable to rate' sentinel (feature-aware via the scale label)."""
    if pd.isna(value):
        return False
    label = _value_label(metadata.get("value_encoding_or_scale", ""), value)
    if label and any(p in label.lower() for p in _MISSING_LABEL_PATTERNS):
        return False
    return True


def enrich_factor(factor):
    """Attach display fields to a SHAP factor dict.

    Adds: short_name, display_value (data-type formatted), interpretation
    ("description. scale"), and displayable (False for NaN / unable-to-rate). The original
    SHAP fields (feature, patient_value, shap_contribution, direction) are preserved.
    """
    meta = get_feature_metadata(factor["feature"])
    description = (meta.get("description") or "").strip()
    scale = (meta.get("value_encoding_or_scale") or "").strip()
    if description and scale:
        interpretation = f"{description}. {scale}"
    else:
        interpretation = description or scale
    return {
        **factor,
        "short_name": (meta.get("short_name") or description or factor["feature"]).strip(),
        "display_value": _format_display_value(factor["patient_value"], meta),
        "interpretation": interpretation,
        "displayable": _is_displayable(factor["patient_value"], meta),
    }


def build_patient_explanation_data_full(patient_idx, coverage_threshold=COVERAGE_THRESHOLD):
    """Build structured per-patient explanation data for both stages.

    Returns a flattened dictionary with explanation data for both stages.

    Args:
        patient_idx: Row index in the test set
        coverage_threshold: Target cumulative |SHAP| coverage (default 0.90)

    Returns:
        Dictionary with full two-stage explanation data (flattened structure)
    """
    # Get Stage 1 data
    patient_shap_s1 = shap_values[patient_idx]
    patient_values = X_test.iloc[patient_idx]
    fall_probability = float(y_pred_proba[patient_idx])
    patient_id = int(patient_ids.iloc[patient_idx][PATIENT_ID_COLUMN])

    # Stage 1 features using coverage-based selection
    risk_increasing_features, risk_decreasing_features = select_features_by_coverage(
        patient_shap_s1, X_test.columns, patient_values, coverage_threshold
    )

    # Determine routing and final prediction
    has_severity_assessment = False
    severity_increasing_features = []
    severity_decreasing_features = []
    final_prediction = 0  # Default: no fall

    # Stage 2 artifacts are required (module raises error at import if missing)
    has_severity_assessment = bool(routing_mask[patient_idx])
    final_prediction = int(y_pred_final[patient_idx])

    if has_severity_assessment:
        s2_idx = get_stage2_index(patient_idx)
        if s2_idx is not None:
            patient_shap_s2 = shap_values_stage2[s2_idx]

            # Stage 2 features using coverage-based selection
            severity_increasing_features, severity_decreasing_features = select_features_by_coverage(
                patient_shap_s2, X_test.columns, patient_values, coverage_threshold
            )

    # Attach display fields (short_name, display_value, interpretation, displayable) once,
    # centrally — so consumers (the LLM prompt) don't re-retrieve from feature_map.
    risk_increasing_features = [enrich_factor(f) for f in risk_increasing_features]
    risk_decreasing_features = [enrich_factor(f) for f in risk_decreasing_features]
    severity_increasing_features = [enrich_factor(f) for f in severity_increasing_features]
    severity_decreasing_features = [enrich_factor(f) for f in severity_decreasing_features]

    return {
        "patient_id": patient_id,
        "patient_idx": patient_idx,
        "final_prediction": final_prediction,
        "final_prediction_label": get_final_prediction_label(final_prediction),
        "fall_probability": fall_probability,
        "risk_increasing_features": risk_increasing_features,
        "risk_decreasing_features": risk_decreasing_features,
        "has_severity_assessment": has_severity_assessment,
        "severity_increasing_features": severity_increasing_features,
        "severity_decreasing_features": severity_decreasing_features,
    }


def build_patient_explanation_data_full_by_id(patient_id, coverage_threshold=COVERAGE_THRESHOLD):
    """Build full two-stage explanation data using stable patient ID lookup."""
    patient_idx = get_patient_index(patient_id)
    return build_patient_explanation_data_full(patient_idx, coverage_threshold=coverage_threshold)
