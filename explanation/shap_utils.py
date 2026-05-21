"""SHAP utilities for feature selection and formatting."""
import numpy as np
import pandas as pd

from .data_loader import COVERAGE_THRESHOLD


def format_feature_value(val):
    """Format feature values safely for prompts and text output."""
    # Avoid raw NaN appearing in clinician-facing text.
    if pd.isna(val):
        return "missing"
    # Keep numeric values consistent and easy to read in prompts.
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{float(val):.2f}"
    # Fall back to plain string formatting for any non-numeric value.
    return str(val)


def select_features_by_coverage(shap_vals, feature_names, patient_values, coverage_threshold=COVERAGE_THRESHOLD):
    """Select features until cumulative |SHAP| coverage threshold is met.

    Instead of selecting a fixed number of top features, we select features
    until we reach a cumulative SHAP coverage threshold (e.g., 90%).
    This allows for more flexible explanations that adapt to the distribution
    of SHAP values for each patient.

    Args:
        shap_vals: 1D array of SHAP values for one patient
        feature_names: List/Index of feature names
        patient_values: Series/array of patient feature values
        coverage_threshold: Target cumulative |SHAP| coverage (default 0.90)

    Returns:
        (positive_features, negative_features) - lists of feature dicts
    """
    # Separate positive and negative contributions
    pos_mask = shap_vals > 0
    neg_mask = shap_vals < 0

    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(neg_mask)[0]

    pos_shap = shap_vals[pos_mask]
    neg_shap = shap_vals[neg_mask]

    # Total absolute SHAP for positive and negative separately
    total_pos = np.abs(pos_shap).sum()
    total_neg = np.abs(neg_shap).sum()

    def select_until_coverage(indices, shap_subset, total_abs, threshold):
        """Select features until cumulative coverage reaches threshold."""
        if len(indices) == 0 or total_abs == 0:
            return []

        # Sort by absolute SHAP value descending
        sorted_order = np.argsort(np.abs(shap_subset))[::-1]
        sorted_indices = indices[sorted_order]
        sorted_shap = shap_subset[sorted_order]

        selected = []
        cumulative = 0.0

        for i, (idx, shap_val) in enumerate(zip(sorted_indices, sorted_shap)):
            cumulative += np.abs(shap_val)
            name = feature_names[idx]
            val = patient_values.iloc[idx] if hasattr(patient_values, 'iloc') else patient_values[idx]
            selected.append({
                "feature": name,
                "patient_value": val,
                "patient_value_formatted": format_feature_value(val),
                "shap_contribution": float(shap_val),
                "direction": "increases" if shap_val > 0 else "decreases",
            })
            if cumulative / total_abs >= threshold:
                break

        return selected

    positive_features = select_until_coverage(pos_indices, pos_shap, total_pos, coverage_threshold)
    negative_features = select_until_coverage(neg_indices, neg_shap, total_neg, coverage_threshold)

    return positive_features, negative_features
