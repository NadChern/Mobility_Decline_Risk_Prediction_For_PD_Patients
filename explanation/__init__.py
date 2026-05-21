"""Explanation pipeline for Fall Prediction Model.

This package provides explanation capabilities for the full two-stage fall prediction pipeline:
- Stage 1: No falls (0) vs Any falls (1) - Random Forest classifier
- Stage 2: Mild falls (1) vs Moderate falls (2) - LinearSVC (calibrated) - only for patients
           predicted as "any fall" by Stage 1

The explanation layer loads precomputed SHAP values and predictions from both stages
to generate patient-specific explanations.
"""
from .data_loader import STAGE2_AVAILABLE, COVERAGE_THRESHOLD, DEBUG
from .explanation_builder import (
    get_patient_index,
    build_patient_explanation_data_full,
    build_patient_explanation_data_full_by_id,
)
from .llm import generate_explanation

__all__ = [
    "STAGE2_AVAILABLE",
    "COVERAGE_THRESHOLD",
    "DEBUG",
    "get_patient_index",
    "build_patient_explanation_data_full",
    "build_patient_explanation_data_full_by_id",
    "generate_explanation",
]
