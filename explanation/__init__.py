"""Explanation pipeline for Fall Prediction Model.

This package provides explanation capabilities for the full two-stage fall prediction pipeline:
- Stage 1: No falls (0) vs Any falls (1) - Random Forest classifier
- Stage 2: Rare Fall (1) vs Recurrent Fall (2) - XGBoost classifier - only for patients
           predicted as "any fall" by Stage 1

The explanation layer loads precomputed SHAP values and predictions from both stages
to generate patient-specific explanations.

Imports are resolved lazily (PEP 562) so that importing the package — e.g. to run the
*producer* ``explanation.export`` from a modelling notebook — does NOT load ``data_loader``,
which reads the artifacts at import time. This avoids a chicken-and-egg failure when the
artifacts do not exist yet (the producer is what creates them).
"""

__all__ = [
    "STAGE2_AVAILABLE",
    "COVERAGE_THRESHOLD",
    "DEBUG",
    "get_patient_index",
    "build_patient_explanation_data_full",
    "build_patient_explanation_data_full_by_id",
    "generate_explanation",
    "export_explanation",
    "save_document",
]

_LAZY = {
    "STAGE2_AVAILABLE": "data_loader",
    "COVERAGE_THRESHOLD": "data_loader",
    "DEBUG": "data_loader",
    "get_patient_index": "explanation_builder",
    "build_patient_explanation_data_full": "explanation_builder",
    "build_patient_explanation_data_full_by_id": "explanation_builder",
    "generate_explanation": "llm",
    "export_explanation": "render",
    "save_document": "render",
}


def __getattr__(name):
    """Resolve public names on first access (PEP 562).

    Touching any of the consumer names below imports the relevant submodule, which is
    when ``data_loader`` (and thus the artifacts) is actually required.
    """
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    submodule = importlib.import_module(f".{module}", __name__)
    return getattr(submodule, name)


def __dir__():
    return sorted(__all__)
