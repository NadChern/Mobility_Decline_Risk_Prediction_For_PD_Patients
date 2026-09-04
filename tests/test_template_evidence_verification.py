"""Tests for historical-versus-current evidence verification in the paired pilot builder."""

from types import SimpleNamespace
from unittest import TestCase

from template_experiment.evaluation.build_gemini_pairs import (
    verify_historical_evidence_matches_current,
)
from template_experiment.evaluation.build_pilot_dataset import DEFAULT_SOURCE


_ASSERT = TestCase()


def test_pilot_default_source_is_the_grouping_v2_generation():
    assert DEFAULT_SOURCE.name == "llm_generations_grouping_v2.csv"
    assert DEFAULT_SOURCE.parent.name == "evaluation_results"


def _factor(name, shap):
    return {
        "short_name": name,
        "shap_contribution": shap,
        "displayable": True,
    }


def _source(**changes):
    values = {
        "patient_id": 123,
        "stage": 2,
        "contributing": "Age|Gait",
        "contributing_shap": "0.123457|0.100000",
        "mitigating": "MoCA",
        "mitigating_shap": "0.234568",
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _patient_info(**changes):
    values = {
        "patient_id": 123,
        "has_severity_assessment": True,
        "severity_increasing_features": [
            _factor("Age", 0.1234567),
            _factor("Gait", 0.1),
        ],
        "severity_decreasing_features": [_factor("MoCA", -0.2345678)],
        "risk_increasing_features": [],
        "risk_decreasing_features": [],
    }
    values.update(changes)
    return values


def test_matching_historical_and_current_evidence_passes():
    assert verify_historical_evidence_matches_current(_source(), _patient_info()) is True


def test_factor_order_drift_fails():
    with _ASSERT.assertRaisesRegex(ValueError, "factor names/order"):
        verify_historical_evidence_matches_current(
            _source(contributing="Gait|Age"),
            _patient_info(),
        )


def test_shap_magnitude_drift_fails():
    with _ASSERT.assertRaisesRegex(ValueError, "historical magnitudes"):
        verify_historical_evidence_matches_current(
            _source(contributing_shap="0.900000|0.100000"),
            _patient_info(),
        )


def test_stage_drift_fails():
    with _ASSERT.assertRaisesRegex(ValueError, "historical stage 1, current stage 2"):
        verify_historical_evidence_matches_current(_source(stage=1), _patient_info())


def test_hidden_factor_is_excluded_from_displayed_evidence_comparison():
    info = _patient_info()
    info["severity_increasing_features"].append({
        **_factor("Unable to rate", 0.05),
        "displayable": False,
    })
    assert verify_historical_evidence_matches_current(_source(), info) is True
