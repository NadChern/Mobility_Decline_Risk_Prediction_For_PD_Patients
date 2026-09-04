"""Build a paired Gemini/template explanation dataset without making LLM calls.

The script reads previously generated Gemini explanations, selects one production output
per patient (temperature 0, run 1 by default), generates a deterministic explanation for
the same patient IDs, and writes both texts side by side. It does not modify the source CSV
or calculate comparison metrics.

Run from the project root:

    ./.venv/bin/python -m template_experiment.evaluation.build_pilot_dataset
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from explanation.explanation_builder import (
    build_patient_explanation_data_full,
    get_patient_index,
)
from explanation.llm import _interpretation_factor_names
from template_experiment import (
    METHOD_NAME,
    generate_template_explanation,
    generate_template_grouped_explanation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "evaluation_results"
    / "llm_generations_grouping_v2.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "template_experiment"
    / "outputs"
    / "pilot"
    / "pilot_reports.csv"
)

_REQUIRED_SOURCE_COLUMNS = {
    "patient_id",
    "category",
    "stage",
    "model",
    "temperature",
    "run",
    "explanation",
    "contributing",
    "contributing_shap",
    "mitigating",
    "mitigating_shap",
}

GEMINI_NORMALIZATION_VERSION = "legacy-fall-labels-to-current-v1"
EVIDENCE_VERIFICATION_VERSION = "historical-vs-current-stage-factors-v1"

# Historical SHAP magnitudes were saved to six decimal places. Allow only the maximum rounding
# error implied by that serialization when comparing them with the full-precision current packet.
_HISTORICAL_SHAP_ATOL = 5.000001e-7

# These replacements change outcome terminology only. They do not change factors, patient values,
# directions, ranks, scales, or free-text clinical claims. The untouched historical output is saved
# beside the normalized copy in `gemini_explanation_raw`.
_LEGACY_OUTCOME_REPLACEMENTS = (
    ("Fall Severity Classification: Moderate Falls",
     "Fall Severity Classification: Recurrent Fall"),
    ("Fall Severity Classification: Mild Falls",
     "Fall Severity Classification: Rare Fall"),
    ("Factors Pushing the Prediction Toward Higher Severity (Moderate)",
     "Factors Pushing the Prediction Toward Higher Severity (Recurrent Fall)"),
    ("Factors Pushing the Prediction Toward Lower Severity (Mild)",
     "Factors Pushing the Prediction Toward Lower Severity (Rare Fall)"),
    ("Moderate Fall classification", "Recurrent Fall classification"),
    ("Mild Fall classification", "Rare Fall classification"),
    ("Moderate Fall prediction", "Recurrent Fall prediction"),
    ("Mild Fall prediction", "Rare Fall prediction"),
    ("Moderate Fall category", "Recurrent Fall category"),
    ("Mild Fall category", "Rare Fall category"),
)


def normalize_gemini_outcome_terminology(text):
    """Return a comparison copy using the current Rare/Recurrent outcome terminology."""
    normalized = str(text)
    for old, new in _LEGACY_OUTCOME_REPLACEMENTS:
        normalized = normalized.replace(old, new)
    return normalized


def select_gemini_outputs(source_df, temperature=0.0, run=1):
    """Return one saved Gemini output per patient for the requested generation settings."""
    missing = sorted(_REQUIRED_SOURCE_COLUMNS - set(source_df.columns))
    if missing:
        raise ValueError(f"Gemini source is missing required columns: {missing}")

    selected = source_df[
        np.isclose(pd.to_numeric(source_df["temperature"]), temperature)
        & (pd.to_numeric(source_df["run"]) == run)
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No Gemini rows found for temperature={temperature}, run={run}."
        )

    duplicates = selected["patient_id"].duplicated(keep=False)
    if duplicates.any():
        duplicate_ids = sorted(selected.loc[duplicates, "patient_id"].unique().tolist())
        raise ValueError(
            "Expected one selected Gemini output per patient; duplicate IDs found: "
            f"{duplicate_ids}"
        )

    return selected.sort_values(["category", "patient_id"]).reset_index(drop=True)


def _pipe_list(value):
    """Split a pipe-delimited historical evidence field, preserving its recorded order."""
    return [item.strip() for item in str(value).split("|") if item.strip()]


def verify_historical_evidence_matches_current(source_row, patient_info):
    """Fail if historical Gemini evidence differs from the current artifact-derived packet.

    The historical generation file records positive-SHAP factors as ``contributing`` and
    negative-SHAP factors as ``mitigating``. This check verifies the patient, displayed stage,
    exact factor names and order, SHAP direction, and six-decimal SHAP magnitudes before the
    historical Gemini text is paired with a newly rendered template.

    Returns ``True`` for convenient recording in the paired dataset; otherwise raises
    ``ValueError`` with the first detected mismatch.
    """
    patient_id = int(source_row.patient_id)
    current_patient_id = int(patient_info["patient_id"])
    if current_patient_id != patient_id:
        raise ValueError(
            f"Evidence verification failed for patient {patient_id}: current packet belongs to "
            f"patient {current_patient_id}."
        )

    historical_stage = int(source_row.stage)
    current_stage = 2 if patient_info["has_severity_assessment"] else 1
    if historical_stage != current_stage:
        raise ValueError(
            f"Evidence verification failed for patient {patient_id}: historical stage "
            f"{historical_stage}, current stage {current_stage}."
        )

    stage_prefix = "severity" if current_stage == 2 else "risk"
    groups = (
        ("contributing", "contributing_shap", f"{stage_prefix}_increasing_features", 1),
        ("mitigating", "mitigating_shap", f"{stage_prefix}_decreasing_features", -1),
    )
    for names_column, shap_column, packet_key, expected_sign in groups:
        current_factors = [
            factor for factor in patient_info[packet_key]
            if factor.get("displayable", True)
        ]
        historical_names = _pipe_list(getattr(source_row, names_column))
        current_names = [str(factor["short_name"]) for factor in current_factors]
        if historical_names != current_names:
            raise ValueError(
                f"Evidence verification failed for patient {patient_id}, {names_column}: "
                f"historical factor names/order={historical_names!r}; "
                f"current factor names/order={current_names!r}."
            )

        historical_shap_text = _pipe_list(getattr(source_row, shap_column))
        try:
            historical_shap = [float(value) for value in historical_shap_text]
        except ValueError as exc:
            raise ValueError(
                f"Evidence verification failed for patient {patient_id}, {shap_column}: "
                "historical SHAP values are not numeric."
            ) from exc
        current_signed_shap = [float(factor["shap_contribution"]) for factor in current_factors]
        if any(expected_sign * value <= 0 for value in current_signed_shap):
            expected_direction = "positive" if expected_sign > 0 else "negative"
            raise ValueError(
                f"Evidence verification failed for patient {patient_id}, {names_column}: "
                f"current packet contains a non-{expected_direction} SHAP contribution."
            )
        current_shap_magnitudes = [abs(value) for value in current_signed_shap]
        if len(historical_shap) != len(current_shap_magnitudes) or not np.allclose(
            historical_shap,
            current_shap_magnitudes,
            rtol=0.0,
            atol=_HISTORICAL_SHAP_ATOL,
        ):
            raise ValueError(
                f"Evidence verification failed for patient {patient_id}, {shap_column}: "
                f"historical magnitudes={historical_shap!r}; "
                f"current magnitudes={current_shap_magnitudes!r}."
            )

    return True


def build_pilot_dataset(source_path=DEFAULT_SOURCE, output_path=DEFAULT_OUTPUT):
    """Generate deterministic reports for the IDs in the selected Gemini results."""
    source_path = Path(source_path)
    output_path = Path(output_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Gemini generations file not found: {source_path}")

    source_df = pd.read_csv(source_path, keep_default_na=False)
    selected = select_gemini_outputs(source_df)
    try:
        source_label = str(source_path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        source_label = str(source_path)

    rows = []
    total = len(selected)
    for position, row in enumerate(selected.itertuples(index=False), start=1):
        patient_id = int(row.patient_id)
        patient_info = build_patient_explanation_data_full(get_patient_index(patient_id))
        evidence_verified = verify_historical_evidence_matches_current(row, patient_info)
        expected_supporting, expected_opposing = _interpretation_factor_names(patient_info)
        roles_saved = hasattr(row, "supporting_factors") and hasattr(row, "opposing_factors")
        saved_supporting = (
            _pipe_list(getattr(row, "supporting_factors")) if roles_saved else expected_supporting
        )
        saved_opposing = (
            _pipe_list(getattr(row, "opposing_factors")) if roles_saved else expected_opposing
        )
        if roles_saved and (
            saved_supporting != expected_supporting or saved_opposing != expected_opposing
        ):
            raise ValueError(
                f"Interpretation-role verification failed for patient {patient_id}: "
                f"saved supporting/opposing={saved_supporting!r}/{saved_opposing!r}; "
                f"current top-5/top-3={expected_supporting!r}/{expected_opposing!r}."
            )
        template_result = generate_template_explanation(
            patient_id=patient_id,
            patient_info=patient_info,
        )
        grouped_result = generate_template_grouped_explanation(
            patient_id=patient_id,
            patient_info=patient_info,
        )
        gemini_raw = row.explanation
        gemini_normalized = normalize_gemini_outcome_terminology(gemini_raw)
        if int(template_result["patient_id"]) != patient_id:
            raise ValueError(
                f"Template returned patient {template_result['patient_id']} for requested "
                f"patient {patient_id}."
            )

        rows.append(
            {
                "patient_id": patient_id,
                "category": row.category,
                "stage": int(row.stage),
                "gemini_model": row.model,
                "source_temperature": float(row.temperature),
                "source_run": int(row.run),
                "source_file": source_label,
                "historical_evidence_matches_current": evidence_verified,
                "evidence_verification_version": EVIDENCE_VERIFICATION_VERSION,
                "contributing": row.contributing,
                "contributing_shap": row.contributing_shap,
                "mitigating": row.mitigating,
                "mitigating_shap": row.mitigating_shap,
                # Always persist the exact top-5/top-3 roles used by both interpretations. Older
                # source files predate these columns, so derive them deterministically above.
                "supporting_factors": "|".join(saved_supporting),
                "opposing_factors": "|".join(saved_opposing),
                "gemini_explanation_raw": gemini_raw,
                "gemini_explanation": gemini_normalized,
                "gemini_terminology_normalized": gemini_normalized != gemini_raw,
                "gemini_normalization_version": GEMINI_NORMALIZATION_VERSION,
                "template_method": template_result["method"],
                "template_explanation": template_result["explanation"],
                "template_grouped_method": grouped_result["method"],
                "template_grouped_explanation": grouped_result["explanation"],
            }
        )
        print(f"[{position:02d}/{total}] patient {patient_id}: paired")

    paired = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_path, index=False)

    print(f"\nSource preserved: {source_path}")
    print(f"Paired dataset:   {output_path}")
    print(f"Rows: {len(paired)} | Patients: {paired['patient_id'].nunique()}")
    print(f"Categories: {paired['category'].value_counts().to_dict()}")
    return paired


# Backward-compatible callable name for notebooks created before the folder refactor.
build_paired_dataset = build_pilot_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Pair saved Gemini temperature-0/run-1 outputs with deterministic reports."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    build_pilot_dataset(args.source, args.output)


if __name__ == "__main__":
    main()
