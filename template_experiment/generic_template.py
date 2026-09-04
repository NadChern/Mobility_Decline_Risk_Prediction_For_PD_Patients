"""Generic deterministic explanation renderer.

The renderer consumes the same enriched patient packet as the LLM prompt builder. It does not
select, infer, regroup, add, or remove evidence. This provides a strong reproducible baseline for
the template-versus-LLM experiment.
"""

from explanation.explanation_builder import (
    build_patient_explanation_data_full,
    get_patient_index,
)

METHOD_NAME = "deterministic-template-v1"

_SEVERITY_LABEL = {1: "Rare Fall", 2: "Recurrent Fall"}
_OUTCOME_PHRASE = {
    0: "No Fall classification",
    1: "Rare Fall classification",
    2: "Recurrent Fall classification",
}

_HIGHER = "Factors Pushing the Prediction Toward Higher Severity (Recurrent Fall)"
_LOWER = "Factors Pushing the Prediction Toward Lower Severity (Rare Fall)"
_TOWARD_FALL = "Factors Pushing the Prediction Toward Fall"
_TOWARD_NOFALL = "Factors Pushing the Prediction Toward No Fall"


def _displayable(factors):
    return [factor for factor in factors if factor.get("displayable", True)]


def _ordered_tables(patient_info):
    """Return prediction-supporting then opposing factors in the supplied SHAP order."""
    prediction = int(patient_info["final_prediction"])
    if patient_info["has_severity_assessment"]:
        higher = (_HIGHER, patient_info["severity_increasing_features"])
        lower = (_LOWER, patient_info["severity_decreasing_features"])
        tables = [higher, lower] if prediction == 2 else [lower, higher]
    else:
        toward_fall = (_TOWARD_FALL, patient_info["risk_increasing_features"])
        toward_no_fall = (_TOWARD_NOFALL, patient_info["risk_decreasing_features"])
        tables = [toward_no_fall, toward_fall]
    return [(heading, _displayable(factors)) for heading, factors in tables]


def _prediction_lines(patient_info):
    prediction = int(patient_info["final_prediction"])
    fall_label = "Fall" if prediction in (1, 2) else "No Fall"
    lines = [f"- Fall Classification: {fall_label}"]
    if patient_info["has_severity_assessment"]:
        lines.append(
            f"- Fall Severity Classification: {_SEVERITY_LABEL[prediction]}"
        )
    return lines


def _escape_cell(value):
    """Keep copied evidence inside one Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _render_table(heading, factors):
    lines = [
        heading,
        "| # | Factor | Patient Value | Interpretation / Scale |",
        "| --- | --- | --- | --- |",
    ]
    if not factors:
        return f"{heading}\n\nNo qualifying factors."
    for number, factor in enumerate(factors, start=1):
        lines.append(
            "| "
            f"{number} | {_escape_cell(factor['short_name'])} | "
            f"{_escape_cell(factor['display_value'])} | "
            f"{_escape_cell(factor['interpretation'])} |"
        )
    return "\n".join(lines)


def _join_names(names):
    if not names:
        return "no qualifying factors"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _interpretation(patient_info, first_factors, second_factors):
    """Create a generic, deterministic synthesis without feature-specific rules."""
    prediction = int(patient_info["final_prediction"])
    outcome = _OUTCOME_PHRASE[prediction]
    supporting = _join_names([f["short_name"] for f in first_factors[:5]])
    opposing = _join_names([f["short_name"] for f in second_factors[:3]])

    first_sentence = (
        f"Within this model, the {outcome} was primarily associated with {supporting}."
    )
    if second_factors:
        second_sentence = (
            f"Factors supporting the opposite direction included {opposing}; however, these "
            f"did not outweigh the factors supporting the {outcome}."
        )
    else:
        second_sentence = "No qualifying factors supported the opposite direction."
    return f"{first_sentence} {second_sentence}"


def render_template_explanation(patient_info, interpretation_fn=_interpretation):
    """Render a deterministic Markdown document from one enriched patient packet.

    ``interpretation_fn`` produces the Model Interpretation paragraph; the default is the generic
    factor-listing baseline. The domain-enriched arm injects a grouping interpretation instead, so
    everything except that one paragraph is identical across the two template arms.
    """
    patient_id = int(patient_info["patient_id"])
    (first_heading, first_factors), (second_heading, second_factors) = _ordered_tables(
        patient_info
    )

    sections = [
        f"Clinical Fall Risk Summary for Patient ID: {patient_id}",
        "Model Prediction\n" + "\n".join(_prediction_lines(patient_info)),
        (
            "Factors are ordered from most to least influential based on their "
            "contribution to the model prediction."
        ),
        _render_table(first_heading, first_factors),
        _render_table(second_heading, second_factors),
        "Model Interpretation\n"
        + interpretation_fn(patient_info, first_factors, second_factors),
        (
            "Clinical Note\n"
            "- Model predictions are intended to support clinical review and should not "
            "replace clinical judgment.\n"
            "- The factors shown above identify the patient characteristics that most "
            "influenced the model prediction and provide insight into the model’s reasoning.\n"
            "- For definitions and interpretation of all variables used by the model, please "
            "refer to the Feature Reference Guide (feature_map.xlsx)."
        ),
    ]
    return "\n\n".join(sections)


def generate_template_explanation(patient_id=None, patient_info=None):
    """Return a deterministic explanation, optionally from a prebuilt shared packet."""
    if patient_info is None:
        if patient_id is None:
            raise ValueError("Provide either patient_id or patient_info.")
        patient_info = build_patient_explanation_data_full(get_patient_index(patient_id))
    elif patient_id is not None and int(patient_info["patient_id"]) != int(patient_id):
        raise ValueError(
            f"patient_id {patient_id} does not match patient_info patient "
            f"{patient_info['patient_id']}."
        )
    return {
        "method": METHOD_NAME,
        "patient_id": int(patient_info["patient_id"]),
        "evidence": patient_info,
        "explanation": render_template_explanation(patient_info),
    }
