"""Mutation tests for the experiment-local metric implementations."""

from pathlib import Path

import numpy as np
import pandas as pd

from template_experiment.evaluation.completeness_comparison import compute_patient_completeness
from template_experiment.evaluation.fidelity_comparison import compute_patient_fidelity, summarize
from template_experiment.evaluation.structure_comparison import compute_patient_structure
from template_experiment.evaluation.unsupported_information_comparison import (
    compute_patient_unsupported,
)


PAIRED = (Path(__file__).resolve().parents[1]
          / "template_experiment/outputs/pilot/pilot_reports.csv")


def _source():
    return pd.read_csv(PAIRED, keep_default_na=False).iloc[0].copy()


def _score(row):
    fidelity = compute_patient_fidelity(
        row, "template", "template_method", "template_explanation")
    completeness = compute_patient_completeness(
        row, "template", "template_method", "template_explanation")
    unsupported = compute_patient_unsupported(
        row, "template", "template_method", "template_explanation")
    return fidelity, completeness, unsupported


def _first_data_row_index(lines):
    return next(i for i, line in enumerate(lines) if line.startswith("| 1 |"))


def test_clean_template_reaches_structured_ceiling():
    fidelity, completeness, unsupported = _score(_source())
    assert fidelity["outcome_correct"]
    assert fidelity["direction_accuracy"] == 1
    assert fidelity["value_fidelity"] == 1
    assert fidelity["scale_fidelity"] == 1
    assert fidelity["rank_order_fidelity"] == 1
    assert completeness["factor_recall"] == 1
    assert completeness["expected_table_field_completeness"] == 1
    assert unsupported["automated_unsupported_information_free"]


def test_missing_factor_hits_completeness_not_conditional_fidelity():
    row = _source()
    lines = row.template_explanation.splitlines()
    lines.pop(_first_data_row_index(lines))
    row.template_explanation = "\n".join(lines)
    fidelity, completeness, _ = _score(row)
    # The factors still shown remain correct, so conditional fidelity is unaffected...
    assert fidelity["value_fidelity"] == 1
    assert fidelity["direction_accuracy"] == 1
    # ...the omission is a completeness failure.
    assert completeness["factor_recall"] < 1
    assert completeness["expected_table_field_completeness"] < 1


def test_direction_swap_is_detected():
    row = _source()
    text = row.template_explanation
    higher = "Factors Pushing the Prediction Toward Higher Severity (Recurrent Fall)"
    lower = "Factors Pushing the Prediction Toward Lower Severity (Rare Fall)"
    row.template_explanation = text.replace(higher, "__TEMP__").replace(
        lower, higher).replace("__TEMP__", lower)
    fidelity, _, _ = _score(row)
    assert fidelity["direction_accuracy"] < 1


def test_value_and_scale_alterations_are_detected():
    row = _source()
    lines = row.template_explanation.splitlines()
    index = _first_data_row_index(lines)
    cells = [cell.strip() for cell in lines[index].strip("|").split("|")]
    cells[2] = "999"
    cells[3] = "invented scale"
    lines[index] = "| " + " | ".join(cells) + " |"
    row.template_explanation = "\n".join(lines)
    fidelity, _, unsupported = _score(row)
    assert fidelity["value_fidelity"] < 1
    assert fidelity["scale_fidelity"] < 1
    assert unsupported["unsupported_value_count"] == 1
    assert unsupported["unsupported_scale_count"] == 1


def test_extra_and_duplicate_factors_are_caught_by_unsupported_not_fidelity():
    row = _source()
    lines = row.template_explanation.splitlines()
    index = _first_data_row_index(lines)
    lines.insert(index + 1, lines[index])  # duplicate the first data row verbatim
    separator = next(i for i, line in enumerate(lines) if line == "| --- | --- | --- | --- |")
    lines.insert(separator + 1, "| 99 | Invented Factor | 1 | invented scale |")
    row.template_explanation = "\n".join(lines)
    fidelity, _, unsupported = _score(row)
    # A duplicate is deduped-by-first and an out-of-set factor is not eligible, so conditional
    # fidelity is unchanged; both belong to the unsupported-information family.
    assert fidelity["value_fidelity"] == 1
    assert fidelity["direction_accuracy"] == 1
    assert unsupported["duplicate_factor_count"] >= 1
    assert unsupported["unsupported_factor_count"] >= 1
    assert unsupported["unsupported_factor_rate"] > 0


def test_unsupported_drops_fidelity_mirrors_and_is_group_aware():
    row = _source()
    _, _, unsupported = _score(row)
    # Rate-mirrors of fidelity were removed (kept as raw counts instead).
    for gone in ("factor_precision", "unsupported_value_rate", "unsupported_scale_rate"):
        assert gone not in unsupported
    # Group-aware grounding: the domain template's grouped prose is not falsely flagged as
    # unsupported when a group label maps to a present clinical category.
    grouped = compute_patient_unsupported(
        row, "template_grouped", "template_grouped_method", "template_grouped_explanation")
    assert grouped["prose_unsupported_feature_count"] == 0


def test_completeness_drops_redundant_metrics_and_is_group_aware():
    row = _source()
    _, generic, _ = _score(row)
    # Redundant metrics removed (identical-to / subsumed-by another metric).
    assert "required_stage_decision_coverage" not in generic
    assert "required_section_coverage" not in generic
    assert generic["classification_field_coverage"] == 1  # the survivor still works

    # Group-aware interpretation coverage: the domain template (which groups) still fully covers its
    # required key factors via category labels, not only literal names.
    grouped = compute_patient_completeness(
        row, "template_grouped", "template_grouped_method", "template_grouped_explanation")
    assert grouped["interpretation_key_factor_coverage"] == 1.0


def test_empty_tables_make_fidelity_not_evaluable():
    row = _source()
    # Neutralize both factor-section headings so no rows are attributed to a direction.
    row.template_explanation = row.template_explanation.replace(
        "Factors Pushing the Prediction Toward", "Section removed:", 2)
    fidelity, completeness, _ = _score(row)
    assert np.isnan(fidelity["value_fidelity"])
    assert np.isnan(fidelity["scale_fidelity"])
    assert np.isnan(fidelity["direction_accuracy"])
    assert np.isnan(fidelity["rank_order_fidelity"])
    # Fidelity is 'not evaluable', but the empty output is penalized by completeness.
    assert completeness["factor_recall"] == 0


def test_empty_interpretation_and_causal_claim_are_detected():
    row = _source()
    start, end = row.template_explanation.split("Model Interpretation", 1)
    _, clinical = end.split("Clinical Note", 1)
    row.template_explanation = start + "Model Interpretation\n\nClinical Note" + clinical
    _, completeness, _ = _score(row)
    assert completeness["nonempty_section_coverage"] < 1

    row = _source()
    row.template_explanation = row.template_explanation.replace(
        "Model Interpretation\n", "Model Interpretation\nThis factor causes falls. ", 1)
    _, _, unsupported = _score(row)
    assert unsupported["causal_claim_sentence_count"] >= 1


def test_contradictory_outcome_and_rank_reversal_are_detected():
    row = _source()
    row.template_explanation = row.template_explanation.replace(
        "- Fall Classification: Fall",
        "- Fall Classification: Fall\n- Fall Classification: No Fall",
        1,
    )
    fidelity, _, _ = _score(row)
    assert not fidelity["outcome_correct"]

    row = _source()
    lines = row.template_explanation.splitlines()
    indices = [i for i, line in enumerate(lines) if line.startswith("| ")
               and line.split("|")[1].strip().isdigit()]
    lines[indices[0]], lines[indices[1]] = lines[indices[1]], lines[indices[0]]
    row.template_explanation = "\n".join(lines)
    fidelity, _, _ = _score(row)
    assert fidelity["rank_order_fidelity"] < 1


def test_reversed_ranking_scores_zero_concordance():
    row = _source()
    lines = row.template_explanation.splitlines()
    # Reverse the data rows inside *every* factor table so both directions are fully reversed;
    # every comparable pair is then out of order, giving concordance 0.0.
    separators = [i for i, line in enumerate(lines) if line == "| --- | --- | --- | --- |"]
    for separator in reversed(separators):  # back-to-front keeps earlier indices valid
        block_end = next((i for i in range(separator + 1, len(lines))
                          if not lines[i].startswith("|")), len(lines))
        lines[separator + 1:block_end] = list(reversed(lines[separator + 1:block_end]))
    row.template_explanation = "\n".join(lines)
    fidelity, _, _ = _score(row)
    assert fidelity["rank_order_fidelity"] == 0


def test_correct_label_in_prose_does_not_rescue_a_wrong_field():
    # Strict field parse must fail even when the correct outcome phrase appears elsewhere in prose
    # (a substring-presence check would wrongly pass this).
    row = _source()  # stage-2 faller: 'Fall Classification: Fall' is expected
    text = row.template_explanation.replace(
        "- Fall Classification: Fall", "- Fall Classification: No Fall", 1)
    text = text.replace(
        "Model Interpretation\n",
        "Model Interpretation\nThe model indicates a Fall classification here. ", 1)
    row.template_explanation = text
    fidelity, _, _ = _score(row)
    assert not fidelity["outcome_correct"]


def test_summary_reports_evaluable_denominator_excluding_nan():
    prose = {
        "prose_direction_fidelity": 1.0,
        "prose_grouping_fidelity": np.nan,
        "selected_factor_precision": np.nan,
        "minimum_group_size_compliance": np.nan,
        "prose_value_descriptor_fidelity": np.nan,
        "taxonomy_grouping_recall": np.nan,
        "taxonomy_grouping_precision": np.nan,
    }
    df = pd.DataFrame([
        {"patient_id": 1, "category": "mild", "stage": 2, "method": "template", "model": "m",
         "outcome_correct": True, "direction_accuracy": 1.0, "value_fidelity": 1.0,
         "scale_fidelity": 1.0, "rank_order_fidelity": 1.0, **prose},
        {"patient_id": 2, "category": "mild", "stage": 2, "method": "template", "model": "m",
         "outcome_correct": True, "direction_accuracy": np.nan, "value_fidelity": np.nan,
         "scale_fidelity": 1.0, "rank_order_fidelity": np.nan, **prose},
    ])
    overall = summarize(df).query("category == 'overall'").set_index("method")
    # NaN observations are excluded from the denominator, which is reported explicitly.
    assert overall.loc["template", "outcome_correct_n_evaluable"] == 2
    assert overall.loc["template", "scale_fidelity_n_evaluable"] == 2
    assert overall.loc["template", "value_fidelity_n_evaluable"] == 1
    assert overall.loc["template", "direction_accuracy_n_evaluable"] == 1
    assert overall.loc["template", "rank_order_fidelity_n_evaluable"] == 1
    # The mean is taken over evaluable observations only.
    assert overall.loc["template", "value_fidelity_mean"] == 1.0


def test_structure_checker_detects_malformed_and_forbidden_content():
    row = _source()
    clean = compute_patient_structure(
        row, "template", "template_method", "template_explanation")
    assert clean["overall_structure_compliant"]
    assert clean["n_components_passed"] == 8
    assert "structure_score" not in clean  # dropped: mean-of-booleans was not interpretable

    lines = row.template_explanation.splitlines()
    index = _first_data_row_index(lines)
    lines[index] = lines[index].replace("| 1 |", "| not-a-number |", 1)
    lines.append("The prediction likelihood is 75%.")
    row.template_explanation = "\n".join(lines)
    changed = compute_patient_structure(
        row, "template", "template_method", "template_explanation")
    assert changed["table_row_format"] < 1
    assert not changed["forbidden_content_clean"]
    assert not changed["overall_structure_compliant"]
    assert changed["n_components_passed"] < 8
