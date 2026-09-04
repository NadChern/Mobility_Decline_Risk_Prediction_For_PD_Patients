"""Unit tests for the automatic interpretation-prose fidelity panel."""

import numpy as np

from template_experiment.evaluation.interpretation_fidelity import (
    detect_prose_groups,
    prose_direction_fidelity,
    prose_grouping_evaluation,
    prose_grouping_fidelity,
    prose_value_descriptor_fidelity,
)


def _evidence(fog_value="1"):
    """Stage-2 evidence: UPDRS-III/MoCA support higher severity; FOG/Gait oppose."""
    return {
        "stage": 2,
        "factor_set": {"UPDRS-III Total", "MoCA", "FOG (Freezing of Gait)", "Gait"},
        "expected_order": {
            "s2_positive": ["UPDRS-III Total", "MoCA"],
            "s2_negative": ["FOG (Freezing of Gait)", "Gait"],
        },
        "content": {
            "UPDRS-III Total": {"value": "56"},
            "MoCA": {"value": "15"},
            "FOG (Freezing of Gait)": {
                "value": fog_value,
                "scale": (
                    "0 = None; 1 = Rare freezing (may have start hesitation); "
                    "2 = Occasional freezing when walking; "
                    "3 = Frequent freezing with occasional falls; "
                    "4 = Frequent falls from freezing"
                ),
            },
            "Gait": {"value": "1"},
        },
    }


# 'moderate' => supporting side is s2_positive (UPDRS-III Total, MoCA).
CATEGORY = "moderate"


def test_factor_names_are_not_misread_as_groups():
    # A plain factor listing must register no group phrases.
    assert detect_prose_groups("UPDRS-III Total, MoCA, Gait, and FOG (Freezing of Gait)") == set()


def test_direction_fidelity_rewards_correct_side_and_flags_wrong_side():
    good = ("Within this model, the Recurrent Fall classification was primarily associated with "
            "UPDRS-III Total and MoCA. While Gait supported the opposite direction, it did not "
            "outweigh these.")
    assert prose_direction_fidelity(good, _evidence(), CATEGORY) == 1.0

    bad = ("Within this model, the Recurrent Fall classification was primarily associated with "
           "Gait. While UPDRS-III Total supported the opposite direction, it did not outweigh it.")
    assert prose_direction_fidelity(bad, _evidence(), CATEGORY) < 1.0


def test_direction_fidelity_is_nan_without_grounded_mentions():
    assert np.isnan(prose_direction_fidelity("The model produced a prediction.", _evidence(),
                                             CATEGORY))


def test_direction_fidelity_recognizes_toward_a_fall_classification():
    evidence = {
        "stage": 1,
        "factor_set": {"GDS-15", "UPDRS-I Total", "Age", "PD Duration"},
        "expected_order": {
            "s1_negative": ["GDS-15", "UPDRS-I Total"],
            "s1_positive": ["Age", "PD Duration"],
        },
        "content": {},
    }
    prose = (
        "Within this model, the No Fall classification was primarily associated with GDS-15 and "
        "UPDRS-I Total. Factors pushing the prediction toward a Fall classification included Age "
        "and PD Duration, though these did not outweigh the supporting factors."
    )
    assert prose_direction_fidelity(prose, evidence, "no_falls") == 1.0


def test_direction_fidelity_splits_mixed_contrast_clauses():
    prose = (
        "UPDRS-III Total and MoCA supported the prediction, while Gait and "
        "FOG (Freezing of Gait) supported the opposite direction."
    )
    assert prose_direction_fidelity(prose, _evidence(), CATEGORY) == 1.0


def test_direction_fidelity_handles_leading_contrast_clause():
    prose = (
        "Although Gait and FOG (Freezing of Gait) supported the opposite direction, "
        "UPDRS-III Total and MoCA primarily supported the prediction."
    )
    assert prose_direction_fidelity(prose, _evidence(), CATEGORY) == 1.0


def test_direction_fidelity_keeps_commas_inside_leading_opposing_factor_list():
    prose = (
        "Although factors pushing toward recurrent fall included gait impairment, supported by "
        "Gait and FOG (Freezing of Gait), these did not outweigh the supporting factors."
    )
    evidence = _evidence()
    evidence["supporting_order"] = ["UPDRS-III Total", "MoCA"]
    evidence["opposing_order"] = ["Gait", "FOG (Freezing of Gait)"]
    assert prose_direction_fidelity(prose, evidence, CATEGORY) == 1.0


def test_direction_fidelity_handles_passive_not_outweighed_by_opposing_factors():
    prose = (
        "The supporting influences were not outweighed by factors pushing toward lower severity, "
        "including Gait and FOG (Freezing of Gait)."
    )
    assert prose_direction_fidelity(prose, _evidence(), CATEGORY) == 1.0


def test_direction_fidelity_recognizes_counterbalanced_by_as_opposing():
    prose = "The prediction was counterbalanced by Gait and FOG (Freezing of Gait)."
    assert prose_direction_fidelity(prose, _evidence(), CATEGORY) == 1.0


def test_direction_fidelity_counts_repeated_same_side_factor_once():
    prose = (
        "UPDRS-III Total supported the prediction. The UPDRS-III Total was influential. "
        "While Gait supported the opposite direction, it did not outweigh it."
    )
    assert prose_direction_fidelity(prose, _evidence(), CATEGORY) == 1.0


def test_direction_fidelity_marks_factor_on_both_sides_incorrect_once():
    prose = (
        "UPDRS-III Total supported the prediction. The UPDRS-III Total remained influential. "
        "While UPDRS-III Total and Gait supported the opposite direction, they did not outweigh it."
    )
    # UPDRS-III appears on both sides and is incorrect; Gait appears only on its correct side.
    assert prose_direction_fidelity(prose, _evidence(), CATEGORY) == 0.5


def test_grouped_template_direction_uses_explicit_members():
    prose = (
        "Other factors supported the prediction. While gait and mobility (Gait, FOG (Freezing of "
        "Gait)) supported the opposite "
        "direction, they did not outweigh them."
    )
    assert prose_direction_fidelity(
        prose, _evidence(), CATEGORY, expand_template_groups=True) == 1.0


def test_grouped_template_direction_flags_category_on_wrong_side():
    prose = "The prediction was primarily associated with gait and mobility (Gait, FOG (Freezing of Gait))."
    assert prose_direction_fidelity(
        prose, _evidence(), CATEGORY, expand_template_groups=True) == 0.0


def test_llm_direction_does_not_infer_unnamed_factors_from_category():
    prose = "The prediction was primarily associated with gait and mobility."
    assert np.isnan(prose_direction_fidelity(prose, _evidence(), CATEGORY))


def test_grouping_fidelity_passes_complete_llm_contract():
    good = (
        "The Recurrent Fall classification was associated with UPDRS-III Total and MoCA. "
        "While gait and mobility, represented by Gait and FOG (Freezing of Gait), supported the "
        "opposite direction, they did not outweigh them."
    )
    assert prose_grouping_fidelity(good, _evidence(), CATEGORY) == 1.0


def test_grouping_fidelity_flags_group_attributed_to_wrong_side():
    bad = (
        "The Recurrent Fall classification was primarily associated with gait and mobility, "
        "represented by Gait and FOG (Freezing of Gait), plus UPDRS-III Total and MoCA."
    )
    # Group construction is correct; direction is deliberately owned by direction fidelity.
    assert prose_grouping_fidelity(bad, _evidence(), CATEGORY) == 1.0
    assert prose_direction_fidelity(bad, _evidence(), CATEGORY) < 1.0


def test_grouping_fidelity_fails_plain_listing_when_grouping_was_expected():
    listing = (
        "The Recurrent Fall classification was associated with UPDRS-III Total and MoCA. "
        "While Gait and FOG (Freezing of Gait) supported the opposite direction, they did not "
        "outweigh them."
    )
    result = prose_grouping_evaluation(listing, _evidence(), CATEGORY)
    assert result["taxonomy_grouping_recall"] == 0.0
    assert np.isnan(result["taxonomy_grouping_precision"])
    assert np.isnan(result["prose_grouping_fidelity"])
    assert "opposing:gait_mobility" in result["missing_expected_groups"]


def test_grouping_fidelity_is_not_applicable_to_generic_template():
    listing = (
        "The Recurrent Fall classification was associated with UPDRS-III Total and MoCA. "
        "While Gait and FOG (Freezing of Gait) supported the opposite direction."
    )
    result = prose_grouping_evaluation(listing, _evidence(), CATEGORY, method="template")
    for metric in (
        "prose_grouping_fidelity",
        "taxonomy_grouping_recall",
        "taxonomy_grouping_precision",
        "selected_factor_precision",
        "minimum_group_size_compliance",
    ):
        assert np.isnan(result[metric])


def test_grouped_template_requires_and_passes_with_explicit_group_members():
    prose = (
        "The Recurrent Fall classification was associated with UPDRS-III Total and MoCA. "
        "While gait and mobility (Gait, FOG (Freezing of Gait)) supported the opposite direction, "
        "it did not outweigh them."
    )
    assert prose_grouping_fidelity(
        prose, _evidence(), CATEGORY, method="template_grouped") == 1.0


def test_grouping_fidelity_rejects_one_member_category():
    prose = (
        "Motor impairment was represented by UPDRS-III Total. MoCA also supported the prediction. "
        "While gait and mobility, represented by Gait and FOG (Freezing of Gait), supported the "
        "opposite direction."
    )
    result = prose_grouping_evaluation(prose, _evidence(), CATEGORY)
    assert result["minimum_group_size_compliance"] == 0.5
    assert result["prose_grouping_fidelity"] == 0.75
    assert "supporting:motor_impairment" in result["undersized_groups"]


def test_grouping_fidelity_rejects_wrong_category_member():
    prose = (
        "Motor impairment, represented by UPDRS-III Total and MoCA, supported the prediction. "
        "While gait and mobility, represented by Gait and FOG (Freezing of Gait), supported the "
        "opposite direction."
    )
    result = prose_grouping_evaluation(prose, _evidence(), CATEGORY)
    assert result["taxonomy_grouping_precision"] == 0.75
    # The members are selected and both groups meet the size rule; taxonomy agreement is separate.
    assert result["prose_grouping_fidelity"] == 1.0
    assert "MoCA" in result["invalid_category_members"]


def test_factor_value_gloss_does_not_truncate_outer_group_members():
    evidence = {
        "stage": 1,
        "factor_set": {
            "UPDRS-IV Total", "FOG (Freezing of Gait)", "Postural Stability",
            "GDS-15", "Urinary Problems", "UPDRS-III Total", "Age",
            "Dopaminergic Therapy",
        },
        "supporting_order": [
            "UPDRS-IV Total", "FOG (Freezing of Gait)", "Postural Stability",
            "GDS-15", "Urinary Problems",
        ],
        "opposing_order": ["UPDRS-III Total", "Dopaminergic Therapy", "Age"],
        "expected_order": {},
        "content": {},
    }
    prose = (
        "The No Fall classification was associated with motor characteristics, specifically the "
        "absence of motor complications (UPDRS-IV Total), absence of freezing of gait (FOG), and "
        "normal postural stability (Postural Stability), alongside GDS-15 and Urinary Problems."
    )
    result = prose_grouping_evaluation(prose, evidence, "no_falls")
    assert "UPDRS-IV Total" in result["observed_groups"]
    assert "FOG (Freezing of Gait)" in result["observed_groups"]
    assert "Postural Stability" in result["observed_groups"]
    assert result["selected_factor_precision"] == 1.0


def test_parallel_individual_factor_does_not_join_preceding_group():
    prose = (
        "Gait characteristics were supported by FOG and Gait. The prediction was also supported "
        "by motor examination findings, reflected by UPDRS-III Total, and a daily activity "
        "Mobility Summary score."
    )
    result = prose_grouping_evaluation(prose, _evidence(), CATEGORY)
    assert "motor_impairment[UPDRS-III Total]" in result["observed_groups"]
    assert "motor_impairment[Mobility Summary,UPDRS-III Total]" not in result["observed_groups"]
    assert result["minimum_group_size_compliance"] == 0.5


def test_mobility_and_at_diagnosis_category_phrases_are_detected():
    evidence = {
        "stage": 2,
        "factor_set": {"Gait", "Mobility Summary", "Postural Instability at Dx", "Rigidity at Dx"},
        "supporting_order": ["Gait", "Mobility Summary"],
        "opposing_order": ["Postural Instability at Dx", "Rigidity at Dx"],
        "expected_order": {},
        "content": {},
    }
    prose = (
        "Mobility (supported by Gait and Mobility Summary) supported the prediction. Conversely, "
        "motor features at diagnosis (Postural Instability at Dx and Rigidity at Dx) opposed it."
    )
    result = prose_grouping_evaluation(prose, evidence, CATEGORY)
    assert "supporting:gait_mobility[Gait,Mobility Summary]" in result["observed_groups"]
    assert "opposing:motor_impairment[Postural Instability at Dx,Rigidity at Dx]" in result[
        "observed_groups"
    ]


def test_grouping_fidelity_rejects_nonselected_factor():
    evidence = _evidence()
    evidence["factor_set"].add("Age")
    evidence["supporting_order"] = ["UPDRS-III Total", "MoCA"]
    evidence["opposing_order"] = ["Gait", "FOG (Freezing of Gait)"]
    prose = (
        "UPDRS-III Total, MoCA, and Age supported the prediction. While gait and mobility, "
        "represented by Gait and FOG (Freezing of Gait), supported the opposite direction."
    )
    result = prose_grouping_evaluation(prose, evidence, CATEGORY)
    # Age is not grouped, so grouping fidelity remains intact; unsupported prose owns this error.
    assert result["prose_grouping_fidelity"] == 1.0
    assert result["nonselected_interpretation_factors"] == "Age"


def test_selected_factor_precision_rejects_nonselected_group_member():
    evidence = _evidence()
    evidence["factor_set"].add("Age")
    evidence["supporting_order"] = ["UPDRS-III Total", "MoCA"]
    evidence["opposing_order"] = ["Gait", "FOG (Freezing of Gait)"]
    prose = (
        "Gait and mobility were represented by Gait, FOG (Freezing of Gait), and Age."
    )
    result = prose_grouping_evaluation(prose, evidence, CATEGORY)
    assert result["selected_factor_precision"] == 2 / 3


def test_grouping_fidelity_rejects_missing_singleton():
    prose = (
        "UPDRS-III Total supported the prediction. While gait and mobility, represented by Gait "
        "and FOG (Freezing of Gait), supported the opposite direction."
    )
    result = prose_grouping_evaluation(prose, _evidence(), CATEGORY)
    # Required-factor representation belongs to completeness, not grouping fidelity.
    assert result["prose_grouping_fidelity"] == 1.0
    assert result["missing_singletons"] == "MoCA"


def test_grouping_fidelity_passes_individual_listing_when_no_group_is_possible():
    evidence = {
        "stage": 2,
        "factor_set": {"UPDRS-III Total", "MoCA", "Age", "BMI"},
        "supporting_order": ["UPDRS-III Total", "MoCA"],
        "opposing_order": ["Age", "BMI"],
        "expected_order": {},
        "content": {},
    }
    prose = (
        "UPDRS-III Total and MoCA supported the prediction. While Age and BMI supported the "
        "opposite direction."
    )
    assert np.isnan(prose_grouping_fidelity(prose, evidence, CATEGORY))


def test_value_descriptor_flags_contradiction_and_rewards_consistency():
    # FOG = 1 of 4 is low; calling it 'severe' contradicts the value.
    contradiction = "There was severe freezing of gait."
    assert prose_value_descriptor_fidelity(contradiction, _evidence(fog_value="1")) < 1.0

    # FOG = 0 described as 'absence of' is consistent.
    consistent = "There was an absence of freezing of gait."
    assert prose_value_descriptor_fidelity(consistent, _evidence(fog_value="0")) == 1.0


def test_value_descriptor_is_nan_without_descriptors():
    assert np.isnan(prose_value_descriptor_fidelity("UPDRS-III Total and MoCA were influential.",
                                                    _evidence()))


def test_value_descriptor_does_not_invent_continuous_score_thresholds():
    evidence = {
        "content": {
            "GDS-15": {
                "value": "1",
                "scale": "0-15; higher = more depressive burden",
            }
        }
    }
    assert np.isnan(prose_value_descriptor_fidelity("low GDS-15", evidence))
