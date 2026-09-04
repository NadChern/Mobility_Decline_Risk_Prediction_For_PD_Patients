"""Tests for pairwise synthesis metrics and complexity/scalability."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from template_experiment.evaluation.synthesis_comparison import (
    compute_patient_synthesis,
    pairwise_grouping_scores,
)
from template_experiment.evaluation.complexity_scalability import (
    static_complexity,
    flexibility_unseen_feature,
)

PAIRED = (Path(__file__).resolve().parents[1]
          / "template_experiment/outputs/pilot/pilot_reports.csv")


def _row():
    return pd.read_csv(PAIRED, keep_default_na=False).iloc[0].copy()


def test_generic_template_does_not_aggregate():
    m = compute_patient_synthesis(_row(), "template", "template_method", "template_explanation")
    assert m["predicted_pair_count"] == 0
    assert m["predicted_group_count"] == 0
    if m["gold_pair_count"]:
        assert m["pairwise_grouping_recall"] == 0.0
        assert m["pairwise_grouping_f1"] == 0.0


def test_domain_template_matches_its_reference_pair_set():
    row = _row()
    d = compute_patient_synthesis(row, "template_grouped", "template_grouped_method",
                                  "template_grouped_explanation")
    assert d["exact_pair_set_agreement"] is True
    if d["gold_pair_count"]:
        assert d["pairwise_grouping_precision"] == 1.0
        assert d["pairwise_grouping_recall"] == 1.0
        assert d["pairwise_grouping_f1"] == 1.0
        assert d["explicit_membership_recall"] == 1.0
        assert "span_start" in d["group_claims_with_spans"]


def test_llm_named_category_members_still_count_as_grouped():
    row = _row()
    result = compute_patient_synthesis(
        row, "gemini", "gemini_model", "gemini_explanation"
    )
    assert result["predicted_group_count"] >= 1
    assert result["predicted_pair_count"] >= 1
    assert 0 < result["explicit_membership_recall"] <= 1.0


def test_pairwise_empty_set_conventions():
    precision, recall, f1 = pairwise_grouping_scores({("a", "b")}, set())
    assert np.isnan(precision)
    assert recall == 0.0
    assert f1 == 0.0

    precision, recall, f1 = pairwise_grouping_scores(set(), set())
    assert np.isnan(precision)
    assert np.isnan(recall)
    assert np.isnan(f1)

    precision, recall, f1 = pairwise_grouping_scores(set(), {("a", "b")})
    assert precision == 0.0
    assert np.isnan(recall)
    assert f1 == 0.0


def test_static_complexity_counts():
    df = static_complexity().set_index("method")
    assert df.loc["Domain template", "map_entries"] == 21
    assert df.loc["Domain template", "grouping_rules"] == 1
    assert df.loc["Generic template", "grouping_rules"] == 0
    assert df.loc["LLM (Gemini)", "prompts"] == 1
    assert df.loc["LLM (Gemini)", "prompt_words"] > 0  # prompt counted as manual logic


def test_flexibility_table_marks_llm_claim_not_fact():
    df = flexibility_unseen_feature().set_index("method")
    assert bool(df.loc["Generic template", "renders_unseen_no_edits"]) is True
    assert bool(df.loc["Domain template", "groups_unseen_no_edits"]) is False
    assert "VERIFY" in str(df.loc["LLM (Gemini)", "edits_to_group_new_feature"])


def test_flexibility_stress_test_quantifies_unseen_feature_gap(tmp_path):
    from template_experiment.evaluation.flexibility_stress_test import compute_results
    df = compute_results(output_dir=tmp_path)
    unseen = df[df["case_id"] == "C01_existing_gait_domain"].set_index("method")
    assert bool(unseen.loc["generic_template", "renders_without_task_specific_edits"]) is True
    assert unseen.loc["domain_template", "compositional_fidelity"] == 1.0
    assert unseen.loc["map_free_gemini", "status"] == "pending_api_generation"


def test_paired_judge_parser_and_aggregation():
    from template_experiment.evaluation.geval_synthesis_judge import (
        parse_judgment, aggregate, RUBRIC,
    )
    payload = {"scores_a": {name: 4 for name in RUBRIC},
               "scores_b": {name: 3 for name in RUBRIC},
               "preference": "A", "rationale": "A is more traceable."}
    parsed = parse_judgment(json.dumps(payload))
    assert parsed == payload
    key = pd.DataFrame([{"audit_id": "x", "patient_id": 1,
                         "method_a": "taxonomy_conditioned_llm",
                         "method_b": "domain_template"}])
    scores, patient, summary = aggregate(
        [{"audit_id": "x", "judge_model": "judge", "parsed": parsed}], key)
    assert len(scores) == 2 and len(patient) == 2
    assert summary["llm_wins"] == 1


def test_common_parser_validation_corpus_passes():
    from template_experiment.evaluation.parser_validation import run
    results = run()
    assert results["exact_relation_match"].mean() >= 0.95
    assert results["source_spans_valid"].all()


def test_domain_template_contract_requires_explicit_membership():
    from explanation.explanation_builder import build_patient_explanation_data_full, get_patient_index
    from template_experiment.domain_template import render_template_grouped_explanation
    from template_experiment.evaluation.shared import parse_report
    from template_experiment.evaluation.group_claims import extract_group_claims
    from template_experiment.synthesis_protocol import validate_interpretation_contract

    patient_info = build_patient_explanation_data_full(get_patient_index(int(_row().patient_id)))
    interpretation = parse_report(render_template_grouped_explanation(patient_info))["interpretation"]
    assert validate_interpretation_contract(interpretation, patient_info) == []
    claims = extract_group_claims(interpretation, str(_row().category))
    assert claims
    hidden = interpretation
    for member in claims[0].members:
        hidden = hidden.replace(member, "unnamed member")
    assert any("missing_or_inexact_group" in error
               for error in validate_interpretation_contract(hidden, patient_info))


def test_prompt_uses_same_mapped_selected_factor_packet():
    from explanation.explanation_builder import build_patient_explanation_data_full, get_patient_index
    from explanation.llm import _build_prompt
    from template_experiment.synthesis_protocol import (
        build_taxonomy_conditioned_prompt, evidence_packet, remove_taxonomy_conditioning,
    )

    patient_info = build_patient_explanation_data_full(get_patient_index(int(_row().patient_id)))
    prompt = build_taxonomy_conditioned_prompt(patient_info)
    assert remove_taxonomy_conditioning(prompt, patient_info) == _build_prompt(patient_info)
    assert "no more than 170 words" not in prompt
    for factor in evidence_packet(patient_info)["selected_factors"]:
        assert factor["factor"] in prompt
        assert factor["direction"] in prompt
        assert factor["domain"] is None or factor["domain"] in prompt
