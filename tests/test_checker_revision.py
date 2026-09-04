"""Pilot-derived regressions and counterexamples; no network, no output regeneration."""

from pathlib import Path

import pytest

from template_experiment.evaluation.group_claims import extract_group_claims
from template_experiment.evaluation.interpretation_contract import check_interpretation
from template_experiment.evaluation.prose_checks import _detect_segment, _llm_group_relations
from template_experiment.run_support import read_jsonl
from template_experiment.taxonomy import FEATURE_CATEGORY


def factor(name, direction="supports_prediction"):
    return {"factor": name, "domain": FEATURE_CATEGORY[name], "direction": direction}


FACTORS = [factor("FOG (Freezing of Gait)"), factor("Gait"), factor("Age"),
           factor("PD Duration", "opposes_prediction")]


def report(group="gait and mobility (FOG, Gait)", age="age", opposing="PD duration"):
    return (f"Within this model, the result was associated with {group}, alongside {age}. "
            f"Factors pushing toward higher severity included {opposing}. "
            "These opposing factors did not outweigh the factors supporting the classification.")


@pytest.mark.parametrize("group", [
    "gait and mobility factors (FOG, Gait)",
    "gait and mobility domain (FOG [Freezing of Gait], Gait)",
    "gait and mobility (including FOG (Freezing of Gait), Gait)",
    "gait and mobility, which encompasses FOG and Gait",
    "gait and mobility factors (encompassing freezing of gait and normal gait)",
    "gait and mobility domain—specifically freezing of gait and slight gait impairment",
    "GAIT AND MOBILITY FACTORS (FOG, GAIT)",
    "gait  and\tmobility factors (FOG, Gait)",
])
def test_equivalent_membership_forms(group):
    assert check_interpretation(report(group), FACTORS, "mild")["issues"] == []


@pytest.mark.parametrize("name,phrase", [
    ("Age", "age"), ("PD Duration", "PD duration"),
    ("Dopaminergic Therapy", "no dopaminergic therapy started"),
    ("Daytime Sleepiness", "mild daytime sleepiness"),
    ("Mobility Summary", "mobility summary score"),
    ("FOG (Freezing of Gait)", "absence of freezing of gait"),
    ("FOG (Freezing of Gait)", "freezing of gait (FOG)"),
    ("Gait", "clinician-rated mild gait impairment"),
    ("Postural Instability at Dx", "postural instability present at diagnosis"),
    ("Rigidity at Dx", "rigidity present at diagnosis"),
    ("Rigidity at Dx", "absence of rigidity at diagnosis"),
    ("Urinary Problems", "normal urinary function with no urine control problems"),
    ("Urinary Problems", "absence of urinary problems"),
    ("UPDRS-IV Total", "MDS-UPDRS Part IV total score of 9"),
    ("UPDRS-IV Total", "UPDRS–IV Total"),
    ("Dopaminergic Therapy", "dopaminergic\t therapy"),
])
def test_reviewed_factor_variants(name, phrase):
    text = f"Within this model, the result was associated with {phrase}. This describes the model."
    result = check_interpretation(text, [factor(name)], "mild")
    assert result["issues"] == []
    assert any(m["factor"] == name and m["matched_text"] in text for m in result["factor_mentions"])


def test_parenthesis_stops_members_before_parallel_domains():
    text = ("Gait and mobility factors (including FOG, Mobility Summary, and Gait), "
            "mood symptoms (GDS-15), and disease-course characteristics (PD Duration).")
    relations = _llm_group_relations(text)
    assert relations == [("gait_mobility", {"FOG (Freezing of Gait)", "Mobility Summary", "Gait"})]


@pytest.mark.parametrize("noun", ["impairment", "impairments", "difficulty", "difficulties",
                                  "dysfunction", "function", "symptoms", "burden", "deficits",
                                  "problems", "involvement", "complaints"])
@pytest.mark.parametrize("membership", ["—supported by FOG, Gait, and Mobility Summary",
                                       " (FOG, Gait, Mobility Summary)"])
def test_clinical_noun_before_connector_or_parenthesis(noun, membership):
    text = f"Gait and mobility {noun}{membership}."
    assert _llm_group_relations(text) == [
        ("gait_mobility", {"FOG (Freezing of Gait)", "Gait", "Mobility Summary"})]


def test_p40538_map_free_group_is_exact_and_not_a_disagreement():
    import pandas as pd
    from template_experiment.evaluation.map_free_analysis import (
        PILOT_DATASET, METHOD, compute_results, disagreement_worksheet,
    )
    frame = pd.read_csv(PILOT_DATASET, keep_default_na=False)
    result = compute_results(frame[frame.patient_id == 40538], METHOD)
    assert result.iloc[0].exact_pair_set_agreement
    assert result.iloc[0].pairwise_grouping_f1 == 1.0
    assert disagreement_worksheet(result).empty


def test_clinical_noun_revision_changes_only_p40538_against_preserved_v2_audit():
    import json
    import numpy as np
    import pandas as pd
    from template_experiment.evaluation.map_free_analysis import PILOT_DATASET, METHOD, compute_results

    root = Path(__file__).resolve().parents[1]
    baseline = pd.read_csv(root / "template_experiment/outputs/synthesis_v2/history/checker_v2/part2a_map_free/patients.csv",
                           keep_default_na=False).set_index("patient_id")
    current = compute_results(pd.read_csv(PILOT_DATASET, keep_default_na=False), METHOD).set_index("patient_id")
    changed = [pid for pid in baseline.index
               if json.loads(baseline.loc[pid, "predicted_pairs"]) != json.loads(current.loc[pid, "predicted_pairs"])]
    assert changed == [40538]
    for pid in baseline.index.difference([40538]):
        old, new = baseline.loc[pid, "pairwise_grouping_f1"], current.loc[pid, "pairwise_grouping_f1"]
        assert (old == "" and pd.isna(new)) or np.isclose(float(old), float(new), equal_nan=True)


def test_member_description_is_not_a_list_boundary():
    text = ("Gait and mobility domain—specifically absence of freezing of gait, slight gait impairment, "
            "and the absence of postural instability at diagnosis—along with BMI.")
    assert _llm_group_relations(text) == [
        ("gait_mobility", {"FOG (Freezing of Gait)", "Gait", "Postural Instability at Dx"})]


@pytest.mark.parametrize("group", [
    "gait and mobility",  # Membership must never be filled in from the reference taxonomy.
    "gait and mobility (FOG)",
    "gait and mobility (FOG, MoCA)",
    "gait and mobility (FOG, Gait, MoCA)",
    "gait and mobility (including FOG, Gait",  # Broken boundary remains uncertain.
    "gait and mobility (FOG, GAITT)",  # No fuzzy guessing.
])
def test_wrong_or_unextractable_groups_still_flag(group):
    result = check_interpretation(report(group), FACTORS, "mild")
    assert any(i["code"].startswith("missing_or_inexact_group") for i in result["issues"])


def test_wrong_group_has_different_diagnostic_from_unknown_structure():
    wrong = check_interpretation(report("gait and mobility (FOG, MoCA)"), FACTORS, "mild")
    unknown = check_interpretation(report("gait and mobility"), FACTORS, "mild")
    assert any(i["kind"] == "content_mismatch" for i in wrong["issues"])
    assert any(i["kind"] == "extraction_uncertain" for i in unknown["issues"])


def test_gait_inside_freezing_of_gait_is_not_separate_factor():
    result = check_interpretation(report("gait and mobility (FOG (Freezing of Gait))"), FACTORS, "mild")
    assert any(i["code"] == "unrecognized_factor=Gait" for i in result["issues"])


def test_age_does_not_match_stage():
    result = check_interpretation(report(age="stage"), FACTORS, "mild")
    assert any(i["code"] == "unrecognized_factor=Age" for i in result["issues"])


@pytest.mark.parametrize("name,domain", [("MoCA", "cognition"), ("GDS-15", "depression"),
                                        ("UPDRS-IV Total", "motor complications")])
def test_broad_domain_word_does_not_substitute_for_named_instrument(name, domain):
    text = f"The model used {domain}. This describes the model."
    result = check_interpretation(text, [factor(name)], "mild")
    assert any(i["code"] == f"unrecognized_factor={name}" for i in result["issues"])


def test_current_postural_stability_does_not_satisfy_diagnosis_variable():
    text = "The model used normal postural stability. This describes the model."
    result = check_interpretation(text, [factor("Postural Instability at Dx")], "mild")
    assert any(i["code"] == "unrecognized_factor=Postural Instability at Dx" for i in result["issues"])
    assert _detect_segment("postural instability present at diagnosis")[0] == {"Postural Instability at Dx"}


def test_reversed_factor_direction_is_not_a_pass():
    text = ("Within this model, PD duration supported the classification. "
            "Factors pushing toward higher severity included gait and mobility (FOG, Gait) and age. "
            "These opposing factors did not outweigh the supporting factors.")
    result = check_interpretation(text, FACTORS, "mild")
    assert any(i["code"] == "wrong_evidence_side=Age" for i in result["issues"])


def test_passive_contrast_is_equivalent():
    text = ("Within this model, the result was associated with gait and mobility (FOG, Gait) and age. "
            "These influences were not outweighed by factors pushing toward higher severity, namely PD duration.")
    assert check_interpretation(text, FACTORS, "mild")["issues"] == []


@pytest.mark.parametrize("contrast", [
    "Supporting factors did not outweigh opposing factors.",
    "The opposing factors were not outweighed by supporting factors.",
    "The opposing factors outweighed the supporting factors.",
])
def test_reversed_or_absent_non_dominance_is_not_accepted(contrast):
    text = report().rsplit("These opposing", 1)[0] + contrast
    result = check_interpretation(text, FACTORS, "mild")
    assert any(i["code"] == "unrecognized_contrastive_outweigh_statement" for i in result["issues"])


def test_singleton_domain_gloss_does_not_invent_aggregation():
    text = ("Within this model, the result was associated with autonomic symptoms, represented by Fainting. "
            "This describes the model.")
    assert check_interpretation(text, [factor("Fainting")], "mild")["issues"] == []


def test_duplicate_conflicting_claim_is_not_overwritten_by_correct_claim():
    text = report("gait and mobility (FOG, MoCA) and gait and mobility (FOG, Gait)")
    result = check_interpretation(text, FACTORS, "mild")
    assert any(i["kind"] == "content_mismatch" for i in result["issues"])


def test_spans_still_reference_original_text_with_whitespace_and_dash_variants():
    text = "Within this model, gait  and\tmobility factors (FOG, Gait) mattered."
    claims = extract_group_claims(text, "mild")
    assert len(claims) == 1
    assert text[claims[0].span_start:claims[0].span_end] == claims[0].source_text


# These are development regressions from the pilot, not independent test-set validation.
PILOT = Path(__file__).resolve().parents[1] / "template_experiment/outputs/synthesis_v2/dataset_a/records.jsonl"
PILOT_ROWS = read_jsonl(PILOT)


@pytest.mark.parametrize("row", PILOT_ROWS, ids=lambda r: f"{r['patient_id']}-{r['method']}")
def test_saved_pilot_report_regression(row):
    category = "moderate" if "Recurrent" in row["prediction"] else "mild" if "Rare" in row["prediction"] else "no_falls"
    result = check_interpretation(row["interpretation"], row["selected_factors"], category)
    assert result["issues"] == []


def test_versioned_recheck_preserves_source_and_original_flags(tmp_path):
    from template_experiment.evaluation.recheck_dataset_a import run
    from template_experiment.run_support import write_jsonl

    source = tmp_path / "records.jsonl"
    rows = [r for r in PILOT_ROWS if r["patient_id"] == 3223]
    write_jsonl(source, rows)
    before = source.read_bytes()
    output = tmp_path / "audit"
    path = run(source, output)
    assert before == source.read_bytes() == (output / "provenance/source_records.jsonl").read_bytes()
    updated = read_jsonl(path)
    llm = next(r for r in updated if r["method"] == "taxonomy_conditioned_llm")
    original = next(r for r in rows if r["method"] == "taxonomy_conditioned_llm")
    assert llm["original_contract_errors"] == original["contract_errors"]
    assert llm["contract_errors"] == [] and llm["status"] == "complete"
    for key in ("raw_response", "interpretation", "prompt", "evidence_hash"):
        assert llm[key] == original[key]
    assert run(source, output) == path  # Same audit is reproducible.


def test_recheck_refuses_to_overwrite_audit_after_source_changes(tmp_path):
    from template_experiment.evaluation.recheck_dataset_a import run
    from template_experiment.run_support import write_jsonl

    source = tmp_path / "records.jsonl"
    rows = [dict(PILOT_ROWS[0])]
    write_jsonl(source, rows)
    output = tmp_path / "audit"
    run(source, output)
    snapshot = (output / "provenance/source_records.jsonl").read_bytes()
    rows[0]["interpretation"] += " Changed."
    write_jsonl(source, rows)
    with pytest.raises(ValueError, match="Audit data or checker code changed"):
        run(source, output)
    assert snapshot == (output / "provenance/source_records.jsonl").read_bytes()


def test_received_contract_failures_are_not_excluded_from_analysis_or_judging():
    from template_experiment.evaluation.part1_structural import build_paired_frame
    from template_experiment.experimental.llm_judge import prepare_blinded_items

    records = [dict(r) for r in PILOT_ROWS if r["patient_id"] == 3223]
    paired = build_paired_frame(records)
    assert paired.loc[paired.patient_id == 3223, "taxonomy_llm_explanation"].notna().all()
    items, key = prepare_blinded_items(records)
    assert len(items) > 0 and set(key.patient_id) == {3223}


def test_primary_analysis_pauses_for_uncertain_extraction_without_dropping_report(tmp_path):
    from template_experiment.evaluation.part1_structural import run
    from template_experiment.run_support import write_jsonl

    records = [dict(r) for r in PILOT_ROWS if r["patient_id"] == 3223]
    records[1]["checker_issues"] = [{"code": "test_unrecognized", "kind": "extraction_uncertain"}]
    path = tmp_path / "records.jsonl"
    write_jsonl(path, records)
    assert run(path, tmp_path / "analysis") is None
    assert "All received reports remain" in (tmp_path / "analysis/report.md").read_text()
