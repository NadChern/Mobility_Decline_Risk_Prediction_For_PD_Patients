import json

import pandas as pd
import pytest

from template_experiment.evaluation.map_free_analysis import (
    _merge_prior_adjudications, adjudication_status, render_report,
)


def worksheet():
    return pd.DataFrame([{
        "patient_id": i, "category": "mild", "stage": 2,
        "gold_groups": "[]", "map_free_groups": "[]", "interpretation": f"report {i}",
        "suggested_category": "over_grouping", "author_adjudication": "",
        "author_rationale": "", "review_status": "pending_blinded_author_review",
    } for i in (1, 2)])


def test_report_leads_with_authors_not_automatic_suggestions():
    frame = worksheet()
    frame["author_adjudication"] = ["granularity_difference", "plausible_alternative_abstraction"]
    frame["author_rationale"] = ["Reviewed and explained", ""]
    frame["review_status"] = "author_reviewed"
    result = adjudication_status(frame)
    assert result["author_counts"] == {"granularity_difference": 1, "plausible_alternative_abstraction": 1}
    assert result["automatic_counts"] == {"over_grouping": 2}
    assert result["complete_reviews"] == 1
    report = render_report(frame)
    assert report.index("## Author adjudication") < report.index("## Automatic routing")
    assert "provisional" in report and "Missing rationales: 2" in report
    assert "| granularity_difference | 1 | 1 |" in report


def test_complete_author_review_is_labeled_complete():
    frame = worksheet()
    frame["author_adjudication"] = "granularity_difference"
    frame["author_rationale"] = "Explicit reason"
    frame["review_status"] = "author_reviewed"
    assert "provisional" not in render_report(frame)
    assert adjudication_status(frame)["complete_reviews"] == 2


def test_merge_preserves_labels_and_rationale_only_drafts(tmp_path):
    prior = worksheet()
    prior.loc[0, ["author_adjudication", "author_rationale", "review_status"]] = [
        "granularity_difference", "My decision", "author_reviewed"]
    prior.loc[1, "author_rationale"] = "Unfinished draft"
    path = tmp_path / "prior.csv"
    prior.to_csv(path, index=False)
    result = _merge_prior_adjudications(worksheet(), path)
    assert result.loc[0, "author_adjudication"] == "granularity_difference"
    assert result.loc[1, "author_rationale"] == "Unfinished draft"
    assert result.loc[0, "review_status"] == "author_reviewed"
    # A second regeneration does not incorrectly invalidate a migrated review.
    result.to_csv(path, index=False)
    again = _merge_prior_adjudications(worksheet(), path)
    assert again.loc[0, "review_status"] == "author_reviewed"


@pytest.mark.parametrize("column", ["gold_groups", "map_free_groups", "interpretation", "stage"])
def test_changed_review_input_retains_text_but_requires_re_review(tmp_path, column):
    prior = worksheet()
    prior.loc[0, ["author_adjudication", "author_rationale", "review_status"]] = [
        "granularity_difference", "Original human note", "author_reviewed"]
    path = tmp_path / "prior.csv"
    prior.to_csv(path, index=False)
    changed = worksheet()
    changed.loc[0, column] = 1 if column == "stage" else "changed"
    result = _merge_prior_adjudications(changed, path)
    assert result.loc[0, "author_rationale"] == "Original human note"
    assert result.loc[0, "review_status"] == "needs_re_review_inputs_changed"
    assert adjudication_status(result)["author_counts"] == {}


def test_duplicate_ids_are_not_silently_merged(tmp_path):
    prior = worksheet()
    prior["patient_id"] = 1
    path = tmp_path / "prior.csv"
    prior.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate patient"):
        _merge_prior_adjudications(worksheet(), path)


def test_empty_worksheet_and_invalid_labels_are_reported_without_fabrication():
    assert adjudication_status(worksheet().iloc[:0])["labels_entered"] == 0
    assert "Non-exact reports: 0" in render_report(worksheet().iloc[:0])
    frame = worksheet()
    frame.loc[0, "author_adjudication"] = "not_a_category"
    assert "Invalid author labels: 1" in render_report(frame)


def test_run_imports_authoring_source_and_snapshots_without_changing_decisions(tmp_path):
    from template_experiment.evaluation.map_free_analysis import run, AUTHOR_WORKSHEET

    source = tmp_path / "authoring.csv"
    source.write_bytes(AUTHOR_WORKSHEET.read_bytes())
    before = source.read_bytes()
    output = tmp_path / "active"
    _, _, result = run(output_dir=output, adjudications_path=source)
    original = pd.read_csv(source, keep_default_na=False)
    assert source.read_bytes() == before
    for column in ["author_adjudication", "author_rationale", "review_status"]:
        assert result.set_index("patient_id")[column].to_dict() == original.set_index("patient_id")[column].to_dict()
    assert len(list((output / "history/worksheet_snapshots").glob("*.csv"))) == 1
    first = (output / "disagreement_worksheet.csv").read_bytes()
    run(output_dir=output, adjudications_path=source)
    assert first == (output / "disagreement_worksheet.csv").read_bytes()
    manifest = json.loads((output / "report_manifest.json").read_text())
    assert manifest["parser_version"] == "explicit-group-claims-v2.1-pilot"
    assert manifest["adjudication_status"]["labels_entered"] == 20
    from template_experiment.evaluation.map_free_analysis import ADJUDICATION_RUBRIC
    from template_experiment.synthesis_protocol import sha256_file
    rubric = manifest["adjudication_rubric"]
    assert rubric["version"] == "adjudication-rubric-v1.1.1"
    assert rubric["sha256"] == sha256_file(ADJUDICATION_RUBRIC)
    assert (output / rubric["snapshot"]).read_bytes() == ADJUDICATION_RUBRIC.read_bytes()
    assert rubric["version"] in (output / "report.md").read_text()


def test_missing_rubric_version_fails_before_output_writes(tmp_path, monkeypatch):
    import template_experiment.evaluation.map_free_analysis as module
    rubric = tmp_path / "rubric.md"
    rubric.write_text("Unversioned rubric")
    monkeypatch.setattr(module, "ADJUDICATION_RUBRIC", rubric)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="declare its version"):
        module.run(output_dir=output)
    assert not (output / "patients.csv").exists()
