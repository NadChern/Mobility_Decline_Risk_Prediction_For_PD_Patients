from template_experiment.evaluation.validation_readiness import run
from template_experiment.run_support import read_jsonl, write_jsonl


def test_challenges_and_blinded_review_packet(tmp_path):
    results = run(tmp_path)
    assert len(results) == 16 and all(r["expectation_met"] for r in results)
    packet = read_jsonl(tmp_path / "blinded_reports.jsonl")
    assert len(packet) == 60
    assert all(set(r) == {"review_id", "prediction", "interpretation"} for r in packet)
    assert all(r["review_status"] == "pending" and not r["reviewer"]
               for r in read_jsonl(tmp_path / "human_annotations.jsonl"))


def test_human_annotations_are_not_overwritten_on_rerun(tmp_path):
    run(tmp_path)
    path = tmp_path / "human_annotations.jsonl"
    annotations = read_jsonl(path)
    annotations[0]["notes"] = "Human draft annotation: do not overwrite"
    write_jsonl(path, annotations)
    before = path.read_bytes()
    run(tmp_path)
    assert before == path.read_bytes()
