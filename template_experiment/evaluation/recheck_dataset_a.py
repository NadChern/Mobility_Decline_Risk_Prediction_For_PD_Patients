"""Recheck saved Dataset A reports locally, preserving original records and findings.

The separate derived records are consumed by analysis; the generation checkpoint is never
rewritten here. Each audit directory is tied to exact source data and checker source hashes.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from .group_claims import PARSER_VERSION
from .interpretation_contract import CHECKER_VERSION, check_interpretation
from ..run_support import atomic_text, indexed, output_lock, progress, read_jsonl, write_jsonl
from ..synthesis_protocol import PROJECT_ROOT, SENTENCE_RANGE, sha256_file


SOURCE = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/dataset_a/records.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/checker_v2_1"
DATA_DIR = OUTPUT_DIR / "data"
PROVENANCE_DIR = OUTPUT_DIR / "provenance"
RECHECKED_RECORDS = DATA_DIR / "records_rechecked.jsonl"


def category_for_prediction(prediction):
    categories = {"Rare Fall classification": "mild", "Recurrent Fall classification": "moderate",
                  "No Fall classification": "no_falls"}
    try:
        return categories[prediction]
    except KeyError as exc:
        raise ValueError(f"Unsupported saved prediction: {prediction!r}") from exc


def source_hashes():
    paths = ["evaluation/interpretation_contract.py", "evaluation/group_claims.py",
             "evaluation/prose_checks.py", "evaluation/shared.py", "taxonomy.py",
             "synthesis_protocol.py", "evaluation/recheck_dataset_a.py"]
    root = PROJECT_ROOT / "template_experiment"
    return {path: sha256_file(root / path) for path in paths}


def run(records_path=SOURCE, output_dir=OUTPUT_DIR):
    records_path, output_dir = Path(records_path), Path(output_dir)
    if records_path.resolve() in {
        (output_dir / "provenance/source_records.jsonl").resolve(),
        (output_dir / "data/records_rechecked.jsonl").resolve(),
    }:
        raise ValueError("Recheck requires the original generation checkpoint, not a derived audit file.")
    with output_lock(output_dir):
        return _run(records_path, output_dir)


def _run(records_path, output_dir):
    data_dir = output_dir / "data"
    provenance_dir = output_dir / "provenance"
    original_hash = sha256_file(records_path)
    records = read_jsonl(records_path)
    indexed(records, ("patient_id", "method"))  # Refuse duplicate identities.
    manifest = {"checker_version": CHECKER_VERSION, "parser_version": PARSER_VERSION,
                "source_records_sha256": original_hash, "code_sha256": source_hashes(),
                "validation_scope": "Pilot development/recheck, not independent final-test validation"}
    manifest_path = provenance_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Audit data or checker code changed. Choose a new --output-dir/version; "
                         "the previous audit will not be overwritten.")
    snapshot = provenance_dir / "source_records.jsonl"
    if snapshot.exists() and sha256_file(snapshot) != original_hash:
        raise ValueError("Source snapshot differs; use a new audit output directory.")
    if not snapshot.exists():
        atomic_text(snapshot, records_path.read_text(encoding="utf-8"))
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    audits, derived = [], []
    for record in records:
        updated = dict(record)
        if record["status"] not in {"complete", "contract_failed"}:
            derived.append(updated)
            continue
        result = check_interpretation(record["interpretation"], record["selected_factors"],
                                      category_for_prediction(record["prediction"]), SENTENCE_RANGE)
        errors = [i["code"] for i in result["issues"]]
        audits.append({"patient_id": record["patient_id"], "method": record["method"],
                       "original_status": record["status"],
                       "original_contract_errors": record.get("contract_errors", []),
                       "human_review": "pending", **result})
        updated.update(original_generation_status=record["status"],
                       original_contract_errors=record.get("contract_errors", []),
                       contract_errors=errors, checker_version=CHECKER_VERSION,
                       checker_issues=result["issues"],
                       checker_assessment=result["assessment"],
                       status="contract_failed" if errors else "complete")
        derived.append(updated)
    write_jsonl(data_dir / "records_rechecked.jsonl", derived)
    write_jsonl(data_dir / "findings.jsonl", audits)
    llm = [r for r in audits if r["method"] == "taxonomy_conditioned_llm"]
    old_flagged = sum(bool(r["original_contract_errors"]) for r in llm)
    old_flags = sum(len(r["original_contract_errors"]) for r in llm)
    counts = Counter(r["assessment"] for r in llm)
    summary = pd.DataFrame([{
        "patient_id": r["patient_id"], "method": r["method"],
        "original_status": r["original_status"], "assessment": r["assessment"],
        "old_flag_count": len(r["original_contract_errors"]), "new_flag_count": len(r["issues"]),
        "old_flags": json.dumps(r["original_contract_errors"]),
        "new_findings": json.dumps(r["issues"]), "human_review": r["human_review"],
    } for r in audits])
    atomic_text(output_dir / "summary.csv", summary.to_csv(index=False))
    lines = ["# Dataset A checker-v2.1 pilot recheck", "",
             f"Checker: `{CHECKER_VERSION}`; parser: `{PARSER_VERSION}`.", "",
             f"Reviewed {len(llm)} saved Gemini interpretations. Before: {old_flagged} flagged reports "
             f"with {old_flags} flags. After: {counts['pass']} pass, {counts['needs_review']} need review.", "",
             "No API calls. Original prompts, responses, interpretation text, and generation flags "
             "are unchanged in the source checkpoint. The original records are also copied to "
             "`provenance/source_records.jsonl`; `data/records_rechecked.jsonl` contains derived "
             "checker statuses and `data/findings.jsonl` contains detailed matches.", "",
             "A pass means no issue detected by this checker, not clinical correctness or a perfect "
             "synthesis score. Missing recognition is distinct from a recognized content mismatch. "
             "These are pilot-development results; independent human review and validation on a "
             "separate corpus remain required before final-test freezing.", "",
             "| Patient | Original flags | New findings | Assessment |",
             "| --- | ---: | ---: | --- |"]
    for row in llm:
        lines.append(f"| {row['patient_id']} | {len(row['original_contract_errors'])} | "
                     f"{len(row['issues'])} | {row['assessment']} |")
    for row in llm:
        if not row["original_contract_errors"] and not row["issues"]:
            continue
        lines.extend(["", f"## Patient {row['patient_id']}", "", "Original flags:", ""])
        lines.extend(f"- `{error}`" for error in row["original_contract_errors"])
        lines.extend(["", "Rechecked explicit groups:", ""])
        lines.extend(f"- {claim['side']} / {claim['group']}: {', '.join(claim['members'])}"
                     for claim in row["group_claims"])
        if not row["group_claims"]:
            lines.append("No explicit multi-factor relation extracted; see the factor mentions below.")
        lines.extend(["", "Recognized factor mentions (source text → factor):", ""])
        seen = set()
        for mention in row["factor_mentions"]:
            key = (mention["side"], mention["matched_text"], mention["factor"])
            if key not in seen:
                seen.add(key)
                lines.append(f"- {mention['side']}: “{mention['matched_text']}” → {mention['factor']}")
        lines.extend(["", f"New findings: {json.dumps(row['issues'], ensure_ascii=False)}", "",
                      "Human confirmation: pending."])
    atomic_text(output_dir / "report.md", "\n".join(lines) + "\n")
    assert sha256_file(records_path) == original_hash, "Source checkpoint changed during recheck"
    progress(f"Checker recheck: {len(llm)} Gemini reports; {counts['pass']} pass, "
             f"{counts['needs_review']} need review. Original outputs retained. See {output_dir / 'report.md'}")
    return data_dir / "records_rechecked.jsonl"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default=str(SOURCE))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run(args.records, args.output_dir)
