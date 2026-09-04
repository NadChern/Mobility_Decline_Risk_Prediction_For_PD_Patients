"""Regenerate deterministic synthesis artifacts in isolation and compare SHA-256 hashes."""

from __future__ import annotations

import tempfile
import json
from pathlib import Path

import pandas as pd

from .map_free_analysis import run as run_map_free
from .parser_validation import run as run_parser_validation
from ..experimental.flexibility_stress_test import run as run_flexibility
from ..generate_dataset_a import run as run_dataset_a
from ..synthesis_protocol import PROJECT_ROOT, canonical_json, sha256_file, sha256_text


CURRENT = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/checker_v2_1"
OUTPUT = CURRENT / "provenance/freeze_checksums.csv"


def deterministic_flex_hash(frame):
    """Hash typed deterministic values, independent of nullable LLM CSV columns."""
    frame = frame[frame.method != "map_free_gemini"].sort_values(["case_id", "method"])
    booleans = {"renders_without_task_specific_edits", "grouping_success",
                "explicit_traceability", "per_case_success"}
    rows = frame.to_dict("records")
    for row in rows:
        for key, value in row.items():
            if key in booleans:
                if str(value).lower() not in {"true", "false"}:
                    raise ValueError(f"Invalid deterministic boolean: {key}={value}")
                row[key] = str(value).lower() == "true"
            elif key in {"compositional_fidelity", "unsupported_content_count"}:
                row[key] = None if value == "" or pd.isna(value) else float(value)
            elif key == "explicit_claims":
                row[key] = json.loads(value)
    return sha256_text(canonical_json(rows))


def run(output_path=OUTPUT, *, map_free_output_dir=None, parser_output_dir=None):
    map_free_output_dir = Path(map_free_output_dir or CURRENT / "part2a_map_free")
    parser_output_dir = Path(parser_output_dir or CURRENT / "parser_validation")
    with tempfile.TemporaryDirectory(prefix="synthesis-v2-") as temporary:
        temp = Path(temporary)
        run_dataset_a(generate=False, output_dir=temp / "dataset_a")
        run_map_free(output_dir=temp / "map_free")
        run_flexibility(output_dir=temp / "flexibility", generate=False)
        run_parser_validation(output_dir=temp / "parser")
        pairs = (
            (map_free_output_dir / "patients.csv", temp / "map_free/patients.csv"),
            (parser_output_dir / "parser_validation.csv", temp / "parser/parser_validation.csv"),
        )
        rows = []
        for current, regenerated in pairs:
            current_hash, regenerated_hash = sha256_file(current), sha256_file(regenerated)
            rows.append({"artifact": str(current.relative_to(PROJECT_ROOT)) if current.is_relative_to(PROJECT_ROOT) else str(current),
                         "current_sha256": current_hash,
                         "regenerated_sha256": regenerated_hash,
                         "deterministic_match": current_hash == regenerated_hash})
        current_records = [json.loads(line) for line in
                           (PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/dataset_a/records.jsonl").read_text(encoding="utf-8").splitlines()]
        regenerated_records = [json.loads(line) for line in
                               (temp / "dataset_a/records.jsonl").read_text(encoding="utf-8").splitlines()]
        current_domain = [record for record in current_records if record["method"] == "domain_template"]
        regenerated_domain = [record for record in regenerated_records if record["method"] == "domain_template"]
        current_hash = sha256_text(canonical_json(current_domain))
        regenerated_hash = sha256_text(canonical_json(regenerated_domain))
        rows.append({"artifact": "dataset_a/domain_template_records",
                     "current_sha256": current_hash, "regenerated_sha256": regenerated_hash,
                     "deterministic_match": current_hash == regenerated_hash})
        current_flex = pd.read_csv(
            PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/part2b_held_out/cases.csv",
            keep_default_na=False).query("method != 'map_free_gemini'")
        regenerated_flex = pd.read_csv(temp / "flexibility/cases.csv", keep_default_na=False).query(
            "method != 'map_free_gemini'")
        current_hash = deterministic_flex_hash(current_flex)
        regenerated_hash = deterministic_flex_hash(regenerated_flex)
        rows.append({"artifact": "part2b_held_out/deterministic_arms",
                     "current_sha256": current_hash, "regenerated_sha256": regenerated_hash,
                     "deterministic_match": current_hash == regenerated_hash})
    result = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    if not result.deterministic_match.all():
        failed = result.loc[~result.deterministic_match, "artifact"].tolist()
        raise AssertionError(f"Deterministic synthesis artifacts changed: {', '.join(failed)}. "
                             f"See {output_path}. Saved LLM responses are unaffected.")
    return result


if __name__ == "__main__":
    run()
