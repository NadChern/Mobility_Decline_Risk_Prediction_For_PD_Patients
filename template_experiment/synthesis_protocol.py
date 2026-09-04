"""Frozen evidence serialization and prompts for synthesis evaluation v2."""

from __future__ import annotations

import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from explanation.explanation_builder import build_patient_explanation_data_full, get_patient_index
from explanation.llm import _build_prompt

from .generic_template import _OUTCOME_PHRASE, _ordered_tables
from .taxonomy import CATEGORY_LABELS, FEATURE_CATEGORY, TAXONOMY_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = PROJECT_ROOT / "template_experiment/protocol"
PILOT_DATASET = PROJECT_ROOT / "template_experiment/outputs/pilot/pilot_reports.csv"
PROTOCOL_VERSION = "synthesis-v2.1-prompt-twin"
SENTENCE_RANGE = (2, 4)


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_factor_records(patient_info):
    """Serialize the exact top-5/top-3 interpretation evidence and taxonomy mapping."""
    stage = 2 if patient_info["has_severity_assessment"] else 1
    records = []
    for side, (_heading, factors), limit in zip(
        ("supports_prediction", "opposes_prediction"),
        _ordered_tables(patient_info),
        (5, 3),
    ):
        for rank, factor in enumerate(factors[:limit], start=1):
            domain = FEATURE_CATEGORY.get(factor["short_name"])
            records.append({
                "factor": factor["short_name"],
                "domain": domain,
                "domain_label": CATEGORY_LABELS.get(domain) if domain else None,
                "direction": side,
                "rank": rank,
                "stage": stage,
                "value": str(factor["display_value"]),
                "scale": str(factor["interpretation"]),
            })
    return records


def evidence_packet(patient_info):
    packet = {
        "patient_id": int(patient_info["patient_id"]),
        "prediction": _OUTCOME_PHRASE[int(patient_info["final_prediction"])],
        "stage": 2 if patient_info["has_severity_assessment"] else 1,
        "selected_factors": selected_factor_records(patient_info),
    }
    packet["evidence_hash"] = sha256_text(canonical_json(packet))
    return packet


TAXONOMY_INSTRUCTION = (
    "- TAXONOMY CONDITION ONLY: For the Model Interpretation, use the supplied domain assignments "
    "instead of inferring categories. When at least two selected factors on the same evidence side "
    "share a domain, combine them under its supplied domain label and name the individual members. "
    "Keep factors that do not share a domain with another selected same-side factor as individual "
    "factors. Treat this taxonomy as an experimental organization, not independent clinical truth."
)

TAXONOMY_CONDITIONED_PROMPT_SPEC = f"""# Taxonomy-conditioned prompt twin

Base prompt: the exact text returned by `explanation.llm._build_prompt(patient_info)`.

The taxonomy condition adds only:

1. A data block before `TASK:` containing factor, evidence side, rank, domain ID, and domain label
   for the same top-5/top-3 interpretation factors.
2. This instruction immediately before the unchanged `MODEL INTERPRETATION` instruction:

{TAXONOMY_INSTRUCTION}

The complete per-patient prompts and hashes are stored in Dataset A records.
"""


def _taxonomy_mapping_block(patient_info):
    lines = ["TAXONOMY CONDITION FOR THE SELECTED INTERPRETATION FACTORS:"]
    for item in selected_factor_records(patient_info):
        lines.append(
            f"- {item['direction']} rank {item['rank']} | factor: {item['factor']} | "
            f"domain_id: {item['domain'] or 'unmapped'} | "
            f"domain_label: {item['domain_label'] or 'unmapped'}"
        )
    return "\n".join(lines)


def build_taxonomy_conditioned_prompt(patient_info):
    """Return a full-document twin of the map-free prompt plus taxonomy data/instruction only."""
    prompt = _build_prompt(patient_info)
    taxonomy_block = _taxonomy_mapping_block(patient_info)
    task_marker = "\n\nTASK:\n"
    interpretation_marker = "\n- MODEL INTERPRETATION:"
    if task_marker not in prompt or interpretation_marker not in prompt:
        raise ValueError("Map-free prompt markers changed; review the taxonomy-twin transformation.")
    prompt = prompt.replace(task_marker, f"\n\n{taxonomy_block}{task_marker}", 1)
    return prompt.replace(
        interpretation_marker,
        f"\n{TAXONOMY_INSTRUCTION}{interpretation_marker}",
        1,
    )


def remove_taxonomy_conditioning(prompt, patient_info):
    """Remove the two taxonomy-only additions; used to prove the remaining prompt is identical."""
    taxonomy_block = _taxonomy_mapping_block(patient_info)
    result = str(prompt).replace(f"\n\n{taxonomy_block}\n\nTASK:\n", "\n\nTASK:\n", 1)
    return result.replace(f"\n{TAXONOMY_INSTRUCTION}\n- MODEL INTERPRETATION:",
                          "\n- MODEL INTERPRETATION:", 1)


def validate_interpretation_contract(interpretation, patient_info):
    """Compatibility API: versioned diagnostics without rewriting the generated text."""
    from .evaluation.interpretation_contract import check_interpretation

    category = "moderate" if int(patient_info["final_prediction"]) == 2 else (
        "mild" if int(patient_info["final_prediction"]) == 1 else "no_falls"
    )
    result = check_interpretation(
        interpretation, selected_factor_records(patient_info), category, SENTENCE_RANGE,
    )
    return [issue["code"] for issue in result["issues"]]


def freeze_protocol(force=False):
    """Write auditable protocol inputs and a checksum manifest; safe to rerun before generation."""
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PROTOCOL_DIR / "synthesis_v2_manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    paired = pd.read_csv(PILOT_DATASET, keep_default_na=False)
    patient_ids = [int(value) for value in paired["patient_id"]]
    packets = [
        evidence_packet(build_patient_explanation_data_full(get_patient_index(patient_id)))
        for patient_id in patient_ids
    ]

    taxonomy_path = PROJECT_ROOT / "template_experiment/taxonomy.py"
    domain_path = PROJECT_ROOT / "template_experiment/domain_template.py"
    parser_path = PROJECT_ROOT / "template_experiment/evaluation/group_claims.py"
    metric_path = PROJECT_ROOT / "template_experiment/evaluation/synthesis.py"
    prompt_path = PROTOCOL_DIR / "taxonomy_conditioned_prompt.txt"
    map_free_path = PROTOCOL_DIR / "map_free_prompt.txt"
    packet_path = PROTOCOL_DIR / "dataset_b_evidence_packets.jsonl"
    prompt_path.write_text(TAXONOMY_CONDITIONED_PROMPT_SPEC, encoding="utf-8")
    map_free_path.write_text(inspect.getsource(_build_prompt), encoding="utf-8")
    packet_path.write_text(
        "\n".join(canonical_json(packet) for packet in packets) + "\n", encoding="utf-8"
    )

    source_metadata = (
        paired[[column for column in ("gemini_model", "source_temperature", "source_run", "source_file")
                if column in paired.columns]]
        .drop_duplicates().to_dict("records")
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "taxonomy_version": TAXONOMY_VERSION,
        "patient_selection": "30-patient frozen pilot: 10 no_falls, 10 mild, 10 moderate",
        "patient_ids": patient_ids,
        "dataset_b": {
            "path": str(PILOT_DATASET.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(PILOT_DATASET),
            "map_free": True,
            "explicit_taxonomy_mapping_supplied": False,
            "taxonomy_input_verification": "Frozen production prompt source contains no feature-to-domain mapping table.",
            "generation_metadata": source_metadata,
        },
        "output_contract": {
            "explicit_group_membership": (
                "Category label plus canonical named members; the template uses "
                "category label (Canonical Factor A, Canonical Factor B), while equivalent "
                "unambiguous connector forms are accepted."
            ),
            "sentence_min": SENTENCE_RANGE[0],
            "sentence_max": SENTENCE_RANGE[1],
            "max_interpretation_words": None,
            "prompt_matching": "Full-document twins; taxonomy condition adds only mapping block and one taxonomy-use instruction.",
        },
        "files": {},
    }
    additional_protocol = [
        path for path in (PROTOCOL_DIR / "judge_prompt.txt", PROTOCOL_DIR / "judge_rubric.md",
                          PROTOCOL_DIR / "held_out_cases.json", PROTOCOL_DIR / "parser_gold.csv")
        if path.exists()
    ]
    implementation_paths = (
        PROJECT_ROOT / "template_experiment/generate_dataset_a.py",
        PROJECT_ROOT / "template_experiment/evaluation/part1_structural.py",
        PROJECT_ROOT / "template_experiment/evaluation/map_free_analysis.py",
        PROJECT_ROOT / "template_experiment/evaluation/parser_validation.py",
        PROJECT_ROOT / "template_experiment/evaluation/prose_checks.py",
        PROJECT_ROOT / "template_experiment/evaluation/build_pilot_dataset.py",
        PROJECT_ROOT / "template_experiment/evaluation/reproducibility.py",
        PROJECT_ROOT / "template_experiment/experimental/flexibility_stress_test.py",
        PROJECT_ROOT / "template_experiment/experimental/llm_judge.py",
        PROJECT_ROOT / "template_experiment/experimental/llm_judge_map_free.py",
        PROJECT_ROOT / "template_experiment/run_synthesis_v2.py",
        PROJECT_ROOT / "explanation/config.yaml",
        PROJECT_ROOT / "requirements-dev.txt",
    )
    for path in (taxonomy_path, domain_path, parser_path, metric_path, prompt_path, map_free_path,
                 packet_path, *additional_protocol, *implementation_paths):
        manifest["files"][str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    freeze_protocol(force=args.force)
