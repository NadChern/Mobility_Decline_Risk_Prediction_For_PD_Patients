"""Additional synthetic challenge checks and a blinded human annotation packet.

Assistant-authored challenges are development checks, not independently labeled validation.
The checker is not modified by this module, and human judgments are never prefilled.
"""

import json
from pathlib import Path

from .interpretation_contract import CHECKER_VERSION, check_interpretation
from .recheck_dataset_a import SOURCE, source_hashes
from ..run_support import atomic_text, read_jsonl, write_jsonl
from ..synthesis_protocol import PROJECT_ROOT, sha256_file, sha256_text
from ..taxonomy import FEATURE_CATEGORY


OUTPUT = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/checker_v2_1/validation_readiness"


def challenge_cases():
    factors = [{"factor": name, "domain": FEATURE_CATEGORY[name], "direction": side}
               for name, side in [("FOG (Freezing of Gait)", "supports_prediction"),
                                  ("Gait", "supports_prediction"), ("Age", "supports_prediction"),
                                  ("PD Duration", "opposes_prediction")]]
    def report(group, singleton="age", other="PD duration"):
        return (f"The prediction was associated with {group}, alongside {singleton}. "
                f"Factors pushing toward higher severity included {other}. "
                "These opposing factors did not outweigh the supporting factors.")
    cases = []
    for index, group in enumerate([
        "gait and mobility (Gait; FOG)", "gait and mobility (Gait / FOG)",
        "gait and mobility features (Gait and FOG)",
        "gait and mobility, comprising Gait and FOG",
        "gait and mobility domain (Gait and freezing of gait)",
        "gait and mobility, which encompasses Gait and FOG",
    ], 1):
        cases.append({"case_id": f"VALID-{index:02d}", "expected_assessment": "pass",
                      "prose": report(group), "factors": factors, "category": "mild"})
    invalid = [
        ("omitted member", report("gait and mobility (FOG)")),
        ("wrong group member", report("gait and mobility (FOG, MoCA)")),
        ("extra group member", report("gait and mobility (Gait, FOG, MoCA)")),
        ("wrong domain", report("motor impairment (Gait, FOG)")),
        ("missing singleton", report("gait and mobility (Gait, FOG)", singleton="an unnamed factor")),
        ("wrong opposing factor", report("gait and mobility (Gait, FOG)", other="MoCA")),
        ("unsupported spelling guess", report("gait and mobility (Gaitt, FOG)")),
        ("reversed direction", report("gait and mobility (Gait, FOG)").replace("higher severity", "lower severity")),
    ]
    for index, (description, prose) in enumerate(invalid, 1):
        cases.append({"case_id": f"INVALID-{index:02d}", "description": description,
                      "expected_assessment": "needs_review", "prose": prose,
                      "factors": factors, "category": "mild"})
    # Semantically intelligible, but deliberately outside the currently recognized grammar.
    # Review is the safe outcome; it must not be presented as a demonstrated model error.
    for index, group in enumerate(["gait and mobility: Gait and FOG",
                                  "gait and mobility, consisting of Gait and FOG"], 1):
        cases.append({"case_id": f"REVIEW-{index:02d}", "expected_assessment": "needs_review",
                      "description": "Unrecognized yet intelligible group introduction",
                      "prose": report(group), "factors": factors, "category": "mild"})
    return cases


def write_once(path, rows):
    if not Path(path).exists():
        write_jsonl(path, rows)


def run(output_dir=OUTPUT, records_path=SOURCE):
    output_dir = Path(output_dir)
    manifest = {"source_sha256": sha256_file(records_path), "checker_version": CHECKER_VERSION,
                "checker_code_sha256": source_hashes(),
                "challenge_sha256": sha256_text(json.dumps(challenge_cases(), sort_keys=True))}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Readiness inputs changed; choose a new output directory to preserve annotations.")
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    results = []
    for case in challenge_cases():
        result = check_interpretation(case["prose"], case["factors"], case["category"])
        results.append({**case, "observed_assessment": result["assessment"],
                        "expectation_met": result["assessment"] == case["expected_assessment"],
                        "findings": result["issues"]})
    write_jsonl(output_dir / "challenge_results.jsonl", results)

    packet, key, forms = [], [], []
    for row in read_jsonl(records_path):
        if row["status"] not in {"complete", "contract_failed"}:
            continue
        identifier = "R-" + sha256_text(f"{row['patient_id']}:{row['method']}")[:12]
        packet.append({"review_id": identifier, "prediction": row["prediction"],
                       "interpretation": row["interpretation"]})
        key.append({"review_id": identifier, "patient_id": row["patient_id"], "method": row["method"]})
        forms.append({"review_id": identifier, "reviewer": "", "review_status": "pending",
                      "explicit_groups": [], "factor_mentions_with_sides": [],
                      "ambiguous_spans": [], "notes": ""})
    packet.sort(key=lambda r: r["review_id"])
    forms.sort(key=lambda r: r["review_id"])
    write_once(output_dir / "blinded_reports.jsonl", packet)
    write_once(output_dir / "review_identity_key.jsonl", key)
    write_once(output_dir / "human_annotations.jsonl", forms)
    # Keep predictions and vocabulary but hide method identity, parser output, and reference groups.
    instructions = ["# Human extraction review packet", "",
        "Purpose: independently label what the interpretation explicitly says. This is not clinical review.", "",
        "Do not open the identity key, checker findings, or reference taxonomy before annotating. "
        "Record the stated group label, explicitly named members, supporting/opposing side, and "
        "a source quote. Do not add members based on clinical knowledge or the expected taxonomy. "
        "Mark unclear wording as ambiguous rather than guessing.", "",
        "Fill human_annotations.jsonl: reviewer name, review_status=reviewed, explicit_groups "
        "(label/members/side/quote), factor_mentions_with_sides (factor/side/quote), and any ambiguous_spans. "
        "An empty group list can be a valid annotation for a genuinely ungrouped report.", "",
        "Canonical factor vocabulary: " + "; ".join(sorted(FEATURE_CATEGORY)) + ".", ""]
    for row in packet:
        instructions.extend([f"## {row['review_id']}", "", f"Prediction: {row['prediction']}", "",
                             row["interpretation"], ""])
    if not (output_dir / "human_review_packet.md").exists():
        atomic_text(output_dir / "human_review_packet.md", "\n".join(instructions))
    annotations = read_jsonl(output_dir / "human_annotations.jsonl")
    reviewed = sum(r.get("review_status") == "reviewed" and bool(r.get("reviewer", "").strip())
                   for r in annotations)
    met = sum(r["expectation_met"] for r in results)
    lines = ["# Synthesis evaluation: readiness and next steps", "",
        f"Additional synthetic challenge checks: {met}/{len(results)} expected outcomes matched. "
        "These are assistant-authored development checks, not independent validation.", "",
        f"Human annotations completed: {reviewed}/{len(packet)}. Annotations are not filled in by the assistant.", "",
        "## Current limitations", "",
        "The checker still routes unfamiliar introductions such as `gait and mobility: Gait and FOG` "
        "or `consisting of` to extraction review. An unfamiliar expression must not be reported as "
        "an established report error. Overall clinical correctness and every possible phrasing are not validated.", "",
        "## Before final testing", "",
        "- [ ] Human reviewer labels the blinded packet; compare with parser extraction and resolve disagreements.",
        "- [ ] Obtain a separately authored/labeled validation corpus not used for checker tuning. "
        "The pilot and these synthetic cases do not satisfy this requirement.",
        "- [ ] Predefine acceptance criteria, review fallback, dataset size/IDs and reporting denominators.",
        "- [ ] Freeze the checker, taxonomy, prompts, patient selection and analysis settings before final generation.",
        "- [ ] Do not tune the checker on final-test results; report uncertain cases with the frozen fallback.", "",
        "## Next paid step (not executed)", "",
        "After reviewing the pilot extraction, run a small capped judge smoke test, inspect both judges' "
        "structured replies, then explicitly authorize the full panel. Current prepared workload: "
        "360 paired narrative calls plus 186 map-free claim calls = 546 calls. Costs are not estimated "
        "without current provider pricing and actual token use. The same commands resume saved requests.", ""]
    for result in results:
        if not result["expectation_met"]:
            lines.append(f"- Challenge requiring investigation: {result['case_id']} "
                         f"(expected {result['expected_assessment']}, observed {result['observed_assessment']}).")
    atomic_text(output_dir / "NEXT_STEPS.md", "\n".join(lines) + "\n")
    return results


if __name__ == "__main__":
    run()
