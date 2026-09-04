"""Versioned retrospective Dataset B analysis and six-category disagreement worksheet."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .synthesis import compute_results, summarize
from .group_claims import PARSER_VERSION
from ..run_support import atomic_text, output_lock, progress
from ..synthesis_protocol import PILOT_DATASET, PROJECT_ROOT, sha256_file, sha256_text


OUTPUT_DIR = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/checker_v2_1/part2a_map_free"
HISTORY_DIR = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/history/map_free_versions"
AUTHOR_WORKSHEET = OUTPUT_DIR / "disagreement_worksheet.csv"
REPORT_VERSION = "map-free-adjudicated-report-v1.1"
ADJUDICATION_RUBRIC = PROJECT_ROOT / "template_experiment/protocol/adjudication_rubric.md"
METHOD = {"map_free_gemini": ("gemini_model", "gemini_explanation")}
ADJUDICATION_CATEGORIES = (
    "reference_match",
    "granularity_difference",
    "plausible_alternative_abstraction",
    "over_grouping",
    "missed_grouping_opportunity",
    "unsupported_or_unclear",
)


def _pairs(value):
    return {
        (item["stage"], item["side"], *item["factors"])
        for item in json.loads(value)
    }


def _suggest_category(row):
    gold, predicted = _pairs(row.gold_pairs), _pairs(row.predicted_pairs)
    if gold == predicted:
        return "reference_match"
    if not gold and predicted:
        return "over_grouping"
    if gold and not predicted:
        return "missed_grouping_opportunity"
    if gold & predicted:
        return "granularity_difference"
    if gold and predicted:
        return "plausible_alternative_abstraction"
    return "unsupported_or_unclear"


# Author-entered columns are preserved across regenerations; everything else is recomputed.
_AUTHOR_COLUMNS = ("author_adjudication", "author_rationale", "review_status")
_REVIEW_INPUTS = ("patient_id", "category", "stage", "gold_groups", "map_free_groups", "interpretation")


def _review_input_hash(row):
    return sha256_text(json.dumps({column: str(row.get(column, "")) for column in _REVIEW_INPUTS}, sort_keys=True))


def _merge_prior_adjudications(worksheet, prior_path):
    """Carry hand-entered author columns forward, keyed by patient_id, on regeneration.

    Preserve drafts as well as completed decisions. If review inputs changed, retain the
    author's text but require re-review; never silently apply it to different groupings.
    """
    if not prior_path.exists():
        return worksheet
    prior = pd.read_csv(prior_path, keep_default_na=False)
    if "author_adjudication" not in prior.columns:
        return worksheet
    if prior.patient_id.duplicated().any():
        raise ValueError(f"Duplicate patient IDs in adjudication source: {prior_path}")
    reviewed = prior[
        prior["author_adjudication"].astype(str).str.strip().ne("")
        | prior.get("author_rationale", pd.Series("", index=prior.index)).astype(str).str.strip().ne("")
    ]
    prior_by_id = reviewed.set_index("patient_id")
    for pid, prior_row in prior_by_id.iterrows():
        mask = worksheet["patient_id"] == pid
        if mask.any():
            current_hash = _review_input_hash(worksheet.loc[mask].iloc[0])
            prior_hash = prior_row.get("adjudication_input_hash", "") or _review_input_hash(
                {"patient_id": pid, **prior_row.to_dict()})
            for column in _AUTHOR_COLUMNS:
                if column in prior_row:
                    worksheet.loc[mask, column] = prior_row[column]
            worksheet.loc[mask, "adjudication_input_hash"] = prior_hash
            if current_hash != prior_hash:
                worksheet.loc[mask, "review_status"] = "needs_re_review_inputs_changed"
    return worksheet


def disagreement_worksheet(patients, prior_path=None):
    rows = []
    for row in patients.itertuples(index=False):
        suggested = _suggest_category(row)
        if suggested == "reference_match":
            continue
        rows.append({
            "patient_id": row.patient_id,
            "category": row.category,
            "stage": row.stage,
            "gold_groups": row.gold_groups,
            "map_free_groups": row.predicted_groups,
            "suggested_category": suggested,
            "allowed_categories": "|".join(ADJUDICATION_CATEGORIES),
            "author_adjudication": "",
            "author_rationale": "",
            "review_status": "pending_blinded_author_review",
            "interpretation": row.interpretation,
        })
    worksheet = pd.DataFrame(rows, columns=["patient_id", "category", "stage", "gold_groups",
        "map_free_groups", "suggested_category", "allowed_categories", "author_adjudication",
        "author_rationale", "review_status", "interpretation"])
    if prior_path is not None:
        worksheet = _merge_prior_adjudications(worksheet, Path(prior_path))
    return worksheet


def adjudication_status(worksheet):
    labels = worksheet.author_adjudication.astype(str).str.strip()
    rationale = worksheet.author_rationale.astype(str).str.strip()
    status = worksheet.review_status.astype(str).str.strip()
    valid = labels.isin(ADJUDICATION_CATEGORIES)
    current = status.ne("needs_re_review_inputs_changed")
    complete = valid & current & rationale.ne("") & status.eq("author_reviewed")
    return {
        "total_disagreements": len(worksheet), "labels_entered": int(labels.ne("").sum()),
        "complete_reviews": int(complete.sum()),
        "author_counts": labels[valid & current].value_counts().to_dict(),
        "complete_counts": labels[complete].value_counts().to_dict(),
        "automatic_counts": worksheet.suggested_category.value_counts().to_dict(),
        "missing_rationale_ids": worksheet.loc[valid & rationale.eq(""), "patient_id"].tolist(),
        "invalid_label_ids": worksheet.loc[labels.ne("") & ~valid, "patient_id"].tolist(),
        "stale_review_ids": worksheet.loc[~current, "patient_id"].tolist(),
    }


def rubric_provenance():
    text = ADJUDICATION_RUBRIC.read_text(encoding="utf-8")
    version = re.search(r"^- Version: `([^`]+)`", text, re.MULTILINE)
    if not version:
        raise ValueError("Adjudication rubric must declare its version.")
    digest = sha256_text(text)
    return text, {"version": version.group(1), "sha256": digest,
                  "source": str(ADJUDICATION_RUBRIC.resolve()),
                  "snapshot": f"rubric_snapshots/{digest}.md",
                  "development_scope": "pilot-developed author adjudication; not independent clinical validation"}


def render_report(worksheet, rubric=None):
    if rubric is None:
        _, rubric = rubric_provenance()
    status = adjudication_status(worksheet)
    total = status["total_disagreements"]
    final = status["complete_reviews"] == total
    lines = ["# Part 2A — retrospective map-free reference agreement", "",
        f"Parser: `{PARSER_VERSION}`. Report: `{REPORT_VERSION}`.", "",
        f"Adjudication rubric: [`{rubric['version']}`]({rubric['snapshot']}) (pilot-developed).", "",
        "The frozen historical Gemini reports did not receive the experiment taxonomy. Pairwise "
        "scores measure agreement with that authored reference, not clinical accuracy. Author "
        "adjudications below are qualitative explanations of disagreement; they do not change "
        "the primary pairwise scores or establish clinical validity.", "",
        "## Author adjudication" + ("" if final else " — provisional"), "",
        f"Non-exact reports: {total}. Author labels entered: {status['labels_entered']}/{total}. "
        f"Complete reviews (valid label, rationale, and author_reviewed status): {status['complete_reviews']}/{total}.", "",
        "The complete-review column requires a valid label, rationale, and author-reviewed status. "
        "Stale decisions are excluded from both columns. Zero unsupported labels, if observed, "
        "does not establish an absence of clinical errors.", "",
        "| Category | Author labels | Complete reviews |", "| --- | ---: | ---: |"]
    for category in ADJUDICATION_CATEGORIES:
        lines.append(f"| {category} | {status['author_counts'].get(category, 0)} | {status['complete_counts'].get(category, 0)} |")
    for key, label in [("missing_rationale_ids", "Missing rationales"),
                       ("invalid_label_ids", "Invalid author labels"),
                       ("stale_review_ids", "Changed review inputs; re-review required")]:
        if status[key]:
            lines.extend(["", f"{label}: {', '.join(str(i) for i in status[key])}."])
    lines.extend(["", "## Automatic routing — not author adjudication", "",
                  "These suggestions are retained for traceability only; do not present them as "
                  "the author's decisions.", "",
                  json.dumps(status["automatic_counts"], sort_keys=True), "",
                  "Review scope: the disagreement subset, not all reports. Exact matches remain "
                  "in full-cohort metric denominators and do not require disagreement adjudication.", ""])
    return "\n".join(lines)


def run(paired_path=PILOT_DATASET, output_dir=OUTPUT_DIR, adjudications_path=None):
    with output_lock(output_dir):
        return _run(paired_path, Path(output_dir), adjudications_path)


def _run(paired_path, output_dir, adjudications_path):
    rubric_text, rubric = rubric_provenance()
    paired = pd.read_csv(paired_path, keep_default_na=False)
    patients = compute_results(paired, METHOD)
    summary = summarize(patients)
    output_dir = Path(output_dir)
    worksheet = disagreement_worksheet(
        patients, prior_path=Path(adjudications_path) if adjudications_path is not None else output_dir / "disagreement_worksheet.csv"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = HISTORY_DIR if output_dir.resolve() == OUTPUT_DIR.resolve() else output_dir / "history"
    source = Path(adjudications_path) if adjudications_path is not None else output_dir / "disagreement_worksheet.csv"
    source_digest = sha256_file(source) if source.exists() else None
    # Keep immutable pre-regeneration snapshots, including notes on rows now exact/removed.
    for path in {source, output_dir / "disagreement_worksheet.csv"}:
        if path.exists():
            snapshot = history_dir / "worksheet_snapshots" / f"{sha256_file(path)}.csv"
            if not snapshot.exists():
                atomic_text(snapshot, path.read_text(encoding="utf-8"))
    for name in ("patients.csv", "summary.csv", "report.md", "report_manifest.json"):
        previous = output_dir / name
        if previous.exists():
            snapshot = history_dir / "report_history" / sha256_file(previous) / name
            if not snapshot.exists():
                atomic_text(snapshot, previous.read_text(encoding="utf-8"))
    atomic_text(output_dir / "patients.csv", patients.to_csv(index=False))
    atomic_text(output_dir / "summary.csv", summary.to_csv(index=False))
    atomic_text(output_dir / "disagreement_worksheet.csv", worksheet.to_csv(index=False))
    rubric_snapshot = output_dir / rubric["snapshot"]
    if not rubric_snapshot.exists():
        atomic_text(rubric_snapshot, rubric_text)
    elif rubric_snapshot.read_text(encoding="utf-8") != rubric_text:
        raise ValueError("Adjudication rubric snapshot differs from its content hash.")
    atomic_text(output_dir / "report.md", render_report(worksheet, rubric))
    provenance = {"report_version": REPORT_VERSION, "parser_version": PARSER_VERSION,
                  "adjudication_rubric": rubric,
                  "report_generator_sha256": sha256_file(Path(__file__)),
                  "adjudication_source": str(source.resolve()), "adjudication_source_sha256": source_digest,
                  "worksheet_sha256": sha256_file(output_dir / "disagreement_worksheet.csv"),
                  "patient_metrics_sha256": sha256_file(output_dir / "patients.csv"),
                  "adjudication_status": adjudication_status(worksheet)}
    atomic_text(output_dir / "report_manifest.json", json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    status = provenance["adjudication_status"]
    progress(f"Part 2A author review: {status['labels_entered']}/{len(worksheet)} labels; "
             f"{status['complete_reviews']}/{len(worksheet)} complete reviews. See {output_dir / 'report.md'}")
    return patients, summary, worksheet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", default=str(PILOT_DATASET))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--adjudications", help="Explicit authoring worksheet to import; input identity is checked.")
    args = parser.parse_args()
    run(args.paired, args.output_dir, args.adjudications)


if __name__ == "__main__":
    main()
