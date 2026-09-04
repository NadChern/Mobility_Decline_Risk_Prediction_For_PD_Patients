"""Pairwise structural-synthesis evaluation for interpretation prose.

The current pilot Gemini reports were generated without the explicit factor-to-domain taxonomy.
Accordingly, results written from the default pilot dataset are a retrospective map-free analysis
(Part 2A of ``SYNTHESIS_EVALUATION_IMPLEMENTATION_PLAN.md``), not the future same-information
taxonomy-conditioned comparison (Part 1).

Metrics operate on same-group factor pairs and retain patient, stage, and evidence side in the pair
identity. Explicit-membership recall and contrastive linkage are reported separately. Old proxies
such as grouping extent, distinct-n, factors-per-sentence, and output length are excluded from the
synthesis results.
"""

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .group_claims import extract_group_claims
from .prose_checks import _detect_segment, _expected_grouping, _segments, _sides
from .shared import METHODS, canonical_labels, expected_evidence, parse_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRED = PROJECT_ROOT / "template_experiment/outputs/pilot/pilot_reports.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "template_experiment/outputs/synthesis_v2/history/map_free_three_method"
)

MACRO_METRICS = (
    "pairwise_grouping_precision",
    "pairwise_grouping_recall",
    "pairwise_grouping_f1",
    "explicit_membership_recall",
    "contrastive_linkage",
)

_CONTRASTIVE_PATTERNS = (
    r"\balthough\b",
    r"\bhowever\b",
    r"\bconversely\b",
    r"\bin contrast\b",
    r"\bwhile\b",
    r"\bbut\b",
    r"\bdid not outweigh\b",
    r"\bdespite\b",
)


def _expected_relations(evidence, category):
    """Return taxonomy-defined eligible groups as ``(side, category, members)`` tuples."""
    supporting, opposing = _sides(evidence, category)
    expected = _expected_grouping(supporting, opposing)
    return [
        (side, group, set(members))
        for side in ("supporting", "opposing")
        for group, members in expected[side]["groups"].items()
    ]


def _observed_relations(prose, evidence, category):
    """Return explicit claims and conditional membership counts using one parser for all methods."""
    supporting, opposing = _sides(evidence, category)
    expected = _expected_grouping(supporting, opposing)
    claims = extract_group_claims(prose, category)
    relations = [(claim.side, claim.group, set(claim.members)) for claim in claims]
    explicit_correct = 0
    intended_total = 0
    for claim in claims:
        reference_members = expected[claim.side]["groups"].get(claim.group)
        if reference_members:
            intended_total += len(reference_members)
            explicit_correct += len(set(claim.members) & set(reference_members))
    return relations, explicit_correct, intended_total, claims


def _pair_set(relations, stage):
    """Convert group relations to stage/side-aware unordered factor-pair identities."""
    pairs = set()
    for side, _group, members in relations:
        for left, right in combinations(sorted(set(members)), 2):
            pairs.add((int(stage), side, left, right))
    return pairs


def pairwise_grouping_scores(gold_pairs, predicted_pairs):
    """Return precision, recall, and F1 using the frozen empty-set conventions."""
    gold_pairs = set(gold_pairs)
    predicted_pairs = set(predicted_pairs)
    hits = len(gold_pairs & predicted_pairs)

    precision = hits / len(predicted_pairs) if predicted_pairs else np.nan
    recall = hits / len(gold_pairs) if gold_pairs else np.nan

    if not gold_pairs and not predicted_pairs:
        f1 = np.nan
    elif not gold_pairs or not predicted_pairs:
        f1 = 0.0
    elif precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


def _contrastive_linkage(prose, evidence, category):
    """Score correct two-sided representation and explicit linkage as 0, 0.5, 1, or N/A."""
    supporting, opposing = _sides(evidence, category)
    if not supporting or not opposing:
        return np.nan

    required_by_side = {"supporting": supporting, "opposing": opposing}
    represented = set()
    for segment, side in _segments(prose, category):
        features, groups = _detect_segment(segment)
        if (features & required_by_side[side]) or groups:
            represented.add(side)
    if represented != {"supporting", "opposing"}:
        return 0.0

    linked = any(re.search(pattern, str(prose), re.I) for pattern in _CONTRASTIVE_PATTERNS)
    return 1.0 if linked else 0.5


def _serialize_relations(relations):
    records = sorted([
        {"side": side, "group": group, "members": sorted(members)}
        for side, group, members in relations
    ], key=lambda item: (item["side"], item["group"], item["members"]))
    return json.dumps(records, sort_keys=True)


def _serialize_pairs(pairs):
    records = [
        {"stage": stage, "side": side, "factors": [left, right]}
        for stage, side, left, right in sorted(pairs)
    ]
    return json.dumps(records, sort_keys=True)


def compute_patient_synthesis(source_row, method, model_col, explanation_col,
                              norm_set=None, norm_to_short=None):
    """Compute the structural-synthesis panel for one patient and one method."""
    if norm_set is None or norm_to_short is None:
        norm_set, norm_to_short = canonical_labels()
    report = parse_report(source_row[explanation_col], norm_set, norm_to_short)
    evidence = expected_evidence(source_row, norm_to_short)
    prose = report["interpretation"]
    category = source_row["category"]
    stage = int(source_row["stage"])

    gold_relations = _expected_relations(evidence, category)
    observed_relations, explicit_correct, intended_total, claims = _observed_relations(
        prose, evidence, category
    )
    gold_pairs = _pair_set(gold_relations, stage)
    predicted_pairs = _pair_set(observed_relations, stage)
    precision, recall, f1 = pairwise_grouping_scores(gold_pairs, predicted_pairs)
    true_positive_pairs = len(gold_pairs & predicted_pairs)

    return {
        "patient_id": int(source_row["patient_id"]),
        "category": category,
        "stage": stage,
        "method": method,
        "model": source_row[model_col],
        "has_gold_grouping_opportunity": bool(gold_pairs),
        "has_group_claim": bool(observed_relations),
        "gold_group_count": len(gold_relations),
        "predicted_group_count": len(observed_relations),
        "undersized_group_count": sum(
            len(members) < 2 for _side, _group, members in observed_relations
        ),
        "gold_pair_count": len(gold_pairs),
        "predicted_pair_count": len(predicted_pairs),
        "true_positive_pair_count": true_positive_pairs,
        "pairwise_grouping_precision": precision,
        "pairwise_grouping_recall": recall,
        "pairwise_grouping_f1": f1,
        "exact_pair_set_agreement": bool(gold_pairs == predicted_pairs),
        "explicit_membership_correct_count": explicit_correct,
        "explicit_membership_intended_count": intended_total,
        "explicit_membership_recall": (
            explicit_correct / intended_total if intended_total else np.nan
        ),
        "contrastive_linkage": _contrastive_linkage(prose, evidence, category),
        "gold_groups": _serialize_relations(gold_relations),
        "predicted_groups": _serialize_relations(observed_relations),
        "group_claims_with_spans": json.dumps(
            [claim.as_dict() for claim in claims], sort_keys=True
        ),
        "gold_pairs": _serialize_pairs(gold_pairs),
        "predicted_pairs": _serialize_pairs(predicted_pairs),
        "interpretation": prose,
    }


def compute_results(paired_df, methods=None):
    """Compute results for a column mapping; defaults to the historical three pilot arms."""
    methods = methods or METHODS
    norm_set, norm_to_short = canonical_labels()
    rows = [
        compute_patient_synthesis(
            source_row, method, model_col, explanation_col, norm_set, norm_to_short
        )
        for _, source_row in paired_df.iterrows()
        for method, (model_col, explanation_col) in methods.items()
    ]
    return pd.DataFrame(rows).sort_values(["patient_id", "method"]).reset_index(drop=True)


def _micro_f1(precision, recall, gold_total, predicted_total):
    if not gold_total and not predicted_total:
        return np.nan
    if not gold_total or not predicted_total:
        return 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def summarize(patient_results):
    """Summarize patient-macro and pooled-micro results with visible denominators."""
    rows = []
    groups = list(patient_results.groupby(["method", "category"]))
    groups += [
        ((method, "overall"), group)
        for method, group in patient_results.groupby("method")
    ]

    for (method, category), group in groups:
        result = {
            "method": method,
            "category": category,
            "n_patients": len(group),
            "n_gold_grouping_opportunity": int(group["has_gold_grouping_opportunity"].sum()),
            "n_with_group_claim": int(group["has_group_claim"].sum()),
        }
        for metric in MACRO_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            result[f"{metric}_macro_mean"] = float(values.mean()) if len(values) else np.nan
            result[f"{metric}_n_evaluable"] = int(len(values))

        gold_total = int(group["gold_pair_count"].sum())
        predicted_total = int(group["predicted_pair_count"].sum())
        true_positive_total = int(group["true_positive_pair_count"].sum())
        micro_precision = true_positive_total / predicted_total if predicted_total else np.nan
        micro_recall = true_positive_total / gold_total if gold_total else np.nan
        result.update({
            "gold_pair_count": gold_total,
            "predicted_pair_count": predicted_total,
            "true_positive_pair_count": true_positive_total,
            "pairwise_grouping_precision_micro": micro_precision,
            "pairwise_grouping_recall_micro": micro_recall,
            "pairwise_grouping_f1_micro": _micro_f1(
                micro_precision, micro_recall, gold_total, predicted_total
            ),
            "exact_pair_set_agreement_mean": float(group["exact_pair_set_agreement"].mean()),
            "exact_pair_set_agreement_n_evaluable": int(len(group)),
        })

        opportunity = group[group["has_gold_grouping_opportunity"]]
        result["exact_pair_set_agreement_with_opportunity_mean"] = (
            float(opportunity["exact_pair_set_agreement"].mean())
            if len(opportunity) else np.nan
        )
        result["exact_pair_set_agreement_with_opportunity_n_evaluable"] = int(
            len(opportunity)
        )

        no_opportunity = group[~group["has_gold_grouping_opportunity"]]
        result["spurious_group_claim_rate_no_opportunity"] = (
            float(no_opportunity["has_group_claim"].mean())
            if len(no_opportunity) else np.nan
        )
        result["spurious_group_claim_n_evaluable"] = int(len(no_opportunity))
        rows.append(result)

    return pd.DataFrame(rows).sort_values(["category", "method"]).reset_index(drop=True)


def _percent(value):
    return "N/A" if pd.isna(value) else f"{100 * value:.1f}%"


def _metric_cell(row, value_column, denominator_column=None):
    value = _percent(row[value_column])
    if denominator_column is None:
        return value
    return f"{value} ({int(row[denominator_column])}/{int(row['n_patients'])})"


def write_report(summary, output_path):
    overall = summary[summary["category"] == "overall"].set_index("method")
    method_order = [
        method for method in ("template", "template_grouped", "gemini")
        if method in overall.index
    ]
    labels = {
        "template": "Generic template",
        "template_grouped": "Domain template",
        "gemini": "Map-free LLM (Gemini)",
    }

    definitions = [
        ("Macro pairwise precision", "Mean report-level precision over evaluable reports.",
         "pairwise_grouping_precision_macro_mean", "pairwise_grouping_precision_n_evaluable"),
        ("Macro pairwise recall", "Mean report-level recall over reports with reference pairs.",
         "pairwise_grouping_recall_macro_mean", "pairwise_grouping_recall_n_evaluable"),
        ("Macro pairwise F1", "Mean report-level F1 using the frozen empty-set rules.",
         "pairwise_grouping_f1_macro_mean", "pairwise_grouping_f1_n_evaluable"),
        ("Micro pairwise precision", "Correct pooled predicted pairs / all pooled predicted pairs.",
         "pairwise_grouping_precision_micro", None),
        ("Micro pairwise recall", "Correct pooled predicted pairs / all pooled reference pairs.",
         "pairwise_grouping_recall_micro", None),
        ("Micro pairwise F1", "Harmonic mean of pooled precision and recall.",
         "pairwise_grouping_f1_micro", None),
        ("Exact pair-set agreement", "Reports with identical predicted and reference pair sets.",
         "exact_pair_set_agreement_mean", "exact_pair_set_agreement_n_evaluable"),
        ("Exact agreement with opportunity", "Exact agreement among reports with reference pairs.",
         "exact_pair_set_agreement_with_opportunity_mean",
         "exact_pair_set_agreement_with_opportunity_n_evaluable"),
        ("Explicit-membership recall", "Named intended members / intended members in produced claims.",
         "explicit_membership_recall_macro_mean", "explicit_membership_recall_n_evaluable"),
        ("Contrastive linkage", "Correct two-sided evidence explicitly linked to the model output.",
         "contrastive_linkage_macro_mean", "contrastive_linkage_n_evaluable"),
        ("Spurious claim without opportunity", "Group claims in reports with no reference pair.",
         "spurious_group_claim_rate_no_opportunity", "spurious_group_claim_n_evaluable"),
    ]

    lines = [
        "| Metric | Definition | " + " | ".join(labels[m] for m in method_order) + " |",
        "| --- | --- | " + " | ".join(["---:"] * len(method_order)) + " |",
    ]
    for label, definition, value_column, denominator_column in definitions:
        cells = [
            _metric_cell(overall.loc[method], value_column, denominator_column)
            for method in method_order
        ]
        lines.append(f"| {label} | {definition} | " + " | ".join(cells) + " |")

    pooled_lines = [
        "| Method | Reference pairs | Predicted pairs | Correct pairs | Reports with opportunity |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in method_order:
        row = overall.loc[method]
        pooled_lines.append(
            f"| {labels[method]} | {int(row['gold_pair_count'])} | "
            f"{int(row['predicted_pair_count'])} | {int(row['true_positive_pair_count'])} | "
            f"{int(row['n_gold_grouping_opportunity'])}/{int(row['n_patients'])} |"
        )

    output_path.write_text(
        "# Synthesis pilot — pairwise structural evaluation\n\n"
        "## Scope\n\n"
        "These results overwrite the former grouping-extent/Distinct-2 synthesis report. The saved "
        "Gemini interpretations were generated **without** the explicit task taxonomy, so this is "
        "the retrospective **map-free reference-taxonomy analysis (Part 2A)**. It is not the future "
        "same-taxonomy Part 1 noninferiority comparison.\n\n"
        "## Results\n\n" + "\n".join(lines)
        + "\n\nCells with parentheses show `n evaluable / 30 patients`. Macro metrics give each "
          "evaluable patient equal weight; micro metrics pool factor pairs and therefore give more "
          "weight to reports with larger grouping opportunities.\n\n"
        "## Pooled pair counts\n\n" + "\n".join(pooled_lines)
        + "\n\n## Interpretation limits\n\n"
          "The domain template's taxonomy agreement is expected by construction. Its regenerated "
          "v2 prose now names every group member explicitly, and the same method-agnostic parser is "
          "used for template and map-free prose. The map-free traceability result remains conditional "
          "on claims the parser can recognize; parser-validation results are published separately.\n\n"
          "Pairwise agreement measures concordance with one researcher-defined taxonomy, not clinical "
          "accuracy. Contrastive linkage is a structural compliance measure and is expected to show "
          "a ceiling because the generation instructions require opposing evidence and 'did not "
          "outweigh' language. No composite synthesis score is reported. Narrative-quality judgments "
          "from the two configured separate-model judges will remain a secondary analysis.\n",
        encoding="utf-8",
    )


def run(paired_path=DEFAULT_PAIRED, output_dir=DEFAULT_OUTPUT_DIR):
    paired = pd.read_csv(paired_path, keep_default_na=False)
    if len(paired) != 30 or paired["patient_id"].nunique() != 30:
        raise ValueError("Expected the 30-patient paired pilot dataset.")
    patient_results = compute_results(paired)
    summary = summarize(patient_results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_results.to_csv(output_dir / "patients.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    write_report(summary, output_dir / "report.md")
    return patient_results, summary


def main():
    parser = argparse.ArgumentParser(
        description="Run pairwise structural-synthesis evaluation."
    )
    parser.add_argument("--paired", default=str(DEFAULT_PAIRED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    run(args.paired, args.output_dir)


if __name__ == "__main__":
    main()
