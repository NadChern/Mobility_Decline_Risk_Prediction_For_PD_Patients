"""Dataset A structural comparison with patient-level paired bootstrap inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .synthesis import compute_results, summarize
from .completeness import compute_patient_completeness
from .fidelity import compute_patient_fidelity
from .structure import compute_patient_structure
from .unsupported_information import compute_patient_unsupported
from ..synthesis_protocol import PILOT_DATASET, PROJECT_ROOT
from ..run_support import progress


DATASET_A = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/dataset_a/records.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/checker_v2_1/part1_structural"
METHODS_A = {
    "domain_template": ("domain_model", "domain_explanation"),
    "taxonomy_conditioned_llm": ("taxonomy_llm_model", "taxonomy_llm_explanation"),
}
NONINFERIORITY_MARGIN = 0.05
BOOTSTRAP_SEED = 20260903
BOOTSTRAP_REPLICATES = 10000


def _read_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def build_paired_frame(records):
    frame = pd.DataFrame(records)
    # Received outputs with real errors still belong in the paired evaluation.
    # Contract success must not select an artificially easy subset of reports.
    complete = frame[frame["status"].isin(["complete", "contract_failed"])]
    wide = {}
    for method, prefix in (("domain_template", "domain"),
                           ("taxonomy_conditioned_llm", "taxonomy_llm")):
        subset = complete[complete["method"] == method].set_index("patient_id")
        wide[f"{prefix}_model"] = subset["model_or_template_version"]
        wide[f"{prefix}_explanation"] = subset["explanation"]
        wide[f"{prefix}_evidence_hash"] = subset["evidence_hash"]
    result = pd.DataFrame(wide).reset_index()
    pilot = pd.read_csv(PILOT_DATASET, keep_default_na=False)
    evidence_columns = [
        "patient_id", "category", "stage", "contributing", "contributing_shap",
        "mitigating", "mitigating_shap", "supporting_factors", "opposing_factors",
    ]
    return pilot[evidence_columns].merge(result, on="patient_id", how="left", validate="one_to_one")


def paired_bootstrap(patient_results, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    eligible = patient_results[patient_results["has_gold_grouping_opportunity"]]
    pivot = eligible.pivot(index="patient_id", columns="method", values="pairwise_grouping_f1")
    pivot = pivot.dropna(subset=list(METHODS_A))
    if pivot.empty:
        return None, pd.DataFrame()
    differences = (
        pivot["taxonomy_conditioned_llm"] - pivot["domain_template"]
    ).rename("f1_difference").reset_index()
    rng = np.random.default_rng(seed)
    strata = eligible.drop_duplicates("patient_id").set_index("patient_id")["category"]
    sampled_means = []
    arrays = [
        differences.set_index("patient_id").loc[strata[strata == category].index, "f1_difference"].to_numpy()
        for category in sorted(strata.unique())
    ]
    for _ in range(replicates):
        sample = np.concatenate([rng.choice(values, len(values), replace=True) for values in arrays if len(values)])
        sampled_means.append(float(sample.mean()))
    estimate = float(differences["f1_difference"].mean())
    lower, upper = np.quantile(sampled_means, [0.025, 0.975])
    stats = {
        "endpoint": "patient-level pairwise grouping F1 with grouping opportunity",
        "n_patients": len(differences),
        "mean_difference_llm_minus_template": estimate,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "noninferior": bool(lower > -NONINFERIORITY_MARGIN),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }
    return stats, differences


def run_contract_verification(paired, output_dir):
    """Run all retained report checks on the exact Dataset A pairs."""
    metric_functions = {
        "fidelity": compute_patient_fidelity,
        "completeness": compute_patient_completeness,
        "unsupported_information": compute_patient_unsupported,
        "structure": compute_patient_structure,
    }
    frames = {}
    for name, function in metric_functions.items():
        rows = []
        for _, source in paired.iterrows():
            for public_method, legacy_method, model_col, explanation_col in (
                ("domain_template", "template_grouped", "domain_model", "domain_explanation"),
                ("taxonomy_conditioned_llm", "gemini", "taxonomy_llm_model", "taxonomy_llm_explanation"),
            ):
                result = function(source, legacy_method, model_col, explanation_col)
                result["method"] = public_method
                rows.append(result)
        frames[name] = pd.DataFrame(rows).sort_values(["patient_id", "method"])
        frames[name].to_csv(output_dir / f"verification_{name}.csv", index=False)
    return frames


def run(records_path=DATASET_A, output_dir=OUTPUT_DIR):
    records = _read_records(records_path)
    paired = build_paired_frame(records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    uncertain = [r for r in records if any(i["kind"] == "extraction_uncertain"
                                         for i in r.get("checker_issues", []))]
    if uncertain:
        ids = sorted({r["patient_id"] for r in uncertain})
        progress(f"Part 1 structural analysis needs extraction review for patients {ids}; no reports excluded.")
        (output_dir / "report.md").write_text(
            "# Part 1 structural synthesis\n\n"
            f"Extraction requires review for patients {ids}. All received reports remain in the cohort; "
            "resolve extraction uncertainty before interpreting primary structural results. "
            "Recognized content errors are scored, not excluded. No regeneration is requested.\n",
            encoding="utf-8")
        return None
    if paired[["domain_explanation", "taxonomy_llm_explanation"]].isna().any(axis=None):
        llm_rows = [r for r in records if r["method"] == "taxonomy_conditioned_llm"]
        flagged = sum(r["status"] == "contract_failed" for r in llm_rows)
        missing = sum(r["status"] not in {"complete", "contract_failed"} for r in llm_rows)
        progress(f"Part 1 analysis blocked: {flagged} saved contract flags; "
                 f"{missing} missing/unprocessed responses. Review flags; do not regenerate them.")
        (output_dir / "report.md").write_text(
            "# Part 1 structural synthesis\n\nAnalysis is blocked under the current contract gate. "
            f"{flagged} saved responses have contract flags; {missing} requests are missing or need processing. "
            "Contract flags need review and are not API failures; --generate will not resample them. "
            "Resume missing requests with `python -m template_experiment.generate_dataset_a --generate`. "
            "No inferential result has been fabricated from incomplete pairs.\n",
            encoding="utf-8",
        )
        return None
    if not (paired["domain_evidence_hash"] == paired["taxonomy_llm_evidence_hash"]).all():
        raise ValueError("Paired methods do not share identical evidence hashes.")
    patients = compute_results(paired, METHODS_A)
    summary = summarize(patients)
    verification = run_contract_verification(paired, output_dir)
    stats, differences = paired_bootstrap(patients)
    patients.to_csv(output_dir / "patients.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    differences.to_csv(output_dir / "paired_differences.csv", index=False)
    pd.DataFrame([stats]).to_csv(output_dir / "noninferiority.csv", index=False)
    conclusion = (
        "The frozen noninferiority rule was met."
        if stats["noninferior"] else
        "The frozen noninferiority rule was not met; this does not establish inferiority."
    )
    overall = summary[summary["category"] == "overall"].set_index("method")
    template_f1 = overall.loc["domain_template", "pairwise_grouping_f1_macro_mean"]
    llm_f1 = overall.loc["taxonomy_conditioned_llm", "pairwise_grouping_f1_macro_mean"]
    decision_boundary = -NONINFERIORITY_MARGIN
    (output_dir / "report.md").write_text(
        "# Part 1 structural synthesis\n\n"
        "This report compares the domain template and taxonomy-conditioned LLM on the same "
        "patients and evidence. It does not use LLM-judge results.\n\n"
        "## Primary result\n\n"
        "| Result | Value |\n| --- | ---: |\n"
        f"| Template mean patient-level pairwise grouping F1 | {template_f1:.3f} |\n"
        f"| Taxonomy-conditioned LLM mean patient-level pairwise grouping F1 | {llm_f1:.3f} |\n"
        f"| Mean paired difference (LLM − template) | {stats['mean_difference_llm_minus_template']:.3f} |\n"
        f"| Stratified paired-bootstrap 95% CI | [{stats['ci95_lower']:.3f}, {stats['ci95_upper']:.3f}] |\n"
        f"| Eligible paired patients | {stats['n_patients']} |\n"
        f"| Bootstrap replicates | {stats['bootstrap_replicates']:,} |\n"
        f"| Bootstrap seed | {stats['bootstrap_seed']} |\n"
        f"| Prespecified noninferiority margin (Δ) | {NONINFERIORITY_MARGIN:.2f} |\n"
        f"| Decision boundary for LLM − template | {decision_boundary:.2f} |\n"
        f"| Noninferiority criterion met | {'Yes' if stats['noninferior'] else 'No'} |\n\n"
        f"Decision rule: the lower confidence bound must be greater than {decision_boundary:.2f}. "
        f"{conclusion}\n\n"
        "The primary endpoint includes patients with at least one reference grouping opportunity. "
        "Patients without an opportunity remain in the complete results and are checked for "
        "spurious grouping claims.\n\n"
        "## Reusable result files\n\n"
        "- `noninferiority.csv`: inference inputs, confidence interval, margin, and decision.\n"
        "- `paired_differences.csv`: one LLM-minus-template F1 difference per eligible patient.\n"
        "- `summary.csv`: macro and micro grouping metrics for both methods.\n"
        "- `patients.csv`: all patient-level claims, spans, pair sets, and metric values.\n"
        "- `verification_*.csv`: fidelity, completeness, unsupported-information, and structure checks.\n\n"
        "Taxonomy agreement is operational, not proof of clinical correctness. These are pilot "
        "results, not independent final-test validation.\n",
        encoding="utf-8",
    )
    return patients, summary, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default=str(DATASET_A))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run(args.records, args.output_dir)


if __name__ == "__main__":
    main()
