"""Paired, blinded, reversed-order synthesis judging with two OpenRouter models."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score

from explanation.data_loader import (
    OPENROUTER_API_KEY,
    SYNTHESIS_JUDGE_MODELS,
    SYNTHESIS_JUDGE_REPETITIONS_PER_ORDER,
    SYNTHESIS_JUDGE_REVERSE_ORDER,
    SYNTHESIS_JUDGE_TEMPERATURE,
)
from .judge_runtime import (build_judge_client, collection_guard, extract_final_content,
                            panel_runtime_settings)

from ..synthesis_protocol import PROJECT_ROOT, PROTOCOL_DIR, canonical_json
from ..run_support import progress, output_lock
from .judge_collection import collect_judgments, collection_complete, verify_frozen_requests


DATASET_A = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/dataset_a/records.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/checker_v2_1/judge_part1_runtime_v3"
RUBRIC_VERSION = "paired-synthesis-v1"
PRIMARY_OUTCOME = "overall pairwise preference excluding ties"
SENSITIVITY_GATE = "both methods pass core structured fidelity and automated unsupported-information checks"
RANDOM_SEED = 20260903
RUBRIC = {
    "evidence_organization": "1=mostly a list or incoherent; 3=some meaningful organization; 5=clear, defensible higher-level organization.",
    "coherence": "1=fragmented/contradictory; 3=understandable with weak flow; 5=logically connected throughout.",
    "factor_traceability": "1=group membership is hidden/unclear; 3=partly traceable; 5=every synthesis claim clearly names its factors.",
    "contrastive_integration": "1=ignores/misuses opposing evidence; 3=mentions both sides; 5=clearly weighs both sides against the output.",
    "concision_nonredundancy": "1=verbose/repetitive; 3=minor excess; 5=concise with no material repetition.",
    "naturalness": "1=very awkward; 3=acceptable; 5=natural and grammatical. Do not treat naturalness as clinical usefulness.",
}


def build_judge_prompt(evidence, output_a, output_b):
    criteria = "\n".join(f"- {name}: {anchor}" for name, anchor in RUBRIC.items())
    score_shape = ", ".join(f'"{name}": 1' for name in RUBRIC)
    return f"""You are comparing two anonymous interpretations of exactly the same model evidence.
Method identity is intentionally hidden. Evaluate only what is stated. The supplied taxonomy is an
experimental reference, not clinical ground truth. Do not prefer an output merely because it sounds
more like an LLM. Use the anchored 1-5 criteria below.

{criteria}

EVIDENCE AND REFERENCE TAXONOMY:
{json.dumps(evidence, indent=2, ensure_ascii=False)}

OUTPUT A:
{output_a}

OUTPUT B:
{output_b}

Return only valid JSON with this exact shape:
{{"scores_a": {{{score_shape}}}, "scores_b": {{{score_shape}}},
  "preference": "A|B|tie", "rationale": "short criterion-based rationale"}}
"""


def _records(path=DATASET_A):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def prepare_blinded_items(records=None, seed=RANDOM_SEED):
    records = records or _records()
    by_patient = {}
    for record in records:
        # Poor compliance is an evaluation outcome, not grounds to discard a received report.
        if record["status"] in {"complete", "contract_failed"}:
            by_patient.setdefault(int(record["patient_id"]), {})[record["method"]] = record
    required = {"domain_template", "taxonomy_conditioned_llm"}
    complete = {pid: methods for pid, methods in by_patient.items() if required <= methods.keys()}
    rng = np.random.default_rng(seed)
    items, keys = [], []
    orders = ("primary", "reversed") if SYNTHESIS_JUDGE_REVERSE_ORDER else ("primary",)
    for patient_id, methods in sorted(complete.items()):
        primary_a = "taxonomy_conditioned_llm" if rng.integers(2) else "domain_template"
        for order in orders:
            method_a = primary_a if order == "primary" else (required - {primary_a}).pop()
            method_b = (required - {method_a}).pop()
            for repetition in range(1, SYNTHESIS_JUDGE_REPETITIONS_PER_ORDER + 1):
                for judge_model in SYNTHESIS_JUDGE_MODELS:
                    audit_id = f"J-{patient_id}-{order[0].upper()}-{repetition}-{hashlib.sha1(judge_model.encode()).hexdigest()[:6]}"
                    evidence = {key: methods[method_a][key] for key in
                                ("patient_id", "prediction", "stage", "evidence_hash", "selected_factors")}
                    prompt = build_judge_prompt(
                        evidence, methods[method_a]["interpretation"], methods[method_b]["interpretation"])
                    items.append({"audit_id": audit_id, "patient_id": patient_id,
                                  "judge_model": judge_model, "order": order,
                                  "repetition": repetition, "output_a": methods[method_a]["interpretation"],
                                  "output_b": methods[method_b]["interpretation"], "judge_prompt": prompt,
                                  "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()})
                    keys.append({"audit_id": audit_id, "patient_id": patient_id,
                                 "method_a": method_a, "method_b": method_b})
    return pd.DataFrame(items), pd.DataFrame(keys)


def parse_judgment(text):
    try:
        value = json.loads(str(text).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    if value.get("preference") not in {"A", "B", "tie"}:
        return None
    for side in ("scores_a", "scores_b"):
        if not isinstance(value.get(side), dict) or set(value[side]) != set(RUBRIC):
            return None
        if any(type(score) is not int or not 1 <= score <= 5 for score in value[side].values()):
            return None
    if not isinstance(value.get("rationale"), str) or not value["rationale"].strip():
        return None
    return value


def score_items(items, output_dir=OUTPUT_DIR, request_timeout=120, max_calls=None):
    return collect_judgments(items.to_dict("records"), output_dir,
        temperature=SYNTHESIS_JUDGE_TEMPERATURE,
        client_factory=lambda model: build_judge_client(model, SYNTHESIS_JUDGE_TEMPERATURE, request_timeout),
        extract=extract_final_content, parse=parse_judgment, label="Part 1 judges",
        max_calls=max_calls,
        runtime_config=lambda model: panel_runtime_settings(model, request_timeout))


def aggregate(raw_rows, key):
    parsed_rows, call_rows = [], []
    key_map = key.set_index("audit_id").to_dict("index")
    for row in raw_rows:
        if not row["parsed"]:
            continue
        identity = key_map[row["audit_id"]]
        value = row["parsed"]
        winner = "tie" if value["preference"] == "tie" else identity[f"method_{value['preference'].lower()}"]
        base = {"audit_id": row["audit_id"], "patient_id": identity["patient_id"],
                "judge_model": row["judge_model"], "winner": winner,
                "order": row.get("order"), "repetition": row.get("repetition")}
        call_rows.append(base)
        for letter, method in (("a", identity["method_a"]), ("b", identity["method_b"])):
            parsed_rows.append({**base, "method": method, **value[f"scores_{letter}"]})
    scores = pd.DataFrame(parsed_rows)
    calls = pd.DataFrame(call_rows)
    if scores.empty:
        return scores, pd.DataFrame(), {}
    criteria = list(RUBRIC)
    aggregates = scores.groupby(["patient_id", "judge_model", "method"])[criteria].mean().reset_index()

    votes = scores.drop_duplicates("audit_id").groupby(["patient_id", "judge_model"])["winner"].agg(list)
    winner_rows = []
    for (patient_id, judge), values in votes.items():
        counts = Counter(values)
        top = counts.most_common()
        winner = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else "tie"
        winner_rows.append({"patient_id": patient_id, "judge_model": judge, "winner": winner})
    winners = pd.DataFrame(winner_rows)

    patient_winners = []
    for patient_id, group in winners.groupby("patient_id"):
        values = group.winner.tolist()
        patient_winners.append({"patient_id": patient_id,
                                "winner": values[0] if len(set(values)) == 1 else "tie"})
    patient_winners = pd.DataFrame(patient_winners)
    non_ties = patient_winners[patient_winners.winner != "tie"]
    llm_wins = int((non_ties.winner == "taxonomy_conditioned_llm").sum())
    binomial = binomtest(llm_wins, len(non_ties), 0.5) if len(non_ties) else None
    summary = {"primary_outcome": PRIMARY_OUTCOME, "n_patients": len(patient_winners),
               "llm_wins": llm_wins,
               "template_wins": int((non_ties.winner == "domain_template").sum()),
               "ties": int((patient_winners.winner == "tie").sum()),
               "exact_binomial_pvalue": float(binomial.pvalue) if binomial else None}
    if calls["order"].notna().any():
        order_pairs = calls.pivot_table(index=["patient_id", "judge_model", "repetition"],
                                        columns="order", values="winner", aggfunc="first").dropna()
        summary["order_consistency"] = float(
            (order_pairs["primary"] == order_pairs["reversed"]).mean()
        ) if {"primary", "reversed"} <= set(order_pairs.columns) and len(order_pairs) else None
    repeat_consistency = []
    for _key, group in calls.groupby(["patient_id", "judge_model", "order"], dropna=False):
        repeat_consistency.append(group.winner.value_counts().iloc[0] / len(group))
    summary["within_judge_repeat_consistency"] = float(np.mean(repeat_consistency))
    if len(SYNTHESIS_JUDGE_MODELS) == 2 and winners["judge_model"].nunique() == 2:
        cross = winners.pivot(index="patient_id", columns="judge_model", values="winner").dropna()
        if not cross.empty:
            summary["cross_judge_raw_agreement"] = float((cross.iloc[:, 0] == cross.iloc[:, 1]).mean())
            summary["cross_judge_winner_kappa"] = float(cohen_kappa_score(cross.iloc[:, 0], cross.iloc[:, 1]))
        criterion_kappas = []
        score_wide = aggregates.pivot(index=["patient_id", "method"], columns="judge_model",
                                      values=list(RUBRIC)).dropna()
        if len(score_wide):
            for criterion in RUBRIC:
                left = score_wide[criterion].iloc[:, 0].round().astype(int)
                right = score_wide[criterion].iloc[:, 1].round().astype(int)
                criterion_kappas.append(cohen_kappa_score(left, right, weights="quadratic"))
        summary["cross_judge_weighted_criterion_kappa_mean"] = (
            float(np.nanmean(criterion_kappas)) if criterion_kappas else None
        )
    return scores, aggregates.merge(winners, on=["patient_id", "judge_model"]), summary


def add_faithfulness_sensitivity(summary, patient_aggregates, structural_dir=None):
    """Add the predefined both-method faithfulness gate without replacing the full-pair result."""
    structural_dir = Path(
        structural_dir
        or PROJECT_ROOT / "template_experiment/outputs/synthesis_v2/checker_v2_1/part1_structural"
    )
    fidelity_path = structural_dir / "verification_fidelity.csv"
    unsupported_path = structural_dir / "verification_unsupported_information.csv"
    if not fidelity_path.exists() or not unsupported_path.exists() or patient_aggregates.empty:
        return summary
    fidelity = pd.read_csv(fidelity_path)
    unsupported = pd.read_csv(unsupported_path)
    core = ["outcome_correct", "direction_accuracy", "value_fidelity", "scale_fidelity",
            "rank_order_fidelity"]
    fidelity["gate"] = fidelity[core].apply(
        lambda row: all(pd.notna(value) and np.isclose(float(value), 1.0) for value in row), axis=1)
    unsupported["gate"] = unsupported["automated_unsupported_information_free"].map(
        lambda value: str(value).strip().lower() == "true"
    )
    passed = fidelity[["patient_id", "method", "gate"]].merge(
        unsupported[["patient_id", "method", "gate"]], on=["patient_id", "method"],
        suffixes=("_fidelity", "_unsupported"))
    passed["gate"] = passed.gate_fidelity & passed.gate_unsupported
    eligible = passed.groupby("patient_id").gate.agg(lambda values: len(values) == 2 and values.all())
    eligible_ids = set(eligible[eligible].index)
    judge_winners = patient_aggregates[["patient_id", "judge_model", "winner"]].drop_duplicates()
    consensus = []
    for patient_id, group in judge_winners[judge_winners.patient_id.isin(eligible_ids)].groupby("patient_id"):
        values = group.winner.tolist()
        consensus.append(values[0] if len(set(values)) == 1 else "tie")
    non_ties = [winner for winner in consensus if winner != "tie"]
    llm_wins = sum(winner == "taxonomy_conditioned_llm" for winner in non_ties)
    summary.update({"sensitivity_gate": SENSITIVITY_GATE,
                    "sensitivity_n_patients": len(consensus),
                    "sensitivity_llm_wins": llm_wins,
                    "sensitivity_template_wins": sum(winner == "domain_template" for winner in non_ties),
                    "sensitivity_ties": sum(winner == "tie" for winner in consensus),
                    "sensitivity_exact_binomial_pvalue": (
                        float(binomtest(llm_wins, len(non_ties), 0.5).pvalue) if non_ties else None)})
    return summary


def run(records_path=DATASET_A, output_dir=OUTPUT_DIR, execute=False, request_timeout=120, max_calls=None):
    if request_timeout <= 0 or (max_calls is not None and max_calls < 0):
        raise ValueError("Timeout must be positive and max_calls nonnegative.")
    with collection_guard(output_dir, execute), output_lock(output_dir):
        return _run(records_path, Path(output_dir), execute, request_timeout, max_calls)


def _run(records_path, output_dir, execute, request_timeout, max_calls):
    records = _records(records_path)
    items, key = prepare_blinded_items(records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_profiles = {model: panel_runtime_settings(model, request_timeout)
                        for model in SYNTHESIS_JUDGE_MODELS}
    verify_frozen_requests(items.to_dict("records"), output_dir, SYNTHESIS_JUDGE_TEMPERATURE,
                           lambda model: panel_runtime_settings(model, request_timeout))
    (output_dir / "runtime_profiles.json").write_text(
        json.dumps(runtime_profiles, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    items.to_json(output_dir / "blinded_items.jsonl", orient="records", lines=True, force_ascii=False)
    key.to_csv(output_dir / "identity_key.csv", index=False)
    (output_dir / "judge_rubric.md").write_text(
        f"# Judge rubric ({RUBRIC_VERSION})\n\nPrimary outcome: {PRIMARY_OUTCOME}.\n\n" +
        "\n".join(f"- **{name}**: {anchor}" for name, anchor in RUBRIC.items()) +
        "\n\nSecondary model-based evidence only; not human or clinical validation.\n", encoding="utf-8")
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    (PROTOCOL_DIR / "judge_rubric.md").write_text(
        (output_dir / "judge_rubric.md").read_text(encoding="utf-8"), encoding="utf-8")
    (PROTOCOL_DIR / "judge_prompt.txt").write_text(
        "Paired prompt template is implemented by build_judge_prompt in "
        "template_experiment/experimental/llm_judge.py. Required JSON keys: scores_a, scores_b, "
        "preference, rationale. Method identity is excluded; both presentation orders are run.\n",
        encoding="utf-8")
    if not execute and (output_dir / "raw_judgments.jsonl").exists():
        progress("Existing Part 1 judge collection retained; no API calls made. See collection_status.md.")
        return items, key, None
    if not execute:
        expected = 30 * len(SYNTHESIS_JUDGE_MODELS) * SYNTHESIS_JUDGE_REPETITIONS_PER_ORDER * (2 if SYNTHESIS_JUDGE_REVERSE_ORDER else 1)
        progress(f"Part 1 judges: {len(items)}/{expected} calls prepared; no judge API calls made."
                 + (" Dataset A still has missing responses." if len(items) < expected else ""))
        (output_dir / "report.md").write_text(
            "# Part 1 paired judge evaluation\n\n"
            f"Prepared {len(items)} of {expected} expected blinded calls. "
            + ("Dataset A has missing responses. No judge results were fabricated. " if len(items) < expected else "All received reports are included, regardless of contract flags. ")
            + "Run this module with `--execute` after setting `OPENROUTER_API_KEY`. Raw responses and failures are retained.\n",
            encoding="utf-8")
        return items, key, None
    expected = 30 * len(SYNTHESIS_JUDGE_MODELS) * SYNTHESIS_JUDGE_REPETITIONS_PER_ORDER * (2 if SYNTHESIS_JUDGE_REVERSE_ORDER else 1)
    if len(items) != expected:
        raise ValueError(
            f"Expected {expected} complete blinded calls before judging; found {len(items)}. Complete Dataset A first."
        )
    raw = score_items(items, output_dir, request_timeout, max_calls)
    if not collection_complete(raw):
        (output_dir / "report.md").write_text(
            "# Part 1 paired judge evaluation\n\nCollection is partial or contains invalid structured "
            "responses. See collection_status.md and raw_judgments.jsonl. No final inference is "
            "computed from an incomplete judge panel. Resume missing requests with the same command; "
            "invalid responses need review, not automatic resampling.\n", encoding="utf-8")
        return items, key, None
    scores, patient_aggregates, summary = aggregate(raw, key)
    summary = add_faithfulness_sensitivity(summary, patient_aggregates, output_dir.parent / "part1_structural")
    scores.to_csv(output_dir / "parsed_scores.csv", index=False)
    patient_aggregates.to_csv(output_dir / "patient_aggregates.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "report.md").write_text(
        "# Part 1 paired judge evaluation\n\n" + json.dumps(summary, indent=2) +
        "\n\nCalls were aggregated within patient and judge before patient-level analysis.\n",
        encoding="utf-8")
    return items, key, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default=str(DATASET_A))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--max-calls", type=int, help="Maximum new API calls in this run; omit to resume all.")
    args = parser.parse_args()
    run(args.records, args.output_dir, args.execute, args.request_timeout, args.max_calls)


if __name__ == "__main__":
    main()
