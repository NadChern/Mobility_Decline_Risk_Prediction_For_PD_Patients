"""Explanation Stability Evaluation — Section 1 of llm_eval_plan.md.

Standalone and runnable; performs **no LLM calls**. It reads the saved explanations from
``evaluation_results/llm_generations.csv`` and computes, per patient and then aggregated
per category (``no_falls`` / ``mild`` / ``moderate``):

§1.1 Semantic Stability
  - Determinism (temp=0): exact-match rate after whitespace normalization. If a patient is
    non-deterministic, a diagnostic cosine similarity is recorded.
  - Robustness (temp=0.3): pairwise cosine similarity over the 5 runs (C(5,2)=10 pairs),
    embedded with the local Ollama model ``qwen3-embedding:0.6b`` → mean_sim, std_sim, min_sim.

§1.2 Feature Consistency (Jaccard)
  - Coverage Jaccard: SHAP features sent in the prompt (stored in the CSV as short_name) vs.
    the features the LLM actually lists in its two output tables — per run, averaged.
  - Feature Stability: pairwise Jaccard of the mentioned-feature sets across the 5 runs.
  - Unknown-label rate: table "Factor" cells that match no feature_map short_name.

Robustness and Feature Consistency are reported at temp=0.3 (headline) and also computed at
temp=0 as a diagnostic (expected ≈1.0; a determinism cross-check).

Usage:
    python -m explanation.evaluation.llm_stability
    python -m explanation.evaluation.llm_stability --generations <path> --out-dir <dir>
"""

import argparse
import json
import os
import re
import urllib.request
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASE_DIR.parent.parent
RESULTS_DIR = _PROJECT_ROOT / "evaluation_results"
GENERATIONS_PATH = RESULTS_DIR / "llm_generations.csv"
FEATURE_MAP_PATH = _BASE_DIR.parent / "feature_map.csv"
PATIENT_RESULTS_PATH = RESULTS_DIR / "llm_eval_patient_results.csv"
SUMMARY_PATH = RESULTS_DIR / "llm_eval_summary.csv"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"  # local Ollama, 1024-dim
# Talk to the already-running Ollama server over HTTP — no Python client package needed.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"
CATEGORIES = ["no_falls", "mild", "moderate"]
TEMP_DETERMINISM = 0.0
TEMP_ROBUSTNESS = 0.3


# -----------------------------------------------------------------------------
# Text / embedding helpers
# -----------------------------------------------------------------------------
def normalize_for_comparison(text):
    """Normalize text for deterministic exact-match comparison (per §1.1.1)."""
    text = text.strip()
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text


def embed(texts):
    """Embed texts with the local Qwen3 embedding model via the Ollama HTTP API.

    Calls the already-running Ollama server's ``/api/embed`` endpoint directly (standard
    library only — no client package required). Symmetric similarity (explanations compared
    to each other, no query/document asymmetry), so every text is embedded identically — no
    retrieval instruction prefix.

    Returns:
        np.ndarray of shape (len(texts), 1024).
    """
    payload = json.dumps({"model": EMBEDDING_MODEL, "input": list(texts)}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except Exception as exc:  # connection refused, model not pulled, etc.
        raise RuntimeError(
            f"Ollama embedding request to {OLLAMA_HOST}/api/embed failed for model "
            f"'{EMBEDDING_MODEL}'. Ensure the Ollama server is running and the model is pulled "
            f"(`ollama pull {EMBEDDING_MODEL}`).\nOriginal error: {exc}"
        ) from exc

    embeddings = data.get("embeddings")
    if not embeddings:
        raise RuntimeError(f"Ollama returned no embeddings: {data}")
    return np.array(embeddings, dtype=float)


def _pairwise_cosine(outputs):
    """Return the C(n,2) pairwise cosine similarities between embedded outputs."""
    embeddings = embed(outputs)
    sims = []
    for i, j in combinations(range(len(outputs)), 2):
        sims.append(
            float(cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0, 0])
        )
    return np.array(sims)


# -----------------------------------------------------------------------------
# Feature extraction (§1.2)
# -----------------------------------------------------------------------------
def _normalize_label(text):
    """Light normalizer for cosmetic drift, e.g. 'FOG (Freezing of Gait)' -> 'FOG'."""
    return text.strip().split(" (")[0].strip().lower()


def load_canonical_labels(feature_map_path=FEATURE_MAP_PATH):
    """Return (norm_set, norm_to_short) from feature_map.csv short_name column.

    norm_set: set of normalized canonical labels.
    norm_to_short: normalized label -> canonical short_name (for de-duplicating mentions).
    """
    short_names = pd.read_csv(feature_map_path)["short_name"].tolist()
    norm_to_short = {}
    for sn in short_names:
        norm_to_short[_normalize_label(sn)] = sn
    return set(norm_to_short), norm_to_short


def extract_table_factors(text):
    """Return the raw 'Factor' cells (in reading order) from the two markdown tables.

    The output format is `| # | Factor | Patient Value | Interpretation / Scale |`. We read the
    second cell of each data row, skipping header and separator rows.
    """
    factors = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        factor = cells[1]
        if factor.lower() == "factor":          # header row
            continue
        if factor and set(factor) <= set("-: "):  # separator row (---, :--, etc.)
            continue
        if not factor:
            continue
        factors.append(factor)
    return factors


def extract_features(text, norm_set, norm_to_short):
    """Parse one explanation → (mentioned_set, n_unknown, n_cells).

    mentioned_set: canonical short_names matched in the tables.
    n_unknown: table Factor cells matching no known short_name.
    n_cells: total table Factor cells parsed.
    """
    cells = extract_table_factors(text)
    mentioned, unknown = set(), 0
    for cell in cells:
        key = _normalize_label(cell)
        if key in norm_set:
            mentioned.add(norm_to_short[key])
        else:
            unknown += 1
    return mentioned, unknown, len(cells)


def extract_model_interpretation(text):
    """Return the free-text 'Model Interpretation' paragraph (between that header and
    'Clinical Note'). This is the only section the LLM composes freely rather than copies,
    so cosine over it is a sharper, less-templated robustness signal."""
    m = re.search(r"Model Interpretation\s*(.*?)\s*Clinical Note", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    idx = text.find("Model Interpretation")
    return text[idx + len("Model Interpretation"):].strip() if idx >= 0 else ""


def jaccard(a, b):
    """Jaccard similarity of two sets. Two empty sets are defined as identical (1.0)."""
    a, b = set(a), set(b)
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _lcs_length(a, b):
    """Length of the longest common subsequence of two token lists (space-optimised DP)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l(a_text, b_text):
    """ROUGE-L F1: word-level lexical overlap based on the longest common subsequence.

    Returns 2PR/(P+R) where P = LCS/len(b), R = LCS/len(a) over whitespace tokens.
    1.0 = identical word sequence, 0.0 = no words shared. A graded companion to the
    binary exact-match check at temperature 0.
    """
    a, b = a_text.lower().split(), b_text.lower().split()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    lcs = _lcs_length(a, b)
    if lcs == 0:
        return 0.0
    p, r = lcs / len(b), lcs / len(a)
    return 2 * p * r / (p + r)


def _pairwise_rouge_l(outputs):
    """Return (mean, std, min) of the C(n,2) pairwise ROUGE-L F1 scores over a list of texts."""
    pairs = [rouge_l(a, b) for a, b in combinations(outputs, 2)]
    if not pairs:
        return np.nan, np.nan, np.nan
    return float(np.mean(pairs)), float(np.std(pairs)), float(np.min(pairs))


# -----------------------------------------------------------------------------
# Per-patient metrics
# -----------------------------------------------------------------------------
def _ground_truth_features(row, norm_set, norm_to_short):
    """SHAP ground-truth feature set for a patient (contributing ∪ mitigating short_names)."""
    gt = set()
    for col in ("contributing", "mitigating"):
        val = str(row.get(col, "") or "")
        for name in val.split("|"):
            key = _normalize_label(name)
            if name and key in norm_set:
                gt.add(norm_to_short[key])
    return gt


def _feature_consistency(outputs, ground_truth, norm_set, norm_to_short):
    """Coverage Jaccard, Feature Stability, unknown-label rate and count stats for one patient."""
    mentioned_sets, coverage, unknown_total, cells_total, counts = [], [], 0, 0, []
    for text in outputs:
        mentioned, n_unknown, n_cells = extract_features(text, norm_set, norm_to_short)
        mentioned_sets.append(mentioned)
        coverage.append(jaccard(ground_truth, mentioned))
        unknown_total += n_unknown
        cells_total += n_cells
        counts.append(len(mentioned))

    stability_pairs = [jaccard(a, b) for a, b in combinations(mentioned_sets, 2)]

    return {
        "coverage_jaccard": float(np.mean(coverage)),
        "feature_stability": float(np.mean(stability_pairs)) if stability_pairs else 1.0,
        "unknown_label_rate": (unknown_total / cells_total) if cells_total else 0.0,
        "feature_count_std": float(np.std(counts)),
    }


def _ordered_outputs(df, patient_id, model, temperature):
    """Return the explanation strings for one (patient, model, temperature), sorted by run."""
    rows = df[
        (df["patient_id"] == patient_id)
        & (df["model"] == model)
        & (np.isclose(df["temperature"].astype(float), temperature))
    ].sort_values("run")
    return rows["explanation"].tolist(), rows


def compute_patient_stability(df, patient_id, model, norm_set, norm_to_short):
    """Compute all Section-1 metrics for a single (patient, model)."""
    det_outputs, det_rows = _ordered_outputs(df, patient_id, model, TEMP_DETERMINISM)
    rob_outputs, rob_rows = _ordered_outputs(df, patient_id, model, TEMP_ROBUSTNESS)

    meta = det_rows.iloc[0] if len(det_rows) else rob_rows.iloc[0]
    result = {
        "patient_id": patient_id,
        "category": meta["category"],
        "stage": int(meta["stage"]),
        "model": model,
    }

    # --- §1.1.1 Determinism (temp=0) ---
    normalized = [normalize_for_comparison(o) for o in det_outputs]
    n_unique = len(set(normalized))
    result["deterministic"] = (n_unique == 1)
    result["n_unique_outputs"] = n_unique
    result["diagnostic_sim"] = (
        float(np.mean(_pairwise_cosine(det_outputs))) if (n_unique > 1 and len(det_outputs) > 1) else np.nan
    )
    # Lexical overlap (temp=0): graded companion to the binary exact-match. ROUGE-L F1 over
    # the LCS of words between each pair of the normalized deterministic outputs, so a few-word
    # difference reads as ~0.99 overlap rather than collapsing the patient to "not identical".
    if len(normalized) > 1:
        rl_mean, rl_std, rl_min = _pairwise_rouge_l(normalized)
        result["rougeL_t0_mean"] = rl_mean
        result["rougeL_t0_std"] = rl_std
        result["rougeL_t0_min"] = rl_min
    else:
        result["rougeL_t0_mean"] = result["rougeL_t0_std"] = result["rougeL_t0_min"] = np.nan

    # --- §1.1.2 Robustness (temp=0.3, headline) ---
    if len(rob_outputs) > 1:
        sims = _pairwise_cosine(rob_outputs)
        result["mean_sim"] = float(np.mean(sims))
        result["std_sim"] = float(np.std(sims))
        result["min_sim"] = float(np.min(sims))
    else:
        result["mean_sim"] = result["std_sim"] = result["min_sim"] = np.nan

    # --- Complementary: Model-Interpretation-only robustness (temp=0.3) ---
    # Sharper signal: cosine over just the free-text synthesis paragraph (not the templated
    # tables), showing semantic stability holds even where the model composes freely.
    rob_mi = [extract_model_interpretation(o) for o in rob_outputs]
    if len(rob_mi) > 1 and all(rob_mi):
        mi_sims = _pairwise_cosine(rob_mi)
        result["mi_mean_sim"] = float(np.mean(mi_sims))
        result["mi_std_sim"] = float(np.std(mi_sims))
        result["mi_min_sim"] = float(np.min(mi_sims))
    else:
        result["mi_mean_sim"] = result["mi_std_sim"] = result["mi_min_sim"] = np.nan

    # --- §1.2 Feature Consistency (temp=0.3, headline) ---
    ground_truth = _ground_truth_features(meta, norm_set, norm_to_short)
    if rob_outputs:
        fc = _feature_consistency(rob_outputs, ground_truth, norm_set, norm_to_short)
        result.update(fc)

    # --- Diagnostics (temp=0): expected ≈1.0, determinism cross-check ---
    if len(det_outputs) > 1:
        result["mean_sim_t0"] = float(np.mean(_pairwise_cosine(det_outputs)))
        fc0 = _feature_consistency(det_outputs, ground_truth, norm_set, norm_to_short)
        result["coverage_jaccard_t0"] = fc0["coverage_jaccard"]
        result["feature_stability_t0"] = fc0["feature_stability"]

    return result


# -----------------------------------------------------------------------------
# Aggregation (patient-wise → per category)
# -----------------------------------------------------------------------------
def aggregate_by_category(patient_df):
    """Aggregate per-patient metrics per (model, category) — patient-wise → mean ± std."""
    rows = []
    for (model, category), grp in patient_df.groupby(["model", "category"]):
        means = grp["mean_sim"].dropna()
        mins = grp["min_sim"].dropna()
        row = {
            "model": model,
            "category": category,
            "n_patients": len(grp),
            # §1.1.1 Determinism
            "exact_match_rate": float(grp["deterministic"].mean()),
            # §1.1.1 Lexical overlap (temp=0): graded determinism companion to exact-match
            "rougeL_t0_mean": float(grp["rougeL_t0_mean"].dropna().mean()) if grp.get("rougeL_t0_mean") is not None and grp["rougeL_t0_mean"].notna().any() else np.nan,
            "rougeL_t0_std": float(grp["rougeL_t0_mean"].dropna().std(ddof=0)) if grp.get("rougeL_t0_mean") is not None and grp["rougeL_t0_mean"].notna().any() else np.nan,
            "rougeL_t0_worst": float(grp["rougeL_t0_min"].dropna().min()) if grp.get("rougeL_t0_min") is not None and grp["rougeL_t0_min"].notna().any() else np.nan,
            # §1.1.2 Robustness (temp=0.3)
            "mean_sim_mean": float(means.mean()) if len(means) else np.nan,
            "mean_sim_std": float(means.std(ddof=0)) if len(means) else np.nan,
            "worst_case_mean": float(mins.mean()) if len(mins) else np.nan,
            "worst_case_min": float(mins.min()) if len(mins) else np.nan,
            # Complementary: Model-Interpretation-only robustness (temp=0.3)
            "mi_mean_sim_mean": float(grp["mi_mean_sim"].dropna().mean()) if grp["mi_mean_sim"].notna().any() else np.nan,
            "mi_mean_sim_std": float(grp["mi_mean_sim"].dropna().std(ddof=0)) if grp["mi_mean_sim"].notna().any() else np.nan,
            "mi_worst_case_min": float(grp["mi_min_sim"].dropna().min()) if grp["mi_min_sim"].notna().any() else np.nan,
            # §1.2 Feature Consistency (temp=0.3)
            "coverage_jaccard_mean": float(grp["coverage_jaccard"].mean()),
            "coverage_jaccard_std": float(grp["coverage_jaccard"].std(ddof=0)),
            "feature_stability_mean": float(grp["feature_stability"].mean()),
            "feature_stability_std": float(grp["feature_stability"].std(ddof=0)),
            "unknown_label_rate_mean": float(grp["unknown_label_rate"].mean()),
            # Diagnostics (temp=0)
            "mean_sim_t0_mean": float(grp["mean_sim_t0"].mean()) if "mean_sim_t0" in grp else np.nan,
            "coverage_jaccard_t0_mean": float(grp["coverage_jaccard_t0"].mean()) if "coverage_jaccard_t0" in grp else np.nan,
            "feature_stability_t0_mean": float(grp["feature_stability_t0"].mean()) if "feature_stability_t0" in grp else np.nan,
        }
        # Percentiles of the patient mean_sim distribution within the category
        if len(means):
            for p in (10, 25, 50, 75, 90):
                row[f"mean_sim_p{p}"] = float(np.percentile(means, p))
        rows.append(row)

    # Order categories sensibly
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["category"] = pd.Categorical(summary["category"], CATEGORIES, ordered=True)
        summary = summary.sort_values(["model", "category"]).reset_index(drop=True)
    return summary


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def load_generations(path=GENERATIONS_PATH):
    """Load llm_generations.csv, preserving empty strings in feature columns."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Generations file not found: {path}\n"
            "Generate it first:  python -m explanation.evaluation.llm_eval_runner --runs 5 --temperature 0 0.3"
        )
    return pd.read_csv(path, keep_default_na=False)


def run_stability_evaluation(generations_df):
    """Compute Section-1 metrics. Returns (patient_results_df, summary_df)."""
    norm_set, norm_to_short = load_canonical_labels()

    patient_rows = []
    pairs = generations_df[["patient_id", "model"]].drop_duplicates().itertuples(index=False)
    for patient_id, model in pairs:
        patient_rows.append(
            compute_patient_stability(generations_df, patient_id, model, norm_set, norm_to_short)
        )

    patient_df = pd.DataFrame(patient_rows)
    summary_df = aggregate_by_category(patient_df)
    return patient_df, summary_df


def _print_summary(summary_df):
    cols = [
        "model", "category", "n_patients", "exact_match_rate",
        "rougeL_t0_mean", "rougeL_t0_std", "rougeL_t0_worst",
        "mean_sim_mean", "mean_sim_std", "worst_case_min",
        "mi_mean_sim_mean", "mi_mean_sim_std", "mi_worst_case_min",
        "coverage_jaccard_mean", "feature_stability_mean", "unknown_label_rate_mean",
    ]
    cols = [c for c in cols if c in summary_df.columns]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\nPer-category stability summary:\n")
        print(summary_df[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Explanation Stability Evaluation (Section 1).")
    parser.add_argument("--generations", default=str(GENERATIONS_PATH), help="Path to llm_generations.csv")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR), help="Directory for output CSVs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    generations_df = load_generations(args.generations)
    print(
        f"Loaded {len(generations_df)} generation rows | "
        f"patients={generations_df['patient_id'].nunique()} | "
        f"models={list(generations_df['model'].unique())} | "
        f"temps={sorted(generations_df['temperature'].astype(float).unique())}"
    )

    patient_df, summary_df = run_stability_evaluation(generations_df)

    patient_path = out_dir / PATIENT_RESULTS_PATH.name
    summary_path = out_dir / SUMMARY_PATH.name
    patient_df.to_csv(patient_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved per-patient metrics → {patient_path}  ({len(patient_df)} rows)")
    print(f"Saved per-category summary → {summary_path}  ({len(summary_df)} rows)")
    _print_summary(summary_df)


if __name__ == "__main__":
    main()
