"""Faithfulness Evaluation — Section 2 of llm_eval_plan.md.

Standalone, no LLM calls. Checks whether the explanation is *grounded* in the model's SHAP
reasoning (a quality metric, distinct from stability). Computed once on the production output
(temperature=0, run=1) — one value per patient, then aggregated patient-wise → per category.

Two parts (per the agreed scope):

Part A — Tables (exact):
    Parse the two output tables (direction given by the header) and compare against the SHAP
    ground truth stored in llm_generations.csv:
      - Precision / Recall  — right features mentioned, none missing
      - Ghost rate          — table features not in the SHAP ground truth (hallucinated)
      - Direction accuracy  — each shared feature sits in the table matching its SHAP sign
      - Rank correlation    — table row order vs |SHAP| order (within each direction)
      - Faithfulness score  — F1 × direction accuracy

Part B — Model Interpretation prose (lexical mapping):
    The free-text synthesis uses clinical descriptions, not the exact short_name. A curated
    keyword→feature lexicon (built from feature_map.csv short_name + clinical phrasing) maps
    prose mentions to features, then:
      - prose_hallucination_rate — specific features named in prose but NOT in the patient's
                                    tables (the genuine hallucination risk)
      - prose_direction_acc      — features named as drivers sit in the prediction-supporting
                                    table; features named as opposing sit in the other table
                                    (clause split; approximate)

Usage:
    python -m explanation.evaluation.llm_faithfulness
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .llm_stability import (
    GENERATIONS_PATH, RESULTS_DIR, CATEGORIES,
    load_generations, load_canonical_labels, _normalize_label,
    extract_model_interpretation,
)
from ..explanation_builder import get_patient_index, build_patient_explanation_data_full

PATIENT_RESULTS_PATH = RESULTS_DIR / "llm_eval_faithfulness_patient.csv"
SUMMARY_PATH = RESULTS_DIR / "llm_eval_faithfulness_summary.csv"

TEMP_PRODUCTION = 0.0
RUN_PRODUCTION = 1

_OUTCOME_PHRASE = {"no_falls": "No Fall classification",
                   "mild": "Rare Fall classification",
                   "moderate": "Recurrent Fall classification"}


def _norm_ws(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def _is_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# -----------------------------------------------------------------------------
# Part B lexicon — curated clinical phrasing → canonical short_name
# -----------------------------------------------------------------------------
# Phrases the LLM uses in the free-text synthesis (not the exact short_name). Matched
# longest-first with word-boundary lookarounds; broad category words ("motor", "gait" alone)
# are deliberately excluded to avoid false hallucination hits.
FEATURE_PROSE_SYNONYMS = {
    "PD Duration": ["duration of parkinson", "disease duration", "pd duration", "longer disease duration"],
    "Age": ["advanced age", "patient's age", "age"],
    "Postural Instability at Dx": ["postural instability at the time of diagnosis",
                                   "postural instability at diagnosis", "postural instability at dx"],
    "Rigidity at Dx": ["rigidity at diagnosis", "rigidity at dx"],
    "Dopaminergic Therapy": ["dopaminergic therapy", "dopaminergic"],
    "MoCA": ["montreal cognitive assessment", "cognitive assessment", "cognitive performance",
             "cognitive score", "moca", "cognition"],
    "FOG (Freezing of Gait)": ["freezing of gait", "freezing", "fog"],
    "Lightheadedness on Standing": ["orthostatic lightheadedness", "lightheadedness on standing", "lightheaded"],
    "Fainting": ["fainting", "fainted", "faint"],
    "GDS-15": ["geriatric depression", "depressive", "depression", "gds-15", "gds"],
    "Mobility Summary": ["mobility summary"],
    "UPDRS-IV Total": ["mds-updrs part iv", "updrs part iv", "updrs-iv", "motor complications", "part iv"],
    # NOTE: bare "non-motor" / "non-motor symptoms" are category language (the prose uses them
    # as an umbrella then names specifics like cognition/depression), so they are NOT mapped to
    # UPDRS-I — only explicit references are.
    "UPDRS-I Total": ["non-motor experiences of daily living", "non-motor experiences",
                      "mds-updrs part i", "updrs part i", "updrs-i", "part i"],
    "BMI": ["body mass index", "bmi"],
    "Daytime Sleepiness": ["daytime sleepiness", "sleepiness"],
    "Urinary Problems": ["urinary problems", "urinary symptoms", "urinary"],
    "Gait": ["clinician-rated gait impairment", "clinician-rated gait", "gait impairment", "gait-related"],
    "Postural Stability": ["clinician-rated postural stability", "severe postural instability",
                           "postural stability"],
    "H&Y Stage": ["hoehn and yahr", "hoehn & yahr", "hoehn", "h&y"],
    "Postural Hypotension": ["postural hypotension", "orthostatic hypotension"],
    "UPDRS-III Total": ["total motor examination score", "total motor examination", "motor examination",
                        "mds-updrs part iii", "updrs part iii", "updrs-iii", "motor impairment", "part iii"],
}

# --- Scale-interpretation metadata (Part C) ---------------------------------------------
# Per-feature scale facts needed to check the prose's value descriptors against the actual
# value. `normal_level` = the value meaning none/normal (for negation checks); `max` + `higher`
# give severity position for directional checks. Features without scale semantics
# (Mobility Summary is non-monotonic; Age/BMI/PD Duration are demographic) are omitted → skipped.
SCALE_META = {
    "FOG (Freezing of Gait)":      {"normal_level": 0, "max": 4, "higher": "worse"},
    "Gait":                        {"normal_level": 0, "max": 4, "higher": "worse"},
    "Postural Stability":          {"normal_level": 0, "max": 4, "higher": "worse"},
    "Daytime Sleepiness":          {"normal_level": 0, "max": 4, "higher": "worse"},
    "Urinary Problems":            {"normal_level": 0, "max": 4, "higher": "worse"},
    "Lightheadedness on Standing": {"normal_level": 0, "max": 3, "higher": "worse"},
    "Fainting":                    {"normal_level": 0, "max": 3, "higher": "worse"},
    "H&Y Stage":                   {"normal_level": 0, "max": 5, "higher": "worse"},
    "UPDRS-I Total":               {"normal_level": 0, "max": 24, "higher": "worse"},
    "UPDRS-III Total":             {"normal_level": 0, "max": 132, "higher": "worse"},
    "UPDRS-IV Total":              {"normal_level": 0, "max": 24, "higher": "worse"},
    "GDS-15":                      {"normal_level": 0, "max": 15, "higher": "worse"},
    "MoCA":                        {"normal_level": None, "max": 30, "higher": "better"},
    # Categorical (negation check only — "absence of"/"no" ⇒ value 0):
    "Rigidity at Dx":              {"normal_level": 0, "max": None, "higher": None},
    "Postural Instability at Dx":  {"normal_level": 0, "max": None, "higher": None},
    "Postural Hypotension":        {"normal_level": 0, "max": None, "higher": None},
    "Dopaminergic Therapy":        {"normal_level": 0, "max": None, "higher": None},
}

# Value descriptors used in the prose, grouped by what they imply about the value.
_VALUE_DESCRIPTORS = {
    "NEG":  ["absence of", "without", "lack of", "unremarkable", "normal", "absent", "no"],
    "LOW":  ["lower", "low", "slight", "mild", "minor", "rare", "reduced", "minimal", "early"],
    "HIGH": ["higher", "high", "severe", "frequent", "significant", "marked", "advanced",
             "substantial", "considerable", "extensive", "elevated"],
}

# Cue phrases that mark a sentence as describing the *opposing* direction. The class-agnostic
# cues always apply; the severity cue is class-dependent (for a moderate patient "higher
# severity" is the SUPPORTING direction, so only "lower severity" marks opposing, and vice
# versa).
_OPPOSING_CUES_BASE = ["opposite direction", "while ", "conversely", "although"]


def _opposing_cues(category):
    cues = list(_OPPOSING_CUES_BASE)
    cues.append("lower severity" if category == "moderate" else "higher severity")
    if category != "moderate":
        cues.append("toward fall")
    return cues


def build_prose_lexicon():
    """Return [(phrase, short_name)] sorted longest-phrase-first for greedy matching."""
    lex = []
    for short_name, phrases in FEATURE_PROSE_SYNONYMS.items():
        for p in phrases:
            lex.append((p.lower(), short_name))
    lex.sort(key=lambda x: len(x[0]), reverse=True)
    return lex


def detect_prose_features(prose, lexicon):
    """Map a prose passage to the set of specific features it names (greedy, non-overlapping)."""
    text = prose.lower()
    claimed = []  # spans already attributed to a (longer) phrase
    found = set()
    for phrase, feat in lexicon:
        for m in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
            if any(not (m.end() <= s or m.start() >= e) for s, e in claimed):
                continue
            claimed.append((m.start(), m.end()))
            found.add(feat)
    return found


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------
def parse_tables_with_direction(text, norm_set, norm_to_short):
    """Return {'contributing': [...], 'mitigating': [...]} of canonical short_names in row order.

    Direction comes from the table header: 'Toward Fall' / 'Toward Higher Severity' = contributing
    (SHAP>0); 'Toward No Fall' / 'Toward Lower Severity' = mitigating (SHAP<0). Only the tables
    region (before 'Model Interpretation') is scanned.
    """
    mi = text.find("Model Interpretation")
    region = text[:mi] if mi >= 0 else text
    out = {"contributing": [], "mitigating": []}
    current = None
    for line in region.splitlines():
        s = line.strip()
        if "Toward" in s and ("Fall" in s or "Severity" in s) and not s.startswith("|"):
            if "No Fall" in s or "Lower Severity" in s:
                current = "mitigating"
            elif "Higher Severity" in s or "Toward Fall" in s:
                current = "contributing"
            continue
        if s.startswith("|") and current:
            cells = [c.strip() for c in s.strip("|").split("|")]
            if len(cells) < 2:
                continue
            factor = cells[1]
            if factor.lower() == "factor" or (factor and set(factor) <= set("-: ")) or not factor:
                continue
            key = _normalize_label(factor)
            if key in norm_set:
                out[current].append(norm_to_short[key])
    return out


def parse_table_values(text, norm_set, norm_to_short):
    """Return {short_name: float patient_value} parsed from the table rows."""
    mi = text.find("Model Interpretation")
    region = text[:mi] if mi >= 0 else text
    values = {}
    for line in region.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 3:
            continue
        factor, value = cells[1], cells[2]
        if factor.lower() == "factor" or (factor and set(factor) <= set("-: ")) or not factor:
            continue
        key = _normalize_label(factor)
        if key in norm_set:
            try:
                values[norm_to_short[key]] = float(value)
            except ValueError:
                pass
    return values


def parse_table_rows(text, norm_set, norm_to_short):
    """Return {canonical_short_name: {'value': str, 'interp': str}} from the table rows."""
    mi = text.find("Model Interpretation")
    region = text[:mi] if mi >= 0 else text
    rows = {}
    for line in region.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 4:
            continue
        factor, value, interp = cells[1], cells[2], cells[3]
        if factor.lower() == "factor" or (factor and set(factor) <= set("-: ")) or not factor:
            continue
        key = _normalize_label(factor)
        if key in norm_set:
            rows[norm_to_short[key]] = {"value": value, "interp": interp}
    return rows


def builder_ground_truth(patient_id, norm_set, norm_to_short):
    """Reconstruct the values/scales the LLM was given, from explanation_artifacts + feature_map.

    Uses build_patient_explanation_data_full (the same deterministic packet the prompt was built
    from). Returns {canonical_short_name: {'value': display_value, 'interp': interpretation}} for
    the displayable factors of the *shown* stage (Stage 1 for non-fallers, Stage 2 for fallers).
    """
    info = build_patient_explanation_data_full(get_patient_index(patient_id))
    if info["has_severity_assessment"]:
        factors = info["severity_increasing_features"] + info["severity_decreasing_features"]
    else:
        factors = info["risk_increasing_features"] + info["risk_decreasing_features"]
    gt = {}
    for f in factors:
        if not f.get("displayable", True):
            continue
        key = _normalize_label(f["short_name"])
        if key in norm_set:
            gt[norm_to_short[key]] = {"value": f["display_value"], "interp": f["interpretation"]}
    return gt


def table_content_fidelity(table_rows, gt):
    """Compare table value/scale cells against builder ground truth.

    Returns (value_fidelity, scale_fidelity, value_mismatches, scale_mismatches) over the
    factors present in both. Exact match after whitespace normalization.
    """
    shared = [f for f in table_rows if f in gt]
    if not shared:
        return np.nan, np.nan, [], []
    val_ok = scale_ok = 0
    val_bad, scale_bad = [], []
    for f in shared:
        if _norm_ws(table_rows[f]["value"]) == _norm_ws(gt[f]["value"]):
            val_ok += 1
        else:
            val_bad.append(f"{f}: '{table_rows[f]['value']}'≠'{gt[f]['value']}'")
        if _norm_ws(table_rows[f]["interp"]) == _norm_ws(gt[f]["interp"]):
            scale_ok += 1
        else:
            scale_bad.append(f)
    n = len(shared)
    return val_ok / n, scale_ok / n, val_bad, scale_bad


# -----------------------------------------------------------------------------
# Part C — scale-interpretation faithfulness
# -----------------------------------------------------------------------------
def _descriptor_bucket(window):
    """Return the value-descriptor bucket (NEG/LOW/HIGH) closest to the end of `window`."""
    best_pos, best_bucket = -1, None
    for bucket, words in _VALUE_DESCRIPTORS.items():
        for w in words:
            for m in re.finditer(rf"(?<!\w){re.escape(w)}(?!\w)", window):
                if m.start() > best_pos:
                    best_pos, best_bucket = m.start(), bucket
    return best_bucket


def _value_claims(prose, lexicon):
    """Return [(feature, bucket)] — features named in prose with an adjacent value descriptor."""
    text = prose.lower()
    spans, claims = [], []
    for phrase, feat in lexicon:  # longest-first
        for m in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
            if any(not (m.end() <= s or m.start() >= e) for s, e in spans):
                continue
            spans.append((m.start(), m.end()))
            # Descriptor must be in the same clause — don't let it cross a conjunction/list
            # boundary from a neighbouring feature ("normal postural stability and early H&Y").
            window = text[max(0, m.start() - 38):m.start()]
            seps = [" and ", ", ", "; ", " including ", " alongside ", " while ", " though "]
            cuts = [window.rfind(s) + len(s) for s in seps if window.rfind(s) != -1]
            if cuts:
                window = window[max(cuts):]
            bucket = _descriptor_bucket(window)
            if bucket:
                claims.append((feat, bucket))
    return claims


def _scale_consistent(bucket, value, meta):
    """True/False if checkable, None if not. Flags only CLEAR contradictions."""
    normal = meta.get("normal_level")
    mx, higher = meta.get("max"), meta.get("higher")
    if bucket == "NEG":
        return abs(value - normal) < 1e-9 if normal is not None else None
    if mx is None or higher is None:
        return None
    sf = (value / mx) if higher == "worse" else (1 - value / mx)  # severity fraction
    if bucket == "HIGH":
        return sf > 0.33   # contradiction only if value is clearly low
    if bucket == "LOW":
        return sf < 0.66   # contradiction only if value is clearly high
    return None


def scale_interpretation(prose, values, lexicon):
    """Return (n_claims, accuracy, flagged) for value-descriptor consistency in the prose."""
    n = consistent = 0
    flagged = []
    for feat, bucket in _value_claims(prose, lexicon):
        meta = SCALE_META.get(feat)
        if not meta or feat not in values:
            continue
        ok = _scale_consistent(bucket, values[feat], meta)
        if ok is None:
            continue
        n += 1
        if ok:
            consistent += 1
        else:
            flagged.append(f"{feat}={values[feat]:g}:{bucket}")
    return n, (consistent / n if n else np.nan), flagged


def _ground_truth(row, norm_set, norm_to_short):
    """Return (contributing_set, mitigating_set, {feature: |shap|}) from the CSV row."""
    def parse(names_col, shap_col):
        names = [n for n in str(row.get(names_col, "") or "").split("|") if n]
        shaps = [s for s in str(row.get(shap_col, "") or "").split("|") if s != ""]
        feats, mags = set(), {}
        for i, n in enumerate(names):
            key = _normalize_label(n)
            if key in norm_set:
                sn = norm_to_short[key]
                feats.add(sn)
                if i < len(shaps):
                    try:
                        mags[sn] = abs(float(shaps[i]))
                    except ValueError:
                        pass
        return feats, mags

    contrib, mc = parse("contributing", "contributing_shap")
    mitig, mm = parse("mitigating", "mitigating_shap")
    mags = {**mc, **mm}
    return contrib, mitig, mags


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def _f1(precision, recall):
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def _rank_corr(order_features, mags):
    """Spearman ρ between LLM row order and |SHAP| order, within a direction (n>=2)."""
    feats = [f for f in order_features if f in mags]
    if len(feats) < 2:
        return None
    from scipy.stats import spearmanr
    llm_rank = list(range(len(feats)))
    shap_vals = [mags[f] for f in feats]
    # higher |SHAP| should come first → rank by descending magnitude
    shap_rank = pd.Series(shap_vals).rank(ascending=False, method="first").tolist()
    rho = spearmanr(llm_rank, shap_rank).correlation
    return float(rho) if rho == rho else None  # guard NaN


def compute_patient_faithfulness(row, norm_set, norm_to_short, lexicon):
    """All Section-2 metrics for one patient (one production explanation)."""
    text = row["explanation"]
    category = row["category"]
    contrib_gt, mitig_gt, mags = _ground_truth(row, norm_set, norm_to_short)
    gt_all = contrib_gt | mitig_gt

    tables = parse_tables_with_direction(text, norm_set, norm_to_short)
    tbl_contrib = set(tables["contributing"])
    tbl_mitig = set(tables["mitigating"])
    mentioned = tbl_contrib | tbl_mitig

    # --- Part A: table-level ---
    tp = len(mentioned & gt_all)
    precision = tp / len(mentioned) if mentioned else 0.0
    recall = tp / len(gt_all) if gt_all else 0.0
    ghost_rate = len(mentioned - gt_all) / len(mentioned) if mentioned else 0.0

    shared = mentioned & gt_all
    correct_dir = sum(
        1 for f in shared
        if (f in tbl_contrib and f in contrib_gt) or (f in tbl_mitig and f in mitig_gt)
    )
    direction_acc = correct_dir / len(shared) if shared else 1.0

    rc_c = _rank_corr(tables["contributing"], mags)
    rc_m = _rank_corr(tables["mitigating"], mags)
    rcs = [r for r in (rc_c, rc_m) if r is not None]
    rank_corr = float(np.mean(rcs)) if rcs else np.nan

    faithfulness = _f1(precision, recall) * direction_acc

    # --- Part B: prose-level ---
    prose = extract_model_interpretation(text)
    prose_feats = detect_prose_features(prose, lexicon)
    prose_ghosts = prose_feats - gt_all
    prose_hallucination_rate = len(prose_ghosts) / len(prose_feats) if prose_feats else 0.0

    # Direction in prose: supporting side depends on the predicted class.
    supporting = contrib_gt if category == "moderate" else mitig_gt
    opposing = mitig_gt if category == "moderate" else contrib_gt
    prose_direction_acc = _prose_direction_accuracy(prose, lexicon, supporting, opposing, category)

    # --- Part A (content): table values & scales vs builder ground truth (artifacts + feature_map) ---
    gt_content = builder_ground_truth(row["patient_id"], norm_set, norm_to_short)
    table_rows = parse_table_rows(text, norm_set, norm_to_short)
    value_fidelity, scale_fidelity, value_mismatches, scale_mismatches = table_content_fidelity(
        table_rows, gt_content)
    outcome_correct = _OUTCOME_PHRASE[category].lower() in text.lower()

    # --- Part C: scale-interpretation faithfulness (prose value descriptors vs actual value) ---
    values = {f: float(v["value"]) for f, v in table_rows.items()
              if _is_float(v["value"])}
    scale_n, scale_acc, scale_flags = scale_interpretation(prose, values, lexicon)

    return {
        "patient_id": row["patient_id"],
        "category": category,
        "stage": int(row["stage"]),
        "model": row["model"],
        # Part A
        "precision": precision,
        "recall": recall,
        "ghost_rate": ghost_rate,
        "direction_acc": direction_acc,
        "rank_corr": rank_corr,
        "faithfulness_score": faithfulness,
        # Part B
        "prose_n_features": len(prose_feats),
        "prose_hallucination_rate": prose_hallucination_rate,
        "prose_ghost_features": "|".join(sorted(prose_ghosts)),
        "prose_direction_acc": prose_direction_acc,
        # Part A — table content fidelity (vs builder ground truth)
        "value_fidelity": value_fidelity,
        "scale_fidelity": scale_fidelity,
        "value_mismatches": "|".join(value_mismatches),
        "scale_mismatches": "|".join(scale_mismatches),
        "outcome_correct": outcome_correct,
        # Part C
        "scale_n_claims": scale_n,
        "scale_interpretation_acc": scale_acc,
        "scale_misinterpretations": "|".join(scale_flags),
    }


def _prose_direction_accuracy(prose, lexicon, supporting, opposing, category):
    """Per-sentence, class-aware direction check.

    Each sentence is classified 'opposing' if it contains a (class-aware) opposing cue, else
    'driver'. Two refinements avoid metric artifacts:
      - Sentence-level (not one index split): the prose names opposing features *before* the
        cue ("X and Y were associated with the opposite direction").
      - The clause after "outweigh" ("did not outweigh the … variables") refers to the
        SUPPORTING factors, so it is always scored as supporting regardless of the cue.
    """
    grounded = supporting | opposing
    cues = _opposing_cues(category)
    correct = total = 0
    for sent in re.split(r"(?<=[.!?])\s+", prose):
        low = sent.lower()
        cut = low.find("outweigh")
        head = sent if cut == -1 else sent[:cut]
        tail = "" if cut == -1 else sent[cut:]
        head_target = opposing if any(c in head.lower() for c in cues) else supporting
        for seg, target in [(head, head_target), (tail, supporting)]:
            for f in detect_prose_features(seg, lexicon) & grounded:
                total += 1
                correct += int(f in target)
    return correct / total if total else np.nan


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------
def aggregate_by_category(patient_df):
    metric_cols = ["precision", "recall", "ghost_rate", "direction_acc", "rank_corr",
                   "faithfulness_score", "value_fidelity", "scale_fidelity", "outcome_correct",
                   "prose_hallucination_rate", "prose_direction_acc",
                   "prose_n_features", "scale_interpretation_acc", "scale_n_claims"]
    rows = []
    for (model, category), grp in patient_df.groupby(["model", "category"]):
        row = {"model": model, "category": category, "n_patients": len(grp)}
        for c in metric_cols:
            vals = grp[c].dropna()
            row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{c}_std"] = float(vals.std(ddof=0)) if len(vals) else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["category"] = pd.Categorical(summary["category"], CATEGORIES, ordered=True)
        summary = summary.sort_values(["model", "category"]).reset_index(drop=True)
    return summary


def run_faithfulness_evaluation(generations_df):
    """Compute Section-2 metrics on the production output. Returns (patient_df, summary_df)."""
    norm_set, norm_to_short = load_canonical_labels()
    lexicon = build_prose_lexicon()

    prod = generations_df[
        np.isclose(generations_df["temperature"].astype(float), TEMP_PRODUCTION)
        & (generations_df["run"] == RUN_PRODUCTION)
    ]
    rows = [compute_patient_faithfulness(r, norm_set, norm_to_short, lexicon)
            for _, r in prod.iterrows()]
    patient_df = pd.DataFrame(rows)
    return patient_df, aggregate_by_category(patient_df)


def _print_summary(summary_df):
    cols = ["model", "category", "n_patients", "precision_mean", "recall_mean",
            "direction_acc_mean", "value_fidelity_mean", "scale_fidelity_mean",
            "outcome_correct_mean", "prose_hallucination_rate_mean", "prose_direction_acc_mean",
            "scale_interpretation_acc_mean"]
    cols = [c for c in cols if c in summary_df.columns]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("\nPer-category faithfulness summary:\n")
        print(summary_df[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Faithfulness Evaluation (Section 2).")
    parser.add_argument("--generations", default=str(GENERATIONS_PATH))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    df = load_generations(args.generations)
    patient_df, summary_df = run_faithfulness_evaluation(df)

    patient_path = out_dir / PATIENT_RESULTS_PATH.name
    summary_path = out_dir / SUMMARY_PATH.name
    patient_df.to_csv(patient_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved per-patient faithfulness → {patient_path}  ({len(patient_df)} rows)")
    print(f"Saved per-category summary     → {summary_path}  ({len(summary_df)} rows)")
    _print_summary(summary_df)

    # Surface any prose hallucinations for inspection
    halluc = patient_df[patient_df["prose_ghost_features"] != ""]
    if len(halluc):
        print("\n⚠ Prose features not found in tables (inspect):")
        for _, r in halluc.iterrows():
            print(f"  patient {r['patient_id']} ({r['category']}): {r['prose_ghost_features']}")
    else:
        print("\n✓ No prose hallucinations: every feature named in the synthesis is grounded in the tables.")

    vbad = patient_df[patient_df["value_mismatches"] != ""]
    sbad = patient_df[patient_df["scale_mismatches"] != ""]
    if len(vbad) or len(sbad):
        print("\n⚠ Table content mismatches vs ground truth (inspect):")
        for _, r in vbad.iterrows():
            print(f"  patient {r['patient_id']} VALUE: {r['value_mismatches']}")
        for _, r in sbad.iterrows():
            print(f"  patient {r['patient_id']} SCALE: {r['scale_mismatches']}")
    else:
        print("\n✓ Table content faithful: all values and scales match the builder ground truth.")
    if (~patient_df["outcome_correct"]).any():
        print("⚠ Wrong outcome stated:", patient_df[~patient_df["outcome_correct"]]["patient_id"].tolist())
    else:
        print("✓ Outcome correct: every explanation states the right predicted class.")

    mis = patient_df[patient_df["scale_misinterpretations"] != ""]
    if len(mis):
        print("\n⚠ Possible scale/value misinterpretations (inspect):")
        for _, r in mis.iterrows():
            print(f"  patient {r['patient_id']} ({r['category']}): {r['scale_misinterpretations']}")
    else:
        print("✓ No scale misinterpretations: prose value descriptors are consistent with the patient values.")


if __name__ == "__main__":
    main()
