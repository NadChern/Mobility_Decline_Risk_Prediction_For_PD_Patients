"""Shared, experiment-local evidence and report parsing utilities.

This module deliberately has no dependency on ``explanation.evaluation``. The template experiment
uses its own frozen metric implementation while continuing to use the production explanation
builder as the source of model-derived ground truth.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from explanation.explanation_builder import (
    build_patient_explanation_data_full,
    get_patient_index,
)


METHODS = {
    "gemini": ("gemini_model", "gemini_explanation"),
    "template": ("template_method", "template_explanation"),
    "template_grouped": ("template_grouped_method", "template_grouped_explanation"),
}


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
    "Lightheadedness on Standing": ["orthostatic lightheadedness", "lightheadedness on standing",
                                     "lightheaded"],
    "Fainting": ["fainting", "fainted", "faint"],
    "GDS-15": ["geriatric depression", "depressive", "depression", "gds-15", "gds"],
    "Mobility Summary": ["mobility summary"],
    "UPDRS-IV Total": ["mds-updrs part iv", "updrs part iv", "updrs-iv", "motor complications",
                        "part iv"],
    "UPDRS-I Total": ["non-motor experiences of daily living", "non-motor experiences",
                       "mds-updrs part i", "updrs part i", "updrs-i", "part i"],
    "BMI": ["body mass index", "bmi"],
    "Daytime Sleepiness": ["daytime sleepiness", "sleepiness"],
    "Urinary Problems": ["urinary problems", "urinary symptoms", "urinary"],
    "Gait": ["clinician-rated gait impairment", "clinician-rated gait", "gait impairment",
             ],
    "Postural Stability": ["clinician-rated postural stability", "severe postural instability",
                            "postural stability"],
    "H&Y Stage": ["hoehn and yahr", "hoehn & yahr", "hoehn", "h&y"],
    "Postural Hypotension": ["postural hypotension", "orthostatic hypotension"],
    "UPDRS-III Total": ["total motor examination score", "total motor examination",
                         "motor examination", "mds-updrs part iii", "updrs part iii",
                         "updrs-iii", "part iii"],
}

CAUSAL_PATTERNS = (
    r"\bcaus(?:e|es|ed|ing)\b",
    r"\bleads? to\b",
    r"\bresults? in\b",
    r"\bprotective factor\b",
    r"\brisk factor\b",
    r"\bincreases? (?:the )?(?:clinical )?risk\b",
    r"\breduces? (?:the )?(?:clinical )?risk\b",
)


def normalize_label(text):
    """Normalize cosmetic label variants while retaining one canonical vocabulary."""
    return str(text).strip().split(" (")[0].strip().lower()


def normalize_space(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def canonical_labels():
    feature_map_path = Path(__file__).resolve().parents[2] / "explanation" / "feature_map.csv"
    names = pd.read_csv(feature_map_path)["short_name"].tolist()
    mapping = {normalize_label(name): name for name in names}
    return set(mapping), mapping


def heading_text(line):
    return str(line).strip().lstrip("#").strip()


def direction_from_heading(line):
    """Map a factor-table heading to a stage/sign key."""
    text = heading_text(line).lower()
    if not text.startswith("factors pushing the prediction toward"):
        return None
    if "no fall" in text:
        return "s1_negative"
    if "higher severity" in text:
        return "s2_positive"
    if "lower severity" in text:
        return "s2_negative"
    if "toward fall" in text:
        return "s1_positive"
    return None


def split_markdown_row(line):
    """Split a Markdown row without treating an escaped pipe as a cell boundary."""
    stripped = str(line).strip().strip("|")
    return [cell.replace(r"\|", "|").strip() for cell in re.split(r"(?<!\\)\|", stripped)]


@dataclass(frozen=True)
class ReportRow:
    number: str
    raw_factor: str
    factor: str | None
    value: str
    scale: str
    direction: str | None
    line_number: int


def parse_report(text, norm_set=None, norm_to_short=None):
    """Parse table rows, headings, classification fields, sections, and interpretation prose."""
    if norm_set is None or norm_to_short is None:
        norm_set, norm_to_short = canonical_labels()

    lines = str(text).splitlines()
    current_direction = None
    directions = []
    rows = []
    for line_number, line in enumerate(lines, start=1):
        possible_direction = direction_from_heading(line)
        if possible_direction:
            current_direction = possible_direction
            directions.append(possible_direction)
            continue
        stripped = line.strip()
        if not stripped.startswith("|") or current_direction is None:
            continue
        cells = split_markdown_row(stripped)
        if len(cells) < 2:
            continue
        first = cells[0]
        if first == "#" or (first and set(first) <= set("-: ")):
            continue
        cells += [""] * max(0, 4 - len(cells))
        normalized = normalize_label(cells[1])
        factor = norm_to_short[normalized] if normalized in norm_set else None
        rows.append(ReportRow(first, cells[1], factor, cells[2], cells[3],
                              current_direction, line_number))

    fall_values = re.findall(
        r"(?mi)^[ \t]*(?:[-*][ \t]*)?Fall Classification[ \t]*:[ \t]*(\S.*?)[ \t]*$",
        str(text),
    )
    severity_values = re.findall(
        r"(?mi)^[ \t]*(?:[-*][ \t]*)?Fall Severity Classification[ \t]*:[ \t]*(\S.*?)[ \t]*$",
        str(text),
    )

    exact_headings = {heading_text(line): i for i, line in enumerate(lines)}
    interpretation = _between_headings(lines, "Model Interpretation", "Clinical Note")
    clinical_note = _after_heading(lines, "Clinical Note")
    prediction = _between_first_factor_or_interpretation(lines, "Model Prediction")
    title_lines = [line.strip() for line in lines if heading_text(line).startswith(
        "Clinical Fall Risk Summary")]

    return {
        "rows": rows,
        "directions": directions,
        "fall_values": [normalize_space(value) for value in fall_values],
        "severity_values": [normalize_space(value) for value in severity_values],
        "title_lines": title_lines,
        "has_prediction_heading": "Model Prediction" in exact_headings,
        "prediction_content": prediction,
        "has_interpretation_heading": "Model Interpretation" in exact_headings,
        "interpretation": interpretation,
        "has_clinical_note_heading": "Clinical Note" in exact_headings,
        "clinical_note": clinical_note,
        "text": str(text),
    }


def _between_headings(lines, start_heading, end_heading):
    start = next((i for i, line in enumerate(lines) if heading_text(line) == start_heading), None)
    if start is None:
        return ""
    end = next((i for i in range(start + 1, len(lines))
                if heading_text(lines[i]) == end_heading), len(lines))
    return "\n".join(lines[start + 1:end]).strip()


def _after_heading(lines, heading):
    start = next((i for i, line in enumerate(lines) if heading_text(line) == heading), None)
    return "" if start is None else "\n".join(lines[start + 1:]).strip()


def _between_first_factor_or_interpretation(lines, heading):
    start = next((i for i, line in enumerate(lines) if heading_text(line) == heading), None)
    if start is None:
        return ""
    end = next((i for i in range(start + 1, len(lines))
                if direction_from_heading(lines[i]) or heading_text(lines[i]) == "Model Interpretation"),
               len(lines))
    return "\n".join(lines[start + 1:end]).strip()


def expected_evidence(source_row, norm_to_short=None):
    """Build experiment ground truth for the displayed stage and both pipeline stages."""
    if norm_to_short is None:
        _, norm_to_short = canonical_labels()
    stage = int(source_row["stage"])
    patient_id = int(source_row["patient_id"])
    info = build_patient_explanation_data_full(get_patient_index(patient_id))

    expected_order = {}
    expected_shap = {}
    for column, shap_column, sign in (
        ("contributing", "contributing_shap", "positive"),
        ("mitigating", "mitigating_shap", "negative"),
    ):
        names = [name for name in str(source_row[column] or "").split("|") if name]
        shaps = [value for value in str(source_row[shap_column] or "").split("|") if value != ""]
        direction = f"s{stage}_{sign}"
        expected_order[direction] = []
        for index, name in enumerate(names):
            canonical = norm_to_short.get(normalize_label(name), name)
            expected_order[direction].append(canonical)
            if index < len(shaps):
                expected_shap[canonical] = abs(float(shaps[index]))

    if stage == 2:
        shown_factors = info["severity_increasing_features"] + info["severity_decreasing_features"]
    else:
        shown_factors = info["risk_increasing_features"] + info["risk_decreasing_features"]
    content = {
        factor["short_name"]: {
            "value": str(factor["display_value"]),
            "scale": str(factor["interpretation"]),
        }
        for factor in shown_factors if factor.get("displayable", True)
    }

    expected_direction = {
        factor: direction
        for direction, factors in expected_order.items()
        for factor in factors
    }

    # Prospective generation rows save the exact first-table/top-5 and second-table/top-3 lists.
    # Historical pilot rows predate these fields, so reconstruct the same selection deterministically
    # from the table order used by both the current prompt and grouped renderer.
    has_saved_roles = (
        "supporting_factors" in source_row.index
        and "opposing_factors" in source_row.index
    )
    if has_saved_roles:
        supporting_order = [
            norm_to_short.get(normalize_label(name), name)
            for name in str(source_row["supporting_factors"] or "").split("|") if name
        ]
        opposing_order = [
            norm_to_short.get(normalize_label(name), name)
            for name in str(source_row["opposing_factors"] or "").split("|") if name
        ]
        unknown_roles = (set(supporting_order) | set(opposing_order)) - set(expected_direction)
        if unknown_roles:
            raise ValueError(
                "Saved interpretation roles contain factors absent from the displayed evidence: "
                f"{sorted(unknown_roles)}"
            )
        overlap = set(supporting_order) & set(opposing_order)
        if overlap:
            raise ValueError(
                "Saved interpretation roles place factors on both sides: "
                f"{sorted(overlap)}"
            )
    else:
        supporting_sign = "positive" if str(source_row["category"]) == "moderate" else "negative"
        opposing_sign = "negative" if supporting_sign == "positive" else "positive"
        supporting_order = list(expected_order.get(f"s{stage}_{supporting_sign}", []))[:5]
        opposing_order = list(expected_order.get(f"s{stage}_{opposing_sign}", []))[:3]

    return {
        "patient_info": info,
        "stage": stage,
        "factor_set": set(expected_direction),
        "expected_direction": expected_direction,
        "expected_order": expected_order,
        "expected_shap": expected_shap,
        "content": content,
        "supporting_order": supporting_order,
        "opposing_order": opposing_order,
        "interpretation_roles_saved": has_saved_roles,
    }


def expected_outcomes(source_row):
    category = str(source_row["category"])
    stage = int(source_row["stage"])
    return {
        "fall": "No Fall" if stage == 1 else "Fall",
        "severity": None if stage == 1 else (
            "Rare Fall" if category == "mild" else "Recurrent Fall"
        ),
    }


def normalized_outcome(value):
    text = normalize_space(value).lower()
    aliases = {
        "no fall": "No Fall",
        "fall": "Fall",
        "mild falls": "Rare Fall",
        "mild fall": "Rare Fall",
        "rare fall": "Rare Fall",
        "moderate falls": "Recurrent Fall",
        "moderate fall": "Recurrent Fall",
        "recurrent fall": "Recurrent Fall",
    }
    return aliases.get(text, normalize_space(value))


def strict_outcome_correct(report, source_row):
    expected = expected_outcomes(source_row)
    fall = [normalized_outcome(value) for value in report["fall_values"]]
    severity = [normalized_outcome(value) for value in report["severity_values"]]
    fall_ok = fall == [expected["fall"]]
    severity_ok = severity == ([] if expected["severity"] is None else [expected["severity"]])
    return fall_ok and severity_ok


def expected_display_directions(stage):
    return {f"s{int(stage)}_positive", f"s{int(stage)}_negative"}


def detect_prose_features(prose):
    """Return frozen-lexicon feature mentions; this is traceability, not semantic claim review."""
    lexicon = []
    for feature, phrases in FEATURE_PROSE_SYNONYMS.items():
        for phrase in set(phrases + [feature]):
            lexicon.append((phrase.lower(), feature))
    lexicon.sort(key=lambda item: len(item[0]), reverse=True)
    text = str(prose).lower()
    occupied = []
    found = set()
    for phrase, feature in lexicon:
        for match in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
            if any(not (match.end() <= start or match.start() >= end)
                   for start, end in occupied):
                continue
            occupied.append((match.start(), match.end()))
            found.add(feature)
    return found


def causal_sentences(prose):
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", str(prose))
                 if sentence.strip()]
    flagged = [sentence for sentence in sentences
               if any(re.search(pattern, sentence, re.I) for pattern in CAUSAL_PATTERNS)]
    return sentences, flagged


def row_counter(report):
    return Counter(row.factor if row.factor is not None else f"__unknown__:{normalize_label(row.raw_factor)}"
                   for row in report["rows"])
