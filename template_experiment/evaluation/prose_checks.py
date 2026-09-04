"""Automatic interpretation-prose fidelity for the Model Interpretation paragraph.

The five table/label fidelity metrics cover the structured parts of the report. This module adds a
*conservative, human-free* panel for the free-text synthesis, checking only mechanically verifiable
claim types against the model evidence:

  - prose_direction_fidelity        — each named factor, or factor deterministically represented by
                                       a grouped-template label, is discussed on the side
                                       (supporting vs. opposing) that matches its SHAP table.
  - prose_grouping_fidelity         — mean of two objectively verifiable grouping constraints:
                                       selected-factor precision and minimum-size compliance.
  - taxonomy grouping recall / precision — descriptive agreement with the experiment's authored
                                       taxonomy, not unique clinical correctness.
  - prose_value_descriptor_fidelity — a value word ("severe", "mild", "absence of") next to a factor
                                       is consistent with that factor's actual value.

Feature *hallucination* (naming a factor absent from the tables) and *causal language* are precision
concerns and stay in the unsupported-information module, not here, so nothing is double-counted.
Every metric is conditional: NaN when there is nothing checkable, with the evaluable denominator
reported by the caller.

The taxonomy is an AUTHORED artifact. It must be frozen and version-controlled before final results
(see ``TAXONOMY_VERSION``), and its scores are reported only as agreement with that operational map.
"""

import re

import numpy as np

from ..taxonomy import CATEGORY_LABELS, CATEGORY_PHRASES, CATEGORY_TO_FEATURES, FEATURE_CATEGORY
from .shared import FEATURE_PROSE_SYNONYMS, detect_prose_features

_VALUE_DESCRIPTORS = {
    "NEG": [
        "absence of", "without", "lack of", "unremarkable", "normal", "absent",
        "none", "never", "no",
    ],
    "LOW": ["lower", "low", "slight", "mild", "minor", "rare", "reduced", "minimal", "early"],
    "HIGH": ["higher", "high", "severe", "frequent", "significant", "marked", "advanced",
             "substantial", "considerable", "extensive", "elevated"],
}

_OPPOSING_PATTERNS_BASE = (
    r"\bopposite[-\s]+direction\b",
    r"\bconversely\b",
    r"\bcounterbalanced by\b",
)
_CONTRAST_MARKER = re.compile(r"\b(?:while|although|conversely)\b", re.I)
_EXPLICIT_SUPPORTING_CLAUSE = re.compile(
    r"\b(?:primarily\s+(?:supported|associated)|supported\s+the\s+(?:prediction|classification))\b",
    re.I,
)
_ACTIVE_OUTWEIGH = re.compile(r"\bdid\s+not\s+outweigh\b", re.I)

# Explicit pilot-reviewed aliases, not fuzzy matching. Diagnosis-time variables remain
# distinct from current examination items. Matching preserves source offsets.
REVIEWED_FACTOR_ALIASES = {
    "Postural Instability at Dx": ("postural instability present at diagnosis",),
    "Rigidity at Dx": ("rigidity present at diagnosis",),
    "Urinary Problems": ("normal urinary function", "urine control problems"),
}

# These older broad prose synonyms describe domains/symptoms, not a uniquely named
# instrument or variable. They may not stand in for explicit factor membership.
AMBIGUOUS_FACTOR_ALIASES = {
    "MoCA": {"cognitive assessment", "cognitive performance", "cognitive score", "cognition"},
    "GDS-15": {"geriatric depression", "depressive", "depression", "gds"},
    "UPDRS-IV Total": {"motor complications", "part iv"},
    "UPDRS-III Total": {"motor examination", "part iii"},
    "UPDRS-I Total": {"non-motor experiences", "part i"},
    "FOG (Freezing of Gait)": {"freezing"},
    "H&Y Stage": {"hoehn"},
}


# --- Lexicons ---------------------------------------------------------------------------------
def _feature_lexicon():
    lex = [(phrase.lower(), feature)
           for feature, phrases in FEATURE_PROSE_SYNONYMS.items() for phrase in phrases
           if phrase.lower() not in AMBIGUOUS_FACTOR_ALIASES.get(feature, set())]
    lex.extend((phrase.lower(), feature)
               for feature, phrases in REVIEWED_FACTOR_ALIASES.items() for phrase in phrases)
    # Also match the exact factor display names (and their pre-parenthesis base), which the template
    # prints verbatim, so a plain factor listing is claimed as features and never misread as a group.
    for feature in FEATURE_CATEGORY:
        lex.append((feature.lower(), feature))
        base = feature.split(" (")[0].strip().lower()
        if base != feature.lower():
            lex.append((base, feature))
    lex = list(dict.fromkeys(lex))
    lex.sort(key=lambda item: len(item[0]), reverse=True)
    return lex


def _category_lexicon():
    lex = [(phrase.lower(), category)
           for category, phrases in CATEGORY_PHRASES.items() for phrase in phrases]
    # The display label a renderer emits must always be detectable as its own group.
    for category, display in CATEGORY_LABELS.items():
        lex.append((display.lower(), category))
    lex = list(dict.fromkeys(lex))
    lex.sort(key=lambda item: len(item[0]), reverse=True)
    return lex


_FEATURE_LEXICON = _feature_lexicon()
_CATEGORY_LEXICON = _category_lexicon()

# One combined lexicon, longest-phrase-first, each entry tagged feature/group. A longer group phrase
# ("gait and mobility") claims its whole span before the shorter feature word ("gait") inside it; a
# bare factor name ties on length with its category word and, because features are listed first,
# resolves to the feature. This is what keeps a group label from being misread as a feature and a
# plain factor listing from being misread as a group.
_COMBINED_LEXICON = sorted(
    [(phrase, "feature", label) for phrase, label in _FEATURE_LEXICON]
    + [(phrase, "group", label) for phrase, label in _CATEGORY_LEXICON],
    key=lambda item: len(item[0]), reverse=True,
)


def _lexicon_pattern(phrase, kind):
    prefix = r"(?<!\w)"
    suffix = r"(?!\w)"
    if kind == "group" and phrase.startswith("motor "):
        # Do not extract "motor burden" from the distinct phrase "non-motor burden".
        prefix = r"(?<!non-)(?<!\w)"
    if kind == "group" and phrase == "gait and mobility":
        # In "Gait and Mobility Summary", both capitalized strings are factor names, not a label.
        suffix = r"(?!\s+summary\b)(?!\w)"
    # Treat typography and runs of whitespace equivalently without rewriting the source
    # string: extraction spans must still point to the exact original text.
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\-", r"[-‐‑–—]")
    return rf"{prefix}{escaped}{suffix}"


def _detect_segment(segment):
    """Return (features, groups) for a segment via one combined longest-first, non-overlapping pass.

    A plain factor listing ("Mobility Summary", "Gait") resolves to features; a genuine multi-factor
    group phrase ("gait and mobility") resolves to a group and consumes the feature word inside it.
    """
    text = str(segment).lower()
    occupied, features, groups = [], set(), set()
    for phrase, kind, label in _COMBINED_LEXICON:
        for match in re.finditer(_lexicon_pattern(phrase, kind), text):
            if any(not (match.end() <= start or match.start() >= end)
                   for start, end in occupied):
                continue
            occupied.append((match.start(), match.end()))
            (features if kind == "feature" else groups).add(label)
    return features, groups


def _detect_segment_spans(segment):
    """Return non-overlapping feature/group mentions with character spans."""
    text = str(segment).lower()
    occupied, features, groups = [], [], []
    for phrase, kind, label in _COMBINED_LEXICON:
        for match in re.finditer(_lexicon_pattern(phrase, kind), text):
            if any(not (match.end() <= start or match.start() >= end)
                   for start, end in occupied):
                continue
            occupied.append((match.start(), match.end()))
            target = features if kind == "feature" else groups
            target.append((match.start(), match.end(), label, phrase))
    return sorted(features), sorted(groups)


def detect_prose_groups(prose):
    """Return the set of genuine group phrases used (feature mentions excluded)."""
    return _detect_segment(prose)[1]


# --- Direction / grouping (per-sentence, class-aware) -----------------------------------------
def _opposing_patterns(category):
    """Return class-aware regexes for natural descriptions of the non-predicted direction."""
    patterns = list(_OPPOSING_PATTERNS_BASE)
    if str(category) == "moderate":
        patterns.extend((
            r"\blower severity\b",
            r"\btoward\s+(?:(?:a|the)\s+)?rare fall(?:\s+classification)?\b",
        ))
    else:
        patterns.extend((
            r"\bhigher severity\b",
            r"\btoward\s+(?:(?:a|the)\s+)?fall(?:\s+classification)?\b",
            r"\btoward\s+(?:(?:a|the)\s+)?recurrent fall(?:\s+classification)?\b",
        ))
    return patterns


def _contrast_clauses(text):
    """Split a sentence head where a contrast marker changes the attributed direction.

    This covers both ``supporting ..., while opposing ...`` and
    ``Although opposing ..., supporting ...``. The prompt normally emits separate sentences, but
    clause splitting prevents a single contrast word from assigning every factor in a mixed
    sentence to the same side.
    """
    match = _CONTRAST_MARKER.search(text)
    if not match:
        return [text]

    before = text[:match.start()]
    marked = text[match.start():]
    clauses = [before] if before.strip() else []

    marker = match.group(0).lower()
    if not before.strip() and marker in {"while", "although"}:
        # A leading concessive clause may itself contain commas in a factor list. Split only at a
        # comma whose remainder explicitly switches back to the supporting prediction. Otherwise
        # keep the whole factor-bearing clause opposing (for example: "Although factors pushing
        # toward Fall included motor symptoms, supported by FOG and UPDRS-IV, these did not ...").
        for comma in (match.start() for match in re.finditer(",", marked)):
            remainder = marked[comma + 1:]
            if _EXPLICIT_SUPPORTING_CLAUSE.search(remainder):
                clauses.extend((marked[:comma + 1], remainder))
                return [clause for clause in clauses if clause.strip()]
        clauses.append(marked)
        return [clause for clause in clauses if clause.strip()]

    clauses.append(marked)
    return [clause for clause in clauses if clause.strip()]


def _is_opposing(segment, category):
    """Return whether a prose segment attributes its factors to the opposite direction."""
    if any(re.search(pattern, segment, re.I) for pattern in _opposing_patterns(category)):
        return True
    # A clause introduced by while/although is opposing under the required report structure even
    # when the model omits the literal phrase "opposite direction".
    return bool(re.match(r"\s*(?:while|although)\b", segment, re.I))


def _segments(prose, category):
    """Split prose into (segment_text, attributed_side) pairs.

    A sentence is 'opposing' if it carries a class-aware opposing cue, else 'supporting'. Only the
    active phrase "did not outweigh" creates a supporting tail. Passive wording such as "were not
    outweighed by factors pushing toward higher severity" keeps the factors after ``by`` opposing.
    """
    segments = []
    for sentence in re.split(r"(?<=[.!?])\s+", str(prose)):
        outweigh = _ACTIVE_OUTWEIGH.search(sentence)
        head = sentence if outweigh is None else sentence[:outweigh.start()]
        tail = "" if outweigh is None else sentence[outweigh.start():]
        for clause in _contrast_clauses(head):
            side = "opposing" if _is_opposing(clause, category) else "supporting"
            segments.append((clause, side))
        if tail:
            segments.append((tail, "supporting"))
    return segments


def _sides(evidence, category):
    """Return the factors the interpretation is expected to discuss on each side.

    Prospective evidence supplies the exact first-table top 5 and second-table top 3 roles saved at
    generation time. The SHAP-sign/category rule remains only for historical rows and test fixtures.
    """
    if "supporting_order" in evidence and "opposing_order" in evidence:
        supporting = set(evidence["supporting_order"])
        opposing = set(evidence["opposing_order"])
        return supporting, opposing

    # Compatibility for small unit-test fixtures and historical callers without saved roles.
    stage = int(evidence["stage"])
    supporting_sign = "positive" if str(category) == "moderate" else "negative"
    supporting = set(evidence["expected_order"].get(f"s{stage}_{supporting_sign}", []))
    opposing = set(evidence["factor_set"]) - supporting
    return supporting, opposing


def _grouped_template_members(group, supporting, opposing, attributed_side):
    """Recover the selected factors represented by one grouped-template category label.

    The renderer emits a category only when at least two selected factors on one side belong to it.
    If only one side meets that rule, those are the factors represented by the label even when the
    prose segment is accidentally attributed to the other side. If both sides independently meet
    the rule, the segment's attributed side disambiguates the otherwise identical label.
    """
    category_members = CATEGORY_TO_FEATURES.get(group, set())
    candidates = {
        "supporting": category_members & supporting,
        "opposing": category_members & opposing,
    }
    eligible = {side: members for side, members in candidates.items() if len(members) >= 2}
    if len(eligible) == 1:
        return next(iter(eligible.values()))
    return eligible.get(attributed_side, set())


def prose_direction_fidelity(prose, evidence, category, expand_template_groups=False):
    """Distinct grounded factors attributed exclusively to the correct side / factors mentioned.

    Repetition on one side never gives a factor extra weight. A factor appearing on both sides is
    contradictory and therefore counted once as incorrect, even if one occurrence is correct. For
    the grouped-template arm, a rendered category is expanded back to the exact selected factors
    that met the renderer's >=2-members grouping rule. LLM categories are not expanded because its
    prompt requires the supporting individual factors to be named explicitly.
    """
    supporting, opposing = _sides(evidence, category)
    grounded = supporting | opposing
    mentioned = {"supporting": set(), "opposing": set()}
    for segment, side in _segments(prose, category):
        features, groups = _detect_segment(segment)
        mentioned[side] |= features & grounded

    all_mentioned = mentioned["supporting"] | mentioned["opposing"]
    if not all_mentioned:
        return np.nan
    both_sides = mentioned["supporting"] & mentioned["opposing"]
    correct = (
        (supporting & mentioned["supporting"])
        | (opposing & mentioned["opposing"])
    ) - both_sides
    return len(correct) / len(all_mentioned)


def represented_factors_by_side(prose, evidence, category, method):
    """Return selected factors represented in supporting/opposing interpretation segments.

    Every method must name factors explicitly. Method identity is accepted only for API
    compatibility and never changes extraction.
    """
    supporting, opposing = _sides(evidence, category)
    selected = supporting | opposing
    represented = {"supporting": set(), "opposing": set()}
    for segment, side in _segments(prose, category):
        features, groups = _detect_segment(segment)
        represented[side] |= features & selected
    return represented


_GROUP_MEMBER_CONNECTOR = re.compile(
    r"(?:\(|[,—-]\s*)?(?:supported|represented|demonstrated|reflected|characterized)\s+by\b"
    r"|(?:\(|[,—-]\s*)?(?:specifically|including|namely|comprising|comprised of|encompasses|encompassing)\b",
    re.I,
)
_GROUP_MEMBER_BOUNDARY = re.compile(
    r"—|;|,\s+(?:along(?:side)?|as well as|in addition to|plus)\b"
    r"|,\s+and\s+(?:a|an|the|patient(?:'s)?)\b(?!\s+(?:absence|presence)\s+of\b)",
    re.I,
)


def _expected_grouping(supporting, opposing):
    """Return expected multi-factor groups and remaining singletons for both directions."""
    result = {}
    for side, factors in (("supporting", supporting), ("opposing", opposing)):
        category_members = {}
        for factor in factors:
            factor_category = FEATURE_CATEGORY.get(factor)
            if factor_category:
                category_members.setdefault(factor_category, set()).add(factor)
        groups = {
            factor_category: members
            for factor_category, members in category_members.items()
            if len(members) >= 2
        }
        grouped = set().union(*groups.values()) if groups else set()
        result[side] = {"groups": groups, "singletons": set(factors) - grouped}
    return result


def _relation_end(text, start, next_group_start=None):
    """Find the conservative end of a category's explicitly introduced member list."""
    ends = [len(text)]
    if next_group_start is not None:
        ends.append(next_group_start)
    boundary = _GROUP_MEMBER_BOUNDARY.search(text, start)
    if boundary:
        ends.append(boundary.start())
    return min(ends)


def _near_member_parenthesis(text, category_end):
    """Return a nearby member-list opening parenthesis, excluding ordinary descriptive prose."""
    open_paren = text.find("(", category_end, min(len(text), category_end + 45))
    if open_paren == -1:
        return None
    between = text[category_end:open_paren]
    allowed = re.fullmatch(
        r"\s*(?:(?:factors?|domain|measures?|features?|characteristics?|assessments?|performance|severity"
        r"|impairments?|difficult(?:y|ies)|dysfunction|function|symptoms?|burden|deficits?|problems?"
        r"|involvement|complaints?)\s*)?"
        r"(?:at diagnosis\s*)?",
        between,
        re.I,
    )
    return open_paren if allowed else None


def _near_member_connector(text, category_end):
    """Only attach a connector introducing this group, not one in a later factor gloss."""
    connector = _GROUP_MEMBER_CONNECTOR.search(text, category_end, category_end + 65)
    if connector is None:
        return None
    gap = text[category_end:connector.start()]
    if re.fullmatch(
        r"\s*[,：:—-]?\s*(?:(?:factors?|domain|measures?|features?|characteristics?|assessments?|"
        r"findings|performance|severity|impairments?|difficult(?:y|ies)|dysfunction|function|"
        r"symptoms?|burden|deficits?|problems?|involvement|complaints?|"
        r"which|were|was|are|is)\s*[,：:—-]?\s*)*", gap, re.I
    ):
        return connector
    return None


def _matching_parenthesis(text, opening):
    """Return the matching close for ``opening``, respecting nested factor-name parentheses."""
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _llm_group_relations(segment):
    """Extract explicit ``category -> named member factors`` relations from one prose segment.

    The prompt requires the LLM to name category members. We therefore accept parenthesized members
    or an explicit connector such as "supported by", "represented by", or "specifically". Merely
    placing a category word and a factor somewhere in the same sentence is not enough.
    """
    text = str(segment)

    # Scan categories independently from factors. This allows a category phrase such as
    # "gait-related measures" to coexist with explicit member names, while the normal combined
    # detector continues to prevent a category word from masquerading as an individual factor.
    feature_spans, occupied = [], []
    low = text.lower()
    for phrase, factor in _FEATURE_LEXICON:
        for match in re.finditer(_lexicon_pattern(phrase, "feature"), low):
            if any(not (match.end() <= start or match.start() >= end)
                   for start, end in occupied):
                continue
            occupied.append((match.start(), match.end()))
            feature_spans.append((match.start(), match.end(), factor, phrase))

    group_spans, occupied = [], []
    for phrase, group in _CATEGORY_LEXICON:
        for match in re.finditer(_lexicon_pattern(phrase, "group"), low):
            if any(not (match.end() <= start or match.start() >= end)
                   for start, end in occupied):
                continue
            occupied.append((match.start(), match.end()))
            group_spans.append((match.start(), match.end(), group, phrase))
    feature_spans.sort()
    group_spans.sort()

    # A category-looking phrase is a group relation only when it has an explicit connector or a
    # parenthetical list containing at least two factors. A one-factor gloss such as
    # "motor complications (UPDRS-IV Total)" names a factor; it must not truncate an outer relation
    # such as "motor characteristics, specifically ...".
    filtered_groups = []
    for span in group_spans:
        _start, end, _group, _phrase = span
        connector = _near_member_connector(text, end)
        opening = _near_member_parenthesis(text, end)
        parenthetical_members = set()
        if opening is not None:
            closing = _matching_parenthesis(text, opening)
            if closing is not None:
                parenthetical_members = {
                    factor for fstart, fend, factor, _fphrase in feature_spans
                    if fstart >= opening + 1 and fend <= closing
                }
        if connector is not None or len(parenthetical_members) >= 2:
            filtered_groups.append(span)
    group_spans = filtered_groups
    relations = []
    for index, (_gstart, gend, group, _phrase) in enumerate(group_spans):
        next_group_start = group_spans[index + 1][0] if index + 1 < len(group_spans) else None
        member_start = None
        connector_used = False

        # ``mobility assessments (Gait and Mobility Summary)`` or
        # ``gait-related measures (FOG and Gait)``.
        open_paren = _near_member_parenthesis(text, gend)
        connector = _near_member_connector(text, gend)
        if (open_paren is not None
                and (connector is None or open_paren <= connector.start())
                and (next_group_start is None or open_paren < next_group_start)):
            member_start = open_paren + 1
            close = _matching_parenthesis(text, open_paren)
            if close is None:
                continue  # Unclosed membership is ambiguous; never extend it across the report.
            member_end = close
        else:
            # Do not attach a distant connector belonging to another category or clause.
            if (connector is None or connector.start() - gend > 45
                    or (next_group_start is not None and connector.start() >= next_group_start)):
                continue
            connector_used = True
            member_start = connector.end()
            member_end = _relation_end(text, member_start, next_group_start)

        members = {
            factor for start, end, factor, _phrase in feature_spans
            if start >= member_start and end <= member_end
        }
        # A parenthetical one-factor gloss such as "cognitive function (MoCA)" is an individual
        # description, not necessarily an attempted group. An explicit grouping connector with one
        # member remains an evaluable undersized-group claim.
        if len(members) < 2 and not connector_used:
            continue
        relations.append((group, members))
    return relations


def _serialize_groups(items):
    return ";".join(
        f"{side}:{group}[{','.join(sorted(members))}]"
        for side, group, members in sorted(items, key=lambda item: (item[0], item[1]))
    )


def _serialize_factors(items):
    return "|".join(sorted(items))


def prose_grouping_evaluation(prose, evidence, category, method="gemini"):
    """Evaluate grounded grouping constraints and taxonomy agreement separately.

    Prose grouping fidelity averages only selected-factor precision and >=2-member compliance.
    These have objective answers from the saved top-5/top-3 packet and prompt. Taxonomy recall and
    precision are returned separately: they measure agreement with one authored operational scheme,
    not whether an alternative clinical grouping is wrong. Direction and factor representation are
    handled by prose direction fidelity and completeness, respectively.
    """
    supporting, opposing = _sides(evidence, category)
    sides = {"supporting": supporting, "opposing": opposing}
    selected = supporting | opposing
    expected = _expected_grouping(supporting, opposing)

    observed_relations = []
    direct_features = {"supporting": set(), "opposing": set()}
    all_detected_features = {"supporting": set(), "opposing": set()}

    for segment, side in _segments(prose, category):
        features, groups = _detect_segment(segment)
        all_detected_features[side] |= features

        assigned = set()
        if method in {"template_grouped", "gemini"}:
            for group, members in _llm_group_relations(segment):
                observed_relations.append((side, group, set(members)))
                assigned |= members

        direct_features[side] |= features - assigned

    observed_by_key = {}
    for side, group, members in observed_relations:
        observed_by_key.setdefault((side, group), set()).update(members)

    missing_expected_groups = []
    for side in ("supporting", "opposing"):
        for group, members in expected[side]["groups"].items():
            if observed_by_key.get((side, group), set()) != members:
                missing_expected_groups.append((side, group, members))

    unexpected_groups = []
    undersized_groups = []
    invalid_category_members = set()
    wrong_side_group_members = set()
    nonselected_group_members = set()
    factor_group_categories = {}
    for side, group, members in observed_relations:
        if group not in expected[side]["groups"]:
            unexpected_groups.append((side, group, members))
        if len(members) < 2:
            undersized_groups.append((side, group, members))
        for factor in members:
            factor_group_categories.setdefault((side, factor), set()).add(group)
            if factor not in selected:
                nonselected_group_members.add(factor)
            elif factor not in sides[side]:
                wrong_side_group_members.add(factor)
            if FEATURE_CATEGORY.get(factor) != group:
                invalid_category_members.add(factor)

    duplicated_group_members = {
        factor for (_side, factor), groups in factor_group_categories.items() if len(groups) > 1
    }
    missing_singletons = set()
    wrong_side_singletons = set()
    for side, other_side in (("supporting", "opposing"), ("opposing", "supporting")):
        for factor in expected[side]["singletons"]:
            if factor not in direct_features[side]:
                missing_singletons.add(factor)
                if factor in direct_features[other_side]:
                    wrong_side_singletons.add(factor)

    detected_any_side = all_detected_features["supporting"] | all_detected_features["opposing"]
    nonselected_factors = detected_any_side - selected

    expected_group_items = [
        (side, group, members)
        for side in ("supporting", "opposing")
        for group, members in expected[side]["groups"].items()
    ]

    # Category/factor relationships are scored without side so a directional error is not counted
    # here and again in prose_direction_fidelity.
    expected_memberships = {
        (group, factor)
        for _side, group, members in expected_group_items
        for factor in members
    }
    observed_memberships = {
        (group, factor)
        for _side, group, members in observed_relations
        for factor in members
    }
    taxonomy_grouping_recall = (
        len(expected_memberships & observed_memberships) / len(expected_memberships)
        if expected_memberships else np.nan
    )
    taxonomy_grouping_precision = (
        sum(FEATURE_CATEGORY.get(factor) == group
            for group, factor in observed_memberships) / len(observed_memberships)
        if observed_memberships else np.nan
    )
    observed_grouped_factors = {factor for _group, factor in observed_memberships}
    selected_factor_precision = (
        len(observed_grouped_factors & selected) / len(observed_grouped_factors)
        if observed_grouped_factors else np.nan
    )
    minimum_group_size_compliance = (
        sum(len(members) >= 2 for _side, _group, members in observed_relations)
        / len(observed_relations)
        if observed_relations else np.nan
    )

    expected_singletons = (
        expected["supporting"]["singletons"] | expected["opposing"]["singletons"]
    )
    components = [
        selected_factor_precision,
        minimum_group_size_compliance,
    ]
    evaluable_components = [score for score in components if not np.isnan(score)]
    composite = float(np.mean(evaluable_components)) if evaluable_components else np.nan

    # The generic template is intentionally a non-grouping baseline. Grouping quality is therefore
    # outside its design scope rather than a failed grouping attempt. Its factor representation is
    # measured by completeness and its prose structure by the synthesis metrics.
    if method == "template":
        taxonomy_grouping_recall = np.nan
        taxonomy_grouping_precision = np.nan
        selected_factor_precision = np.nan
        minimum_group_size_compliance = np.nan
        composite = np.nan

    return {
        "prose_grouping_fidelity": composite,
        "taxonomy_grouping_recall": taxonomy_grouping_recall,
        "taxonomy_grouping_precision": taxonomy_grouping_precision,
        "selected_factor_precision": selected_factor_precision,
        "minimum_group_size_compliance": minimum_group_size_compliance,
        "expected_groups": _serialize_groups(expected_group_items),
        "observed_groups": _serialize_groups(observed_relations),
        "expected_singletons": _serialize_factors(expected_singletons),
        "expected_supporting_singletons": _serialize_factors(
            expected["supporting"]["singletons"]),
        "expected_opposing_singletons": _serialize_factors(
            expected["opposing"]["singletons"]),
        "observed_singletons": _serialize_factors(
            direct_features["supporting"] | direct_features["opposing"]),
        "observed_supporting_singletons": _serialize_factors(direct_features["supporting"]),
        "observed_opposing_singletons": _serialize_factors(direct_features["opposing"]),
        "missing_expected_groups": _serialize_groups(missing_expected_groups),
        "unexpected_groups": _serialize_groups(unexpected_groups),
        "undersized_groups": _serialize_groups(undersized_groups),
        "invalid_category_members": _serialize_factors(invalid_category_members),
        "wrong_side_group_members": _serialize_factors(wrong_side_group_members),
        "nonselected_group_members": _serialize_factors(nonselected_group_members),
        "duplicated_group_members": _serialize_factors(duplicated_group_members),
        "missing_singletons": _serialize_factors(missing_singletons),
        "wrong_side_singletons": _serialize_factors(wrong_side_singletons),
        "nonselected_interpretation_factors": _serialize_factors(nonselected_factors),
    }


def prose_grouping_fidelity(prose, evidence, category, method="gemini"):
    """Return mean selected-factor precision and >=2-member compliance, or NaN without groups."""
    return prose_grouping_evaluation(prose, evidence, category, method)[
        "prose_grouping_fidelity"
    ]


# --- Value-descriptor consistency (ported Part C) ---------------------------------------------
def _descriptor_bucket(window):
    window = str(window).lower()
    best_pos, best_bucket = -1, None
    for bucket, words in _VALUE_DESCRIPTORS.items():
        for word in words:
            for match in re.finditer(rf"(?<!\w){re.escape(word)}(?!\w)", window):
                if match.start() > best_pos:
                    best_pos, best_bucket = match.start(), bucket
    return best_bucket


def _value_claims(prose):
    """Return [(feature, bucket)] for features named with an adjacent value descriptor."""
    text = str(prose).lower()
    spans, claims = [], []
    for phrase, feature in _FEATURE_LEXICON:
        for match in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
            if any(not (match.end() <= start or match.start() >= end) for start, end in spans):
                continue
            spans.append((match.start(), match.end()))
            window = text[max(0, match.start() - 38):match.start()]
            separators = [" and ", ", ", "; ", " including ", " alongside ", " while ", " though "]
            cuts = [window.rfind(sep) + len(sep) for sep in separators if window.rfind(sep) != -1]
            if cuts:
                window = window[max(cuts):]
            bucket = _descriptor_bucket(window)
            if bucket:
                claims.append((feature, bucket))
    return claims


def _assigned_scale_bucket(value, scale):
    """Return the qualitative bucket explicitly assigned to ``value`` in a categorical scale.

    Continuous ranges such as ``0-132; higher = worse`` intentionally return None. No clinical
    cutoffs are invented. Parenthetical detail is ignored so a leading label such as ``Mild`` is
    not confused by words later in its explanation.
    """
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    pattern = re.compile(
        r"(?:^|[.;])\s*(-?\d+(?:\.\d+)?)\s*=\s*(.*?)"
        r"(?=;\s*-?\d+(?:\.\d+)?\s*=|$)",
        re.I,
    )
    for match in pattern.finditer(str(scale)):
        if abs(float(match.group(1)) - numeric_value) < 1e-9:
            assigned_label = match.group(2).split("(", 1)[0].strip()
            return _descriptor_bucket(assigned_label)
    return None


def prose_value_descriptor_fidelity(prose, evidence):
    """Consistency with qualitative labels explicitly assigned by the supplied scale.

    Claims about continuous scores without categorical cutoffs are non-evaluable rather than being
    judged through authored low/high thresholds.
    """
    checkable = consistent = 0
    for feature, bucket in _value_claims(prose):
        content = evidence["content"].get(feature, {})
        expected_bucket = _assigned_scale_bucket(
            content.get("value"), content.get("scale", "")
        )
        if expected_bucket is None:
            continue
        checkable += 1
        consistent += int(bucket == expected_bucket)
    return consistent / checkable if checkable else np.nan


PROSE_METRICS = [
    "prose_direction_fidelity",
    "prose_grouping_fidelity",
    "selected_factor_precision",
    "minimum_group_size_compliance",
    "prose_value_descriptor_fidelity",
    "taxonomy_grouping_recall",
    "taxonomy_grouping_precision",
]


def compute_interpretation_fidelity(report, evidence, source_row, method=None):
    """The automatic interpretation-prose fidelity panel for one patient."""
    prose = report["interpretation"]
    category = source_row["category"]
    grouping = prose_grouping_evaluation(prose, evidence, category, method or "gemini")
    return {
        "prose_direction_fidelity": prose_direction_fidelity(
            prose, evidence, category, expand_template_groups=(method == "template_grouped")),
        **grouping,
        "prose_value_descriptor_fidelity": prose_value_descriptor_fidelity(prose, evidence),
    }
