"""Versioned pilot checker: explicit recognition, never hidden-member reconstruction.

Flags are diagnostics, not admission criteria for the scientific evaluation. Unknown wording
requires review; recognized incorrect memberships/directions are retained as measured errors.
"""

import re

from .group_claims import extract_group_claims
from .prose_checks import _detect_segment_spans, _is_opposing, _segments


CHECKER_VERSION = "interpretation-contract-v2.1-pilot"


def contrastive_outweigh_evidence(text, category):
    """Recognize active/passive non-dominance, rejecting an explicitly reversed claim."""
    for sentence in re.split(r"(?<=[.!?])\s+", str(text)):
        active = re.search(r"\bdid\s+not\s+outweigh\b", sentence, re.I)
        if active:
            left, right = sentence[:active.start()], sentence[active.end():]
            if re.search(r"\b(?:supporting|supportive)\s+(?:factors|evidence|influences)\b", left, re.I):
                continue
            if re.match(r"\s*(?:the\s+)?opposing\s+(?:factors|evidence|influences)\b", right, re.I):
                continue
            return sentence
        passive = re.search(r"\b(?:was|were|is|are)\s+not\s+outweighed\s+by\b", sentence, re.I)
        if passive:
            left, right = sentence[:passive.start()], sentence[passive.end():]
            if re.search(r"\bopposing\s+(?:factors|evidence|influences)\b", left, re.I):
                continue
            if _is_opposing(right, category) or re.search(r"\bopposing\b", right, re.I):
                return sentence
    return None


def check_interpretation(text, factors, category, sentence_range=(2, 4)):
    """Check frozen selected factors, returning findings and auditable extraction evidence."""
    text = str(text).strip()
    issues = []

    def issue(code, kind, **detail):
        issues.append({"code": code, "kind": kind, **detail})

    sentences = len([s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()])
    if not sentence_range[0] <= sentences <= sentence_range[1]:
        issue(f"sentence_count={sentences}", "format_error")

    by_side = {"supporting": set(), "opposing": set()}
    mentions = []
    for segment, side in _segments(text, category):
        spans, _ = _detect_segment_spans(segment)
        for start, end, factor, _alias in spans:
            by_side[side].add(factor)
            mentions.append({"factor": factor, "side": side, "matched_text": segment[start:end],
                             "source_segment": segment})
    expected = {}
    side_name = {"supports_prediction": "supporting", "opposes_prediction": "opposing"}
    for factor in factors:
        side = side_name[factor["direction"]]
        if factor["domain"]:
            expected.setdefault((side, factor["domain"]), set()).add(factor["factor"])
    expected = {key: members for key, members in expected.items() if len(members) >= 2}
    claims = extract_group_claims(text, category)
    observed = {}
    for claim in claims:
        observed.setdefault((claim.side, claim.group), []).append(set(claim.members))
    for (side, domain), members in expected.items():
        predictions = observed.get((side, domain), [])
        # Do not merge separate claims or let a later valid claim erase an earlier wrong one.
        if not predictions or any(prediction != members for prediction in predictions):
            direction = "supports_prediction" if side == "supporting" else "opposes_prediction"
            extras = set().union(*predictions) - members if predictions else set()
            issue(f"missing_or_inexact_group={direction}:{domain}",
                  "content_mismatch" if extras else "extraction_uncertain",
                  expected_members=sorted(members), observed_members=[sorted(p) for p in predictions])
    for (side, domain), predictions in observed.items():
        # A correct category gloss for one explicitly named singleton does not combine
        # factors and is not an unexpected multi-factor group.
        multi_factor = [p for p in predictions if len(p) >= 2]
        if (side, domain) not in expected and multi_factor:
            issue(f"unexpected_group={side}:{domain}", "content_mismatch",
                  observed_members=[sorted(p) for p in multi_factor])
    for factor in factors:
        side = side_name[factor["direction"]]
        name = factor["factor"]
        other = "opposing" if side == "supporting" else "supporting"
        if name not in by_side[side]:
            if name in by_side[other]:
                issue(f"wrong_evidence_side={name}", "content_mismatch", expected_side=side)
            else:
                # Missing recognition could be an omission OR an unreviewed paraphrase.
                issue(f"unrecognized_factor={name}", "extraction_uncertain", expected_side=side)
        elif name in by_side[other]:
            issue(f"factor_on_both_sides={name}", "content_mismatch", expected_side=side)
    contrast = contrastive_outweigh_evidence(text, category)
    if any(f["direction"] == "opposes_prediction" for f in factors) and contrast is None:
        issue("unrecognized_contrastive_outweigh_statement", "extraction_uncertain")
    return {"checker_version": CHECKER_VERSION,
            "assessment": "pass" if not issues else "needs_review",
            "issues": issues, "factor_mentions": mentions,
            "group_claims": [claim.as_dict() for claim in claims],
            "contrastive_evidence": contrast, "sentence_count": sentences}
