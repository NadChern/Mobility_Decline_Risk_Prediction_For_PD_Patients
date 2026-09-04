"""Method-agnostic extraction of explicit factor-to-group claims.

The extractor accepts the frozen ``category (Factor A, Factor B)`` syntax and the explicit
connector forms used by the historical map-free reports (for example, ``category, supported by
Factor A and Factor B``).  It never receives a method identity and never imputes hidden members.
"""

from dataclasses import asdict, dataclass

from .prose_checks import _llm_group_relations, _segments


PARSER_VERSION = "explicit-group-claims-v2.1-pilot"


@dataclass(frozen=True)
class GroupClaim:
    side: str
    group: str
    members: tuple[str, ...]
    source_text: str
    span_start: int
    span_end: int

    def as_dict(self):
        return asdict(self)


def extract_group_claims(prose, category):
    """Return explicit claims with source spans, without method-specific reconstruction."""
    text = str(prose)
    claims = []
    cursor = 0
    for segment, side in _segments(text, category):
        start = text.find(segment, cursor)
        if start < 0:
            start = text.find(segment)
        if start < 0:
            start = 0
        end = start + len(segment)
        cursor = max(cursor, end)
        for group, members in _llm_group_relations(segment):
            claims.append(GroupClaim(
                side=side,
                group=group,
                members=tuple(sorted(set(members))),
                source_text=segment,
                span_start=start,
                span_end=end,
            ))
    return claims
