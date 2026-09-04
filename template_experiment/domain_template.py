"""Domain-enriched deterministic renderer (grouping template arm).

Identical to the generic template in every section except the Model Interpretation paragraph, which
groups the top factors into frozen clinical categories instead of listing them individually. This is
the *fair* deterministic comparator for the LLM: Both methods are allowed to group factors, but the domain template uses
 an explicit frozen taxonomy, whereas the LLM infers groupings from factor names, metadata, prompt examples, and its learned knowledge.

 
 Known implementation burden:

1. FEATURE_CATEGORY contains 21 manually authored feature assignments.
2. CATEGORY_LABELS contains 8 manually authored report labels.
3. Grouping is controlled by one explicit threshold rule:
   two or more same-category factors produce a group label;
   singletons and unmapped features remain individual names.

Hypothesis to test (scalability):

The LLM can produce defensible group-level synthesis for unseen
features and combinations without prompt modification, while
preserving factual fidelity. This must be evaluated prospectively;
it is not assumed from the implementation.
"""

from explanation.explanation_builder import (
    build_patient_explanation_data_full,
    get_patient_index,
)

from .generic_template import _OUTCOME_PHRASE, _join_names, render_template_explanation
from .taxonomy import CATEGORY_LABELS, FEATURE_CATEGORY

GROUPED_METHOD_NAME = "deterministic-template-grouped-v2-explicit-members"

# at least 2 factors from known category, use category label
# 1 factor from category, use feature nme (same with unknown feature)
def _grouped_labels(factors, top_k):
    """Render explicit category membership (>=2 members) or bare singleton names.

    The canonical group contract is ``category (Factor A, Factor B)``.  This makes every
    structural claim auditable from the text itself; evaluation must never reconstruct hidden
    membership from knowledge of which renderer produced the text.
    """
    order, members = [], {}
    for factor in factors[:top_k]:
        category = FEATURE_CATEGORY.get(factor["short_name"])
        if category not in members:
            members[category] = []
            order.append(category)
        members[category].append(factor["short_name"])
    labels = []
    for category in order:
        if category and len(members[category]) >= 2:
            member_text = ", ".join(members[category])
            labels.append(f"{CATEGORY_LABELS.get(category, category)} ({member_text})")
        else:
            labels.extend(members[category])
    return _join_names(labels)


def _grouped_interpretation(patient_info, first_factors, second_factors):
    """Deterministic synthesis that groups factors into frozen clinical categories."""
    outcome = _OUTCOME_PHRASE[int(patient_info["final_prediction"])]
    supporting = _grouped_labels(first_factors, 5)
    first_sentence = f"Within this model, the {outcome} was primarily associated with {supporting}."
    if second_factors:
        opposing = _grouped_labels(second_factors, 3)
        second_sentence = (
            f"Factors supporting the opposite direction included {opposing}; however, these "
            f"did not outweigh the factors supporting the {outcome}."
        )
    else:
        second_sentence = "No qualifying factors supported the opposite direction."
    return f"{first_sentence} {second_sentence}"


def render_template_grouped_explanation(patient_info):
    return render_template_explanation(patient_info, interpretation_fn=_grouped_interpretation)


def generate_template_grouped_explanation(patient_id=None, patient_info=None):
    """Return a domain-enriched (grouping) deterministic explanation."""
    if patient_info is None:
        if patient_id is None:
            raise ValueError("Provide either patient_id or patient_info.")
        patient_info = build_patient_explanation_data_full(get_patient_index(patient_id))
    elif patient_id is not None and int(patient_info["patient_id"]) != int(patient_id):
        raise ValueError(
            f"patient_id {patient_id} does not match patient_info patient "
            f"{patient_info['patient_id']}.")
    return {
        "method": GROUPED_METHOD_NAME,
        "patient_id": int(patient_info["patient_id"]),
        "evidence": patient_info,
        "explanation": render_template_grouped_explanation(patient_info),
    }
