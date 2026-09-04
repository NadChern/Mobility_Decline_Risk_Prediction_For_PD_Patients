# Complexity / scalability / flexibility (Metric 5)

## Part A — static complexity (authored machinery)

| method | templates | grouping_rules | map_entries | category_labels | prompts | embedded_category_examples | interp_sloc | prompt_words |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Generic template | 1 | 0 | 0 | 0 | 0 | 0 | 25 | 0 |
| Domain template | 1 | 1 | 21 | 12 | 0 | 0 | 35 | 0 |
| LLM (Gemini) | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 199 |

The prompt is counted as manual logic. It contains grouping instructions but receives no explicit experiment taxonomy labels or mappings. The honest claim is that the template's authored burden **grows with the feature/category set**, while the prompt's is roughly fixed. SLOC is approximate (non-blank, non-comment).

## Part B — flexibility on an unseen feature (systematicity)

| method | renders_unseen_no_edits | groups_unseen_no_edits | edits_to_group_new_feature | note |
| --- | --- | --- | --- | --- |
| Generic template | True | False | n/a (never groups) | Generic loop renders any factor list; it never groups, so nothing to maintain. |
| Domain template | True | False | 1 map entry for an existing category; +1 display label for a new category | Unmapped feature silently falls back to listing; grouping requires new artifacts. |
| LLM (Gemini) | True | CLAIM — verify | 0 (from the same prompt) — VERIFY compositional fidelity | Must be confirmed by generation + re-running the fidelity panel on the new case; neural D2T can hallucinate/omit on unseen combinations. |

The LLM's zero-edit rows are **claims to be confirmed by generation**, not assumptions: we must generate on the held-out feature/combination and re-run the fidelity panel (compositional fidelity), because neural data-to-text models also hallucinate or omit on unseen combinations. Report whatever we find, including cases where the LLM needs no edits but is less faithful.

**Next (needs generation + frozen dev/held-out split):** per held-out case, log edits required, success-without-modification, and compositional fidelity for all three arms.
