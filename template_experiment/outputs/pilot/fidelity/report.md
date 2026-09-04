# Fidelity pilot

## Tables and label

| Metric | Definition | LLM (Gemini) (n eval) | Generic template (n eval) | Domain template (n eval) |
| --- | --- | ---: | ---: | ---: |
| Outcome correctness | Is the reported fall classification correct? | 100.00% (30/30) | 100.00% (30/30) | 100.00% (30/30) |
| Direction accuracy | Are displayed factors in the correct supporting or opposing table? | 100.00% (30/30) | 100.00% (30/30) | 100.00% (30/30) |
| Value fidelity | Are displayed patient values copied exactly from the supplied evidence? | 100.00% (30/30) | 100.00% (30/30) | 100.00% (30/30) |
| Scale fidelity | Is each displayed scale explanation copied completely and exactly? | 100.00% (30/30) | 100.00% (30/30) | 100.00% (30/30) |
| Rank-order fidelity | Are factors kept in their supplied order within each table? | 100.00% (30/30) | 100.00% (30/30) | 100.00% (30/30) |

## Interpretation prose (automatic panel)

| Metric | Definition | LLM (Gemini) (n eval) | Generic template (n eval) | Domain template (n eval) |
| --- | --- | ---: | ---: | ---: |
| Prose direction fidelity | Of the factors mentioned, how many are discussed on the correct supporting or opposing side? | 100.00% (30/30) | 100.00% (30/30) | 100.00% (30/30) |
| Prose grouping fidelity | For each report with groups, what is the average of the two checks below? | 99.14% (29/30) | NA (0/30) | 100.00% (27/30) |
| — Selected-factor precision | Of the factors placed in groups, how many belong to the required top 5/top 3? | 100.00% (29/30) | NA (0/30) | 100.00% (27/30) |
| — Minimum-size compliance | Of the groups created, how many contain at least two factors? | 98.28% (29/30) | NA (0/30) | 100.00% (27/30) |
| Prose value-descriptor fidelity | Of descriptions with an exact categorical scale label, how many match that label? | 100.00% (26/30) | NA (0/30) | NA (0/30) |

**Grouping denominators:** The Domain template is evaluable for 27/30 reports; the other 3 had no category containing at least two selected factors. The LLM is evaluable for 29/30; it formed no group in 1 report.

This supports the conclusion:

> When the LLM formed groups, it used grounded selected factors and followed the minimum two-factor requirement.

It does **not** support:

> The LLM’s clinical categories were definitively correct.

## Agreement with the reference taxonomy

| Metric | Definition | LLM (Gemini) (n eval) | Generic template (n eval) | Domain template (n eval) |
| --- | --- | ---: | ---: | ---: |
| Taxonomy grouping recall | Of the factor–category links defined by the reference taxonomy, how many did the prose reproduce? | 54.94% (27/30) | NA (0/30) | 100.00% (27/30) |
| Taxonomy grouping precision | Of the factor–category links made in the prose, how many match the reference taxonomy? | 67.26% (29/30) | NA (0/30) | 100.00% (27/30) |

These are descriptive agreement metrics, not clinical-validity metrics. The taxonomy is one predefined operational scheme for the rule-based template. The Domain template’s 100% agreement is expected because it uses that same taxonomy; a different LLM grouping may still be clinically defensible.

Table/label fidelity is conditional: it scores only content that is actually displayed. A missing factor is recorded by completeness (recall) and an extra or duplicate factor by unsupported-information (precision), so neither is double-counted here. Metrics return NaN when nothing is evaluable; the evaluable denominator `(n eval / n patients)` is shown so empty output cannot look artificially faithful. Value and scale fidelity are verbatim (whitespace-normalized) matches, so a paraphrase counts as a mismatch.

The interpretation-prose panel is a conservative, human-free screen. Prose grouping fidelity is the unweighted mean of selected-factor precision and minimum-size compliance. It is conditional on a group being formed. Direction is scored separately, and required factor representation is assessed by completeness. The generic template has no grouping mechanism, so grouping metrics are not applicable for that arm. Value-description fidelity uses only qualitative labels explicitly assigned to the patient’s exact value in the supplied scale; continuous scores without categorical cutoffs are not evaluated. The panel does not judge synthesis quality, and it cannot catch a subtly worded invented interaction. Feature hallucination and causal language are reported by the unsupported-information module.
