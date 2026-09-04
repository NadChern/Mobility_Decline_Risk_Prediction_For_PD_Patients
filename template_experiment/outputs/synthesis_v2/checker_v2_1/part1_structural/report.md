# Part 1 structural synthesis

This report compares the domain template and taxonomy-conditioned LLM on the same patients and evidence. It does not use LLM-judge results.

## Primary result

| Result | Value |
| --- | ---: |
| Template mean patient-level pairwise grouping F1 | 1.000 |
| Taxonomy-conditioned LLM mean patient-level pairwise grouping F1 | 1.000 |
| Mean paired difference (LLM − template) | 0.000 |
| Stratified paired-bootstrap 95% CI | [0.000, 0.000] |
| Eligible paired patients | 27 |
| Bootstrap replicates | 10,000 |
| Bootstrap seed | 20260903 |
| Prespecified noninferiority margin (Δ) | 0.05 |
| Decision boundary for LLM − template | -0.05 |
| Noninferiority criterion met | Yes |

Decision rule: the lower confidence bound must be greater than -0.05. The frozen noninferiority rule was met.

The primary endpoint includes patients with at least one reference grouping opportunity. Patients without an opportunity remain in the complete results and are checked for spurious grouping claims.

## Reusable result files

- `noninferiority.csv`: inference inputs, confidence interval, margin, and decision.
- `paired_differences.csv`: one LLM-minus-template F1 difference per eligible patient.
- `summary.csv`: macro and micro grouping metrics for both methods.
- `patients.csv`: all patient-level claims, spans, pair sets, and metric values.
- `verification_*.csv`: fidelity, completeness, unsupported-information, and structure checks.

Taxonomy agreement is operational, not proof of clinical correctness. These are pilot results, not independent final-test validation.
