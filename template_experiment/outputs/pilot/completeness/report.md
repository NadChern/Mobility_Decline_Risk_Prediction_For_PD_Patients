# Completeness pilot

| Metric | Generic template | Domain template | LLM (Gemini) |
| --- | ---: | ---: | ---: |
| Supplied-factor recall | 100.00% | 100.00% | 100.00% |
| Required-section nonempty coverage | 100.00% | 100.00% | 100.00% |
| Classification-field coverage | 100.00% | 100.00% | 100.00% |
| Expected table-cell completeness | 100.00% | 100.00% | 100.00% |
| Displayed-stage evidence coverage | 100.00% | 100.00% | 100.00% |
| Full-pipeline stage-evidence coverage | 66.67% | 66.67% | 66.67% |
| Interpretation key-factor coverage | 100.00% | 100.00% | 100.00% |
| Opposing-evidence coverage | 100.00% | 100.00% | 100.00% |
| Displayed-contract complete rate | 100.00% | 100.00% | 100.00% |
| Full-pipeline complete rate | 33.33% | 33.33% | 33.33% |

Displayed-contract completeness evaluates the current single-stage tables and requires a non-empty interpretation, but does not fold the lexical interpretation-coverage screens into the overall pass. Full-pipeline completeness additionally requires Stage-1 evidence for routed fallers (so a routed faller caps at 0.5 until both stages are shown). Interpretation coverage uses the saved top-5/top-3 roles and the correct prose side. All methods must name represented factors explicitly; hidden group membership receives no coverage credit. Grouping correctness is reported only under prose grouping fidelity.
