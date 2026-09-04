# Synthesis pilot — pairwise structural evaluation

## Scope

These results overwrite the former grouping-extent/Distinct-2 synthesis report. The saved Gemini interpretations were generated **without** the explicit task taxonomy, so this is the retrospective **map-free reference-taxonomy analysis (Part 2A)**. It is not the future same-taxonomy Part 1 noninferiority comparison.

## Results

| Metric | Definition | Generic template | Domain template | Map-free LLM (Gemini) |
| --- | --- | ---: | ---: | ---: |
| Macro pairwise precision | Mean report-level precision over evaluable reports. | N/A (0/30) | 100.0% (27/30) | 66.9% (28/30) |
| Macro pairwise recall | Mean report-level recall over reports with reference pairs. | 0.0% (27/30) | 100.0% (27/30) | 61.7% (27/30) |
| Macro pairwise F1 | Mean report-level F1 using the frozen empty-set rules. | 0.0% (27/30) | 100.0% (27/30) | 53.2% (29/30) |
| Micro pairwise precision | Correct pooled predicted pairs / all pooled predicted pairs. | N/A | 100.0% | 54.9% |
| Micro pairwise recall | Correct pooled predicted pairs / all pooled reference pairs. | 0.0% | 100.0% | 46.7% |
| Micro pairwise F1 | Harmonic mean of pooled precision and recall. | 0.0% | 100.0% | 50.5% |
| Exact pair-set agreement | Reports with identical predicted and reference pair sets. | 10.0% (30/30) | 100.0% (30/30) | 30.0% (30/30) |
| Exact agreement with opportunity | Exact agreement among reports with reference pairs. | 0.0% (27/30) | 100.0% (27/30) | 29.6% (27/30) |
| Explicit-membership recall | Named intended members / intended members in produced claims. | N/A (0/30) | 100.0% (27/30) | 83.3% (19/30) |
| Contrastive linkage | Correct two-sided evidence explicitly linked to the model output. | 100.0% (30/30) | 100.0% (30/30) | 98.3% (30/30) |
| Spurious claim without opportunity | Group claims in reports with no reference pair. | 0.0% (3/30) | 0.0% (3/30) | 66.7% (3/30) |

Cells with parentheses show `n evaluable / 30 patients`. Macro metrics give each evaluable patient equal weight; micro metrics pool factor pairs and therefore give more weight to reports with larger grouping opportunities.

## Pooled pair counts

| Method | Reference pairs | Predicted pairs | Correct pairs | Reports with opportunity |
| --- | ---: | ---: | ---: | ---: |
| Generic template | 60 | 0 | 0 | 27/30 |
| Domain template | 60 | 60 | 60 | 27/30 |
| Map-free LLM (Gemini) | 60 | 51 | 28 | 27/30 |

## Interpretation limits

The domain template's taxonomy agreement is expected by construction. Its regenerated v2 prose now names every group member explicitly, and the same method-agnostic parser is used for template and map-free prose. The map-free traceability result remains conditional on claims the parser can recognize; parser-validation results are published separately.

Pairwise agreement measures concordance with one researcher-defined taxonomy, not clinical accuracy. Contrastive linkage is a structural compliance measure and is expected to show a ceiling because the generation instructions require opposing evidence and 'did not outweigh' language. No composite synthesis score is reported. Narrative-quality judgments from the two configured separate-model judges will remain a secondary analysis.
