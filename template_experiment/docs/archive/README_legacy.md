# Template Explanation Experiment

This folder contains the deterministic comparator for the LLM explanation pipeline. It is separate
from `explanation/` so the existing production behavior remains unchanged while the comparison is
developed and frozen.

The baseline consumes the same enriched dictionary returned by
`build_patient_explanation_data_full()` and performs no model inference, feature selection, clinical
inference, or feature-combination lookup. It renders arbitrary factor lists using one generic set of
rules.

## Quick use

```python
from template_experiment import generate_template_explanation

result = generate_template_explanation(3207)
print(result["explanation"])
```

The result contains:

- `method`: the fixed baseline identifier;
- `patient_id`;
- `evidence`: the enriched patient packet used by the renderer; and
- `explanation`: the deterministic Markdown document.

No API key or LLM call is required.

## Generate both methods from one evidence packet

For the final paired experiment, use the shared-packet entry point. It builds
`patient_info` once and passes the same in-memory object to the LLM and deterministic renderer:

```python
from template_experiment import generate_paired_explanation

result = generate_paired_explanation(
    patient_id=3207,
    provider="google",
    temperature=0,
)

patient_info = result["evidence"]
llm_output = result["llm"]
template_output = result["template"]
```

This entry point makes one LLM call. Template generation requires no additional tokens. Existing
`generate_template_explanation(patient_id)` and `generate_explanation(patient_id)` calls remain
available for standalone use.

## Baseline scope

Version 1 intentionally uses no manually curated feature-domain rules. Its interpretation names the
top supporting and opposing factors in their supplied SHAP order. A later, separately identified
condition may add frozen domain mappings; it must not silently replace this baseline.

## Thirty-patient metric-validation pilot

The current pilot pairs 30 saved Gemini explanations with deterministic reports for the same
patients: 10 No Fall, 10 Rare Fall, and 10 Recurrent Fall predictions. It is intended to validate
metric definitions, mutation sensitivity, output tables, and analysis code while the ML pipeline is
being finalized. It is not the definitive experiment for establishing LLM superiority.

Build or refresh the paired pilot without making LLM calls:

```bash
./.venv/bin/python -m template_experiment.evaluation.build_gemini_pairs
```

The historical Gemini outputs use earlier Mild/Moderate terminology. The pilot retains the raw text
and creates a transparently normalized comparison copy using current Rare/Recurrent terminology.
Final paper-level outputs must be regenerated prospectively with one frozen terminology and evidence
contract.

## Evaluation families

The comparison deliberately keeps four metric families separate:

| Family | Primary question | Included measures |
| --- | --- | --- |
| Fidelity | Is present content correct? | Outcome, factor-set, direction, value, scale, rank, stage/routing, lexical prose-feature grounding |
| Completeness | Is required content present? | Factor recall, sections and non-empty content, expected table cells, classification fields, displayed-stage and full-pipeline stage coverage, interpretation coverage screens |
| Unsupported information | Was evidence added, duplicated, or altered? | Factor precision, extra/duplicate factors, changed values/scales, unsupported prose features, causal-language screen, manual claim audit |
| Structure compliance | Does the report follow the required format? | Title and section order, stage template, table headers/rows/numbering/order, forbidden content, overall structure score |

Structure compliance is not used as a substitute for factual fidelity. Likewise, a complete report
may still contain incorrect or unsupported information.

Run all four comparisons from the project root:

```bash
./.venv/bin/python -m template_experiment.evaluation.fidelity_comparison
./.venv/bin/python -m template_experiment.evaluation.completeness_comparison
./.venv/bin/python -m template_experiment.evaluation.unsupported_information_comparison
./.venv/bin/python -m template_experiment.evaluation.structure_comparison
```

Generated artifacts are kept under:

```text
template_experiment/outputs/gemini/
  paired_explanations.csv
  fidelity_comparison/
  completeness_comparison/
  unsupported_information_comparison/
  structure_comparison/
```

Each comparison writes patient-level results, category/overall summaries, paired
template-minus-Gemini differences, and a Markdown report. The unsupported-information analysis also
writes a randomized blinded prose-claim worksheet and a separate method-identity key.

## Interpretation and prose limits

Automatic interpretation metrics use a frozen feature-phrase lexicon. They are screening measures,
not general semantic evaluation. Broader clinical-domain wording can be valid synthesis without
repeating a feature label, so lexical interpretation coverage does not determine the overall
completeness pass.

General unsupported-claim rate requires blinded atomic-claim review:

```text
unsupported checkable atomic claims / all checkable atomic claims
```

The generated claim-audit worksheet supports this review. Claims about clinical usefulness,
actionability, comprehension, or trust require evaluation by appropriate clinical end users.

## Current stage-coverage interpretation

Current routed-faller reports state both classification decisions but show only Stage-2 evidence.
Consequently:

- displayed-stage coverage can be 100%;
- Stage-1/Stage-2 decision coverage can be 100%;
- full-pipeline stage-evidence coverage is 50% for a routed faller;
- full-pipeline completeness fails until Stage-1 evidence is added.

This distinction directly represents the reviewer concern that the final faller classification
depends on both stages.

## Validation

Mutation tests are located in `tests/test_template_metrics.py`. They verify detection of missing
factors, direction swaps, altered values and scales, invented and duplicated factors, empty
interpretations, causal claims, contradictory outcomes, rank reversals, malformed table rows, and
forbidden probability content.

The experiment-local metric code is implemented in `template_experiment/evaluation/` and does not
reuse metric functions from the existing `explanation/evaluation` pipeline. It continues to use the
production explanation builder as the source of model-derived ground truth.

## Final-experiment checklist

Before final generation:

- export and freeze the final ML explanation artifacts;
- freeze `feature_map.csv`, the 0.90 coverage threshold, and builder code;
- freeze the LLM prompt, provider, model, temperature, terminology, and template version;
- generate both methods from the same `patient_info` object;
- freeze automatic metric definitions and the prose/synthesis rubrics;
- retain raw outputs, generation failures, token use, latency, and method versions;
- run mutation tests before calculating final results.

See [`TEMPLATE_EXPERIMENT_PLAN.md`](../TEMPLATE_EXPERIMENT_PLAN.md) for the full protocol.
