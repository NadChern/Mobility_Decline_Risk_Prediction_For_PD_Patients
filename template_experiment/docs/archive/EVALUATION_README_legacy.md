# Evaluation workspace

This directory holds scripts for the paired deterministic-template versus LLM comparison.
Generated results are written under `template_experiment/outputs/`, not mixed into the existing
single-method `evaluation_results/` directory.

## Build the Gemini paired dataset

From the project root:

```bash
./.venv/bin/python -m template_experiment.evaluation.build_gemini_pairs
```

This reads the existing Gemini generations, selects temperature 0/run 1, generates a deterministic
report for the same patient IDs, and writes:

```text
template_experiment/outputs/gemini/paired_explanations.csv
```

The original Gemini results are read-only inputs and remain unchanged. This script performs no LLM
calls and calculates no comparison metrics.

### Pilot compatibility note

The saved Gemini file uses the earlier outcome wording `Mild Falls` / `Moderate Falls`, whereas the
current pipeline and deterministic renderer use `Rare Fall` / `Recurrent Fall`. The paired builder
preserves that historical text in `gemini_explanation_raw` and creates `gemini_explanation` as a
comparison copy with only the outcome labels and related headings normalized to current wording. All
30 saved feature lists, directions, ranks, and six-decimal SHAP magnitudes match the evidence
reconstructed from the current artifacts. This normalization supports pilot predicted-class checks,
but final paper outputs must still be regenerated prospectively under one frozen output contract.

**Deferred action:** revise and finalize the deterministic template first. Then freeze the shared
evidence/output contract and regenerate both LLM and template reports for the same IDs. Treat the
current `paired_explanations.csv` as a disposable pilot artifact and do not use it for paper-level
comparative metrics.

## Metric-family boundary

The comparison keeps four concepts separate:

- **Fidelity:** whether present content is correct relative to supplied evidence.
- **Completeness:** whether all required content is present and non-empty.
- **Unsupported information:** whether the report adds, duplicates, or alters evidence.
- **Structure compliance:** whether the document follows the required format. This remains a
  separate family and is not folded into completeness.

The four comparison modules use parsing and scoring code located entirely inside
`template_experiment/evaluation/metric_utils.py`; they do not reuse the existing
`explanation/evaluation` metric implementation. They do use the production explanation builder to
reconstruct model-derived ground truth for this pilot.

## Run the fidelity pilot

For a preliminary presentation, the comparable evidence-bearing table metrics can be calculated
without treating the legacy label difference as an error:

```bash
./.venv/bin/python -m template_experiment.evaluation.fidelity_comparison
```

This evaluates strict outcome correctness, exact factor-set agreement and Jaccard, conditional and
strict direction fidelity, conditional and end-to-end value/scale fidelity, pairwise and exact
within-direction SHAP order, stage/routing correctness, and frozen-lexicon prose-feature grounding.
The prose measure is traceability only; it is not presented as general semantic claim fidelity.

## Run the completeness comparison

```bash
./.venv/bin/python -m template_experiment.evaluation.completeness_comparison
```

This checks factor recall, required section presence and non-empty content, classification-field
coverage, expected table-cell coverage, current displayed-stage coverage, full-pipeline stage
coverage, and frozen-lexicon coverage of required supporting and opposing factors in the
interpretation. It reports both current-contract completeness and full-pipeline completeness. Under
the current report design, routed fallers necessarily receive only 50% full-pipeline stage-evidence
coverage because their Stage-1 evidence is absent. Lexical interpretation-coverage screens are
reported separately and do not determine the overall completeness pass, because broader category
wording can be valid synthesis without repeating every exact factor label.

## Run the unsupported-information comparison

```bash
./.venv/bin/python -m template_experiment.evaluation.unsupported_information_comparison
```

This reports factor precision, extra and duplicate factor rates, altered-value and altered-scale
rates, unsupported frozen-lexicon prose-feature mentions, prohibited causal-language screening, raw
counts, and an all-clear rate. It also writes a randomized `prose_claim_audit_blinded.csv` plus a
separate identity key. Blinded raters must use that worksheet to calculate general unsupported-claim
rate; regular expressions cannot establish the semantic support of arbitrary clinical prose.

## Run the structure-compliance comparison

```bash
./.venv/bin/python -m template_experiment.evaluation.structure_comparison
```

This paired, experiment-local checker measures title format, section ordering, stage-specific
template selection, table-header and row format, row numbering, supporting-table order, forbidden
content, an arithmetic structure score, and an all-components compliance rate. It evaluates format
only and remains separate from fidelity, completeness, and unsupported information.

## Mutation validation

`tests/test_template_metrics.py` verifies that the metrics detect missing factors, direction swaps,
value and scale alterations, invented and duplicated factors, empty interpretation prose, causal
claims, contradictory outcomes, and rank reversals.

Planned components are documented in the repository-level `TEMPLATE_EXPERIMENT_PLAN.md`.

Before final data generation, freeze and version:

- the shared evidence schema;
- deterministic renderer version;
- LLM prompt, provider, model, and temperature;
- automatic metric definitions;
- synthesis rubric; and
- unseen-combination split.
