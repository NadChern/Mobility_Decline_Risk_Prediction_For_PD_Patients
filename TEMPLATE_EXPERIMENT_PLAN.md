# Deterministic Template vs LLM Explanation Experiment

## 1. Purpose

This experiment will determine whether an LLM adds measurable value beyond a deterministic
template when both methods receive exactly the same model-derived evidence.

The comparison is intended to answer the reviewer concern that the current LLM primarily
reformats fixed SHAP evidence into a predefined document. The experiment will not assume that
the LLM is superior. If it does not demonstrate an advantage, the paper will describe it as an
implementation choice rather than a central scientific contribution.

## 2. Primary research question

When supplied with the same prediction label, selected SHAP factors, feature values, factor
directions, rank order, and feature metadata, how do deterministic-template and LLM-generated
explanations compare in:

1. Fidelity
2. Completeness
3. Unsupported information
4. Synthesis quality
5. Complexity, scalability, and flexibility

## 3. Experimental conditions

### 3.1 Shared evidence packet

Both conditions must consume the output of
`explanation.explanation_builder.build_patient_explanation_data_full()` without changing:

- the patient or prediction;
- the prediction stage being explained;
- the SHAP coverage threshold;
- selected factors;
- factor direction and order;
- patient values;
- display names and scale descriptions; or
- missing/unrateable-factor filtering.

For auditability, save a canonical JSON representation and SHA-256 hash of each patient's
displayable evidence packet. Store the same hash beside both generated explanations.

### 3.2 Deterministic-template condition

Use one generic renderer that:

- renders arbitrary-length supporting and opposing factor lists;
- copies factual values and scale descriptions verbatim;
- creates the same prediction, table, interpretation, and clinical-note sections;
- selects the first 3-5 supporting and first 2-3 opposing factors for the interpretation; and
- uses no patient-specific or feature-combination-specific rules.

The initial baseline is implemented in `template_experiment/renderer.py`. It is deliberately a
strong generic baseline rather than a separate template for every feature combination.

### 3.3 LLM condition

Use the current production provider/model at temperature 0 for the primary comparison. Generate
only one primary output per patient. Repeated generations at temperatures 0 and 0.3 remain a
separate stability analysis and should not be treated as additional independent patients.

For the cleanest ablation, add a second LLM condition in which the prediction, tables, and
clinical note are rendered deterministically and the LLM generates only the Model Interpretation
paragraph. This isolates the component where the LLM might add value.

### 3.4 Optional stronger deterministic condition

Add a rule-enriched template that groups features into a frozen, version-controlled set of
clinical domains such as gait/mobility, motor impairment, motor complications, cognition,
autonomic symptoms, mood, and demographics. Freeze these mappings before testing.

This produces three useful comparison arms:

1. Generic deterministic template
2. Domain-enriched deterministic template
3. LLM synthesis

## 4. Cohort and sampling

Use all 351 patients in the saved test artifacts where feasible. The current distribution is:

- No Fall: 230
- Rare Fall: 92
- Recurrent Fall: 29

Report both:

- macro-averaged results, giving each prediction category equal weight; and
- prevalence-weighted results, reflecting the saved test cohort.

If LLM cost prevents evaluation of all patients, predefine a stratified random sample larger
than the current 10 patients per category and retain the seed. Perform the deterministic
condition on all patients even if the paired primary analysis uses a subset.

The unit of analysis is the patient, not an individual LLM generation.

## 5. Primary metrics

### 5.1 Fidelity

Calculate against the shared evidence packet:

- factor precision;
- factor recall;
- direction accuracy;
- within-direction rank correlation;
- patient-value exact-match rate;
- scale-description exact-match rate; and
- prediction-label accuracy.

Also perform claim-level review of the Model Interpretation for incorrect clinical meaning,
incorrect interactions, and incorrect causal or severity statements. Table-level feature matching
alone is not sufficient to establish prose fidelity.

### 5.2 Completeness

Keep these concepts separate:

- **Evidence completeness:** proportion of all supplied displayable factors reproduced.
- **Synthesis coverage:** proportion of the required top supporting and opposing factors represented
  in the interpretation.
- **Stage completeness:** whether the evidence shown supports every prediction stage claimed by the
  document.
- **Generation completion rate:** proportion of requested outputs successfully produced. Report this
  as reliability, not content completeness.

Before running the comparison, decide whether routed patients should show both Stage 1 and Stage 2
evidence or whether the document will explicitly state that its factor tables explain only the
severity stage. Apply the decision identically to both methods.

### 5.3 Unsupported information / hallucination

Report separate rates and raw counts for:

- features absent from the packet;
- altered or invented patient values;
- altered scale descriptions;
- incorrect factor directions;
- unsupported clinical interpretations;
- unsupported feature interactions;
- causal claims not licensed by the evidence; and
- contradiction of the predicted class.

Define the denominator for every rate in advance. When no claims are present, report the raw count
and the predefined zero-denominator convention.

### 5.4 Synthesis

Use a frozen rubric that scores:

- prioritization of the most influential factors;
- integration of multiple factors rather than simple enumeration;
- grouping into defensible higher-level concepts;
- contrast between supporting and opposing evidence;
- coherence and logical flow;
- non-redundancy; and
- preservation of factual constraints.

Primary synthesis assessment should use blinded, randomized paired outputs rated by at least two
trained annotators. The raters need not be clinicians for linguistic synthesis, but claims about
clinical usefulness, actionability, trust, or comprehension require appropriate clinical end users.

Record inter-rater agreement and resolve neither disagreements nor rubric definitions after method
labels are revealed. Embedding similarity, readability indices, and an LLM judge may be reported as
secondary analyses but cannot substitute for the primary blinded assessment.

### 5.5 Complexity, scalability, and flexibility

Record prospectively:

- number of generic templates;
- number of manually authored decision rules;
- number of feature-domain mappings;
- implementation source lines and cyclomatic complexity, interpreted cautiously;
- developer time required for initial implementation and each requested change;
- output failure and fallback rates;
- per-explanation latency and monetary cost;
- dependencies on external APIs and model versions; and
- number of code, rule, or prompt modifications required by held-out cases.

Do not equate one prompt with one rule: prompt instructions and embedded examples are manual logic.
Report their size and maintenance burden alongside template rules.

## 6. Unseen-combination experiment

The current 351 patients contain 325 distinct exact displayed feature sets, so feature combinations
are already highly diverse. A generic loop-based template can handle new combinations without one
rule per combination; the experiment must not intentionally make the template combination-specific.

Create a development/held-out split in which the held-out set contains signed feature combinations
not present in development. Freeze template code, domain mappings, and the LLM prompt before opening
held-out results.

Include stress cases covering:

- unseen combinations of familiar features;
- a newly introduced feature with metadata;
- only one factor direction populated;
- unusually short and long factor lists;
- missing or unable-to-rate factors; and
- renamed display labels or changed output-length requirements.

For each case, record whether the method succeeds without modification. Distinguish rendering a new
combination from producing a genuinely useful synthesis of it.

## 7. Statistical analysis

- Compare methods on paired patient-level observations.
- Report estimates with 95% bootstrap confidence intervals.
- For fidelity and completeness, predefine equivalence or non-inferiority margins.
- For paired binary errors, use McNemar's exact test where appropriate.
- For paired ordinal synthesis scores, use a paired permutation test or Wilcoxon signed-rank test.
- Correct or clearly qualify inference across the five metric families.
- Report category-specific results, macro averages, weighted averages, and raw error counts.
- Do not treat repeated generations for one patient as independent samples.

## 8. Bias controls and reproducibility

- Freeze the evidence schema, template, prompt, metrics, lexicon, and synthesis rubric before the
  final evaluation.
- Do not revise lexical mappings after inspecting which method produced an apparent error.
- Randomize output order and mask method identity for human ratings.
- Retain raw outputs, evidence hashes, model/provider identifiers, temperature, timestamps, latency,
  token use, and generation failures.
- Run mutation tests demonstrating that each automatic metric detects deliberately introduced
  omissions, direction swaps, value changes, hallucinated claims, and malformed sections.
- Keep experimental outputs separate from the existing `evaluation_results/` until the protocol is
  frozen.

## 9. Implementation layout

```text
template_experiment/
  README.md                 # package purpose and usage
  __init__.py               # public deterministic API
  renderer.py               # generic deterministic baseline
  evaluation/README.md      # planned comparison artifacts and scripts
  outputs/.gitkeep          # generated outputs go here and should be treated as artifacts
```

Planned additions after protocol approval:

```text
template_experiment/
  evidence.py               # canonical evidence serialization and hashing
  generate_dataset.py       # paired template/LLM output generation
  domain_rules.yaml         # optional frozen domain-enriched baseline
  evaluation/
    compare_factual.py
    compare_synthesis.py
    compare_scalability.py
    analyze_results.py
tests/
  test_template_renderer.py
  test_template_metrics.py
```

## 10. Execution phases

> **Deferred-version note (24 August 2026):** The current 30-patient paired Gemini/template file is
> a technical pilot only. The saved Gemini outputs use the historical `Mild Falls` / `Moderate Falls`
> wording, while the current template uses `Rare Fall` / `Recurrent Fall`; the deterministic template
> may also be revised before the final experiment. Do not calculate or report paper-level comparative
> metrics from this pilot file. After the template and output terminology are finalized, freeze the
> evidence/output contract and regenerate both the LLM and deterministic reports for the same patient
> cohort before running the final comparison.

### Phase 1 — Freeze protocol

- [ ] Resolve the Stage 1/Stage 2 completeness decision.
- [ ] Finalize primary and secondary outcomes.
- [ ] Define equivalence margins and zero-denominator conventions.
- [ ] Freeze the synthesis rubric and rater procedure.

### Phase 2 — Build paired generators

- [x] Scaffold a generic deterministic baseline.
- [x] Build the 30-patient Gemini/template paired pilot dataset from saved Gemini outputs.
- [ ] Add evidence serialization and hashes.
- [ ] Add the LLM-interpretation-only ablation.
- [ ] Add the optional domain-enriched template.
- [ ] Add automated tests and mutation tests.

### Phase 3 — Pilot without hypothesis changes

- [ ] Run a small technical pilot to find parser or logging defects.
- [ ] Repair implementation defects only; do not tune the rubric toward a preferred method.
- [ ] Version and freeze the final protocol.

### Phase 4 — Final evaluation

- [ ] Finalize the deterministic template and outcome terminology.
- [ ] Regenerate both LLM and template reports from the same frozen evidence/output contract.
- [ ] Generate paired outputs on the frozen cohort.
- [ ] Run automatic factual metrics.
- [ ] Conduct blinded synthesis ratings.
- [ ] Run held-out-combination and flexibility tests.
- [ ] Calculate paired statistics and confidence intervals.

### Phase 5 — Paper revision

- [ ] Report ties and template advantages as clearly as LLM advantages.
- [ ] Limit claims to measured outcomes.
- [ ] If synthesis/flexibility advantages are weak, reposition the LLM as an implementation choice.
- [ ] If synthesis is better, describe the factual shell as deterministic and the LLM contribution
  specifically as constrained synthesis.

## 11. Decision rule for the paper

- If the LLM improves blinded synthesis while remaining non-inferior on fidelity and completeness,
  claim a constrained synthesis benefit—not improved clinical utility.
- If it improves flexibility under prospectively defined requirement changes, claim that specific
  maintenance/adaptation advantage with the measured costs.
- If it ties or loses across outcomes, use the deterministic template in the primary system or retain
  the LLM only as an optional implementation component.
- Claims of improved comprehension, actionability, usefulness, or trust remain out of scope unless a
  suitable user study is added.
