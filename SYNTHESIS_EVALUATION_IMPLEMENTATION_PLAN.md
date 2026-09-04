# Synthesis Evaluation: Full Implementation Plan

## Document status

- Status: implementation complete; external generation, judging, and blinded author review pending
- Scope: evaluation of the `Model Interpretation` section only
- Primary comparison: taxonomy-conditioned LLM versus domain-aware deterministic template
- Existing map-free Gemini reports: preserved as a separate retrospective dataset
- Unit of analysis: patient, not report sentence, factor pair, judge call, or repeated generation
- Intended use: reviewer-facing evidence about structural synthesis and configuration flexibility
- Environment status: deterministic pipeline complete; `GOOGLE_API_KEY` and `OPENROUTER_API_KEY`
  were not available, so no external scores were fabricated

This plan replaces the current weak synthesis claim based mainly on grouping extent, lexical
diversity, and compression. It does not replace the existing fidelity, completeness, unsupported-
information, or structure evaluations. Those metric families remain separate.

## 1. Objective and claim boundary

### 1.1 Objective

Determine whether an LLM adds measurable value beyond a strong deterministic template when
converting the same patient-specific model evidence into a short interpretation.

The evaluation separates three questions:

1. **Taxonomy-conditioned structural synthesis:** When both methods receive the same taxonomy, can
   the LLM preserve the deterministic template's structural performance?
2. **Map-free flexibility:** Without receiving the task-specific taxonomy, how closely does the LLM
   agree with the researcher-defined reference, what alternative groupings does it create, and how
   much explicit mapping maintenance does it avoid?
3. **Separate-model judge assessment:** When method identity is hidden, do separate judge models
   prefer one interpretation for organization, coherence, traceability, contrast, concision, or
   naturalness?

### 1.2 What this evaluation does not establish

This study does not establish:

- clinical usefulness;
- clinical correctness of inferred categories;
- improved clinician comprehension, trust, or decision-making;
- safety for autonomous clinical use; or
- that the LLM has never encountered a held-out clinical concept during pretraining.

Use **“factors held out from task configuration”**, not “unseen to the model.”

### 1.3 Expected honest outcomes

Any of the following is an acceptable result:

- the taxonomy-conditioned LLM is noninferior to the domain template;
- the pilot is too imprecise to establish noninferiority;
- the domain template outperforms the LLM structurally;
- the map-free LLM reduces explicit mapping work but requires more review;
- the judge prefers the LLM on linguistic criteria only; or
- the LLM becomes an implementation choice rather than a scientific contribution.

## 2. Current repository baseline

The current 30-patient pilot contains three arms:

- generic deterministic template;
- domain-aware deterministic template; and
- map-free Gemini output.

The former grouping-extent/Distinct-2 synthesis outputs have been retired and overwritten with a
pairwise structural report. Current retrospective map-free results are:

- macro pairwise precision: Gemini 67.2%, domain template 100%, generic template N/A;
- macro pairwise recall: Gemini 63.6%, domain template 100%, generic template 0%;
- macro pairwise F1: Gemini 55.6%, domain template 100%, generic template 0%;
- micro pairwise F1: Gemini 54.1%, domain template 100%, generic template 0%;
- exact predicted/reference pair-set agreement: Gemini 9/30, domain template 30/30, generic
  template 3/30; and
- conditional explicit-membership recall for Gemini: 84.2% across 20 evaluable reports.

The existing map-free reports must not be compared with the taxonomy-driven template and described
as a same-information Part 1 experiment. They belong to Part 2A only.

These current values use the method-agnostic explicit-claim parser. It achieved 16/16 exact relation
matches on the frozen mutation corpus; a blinded manual audit of real outputs is still required.

## 3. Final experimental architecture

### 3.1 Dataset A: taxonomy-conditioned structural synthesis

Purpose: Part 1 comparison under equal domain information.

Dataset A contains:

- the same 30 frozen patient evidence packets;
- newly generated taxonomy-conditioned Gemini interpretations;
- newly rendered, explicitly traceable domain-template interpretations;
- the same selected factors, values, scales, ranks, directions, and stages;
- the same factor-selection policy;
- the same factor-to-domain taxonomy and display labels;
- the same safety and grounding constraints; and
- the same sentence/word requirements.

Dataset A does not reuse the current map-free Gemini interpretations.

### 3.2 Dataset B: frozen retrospective map-free synthesis

Purpose: Part 2A reference-taxonomy agreement and disagreement analysis.

Dataset B contains:

- the existing 30 Gemini interpretations;
- the exact historical production prompt and model metadata;
- no explicit factor-to-domain map in the generation prompt; and
- the frozen researcher-defined taxonomy used only as an evaluation reference.

Do not regenerate Dataset B after reviewing disagreements. Any revised-prompt generation is a new,
separately labelled experiment.

### 3.3 Dataset C: prospective held-out task-configuration cases

Purpose: Part 2B flexibility and maintenance-burden evaluation.

Dataset C contains predefined synthetic or artifact-derived evidence packets with factors and
combinations held out from the task configuration. These cases test explanation-system adaptation,
not predictive-model validity.

Dataset C must include:

- a new factor belonging to an existing domain;
- multiple new factors belonging to one existing domain;
- a new factor with an ambiguous domain;
- a new factor from a previously unrepresented domain;
- new combinations of known and held-out factors;
- a held-out factor that remains a singleton;
- a held-out factor on the opposing evidence side;
- a one-sided evidence packet;
- short and long factor lists without duplicated factor identities; and
- a case with missing or unable-to-rate factor metadata, if supported by the report contract.

### 3.4 Comparator roles

- **Primary deterministic comparator:** domain-aware template.
- **Primary generative method:** Gemini using the frozen condition-specific prompt.
- **Sanity baseline:** generic non-grouping template.

The generic template is useful for showing that grouping metrics distinguish listing from
integration. It must not be used as the main basis for claiming LLM value.

## 4. Freeze and provenance protocol

### 4.1 Freeze order

1. Freeze the existing map-free reports as Dataset B.
2. Freeze the 30 patient IDs and canonical evidence packets.
3. Freeze the taxonomy version and category labels.
4. Freeze the explicit interpretation output contract.
5. Freeze all structural metric definitions and edge cases.
6. Freeze the parser and its validation set.
7. Freeze the domain-template implementation.
8. Freeze the taxonomy-conditioned Part 1 prompt.
9. Freeze the Part 2B map-free prompt and held-out cases.
10. Freeze the judge rubric, prompts, models, repetition count, and aggregation rules.
11. Generate Dataset A and Dataset C outputs.
12. Run the frozen analyses without post-result rule changes.

### 4.2 Frozen manifest

Create a machine-readable manifest containing:

- protocol version;
- Git commit or worktree-state identifier;
- patient IDs and sampling seed;
- evidence-packet SHA-256 hashes;
- taxonomy version and file hash;
- prompt versions and hashes;
- template implementation hash;
- parser and evaluation-code hashes;
- model provider, model/version, temperature, and generation date;
- held-out case IDs and hashes;
- judge model/version and settings; and
- Python and dependency versions.

Do not silently replace an artifact after it is frozen. Create a new protocol version instead.

## 5. Common record and output contract

### 5.1 Per-report record

Each comparison record must include at least:

```json
{
  "dataset": "A",
  "patient_id": "...",
  "prediction": "Rare Fall",
  "stage": 2,
  "evidence_hash": "...",
  "selected_factors": [
    {
      "factor": "FOG (Freezing of Gait)",
      "domain": "gait_mobility",
      "domain_label": "gait and mobility",
      "direction": "supports_prediction",
      "rank": 1,
      "value": "0",
      "scale": "..."
    }
  ],
  "method": "taxonomy_conditioned_llm",
  "model_or_template_version": "...",
  "interpretation": "..."
}
```

### 5.2 Explicit group membership

Both Dataset A methods must state group membership explicitly. The deterministic template uses the
canonical form:

> gait and mobility (FOG, Gait, Mobility Summary)

Requirements:

- every produced group names its category and members, either with the canonical parenthetical form
  or an unambiguous connector such as “represented by” or “supported by”;
- members are written using canonical factor display names;
- factors are grouped only within the same patient, stage, and evidence side;
- a group requires at least two factors;
- singletons remain individually named;
- implied membership receives no traceability credit; and
- the LLM is not given the template's exact sentences.

The domain template must be regenerated after implementing this format.

### 5.3 Length matching

Use the same content and length requirements for the two Dataset A methods.

Before generation, freeze:

- allowed sentence range;
- required representation of supporting and opposing evidence; and
- required explicit group membership.

The historical map-free prompt has no maximum word count, so the taxonomy-conditioned twin must not
introduce one. Both use the same 2–4-sentence instruction. If a material length imbalance remains,
disclose it as a judge-design limitation rather than introducing output-length statistics as a
synthesis metric.

## 6. Group-claim extraction and parser validation

### 6.1 One extraction procedure

The same parser must process the LLM and template interpretations. The evaluator must not infer
hidden template membership from the taxonomy using method-specific code.

For each group claim, extract:

- patient ID;
- stage;
- evidence side;
- category phrase;
- normalized category, when applicable;
- explicitly named member factors;
- whether membership was complete, partial, or implied; and
- the source character span for auditability.

For Dataset B, category-label normalization is secondary. Same-group factor pairs are primary so
synonymous labels do not count as different partitions.

### 6.2 Parser validation corpus

Create hand-authored cases covering:

- a correct two-factor group;
- a correct three-or-more-factor group;
- a partial member list;
- a category with no explicit members;
- a singleton category claim;
- an incorrect merge;
- a factor on the wrong evidence side;
- a nonselected factor;
- two groups in one sentence;
- nested parentheses in factor names;
- synonymous category phrases;
- a broad or previously unseen category phrase;
- no group when a reference group exists; and
- no group when no grouping opportunity exists.

### 6.3 Manual extraction audit

Blind method identity and manually annotate all Dataset A and Dataset B interpretations if feasible.
At minimum, audit every detected disagreement and a random sample of agreements.

Report parser agreement with manual annotations for:

- exact group-member relation extraction;
- evidence-side assignment;
- factor detection; and
- exact pair-set recovery.

Target acceptance criterion: at least 95% exact relation-level agreement. If the parser does not
meet the frozen criterion, use blinded manual annotations as the primary extraction source and the
parser as a reproducibility aid.

## 7. Part 1: taxonomy-conditioned structural synthesis

### 7.1 Research question

When the LLM and domain template receive the same evidence and taxonomy, is the LLM no more than a
predefined acceptable amount worse on taxonomy-defined structural grouping?

### 7.2 Eligible reference groups

A reference group exists only when at least two selected factors:

- have the same supplied domain;
- occur on the same evidence side; and
- occur within the same stage.

The gold pair identity within a report is:

```text
(stage, evidence_side, unordered_factor_A, unordered_factor_B)
```

For pooled calculations, use:

```text
(patient_id, stage, evidence_side, unordered_factor_A, unordered_factor_B)
```

Including `patient_id` prevents the same factor pair in different patients from collapsing into a
single pooled observation.

### 7.3 Pairwise grouping metrics

For report `i`, let:

- `P_i` be the set of factor pairs grouped by the generated interpretation; and
- `G_i` be the set of eligible reference pairs defined by the supplied taxonomy.

Calculate:

```text
precision_i = |P_i ∩ G_i| / |P_i|
recall_i    = |P_i ∩ G_i| / |G_i|
F1_i        = 2 * precision_i * recall_i / (precision_i + recall_i)
```

#### Frozen edge cases

| Gold pairs | Predicted pairs | Precision | Recall | F1 | Exact pair-set |
| --- | --- | ---: | ---: | ---: | ---: |
| Present | None | N/A | 0 | 0 | 0 |
| None | None | N/A | N/A | N/A | 1 |
| None | Present | 0 | N/A | 0 | 0 |
| Present | Present | Formula | Formula | Formula | `P_i == G_i` |

Reports with no gold and no predicted pairs must be reported separately so exact agreement is not
inflated by reports with no grouping opportunity.

### 7.4 Macro and micro reporting

Report:

- macro precision: mean per-report precision over evaluable reports;
- macro recall: mean per-report recall over evaluable reports;
- macro F1: mean per-report F1 over evaluable reports;
- micro precision: pooled true-positive pairs divided by pooled predicted pairs;
- micro recall: pooled true-positive pairs divided by pooled gold pairs;
- micro F1 from pooled precision and recall;
- exact pair-set agreement overall;
- exact pair-set agreement among reports with a gold grouping opportunity;
- spurious-grouping rate among reports with no gold opportunity; and
- the evaluable numerator and denominator for every result.

Macro F1 is the mean of per-report F1 values. It need not equal the harmonic mean of macro precision
and macro recall.

### 7.5 Conditional explicit-membership recall

This metric measures traceability without double-penalizing an entirely missed group.

For each group claim actually produced:

```text
explicit_membership_recall =
    correct intended members explicitly named / intended members represented by that claim
```

Examples:

- Eligible group omitted entirely: grouping recall penalizes the omission; explicit-membership
  recall is N/A for the missing claim.
- “gait and mobility” with no members: group claim detected; explicit-membership recall is 0.
- “gait and mobility (FOG, Gait)” when the intended claimed group contains FOG, Gait, and Postural
  Instability: explicit-membership recall is 2/3.

Report claim-level micro recall, report-level macro recall, and evaluable denominators. Incorrect or
extra members are penalized through pairwise precision and unsupported-merge reporting rather than
being silently absorbed into this recall measure.

### 7.6 Contrastive linkage

Apply only when both supporting and opposing factors are supplied:

- `0`: one side is omitted, placed on the wrong side, or contradicted;
- `0.5`: both sides are represented but only as separate lists or statements; and
- `1`: both sides are correctly represented and explicitly related to the model output through a
  contrastive relationship.

A connective such as “however” is insufficient unless both evidence sides are correct. Because the
current prompt explicitly requests “did not outweigh” language, treat this as a structural
compliance metric with a likely ceiling—not proof that evidence was clinically weighed.

### 7.7 Composite score policy

Do not headline a structural composite.

Primary reporting consists of:

- pairwise precision;
- pairwise recall;
- pairwise F1;
- conditional explicit-membership recall; and
- contrastive linkage reported separately.

If a composite is retained, label it exploratory, predefine its weighting, place it in supplementary
material, and never use it as the sole basis for a conclusion.

### 7.8 Primary endpoint and noninferiority analysis

Proposed primary endpoint:

> Patient-level pairwise grouping F1 among reports with at least one eligible reference grouping
> opportunity.

For each eligible patient:

```text
D_i = F1_LLM,i - F1_template,i
```

Proposed noninferiority rule:

- define and justify margin `Delta` before Dataset A generation;
- provisional margin for protocol discussion: `Delta = 0.05`;
- estimate the paired mean F1 difference;
- calculate a patient-level 95% confidence interval, preferably with a stratified paired bootstrap;
- conclude noninferiority only if the lower confidence bound is greater than `-Delta`.

Do not choose or revise `Delta` after examining Dataset A results. If the confidence interval is too
wide, report that the pilot lacked sufficient precision to establish noninferiority.

Secondary analyses may include a paired permutation test or Wilcoxon signed-rank test, paired effect
sizes, individual patient differences, and prediction-category strata.

## 8. Part 2A: retrospective map-free analysis

### 8.1 Research question

Without an explicit task-specific factor-to-domain dictionary, how does the frozen map-free LLM
partition the selected factors relative to the researcher-defined taxonomy?

### 8.2 Primary reporting

Report:

- macro and micro reference-taxonomy pairwise precision;
- macro and micro reference-taxonomy pairwise recall;
- macro and micro reference-taxonomy pairwise F1;
- exact pair-set agreement;
- reports with no grouping opportunity;
- grouping stability only if separately generated repeated runs exist;
- existing factor-to-category membership precision and recall, clearly labelled as different units;
- all evaluable denominators; and
- full disagreement-category counts.

Use “reference-taxonomy agreement” or “concordance,” never “accuracy.” The taxonomy is a researcher-
defined operational reference, not unique clinical ground truth.

### 8.3 Disagreement worksheet

Assign every disagreement to one predefined category:

1. plausible alternative abstraction;
2. granularity difference;
3. reference-taxonomy ambiguity;
4. missed grouping opportunity;
5. unsupported merge; or
6. uninterpretable extraction.

For every row, record:

- patient and evidence side;
- selected factors;
- reference grouping;
- LLM grouping and exact text span;
- disagreement category;
- author rationale;
- whether a separate-model judge was used;
- judge output, if any; and
- whether adjudication changed after parser review.

Publish counts and representative paraphrased examples that both favor and disfavor the LLM. Author
adjudication must be labelled as author-defined analysis, not independent clinical validation.

## 9. Part 2B: prospective held-out task-configuration flexibility

### 9.1 Research question

Can a frozen map-free LLM produce explicit, defensible grouping for factors held out from task
configuration without prompt changes, while preserving fidelity and avoiding unsupported content?

### 9.2 Predefine every held-out case

Before generation, record for each case:

- case ID and rationale;
- complete evidence packet;
- which factors are held out from task configuration;
- expected evidence side and stage;
- reference grouping;
- acceptable alternative groupings, if any;
- groupings defined as unsupported;
- template changes expected;
- frozen LLM prompt hash;
- scoring rules; and
- case-specific ambiguity notes.

### 9.3 Fair maintenance accounting

For a held-out factor assigned to an existing domain, the domain template usually requires:

- one new factor-to-domain mapping entry;
- zero grouping-algorithm changes; and
- zero new category labels.

Only count a new label or other artifact when the held-out case genuinely introduces a new domain.
Evaluation-parser phrases are not production-template maintenance and must not be added to the
template's edit count.

Measure prospectively:

| Method | Maintenance measures |
| --- | --- |
| Domain template | Mapping entries, category labels, algorithm/code changes, review time |
| Map-free LLM | Prompt changes, manual corrections, regeneration count, review time |
| Generic template | Rendering success only; grouping maintenance is not applicable |

Log time from a predefined start to stop rule. Do not reconstruct review time retrospectively.

### 9.4 Per-case outcomes

Report for every method and case:

- renders without configuration changes;
- names every required factor;
- groups the held-out factor without configuration changes;
- explicit-membership traceability;
- pairwise reference agreement;
- accepted-alternative grouping result;
- unsupported-merge count;
- omitted-factor count;
- altered-value or scale count;
- direction correctness;
- total prompt/map/code changes;
- manual correction count;
- review time;
- generation latency and cost, if available; and
- final pass/fail under the frozen case rule.

### 9.5 Per-case success rule

A map-free LLM case succeeds only if it:

- uses the frozen prompt without task-specific edits;
- renders a valid interpretation;
- explicitly groups or correctly leaves individual the held-out factor according to the frozen
  reference or predefined acceptable alternatives;
- preserves required factors, values, scales, and evidence directions;
- adds no unsupported factors or causal claims; and
- requires no manual correction for the scored output.

Report the success rate with its denominator and per-case results. Do not claim unlimited
scalability. The likely defensible conclusion is a tradeoff between fewer explicit mapping updates
and greater review of inferred groups.

## 10. Part 3: blinded separate-model judge evaluation

### 10.1 Role and limitations

Judge results are secondary model-based evidence. They do not replace clinician or human evaluation
and must not be called independent validation.

### 10.2 Fixed judge panel and settings

The separate-model judge panel is fixed to two OpenRouter models:

| Judge | OpenRouter model ID | Role |
| --- | --- | --- |
| Z.ai GLM 5.3 Flash | `z-ai/glm-5.3-flash` | Blinded Part 1 paired scoring and secondary Part 2 plausibility scoring |
| Qwen3.8 Flash | `qwen/qwen3.8-flash` | Blinded Part 1 paired scoring and secondary Part 2 plausibility scoring |

Both judges use temperature 0, both A/B presentation orders, and three repetitions per order. The
model IDs and execution settings live under `evaluation.synthesis_judges` in
`explanation/config.yaml`. Any model substitution creates a new protocol version and must not be
silently pooled with these results.

These models come from different model families than the Gemini explanation generator. They remain
**separate-model judges**, not independent clinical validators.

### 10.3 Part 1 paired judge task

Primary comparison:

- taxonomy-conditioned Gemini interpretation; versus
- explicitly traceable domain-template interpretation.

Provide the judge:

- the shared evidence packet;
- the reference taxonomy;
- anonymous Output A and Output B; and
- the frozen rubric.

Do not provide method identities.

### 10.4 Judge criteria

Use anchored 1–5 ratings for:

1. evidence organization;
2. coherence;
3. factor traceability;
4. contrastive integration;
5. concision and non-redundancy; and
6. naturalness.

Also require:

- Output A preferred, Output B preferred, or tie; and
- a short criterion-based rationale.

Naturalness must not be interpreted as clinical usefulness.

### 10.5 Bias controls

For each patient:

- randomly assign method to A/B;
- evaluate the reversed B/A order;
- run three repetitions per order when feasible;
- use both configured judge-model families: `z-ai/glm-5.3-flash` and
  `qwen/qwen3.8-flash`;
- require structured JSON;
- store raw responses, not only parsed scores;
- record exact model/version, temperature, date, and prompt hash; and
- do not reveal the key until scoring and parsing are complete.

If using 30 patients, two orders, three repetitions, and two judges, the experiment produces 360
judge calls but still only 30 independent patient comparisons.

### 10.6 Aggregation and pseudoreplication control

Aggregate repeated and order-reversed calls within patient and judge before inferential analysis.
Never treat judge calls as independent samples.

Report:

- per-method mean, median, and interquartile range for each criterion;
- LLM/template/tie preference proportions;
- order consistency;
- within-judge repeat consistency;
- cross-judge raw agreement;
- weighted Cohen's kappa for ordinal scores when appropriate;
- Cohen's kappa for winner categories when appropriate; and
- patient-level distributions and paired differences.

If agreement is low, report the judge evidence as inconclusive rather than averaging it into an
authoritative score.

### 10.7 Statistical use

The structural metrics remain primary. For the secondary judge analysis:

- predesignate one primary judge outcome, preferably overall pairwise preference;
- use an exact binomial test after excluding ties while always reporting the tie rate;
- use paired ordinal tests for criterion scores;
- apply Holm correction to multiple secondary criteria or label them descriptive; and
- report patient-level confidence intervals and effect sizes.

Analyze all patient pairs. Optionally report a predefined sensitivity analysis restricted to pairs
where both methods passed fidelity and unsupported-information checks, but do not use filtering to
hide method-specific failures.

### 10.8 Part 2 judge task

Use separate-model judges only as a secondary assessment of map-free disagreements. Provide factor
names, values/scales when relevant, and the anonymous group claim. Do not show the reference group
when asking whether an alternative is coherent.

Score:

- single-construct coherence;
- category-label fit;
- possible over-grouping; and
- keep together, split, or leave individual.

Do not combine Part 1 narrative ratings with Part 2 grouping-plausibility ratings.

## 11. Existing metric families retained separately

### 11.1 Fidelity

Retain outcome correctness, direction accuracy, value fidelity, scale fidelity, rank-order fidelity,
and applicable prose-fidelity checks. Do not move missing content into conditional fidelity.

### 11.2 Completeness

Move or retain factor-reference coverage here, including interpretation key-factor coverage and
opposing-evidence coverage.

### 11.3 Unsupported information

Retain extra factors, duplicate factors, altered values/scales, unsupported prose features, and
causal-language checks. The automated lexical screen does not prove the absence of semantic
hallucination.

### 11.4 Structure

Retain report-contract checks separately. Structure compliance does not imply synthesis quality.

## 12. Repository changes required

### 12.1 Planned source changes

- revise `template_experiment/domain_template.py` to emit explicit group members;
- add a taxonomy-conditioned prompt path without changing the map-free production condition;
- add a common group-claim extraction module;
- replace method-specific hidden membership reconstruction in the new evaluation path;
- implement patient-level pairwise structural metrics;
- implement explicit-membership recall and contrastive linkage;
- create Dataset A generation and verification code;
- create Dataset B pairwise reanalysis code;
- rebuild the disagreement worksheet;
- rewrite the flexibility stress-test cases and scoring;
- wire the map-free LLM generator for Dataset C;
- replace the standalone judge scaffold with paired A/B and B/A evaluation;
- add within-patient judge aggregation and reliability statistics; and
- create a frozen protocol manifest.

### 12.2 Known repository issues to resolve first

- prompt-word counting currently returns zero;
- the embedded-category-example count is hard-coded and needs an auditable definition;
- the flexibility test and implementation disagree about edit counts;
- the LLM flexibility arm is unwired;
- the current “factor absent means grouped” logic contradicts explicit traceability;
- the current unseen-combination case is not a real predefined combination test;
- the existing judge scaffold is unpaired and has no active judge model;
- the existing template scorer reconstructs hidden membership using method-specific logic;
- the test dependency is missing from the current environment; and
- the full relevant test suite must be runnable from a clean documented environment.

### 12.3 Versioned output layout

The obsolete files under `template_experiment/outputs/pilot/synthesis/` were retired after their
replacement pairwise report was superseded by the checker-v2.1 Part 2A analysis. Do not write future
Dataset A or Dataset C results into the pilot directory; use the versioned directory below:

```text
template_experiment/
  protocol/
    synthesis_v2_manifest.json
    taxonomy_conditioned_prompt.txt
    map_free_prompt.txt
    judge_prompt.txt
    judge_rubric.md
    held_out_cases.json
  outputs/
    synthesis_v2/
      dataset_a/
        records.jsonl
        generation_log.csv
      part1_structural/
        patients.csv
        summary.csv
        report.md
      part2a_map_free/
        patients.csv
        summary.csv
        disagreement_worksheet.csv
        report.md
      part2b_held_out/
        cases.csv
        generations.jsonl
        maintenance_log.csv
        report.md
      judge_part1/
        blinded_items.jsonl
        identity_key.csv
        raw_judgments.jsonl
        patient_aggregates.csv
        summary.csv
        report.md
      judge_part2/
        blinded_groups.jsonl
        raw_judgments.jsonl
        summary.csv
        report.md
      audits/
        parser_gold.csv
        parser_validation.csv
        freeze_checksums.csv
```

Keep identity keys and blinded items separate.

## 13. Statistical reporting requirements

For every endpoint:

- identify the patient as the paired unit;
- report raw counts and evaluable denominators;
- report macro and micro estimates where relevant;
- report 95% confidence intervals;
- show patient-level paired differences;
- report prediction-category strata as descriptive pilot analyses;
- distinguish confirmatory, secondary, and exploratory endpoints;
- do not treat factor pairs or repeated judge calls as independent patients; and
- do not infer equivalence from a nonsignificant difference.

With 30 patients, emphasize uncertainty. If confidence intervals are wide, label the work a pilot
and avoid definitive comparability claims.

## 14. Test and quality-assurance plan

### 14.1 Unit tests

Add tests for:

- canonical pair construction;
- inclusion of patient, stage, and evidence side in pooled identity;
- groups of size two, three, and larger;
- partial groups;
- incorrect merges;
- wrong-side groups;
- singleton category claims;
- no predicted group with gold opportunities;
- no gold and no predicted groups;
- predicted groups with no gold opportunity;
- explicit, partial, and absent membership;
- macro versus micro aggregation;
- exact pair-set agreement;
- duplicate factors in malformed input;
- contrast scores 0, 0.5, 1, and N/A;
- blinded A/B and reversed B/A generation;
- judge-output schema validation;
- within-patient judge aggregation;
- held-out mapping-edit accounting; and
- frozen manifest/hash validation.

### 14.2 Mutation tests

Create deliberately altered interpretations to verify detection of:

- omitted group members;
- an unrelated factor merged into a group;
- a group placed on the wrong evidence side;
- hidden membership with no factor names;
- a missing evidence side;
- a connective with incorrect evidence;
- an invented factor;
- an altered value or scale;
- unsupported causal language; and
- a malformed or missing interpretation section.

### 14.3 Reproducibility checks

- install and pin the test dependency;
- run the full relevant test suite;
- run the complete analysis twice from frozen inputs;
- compare file hashes for deterministic artifacts;
- verify no generation calls occur during retrospective Dataset B analysis; and
- verify old pilot outputs remain unchanged.

## 15. Implementation phases and checklist

### Phase 0 — Protocol approval and environment repair

- [x] Approve this three-part architecture.
- [x] Confirm the domain template as the primary deterministic comparator.
- [x] Confirm that the generic template is a sanity baseline only.
- [x] Approve the primary endpoint.
- [x] Freeze the provisional noninferiority margin at `Delta = 0.05` for the pilot.
- [ ] Decide whether all 30 patients are sufficient for a pilot or whether a larger paired cohort is
      feasible.
- [x] Add and pin the missing test dependency.
- [x] Fix prompt-word counting.
- [x] Replace the hard-coded prompt-example count with an auditable measure or remove it.
- [x] Resolve stale flexibility edit-count expectations.
- [x] Run and record the relevant test suite (`68 passed, 2 skipped`).

### Phase 1 — Freeze Dataset B and shared evidence

- [x] Reference the current map-free reports as checksum-frozen Dataset B.
- [x] Save the historical prompt, model/version, temperature, run, and source metadata.
- [x] Freeze the 30 patient IDs and selection procedure.
- [x] Serialize canonical patient evidence packets.
- [x] Add patient-level evidence hashes.
- [x] Freeze the taxonomy and category labels.
- [x] Create the protocol manifest.
- [x] Verify and record that Dataset B used the map-free prompt with no taxonomy mappings.

### Phase 2 — Common explicit output contract

- [x] Define canonical explicit group membership and the template's preferred syntax.
- [x] Modify the domain template to emit category plus factor members.
- [x] Preserve singleton factor names.
- [x] Enforce same-side and same-stage grouping.
- [x] Match the existing map-free 2–4-sentence instruction with no taxonomy-only word limit.
- [x] Create a full-document taxonomy-conditioned twin of the existing map-free LLM prompt.
- [x] Ensure the prompt supplies the same selected-factor mappings and labels used by the template.
- [x] Ensure the prompt does not copy the template's exact sentences.
- [x] Add contract-level unit tests.

### Phase 3 — Parser and metric implementation

- [x] Implement one method-agnostic group-claim parser.
- [x] Store source spans for every extracted claim.
- [x] Build the hand-authored parser validation corpus.
- [x] Add parser mutation tests.
- [ ] Manually annotate the blinded validation set.
- [x] Measure exact relation/side/factor extraction agreement on the frozen corpus.
- [x] Confirm the parser meets the frozen acceptance threshold or designate manual extraction as
      primary.
- [x] Implement patient-aware pair identities in the replacement synthesis module.
- [x] Implement all edge-case conventions.
- [x] Implement macro and micro pairwise metrics.
- [x] Implement exact pair-set agreement and no-opportunity strata.
- [x] Implement provisional conditional explicit-membership recall; validate it with the common
      parser before freezing Part 1.
- [x] Implement contrastive linkage.
- [x] Retire grouping extent, Distinct-1/Distinct-2, factors-per-sentence, sentence/word counts, and
      compression from the synthesis outputs, and overwrite the old pilot synthesis files.

### Phase 4 — Dataset A generation and verification

- [x] Freeze prompt, template, taxonomy, parser, metrics, and manifest before external generation.
- [x] Regenerate all explicitly traceable domain-template interpretations.
- [ ] Generate one primary taxonomy-conditioned Gemini interpretation per patient.
- [ ] Save raw responses and complete generation metadata.
- [ ] Verify evidence hashes match across paired methods.
- [ ] Verify patient, factor, rank, value, scale, side, and stage identity.
- [ ] Run fidelity, completeness, unsupported-information, and structure checks.
- [x] Record failures without silently regenerating them.
- [x] Version any necessary rerun as a separate experiment.

### Phase 5 — Part 1 analysis

- [ ] Produce per-patient predicted and gold pair sets.
- [ ] Produce macro and micro precision, recall, and F1.
- [ ] Produce exact pair-set agreement.
- [ ] Produce explicit-membership recall.
- [ ] Produce contrastive-linkage results.
- [ ] Report all evaluable denominators.
- [ ] Calculate paired patient-level differences.
- [ ] Run the frozen paired bootstrap.
- [ ] Apply the frozen noninferiority rule.
- [ ] Report category-specific distributions descriptively.
- [ ] Write both supported and unsupported conclusions.

### Phase 6 — Part 2A retrospective map-free analysis

- [x] Run the validated parser on frozen Dataset B.
- [x] Produce macro and micro pairwise reference agreement with the common parser.
- [x] Produce exact pair-set agreement with the common parser.
- [x] Preserve conditional explicit-membership recall under a distinct label.
- [x] Build the six-category disagreement worksheet.
- [ ] Review all disagreements blinded to downstream judge results.
- [ ] Publish counts and balanced examples.
- [x] Remove “accuracy” terminology from tables and prose.
- [x] Label all author adjudication appropriately.

### Phase 7 — Part 2B prospective flexibility

- [x] Replace the current stress cases with the predefined Dataset C cases.
- [x] Define references and acceptable alternatives before generation.
- [x] Correct mapping-edit accounting.
- [x] Remove “factor name absent means grouped” logic.
- [x] Wire the frozen map-free LLM generator.
- [ ] Run all three arms without task-specific edits.
- [ ] Record render, grouping, fidelity, unsupported-content, and traceability outcomes.
- [ ] Prospectively time template mapping and LLM review work.
- [x] Record prompt, map, label, code, correction, and regeneration counts.
- [x] Apply the frozen per-case success rule.
- [x] Report every case, including pending/failed LLM rows.

### Phase 8 — Part 1 judge experiment

- [x] Freeze the six anchored criteria and overall preference question.
- [x] Freeze one primary judge outcome.
- [x] Select two non-Gemini judge-model families: `z-ai/glm-5.3-flash` and
      `qwen/qwen3.8-flash` through OpenRouter.
- [x] Add the fixed judge panel and execution settings to `explanation/config.yaml`.
- [x] Implement paired blinded A/B item construction (items materialize after Dataset A completes).
- [x] Implement reversed B/A item construction.
- [x] Configure three repetitions per order.
- [x] Validate structured-output parsing.
- [ ] Run and retain all raw judgments.
- [ ] Aggregate repetitions and orders within patient.
- [ ] Calculate order, repeat, and cross-judge consistency.
- [ ] Run patient-level statistical analysis only.
- [ ] Report the full-pair analysis and any predefined faithfulness-gated sensitivity analysis.
- [ ] Treat low agreement as an inconclusive judge result.

### Phase 9 — Part 2 judge experiment

- [x] Select only predefined explicit map-free disagreement claims.
- [x] Remove method identity and reference grouping from plausibility prompts.
- [x] Evaluate coherence, label fit, over-grouping, and preferred representation.
- [ ] Retain raw judgments and rationales.
- [x] Report separately from Part 1 judge results.
- [x] State explicitly that model judgments do not establish clinical correctness.

### Phase 10 — Reproducibility and paper integration

- [x] Run the full test suite in the pinned environment (`68 passed, 2 skipped`).
- [x] Regenerate all completed reports from frozen inputs.
- [x] Verify deterministic output hashes.
- [ ] Verify the old pilot directory is unchanged.
- [x] Publish formulas and edge cases.
- [x] Publish per-patient structural results as CSV artifacts.
- [ ] Publish judge prompts, model metadata, raw outputs, and aggregation code where permitted.
- [x] Publish the held-out case definitions and maintenance logs.
- [ ] Report expected ties on fidelity and completeness openly.
- [ ] Report template advantages and LLM failures openly.
- [ ] Align every manuscript claim with the observed evidence tier.
- [ ] Include the no-human/no-clinician limitation.

## 16. Definition of done

The revised synthesis evaluation is complete only when:

- Dataset A and Dataset B are separate and auditable;
- Dataset C cases were predefined before generation;
- both Part 1 methods use explicit group membership;
- the same validated extraction procedure is applied to both methods;
- primary structural formulas and edge cases are frozen;
- macro, micro, exact-agreement, and denominator reporting is complete;
- the noninferiority margin was defined before viewing Dataset A results;
- held-out flexibility is measured rather than claimed;
- maintenance accounting is fair to both methods;
- judge comparisons are paired, blinded, reversed, and aggregated by patient;
- judge results remain secondary;
- the test suite passes in a documented environment;
- all raw and derived artifacts are versioned; and
- reviewer-facing claims remain within the boundaries below.

## 17. Reviewer-facing claim guide

### Allowed if supported

**Part 1 noninferiority established**

> Under a shared researcher-defined taxonomy, the taxonomy-conditioned LLM was noninferior to the
> deterministic domain template on the predefined patient-level structural grouping endpoint.

**Part 1 similar estimates but inconclusive interval**

> The methods produced similar point estimates, but this 30-patient pilot did not provide sufficient
> precision to establish noninferiority.

**Part 2A map-free result**

> Without an explicit task-specific dictionary, the frozen LLM outputs showed partial pairwise
> agreement with the researcher-defined taxonomy. Disagreements included narrower, broader,
> ambiguous, missed, and unsupported groupings.

**Part 2B flexibility result**

> For factors held out from task configuration, the map-free LLM required fewer explicit mapping
> updates, while its inferred groups required the reported amount of review and correction.

**Judge advantage**

> Blinded separate-model judges preferred the LLM interpretations on the specified linguistic
> criteria. This secondary model-based result was not validated against human or clinician ratings.

**Balanced overall conclusion**

> The deterministic template provided reproducible, auditable taxonomy-based grouping. The LLM's
> structural performance under the same taxonomy and its lower explicit mapping burden in the
> map-free condition are reported separately; neither result establishes clinical usefulness.

### Claims to avoid

- “The LLM provides better clinical synthesis.”
- “The LLM groupings are clinically correct.”
- “The LLM improves clinician understanding or decision-making.”
- “The LLM requires no taxonomy or maintenance.”
- “The template needs a separate rule for every factor combination.”
- “The LLM saw genuinely unseen clinical concepts.”
- “No significant difference proves equivalence.”
- “The LLM is independently validated by another LLM.”

## 18. Recommended execution priority

1. Repair the environment and known metric/scaffold drift.
2. Freeze Dataset B, patient evidence, taxonomy, definitions, and manifest.
3. Implement explicit domain-template membership and the common parser.
4. Validate extraction and implement the structural metrics.
5. Generate Dataset A and run the Part 1 noninferiority analysis.
6. Complete the retrospective Dataset B disagreement analysis.
7. Run the prospective held-out Dataset C flexibility experiment.
8. Run the blinded separate-model judge experiments last.
9. Integrate results into the paper only after all frozen analyses and checks complete.

## 19. Generation observability and recovery follow-up

- [x] Show pipeline stages and patient/case-level request, response, and save progress.
- [x] Print a waiting heartbeat every 15 seconds; use a configurable 120-second Gemini timeout.
- [x] Disable hidden SDK retries for these generation collectors.
- [x] Atomically checkpoint every raw response before local parsing/evaluation.
- [x] Resume missing/failed/interrupted requests without regenerating saved responses.
- [x] Preserve contract-flagged responses and locally reprocess saved raw responses after parser errors.
- [x] Retain attempt history and prevent concurrent collectors using the same checkpoint directory.
- [x] Verify patient/case, prompt, model, and evidence identity before reusing saved data.
- [x] Correct deterministic auditing for CSV numeric formatting (`0` versus `0.0`) without changing metrics.
- [x] Verify the full saved run with API-client construction blocked: 30 Dataset A and six Dataset C
  responses reused, unchanged, and all six deterministic audit checks passing.
- [x] Add offline regression tests for interruptions, request failures, resume, raw-response retention,
  contract flags, identity mismatches, waiting heartbeats, locking, and semantic hash comparisons.
- [x] Inspect the 20 Dataset A reports and 42 original flags; implement and locally recheck the
  identified case/alias, contrast-wording, connector, and group-boundary limitations.
- [x] Preserve original generation flags and responses; save code/data hashes and revised
  per-report extraction evidence in `history/checker_v2/`.
- [x] Test all 30 saved Gemini interpretations and 30 templates, plus positive paraphrases and
  negative mutations. Current result: 189 tests passed, two skipped.
- [x] Keep received but genuinely noncompliant reports in paired analysis and judge preparation;
  distinguish uncertain extraction requiring review from recognized content mismatches.
- [x] Recompute derived metrics consistently with parser v2 in a separate output tree; preserve
  previous metrics and the original frozen protocol manifest.
- [ ] Obtain human confirmation of pilot extraction (the audit records this as pending).
- [ ] Validate on a separate development/validation corpus and freeze the checker before final testing.
- [x] Extend per-call durable resume to both OpenRouter judge collectors, with frozen request
  identity, timeouts, waiting heartbeats, invalid-response retention, and explicit call caps.
- [x] Ensure paired-judge faithfulness sensitivity uses the matching versioned structural results.
- [x] Prepare a blinded human annotation packet for all 60 pilot method/report paragraphs; preserve
  reviewer drafts on rerun and keep method identity/parser answers outside the packet.
- [x] Run 16 additional synthetic challenge cases against the unchanged checker (16 expected
  outcomes matched). These assistant-authored checks are not independent validation.
- [ ] Review the blinded extraction packet and resolve disagreements before final-test freezing.
- [ ] After pilot review, explicitly authorize a capped judge smoke test (up to four new calls),
  inspect both judges' outputs, then authorize the remaining panel. No paid calls were made here.

Generation is complete for the current saved run; contract review is distinct from missing API
responses. No synthesis prompt, grouping criterion, or contract threshold was changed in this
runtime recovery. The existing protocol manifest remains preserved. A request interrupted before
its response reaches the local checkpoint may be billed again when explicitly resumed.

The subsequent checker-v2 revision changes operational recognition rules, not generated text or
the taxonomy. All 30 saved Gemini interpretations pass its current checks. This is a pilot-tuned
result, not an independent validation claim. See `template_experiment/outputs/synthesis_v2/history/checker_v2/`
for the original snapshot, before/after flags, extraction evidence, revised metrics, and checksums.

### Confirmed clinical-noun repair: checker v2.1

- [x] Confirm and version the expanded clinical-noun allow-lists, without changing raw reports.
- [x] Add connector/parenthesis regression cases and the actual P40538 map-free report.
- [x] Verify that only P40538's predicted pairs change relative to the saved checker-v2 audit;
  verify deterministic template rows remain unchanged.
- [x] Exclude P40538 from disagreement adjudication (21 → 20 reports), retaining it in full-cohort metrics.
- [x] Create the new active `checker_v2_1/` result tree; preserve `history/checker_v2/` and the
  original protocol. All six reproducibility checks pass; tests: 229 passed, two skipped.
- [x] Record separate historical macro-F1 baselines in `checker_v2_1/REVISION_NOTES.md`.
- [ ] Complete human review and independent validation before freezing for final testing.

### Author-adjudicated Part 2A reporting

- [x] Lead the report with author decisions, separating complete reviews from entered labels
  and automatic routing suggestions; preserve the primary pairwise metrics.
- [x] Import the canonical author worksheet into active versioned outputs; preserve drafts,
  snapshot prior artifacts, and flag decisions for re-review if their underlying inputs change.
- [x] Record reporting provenance separately as `map-free-adjudicated-report-v1`; verify the
  existing `explicit-group-claims-v2.1-pilot` parser manifest still matches its source hashes.
- [x] Verify the offline pipeline and all six reproducibility checks; tests: 239 passed, two skipped.
- [x] Add missing author rationales for 3001, 101566, and 292826 (20/20 reviews now complete).
- [x] Clarify the pilot-developed rubric as `adjudication-rubric-v1.1`, preserve v1, and record
  the current rubric version/hash and immutable snapshot in both Part 2A report manifests.
- [x] Add missing-reference-group rationale details for 4098 and 130828 without changing labels;
  synchronize author and active reports at seven granularity / 13 plausible / zero unsupported.
- [x] Verify unchanged structural scores, checker hashes, and existing frozen judge requests;
  regression suite: 240 passed, two skipped. No paid calls initiated during this cleanup.

### Judge runtime v2 safety revision

- [x] Replace the judge-only transport to fix seconds/milliseconds conversion and explicitly
  disable underlying SDK retries; leave report-generation clients unchanged.
- [x] Add a cancellable wall-clock deadline and a configurable 2,048-token output cap.
- [x] Preserve complete response metadata; distinguish empty, truncated, non-final, and invalid
  final content without falling back to reasoning or automatically resampling received responses.
- [x] Freeze runtime settings with request identities in new versioned judge directories.
- [x] Balance capped calls across judges and guard against concurrently running legacy collectors.
- [x] Verify offline regressions: 249 passed, two skipped.
- [x] Stop the verified orphaned legacy collector with user approval; complete four real-prompt
  smoke calls covering both models in both panels, with one attempt per request and no full panel.
- [x] Inspect the saved responses: three valid judgments; GLM paired judging exhausted the
  2,048-token budget on reasoning and returned no final content. Preserve all four outcomes.
- [x] Verify GLM's published capability metadata: reasoning is mandatory, defaults to max effort,
  and supports low effort. Save a separate `glm_paired_low_v1` retest profile with low effort and
  the unchanged 2,048-token cap.
- [x] Retest the same failed GLM paired prompt once: valid final JSON in 20.170 seconds, with
  1,553 input / 268 output tokens (72 reasoning). Preserve the original four-call collection.
- [x] Verify retest isolation and SDK parameter forwarding; full offline suite: 253 passed, two skipped.
- [x] Adopt the selected model-specific settings consistently in new panel-v3 output directories:
  GLM low effort in both panels, Qwen provider default, both capped at 2,048 tokens with a
  120-second deadline and no retries. Preserve the original smoke outcomes separately.
- [ ] Run the full panel only after explicit authorization; preparation alone must make no API calls.
