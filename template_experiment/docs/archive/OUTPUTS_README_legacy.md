# Results guide (plain language)

This guide explains each evaluation metric — what it measures, the approach, and exactly how it is
calculated — so anyone reading the numbers can understand them without asking. Written for reviewers.

**Metrics detailed here:**

1. **Fidelity** — is the presented information correct? → `gemini/fidelity_comparison/`
2. **Completeness** — is all the required content present? → `gemini/completeness_comparison/`
3. **Unsupported information** — was anything added, duplicated, or altered? → `gemini/unsupported_information_comparison/`
4. **Structure compliance** — does the report follow the required format? → `gemini/structure_comparison/`
5. **Synthesis** — does the interpretation *integrate* the factors, and how good is the prose? →
   `gemini/synthesis_comparison/` (deterministic proxies) and `gemini/geval_synthesis/` (LLM-judge scaffold)
6. **Complexity / scalability / flexibility** — what does each method *cost*, and how does it handle
   *unseen* inputs? → `gemini/complexity_scalability/` and `gemini/flexibility_stress_test/`

A companion engineering review of all metrics is in
[`../METRIC_REVIEW.md`](../METRIC_REVIEW.md), and the plan for the two unfinished pieces (the G-Eval
judge and the flexibility generation runs) is in
[`../SYNTHESIS_SCALABILITY_PROPOSAL.md`](../SYNTHESIS_SCALABILITY_PROPOSAL.md).

> **Honest-reporting note.** The framing literature is old and the field moves fast, so we use it only
> for definitions, treat every "expected pattern" as a hypothesis, and report ties and template wins
> as plainly as LLM wins. Where a number surprised us, we say so.

---

## Metric 1 — Fidelity

Generated files: [`gemini/fidelity_comparison/`](gemini/fidelity_comparison/) — `fidelity_report.md`
(summary table), `fidelity_summary.csv` (per-category/overall means with denominators),
`fidelity_patient_results.csv` (one row per patient per method), `fidelity_paired_differences.csv`.

## What "fidelity" means

**Fidelity = is the information the report presents *correct* with respect to the model's evidence?**

Every explanation is compared against the same ground truth: the model-derived evidence packet built
by `build_patient_explanation_data_full()` — the predicted label, the selected SHAP factors, each
factor's value, its scale text, which direction it pushed, and its rank. Fidelity asks only about
what is *shown*: given that the report displays something, is it right?

Two things are deliberately **not** part of fidelity, so that no error is counted twice:

- whether a required factor is **missing** → that is **completeness** (recall);
- whether the report **added** something not in the evidence → that is **unsupported information**
  (precision).

So a low fidelity score always means "what was shown is wrong," never "something was missing or
extra." (See the sibling result folders for those families.)

## Three methods compared

Fidelity (and every other metric) is computed for three explanation methods, so the comparison is
fair:

| Method | What it does with the interpretation paragraph |
| --- | --- |
| **Generic template** | Lists the top factors by name. No grouping, no value words. The rule-free baseline. |
| **Domain template** | Groups the top factors into frozen clinical categories using rules (the "fair" deterministic comparator — same grouping ability as the LLM). |
| **LLM (Gemini)** | Groups and phrases the factors in natural language from a single prompt. |

Giving the domain template the *same* grouping ability is what lets us answer the reviewer's fairness
question: any remaining LLM advantage is about the *quality* of the language, not about *having* a
taxonomy.

## A rule used by every fidelity metric: conditional + NaN + visible denominator

- **Conditional.** Each metric scores only the items that are actually shown. Duplicated factors are
  collapsed to their first occurrence (duplicates are counted in the unsupported family, not
  penalized here).
- **NaN when nothing is checkable.** If a report shows nothing to score, the metric is `NaN`
  ("not evaluable"), never a silent 0 or 1.
- **Visible denominator.** Every mean is reported as `value (n eval / n patients)`. This matters: a
  method that produced empty output would otherwise look perfect because its hard cases quietly
  dropped out. The denominator makes that impossible to hide.

---

## Part A — Tables and label fidelity (5 metrics)

These cover the structured parts of the report: the prediction line and the two factor tables.

| Metric | Question | How it is computed |
| --- | --- | --- |
| **Outcome correctness** | Did it state the right predicted class, and only that? | Parse the `Fall Classification` / `Fall Severity Classification` fields; require them to equal exactly the expected labels (extra or contradictory labels fail). Boolean per patient. |
| **Direction accuracy** | Are shown factors on the correct side? | For each displayed expected factor (first occurrence), check its table matches its SHAP sign. Score = correct ÷ displayed-expected. NaN if none shown. |
| **Value fidelity** | Are the patient values copied correctly? | For each displayed expected factor, check the Patient Value cell equals the evidence value **verbatim** (whitespace-normalized). Score = correct ÷ displayed-expected. A paraphrase counts as wrong. |
| **Scale fidelity** | Is the scale/interpretation text copied correctly? | Same as value fidelity, on the `Interpretation / Scale` cell. |
| **Rank-order fidelity** | Are factors listed in the right order? | Within each table, of all expected factor pairs where both appear, the fraction the report lists in the correct order (pairwise concordance). Correct order → 1.0, fully reversed → 0.0, `<2` comparable → NaN. |

On the pilot these tie near 100% across all three methods — as expected, because all three copy the
same evidence packet. That is the point: fidelity is a **non-inferiority** check (the LLM does not
lose correctness), not where a method wins.

## Part B — Interpretation-prose fidelity (3-metric panel)

The five metrics above never read the free-text **Model Interpretation** paragraph — the only part
of the document that differs between methods, and the place the LLM might add value. This panel
checks whether the *claims in that paragraph* are true, fully automatically (no clinicians).

It reads only the Model Interpretation block (the Clinical Note is fixed boilerplate and is
ignored), and validates each claim against the evidence packet.

| Metric | Question | How it is computed |
| --- | --- | --- |
| **Prose direction fidelity** | Is each factor named in the prose discussed on its correct side? | Split the paragraph into supporting/opposing clauses (class-aware cues + the "did not outweigh" clause). Detect factor mentions; each is correct if the side it is discussed on matches its SHAP table. Score = correct ÷ mentions. NaN if none. |
| **Prose grouping fidelity** | When the prose groups factors ("gait and mobility"), is the group grounded? | Detect genuine multi-factor group phrases. A group named on side S is faithful if the category has ≥1 member factor actually present on side S. Score = faithful ÷ group mentions. NaN if no grouping. |
| **Prose value-descriptor fidelity** | Do value words ("severe", "mild", "absence of") match the real value? | For each value word next to a factor, check consistency against the factor's actual value using frozen scale metadata (e.g. "severe" but the value is 1 of 4 → flagged). Score = consistent ÷ checkable. NaN if none. |

**What the pilot shows (and why it matters):**

- **Generic template:** grouping and value-descriptor fidelity are `NA` — it *never groups* and
  *never describes values*, so there is nothing to score. This is the honest floor.
- **Domain template:** grouping fidelity is ~100% (it groups by rule, correct by construction) but
  value-descriptor is still `NA`. So on grouping *correctness* it matches the LLM.
- **LLM:** grouping fidelity ~95% (it occasionally errs) **and** it uniquely adds value descriptors
  at ~100%.

The takeaway: grouping *fidelity* ties (both grouping methods are correct by construction), so the
LLM's advantage is not correctness — it is the *quality* and *naturalness* of the synthesis (measured
by the separate Synthesis metric), and its ability to group new combinations from one prompt
(the Scalability metric).

### Why this design for the prose

- **No clinicians available**, so the panel is fully automatic and **conservative**: it flags only
  clearly verifiable errors (wrong side, invented/mis-filed grouping, value-word contradictions). It
  does **not** judge writing quality (that is the Synthesis metric) and cannot catch a subtly worded
  invented interaction — a limitation we state openly.
- **Feature hallucination** (naming a factor absent from the tables) and **causal language** are
  *precision* problems and are reported by the **unsupported-information** module, not here, so
  nothing is double-counted.
- To read grouped language, the checker needs to know which factors belong to which clinical
  category. That taxonomy is **not invented by us** — it is adopted from established clinical
  instruments (below), frozen, and version-controlled before results are examined. The same frozen
  map is what the domain template uses to group, keeping the comparison fair.

## Grouping taxonomy — sources

Each factor's clinical category is taken from the instrument it comes from, not authored ad hoc.
Defined in `template_experiment/evaluation/interpretation_fidelity.py`
(`FEATURE_CATEGORY` / `CATEGORY_PROVENANCE`, `TAXONOMY_VERSION`). **References to be verified before
freezing:**

- **MDS-UPDRS Parts I / III / IV** (non-motor experiences; motor examination; motor complications) —
  Goetz CG et al. *Movement Disorder Society-sponsored revision of the Unified Parkinson's Disease
  Rating Scale (MDS-UPDRS).* Mov Disord. 2008;23(15):2129–2170.
- **Hoehn & Yahr staging** — Hoehn MM, Yahr MD. *Parkinsonism: onset, progression and mortality.*
  Neurology. 1967;17(5):427–442.
- **PIGD (postural-instability/gait-difficulty) grouping of gait & postural items** — Stebbins GT et
  al. *How to identify tremor dominant and postural instability/gait difficulty groups with the
  MDS-UPDRS.* Mov Disord. 2013;28(5):668–670.
- **SCOPA-AUT (cardiovascular autonomic symptoms)** — Visser M et al. *Assessment of autonomic
  dysfunction in Parkinson's disease: the SCOPA-AUT.* Mov Disord. 2004;19(11):1306–1312.
- **MoCA (cognition)** — Nasreddine ZS et al. *The Montreal Cognitive Assessment, MoCA.* J Am Geriatr
  Soc. 2005;53(4):695–699.
- **GDS-15 (depression)** — Sheikh JI, Yesavage JA. *Geriatric Depression Scale (GDS): recent
  evidence and development of a shorter version.* 1986.
- **Motor vs. non-motor symptom framework** — Chaudhuri KR, Schapira AHV. *Non-motor symptoms of
  Parkinson's disease: dopaminergic pathophysiology and treatment.* Lancet Neurol.
  2009;8(5):464–474.

Two assignments where the source instrument and the functional domain disagree — **Mobility Summary**
and **FOG (Freezing of Gait)** — are flagged in code (`AMBIGUOUS_ASSIGNMENTS`) and should be resolved
against a clinical reference before the taxonomy is frozen.

## How to reproduce

From the project root:

```bash
./.venv/bin/python -m template_experiment.evaluation.build_gemini_pairs
./.venv/bin/python -m template_experiment.evaluation.fidelity_comparison
```

The first rebuilds the paired dataset (generic template, domain template, and the saved LLM output
per patient); the second computes all eight fidelity metrics and writes this metric's outputs.

---

## Metric 2 — Completeness

Generated files: [`gemini/completeness_comparison/`](gemini/completeness_comparison/).

**Completeness = is all the required content *present* and non-empty?** Where fidelity asks "is what
is shown correct?", completeness asks "is anything *missing*?" (a factor left out, an empty section, a
classification field absent). Extra content is the unsupported family's concern, not this one — so no
error is counted twice.

| Metric | Question | How it is computed |
| --- | --- | --- |
| **Supplied-factor recall** | Are all supplied factors shown? | displayed ∩ expected ÷ expected. |
| **Required-section nonempty coverage** | Are all required sections present *and* non-empty? | Fraction of {title, prediction, each factor table, interpretation, note} that exist with content. |
| **Classification-field coverage** | Are the prediction fields present? | Fall field always; severity field when a severity is expected. |
| **Expected table-cell completeness** | Are all table cells filled? | Filled cells ÷ (4 × expected factors); a missing row = 4 missing cells. |
| **Displayed-stage evidence coverage** | Are both direction tables for the shown stage present? | The two expected direction sections appear. |
| **Full-pipeline stage-evidence coverage** | Is evidence shown for *every* stage the prediction depends on? | A routed faller needs Stage-1 **and** Stage-2 evidence; showing only Stage-2 = 0.5. |
| **Interpretation key-factor / opposing coverage** | Does the prose mention the required top factors? | Fraction of the top supporting / opposing factors represented. **Group-aware**: a named category ("gait and mobility") covers its member factors, so grouped prose is not unfairly penalized. |
| **Displayed-contract complete** | Is the shown single-stage report fully complete? | All of the above for the displayed stage, at 100%. |
| **Full-pipeline complete** | Is the report complete across all stages? | Additionally requires Stage-1 evidence for routed fallers. |

**What the pilot shows:** most metrics tie at 100% (all three methods copy the same evidence). Two do
not, and both are *structural*, not method differences: **full-pipeline stage-evidence coverage is
66.7%** and **full-pipeline complete is 33.3%**, because routed fallers currently show only Stage-2
evidence — exactly the reviewer's point that the final faller class depends on both stages. This is a
report-design fact (see `../TODO_TWO_STAGE.md`), identical across methods.

This module was trimmed from 12 to 10 metrics: two were removed as redundant —
`required_stage_decision_coverage` (numerically identical to classification-field coverage) and a
presence-only section check (subsumed by the non-empty version).

### How to reproduce

```bash
./.venv/bin/python -m template_experiment.evaluation.completeness_comparison
```

---

## Metric 3 — Unsupported information

Generated files: [`gemini/unsupported_information_comparison/`](gemini/unsupported_information_comparison/),
including a blinded prose-claim worksheet and its identity key.

**Unsupported information = did the report *add*, *duplicate*, or *alter* anything relative to the
evidence?** It is the mirror of completeness (which asks "is anything *missing*?"). Raw counts are the
primary evidence; rates are secondary.

| Raw count (primary) | What it counts |
| --- | --- |
| Extra (unsupported) factors | Table factors not in the evidence set. |
| Duplicated factors | Factors shown more than once. |
| Altered patient values / scale texts | Cells that differ from the evidence (verbatim). |
| Unsupported prose features | Features named in the interpretation whose feature **and** clinical category are absent from the evidence (group-aware). |
| Causal-language sentences | Interpretation sentences using causal/risk wording (a screen, not a correctness judgment). |

Plus `automated_unsupported_information_free` — the all-clear rate (no unsupported content at all).

**What the pilot shows:** generic and domain templates are 100% clean; the LLM is 96.7% (one altered
scale text — a paraphrase of a scale description, the same one fidelity flags).

Design notes:
- **Value/scale alteration *rates* are omitted** — they are the exact complement of fidelity's
  value/scale fidelity, so counts are reported here instead of a duplicated rate. `factor_precision`
  (= 1 − unsupported-factor rate) was likewise removed.
- The prose check is **group-aware**: a group label ("motor impairment") whose category is present is
  not misread as a hallucinated specific feature — without this, the domain template showed 6 false
  positives.
- Automated prose measures are **screening only**; the blinded `prose_claim_audit_blinded.csv`
  worksheet (+ key) is generated for the general unsupported-claim rate (unsupported atomic claims ÷
  checkable atomic claims), to be completed by raters.

### How to reproduce

```bash
./.venv/bin/python -m template_experiment.evaluation.unsupported_information_comparison
```

---

## Metric 4 — Structure compliance

Generated files: [`gemini/structure_comparison/`](gemini/structure_comparison/).

**Structure = does the document follow the required format?** Layout only — it does not check facts,
completeness, or claims. It is **eight independent component checks**, each a distinct failure mode:

| Component | What it checks |
| --- | --- |
| Title format | Exact title line with the right patient ID. |
| Section order | Title → prediction → factor tables → interpretation → note, in order. |
| Stage template | The two direction tables are exactly the correct pair for the stage. |
| Table header format | Both tables use the exact 4-column header. |
| Table row format | Every data row has 4 cells and a numeric index. |
| Table numbering | Rows numbered 1..N within each table. |
| Supporting table first | The prediction-supporting table appears first (drivers first). |
| Forbidden content clean | No probabilities/percentages/likelihoods. |

Reported as the eight component pass-rates, plus **`overall_structure_compliant`** (all eight pass)
and **`n_components_passed`** (0–8, an interpretable graded summary). A mean-of-booleans
"structure_score" was removed — averaging unlike format checks produced a percentage with no meaning.

On the pilot all three arms pass every component (8.00/8), the expected result for format compliance.

### How to reproduce

```bash
./.venv/bin/python -m template_experiment.evaluation.structure_comparison
```

---

## Metric 5 — Synthesis

Generated files: [`gemini/synthesis_comparison/`](gemini/synthesis_comparison/) (deterministic
proxies) and [`gemini/geval_synthesis/`](gemini/geval_synthesis/) (LLM-judge scaffold).

### What "synthesis" means

**Synthesis = does the interpretation *integrate* the factors into coherent higher-level prose,
rather than list them?** This is about *quality*, not correctness. Correctness is fidelity; whether
the required factors are mentioned is completeness. A paragraph can be perfectly faithful yet a flat
list (low synthesis), or beautifully written yet wrong (high synthesis, low fidelity) — so the two
are measured separately and never mixed.

This is a **data-to-text generation** problem, so we borrow its established vocabulary: what we call
"grouping/integration" is **aggregation** in the classic NLG pipeline (Reiter & Dale 2000 —
*combining two or more messages into a more complex sentence*; they note templates struggle to do it).

Synthesis has two layers, both **human-free** (no clinicians): objective deterministic proxies
(Layer A) and a reference-free LLM-judge (Layer B, secondary).

### Layer A — objective deterministic proxies

Computed on the Model Interpretation paragraph, per patient, per method, with the same
conditional / NaN / visible-denominator rule as fidelity.

| Metric | Question | How it is computed |
| --- | --- | --- |
| **Aggregation rate** *(primary)* | Does it express factors as higher-level groups instead of names? | (factors covered by a group label) ÷ (all referenced factors), using the frozen taxonomy. A pure listing = 0. |
| **Contrast present** | Does it weigh supporting vs. opposing evidence? | True if the prose references both sides (class-aware clause split). |
| **Non-redundancy (distinct-2)** | Does it avoid repetition? | Distinct bigram ratio (higher = less repeated wording). |
| **Compression** | How densely does it pack factors? | Referenced factors ÷ sentences (descriptive, not better/worse on its own). |

**What the pilot shows — and an honest surprise:**

| | Generic template | Domain template | LLM |
| --- | ---: | ---: | ---: |
| Aggregation rate | 0.0% | **51.4%** | **6.8%** |
| Contrast present | 100% | 100% | 100% |
| Non-redundancy (distinct-2) | 89.3% | 87.3% | 91.8% |

The **rule-based domain template aggregates far more than the LLM (51% vs 7%)**. By this objective
measure the LLM does *not* integrate more — it tends to name specific factors ("absence of freezing
of gait, normal postural stability") rather than umbrella terms. Contrast and non-redundancy tie
(single-stage templates are naturally non-redundant, as the data-to-text literature predicts). So
**raw aggregation is not the LLM's advantage.** If the LLM adds value it is in the *quality* of the
prose (Layer B) or in flexibility (Metric 5) — not in how much it groups. We report this as found.

*(Caveat: the aggregation proxy credits group words; because the LLM uses specific clinical
descriptors, the proxy may also under-count its grouping. Both readings are reported.)*

### Layer B — G-Eval LLM-as-judge (secondary, scaffold)

Layer A cannot judge coherence or fluency. Layer B uses a **reference-free LLM judge** (G-Eval style:
the judge reasons step-by-step over a rubric, then scores) on a **frozen rubric** (`frozen-v0`, 1–5):
integration, coherence, fluency, non-redundancy, contrast.

How it is calculated (and its bias controls): the 30×3 interpretations are turned into **90 blinded
items** — the judge sees the evidence tables (identical across methods) plus one interpretation, with
method identity removed and held in a separate key; item order is randomized; one judge scores all
arms. It is **secondary and discounted**: LLM judges favor LLM-style text, so a judge-only synthesis
win does not count — conclusions are anchored on Layer A.

**Status:** scaffold only. The rubric, the 90 blinded prompts, and the identity key are generated
now; scoring needs a **separate judge model** wired to `JUDGE_MODEL` (must not be the generator).

### How to reproduce

```bash
./.venv/bin/python -m template_experiment.evaluation.synthesis_comparison   # Layer A (runs now)
./.venv/bin/python -m template_experiment.evaluation.geval_synthesis_judge  # Layer B scaffold
```

---

## Metric 6 — Complexity / scalability / flexibility

Generated files: [`gemini/complexity_scalability/`](gemini/complexity_scalability/) (static counts +
unseen-feature edit cost) and [`gemini/flexibility_stress_test/`](gemini/flexibility_stress_test/)
(the stress-test harness).

### What it measures

**What does it *cost* to make each method produce good explanations, and how does each behave on
inputs not seen during development?** The "unseen feature/combination" question is the
**compositional generalization / systematicity** problem in data-to-text generation — with the known
caveat that neural models *also* hallucinate on unseen combinations, so the LLM's flexibility is
something we must **measure**, not assume.

### Part A — static complexity (authored machinery)

Simple counts of how much hand-built logic each method needs.

| Method | templates | grouping rules | map entries | category labels | prompts | prompt words | interp. SLOC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Generic template | 1 | 0 | 0 | 0 | 0 | 0 | 25 |
| Domain template | 1 | 1 | 21 | 8 | 0 | 0 | 30 |
| LLM (Gemini) | 0 | 0 | 0 | 0 | 1 | 148 | 0 |

Computed by importing the taxonomy (map sizes) and counting source lines of the interpretation code;
the prompt is **counted as manual logic** (it embeds 5 example categories), so we do *not* claim "0
vs N". The honest claim is **growth**: the template's authored burden scales with the feature/category
set, while the prompt's is roughly fixed.

### Part B — flexibility stress test (systematicity)

Each stress case is an evidence packet simulating an unseen situation (a brand-new feature, an unseen
combination, a one-sided list, a single factor, a long list). For each, per method, we log: does it
**render** with no code change, does it **group** the new feature, how many **edits** grouping would
cost, and **compositional fidelity** (does the output reproduce the packet verbatim — a self-contained
check that works on synthetic packets).

Concrete result for the **unseen feature** case:

| Method | renders (0 edits) | groups it | edits to group it | fidelity |
| --- | :---: | :---: | ---: | ---: |
| Generic template | yes | never groups | 0 | 1.0 |
| Domain template | yes | no | **10** (1 map + 1 label + 8 phrases) | 1.0 |
| LLM (Gemini) | *requires generation* | *requires generation* | 0 *(claim — VERIFY)* | *requires generation* |

Reading: both templates *render* an unseen feature for free, but the domain template needs **10
authored artifacts to group it**; the generic template never groups. The **LLM row is deliberately
unproven** — its zero-edit grouping is a claim to confirm by generation, and `compositional_fidelity`
is there to catch it hallucinating on the new feature once wired.

**Status:** deterministic arms run now; the LLM arm is a pluggable hook (`LLM_GENERATOR`) that needs
a generation model, and the full experiment needs a frozen development/held-out split built on the
real ML artifacts.

### How to reproduce

```bash
./.venv/bin/python -m template_experiment.evaluation.complexity_scalability     # static counts (runs now)
./.venv/bin/python -m template_experiment.evaluation.flexibility_stress_test    # stress harness (deterministic arms run now)
```
