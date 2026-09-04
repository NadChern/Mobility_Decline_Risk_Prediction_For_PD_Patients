# Proposal: Metric 4 (Synthesis) and Metric 5 (Complexity / Scalability / Flexibility)

Both metrics are **human-free** (no clinicians) and grounded in existing literature rather than
invented. This document proposes what to compute, how, and what each method is expected to show —
for the three arms already built: generic template, domain-enriched template, LLM.

Citations are listed at the end; author/year details marked *(verify)* should be confirmed before
the paper.

---

## Honest-reporting caveat (read first)

Most of the framing literature is **old** (Reiter & Dale 2000; van Deemter 2005; entity-grid 2005/8)
and the field moves fast — a 2026 LLM and a modern template behave nothing like early-2000s systems.
Therefore:

- **Use the old work for definitions and prior art only** (what *aggregation* / *systematicity*
  mean), **not for empirical predictions.** Where recent work exists (G-Eval 2023;
  compositional-generalization D2T 2023–24), prefer it for the empirical claims.
- **Every "expected pattern" in this document is a pre-registered hypothesis, not an assumption.**
  We freeze proxies, rubric, thresholds, and the held-out split **before** looking at results, then
  report what we actually find.
- **Report all outcomes symmetrically.** Ties and template wins are stated as plainly as LLM wins
  (plan §11). Legitimate outcomes include: the enhanced template reads fine (no coherence gap → no
  synthesis advantage); the LLM aggregates unfaithfully on some cases (fidelity gate flags it); or no
  measurable difference at all.
- **Anchor on the objective proxies, discount the LLM judge.** G-Eval is secondary and biased toward
  LLM-style text; if only the judge favors the LLM while Layer-A proxies do not, we do not claim a
  synthesis win.
- **No inflation, no deflation.** High constrained-adherence numbers are earned evidence, not spin;
  equally, a favorable proxy is not a usefulness claim (that needs clinicians).

---

## Framing: this is a data-to-text (D2T) generation problem

Structured SHAP evidence → clinical prose is exactly **data-to-text generation**. That field already
studied template-vs-neural and gives us the vocabulary and metrics:

- What we call "grouping/integration" is **aggregation** in the classic NLG pipeline (Reiter & Dale
  2000): *"determining whether two or more messages can be combined to produce a more complex
  sentence."* Reiter & Dale explicitly note static templates *cannot readily do aggregation,
  referring-expression generation, or lexicalization* — this is our LLM-value argument, already named.
- D2T human studies find neural output **more coherent** but templates **naturally non-redundant**
  (a template cannot repeat itself), while template text is **stilted, no variability**. → Design
  consequence: **non-redundancy is not a single-stage LLM advantage**; it only discriminates in the
  **cross-stage** report, where a rule template must concatenate two stages and duplicates shared
  factors.
- "Unseen combinations of features" is the **compositional generalization / systematicity** problem
  for D2T (Mehta et al. 2023; SPOR 2024), with the key finding that **neural systems also produce
  unfaithful text (hallucination/omission) on unseen combinations** — so we must *measure* LLM
  fidelity on new combinations, not assume it.

---

## Metric 4 — Synthesis (quality of integration)

**Question:** given the same evidence, does the interpretation *integrate* the factors into coherent,
higher-level prose, rather than enumerate them? This is quality, not correctness (correctness is
fidelity; factor coverage is completeness — kept separate).

Two layers. Layer A is deterministic and objective; Layer B is an LLM judge as a secondary,
caveated proxy for the clinician rating we cannot do.

### Layer A — objective structural proxies (deterministic, no LLM)

Each is computed on the Model Interpretation prose, per patient, per method, with the same
conditional / NaN / visible-denominator rules as fidelity.

| Proxy | What it measures | How computed | Grounding |
| --- | --- | --- | --- |
| **Aggregation rate** | Does it group factors into higher-level concepts? | (# factors expressed via a group label) / (# factors referenced), using the frozen taxonomy. | NLG *aggregation* (Reiter & Dale 2000) |
| **Compression / integration density** | Does it say more with less? | (# grounded factors covered) / (# sentences or tokens). | D2T conciseness; summarization conciseness |
| **Contrast presence** | Does it explicitly weigh supporting vs. opposing? | Detect the two-sided structure ("did not outweigh" + both sides referenced). Binary/graded. | rubric dimension (plan §5.4) |
| **Non-redundancy** | Does it avoid repeating content? | distinct-n and/or n-gram repetition rate; **cross-stage:** count duplicated factor mentions across the two stage blocks. | distinct-n; self-BLEU; redundancy metrics (survey 2501.12011) |
| **Local coherence** *(optional)* | Do sentences connect? | entity-grid coherence, or a lighter connective/transition proxy. | entity-grid (Barzilay & Lapata) |

Expected pattern: generic template — aggregation 0, stilted; domain template — aggregation high but
mechanical (low coherence, awkward "X and Y and Z"); LLM — aggregation high **and** fluent. The
**cross-stage non-redundancy** proxy is where the LLM should clearly beat the domain template
(concatenation duplicates).

### Layer B — G-Eval LLM-as-judge (secondary, reference-free)

- Use a **separate** judge model (not Gemini — avoid self-preference bias), scoring a **frozen
  rubric**: integration, coherence, fluency, non-redundancy, contrast. Chain-of-thought +
  form-filling + probability-weighted score (the G-Eval method, Liu et al. 2023).
- **Blinded and randomized** method order per item; report inter-run variance.
- Report as **secondary**, with disclosed limitations: LLM judges favor LLM-style text, and this is
  not a clinician judgment. Anchor it against Layer A — if the objective proxies and the judge agree,
  the finding is credible; if only the judge favors the LLM, treat it as weak.

### Guardrails (no double-counting)

- **Factual correctness → fidelity.** A synthesis containing a fidelity error is *flagged and
  excluded from a high synthesis score* (a gate), never re-scored here.
- **Factor coverage → completeness.** Whether the required factors are mentioned is not a synthesis
  metric.
- **Scope honesty:** these proxies measure *structural integration and coherence*, **not** clinical
  clarity/usefulness/trust — those need clinicians and stay out of scope.

---

## Metric 5 — Complexity / Scalability / Flexibility

**Question:** what does it *cost* to make each method produce (and keep producing) good explanations,
and how does each behave on inputs not anticipated during development?

### Part A — static complexity (deterministic counts)

Report per method (some already gathered for the domain arm):

- number of templates / renderers;
- number of manually authored rules (e.g. the `>=2 → group` rule);
- number of map entries (feature→category = 21, display labels = 8);
- source lines and cyclomatic complexity (interpret cautiously);
- number of prompts **and prompt size** — *be honest: the prompt is manual logic too* (it names 5
  example categories), so report its size and maintenance burden alongside template rules, per plan
  §5.5. Do not claim "0 vs N".

The honest claim is **growth**: the template's authored burden **scales with the feature/category
set**, while the prompt's is roughly **fixed**.

### Part B — flexibility / compositional-generalization stress test (the key experiment)

Formalize your "A+C+D not in the predefined set" idea as **systematicity** (compositional
generalization for D2T). Build a **development / held-out split** where the held-out set contains
signed feature combinations (and one genuinely new feature with metadata) absent from development.
Freeze template code, the taxonomy, and the prompt **before** opening held-out results.

For each held-out case, record **per method**:

1. **Edits required to handle it** — code / rule / map / prompt changes. Expected: template needs a
   new map entry (+ maybe new category, label, detection phrase) for a new feature → N>0; LLM → 0.
2. **Compositional fidelity** — run the *existing fidelity + interpretation panel* on the new-case
   output. **This is essential:** the D2T literature shows neural models also hallucinate/omit on
   unseen combinations, so we must verify the LLM stays faithful, not merely that it "runs".
3. **Succeeds without modification?** — boolean.

Stress cases (plan §6 + literature): unseen combination of known features; a new feature with
metadata; only one direction populated; unusually short / long factor lists; missing/unrateable
factors; renamed display labels.

**Headline metric:** *edits-per-held-out-case* (template N>0 for new feature/category; LLM 0) **paired
with** *compositional fidelity* (the LLM's advantage only counts if the zero-edit output is also
faithful). This is the cleanest, fully-automatic, clinician-free evidence of LLM value.

---

## What this lets you claim (and not)

- **Can claim (automatic):** the LLM aggregates more fluently (Layer A + G-Eval), is non-redundant
  across stages, and handles unseen feature combinations with **zero code changes while staying
  faithful** (Metric 5B). The template achieves comparable grouping only at the cost of a taxonomy +
  rules that must be extended per new feature.
- **Cannot claim (needs clinicians):** improved clarity, usefulness, actionability, or trust. State
  this as out of scope, per the plan.
- **Decision rule (plan §11):** if the LLM improves synthesis proxies + G-Eval while non-inferior on
  fidelity/completeness, claim a *constrained synthesis benefit*; if it wins the
  compositional-generalization test, claim the specific *maintenance/adaptation advantage* with the
  measured edit counts. Ties → template as primary, LLM optional.

---

## Suggested build order

1. **Metric 5A + 5B first** — fully deterministic, no judge model, and it is the strongest clean
   claim. We already have the static counts and the three arms; the held-out split + edit/fidelity
   logging is small.
2. **Metric 4 Layer A** — deterministic proxies (aggregation rate, compression, contrast,
   non-redundancy) reusing the frozen taxonomy and the prose parser.
3. **Metric 4 Layer B (G-Eval)** — last, needs a judge model and careful bias controls; secondary.

---

## References (verify author/year before final paper)

- **Reiter E., Dale R.** *Building Natural Language Generation Systems.* Cambridge University Press,
  2000. (NLG pipeline; **aggregation**; template limitations.)
- **van Deemter K., Krahmer E., Theune M.** *Real vs. template-based NLG: a false opposition?*
  Computational Linguistics, 2005. (template-vs-generation framing.)
- **Liu Y. et al.** *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.* EMNLP 2023.
  (reference-free LLM-as-judge; coherence/fluency/consistency/relevance.) *(verify)*
- **Reference-free Evaluation Metrics for Text Generation: A Survey.* arXiv:2501.12011, 2025.
  (distinct-n, self-BLEU, redundancy, entity-grid coherence.) *(verify)*
- **Barzilay R., Lapata M.** *Modeling Local Coherence: An Entity-Based Approach.* ACL 2005 / CL 2008.
  (entity-grid coherence.) *(verify)*
- **Mehta S. et al.** *Compositional Generalization for Data-to-Text Generation.* Findings of EMNLP
  2023 (arXiv:2312.02748). (unseen predicate combinations; cluster-and-generate; faithfulness under
  systematicity.) *(verify authors)*
- **SPOR: A Comprehensive and Practical Evaluation Method for Compositional Generalization in
  Data-to-Text Generation.* arXiv:2405.10650, 2024. (systematicity + other compositional aspects.)
  *(verify)*
- **XAI natural-language-explanation trustworthiness** — e.g. *LExT: Towards Evaluating
  Trustworthiness of Natural Language Explanations* (arXiv:2504.06227); studies finding NL
  explanations outperform raw SHAP for clinical decisions. (motivates the LLM approach and the
  faithfulness/plausibility split.) *(verify)*
