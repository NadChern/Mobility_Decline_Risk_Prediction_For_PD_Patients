# Metric review: fidelity, completeness, unsupported information, structure

Scope: correctness of the calculations, duplication within/across the four modules, and whether
each metric is precise, focused, and covers what it claims — with particular attention to whether
**fidelity actually measures correctness of *all* parts of the report, including the Model
Interpretation prose.** All duplication and degeneracy claims below were verified numerically on the
current 30-patient pilot outputs.

---

## Progress checklist — updated 2026-08-26

Status of every recommendation in this review, plus work done today that extends beyond it. Each
item cites the section it traces to, so the checklist stays correlated with the findings.

### Done today — Fidelity family (traces to §2, §4, §6, §7)

- [x] **Trimmed fidelity to the conditional 5-metric core** (§2, §7): `outcome_correct`,
      `direction_accuracy`, `value_fidelity`, `scale_fidelity`, `rank_order_fidelity`. Deleted the
      family-mixing composites `factor_set_exact`, `factor_set_jaccard`, `direction_exact`,
      `value_end_to_end_fidelity`, `scale_end_to_end_fidelity`, `rank_order_exact`,
      `stage_routing_correct`.
- [x] **value/scale/direction now NaN-on-empty + dedup-by-first** (§6 zero-denominator; conditional
      definition), so a missing/duplicate/extra factor no longer leaks into fidelity.
- [x] **Evaluable denominator surfaced** in `summarize()` and the report as `value (n eval / n)`
      (§6). Prevents empty output from looking artificially faithful.
- [x] **Rank kept as pairwise concordance** (0..1, NaN < 2 pairs) rather than Spearman (§7).
- [x] **`prose_feature_fidelity` removed from fidelity** (§4). It was the mirror of the unsupported
      screen and mislabeled as interpretation fidelity.
- [x] **Interpretation-prose fidelity now measured** (§3, §4 — the "real gap") as an **automatic**
      3-metric panel: `prose_direction_fidelity`, `prose_grouping_fidelity`,
      `prose_value_descriptor_fidelity`. See deviation note below.
- [x] **Mutation/unit tests** for all of the above (`tests/test_template_metrics.py`,
      `tests/test_interpretation_fidelity.py`), 20 passing — covers §6 robustness cases
      (empty table → NaN; correct label in prose does not rescue a wrong field).

### Done today — beyond the original review (later design decisions)

- [x] **Frozen, literature-grounded grouping taxonomy** with citations (`interpretation_fidelity.py`
      `FEATURE_CATEGORY` / `CATEGORY_PROVENANCE`). Adopted from instruments, not authored.
- [x] **Domain-enriched template arm** (`domain_renderer.py`) so the prose comparison has three arms
      (generic template / domain template / LLM). Makes the grouping comparison fair and feeds the
      future scalability metric (rule counts).
- [x] **`outputs/README.md`** explaining fidelity in plain language + citations.
- [x] **`TODO_TWO_STAGE.md`** capturing the deferred two-stage / `(stage, factor)` work.

### Deviation from the original recommendation (documented, intentional)

- §4/§7 recommended `interpretation_claim_fidelity` via a **blinded human claim audit**. **No
  clinicians are available**, so we built a **conservative automatic panel** instead (direction /
  grouping / value-descriptor), and kept feature hallucination + causal language in the unsupported
  family. Limitation stated in the report: it flags only mechanically verifiable errors and cannot
  catch a subtly worded invented interaction.

### Outstanding — other three families still carry the flagged redundancy

- [x] **Completeness** (§1, §7) — done: deleted `required_stage_decision_coverage` (identical to
      `classification_field_coverage`, and redundant in the full-pipeline composite) and presence-only
      `required_section_coverage` (subsumed by the nonempty version); 12 → 10 metrics. Made the
      interpretation-coverage screens **group-aware** (a named category covers its member factors, so
      grouped LLM prose is not unfairly penalized) — a correctness fix. *Reasoned deviation from §7:*
      kept `interpretation_key_factor_coverage` / `opposing_evidence_coverage` in completeness (they
      measure whether required content is *present* in the prose = completeness, not synthesis
      *quality*) and kept `displayed_stage_coverage` (a completeness presence view; it uses subset,
      not the structure module's exact-equality check). Report now shows all three arms.
- [x] **Unsupported** (§1, §4) — done: dropped the fidelity rate-mirrors `unsupported_value_rate` /
      `unsupported_scale_rate` (kept raw counts, now reported as the *primary* evidence) and
      `factor_precision` (kept `unsupported_factor_rate`). Kept `prose_unsupported_feature_rate` as-is
      (its fidelity twin was already removed, so it is this family's natural metric — no rename).
      Also fixed a correctness bug: made the prose-unsupported check **group-aware** (a group label
      whose category is present is not misread as a hallucinated specific feature), which removed 6
      false positives on the domain template. Report now shows all three arms, counts-first.
- [x] **Structure** (§5) — done: dropped `structure_score` (added interpretable
      `n_components_passed` 0-8 diagnostic); `section_order_correct` now tests order only (no longer
      re-asserts the two-table count that `stage_template_correct` owns); guarded `_supporting_first`
      against unknown categories; report now shows all three arms.
- [ ] **Parser robustness** (§6): tolerant heading match + `parse_failure` flag so a cosmetic LLM
      heading rewrite is not scored as a content error.
- [ ] **Category-string coupling guards** (§6) in `structure._supporting_first` and
      `completeness._interpretation_coverage`.

### Outstanding — the value metrics (where the reviewer is actually answered)

- [~] **Metric 4 — Synthesis** (§4, §7, §8): **Layer A done** (`synthesis_comparison.py`). **Layer B
      G-Eval scaffold done** (`geval_synthesis_judge.py`): frozen rubric + 90 blinded items + key +
      fake-judge-tested scoring pipeline; **needs a separate judge model wired** (`JUDGE_MODEL`).
      Pilot finding: the rule-based domain template aggregates *more* than the LLM (51% vs 7%) — raw
      aggregation is not the LLM's advantage; reported honestly.
- [~] **Metric 5 — Complexity / scalability / flexibility** (§7, §8): **Part A done**
      (`complexity_scalability.py`). **Part B flexibility harness done** (`flexibility_stress_test.py`):
      deterministic arms run (unseen feature → domain needs 10 authored edits to group; generic never
      groups); **LLM arm is a pluggable hook (`LLM_GENERATOR`) needing generation** to confirm
      zero-edit flexibility + compositional fidelity. Frozen dev/held-out split on real ML artifacts
      still outstanding.
- [ ] **Two-stage, interpretation-only report** (cross-stage integration) — see `TODO_TWO_STAGE.md`.

**Correlation check:** every "done" item above traces to a section in this review (§2/§4/§6/§7) or is
an explicitly-labeled later extension; every original recommendation not yet done is listed under
Outstanding with its section. The one intentional divergence (automatic panel vs. blinded audit) is
called out. No finding in §1–§8 is silently dropped.

---

## 0. Headline findings

1. **Massive cross-module redundancy.** Six metric pairs are exact algebraic mirrors of each other
   (verified `allclose` on the pilot). You are reporting the same quantity two or three times under
   different family names.
2. **~35 of ~43 metrics are constant at 1.0** across all 60 rows, so they carry zero discriminative
   signal in the pilot. This is expected (the template is rendered from the same evidence packet the
   ground truth is built from), but it means the four families as currently written **cannot answer
   the reviewer's question.** They confirm "both methods reproduce the evidence table faithfully" —
   which was never in dispute.
3. **The one place the LLM could win — the Interpretation paragraph — is essentially unmeasured for
   fidelity.** `prose_feature_fidelity` only checks whether the *feature names* mentioned are in the
   supplied set. It cannot detect a wrong direction, a wrong value, or an incorrect synthesis claim
   in prose. So "is the presented info correct for the model-interpretation part?" — currently, **no,
   we are not measuring that.**
4. Several metrics are **composites that mix families** (a fidelity metric that silently folds in
   completeness + de-duplication), which defeats the whole point of the README's "keep four families
   separate" boundary.

---

## 1. Confirmed exact duplications (delete one side of each)

Each of these was checked with `np.allclose` on the pilot patient-level CSVs and is an identity, not
a coincidence of this dataset:

| Metric A | Metric B | Relationship | Recommendation |
| --- | --- | --- | --- |
| `unsupported.factor_precision` | `unsupported.unsupported_factor_rate` | `A = 1 − B` | Keep **one** (keep the rate + raw count; drop precision, or vice-versa). |
| `unsupported.unsupported_value_rate` | `fidelity.value_fidelity` | `A = 1 − B` | Same number in two families. Keep value fidelity in **fidelity**; drop the unsupported-value *rate* (keep only a raw count if you want counts). |
| `unsupported.unsupported_scale_rate` | `fidelity.scale_fidelity` | `A = 1 − B` | Same as above. |
| `unsupported.prose_unsupported_feature_rate` | `fidelity.prose_feature_fidelity` | `A = 1 − B` | Same number in two families. Pick one home (see §4 — it belongs in *unsupported*, not fidelity). |
| `completeness.required_stage_decision_coverage` | `completeness.classification_field_coverage` | `A = B` (identical) | **Within the same module.** Delete one outright. |
| `completeness.displayed_stage_coverage` | `structure.stage_template_correct` | equal on pilot; both check the two direction sections | Collapse to one canonical "stage tables present & correct" check (see §5). |

Net: at least **5 columns are pure redundancy** and can be deleted with zero information loss.

---

## 2. Composite metrics that mix families (the boundary leaks)

The README insists fidelity, completeness, unsupported, and structure stay separate. These metrics
violate that by bundling several concepts into one number, which makes them un-interpretable and
double-counts errors:

- **`fidelity.factor_set_exact` / `factor_set_jaccard`** — "displayed set == supplied set" is
  *recall* (a missing factor lowers it → completeness) **and** *precision* (an extra factor lowers it
  → unsupported). It is not fidelity. Fidelity should mean "given a factor is shown, is what's shown
  about it correct." Recall lives in completeness (`factor_recall`), extra-factor rate lives in
  unsupported. **Recommend: drop the factor-set metrics from the fidelity module** (they duplicate
  `completeness.factor_recall` + `unsupported.unsupported_factor_rate`).
  - Also `factor_set_exact == (factor_set_jaccard == 1)` exactly, so even within fidelity these two
    are the binary/graded pair — keep at most one.

- **`fidelity.direction_exact`** = `factor_set_exact` AND (every count == 1, i.e. de-duplication,
  which is `unsupported.duplicate_factor_rate`'s job) AND `direction_accuracy == 1`. It folds
  completeness + unsupported + fidelity into one boolean. **Recommend: delete; keep the graded
  `direction_accuracy`** as the clean fidelity measure.

- **`fidelity.value_end_to_end_fidelity` / `scale_end_to_end_fidelity`** — "every expected factor
  appears exactly once with the correct value." That is (completeness: present) + (unsupported: not
  duplicated) + (fidelity: correct). The pure-fidelity version is already there as the *conditional*
  `value_fidelity` / `scale_fidelity`. **Recommend: keep the conditional ones in fidelity; move the
  end-to-end variants out** (or drop — completeness's `expected_table_field_completeness` +
  duplicate rate already cover the other two ingredients).

- **`fidelity.stage_routing_correct`** = `strict_outcome_correct` (which is *already* the separate
  `outcome_correct` metric — verified `stage_routing ≤ outcome_correct` always) AND the
  direction-set check (which is `structure.stage_template_correct` /
  `completeness.displayed_stage_coverage`). So this single metric re-uses one fidelity metric and one
  structure metric. **Recommend: delete; it is `outcome_correct AND stage_template_correct`.**

After removing the composites, the fidelity module collapses from 13 columns to a clean core:
`outcome_correct`, `direction_accuracy`, `value_fidelity`, `scale_fidelity`, `rank_order_fidelity`
(+ optionally the `*_exact` binary of one of them as a strict companion). That is what "is the shown
content correct?" actually means.

---

## 3. Degenerate (constant) metrics in the pilot

Only these columns vary at all across the 60 rows; **everything else is a flat 1.0**:

- `fidelity.scale_fidelity` / `scale_end_to_end` (0.998 — a single Gemini scale-text mismatch)
- `completeness.required_stage_evidence_coverage` (0.667 — purely the structural routed-faller quirk)
- `completeness.interpretation_key_factor_coverage`, `opposing_evidence_coverage` (lexical prose
  screens)
- `completeness.overall_full_pipeline_complete` (0.333 — driven entirely by the stage-evidence quirk)
- `unsupported.unsupported_scale_rate` (mirror of scale_fidelity), `automated_..._free`
- Structure: **all ten constant at 1.0.**

Implication for the write-up: **do not present 40 near-identical 100% rows as if they were 40
independent pieces of evidence.** A reviewer will (correctly) read a wall of 100%/100% as "the
metrics can't tell the two apart." Collapse to the handful of metrics that can actually move, and be
explicit that table/format/value fidelity is expected to tie *by construction* — the template copies
the same packet the ground truth is built from.

---

## 4. The real gap: interpretation-prose fidelity is not measured

This is the most important correctness issue, and it maps directly onto your synthesis argument.

`fidelity.prose_feature_fidelity` = (feature names mentioned in the interpretation that are in the
supplied set) / (feature names mentioned). It is a **name-membership / traceability** check. It
**cannot** detect:

- a prose sentence that puts a factor in the **wrong direction** ("MoCA pushed toward higher
  severity" when it mitigated);
- a prose sentence that **misstates a value or scale**;
- an **unsupported interaction or causal claim** ("high UPDRS-III combined with low MoCA *causes*…");
- a **contradiction of the predicted class** stated in prose.

So it is mislabeled: it is a prose **precision/grounding screen**, and it is the exact complement of
`unsupported.prose_unsupported_feature_rate`. It should not live in the fidelity table under the name
"prose feature fidelity," because that name promises interpretation fidelity you are not delivering.

Recommendations:

1. **Rename** `prose_feature_fidelity` → `prose_feature_grounding` (or drop it and keep only the
   unsupported-side rate — they are the same number). Stop presenting it as interpretation fidelity.
2. **Interpretation fidelity requires the blinded claim audit**, which you already scaffold
   (`prose_claim_audit_blinded.csv`) but only inside the *unsupported* module. Fidelity of the prose
   (directional correctness, value correctness, licensed vs. unlicensed claims) has **no automatic
   proxy** — the plan (§5.1) already says "table-level matching is not sufficient to establish prose
   fidelity." Make the claim audit produce a *fidelity* number (fraction of checkable atomic claims
   that are correct), not only an unsupported-rate number, and cite it in the fidelity report.
3. This is also where your **synthesis metric (#4)** belongs and is currently absent — none of the
   four modules score integration/grouping/contrast. The pilot's `interpretation_key_factor_coverage`
   / `opposing_evidence_coverage` are *coverage counts of which names appear*, explicitly not
   synthesis quality (the README even says so). Build the frozen synthesis rubric from plan §5.4 as a
   fifth module; do not let coverage stand in for it.

---

## 5. Structure module: your `structure_score` vs `overall_structure_compliant` question

You asked specifically why both exist. They are two aggregations of the **same 8 components**:

- `structure_score` = mean of the 8 booleans (graded, 0–1).
- `overall_structure_compliant` = AND of the 8 (all must pass).

They are not exact duplicates (mean vs. all-pass), but reporting **both plus all 8 components** is
redundant, and `structure_score` is the weaker of the two: it is the arithmetic mean of *unlike*
binary checks (averaging "title format" with "row numbering" produces a number with no natural
interpretation — 0.875 doesn't mean anything clinically or statistically). 

Recommendation: **drop `structure_score`.** Keep the 8 per-component pass rates (those are
interpretable and diagnostic) plus `overall_structure_compliant` as the single headline. If you want
a graded summary for a significance test, use "number of components passed" as an explicit ordinal,
not a mean-of-booleans dressed up as a percentage.

Other structure notes:
- `section_order_correct` re-checks section *presence* (title/prediction/interpretation/note), which
  completeness's `required_section_coverage` also checks. Structure should assume presence and test
  *order only*; let completeness own presence. Minor double-count.
- `stage_template_correct` is the clean canonical "two correct direction sections" check — make the
  fidelity and completeness modules **reference this one** instead of each recomputing it (see §1).

---

## 6. Correctness / robustness issues in the calculations

Not wrong on the pilot, but they will bite on the full cohort or on freer LLM output:

- **Heading-parse brittleness (`metric_utils.direction_from_heading`, `parse_report`).** Direction
  sections are found only if the heading *starts with* "factors pushing the prediction toward" with
  the exact severity words. Any cosmetic rewrite by the LLM ("Factors that push toward a recurrent
  fall") → `direction = None` → those rows are silently dropped → the LLM is penalized on fidelity
  **and** completeness for a purely stylistic difference. On temp-0 Gemini it happened not to fire,
  but for the real experiment either (a) constrain the prompt to emit exact headings, or (b) make the
  matcher tolerant, and log a `parse_failure` flag so a parsing miss is never scored as a content
  error. Right now a parser miss is indistinguishable from a real omission.
- **Category string coupling.** `structure._supporting_first` indexes a dict by
  `category` and `completeness._interpretation_coverage` branches on `category == "moderate"`. Both
  `KeyError`/misclassify if a category label changes or a fourth class appears. Freeze the category
  vocabulary or guard with `.get`.
- **`_content_fidelity` / conditional metrics** use the fallback `1.0 if not expected else 0.0`. That
  is defensible, but document it: a patient with an empty expected set scores a *perfect* 1.0, which
  can inflate means if any such patients exist in the full cohort. State the zero-denominator
  convention explicitly (the plan §5.3 asks for this and it is not yet in the code comments).
- **`causal_claim_sentence_rate`** is presence-detection of causal *lexicon* (`cause`, `leads to`,
  `risk factor`, …), not a judgment of whether a causal claim is *licensed*. That's fine as a screen,
  but label it as such — a legitimately-worded "X is a recognized risk factor for falls" would be
  flagged. Keep it, but it is a red-flag counter, not a correctness metric.

---

## 7. Recommended minimal, non-overlapping metric set

One number per concept, each in exactly one family:

**Fidelity — "is the shown content correct?"** (conditional on presence)
- `outcome_correct` (label)
- `direction_accuracy` (graded)
- `value_fidelity` (conditional)
- `scale_fidelity` (conditional)
- `rank_order_fidelity` (graded)
- **[new]** `interpretation_claim_fidelity` from the blinded audit (correct atomic claims / checkable)

Delete from fidelity: `factor_set_exact`, `factor_set_jaccard`, `direction_exact`,
`value_end_to_end_fidelity`, `scale_end_to_end_fidelity`, `stage_routing_correct`,
`prose_feature_fidelity` (→ rename/move to unsupported as grounding).

**Completeness — "is required content present?"**
- `factor_recall`
- `nonempty_section_coverage` (drop presence-only `required_section_coverage`)
- `expected_table_field_completeness`
- `classification_field_coverage` (drop the identical `required_stage_decision_coverage`)
- `required_stage_evidence_coverage` (the full-pipeline stage check — this one genuinely varies)
- one composite (`overall_full_pipeline_complete`)

Move out of completeness: `interpretation_key_factor_coverage`, `opposing_evidence_coverage` → these
are synthesis-coverage, put them in the synthesis module (or drop in favor of the rubric).
Reference the canonical stage check instead of recomputing `displayed_stage_coverage`.

**Unsupported — "was anything added / altered?"** (report raw counts as primary, rates secondary)
- `unsupported_factor_count`/rate (extra factors)
- `duplicate_factor_count`/rate
- value/scale *alteration counts* (as the complement of fidelity — report the count, don't re-report
  the rate that mirrors fidelity)
- `prose_feature_grounding` (the renamed ex-`prose_feature_fidelity`)
- `causal_claim_sentence_count` (screen)
- **[new]** `unsupported_prose_claim_rate` from the blinded audit

**Structure — "is the format compliant?"**
- 8 component pass-rates + `overall_structure_compliant`. Drop `structure_score`.

**Fifth family (missing): Synthesis** — frozen rubric (prioritization, integration, grouping,
contrast, coherence, non-redundancy, constraint-preservation), blinded paired human rating. This is
the family that actually addresses the reviewer, and none of the current four cover it.

---

## 8. Bottom line for the reviewer response

- The four automatic families are correct on the mechanics but **over-built and largely
  self-duplicating**, and — by design — they **tie at ~100%** because the template copies the same
  packet. They are necessary as a *non-inferiority* check ("the LLM doesn't lose fidelity/
  completeness/format"), and that is exactly how to frame them: evidence of a **tie**, not evidence
  of value.
- The **value argument lives entirely in the two families that are not yet implemented** — synthesis
  quality (#4) and complexity/scalability/flexibility (#5), including the unseen-combination stress
  test. Prioritize those. The current modules should be trimmed to the minimal set above so the
  fidelity/completeness tie is stated cleanly in one small table, freeing the paper's weight for
  synthesis + scalability where the LLM can actually differentiate.
