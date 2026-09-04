# Map-free disagreement adjudication rubric

- Version: `adjudication-rubric-v1`
- Scope: Part 2A retrospective map-free reports (`disagreement_worksheet.csv`).
- Status: pilot-developed. Earlier per-row decisions are preserved as worksheet snapshots under
  `outputs/synthesis_v2/checker_v2_1/part2a_map_free/worksheet_snapshots/`.

This rubric judges each predicted grouping on its **actual wording**, not on which factor pairs it
contains. The reference taxonomy is one operational scheme; disagreement with it is not, by itself,
evidence of a wrong or misleading grouping.

## Per-group classification

Classify **each predicted group**, then label the report by its most severe group.

1. **granularity_difference** — a valid *narrower or broader slice within a single reference
   category*. No cross-category members. Example: the reference groups `gait_mobility(FOG, Gait,
   Postural Instability at Dx)`; the report names `gait_mobility(FOG, Gait)` and leaves the rest
   individual.

2. **plausible_alternative_abstraction** — the group *crosses* reference categories, **but the
   actual wording provides a defensible shared umbrella** and names its members. A broad label such
   as "motor features", "motor assessments", or "non-motor burden" over factors that genuinely share
   that broader construct is a defensible alternative cut. Naming both members explicitly under a
   broad label is **not** misleading synthesis. Use "shares a broader motor construct" rather than
   "subsumes" for separate historical variables (e.g. *Postural Instability at Dx*); reserve
   "encompasses" for a composite that literally contains the member (UPDRS-I for mood/autonomic/sleep
   items).

3. **unsupported_or_unclear** — the wording itself fails, for one of these reasons:
   - the **label does not fit** the members (e.g. calling a motor-complication factor "gait");
   - the report **asserts an unsupported relationship** (a causal or mechanistic claim the evidence
     does not support); or
   - the grouping **cannot be interpreted reliably**.
   If clinical defensibility is genuinely *uncertain*, retain this label **with an explicit
   uncertainty rationale**, rather than asserting the grouping is clinically wrong.

## Report-level label

- The report takes its **most severe** group label: `unsupported_or_unclear` >
  `plausible_alternative_abstraction` > `granularity_difference`.
- **Missed or partial groupings** (the report leaves a reference group's members individual, or omits
  a reference group) are recorded in the rationale, **not** used as the headline label, unless the
  report formed no defensible group at all.
- Every disputed group in a report must be addressed in its `author_rationale`.

## Notes on the FOG + UPDRS-IV case

Grouping freezing of gait (FOG) with motor complications (UPDRS-IV) under a broad motor label is a
*broad but defensible* grouping when both are named — not automatically misleading. The
motor-examination vs motor-complications distinction is real, but that distinction alone does not
make a broad "motor features" umbrella indefensible. It becomes `unsupported_or_unclear` only if the
wording misfits, asserts an unsupported relationship, or is uninterpretable.
