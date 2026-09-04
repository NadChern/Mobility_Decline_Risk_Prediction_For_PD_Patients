# Pilot results

This folder contains the completed non-synthesis metrics for the matched 30-patient pilot. The
shared analysis input is `pilot_reports.csv`, with one report from each method for every patient:
generic template, domain template, and Gemini.

| Metric | What it checks | Start here |
| --- | --- | --- |
| Completeness | Required evidence and report content are present | [`completeness/report.md`](completeness/report.md) |
| Fidelity | Labels, values, scales, ranks, and prose stay grounded in the supplied evidence | [`fidelity/report.md`](fidelity/report.md) |
| Structure | Required sections, table layout, order, and formatting | [`structure/report.md`](structure/report.md) |
| Unsupported information | Added, duplicated, altered, or causal content | [`unsupported_information/report.md`](unsupported_information/report.md) |
| Complexity | Authored machinery and unseen-feature maintenance burden | [`complexity/report.md`](complexity/report.md) |

Most metric folders contain `patients.csv` for patient-level results, `summary.csv` for aggregates,
and `report.md` for the readable result. Complexity uses design-level tables instead.

## Synthesis evaluation

Synthesis is evaluated separately in [`../synthesis_v2/RESULTS.md`](../synthesis_v2/RESULTS.md).
Its design has three main parts:

1. **Part 1 — taxonomy-conditioned comparison:** compare the domain template and an LLM given the
   same taxonomy, patient by patient, using structural grouping metrics and a paired bootstrap.
   Blinded paired judges are a separate, currently pending panel.
2. **Part 2A — map-free evaluation:** measure how historical LLM groupings created without the
   taxonomy agree with the reference taxonomy, then author-adjudicate non-exact groupings.
3. **Part 2B — held-out flexibility:** test whether each method handles unseen factors or groupings
   without task-specific changes while remaining faithful to the evidence.

These are pilot-development results, not independent clinical validation.
