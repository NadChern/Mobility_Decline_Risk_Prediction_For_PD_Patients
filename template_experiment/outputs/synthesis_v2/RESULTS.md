# Synthesis v2 results

This is the entry point for the current synthesis evaluation. Detailed patient-level data remain in
the linked folders. Historical checker and request versions are preserved under `history/` but are
not the current results.

## Current status

| Part | Status | Current result |
| --- | --- | --- |
| Dataset A checker | Complete | 30/30 saved taxonomy-conditioned Gemini reports pass checker v2.1. A checker pass is not clinical validation. |
| Part 1 structural comparison | Complete | Mean patient-level pairwise F1 is 1.000 for both methods. LLM − template = 0.000; stratified bootstrap 95% CI [0.000, 0.000], n=27, 10,000 replicates. The prespecified noninferiority rule was met against the −0.05 boundary. |
| Part 1 paired judges | Prepared, not run | 360 blinded calls are prepared in the panel-v3 directory. No full-panel results exist. |
| Part 2A map-free analysis | Complete | Macro pairwise F1 = 0.566 across 29 evaluable reports. Twenty non-exact reports were adjudicated: 7 granularity differences, 13 plausible alternative abstractions, and 0 unsupported/unclear. |
| Part 2 map-free judges | Prepared, not run | 186 blinded calls are prepared in the panel-v3 directory. No full-panel results exist. |
| Part 2B held-out flexibility | Generated and scored | Six held-out Gemini responses and deterministic comparators are saved. Maintenance time remains unmeasured; treat these as pilot-development results. |

## Interpretation

The taxonomy-conditioned LLM matched the domain template on the operational pairwise grouping
endpoint and met the prespecified noninferiority rule. This shows adherence to the supplied taxonomy,
not independent clinical correctness. Without the taxonomy, Gemini showed partial reference-taxonomy
agreement; the completed author review attributed the 20 non-exact reports to granularity or
plausible alternative abstraction rather than unsupported/unclear grouping. Blinded judge panels
remain pending and must not be described as completed evidence.

## Detailed evidence

- [Checker-v2.1 folder guide](checker_v2_1/README.md)
- [Dataset A checker-v2.1 report](checker_v2_1/report.md)
- [Part 1 structural and noninferiority report](checker_v2_1/part1_structural/report.md)
- [Part 1 patient-level results](checker_v2_1/part1_structural/patients.csv)
- [Part 1 noninferiority values](checker_v2_1/part1_structural/noninferiority.csv)
- [Part 2A map-free report and adjudication](checker_v2_1/part2a_map_free/report.md)
- [Part 2A patient-level results](checker_v2_1/part2a_map_free/patients.csv)
- [Part 2B held-out report](part2b_held_out/report.md)
- [Part 1 judge preparation](checker_v2_1/judge_part1_runtime_v3/report.md)
- [Part 2 judge preparation](checker_v2_1/judge_part2_runtime_v3/report.md)
- [Historical artifact guide](history/README.md)

## Scope

These are pilot-development findings. The supplied taxonomy is an experimental reference rather
than clinical ground truth. The full judge panel and independent final-test validation are not yet
complete.
