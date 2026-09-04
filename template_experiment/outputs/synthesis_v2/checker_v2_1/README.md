# Checker v2.1 output guide

This is the current synthesis-checker release. Start with `report.md` for the Dataset A checker
result, then use these folders for the detailed analyses.

| Path | Contents |
| --- | --- |
| `data/` | Current rechecked records and row-level checker findings |
| `provenance/` | Current source snapshot, checker/code manifest, revision notes, and reproducibility checksums |
| `parser_validation/` | Parser validation cases and report |
| `part1_structural/` | Template-versus-taxonomy-conditioned-LLM structural results and noninferiority analysis |
| `part2a_map_free/` | Map-free scores, canonical author worksheet, current report, and current rubric snapshot |
| `validation_readiness/` | Blinded human-review packet and final-test readiness materials |
| `judge_part1_runtime_v3/` | Current paired-judge requests; prepared but not fully run |
| `judge_part2_runtime_v3/` | Current map-free judge requests; prepared but not fully run |

Older checker results, prior Part 2A report versions, pre-layout manifests, and paid smoke-test
responses are under `../history/`. They are provenance, not current results.
