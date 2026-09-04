# Synthesis-v2 protocol files

This folder contains active study specifications and immutable inputs used to reproduce or audit
the synthesis-v2 experiment. These are not result files.

| File | Role |
| --- | --- |
| `taxonomy_conditioned_prompt.txt` | Defines the taxonomy-only addition to the matched generation prompt |
| `map_free_prompt.txt` | Frozen map-free generation prompt source |
| `dataset_b_evidence_packets.jsonl` | Frozen evidence packets for the retrospective map-free cohort |
| `held_out_cases.json` | Prespecified held-out flexibility cases |
| `parser_gold.csv` | Parser validation examples and expected outputs |
| `judge_prompt.txt` | Paired-judge prompt contract |
| `judge_rubric.md` | Paired-judge scoring rubric |
| `judge_runtime_v2.md` | Current judge transport, timeout, retry, and checkpoint rules |
| `adjudication_rubric.md` | Current author-adjudication rules for Part 2A disagreements |
| `synthesis_v2_manifest.json` | Generation-time frozen protocol and implementation hashes |

The manifest is intentionally preserved as the generation-time snapshot. Some implementation
hashes differ from the current checkout because later versioned checker, judge-runtime, and output
layout changes were made without rewriting the original generation record. Current checker hashes
are recorded in `../outputs/synthesis_v2/checker_v2_1/provenance/manifest.json`.

Superseded adjudication rubrics are stored with the corresponding synthesis history under
`../outputs/synthesis_v2/history/map_free_versions/`; only the active rubric remains here.
