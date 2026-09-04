# Template-versus-LLM pilot

This package compares three report-generation methods using the same 30-patient pilot cohort:

1. **Generic template** — copies the evidence into a fixed report and lists top factors.
2. **Domain template** — uses the same report shell and a frozen clinical taxonomy to group factors.
3. **Gemini** — historical temperature-0/run-1 reports, terminology-normalized for this pilot.

The current report contract is intentionally unchanged: no-fall reports show Stage-1 factors and
routed faller reports show Stage-2 severity factors. Completeness retains both displayed-contract
and full-pipeline stage-coverage metrics so the current report can be described from both views.
These metrics are diagnostics; they do not imply that a two-stage report must be implemented.

## Source layout

| Path | Purpose |
| --- | --- |
| `generic_template.py` | Generic deterministic report generator |
| `domain_template.py` | Domain-grouped deterministic report generator |
| `taxonomy.py` | Shared, versioned feature-domain taxonomy and scale metadata |
| `evaluation/build_pilot_dataset.py` | Builds and verifies the matched three-method pilot dataset |
| `evaluation/shared.py` | Shared report parser and expected-evidence construction |
| `evaluation/prose_checks.py` | Automatic interpretation-prose checks |
| `evaluation/{metric}.py` | One module per completed metric family |
| `evaluation/run_all.py` | Rebuilds the dataset and all completed results |
| `generate_dataset_a.py` | Prepares/runs taxonomy-conditioned Dataset A generation |
| `run_synthesis_v2.py` | Runs synthesis-v2 and deterministic reproducibility checks |
| `experimental/` | API-backed paired-judge and held-out evaluation runners |
| `docs/archive/` | Historical planning and review notes; not current instructions |

Legacy module names remain import-compatible through aliases in `__init__.py` and
`evaluation/__init__.py`, but new code should use the names above.

## Reproduce the completed pilot

From the project root:

```bash
./.venv/bin/python -m template_experiment.evaluation.run_all
```

To rebuild the experiment from a specific saved LLM generation file and regenerate every metric:

```bash
./.venv/bin/python -m template_experiment.evaluation.run_all \
  --source evaluation_results/llm_generations_grouping_v2.csv
```

This `grouping_v2` file is also the default source. It is the source recorded in every row of the
canonical `pilot_reports.csv`; the older `evaluation_results/gemini_results/llm_generations.csv`
is not used to rebuild this pilot.

To reuse an existing `outputs/pilot/pilot_reports.csv`:

```bash
./.venv/bin/python -m template_experiment.evaluation.run_all --skip-dataset
```

Building the dataset makes no LLM calls. It reads the saved Gemini generations, builds both
templates from the current artifact-derived packet, and fails if the historical and current patient,
stage, displayed factor names/order, SHAP direction, or six-decimal SHAP magnitudes differ.

## Output contract

`outputs/pilot/pilot_reports.csv` is the single analysis input. Each completed patient-level metric
folder contains only:

- `patients.csv` — one row per patient and method;
- `summary.csv` — category and overall summaries;
- `report.md` — concise human-readable results.

Synthesis and API-backed outputs are kept separately under `outputs/synthesis_v2/`; prepared or
pending panels are not completed results. See [outputs/README.md](outputs/README.md) for navigation.

## Synthesis v2

Versioned reviewer-facing synthesis outputs live under `outputs/synthesis_v2/`. Prepare all
deterministic artifacts without API calls:

Start with [outputs/synthesis_v2/RESULTS.md](outputs/synthesis_v2/RESULTS.md) for the consolidated
status, findings, and links to detailed evidence.

```bash
./.venv/bin/python -m template_experiment.run_synthesis_v2
```

After setting `GOOGLE_API_KEY`, add `--generate` for Dataset A and Dataset C. After setting
`OPENROUTER_API_KEY`, add `--judge` for the two configured judge families. Incomplete external runs
remain explicitly marked pending; they are never replaced with placeholder scores.

### Generation progress and recovery

```bash
./.venv/bin/python -m template_experiment.run_synthesis_v2 --generate
```

The same command starts or resumes Dataset A (30 matched patients) and Dataset C (six synthetic
cases). It reports the stage, patient/case ID, request attempt, response receipt, and checkpoint.
While a request is running, a heartbeat prints every 15 seconds. Gemini requests default to a
120-second timeout with SDK retries disabled; override with `--request-timeout 180` if needed.

Every received response is saved atomically before parsing. Successful and contract-flagged
responses are reused; missing, failed, or interrupted requests are attempted on the next explicit
run. A local processing failure reuses the saved raw response without calling Gemini again.
Collection stops on a request error rather than silently retrying. Do not run two collectors for
the same output directory; a lock prevents duplicate collection.

The authoritative checkpoints are `outputs/synthesis_v2/dataset_a/records.jsonl` and
`outputs/synthesis_v2/part2b_held_out/generations.jsonl`. Dataset A's `generation_log.csv` and
`status.md`, and Dataset C's `cases.csv` and `report.md`, are refreshed after each checkpoint.
Attempt history is retained in the JSONL files. Resume verifies the patient/case identities,
prompts, and models before reusing responses. Changed conditions require a separate output
directory, not overwriting the existing study run.

A hard interruption can occur after the provider accepts a request but before its response is
saved. Retrying that in-flight request can incur another charge; local checkpoints cannot guarantee
exactly-once provider billing. Previously saved responses are never requested again.

`contract_failed` means a response was received but failed its generation-time local check. It is
not a network failure. Those original flags are preserved even after a newer checker clears them.
The OpenRouter judge collectors now provide equivalent checkpoint/resume protection.

### Checker-v2.1 pilot revision

Run only the local recheck (no API calls):

```bash
./.venv/bin/python -m template_experiment.evaluation.recheck_dataset_a
```

The current pilot recheck is in
[checker_v2_1/report.md](outputs/synthesis_v2/checker_v2_1/report.md). All 30 Gemini
interpretations pass the revised checker; the original 20 flagged reports / 42 flags remain in
the unchanged generation checkpoint. `data/findings.jsonl` records matched source wording, group
membership, original flags, and new diagnostics. `data/records_rechecked.jsonl` provides derived
statuses for analysis; `provenance/source_records.jsonl` preserves the original snapshot.

The v2.1 checker uses case-insensitive, boundary-aware names, reviewed aliases, whitespace/dash
normalization, active/passive contrast wording, and bounded group-member extraction. It does not
use fuzzy matching or fill missing members from the taxonomy. Broad domain words such as
“cognition” alone do not establish that the report explicitly named MoCA.

Findings distinguish recognized content mismatches from uncertain extraction and formatting
issues. All received reports remain eligible for analysis/judging, including genuine contract
failures. Uncertain extraction pauses primary structural inference for review rather than
silently excluding the report or treating a parser miss as an established model error.

The main `run_synthesis_v2` command now rechecks first and writes **revised analysis and judge
preparation artifacts under `outputs/synthesis_v2/checker_v2_1/`**. It uses the revised
map-free worksheet when preparing map-free judges. Earlier metrics remain untouched so parser
revision effects can be audited. The original frozen protocol manifest is also preserved.

The team-facing Part 1 result, including both method means, the paired bootstrap confidence
interval, prespecified noninferiority rule, and links to reusable patient-level tables, is in
[part1_structural/report.md](outputs/synthesis_v2/checker_v2_1/part1_structural/report.md).
This analysis is computed without LLM-judge scores.

Each checker release is bound to source-data and checker-code hashes. For a changed corpus or
checker revision, use a new version/output directory rather than overwriting an existing release.
These are pilot-development regressions, **not independent final-test validation**. Human review
of the extraction and a separate validation corpus are still needed before freezing for final
testing. Do not modify the final test set or tune the checker on its results.

### Judge collection safety and next steps

Both judge panels use the judge-only `openrouter-judge-runtime-v2` transport and the panel-v3
request configuration, writing new collections to `checker_v2_1/judge_part1_runtime_v3/`
and `judge_part2_runtime_v3/`. GLM uses low reasoning effort in both panels; Qwen retains the
provider-default reasoning setting. Both retain the 2,048-token cap, 120-second default deadline,
and zero retries. Exact resolved settings are written to `runtime_profiles.json`. Earlier runtime-v2
smoke collections and the targeted GLM retest remain separate and unchanged. The earlier interrupted
collection is preserved, not resumed under changed settings. The runtime bypasses the installed
LangChain wrapper's millisecond/seconds mismatch and inherited retry defaults; Gemini generation
is unchanged. See [judge_runtime_v2.md](protocol/judge_runtime_v2.md).

Both panels freeze exact requests at first execution, save responses and metadata before parsing,
show waiting heartbeats, and resume missing requests. Changed prompts, models, cohort, temperature,
order, token/reasoning budget, runtime/dependency version, or timeout are rejected against
`frozen_requests.jsonl`. Per-request wall-clock timeout is 120 seconds by default; SDK and HTTP
retries are explicitly disabled. The configured output cap is 2,048 tokens, with provider-default
reasoning. An invalid structured response remains saved and
requires review; it is not automatically regenerated. Partial/invalid panels do not produce final
inferential summaries. `collection_status.md` and `raw_judgments.jsonl` show current progress.

After reviewing the pilot extraction, a deliberately small smoke test can use:

```bash
./.venv/bin/python -m template_experiment.run_synthesis_v2 --judge --judge-max-calls-per-panel 2
```

Resolve any checker/source manifest mismatch before executing this command; the audit guard
intentionally prevents judging under an unversioned parser revision.

This permits **up to four new paid calls total**, two per panel, covering both judges. Saved
responses count toward the full run. Inspect both models' raw JSON and any errors before running
the full `--judge` command. Currently the complete prepared workload is 360 paired narrative calls
plus 186 map-free claim calls (546 total); a resume only makes missing calls. Standalone judge
modules also support `--max-calls` and `--request-timeout` (that cap applies to their single panel).

The confirmed clinical-noun connector fix is versioned in
[REVISION_NOTES.md](outputs/synthesis_v2/checker_v2_1/provenance/REVISION_NOTES.md). Patient 40538 is
now an exact map-free match and is absent from the 20-row disagreement worksheet. It remains
in the full-cohort analysis. Earlier checker-v2 artifacts are historical snapshots, not active inputs.

### Part 2A author adjudication

Edit `outputs/synthesis_v2/checker_v2_1/part2a_map_free/disagreement_worksheet.csv` as the canonical
author worksheet. The offline runner updates the report in that same directory. The report leads
with author labels, separately counts complete reviews, and retains automatic routing only for
traceability. Adjudication explains disagreements; it does not alter primary pairwise scores.

Regeneration preserves labels, rationales, and drafts, snapshots prior worksheets/reports, and
marks decisions for re-review when their underlying inputs change. `report_manifest.json`
records report/parser versions, input/output hashes, and review completeness. It also records the
adjudication rubric version/hash and links an immutable rubric snapshot. All 20 reviews are now
complete: seven granularity differences, 13 plausible alternatives, and zero unsupported/unclear
under the pilot-developed rubric. This is author adjudication, not independent clinical validation.
The v1.1 clarification preserves all labels and adds missing-reference-pair explanations for 4098
and 130828. Version v1.1.1 changes only the documented artifact location. Earlier decisions and
rubrics remain preserved.

When a judge collector is already running, do not launch another collector in a different output
directory. The revised runtime serializes study judge execution and checks legacy collection locks
before calling an API. Older code cannot honor the new guard, so do not restart a legacy process.
The Part 2A reporting/rubric updates do not change judge request identities. Capped selection now
round-robins models: the first two calls per panel cover GLM and Qwen once each. Actual smoke success
requires all four final responses to be valid; a fast connection test or empty content is insufficient.

The first runtime-v2 smoke run returned three valid judgments and one truncated GLM paired reply.
A separate `glm_paired_low_v1` profile in `config.yaml` was then tested with the same paired prompt,
temperature, order, and 2,048-token cap, changing only reasoning effort to `low`. That one-call retest
returned valid JSON in 20.170 seconds (1,553 input / 268 output tokens, including 72 reasoning tokens).
Results and provenance are preserved in
`outputs/synthesis_v2/history/judge_smoke_v2/judge_part1_glm_low_v1_retest/`.

`python -m template_experiment.run_judge_retest` prepares this targeted retest without calling an
API; adding `--execute` permits at most one missing request and reuses a received response.
The ordinary full-panel runner now uses the panel-v3 model-specific settings. Preparation without
`--judge` makes no API calls. Do not copy or merge the earlier runtime-v2 smoke responses into the
new collection; they were obtained under different GLM reasoning settings.

The paired judge faithfulness sensitivity analysis reads the structural results in the same
versioned output tree, not the superseded pre-checker-v2 directory. As with generation, a request
interrupted before its response is saved can be billed again on explicit resume.

See [validation_readiness/NEXT_STEPS.md](outputs/synthesis_v2/checker_v2_1/validation_readiness/NEXT_STEPS.md)
for remaining human review and final-test requirements. A 60-report blinded annotation packet
(30 Gemini + 30 template paragraphs) is prepared there without model identities, parser answers,
or reference groups. The 16 additional synthetic challenge checks are assistant-authored
development checks, not a substitute for independent validation. Annotation drafts are preserved
on rerun. No paid calls were made while preparing these safeguards.

See [METRICS.md](METRICS.md) for definitions and [outputs/README.md](outputs/README.md) for the
result-file guide.
