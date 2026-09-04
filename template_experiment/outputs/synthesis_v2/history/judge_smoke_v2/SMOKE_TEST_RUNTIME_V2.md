# Judge runtime v2 smoke test

Run: 2026-09-04, 08:25–08:28 UTC. Four new API requests; one attempt each. No full panel executed.
The verified orphaned legacy process (PID 95074) was terminated with user approval before this run.
Its historical checkpoint remains unchanged; its old in-progress marker is not evidence of a live job.

Settings: 2,048 maximum output tokens; provider-default reasoning; 120-second wall-clock deadline;
explicitly disabled SDK and HTTP retries. Exact settings and request identities are preserved in
each runtime-v2 panel's `frozen_requests.jsonl`.

| Panel | Judge | Elapsed seconds | Input tokens | Output tokens | Reasoning tokens | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Paired reports | GLM 5.3 Flash | 61.157 | 1553 | 2048 | 2045 | Truncated; empty final content |
| Paired reports | Qwen3.8 Flash | 24.806 | 1731 | 1604 | 1449 | Valid final JSON |
| Map-free claim | GLM 5.3 Flash | 35.080 | 463 | 1285 | 1146 | Valid final JSON |
| Map-free claim | Qwen3.8 Flash | 13.038 | 541 | 784 | 690 | Valid final JSON |

Total: 4,288 input and 5,721 output tokens, including 5,330 reasoning tokens. Dollar cost was not
present in the saved SDK responses; it is unknown, not zero. Provider billing can be checked using
the saved request IDs. The old interrupted request may have incurred an additional charge.

## Assessment

Transport and checkpoint collection completed without a hang or retry. All four responses arrived
before the deadline; this run did not exercise live timeout cancellation (covered by offline tests).
The schema smoke test did not fully pass: only three of four requests returned valid judgments.
GLM paired judging exhausted the token budget on reasoning (`finish_reason=length`) and returned
no final answer. It is retained as `contract_failed` / `truncated_output`, not parsed from reasoning
and not automatically resampled.

The three valid responses have correct schema and permitted score ranges. They are pilot smoke
outputs, not evidence of panel-level judge reliability or clinical validity. No final panel summary
is computed from this partial collection.

## Next decision

Do not start the full panel. Verify supported GLM reasoning controls and decide on a bounded
reasoning/output configuration for paired judging. Freeze any changed settings in a new request
version before an explicitly authorized follow-up smoke test. Keep these four responses and do
not silently reuse them as if they came from changed settings.

Raw responses: `judge_part1_runtime_v2/raw_judgments.jsonl` and
`judge_part2_runtime_v2/raw_judgments.jsonl`. Both include full saved SDK envelopes, request IDs,
finish reasons, usage, parsed results where valid, and attempt durations.
