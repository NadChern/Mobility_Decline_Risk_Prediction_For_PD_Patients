# Judge runtime v2

This is an operational pilot revision, not a change to the taxonomy, parser, author adjudications,
report-generation prompt, or judge rubric. Previous requests and interrupted attempts are retained.

## Confirmed defects in the prior client path

The installed LangChain OpenRouter wrapper interprets `request_timeout` in milliseconds, while the
project passes seconds. A 90-second request became 90 milliseconds. Setting `max_retries=0` in that
wrapper omits the SDK retry configuration, leaving the SDK's default backoff behavior enabled.
Local dependency-level tests confirmed this boundary mismatch. The earlier hang cannot be attributed
to excessive reasoning without a received response and its token metadata.

## Revised request contract

- Judge-only direct asynchronous OpenRouter SDK calls; generation clients are unchanged.
- Convert seconds to milliseconds explicitly; default 120-second wall-clock deadline.
- Cancel the local async operation at the deadline and close its HTTP client. No detached thread.
- Explicitly disable retries at SDK client, SDK call, and HTTP transport levels.
- Output cap: `evaluation.synthesis_judges.max_tokens`, initially 2,048.
- Reasoning: `evaluation.synthesis_judges.reasoning`, initially null (provider default).
- Accept only final content that passes the panel's existing JSON schema. Never promote reasoning
  text to a judgment. Empty, token-limited, non-final, and invalid-JSON responses are retained as
  distinct contract failures and are not automatically regenerated.
- Save the full response envelope, request ID, finish reason, input/output/reasoning token usage,
  and provider-reported cost where supplied. Missing cost is unknown, not zero.
- Freeze runtime settings and dependency versions with each request. Changed settings require a
  new output directory; never mix responses under different budgets silently.
- Preserved smoke outputs: `outputs/synthesis_v2/history/judge_smoke_v2/`. Current panel-v3
  preparations live under `outputs/synthesis_v2/checker_v2_1/`.
- Serialize new study collectors and check existing legacy output locks before execution.

## Four-call smoke test

After the old collector has exited:

```bash
./.venv/bin/python -m template_experiment.run_synthesis_v2 --judge --judge-max-calls-per-panel 2
```

The cap applies to new requests per invocation. Capped selection round-robins models, covering one
GLM and one Qwen response in each panel on a fresh collection. A transport failure stops the run;
there is no automatic reissue. If all four responses are final, schema-valid, and have sensible
usage metadata, they can be reused in the later full panel under identical settings. A truncated
or empty response is a failed smoke outcome, not permission to raise the budget automatically.

Local cancellation cannot guarantee remote cancellation or prevent a charge already incurred.
An explicit resume of an interrupted request can be billed again. Inspect failures before resuming.
Passing the smoke test verifies transport and schema handling, not judge reliability or clinical
validity. Do not infer final panel results from four calls.

Offline verification covers real SDK request construction and single-attempt behavior with a mocked
HTTP transport, cancellation, metadata preservation, empty/reasoning-only output rejection,
token-limit detection, budget freezing, and cross-version locks. Exact suite counts belong to the
current verification run rather than this frozen runtime note.
