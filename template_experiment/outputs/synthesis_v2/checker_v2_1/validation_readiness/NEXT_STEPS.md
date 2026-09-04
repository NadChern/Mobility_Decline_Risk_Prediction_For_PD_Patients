# Synthesis evaluation: readiness and next steps

Additional synthetic challenge checks: 16/16 expected outcomes matched. These are assistant-authored development checks, not independent validation.

Human annotations completed: 0/60. Annotations are not filled in by the assistant.

## Current limitations

The checker still routes unfamiliar introductions such as `gait and mobility: Gait and FOG` or `consisting of` to extraction review. An unfamiliar expression must not be reported as an established report error. Overall clinical correctness and every possible phrasing are not validated.

## Before final testing

- [ ] Human reviewer labels the blinded packet; compare with parser extraction and resolve disagreements.
- [ ] Obtain a separately authored/labeled validation corpus not used for checker tuning. The pilot and these synthetic cases do not satisfy this requirement.
- [ ] Predefine acceptance criteria, review fallback, dataset size/IDs and reporting denominators.
- [ ] Freeze the checker, taxonomy, prompts, patient selection and analysis settings before final generation.
- [ ] Do not tune the checker on final-test results; report uncertain cases with the frozen fallback.

## Next paid step (not executed)

After reviewing the pilot extraction, run a small capped judge smoke test, inspect both judges' structured replies, then explicitly authorize the full panel. Current prepared workload: 360 paired narrative calls plus 186 map-free claim calls = 546 calls. Costs are not estimated without current provider pricing and actual token use. The same commands resume saved requests.

