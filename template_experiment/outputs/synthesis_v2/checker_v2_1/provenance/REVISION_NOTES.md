# Checker v2.1: clinical-noun connector repair

The user confirmed the expansion of the noun allow-lists in `_near_member_connector` and
`_near_member_parenthesis` was intentional. Terms such as impairment(s), difficulty/difficulties,
dysfunction, function, symptoms, burden, deficits, problems, involvement, and complaints may
introduce explicit member lists. The checker still requires explicitly named members; it does
not reconstruct groups from the reference taxonomy.

## Verified impact

- Compared with the preserved `history/checker_v2/part2a_map_free/patients.csv`, only patient 40538's
  predicted pair set changes among the 30 historical map-free reports.
- Its supporting group is gait/mobility: FOG (Freezing of Gait), Gait, Mobility Summary.
- Its predicted pairs exactly match the reference; patient-level pairwise F1 is 1.0.
- Patient 40538 remains in the full-cohort metric denominator, but is removed from disagreement
  adjudication. The map-free disagreement worksheet has 20 rows, previously 21.
- All 60 deterministic template metric rows are unchanged relative to the saved checker-v2 audit.
- All 30 taxonomy-conditioned Dataset A reports still pass the local checker.

## Keep baseline versions explicit

| Artifact/version | Map-free macro pairwise F1 |
| --- | ---: |
| Legacy `outputs/pilot/synthesis/patients.csv` at verification | 0.5555008210 |
| Preserved checker-v2 map-free audit | 0.5316912972 |
| Current checker-v2.1 map-free audit | 0.5661740558 |

The current macro F1 has 29 evaluable patient values under the existing no-opportunity convention.
Do not attribute differences between different historical parser snapshots to a single change.
The one-patient-only change was verified specifically against the preserved checker-v2 audit.

## Provenance and verification

- Active checker: `interpretation-contract-v2.1-pilot`.
- Active parser: `explicit-group-claims-v2.1-pilot`.
- `manifest.json` stores source-data and code hashes for this revision.
- Previous audits, original generation records, and the original frozen protocol are preserved.
- The full test suite and all current reproducibility checks pass; exact counts are reported with
  each verification run rather than frozen in this note.
- Added regression tests for both connector and parenthesis forms across the clinical-noun list,
  P40538's exact match, and the one-patient change against the preserved audit.
- Prepared but not executed: 360 paired narrative judge calls and 186 map-free claim calls.
  Removing P40538 does not remove previously prepared calls: its old parser output had no claims
  to judge. Current total is therefore still 546 calls.
- No Gemini/OpenRouter calls were made during revision or verification.

This is a versioned pilot-development revision, not independent final-test validation. Human
extraction confirmation and a separate validation corpus remain required before final-test freezing.
