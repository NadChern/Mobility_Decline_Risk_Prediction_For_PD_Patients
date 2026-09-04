# Experiment outputs

Use this page to choose the correct result set.

| Folder | Purpose | Entry point |
| --- | --- | --- |
| `pilot/` | Completed completeness, fidelity, structure, unsupported-information, and complexity metrics on the matched 30-patient pilot | [`pilot/README.md`](pilot/README.md) |
| `synthesis_v2/` | Current multi-part synthesis evaluation, checker results, judge readiness, and held-out testing | [`synthesis_v2/RESULTS.md`](synthesis_v2/RESULTS.md) |

`pilot/pilot_reports.csv` is the shared 30-patient input for the completed pilot metrics. Current
synthesis results live under `synthesis_v2/checker_v2_1/`; generated reports and historical
provenance are kept inside the synthesis-v2 tree so they are not confused with the main pilot
metrics.

Read each `report.md` first. Use `patients.csv` when patient-level auditing or paired analysis is
needed, and `summary.csv` for aggregate tables. Files marked prepared or pending are not completed
results.
