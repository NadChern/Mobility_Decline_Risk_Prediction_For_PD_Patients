# Dataset A status

Protocol: `synthesis-v2.1-prompt-twin`. Domain-template records: 30 complete.

Generation-time statuses (preserved; later checker revisions are audited separately):

- complete: 10
- contract_failed: 20

Contract-failed responses were received and saved; they are not API failures. The latest checker-v2.1 assessments are in `../checker_v2_1/report.md`; rechecking does not rewrite these original generation flags. Rerunning `python -m template_experiment.run_synthesis_v2 --generate` resumes missing/failed requests and reprocesses saved raw responses locally if necessary. Request attempt history is in `records.jsonl`.
