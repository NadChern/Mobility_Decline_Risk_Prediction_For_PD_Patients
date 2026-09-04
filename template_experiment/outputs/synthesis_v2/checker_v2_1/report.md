# Dataset A checker-v2.1 pilot recheck

Checker: `interpretation-contract-v2.1-pilot`; parser: `explicit-group-claims-v2.1-pilot`.

Reviewed 30 saved Gemini interpretations. Before: 20 flagged reports with 42 flags. After: 30 pass, 0 need review.

No API calls. Original prompts, responses, interpretation text, and generation flags are unchanged in the source checkpoint. The original records are also copied to `provenance/source_records.jsonl`; `data/records_rechecked.jsonl` contains derived checker statuses and `data/findings.jsonl` contains detailed matches.

A pass means no issue detected by this checker, not clinical correctness or a perfect synthesis score. Missing recognition is distinct from a recognized content mismatch. These are pilot-development results; independent human review and validation on a separate corpus remain required before final-test freezing.

| Patient | Original flags | New findings | Assessment |
| --- | ---: | ---: | --- |
| 3066 | 0 | 0 | pass |
| 3223 | 1 | 0 | pass |
| 3429 | 3 | 0 | pass |
| 3625 | 4 | 0 | pass |
| 3832 | 3 | 0 | pass |
| 4059 | 1 | 0 | pass |
| 41282 | 0 | 0 | pass |
| 50860 | 2 | 0 | pass |
| 116531 | 2 | 0 | pass |
| 186755 | 0 | 0 | pass |
| 3001 | 1 | 0 | pass |
| 3054 | 0 | 0 | pass |
| 3224 | 2 | 0 | pass |
| 3530 | 2 | 0 | pass |
| 3758 | 1 | 0 | pass |
| 4030 | 0 | 0 | pass |
| 4054 | 0 | 0 | pass |
| 4098 | 0 | 0 | pass |
| 40538 | 1 | 0 | pass |
| 179752 | 0 | 0 | pass |
| 3826 | 1 | 0 | pass |
| 100001 | 0 | 0 | pass |
| 101566 | 5 | 0 | pass |
| 102479 | 3 | 0 | pass |
| 130828 | 3 | 0 | pass |
| 190603 | 1 | 0 | pass |
| 218338 | 2 | 0 | pass |
| 224574 | 0 | 0 | pass |
| 241069 | 1 | 0 | pass |
| 292826 | 3 | 0 | pass |

## Patient 3223

Original flags:

- `missing_or_inexact_group=opposes_prediction:gait_mobility`

Rechecked explicit groups:

- supporting / gait_mobility: FOG (Freezing of Gait), Postural Instability at Dx
- opposing / gait_mobility: Gait, Mobility Summary

Recognized factor mentions (source text → factor):

- supporting: “FOG (Freezing of Gait)” → FOG (Freezing of Gait)
- supporting: “Postural Instability at Dx” → Postural Instability at Dx
- supporting: “Urinary Problems” → Urinary Problems
- supporting: “UPDRS-I Total” → UPDRS-I Total
- supporting: “UPDRS-III Total” → UPDRS-III Total
- opposing: “Gait” → Gait
- opposing: “Mobility Summary” → Mobility Summary
- opposing: “UPDRS-IV Total” → UPDRS-IV Total

New findings: []

Human confirmation: pending.

## Patient 3429

Original flags:

- `missing_singleton=Gait`
- `missing_singleton=Age`
- `missing_contrastive_outweigh_statement`

Rechecked explicit groups:

- supporting / gait_mobility: FOG (Freezing of Gait), Postural Instability at Dx
- supporting / autonomic_symptoms: Fainting, Postural Hypotension

Recognized factor mentions (source text → factor):

- supporting: “freezing of gait” → FOG (Freezing of Gait)
- supporting: “postural instability at diagnosis” → Postural Instability at Dx
- supporting: “fainting” → Fainting
- supporting: “postural hypotension” → Postural Hypotension
- supporting: “BMI” → BMI
- opposing: “MoCA” → MoCA
- opposing: “gait impairment” → Gait
- opposing: “age” → Age

New findings: []

Human confirmation: pending.

## Patient 3625

Original flags:

- `missing_or_inexact_group=supports_prediction:gait_mobility`
- `missing_singleton=Age`
- `missing_singleton=Urinary Problems`
- `missing_singleton=Rigidity at Dx`

Rechecked explicit groups:

- supporting / gait_mobility: FOG (Freezing of Gait), Gait, Postural Instability at Dx

Recognized factor mentions (source text → factor):

- supporting: “freezing of gait” → FOG (Freezing of Gait)
- supporting: “clinician-rated gait impairment” → Gait
- supporting: “postural instability at diagnosis” → Postural Instability at Dx
- supporting: “UPDRS-III Total” → UPDRS-III Total
- supporting: “BMI” → BMI
- opposing: “age” → Age
- opposing: “normal urinary function” → Urinary Problems
- opposing: “urine control problems” → Urinary Problems
- opposing: “rigidity at diagnosis” → Rigidity at Dx

New findings: []

Human confirmation: pending.

## Patient 3832

Original flags:

- `missing_or_inexact_group=supports_prediction:gait_mobility`
- `missing_or_inexact_group=opposes_prediction:gait_mobility`
- `missing_singleton=Age`

Rechecked explicit groups:

- supporting / gait_mobility: Gait, Postural Instability at Dx
- opposing / gait_mobility: FOG (Freezing of Gait), Mobility Summary

Recognized factor mentions (source text → factor):

- supporting: “postural instability at diagnosis” → Postural Instability at Dx
- supporting: “gait impairment” → Gait
- supporting: “UPDRS-I Total” → UPDRS-I Total
- supporting: “UPDRS-III Total” → UPDRS-III Total
- supporting: “UPDRS-IV Total” → UPDRS-IV Total
- opposing: “Mobility Summary” → Mobility Summary
- opposing: “freezing of gait” → FOG (Freezing of Gait)
- opposing: “age” → Age

New findings: []

Human confirmation: pending.

## Patient 4059

Original flags:

- `missing_or_inexact_group=supports_prediction:gait_mobility`

Rechecked explicit groups:

- supporting / gait_mobility: FOG (Freezing of Gait), Gait, Mobility Summary, Postural Stability

Recognized factor mentions (source text → factor):

- supporting: “FOG (Freezing of Gait)” → FOG (Freezing of Gait)
- supporting: “Gait” → Gait
- supporting: “Mobility Summary” → Mobility Summary
- supporting: “Postural Stability” → Postural Stability
- supporting: “UPDRS-III Total” → UPDRS-III Total
- opposing: “PD Duration” → PD Duration
- opposing: “UPDRS-I Total” → UPDRS-I Total
- opposing: “BMI” → BMI

New findings: []

Human confirmation: pending.

## Patient 50860

Original flags:

- `missing_singleton=Postural Instability at Dx`
- `missing_singleton=Age`

Rechecked explicit groups:

- supporting / gait_mobility: FOG (Freezing of Gait), Gait, Mobility Summary, Postural Stability

Recognized factor mentions (source text → factor):

- supporting: “Mobility Summary” → Mobility Summary
- supporting: “freezing of gait” → FOG (Freezing of Gait)
- supporting: “gait impairment” → Gait
- supporting: “postural stability” → Postural Stability
- supporting: “BMI” → BMI
- opposing: “postural instability present at diagnosis” → Postural Instability at Dx
- opposing: “age” → Age

New findings: []

Human confirmation: pending.

## Patient 116531

Original flags:

- `missing_singleton=Gait`
- `missing_singleton=Urinary Problems`

Rechecked explicit groups:

- supporting / gait_mobility: FOG (Freezing of Gait), Postural Instability at Dx

Recognized factor mentions (source text → factor):

- supporting: “freezing of gait” → FOG (Freezing of Gait)
- supporting: “FOG” → FOG (Freezing of Gait)
- supporting: “postural instability at diagnosis” → Postural Instability at Dx
- supporting: “Age” → Age
- supporting: “PD Duration” → PD Duration
- supporting: “UPDRS-I Total” → UPDRS-I Total
- opposing: “gait impairment” → Gait
- opposing: “urinary problems” → Urinary Problems
- opposing: “BMI” → BMI

New findings: []

Human confirmation: pending.

## Patient 3001

Original flags:

- `missing_or_inexact_group=opposes_prediction:gait_mobility`

Rechecked explicit groups:

- opposing / gait_mobility: FOG (Freezing of Gait), Gait

Recognized factor mentions (source text → factor):

- supporting: “Mobility Summary” → Mobility Summary
- supporting: “UPDRS-III Total” → UPDRS-III Total
- supporting: “Age” → Age
- supporting: “PD Duration” → PD Duration
- supporting: “UPDRS-IV Total” → UPDRS-IV Total
- opposing: “FOG” → FOG (Freezing of Gait)
- opposing: “Freezing of Gait” → FOG (Freezing of Gait)
- opposing: “Gait” → Gait
- opposing: “MoCA” → MoCA

New findings: []

Human confirmation: pending.

## Patient 3224

Original flags:

- `missing_or_inexact_group=supports_prediction:gait_mobility`
- `missing_or_inexact_group=opposes_prediction:gait_mobility`

Rechecked explicit groups:

- supporting / gait_mobility: Gait, Mobility Summary
- opposing / gait_mobility: FOG (Freezing of Gait), Postural Instability at Dx

Recognized factor mentions (source text → factor):

- supporting: “Gait” → Gait
- supporting: “Mobility Summary” → Mobility Summary
- supporting: “GDS-15” → GDS-15
- supporting: “UPDRS-I Total” → UPDRS-I Total
- supporting: “Age” → Age
- opposing: “FOG (Freezing of Gait)” → FOG (Freezing of Gait)
- opposing: “Postural Instability at Dx” → Postural Instability at Dx
- opposing: “UPDRS-IV Total” → UPDRS-IV Total

New findings: []

Human confirmation: pending.

## Patient 3530

Original flags:

- `missing_singleton=FOG (Freezing of Gait)`
- `missing_singleton=Urinary Problems`

Rechecked explicit groups:

No explicit multi-factor relation extracted; see the factor mentions below.

Recognized factor mentions (source text → factor):

- supporting: “UPDRS-III Total” → UPDRS-III Total
- supporting: “MoCA” → MoCA
- supporting: “Mobility Summary” → Mobility Summary
- supporting: “PD Duration” → PD Duration
- opposing: “freezing of gait” → FOG (Freezing of Gait)
- opposing: “FOG” → FOG (Freezing of Gait)
- opposing: “BMI” → BMI
- opposing: “urinary problems” → Urinary Problems

New findings: []

Human confirmation: pending.

## Patient 3758

Original flags:

- `missing_or_inexact_group=supports_prediction:gait_mobility`

Rechecked explicit groups:

- supporting / gait_mobility: Mobility Summary, Postural Instability at Dx

Recognized factor mentions (source text → factor):

- supporting: “Mobility Summary” → Mobility Summary
- supporting: “Postural Instability at Dx” → Postural Instability at Dx
- supporting: “UPDRS-III Total” → UPDRS-III Total
- supporting: “MoCA” → MoCA
- supporting: “GDS-15” → GDS-15
- opposing: “FOG (Freezing of Gait)” → FOG (Freezing of Gait)
- opposing: “Urinary Problems” → Urinary Problems
- opposing: “UPDRS-IV Total” → UPDRS-IV Total

New findings: []

Human confirmation: pending.

## Patient 40538

Original flags:

- `missing_or_inexact_group=supports_prediction:gait_mobility`

Rechecked explicit groups:

- supporting / gait_mobility: FOG (Freezing of Gait), Gait, Mobility Summary

Recognized factor mentions (source text → factor):

- supporting: “FOG (Freezing of Gait)” → FOG (Freezing of Gait)
- supporting: “Mobility Summary” → Mobility Summary
- supporting: “Gait” → Gait
- supporting: “GDS-15” → GDS-15
- supporting: “PD Duration” → PD Duration
- opposing: “UPDRS-III Total” → UPDRS-III Total
- opposing: “postural instability at diagnosis” → Postural Instability at Dx
- opposing: “Postural Instability at Dx” → Postural Instability at Dx
- opposing: “Urinary Problems” → Urinary Problems

New findings: []

Human confirmation: pending.

## Patient 3826

Original flags:

- `missing_singleton=Dopaminergic Therapy`

Rechecked explicit groups:

- supporting / gait_mobility: FOG (Freezing of Gait), Postural Stability

Recognized factor mentions (source text → factor):

- supporting: “freezing of gait” → FOG (Freezing of Gait)
- supporting: “FOG” → FOG (Freezing of Gait)
- supporting: “postural stability” → Postural Stability
- supporting: “UPDRS-IV Total” → UPDRS-IV Total
- supporting: “MoCA” → MoCA
- supporting: “Age” → Age
- opposing: “GDS-15” → GDS-15
- opposing: “dopaminergic therapy” → Dopaminergic Therapy
- opposing: “UPDRS-III Total” → UPDRS-III Total

New findings: []

Human confirmation: pending.

## Patient 101566

Original flags:

- `missing_singleton=Age`
- `missing_singleton=FOG (Freezing of Gait)`
- `missing_singleton=Mobility Summary`
- `missing_singleton=Dopaminergic Therapy`
- `missing_singleton=Daytime Sleepiness`

Rechecked explicit groups:

No explicit multi-factor relation extracted; see the factor mentions below.

Recognized factor mentions (source text → factor):

- supporting: “UPDRS-IV Total” → UPDRS-IV Total
- supporting: “age” → Age
- supporting: “BMI” → BMI
- supporting: “UPDRS-III Total” → UPDRS-III Total
- supporting: “freezing of gait” → FOG (Freezing of Gait)
- supporting: “FOG” → FOG (Freezing of Gait)
- opposing: “mobility summary” → Mobility Summary
- opposing: “dopaminergic therapy” → Dopaminergic Therapy
- opposing: “daytime sleepiness” → Daytime Sleepiness

New findings: []

Human confirmation: pending.

## Patient 102479

Original flags:

- `missing_singleton=Age`
- `missing_singleton=Dopaminergic Therapy`
- `missing_singleton=Rigidity at Dx`

Rechecked explicit groups:

- supporting / gait_mobility: FOG (Freezing of Gait), Mobility Summary

Recognized factor mentions (source text → factor):

- supporting: “mobility summary” → Mobility Summary
- supporting: “freezing of gait” → FOG (Freezing of Gait)
- supporting: “age” → Age
- supporting: “body mass index” → BMI
- supporting: “BMI” → BMI
- supporting: “GDS-15” → GDS-15
- opposing: “rigidity at diagnosis” → Rigidity at Dx
- opposing: “dopaminergic therapy” → Dopaminergic Therapy

New findings: []

Human confirmation: pending.

## Patient 130828

Original flags:

- `missing_singleton=FOG (Freezing of Gait)`
- `missing_singleton=UPDRS-IV Total`
- `missing_singleton=Dopaminergic Therapy`

Rechecked explicit groups:

- supporting / gait_mobility: Mobility Summary, Postural Stability

Recognized factor mentions (source text → factor):

- supporting: “Mobility Summary” → Mobility Summary
- supporting: “postural stability” → Postural Stability
- supporting: “Age” → Age
- supporting: “PD Duration” → PD Duration
- supporting: “H&Y Stage” → H&Y Stage
- opposing: “freezing of gait” → FOG (Freezing of Gait)
- opposing: “MDS-UPDRS Part IV” → UPDRS-IV Total
- opposing: “dopaminergic therapy” → Dopaminergic Therapy

New findings: []

Human confirmation: pending.

## Patient 190603

Original flags:

- `missing_or_inexact_group=supports_prediction:gait_mobility`

Rechecked explicit groups:

- supporting / gait_mobility: FOG (Freezing of Gait), Gait, Mobility Summary

Recognized factor mentions (source text → factor):

- supporting: “Mobility Summary” → Mobility Summary
- supporting: “freezing of gait” → FOG (Freezing of Gait)
- supporting: “gait” → Gait
- supporting: “UPDRS-IV Total” → UPDRS-IV Total
- supporting: “PD Duration” → PD Duration
- opposing: “GDS-15” → GDS-15
- opposing: “Dopaminergic Therapy” → Dopaminergic Therapy
- opposing: “UPDRS-I Total” → UPDRS-I Total

New findings: []

Human confirmation: pending.

## Patient 218338

Original flags:

- `missing_singleton=Dopaminergic Therapy`
- `missing_singleton=Rigidity at Dx`

Rechecked explicit groups:

- supporting / gait_mobility: FOG (Freezing of Gait), Mobility Summary

Recognized factor mentions (source text → factor):

- supporting: “Mobility Summary” → Mobility Summary
- supporting: “freezing of gait” → FOG (Freezing of Gait)
- supporting: “FOG” → FOG (Freezing of Gait)
- supporting: “Age” → Age
- supporting: “GDS-15” → GDS-15
- supporting: “UPDRS-IV Total” → UPDRS-IV Total
- opposing: “dopaminergic therapy” → Dopaminergic Therapy
- opposing: “rigidity present at diagnosis” → Rigidity at Dx

New findings: []

Human confirmation: pending.

## Patient 241069

Original flags:

- `missing_singleton=Dopaminergic Therapy`

Rechecked explicit groups:

- supporting / gait_mobility: FOG (Freezing of Gait), Mobility Summary

Recognized factor mentions (source text → factor):

- supporting: “Mobility Summary” → Mobility Summary
- supporting: “freezing of gait” → FOG (Freezing of Gait)
- supporting: “dopaminergic therapy” → Dopaminergic Therapy
- supporting: “UPDRS-IV Total” → UPDRS-IV Total
- supporting: “PD Duration” → PD Duration
- opposing: “Age” → Age
- opposing: “UPDRS-III Total” → UPDRS-III Total
- opposing: “BMI” → BMI

New findings: []

Human confirmation: pending.

## Patient 292826

Original flags:

- `missing_singleton=Dopaminergic Therapy`
- `missing_singleton=FOG (Freezing of Gait)`
- `missing_singleton=PD Duration`

Rechecked explicit groups:

No explicit multi-factor relation extracted; see the factor mentions below.

Recognized factor mentions (source text → factor):

- supporting: “dopaminergic therapy” → Dopaminergic Therapy
- supporting: “BMI” → BMI
- supporting: “UPDRS-IV Total” → UPDRS-IV Total
- supporting: “freezing of gait” → FOG (Freezing of Gait)
- supporting: “PD duration” → PD Duration
- opposing: “UPDRS-III Total” → UPDRS-III Total

New findings: []

Human confirmation: pending.
