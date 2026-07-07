# Clinical Fall Risk Summary for Patient ID: 101477


## Model Prediction

- Fall Classification: No Fall

**Factors are ordered from most to least influential based on their contribution to the model prediction.**


## Factors Pushing the Prediction Toward No Fall

| # | Factor | Patient Value | Interpretation / Scale |
| --- | --- | --- | --- |
| 1 | GDS-15 | 1 | Geriatric Depression Scale (GDS-15) total depression score. 0-15; higher = more depressive burden |
| 2 | UPDRS-IV Total | 0 | MDS-UPDRS Part IV total motor complications score (dyskinesia and motor fluctuations). 0-24; higher = more motor complications |
| 3 | FOG (Freezing of Gait) | 0 | Self-reported severity of freezing of gait. 0 = None; 1 = Rare freezing (may have start hesitation); 2 = Occasional freezing when walking; 3 = Frequent freezing with occasional falls; 4 = Frequent falls from freezing |
| 4 | UPDRS-I Total | 0 | MDS-UPDRS Part I (rater-completed) total non-motor experiences of daily living score. 0-24; higher = more non-motor burden |
| 5 | Rigidity at Dx | 0 | Rigidity present at diagnosis. 0 = No; 1 = Yes; 2 = Unknown |
| 6 | UPDRS-III Total | 19 | MDS-UPDRS Part III total motor examination score (sum of 33 items). 0-132; higher = more severe motor impairment |
| 7 | Postural Stability | 0 | Clinician-rated postural stability (MDS-UPDRS Part III). 0 = Normal (recovers with 1-2 steps); 1 = Slight (3-5 steps recovers unaided); 2 = Mild (>5 steps recovers unaided); 3 = Moderate (stands safely but falls if not caught by examiner); 4 = Severe (very unstable loses balance spontaneously); 101 = unable to rate |
| 8 | Urinary Problems | 1 | Urinary problems (MDS-UPDRS Part I). 0 = Normal (no urine control problems); 1 = Slight (urinate often/urgently no daily activity impact); 2 = Mild (some difficulties with daily activities no accidents); 3 = Moderate (a lot of difficulties including urine accidents); 4 = Severe (cannot control urine use protective garment or bladder tube) |
| 9 | Gait | 1 | Clinician-rated gait impairment (MDS-UPDRS Part III). 0 = Normal (no problems); 1 = Slight (independent walking with minor impairment); 2 = Mild (independent walking with substantial impairment); 3 = Moderate (requires walking aid but not a person); 4 = Severe (cannot walk without another person's assistance); 101 = unable to rate |

## Factors Pushing the Prediction Toward Fall

| # | Factor | Patient Value | Interpretation / Scale |
| --- | --- | --- | --- |
| 1 | Age | 80 | Patient age. Years |
| 2 | Dopaminergic Therapy | 1 | Dopaminergic therapy started. 0 = No; 1 = Yes |
| 3 | PD Duration | 5 | Years since Parkinson's disease diagnosis. Non-negative integer years |
| 4 | BMI | 20.77 | Body mass index. kg/m^2 |

## Model Interpretation

Within this model, the No Fall classification was primarily associated with the absence of motor complications and freezing of gait, alongside low scores for non-motor burden and motor impairment. These factors, which reflect a relative stability in motor and non-motor function, were the primary drivers of the model's output. While the patient's age and the presence of dopaminergic therapy were associated with the opposite direction, these factors did not outweigh the influence of the clinical indicators supporting the No Fall classification.

## Clinical Note

- Model predictions are intended to support clinical review and should not replace clinical judgment.
- The factors shown above identify the patient characteristics that most influenced the model prediction and provide insight into the model’s reasoning.
- For definitions and interpretation of all variables used by the model, please refer to the Feature Reference Guide (feature_map.xlsx).