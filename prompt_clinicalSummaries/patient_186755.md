# Clinical Fall Risk Summary for Patient ID: 186755

## Model Prediction

Fall Classification: Fall

Fall Severity Classification: Rare Fall

**Factors are ordered from most to least influential based on their contribution to the model prediction.**

## Factors Pushing the Prediction Toward Lower Severity (Rare Fall)

| #   | Factor                     | Patient Value | Interpretation / Scale                                                                                                                                                                                                                                                                                                                               |
| --- | -------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | FOG (Freezing of Gait)     | 0             | Self-reported severity of freezing of gait. 0 = None; 1 = Rare freezing (may have start hesitation); 2 = Occasional freezing when walking; 3 = Frequent freezing with occasional falls; 4 = Frequent falls from freezing                                                                                                                             |
| 2   | Gait                       | 1             | Clinician-rated gait impairment (MDS-UPDRS Part III). 0 = Normal (no problems); 1 = Slight (independent walking with minor impairment); 2 = Mild (independent walking with substantial impairment); 3 = Moderate (requires walking aid but not a person); 4 = Severe (cannot walk without another person's assistance); 101 = unable to rate         |
| 3   | UPDRS-III Total            | 22            | MDS-UPDRS Part III total motor examination score (sum of 33 items). 0-132; higher = more severe motor impairment                                                                                                                                                                                                                                     |
| 4   | PD Duration                | 4             | Years since Parkinson's disease diagnosis. Non-negative integer years                                                                                                                                                                                                                                                                                |
| 5   | Age                        | 69            | Patient age. Years                                                                                                                                                                                                                                                                                                                                   |
| 6   | Postural Instability at Dx | 0             | Postural instability present at diagnosis. 0 = No; 1 = Yes; 2 = Unknown                                                                                                                                                                                                                                                                              |
| 7   | UPDRS-IV Total             | 7             | MDS-UPDRS Part IV total motor complications score (dyskinesia and motor fluctuations). 0-24; higher = more motor complications                                                                                                                                                                                                                       |
| 8   | Urinary Problems           | 1             | Urinary problems (MDS-UPDRS Part I). 0 = Normal (no urine control problems); 1 = Slight (urinate often/urgently no daily activity impact); 2 = Mild (some difficulties with daily activities no accidents); 3 = Moderate (a lot of difficulties including urine accidents); 4 = Severe (cannot control urine use protective garment or bladder tube) |
| 9   | Fainting                   | 0             | Fainting frequency (SCOPA-AUT). 0 = never; 1 = sometimes; 2 = regularly; 3 = often                                                                                                                                                                                                                                                                   |

## Factors Pushing the Prediction Toward Higher Severity (Recurrent Fall)

| #   | Factor           | Patient Value | Interpretation / Scale                                                                                                                                                                                                                                                                                                                                  |
| --- | ---------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | BMI              | 24.39         | Body mass index. kg/m^2                                                                                                                                                                                                                                                                                                                                 |
| 2   | Mobility Summary | 4.35          | Gaussian-weighted mobility summary across seven daily activities (toilet transfers, curbs, car transfers, bed-to-chair, errands/shopping, getting off the floor, walking 15+ minutes). Higher values indicate more responses in the moderate-difficulty range; this score is non-monotonic and does not simply mean greater overall mobility difficulty |
| 3   | MoCA             | 21            | Montreal Cognitive Assessment (MoCA) total cognition score. 0-30; higher = better cognition                                                                                                                                                                                                                                                             |

## Model Interpretation

Within this model, the Rare Fall classification was primarily associated with the absence of freezing of gait, a low clinician-rated gait impairment score, and a lower total motor examination score. These factors, alongside the patient's disease duration and age, collectively supported the classification of lower severity. While the MoCA score and the mobility summary score pushed the model toward a higher severity classification, these factors did not outweigh the influence of the motor and gait-related variables.

## Clinical Note

- Model predictions are intended to support clinical review and should not replace clinical judgment.
- The factors shown above identify the patient characteristics that most influenced the model prediction and provide insight into the model's reasoning.
- For definitions and interpretation of all variables used by the model, please refer to the Feature Reference Guide (feature_map.xlsx).
