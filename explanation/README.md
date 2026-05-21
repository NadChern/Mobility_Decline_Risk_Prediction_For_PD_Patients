# Explanation Pipeline

## Summary

This folder contains the explanation pipeline for the **full two-stage fall prediction model**.

The two-stage pipeline predicts fall risk and severity for Parkinson's disease patients:

- **Stage 1** (Random Forest): Predicts probability of **any fall**
  - class `0` = no fall
  - class `1` = any fall

- **Stage 2** (LinearSVC, calibrated): Predicts **severity** for patients predicted to fall
  - class `1` = mild falls
  - class `2` = moderate falls

Only patients predicted as "any fall" by Stage 1 are routed to Stage 2 for severity assessment.

The explanation pipeline supports both Stage 1-only mode (backward compatible) and full two-stage mode when Stage 2 artifacts are available.

## How SHAP Values Are Computed

### Stage 1 SHAP Computation

- **Model**: Random Forest (`best_stage1`)
- **Explainer**: `shap.Explainer` (automatically uses TreeExplainer internally for tree-based models)
- **Input**: Full test set (`X_test`)
- **Output**: SHAP values for class 1 (any fall)

### Stage 2 SHAP Computation

- **Model**: Calibrated LinearSVC (`calibrated_stage2`)
  - LinearSVC wrapped with `CalibratedClassifierCV` to enable probability output
- **Explainer**: `shap.KernelExplainer` (required for calibrated wrapper since TreeExplainer doesn't support it)
- **Input**: Only routed patients (`X_routed` where `pred_s1 == 1`)
- **Output**: SHAP values for class 1 (moderate falls)

### Artifacts Created

**Stage 1 artifacts** (required for explanation pipeline):

| File                     | Description                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------- |
| `shap_values_stage1.npy` | SHAP values explaining Stage 1 predictions (one value per feature per patient)      |
| `X_test.csv`             | Feature values for all test patients (used to show "Patient value" in explanations) |
| `y_pred_proba.npy`       | Predicted probability of any fall for each patient                                  |
| `patient_ids_test.csv`   | Patient identifiers to look up specific patients                                    |
| `routing_mask.npy`       | Stage 1 binary prediction (True = any fall predicted, routes to Stage 2)            |

> **Note on naming:** Stage 1 does not have a separate `y_pred_s1.npy` file because its binary prediction serves as the `routing_mask` for Stage 2. The routing mask IS Stage 1's binary prediction (threshold applied to `y_pred_proba.npy`). Stage 2 has both `y_pred_proba_s2.npy` (probability) and `y_pred_s2.npy` (binary) because it doesn't route to another stage.

**Stage 2 artifacts** (optional - enables full two-stage explanations):

| File                     | Description                                                                     |
| ------------------------ | ------------------------------------------------------------------------------- |
| `shap_values_stage2.npy` | SHAP values explaining Stage 2 severity predictions                             |
| `y_pred_proba_s2.npy`    | Predicted probability of moderate falls for routed patients                     |
| `y_pred_s2.npy`          | Stage 2 binary prediction (0=mild, 1=moderate)                                  |
| `y_pred_final.npy`       | Final 3-class predictions combining both stages (0=no fall, 1=mild, 2=moderate) |

## Pipeline Steps

### Step 1. Global Feature Importance

Displays the **Top 10 global feature importance values** for both stages.

For Stage 1 (any fall):

- Computed from Stage 1 SHAP values
- Shows which features most influence fall risk prediction

For Stage 2 (severity):

- Computed from Stage 2 SHAP values
- Shows which features most influence severity prediction
- Only available when Stage 2 artifacts exist

Global feature importance is calculated by:

- Taking the absolute SHAP value for each feature across all patients
- Averaging those absolute values across patients
- Sorting features from highest importance to lowest

### Step 2. Explanation Formatter

Builds a patient-specific explanation summary covering both stages.

For the selected patient, it shows:

**Stage 1 (Fall Risk):**

- Predicted probability of any fall
- Risk level (low/moderate/high)
- Top 3 risk-increasing factors
- Top 2 risk-decreasing factors

**Stage 2 (Severity) - if patient was routed:**

- Predicted probability of moderate falls
- Severity level (likely mild/uncertain/likely moderate)
- Top 3 severity-increasing factors
- Top 2 severity-decreasing factors

**Final Prediction:**

- 3-class prediction (no falls / mild falls / moderate falls)

### Step 3. Prompt Generator

Creates the prompt for the LLM using:

- Full two-stage patient-specific explanation data
- The feature map (`feature_map.csv`) for human-readable context

The prompt instructs the LLM to explain:

1. The final 3-class prediction
2. Stage 1 factors (why fall risk is high/low)
3. Stage 2 factors (why severity is mild/moderate) - if applicable
4. Clinical recommendation

### Step 4. LLM Explanation Generation

Sends the prompt to the LLM and returns the generated explanation.

Uses:

- **LangChain** as the wrapper for the LLM call
- **OpenRouter** as the model provider

The current default model is:

`nvidia/nemotron-3-super-120b-a12b:free`

## Folder Structure

```text
explanation/
  explanation_layer.py
  feature_map.csv
  README.md

explanation_artifacts/
  # Stage 1 artifacts (required)
  shap_values_stage1.npy      # (N_test, N_features) SHAP values for Stage 1
  X_test.csv                  # (N_test, N_features) patient feature values
  y_pred_proba.npy            # (N_test,) P(any fall)
  patient_ids_test.csv        # (N_test,) stable patient identifiers

  # Stage 2 artifacts (optional - enables full pipeline)
  routing_mask.npy            # (N_test,) bool - which patients went to Stage 2
  shap_values_stage2.npy      # (N_routed, N_features) SHAP values for Stage 2
  y_pred_proba_s2.npy         # (N_routed,) P(moderate falls)
  y_pred_s2.npy               # (N_routed,) Stage 2 predictions (0=mild, 1=moderate)
  y_pred_final.npy            # (N_test,) Final 3-class predictions

explanation_results/          # planned output folder
  <PATNO>.md
```

### Artifact Descriptions

| File                     | Shape          | Description                                      |
| ------------------------ | -------------- | ------------------------------------------------ |
| `shap_values_stage1.npy` | (351, 21)      | SHAP values for Stage 1 (any fall) predictions   |
| `X_test.csv`             | (351, 21)      | Patient feature values for the test set          |
| `y_pred_proba.npy`       | (351,)         | Stage 1 predicted probabilities of any fall      |
| `patient_ids_test.csv`   | (351,)         | Patient IDs (PATNO) aligned with other artifacts |
| `routing_mask.npy`       | (351,)         | Boolean mask - True if routed to Stage 2         |
| `shap_values_stage2.npy` | (N_routed, 21) | SHAP values for Stage 2 (severity) predictions   |
| `y_pred_proba_s2.npy`    | (N_routed,)    | Stage 2 P(moderate falls) for routed patients    |
| `y_pred_s2.npy`          | (N_routed,)    | Stage 2 predictions for routed patients          |
| `y_pred_final.npy`       | (351,)         | Final 3-class predictions (0/1/2)                |

## How to Run

### 1. Activate the environment

From the project root:

```bash
source .venv/bin/activate
```

### 2. Run the model notebook to generate artifacts

From the project root:

```bash
jupyter nbconvert --to notebook --execute Model_Development.ipynb --output Model_Development.rerun.ipynb
```

This generates both Stage 1 and Stage 2 explanation artifacts in:

`explanation_artifacts/`

**Note:** Stage 2 artifact generation requires the KernelExplainer which can take several minutes to compute SHAP values.

### 3. Create a `.env` file in the project root

Add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Make sure `.env` is included in `.gitignore`.

### 4. Install explanation-layer dependencies

If needed, install:

```bash
uv pip install --python .venv/bin/python python-dotenv langchain langchain-openrouter
```

### 5. Run the explanation pipeline

From the project root:

```bash
python explanation/explanation_layer.py
```

or:

```bash
.venv/bin/python explanation/explanation_layer.py
```

## Backward Compatibility

The explanation layer automatically detects whether Stage 2 artifacts are available:

- **Stage 2 available**: Full two-stage explanations with severity assessment
- **Stage 2 missing**: Falls back to Stage 1-only explanations

The original Stage 1-only functions remain available:

- `build_patient_explanation_data()` - Stage 1 only
- `format_explanation()` - Stage 1 only
- `generate_llm_prompt()` - Stage 1 only
- `generate_llm_explanation()` - Stage 1 only

New full pipeline functions:

- `build_patient_explanation_data_full()` - Both stages
- `format_explanation_full()` - Both stages
- `generate_llm_prompt_full()` - Both stages
- `generate_llm_explanation_full()` - Both stages

## Edge Cases

| Scenario                               | Behavior                                 |
| -------------------------------------- | ---------------------------------------- |
| Stage 2 artifacts missing              | Falls back to Stage 1-only explanation   |
| Patient not routed (predicted no fall) | Returns Stage 1 only, stage2=None        |
| Patient not in test set                | Raises KeyError with descriptive message |
| Artifact row count mismatch            | Raises ValueError at module load         |

## Example Output

```
======================================================================
FULL TWO-STAGE FALL PREDICTION EXPLANATION
Patient ID: 101477
======================================================================

FINAL PREDICTION: Moderate falls predicted
Routed to Stage 2: Yes

----------------------------------------------------------------------
STAGE 1: Fall Risk Assessment (No Fall vs Any Fall)
----------------------------------------------------------------------
Predicted probability of any fall: 0.73
Risk level: high

Risk-increasing factors:
  1. Neuro_QoL_LE
     Patient value: 35.00
     Contribution: +0.142 (global importance: 0.089)
...

----------------------------------------------------------------------
STAGE 2: Severity Assessment (Mild vs Moderate Falls)
----------------------------------------------------------------------
Predicted probability of moderate falls: 0.62
Severity level: uncertain (borderline)

Severity-increasing factors (toward moderate):
  1. Age
     Patient value: 72.00
     Contribution: +0.085 (global importance: 0.072)
...
======================================================================
```
