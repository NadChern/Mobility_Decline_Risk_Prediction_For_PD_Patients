# Explanation Pipeline

## Summary

This folder contains the current explanation pipeline for the **Stage 1 fall-risk model** only.

Stage 1 predicts the probability of **any fall**:

- class `0` = no fall
- class `1` = any fall

The current explanation workflow does **not** explain the full two-stage pipeline yet. It explains the Stage 1 model output using saved SHAP values, saved test-set feature rows, saved patient IDs, and a feature map.

The explanation pipeline currently has 4 steps.

## Step 1. Global Feature Importance

Step 1 displays the **Top 10 global feature importance values** in the console.

Global feature importance is calculated from SHAP values by:

- taking the absolute SHAP value for each feature across all test patients
- averaging those absolute values across patients
- sorting features from highest importance to lowest

This gives an overall ranking of which features influence the Stage 1 model most strongly across the test set.

## Step 2. Explanation Formatter

Step 2 builds a patient-specific explanation summary.

For the selected patient, it shows:

- top 3 risk-increasing factors
- top 2 risk-decreasing factors

This gives a total of 5 patient-specific factors.

These factors are chosen from the patient's SHAP values:

- positive SHAP values = factors that increase predicted probability of any fall
- negative SHAP values = factors that decrease predicted probability of any fall

Local contribution is the patient-specific SHAP value for that feature.

Patient value is taken from the saved `X_test.csv` row aligned to that patient.

## Step 3. Prompt Generator

Step 3 creates the prompt that will be sent to the LLM.

It uses:

- the patient-specific explanation data from Step 2
- the feature map feature_map.csv.

The feature map provides:

- short feature descriptions
- scale / value interpretation notes

This gives the LLM more human-readable context for selected features.

## Step 4. LLM Explanation Generation

Step 4 sends the prompt to the LLM and prints the generated explanation to the console.

This uses:

- **LangChain** as the wrapper for the LLM call
- **OpenRouter** as the model provider

The current model configured in code is:

`nvidia/nemotron-3-super-120b-a12b:free`

The LLM output is expected to:

- begin with a title
- summarize the predicted probability and risk level
- mention risk-increasing factors
- mention risk-decreasing factors
- end with a short safety-focused clinical summary

## Folder Structure

```text
explanation/
  explanation_layer.py
  feature_map.csv
  README.md

explanation_artifacts/
  shap_values_stage1.npy
  X_test.csv
  y_pred_proba.npy
  patient_ids_test.csv

explanation_results/   # planned
  <PATNO>.md
```

Meaning:

- `explanation/` contains the explanation code and metadata
- `explanation_artifacts/` contains generated files created by `Model_Development.ipynb`
- `explanation_results/` is a planned output folder for saved explanation results

Planned result format:

- one Markdown file per patient
- file name based on patient ID, for example:
  - `101477.md`
  - `170519.md`

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

This generates the explanation artifacts in:

`explanation_artifacts/`

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

## Notes

- The current pipeline explains **Stage 1 only**.
- It explains the probability of **any fall**, not the full final two-stage severity decision.
- The selected patient ID can be changed near the bottom of `explanation_layer.py`.
