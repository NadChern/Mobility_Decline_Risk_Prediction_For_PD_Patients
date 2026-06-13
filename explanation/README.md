# Explanation Pipeline

## Summary

This package generates patient-specific explanations for the current two-stage fall prediction workflow used in this repository.

- **Stage 1** explains the probability of **any fall** using TreeSHAP (exact, deterministic).
- **Stage 2** explains **mild vs moderate fall severity** for patients routed forward by Stage 1 using LinearSHAP (exact, deterministic).
- Feature selection for explanations uses the SHAP coverage threshold defined in [`config.yaml`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/config.yaml).
- The main runtime output is a console clinical summary. When `debug: true`, the full LLM prompt is also printed.

## What The Package Does Now

The explanation layer loads precomputed model artifacts from `explanation_artifacts/`, builds structured patient explanation data from SHAP values, and sends that data to an LLM for a natural-language clinical summary.

Current behavior:

- Stage 1 explains why the model predicted **no falls** or **any falls**.
- Stage 2 explains why routed patients were classified as **mild falls** or **moderate falls**.
- The package expects the **full two-stage artifact set** to be present.
- The default CLI flow uses a sample patient ID hardcoded in [`__main__.py`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/__main__.py).

## Package Structure

The files most relevant to users are:

```text
explanation/
  __init__.py
  __main__.py
  config.yaml
  data_loader.py
  explanation_builder.py
  feature_map.csv
  llm.py
  README.md
  shap_utils.py
  validate.py
```

What they are used for:

- `data_loader.py`: loads config, environment variables, feature metadata, and explanation artifacts.
- `explanation_builder.py`: builds per-patient explanation payloads from SHAP values and predictions.
- `llm.py`: constructs the prompt and calls the configured LLM provider.
- `shap_utils.py`: selects top contributing features based on SHAP coverage.
- `validate.py`: generates a validation spreadsheet with sampled patients and LLM explanations.
- `config.yaml`: controls debug mode, SHAP coverage threshold, provider, model, and temperature.
- `feature_map.csv`: maps model feature names to human-readable clinical descriptions.

## Required Runtime Artifacts

The package currently expects the full two-stage artifact set in `explanation_artifacts/`.

Required files:

- `shap_values_stage1.npy`
- `X_test.csv`
- `y_pred_proba.npy`
- `patient_ids_test.csv`
- `routing_mask.npy`
- `shap_values_stage2.npy`
- `y_pred_s2.npy`
- `y_pred_final.npy`

Important note:

- The current code does **not** fall back to a Stage 1-only mode.
- [`data_loader.py`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/data_loader.py) loads Stage 2 artifacts at import time and raises an error if they are missing or misaligned.

## Configuration

Configuration lives in [`config.yaml`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/config.yaml).

Current top-level settings:

- `debug`: when `true`, prints the generated prompt before the clinical summary.
- `shap.coverage_threshold`: cumulative absolute SHAP coverage target used for feature selection.
- `llm.provider`: selects `"google"` or `"openrouter"`.
- `llm.temperature`: generation temperature passed to the LLM client.
- `llm.openrouter.model`: model name used when the provider is OpenRouter.
- `llm.google.model`: model name used when the provider is Google.

### API Keys

Environment variables are loaded from the project root `.env` file.

- `GOOGLE_API_KEY`: required when `llm.provider: "google"`
- `OPENROUTER_API_KEY`: required when `llm.provider: "openrouter"`

At the time of writing, the default configuration is:

- provider: `google`
- model: `gemini-3.1-flash-lite`

OpenRouter remains supported through the same interface in `llm.py`.

## How To Run

### 1. Activate the local environment

From the project root:

```bash
source .venv/bin/activate
```

### 2. Generate explanation artifacts

Run the model notebook so `explanation_artifacts/` contains the full two-stage outputs:

```bash
jupyter nbconvert --to notebook --execute Model_Development.ipynb --output Model_Development.rerun.ipynb
```

This step is expected to produce the `.npy` and `.csv` files loaded by `data_loader.py`.

### 3. Add the required API key to `.env`

For the current default provider:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

If you switch `llm.provider` to OpenRouter in `config.yaml`, use:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 4. Install missing dependencies if needed

If the explanation dependencies are not already available in `.venv`, install the packages used by the current provider:

```bash
uv pip install --python .venv/bin/python python-dotenv pyyaml langchain langchain-google-genai langchain-openrouter
```

### 5. Run the explanation CLI

From the project root:

```bash
python -m explanation
```

This entrypoint runs [`__main__.py`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/__main__.py), which:

- reads the configured provider and model
- loads the saved explanation artifacts
- generates a clinical summary for one sample patient

To inspect a different patient, edit the `patient_id` value in `__main__.py`.

## Validation Utility

[`validate.py`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/validate.py) creates a spreadsheet for manual review of LLM explanations across sampled patients from each prediction class.

Run it from the project root with:

```bash
python -m explanation.validate
```

What it does:

- samples patients from the final prediction categories
- calls the explanation pipeline for each selected patient
- writes `validation_results.xlsx` in the project root
- writes a resumable partial file during execution if needed

Output location:

- validation spreadsheet: `validation_results.xlsx` in the project root

## SHAP Determinism

Both SHAP methods used in this pipeline are exact and produce bit-for-bit identical results on every run:

- **Stage 1 (TreeSHAP)**: exact computation over the RandomForest tree structure — no sampling involved.
- **Stage 2 (LinearSHAP)**: closed-form `SHAP_i = coef_i × (x_i − E[x_i])` using the full training set mean as background — no randomness.

`Model_Development.ipynb` includes inline determinism checks after each SHAP computation block that assert `np.allclose()` and print max/mean absolute differences (expected: `0.000000000000` for both).

## Output Locations

To avoid confusion, the main output locations are:

- `explanation_artifacts/`: runtime model artifacts consumed by the explanation package
- project root `validation_results.xlsx`: spreadsheet generated by `python -m explanation.validate`

The current package does **not** implement a standard per-patient markdown export such as `explanation_results/<PATNO>.md`.

## Common Failure Modes

- Missing API key for the selected provider in `.env`
- Missing Stage 2 artifacts in `explanation_artifacts/`
- Row-count mismatch across saved artifact files
- Missing provider-specific dependency such as `langchain-google-genai` or `langchain-openrouter`

If you hit one of these errors, first verify `config.yaml`, `.env`, and the contents of `explanation_artifacts/`.
