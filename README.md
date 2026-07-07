# Explanation Pipeline

> This README documents only the `explanation/` package — explanation generation and its
> evaluation pipeline. It does not cover the modeling notebooks in `Revised_Modelling/` or the
> rest of the repository.

## Summary

This package generates patient-specific explanations for the current two-stage fall prediction workflow used in this repository, combining SHAP feature attribution with LLM-based natural-language generation. Rather than reporting only the predicted fall category, it identifies the patient characteristics that most influenced the model's decision and presents them in a structured, clinically interpretable format. The mechanics (attribution method, coverage-based feature selection, determinism/additivity checks) are documented in detail in [SHAP Calculation, Feature Selection & Sanity Checks](#shap-calculation-feature-selection--sanity-checks) below.

- **Stage 1** explains the probability of **any fall** (RandomForest, TreeSHAP).
- **Stage 2** explains **Rare Fall vs Recurrent Fall severity** for patients routed forward by Stage 1 (XGBoost, TreeSHAP).
- Feature selection for explanations uses the SHAP coverage threshold defined in [`config.yaml`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/config.yaml).
- The main runtime output is a console clinical summary. When `debug: true`, the full LLM prompt is also printed.

## Design Approach & Why

The package is split into a **producer** ([`export.py`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/export.py), run from the modelling notebook) and a **consumer** (everything that reads `explanation_artifacts/` at runtime), with a small shared **contract** ([`contract.py`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/contract.py)) defining artifact filenames and the `PATNO` key.

This producer-consumer split gives four concrete benefits: it separates responsibilities cleanly, improves maintainability as the underlying model evolves, keeps explanations reproducible from a fixed set of artifacts, and supports robust validation before anything reaches the LLM step:

- **Model-agnostic SHAP** — `compute_shap()` leans on `shap.Explainer`'s dispatcher and normalises any tree/linear output to a 2-D positive-class array. Swapping the underlying model needs **no explanation-code edits**.
- **Consumer decoupled from the model** — the runtime only depends on the 8-file artifact contract, not on which algorithm produced it. So a new model is just a re-export.
- **Single source of truth** — filenames and the ID column live in `contract.py`; the export is one reusable function instead of notebook copy-paste that drifts.
- **Fail-loud, not silent** — `export.py` validates feature alignment, stage-2 consistency, shapes/finiteness, and SHAP determinism **before writing**, and `compute_shap` rejects mismatched column order. Bad artifacts never reach the LLM step.

## What The Package Does Now

The explanation layer loads precomputed model artifacts from `explanation_artifacts/`, builds structured patient explanation data from SHAP values, and sends that data to an LLM for a natural-language clinical summary.

Current behavior:

- Stage 1 explains why the model predicted **no falls** or **any falls**.
- Stage 2 explains why routed patients were classified as **Rare Fall** or **Recurrent Fall**.
- The package expects the **full two-stage artifact set** to be present.
- The default CLI flow auto-selects the first patient in the saved test set (override in [`__main__.py`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/__main__.py)).

### Output document

The LLM renders one clinician document from the deterministic SHAP packet (it may **not**
alter the facts):

- a **Model Prediction** header (Fall Classification, plus Severity if routed);
- **two factor tables** — _Factors Pushing the Prediction Toward …_ the predicted outcome
  (shown first, the drivers) and the opposite direction — each `# | Factor | Patient Value |
Interpretation / Scale`, ordered by `|SHAP|`, with missing / "unable to rate" factors omitted;
- a short **Model Interpretation** (names the main drivers and the opposing factors that did
  not outweigh them); and
- a fixed **Clinical Note**.

Export to Markdown / HTML / PDF via [`render.py`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/render.py) (see _Export the output_ below).

## Package Structure

The files most relevant to users are:

```text
README.md              # this file (project root)
explanation/
  __init__.py
  __main__.py
  config.yaml
  contract.py
  data_loader.py
  explanation_builder.py
  export.py
  feature_map.csv
  llm.py
  render.py
  shap_utils.py
  validate.py
  evaluation/
    llm_eval_runner.py
    llm_stability.py
    llm_structure.py
    llm_faithfulness.py
    additivity_check.py
```

What they are used for:

- `contract.py`: shared artifact contract — filenames and the `PATNO` column, used by both producer and consumer.
- `export.py`: **producer** — `compute_shap()` and `export_explanation_artifacts()` regenerate, validate, and write all artifacts (called from the modelling notebook).
- `data_loader.py`: loads config, environment variables, feature metadata, and explanation artifacts.
- `explanation_builder.py`: builds per-patient explanation payloads from SHAP values and predictions, and attaches display fields (`enrich_factor`: `short_name`, formatted value, interpretation, `displayable`) from `feature_map.csv`.
- `llm.py`: builds the full-document prompt (two direction tables + Model Interpretation) from the enriched packet and calls the configured LLM provider.
- `render.py`: exports a generated explanation document to Markdown, styled HTML, or PDF (presentation only — does not change content).
- `shap_utils.py`: selects top contributing features based on SHAP coverage.
- `validate.py`: generates a validation spreadsheet with sampled patients and LLM explanations.
- `config.yaml`: controls debug mode, SHAP coverage threshold, provider, model, and temperature.
- `feature_map.csv`: maps each model feature to a clinician-facing `short_name`, an instrument-named `description`, and a standardised value scale (curated, version-controlled).
- `evaluation/`: the explanation-quality evaluation pipeline (stability, structure, faithfulness, additivity) — see [Evaluation](#evaluation) below.

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

Optional (provenance, **not** loaded by the consumer):

- `manifest.json` — records which dataset, source notebook, stage models, SHAP explainers, feature order, `n_test`, `n_routed`, final-prediction counts, and timestamp produced the current artifacts. Written by `export.py`; safe to delete, and older artifact sets without it still load.

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

- provider: `openrouter`
- model: `deepseek/deepseek-v4-flash::alibaba`

Google (`gemini-3.1-flash-lite`) remains supported through the same interface in `llm.py`; switch
`llm.provider` in `config.yaml` to use it.

## How To Run

### 1. Activate the local environment

From the project root:

```bash
source .venv/bin/activate
```

### 2. Generate explanation artifacts (run the modelling notebook)

The artifacts are produced by running `Revised_Modelling/Revised_Model_Results.ipynb`
top-to-bottom. Its final cell calls `export_explanation_artifacts(...)` from `export.py`, which
computes SHAP, validates the contract, runs the determinism check, and writes the `.npy`/`.csv`
files (and `manifest.json`) into `explanation_artifacts/`.

**Option A — headless (terminal):**

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=3600 \
  Revised_Modelling/Revised_Model_Results.ipynb
```

This re-runs every cell (including the model GridSearch — expect a few minutes) and saves the
refreshed outputs back into the notebook. The export cell's summary (shapes, routed count,
determinism diagnostics) is stored in that cell's output; to print it back to the terminal:

```bash
python -c "import json; nb=json.load(open('Revised_Modelling/Revised_Model_Results.ipynb')); [print(''.join(o.get('text',[]))) for c in nb['cells'] for o in c.get('outputs',[]) if 'SHAP determinism' in ''.join(o.get('text',[]))]"
```

**Option B — interactive (Jupyter / VS Code):** open the notebook and choose
**Kernel → Restart & Run All**; the summary prints live in the last cell.

To verify the artifacts were written:

```bash
ls explanation_artifacts          # 8 artifacts + manifest.json
```

> Notes:
>
> - `explanation_artifacts/` is git-ignored, so a fresh clone must run this step to recreate the artifacts.
> - XGBoost on macOS needs the OpenMP runtime: `brew install libomp` if `import xgboost` fails.

### 3. Add the required API key to `.env`

For the current default provider (OpenRouter):

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

If you switch `llm.provider` to Google in `config.yaml`, use:

```env
GOOGLE_API_KEY=your_google_api_key_here
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
- generates a clinical summary for the first patient in the saved test set

To inspect a different patient, set the `patient_id` value in `__main__.py` to any `PATNO` present in `patient_ids_test.csv`.

### 6. Export the output to Markdown / HTML / PDF

The console summary is markdown. To save it as a clean, formatted document — the wide
Scale / Interpretation column wraps inside the table cell, so it renders as a proper bordered
table — use the helpers in [`render.py`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/render.py). The format is inferred from the file extension (`.md` / `.html` / `.pdf`), or pass `fmt=`.

```python
from explanation import export_explanation, save_document

# Generate for a patient and export (one LLM call):
export_explanation(3207, "patient_3207.md")     # raw markdown
export_explanation(3207, "patient_3207.html")   # styled, standalone HTML
export_explanation(3207, "patient_3207.pdf")    # PDF

# Or render an already-generated document string (no LLM call):
from explanation.llm import generate_explanation
doc = generate_explanation(3207, debug=False, temperature=0)["explanation"]
save_document(doc, "patient_3207.pdf")
```

One-liner from the shell (project root):

```bash
./.venv/bin/python -c "from explanation import export_explanation; export_explanation(3207, 'patient_3207.pdf')"
```

Notes:

- **Markdown / HTML** need only the `markdown` package (already installed).
- **PDF** requires `weasyprint` so the PDF preserves the HTML styling, headings, and tables:
  `uv pip install --python .venv/bin/python weasyprint`
- If `weasyprint` is unavailable, export to `.html` and print from a browser rather than
  falling back to a lower-fidelity PDF renderer.
- The `.md` / `.html` files render as a formatted table in any markdown viewer or browser
  (in VS Code, open a `.md` and press `Cmd+Shift+V` for the preview).

## SHAP Calculation, Feature Selection & Sanity Checks

This section documents how the per-patient feature attributions are computed, how the
explanation selects which features to report, and how the artifacts are validated before
they are written. It is intended as a technical reference for the methodology.

### 1. SHAP method

Both stages use **TreeSHAP** via `shap.Explainer(model, background)`
([`compute_shap()` in `export.py`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/export.py)).
`shap.Explainer` is a dispatcher: for the tree ensembles used here it selects the
`TreeExplainer`, confirmed in `manifest.json` (`stage1_shap_explainer` /
`stage2_shap_explainer` = `TreeExplainer`).

|             | Model        | Explainer | SHAP output                             | Reported space                          |
| ----------- | ------------ | --------- | --------------------------------------- | --------------------------------------- |
| **Stage 1** | RandomForest | TreeSHAP  | `(N, F, 2)` → class-1 slice `[:, :, 1]` | probability of _any fall_               |
| **Stage 2** | XGBoost      | TreeSHAP  | `(M, F)` for the positive class         | log-odds margin toward _Recurrent Fall_ |

Key properties:

- **Exact, not sampled.** TreeSHAP computes exact Shapley values from the tree structure
  (no Monte-Carlo sampling), using a **fixed training-set background** (`X_train` for Stage 1,
  the true-faller training rows `X_train_12` for Stage 2). With fixed seeds across the split
  and both models, the attributions are reproducible run-to-run — see the determinism check below.
- **Sign convention.** Positive SHAP pushes the prediction _toward the positive class_
  (any-fall for Stage 1, Recurrent Fall for Stage 2); negative SHAP pushes away. The explanation
  presents these as two tables — _Factors Pushing the Prediction Toward …_ the predicted
  outcome (drivers) vs. the opposite direction — rather than as clinical risk/protective labels.
- **Model-agnostic normalisation.** `compute_shap()` collapses whichever shape the explainer
  returns (`(N,F,2)`, `(M,F)`, or a legacy `list`) into a 2-D positive-class array, and rejects
  a background whose column order differs from the explained matrix — so a future model swap
  needs no changes here.

### 2. Why a coverage threshold (and why not "top-5")

For each patient the explanation does **not** report a fixed number of features. Instead
([`select_features_by_coverage()` in `shap_utils.py`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/shap_utils.py)) it:

1. splits the features into positive (contributing) and negative (mitigating) groups,
2. sorts each group by `|SHAP|` descending, and
3. keeps adding features until the **cumulative `|SHAP|` reaches the coverage threshold** of that
   group's total — applied **separately** to the positive and negative sides.

**Current threshold: `0.90`** (`shap.coverage_threshold` in
[`config.yaml`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/config.yaml)),
i.e. the reported factors account for ≥ 90% of the total attribution magnitude on each side.

Why this is preferable to a fixed "top-5":

- **It is adaptive to the actual decision.** When a prediction is driven by 2–3 dominant
  features, coverage reports just those; when the contribution is diffuse across many small
  features, it reports more. A fixed top-5 mis-sizes both cases: it _pads_ the concentrated
  case with near-zero features, and it _under-explains_ the diffuse case (a top-5 might cover
  only ~40% of the signal, silently omitting most of the reasoning).
- **It carries a guarantee.** "These factors explain ≥ 90% of the model's push in each
  direction" is a defensible statement; "top 5" carries no guarantee about how much of the
  decision is actually captured.
- **It is faithful to magnitude, not count.** Ranking and the cut-off use `|SHAP|`, so the
  threshold is comparable across patients and across model versions (even though raw SHAP
  _units_ differ between a linear and a tree model).

The threshold is a single config value, so it can be tightened (e.g. `0.95`) or loosened
without code changes.

### 3. Write-time validation (before any artifact is written)

`export_explanation_artifacts()` validates the full artifact contract via `_validate()`
**before** saving anything, so malformed artifacts never reach disk or the LLM step:

- **Feature alignment** — `set(X_test.columns)` must equal the feature set in `feature_map.csv`
  (no missing/extra features), so every reported feature has clinical metadata.
- **Row-count alignment** — `shap_values_stage1`, `y_pred_proba`, `routing_mask`, and
  `y_pred_final` all have one row per test patient (`N`).
- **Stage-2 consistency** — `shap_values_stage2` and `y_pred_s2` have exactly `routing_mask.sum()`
  rows; `routing_mask` is boolean; routed patients map `y_pred_s2` (0/1) → `y_pred_final` (1/2);
  non-routed patients are forced to class 0; `y_pred_s2 ⊆ {0,1}`; `y_pred_final ⊆ {0,1,2}`.
- **Numerical sanity** — no `NaN`/`Inf` in either SHAP array, and neither array is all-zero.

If any check fails, the function raises and writes nothing.

### 4. Determinism check

Immediately after computing SHAP, `export.py` **recomputes** each stage's SHAP from the same
fitted model and background and compares it to the values it is about to save:

- the gate is `np.allclose(saved, recomputed)` — if it fails, it raises
  `"Stage N SHAP values are not deterministic!"` and writes nothing;
- it also reports the diagnostics `max|diff|` and `mean|diff|` per stage in the console summary.

**Result (current model):**

```
SHAP determinism (recompute vs saved, expected 0):
  Stage 1: max|diff|=0.00e+00  mean|diff|=0.00e+00
  Stage 2: max|diff|=0.00e+00  mean|diff|=0.00e+00
All write-time validation and determinism checks passed.
```

i.e. recomputation reproduces the saved values **bit-for-bit**, confirming the SHAP step is
deterministic and that what is on disk is exactly what the explainer produced.

### 5. Additivity sanity check (correctness of the values)

Determinism proves _stability_; additivity proves the values are _correct_ Shapley
attributions. TreeSHAP satisfies the **local-accuracy / additivity** property:
`base_value + Σ SHAP = model output` for each patient. We verified this independently against
the saved artifacts:

| Stage                  | Reconstruction tested                                  | Max error   |
| ---------------------- | ------------------------------------------------------ | ----------- |
| Stage 1 (RandomForest) | `base + Σ SHAP` vs `predict_proba` (probability space) | ≈ `1.0e-02` |
| Stage 2 (XGBoost)      | `base + Σ SHAP` vs raw margin (log-odds space)         | ≈ `1.1e-06` |

Stage 2 reconciles to floating-point precision. The small Stage 1 residual comes from SHAP's
`Independent` background masker sub-sampling the background (default 100 rows) when estimating
the expected value — it is a property of the SHAP library's default, **not** of this pipeline
(the old LinearSVC pipeline used the identical `shap.Explainer(model, X_train)` call), and it
does not affect feature ranking or selection, which depend only on `|SHAP|`.

## Output Locations

To avoid confusion, the main output locations are:

- `explanation_artifacts/`: runtime model artifacts consumed by the explanation package
- `evaluation_results/`: per-patient and per-category CSVs written by the evaluation pipeline (git-ignored) — see [Evaluation](#evaluation) below

Per-patient documents can be exported to Markdown / HTML / PDF with `render.py` (see [Export the output to Markdown / HTML / PDF](#6-export-the-output-to-markdown--html--pdf) above). The package does not auto-write them to a fixed location — you choose the output path.

## Evaluation

`explanation/evaluation/` assesses generated explanations along three dimensions, plus an
independent SHAP sanity check. Full methodology (sampling design, metric definitions) is in
`llm_eval_plan.md`; results already produced live under `evaluation_results/` (git-ignored).

- **Stability** ([`llm_stability.py`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/evaluation/llm_stability.py)) — do repeated generations for the same patient agree? Exact-match at temperature 0; semantic similarity + factor-set (Jaccard) consistency at temperature 0.3.
- **Structure** ([`llm_structure.py`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/evaluation/llm_structure.py)) — does each report follow the required template: sections present, correct stage/routing layout, well-formed tables, drivers-table-first ordering, no forbidden probability/percentage language?
- **Faithfulness** ([`llm_faithfulness.py`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/evaluation/llm_faithfulness.py)) — does the report match the deterministic SHAP evidence it was given: same factors, correct direction, correct values/scale text, correct predicted outcome; does the free-text interpretation avoid naming factors it wasn't given?
- **Additivity** ([`additivity_check.py`](/Users/nadin/Documents/Mobility_Decline_Risk_Prediction_For_PD_Patients/explanation/evaluation/additivity_check.py)) — independently re-verifies `base value + Σ SHAP = model output` against the exported artifacts; a numerical check on the attributions themselves, not on the LLM.

### Running the evaluation pipeline

```bash
# 1. Generate the evaluation dataset (5 runs x temperatures 0 and 0.3 -> evaluation_results/llm_generations.csv)
python -m explanation.evaluation.llm_eval_runner --runs 5 --temperature 0 0.3

# 2. Structure and faithfulness are deterministic, rule-based checks - no extra setup needed
python -m explanation.evaluation.llm_structure
python -m explanation.evaluation.llm_faithfulness

# 3. Stability needs local embeddings via Ollama - pull the model once, then run
ollama pull qwen3-embedding:0.6b
python -m explanation.evaluation.llm_stability

# 4. Independent SHAP sanity check (no dependency on step 1)
python -m explanation.evaluation.additivity_check
```

Each command writes a per-patient CSV and a per-category summary CSV to `evaluation_results/`.

## Common Failure Modes

- Missing API key for the selected provider in `.env`
- Missing Stage 2 artifacts in `explanation_artifacts/`
- Row-count mismatch across saved artifact files
- Stability step (`llm_stability.py`) requires a running Ollama server with `qwen3-embedding:0.6b` pulled — `python -m explanation.evaluation.llm_stability` fails otherwise
- Structure/faithfulness/stability all read `evaluation_results/llm_generations.csv` — run `llm_eval_runner` first
- Missing provider-specific dependency such as `langchain-google-genai` or `langchain-openrouter`

If you hit one of these errors, first verify `config.yaml`, `.env`, and the contents of `explanation_artifacts/`.
