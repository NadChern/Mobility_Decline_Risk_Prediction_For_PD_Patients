"""Explanation Layer for Fall Prediction Model (stage 1, for now)"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ARTIFACTS_DIR = PROJECT_ROOT / "explanation_artifacts"
FEATURE_MAP_PATH = BASE_DIR / "feature_map.csv"

OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# Load data
shap_values = np.load(ARTIFACTS_DIR / 'shap_values_stage1.npy')
X_test = pd.read_csv(ARTIFACTS_DIR / 'X_test.csv')
y_pred_proba = np.load(ARTIFACTS_DIR / 'y_pred_proba.npy')  # Stage-1 probability of any fall
patient_ids_path = ARTIFACTS_DIR / 'patient_ids_test.csv'
patient_ids = pd.read_csv(patient_ids_path)
feature_map = pd.read_csv(FEATURE_MAP_PATH)
feature_map_lookup = feature_map.set_index('feature_name').to_dict(orient='index')

# Validation checks
if len(patient_ids) != len(X_test):
    raise ValueError("patient_ids_test.csv and X_test.csv row counts do not match.")
if shap_values.shape[0] != len(X_test):
    raise ValueError("shap_values_stage1.npy row count does not match X_test.csv.")
if len(y_pred_proba) != len(X_test):
    raise ValueError("y_pred_proba.npy row count does not match X_test.csv.")

# Step 1: Global Feature Importance 
# Take absolute SHAP values, compute average across patients for each feature.
# Sort features from highest importance to lowest.
global_importance = pd.DataFrame({
    'feature': X_test.columns,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

def get_top_features(n=10):
    """Return the top n globally influential features."""
    return global_importance.head(n).copy()


# Find which row contains that patient's PATNO
def get_patient_index(patient_id):
    """Return the aligned row index for a PATNO in saved artifacts."""
    matches = patient_ids.index[patient_ids["PATNO"] == patient_id].tolist()
    if not matches:
        # Stop early if the requested PATNO is not in the saved test-set IDs.
        raise KeyError(f"Patient ID {patient_id} not found in patient_ids_test.csv.")
    if len(matches) > 1:
        # Each patient should map to exactly one saved artifact row.
        raise ValueError(f"Patient ID {patient_id} appears multiple times in patient_ids_test.csv.")
    return matches[0]


def format_feature_value(val):
    """Format feature values safely for prompts and text output."""
    # Avoid raw NaN appearing in clinician-facing text.
    if pd.isna(val):
        return "missing"
    # Keep numeric values consistent and easy to read in prompts.
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{float(val):.2f}"
    # Fall back to plain string formatting for any non-numeric value.
    return str(val)


def get_feature_metadata(feature_name):
    """Return concise human-readable metadata for a model feature."""
    return feature_map_lookup.get(
        feature_name,
        {
            "description": feature_name,
            "data_type": "unknown",
            "value_encoding_or_scale": "",
        },
    )

# We should define risk band (example for now)
def get_risk_level(prob):
    """Map Stage-1 probability of any fall into a readable risk band."""
    # These are explanation bands, not model-trained labels.
    if prob < 0.30:
        return "low"
    if prob < 0.60:
        return "moderate"
    return "high"


def build_patient_explanation_data(patient_idx, top_positive=3, top_negative=2):
    """Build structured per-patient explanation data from saved artifacts."""
    patient_shap = shap_values[patient_idx]
    patient_values = X_test.iloc[patient_idx]
    fall_probability = float(y_pred_proba[patient_idx])

    # Separate factors that push this patient's predicted risk up vs down.
    pos_idx = np.where(patient_shap > 0)[0]
    neg_idx = np.where(patient_shap < 0)[0]

    # Rank the strongest positive and negative local contributions separately.
    pos_sorted = pos_idx[np.argsort(patient_shap[pos_idx])[::-1]][:top_positive]
    neg_sorted = neg_idx[np.argsort(np.abs(patient_shap[neg_idx]))[::-1]][:top_negative]

    def build_feature_entry(idx):
        name = X_test.columns[idx]
        val = patient_values.iloc[idx]
        contrib = float(patient_shap[idx])
        return {
            "feature": name,
            "patient_value": val,
            "patient_value_formatted": format_feature_value(val),
            "shap_contribution": contrib,
            "direction": "increases fall risk" if contrib > 0 else "decreases fall risk",
        }

    top_risk_features = [build_feature_entry(idx) for idx in pos_sorted]
    top_protective_features = [build_feature_entry(idx) for idx in neg_sorted]

    return {
        "patient_idx": patient_idx,
        "patient_id": patient_ids.iloc[patient_idx]["PATNO"],
        "predicted_probability_any_fall": fall_probability,
        "risk_level": get_risk_level(fall_probability),
        "top_risk_features": top_risk_features,
        "top_protective_features": top_protective_features,
    }

def build_patient_explanation_data_by_id(patient_id, top_positive=3, top_negative=2):
    """Build structured explanation data using stable patient ID lookup."""
    patient_idx = get_patient_index(patient_id)
    return build_patient_explanation_data(
        patient_idx, top_positive=top_positive, top_negative=top_negative
    )

# Step 2: Explanation Formatter
def format_explanation(patient_idx, top_n=5):
    """Convert SHAP values to structured text for a patient."""
    # Split the requested count across risk-increasing and risk-decreasing factors.
    top_positive = max(1, top_n // 2 + top_n % 2)
    top_negative = max(1, top_n // 2)
    patient_info = build_patient_explanation_data(
        patient_idx, top_positive=top_positive, top_negative=top_negative
    )
    global_imp_dict = dict(zip(global_importance['feature'], global_importance['importance']))

    lines = [
        "Top factors influencing this patient's predicted probability of any fall:",
        f"Predicted probability of any fall: {patient_info['predicted_probability_any_fall']:.2f}",
        f"Risk level: {patient_info['risk_level']}",
    ]
    lines.append("-" * 60)

    if patient_info["top_risk_features"]:
        lines.append("Risk-increasing factors:")
    for rank, feature_info in enumerate(patient_info["top_risk_features"], 1):
        name = feature_info["feature"]
        global_imp = global_imp_dict.get(name, 0)
        lines.append(f"{rank}. {name}")
        lines.append(f"   Global importance: {global_imp:.3f}")
        lines.append(f"   Patient value: {feature_info['patient_value_formatted']}")
        lines.append(f"   Local contribution: {feature_info['shap_contribution']:+.3f}")
        lines.append("")

    if patient_info["top_protective_features"]:
        lines.append("Risk-decreasing factors:")
    for rank, feature_info in enumerate(patient_info["top_protective_features"], 1):
        name = feature_info["feature"]
        global_imp = global_imp_dict.get(name, 0)
        lines.append(f"{rank}. {name}")
        lines.append(f"   Global importance: {global_imp:.3f}")
        lines.append(f"   Patient value: {feature_info['patient_value_formatted']}")
        lines.append(f"   Local contribution: {feature_info['shap_contribution']:+.3f}")
        lines.append("")
    return "\n".join(lines)

def format_explanation_by_id(patient_id, top_n=5):
    """Convert SHAP values to structured text for a patient identified by PATNO."""
    patient_idx = get_patient_index(patient_id)
    return format_explanation(patient_idx, top_n=top_n)

# Step 3: Automatic Prompt Generator
def generate_llm_prompt(patient_info):
    """Generate an LLM prompt from structured patient explanation data."""
    # Turn structured feature dictionaries into readable bullet lists for the LLM.
    risk_lines = []
    for feature_info in patient_info["top_risk_features"]:
        metadata = get_feature_metadata(feature_info["feature"])
        risk_lines.append(
            f"- Feature: {feature_info['feature']}\n"
            f"  Patient value: {feature_info['patient_value_formatted']}\n"
            f"  Feature note: {metadata['description']}\n"
            f"  Scale: {metadata['value_encoding_or_scale']}\n"
            f"  SHAP contribution: {feature_info['shap_contribution']:+.3f} (increases fall risk)"
        )

    protective_lines = []
    for feature_info in patient_info["top_protective_features"]:
        metadata = get_feature_metadata(feature_info["feature"])
        protective_lines.append(
            f"- Feature: {feature_info['feature']}\n"
            f"  Patient value: {feature_info['patient_value_formatted']}\n"
            f"  Feature note: {metadata['description']}\n"
            f"  Scale: {metadata['value_encoding_or_scale']}\n"
            f"  SHAP contribution: {feature_info['shap_contribution']:+.3f} (decreases fall risk)"
        )

    risk_text = chr(10).join(risk_lines) if risk_lines else "- None identified"
    protective_text = chr(10).join(protective_lines) if protective_lines else "- None identified"

    return f"""You are a neurologist explaining a machine learning fall-risk prediction.

Important context:
- The machine learning model predicts fall risk for a patient with Parkinson's disease.
- The LLM does not make the prediction; it only explains the model output.
- A positive SHAP contribution means the feature increases the predicted probability of any fall.
- A negative SHAP contribution means the feature decreases the predicted probability of any fall.
- Explain the result in simple clinical language for a doctor or caregiver.
- Do not overclaim causality. Say "is associated with" or "may contribute to" instead of "causes."

Prediction:
- Patient ID: {patient_info['patient_id']}
- Predicted probability of any fall: {patient_info['predicted_probability_any_fall']:.2f}
- Risk category: {patient_info['risk_level']}

Top patient-specific risk-increasing factors:
{risk_text}

Top patient-specific risk-decreasing factors:
{protective_text}

Task:
Begin the output with this exact title on the first line:
Clinical Fall Risk Summary for Patient {patient_info['patient_id']}

Write a short clinical explanation in 3-5 sentences.
Mention:
1. Whether the estimated fall risk is low, moderate, or high.
2. Which factors increase the risk.
3. Which factors decrease the risk, if any.
4. A short safety-focused conclusion.

Output:
"""

def build_langchain_llm(model_name=None, temperature=0):
    """Create a LangChain ChatOpenRouter client for LLM explanation generation."""
    try:
        from langchain_openrouter import ChatOpenRouter
    except ImportError as exc:
        raise ImportError(
            "langchain-openrouter is not installed. Install it with "
            "`pip install -U langchain langchain-openrouter`."
        ) from exc

    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add your OpenRouter API key to the .env file before "
            "running Step 4."
        )

    return ChatOpenRouter(
        model=model_name or OPENROUTER_MODEL,
        temperature=temperature,
        api_key=OPENROUTER_API_KEY,
    )


def generate_llm_explanation(patient_info, model_name=None, temperature=0):
    """Run Step 4: send the prompt to an LLM through LangChain + OpenRouter."""
    llm = build_langchain_llm(model_name=model_name, temperature=temperature)
    prompt = generate_llm_prompt(patient_info)
    response = llm.invoke(prompt)
    return {
        "model": model_name or OPENROUTER_MODEL,
        "prompt": prompt,
        "explanation": response.content,
    }


if __name__ == "__main__":
    # Change this PATNO to inspect a different patient.
    patient_id = 101477
    patient_idx = get_patient_index(patient_id)

    # Test Step 1
    print("=== STEP 1: Top 10 Global Feature Importance ===")
    print(get_top_features(10).to_string(index=False))

    # Test Step 2
    print(f"\n=== STEP 2: Explanation Formatter (Patient idx={patient_idx}, PATNO={patient_id}) ===")
    print(format_explanation_by_id(patient_id))

    # Test Step 3
    patient_info = build_patient_explanation_data_by_id(patient_id)
    actual_prob = patient_info["predicted_probability_any_fall"]
    print(
        f"\n=== STEP 3: Prompt Generator (Patient idx={patient_idx}, PATNO={patient_id}, "
        f"prob={actual_prob:.2f}) ==="
    )
    prompt = generate_llm_prompt(patient_info)
    print(prompt)

    # Test Step 4
    print("\n=== STEP 4: LLM Explanation Generation ===")
    if OPENROUTER_API_KEY:
        try:
            llm_result = generate_llm_explanation(patient_info)
            print(f"Model: {llm_result['model']}")
            print(llm_result["explanation"])
        except Exception as exc:
            print(f"Step 4 could not run: {exc}")
    else:
        print(
            "Skipping Step 4 because OPENROUTER_API_KEY is not set in .env. "
            "Install `python-dotenv`, `langchain`, and `langchain-openrouter`, "
            "then add OPENROUTER_API_KEY to .env to run LLM explanation generation."
        )
