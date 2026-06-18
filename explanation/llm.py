"""LLM explanation generation and formatting.

All deterministic facts — which features (0.90 |SHAP| coverage), values, SHAP sign,
ordering, routing, prediction, and the display fields (short_name, formatted value,
interpretation, displayable) — are computed upstream in `explanation_builder`. This module
only groups those ready-made factors into the two direction tables and asks the LLM to
render the clinician document (title + Model Prediction + two factor tables + a model
interpretation). The LLM does layout and synthesis only; it must not alter, add, or drop
facts (the evaluation checks this).

Direction is expressed as which way a factor pushed the model's prediction:
  - Stage 1 (No Fall): "Toward No Fall" (SHAP < 0) vs "Toward Fall" (SHAP > 0)
  - Stage 2 (severity): "Toward Lower Severity (Mild)" (SHAP < 0) vs
                        "Toward Higher Severity (Moderate)" (SHAP > 0)
The table supporting the actual prediction is shown first (drivers first).
"""
from .data_loader import (
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OPENROUTER_MODEL,
    OPENROUTER_API_KEY,
    GOOGLE_MODEL,
    GOOGLE_API_KEY,
    DEBUG,
)
from .explanation_builder import (
    get_patient_index,
    build_patient_explanation_data_full,
)


def _extract_text_content(response):
    """Extract text content from LLM response regardless of format.

    Handles differences between providers (OpenRouter, Google, etc.)
    """
    content = response.content

    # Already a string - most common case
    if isinstance(content, str):
        return content

    # List of content blocks (Google, some Anthropic responses)
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and 'text' in block:
                text_parts.append(block['text'])
        return ''.join(text_parts) if text_parts else str(content)

    # Fallback - convert to string
    return str(content)


def build_langchain_llm(provider=None, model_name=None, temperature=LLM_TEMPERATURE):
    """Create a LangChain LLM client for explanation generation.

    Args:
        provider: "openrouter" or "google". If None, uses config.yaml setting.
        model_name: Model name override. If None, uses config.yaml setting for the provider.
        temperature: LLM temperature setting.

    Returns:
        LangChain chat model instance.
    """
    provider = provider or LLM_PROVIDER

    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError as exc:
            raise ImportError(
                "langchain-openrouter is not installed. Install it with "
                "`uv pip install langchain-openrouter`."
            ) from exc

        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Add your OpenRouter API key to the .env file."
            )

        return ChatOpenRouter(
            model=model_name or OPENROUTER_MODEL,
            temperature=temperature,
            api_key=OPENROUTER_API_KEY,
        )

    elif provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is not installed. Install it with "
                "`uv pip install langchain-google-genai`."
            ) from exc

        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Add your Google API key to the .env file."
            )

        return ChatGoogleGenerativeAI(
            model=model_name or GOOGLE_MODEL,
            temperature=temperature,
            google_api_key=GOOGLE_API_KEY,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. Use 'openrouter' or 'google'."
        )


_SEVERITY_LABEL = {1: "Mild Falls", 2: "Moderate Falls"}
_OUTCOME_PHRASE = {
    0: "No Fall classification",
    1: "Mild Fall classification",
    2: "Moderate Fall classification",
}

# Stage-2 table headers (fixed wording).
_HIGHER = "Factors Pushing the Prediction Toward Higher Severity (Moderate)"
_LOWER = "Factors Pushing the Prediction Toward Lower Severity (Mild)"
# Stage-1 table headers.
_TOWARD_FALL = "Factors Pushing the Prediction Toward Fall"
_TOWARD_NOFALL = "Factors Pushing the Prediction Toward No Fall"


def _displayable(factors):
    """Drop factors flagged non-displayable (NaN / unable-to-rate) by the builder."""
    return [f for f in factors if f.get("displayable", True)]


def _ordered_tables(patient_info):
    """Return [(header, factors), (header, factors)] in display order.

    The table supporting the actual prediction (the drivers) is listed first. Non-displayable
    factors are removed. `*_increasing_features` = SHAP > 0, `*_decreasing_features` = SHAP < 0.
    """
    fp = patient_info["final_prediction"]
    if patient_info["has_severity_assessment"]:
        higher = (_HIGHER, patient_info["severity_increasing_features"])   # toward Moderate
        lower = (_LOWER, patient_info["severity_decreasing_features"])     # toward Mild
        tables = [higher, lower] if fp == 2 else [lower, higher]
    else:
        toward_fall = (_TOWARD_FALL, patient_info["risk_increasing_features"])
        toward_nofall = (_TOWARD_NOFALL, patient_info["risk_decreasing_features"])
        tables = [toward_nofall, toward_fall]
    return [(header, _displayable(factors)) for header, factors in tables]


def _bullet_lines(factors):
    """Render factors as bulleted grounding lines for the prompt."""
    if not factors:
        return "  (none)"
    return "\n".join(
        f"- {f['short_name']} | value: {f['display_value']} | scale/interpretation: {f['interpretation']}"
        for f in factors
    )


def _model_prediction_block(patient_info):
    """Render the Model Prediction header lines."""
    fp = patient_info["final_prediction"]
    fall = "Fall" if fp in (1, 2) else "No Fall"
    block = f"- Fall Classification: {fall}"
    if patient_info["has_severity_assessment"]:
        block += f"\n- Fall Severity Classification: {_SEVERITY_LABEL.get(fp, '')}"
    return block


def _build_prompt(patient_info):
    """Build the full-document LLM prompt from the deterministic explanation packet."""
    pid = patient_info["patient_id"]
    fp = patient_info["final_prediction"]
    model_pred_block = _model_prediction_block(patient_info)
    outcome_phrase = _OUTCOME_PHRASE.get(fp, "prediction")
    (first_header, first_factors), (second_header, second_factors) = _ordered_tables(patient_info)

    grounding = (
        f"{first_header}:\n{_bullet_lines(first_factors)}\n\n"
        f"{second_header}:\n{_bullet_lines(second_factors)}"
    )

    return f"""You are a clinician explaining a machine-learning fall-risk prediction for a Parkinson's disease patient.

IMPORTANT CONTEXT:
- The machine-learning model predicts fall risk for a patient with Parkinson's disease.
- You do NOT make the prediction; you only explain the model's output.
- Use the provided factors, values, and scales exactly. Do NOT introduce new features, change any value, or move a factor between the two groups.
- Do NOT infer additional clinical severity or causal relationships.
- Be clinically neutral and avoid deterministic statements. Do NOT include probabilities, percentages, or model scores.
- A factor's group shows the direction it pushed the model's prediction, not a clinical judgement of risk. Do NOT describe a factor as clinically "protective" or a "risk factor" unless its Interpretation / Scale text explicitly supports it.
- When a factor value has an explicit interpretation in the Interpretation / Scale text, use that interpretation (e.g., absence of freezing of gait, normal postural stability, no motor complications).
- Do not assign additional clinical meaning beyond what is explicitly provided in the Interpretation / Scale text.

DATA (already computed by the model):

Patient ID: {pid}
Model Prediction:
{model_pred_block}

{grounding}

TASK:
Produce the clinician document defined under OUTPUT FORMAT — the Model Prediction header, the two factor tables, and a model interpretation — using only the data above.

INSTRUCTIONS:
- Build TWO tables exactly as grounded, in the given order and with the given headers. Keep each factor in its group and order; number rows 1..N within each table. For each row: Factor = the name before '|', Patient Value = the value after 'value:', Interpretation / Scale = the FULL text after 'scale/interpretation:'. Reproduce the Interpretation / Scale text EXACTLY and IN FULL — include every scale level and word; do NOT shorten it to the row's matching label, paraphrase it, or omit any part. Copy the Patient Value verbatim too. If a group has no factors, write "No qualifying factors." in place of that table.
- MODEL INTERPRETATION: write 2-4 sentences explaining the model's {outcome_phrase}.
  1. Begin with "Within this model, the {outcome_phrase} was primarily associated with …".
  2. Summarize the top 3–5 most influential factors from the FIRST table (those supporting the prediction) in 1–2 broader categories when possible (e.g., gait and mobility, motor complications, non-motor symptoms, cognition, disease duration) rather than restating every factor individually.  
  3. Then name the top 2-3 most influential factors from the SECOND table that supported the opposite direction, and note that they did not outweigh the factors supporting the {outcome_phrase}.
  Every factor you mention must appear in a table. Use cautious language ("were associated with", "within this model"). Do NOT imply causality. Do NOT describe factors as clinically protective or risky unless the Interpretation / Scale text supports it. Do NOT introduce features not in the tables. Use "no" or "absence of" for a value of 0.

OUTPUT FORMAT (use this structure exactly):

Clinical Fall Risk Summary for Patient ID: {pid}

Model Prediction
{model_pred_block}

Factors are ordered from most to least influential based on their contribution to the model prediction.

{first_header}
| # | Factor | Patient Value | Interpretation / Scale |
| --- | --- | --- | --- |
[{first_header} rows, numbered 1..N]

{second_header}
| # | Factor | Patient Value | Interpretation / Scale |
| --- | --- | --- | --- |
[{second_header} rows, numbered 1..M]

Model Interpretation
[3-4 sentence interpretation]

Clinical Note
- Model predictions are intended to support clinical review and should not replace clinical judgment.
- The factors shown above identify the patient characteristics that most influenced the model prediction and provide insight into the model’s reasoning.
- For definitions and interpretation of all variables used by the model, please refer to the Feature Reference Guide (feature_map.xlsx)."""


def generate_explanation(patient_id, debug=None, provider=None, model_name=None, temperature=LLM_TEMPERATURE):
    """Generate a clinical explanation document for a patient's fall risk prediction.

    Builds the prompt from the deterministic packet, calls the LLM, and returns the
    rendered document along with the prompt and model identity.

    Args:
        patient_id: Patient PATNO identifier
        debug: If True, prints full prompt to console. If None, uses config.yaml setting.
        provider: "openrouter" or "google". If None, uses config.yaml setting.
        model_name: Optional LLM model name override
        temperature: LLM temperature setting

    Returns:
        Dictionary with 'provider', 'model', 'prompt', and 'explanation' keys
    """
    if debug is None:
        debug = DEBUG

    # Get patient data and build prompt
    patient_idx = get_patient_index(patient_id)
    patient_info = build_patient_explanation_data_full(patient_idx)
    prompt = _build_prompt(patient_info)

    # Determine provider and model
    active_provider = provider or LLM_PROVIDER
    if model_name:
        model = model_name
    elif active_provider == "google":
        model = GOOGLE_MODEL
    else:
        model = OPENROUTER_MODEL

    # Call LLM
    if debug:
        print(f"Connecting to {active_provider} ({model})...")
    llm = build_langchain_llm(provider=active_provider, model_name=model_name, temperature=temperature)
    response = llm.invoke(prompt)
    if debug:
        print("Response received.\n")

    return {
        "provider": active_provider,
        "model": model,
        "prompt": prompt,
        "explanation": _extract_text_content(response),
    }
