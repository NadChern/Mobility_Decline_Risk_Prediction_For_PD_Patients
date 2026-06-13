"""LLM explanation generation and formatting."""
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
    get_feature_metadata,
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


def _parse_value_label(scale_str, value):
    """Extract label for a value from scale string like '0 = No; 1 = Yes'.

    Args:
        scale_str: Scale description string with value mappings
        value: The numeric value to look up

    Returns:
        Label string if found, None otherwise
    """
    import re

    # Handle integer values
    try:
        int_val = int(value) if float(value) == int(float(value)) else None
    except (ValueError, TypeError):
        return None

    if int_val is not None:
        # Match patterns like "1 = Rare freezing" or "1=Yes"
        pattern = rf'\b{int_val}\s*=\s*([^;]+)'
        match = re.search(pattern, scale_str)
        if match:
            return match.group(1).strip()
    return None


def _format_value(value, metadata):
    """Format value consistently based on data type.

    Args:
        value: The raw value to format
        metadata: Feature metadata dict with 'data_type' key

    Returns:
        Formatted value string
    """
    data_type = metadata.get('data_type', '')

    try:
        float_val = float(value)
    except (ValueError, TypeError):
        return str(value)  # Return as-is if not numeric

    # Integer-like types: no decimals
    if data_type in ('continuous_integer', 'ordinal_numeric', 'ordinal_categorical', 'binary_categorical'):
        return str(int(float_val))
    else:
        # Continuous numeric: show decimal only if fractional
        return str(int(float_val)) if float_val == int(float_val) else f"{float_val:.2f}"


def _format_features_list(features):
    """Format features as a numbered list with full context.

    Args:
        features: List of feature dicts with 'feature', 'patient_value_formatted', 'shap_contribution'

    Returns:
        Formatted list string
    """
    if not features:
        return "(none identified)"

    lines = []
    for i, f in enumerate(features, 1):
        metadata = get_feature_metadata(f['feature'])
        value = f['patient_value_formatted']
        impact = f['shap_contribution']

        # Format value consistently
        formatted_value = _format_value(value, metadata)

        # Try to get value label from scale
        try:
            label = _parse_value_label(metadata['value_encoding_or_scale'], float(value))
        except (ValueError, TypeError):
            label = None
        value_display = f"{formatted_value} ({label})" if label else formatted_value

        lines.append(f"{i}. {metadata['description']} (Impact: {impact:+.3f})")
        lines.append(f"   Value: {value_display}")
        lines.append(f"   Scale: {metadata['value_encoding_or_scale']}")
        lines.append("")

    return "\n".join(lines)


def _build_prompt(patient_info):
    """Build the LLM prompt from patient explanation data.

    Args:
        patient_info: Dictionary from build_patient_explanation_data_full()

    Returns:
        String prompt for LLM
    """
    # Build DATA INPUTS section
    data_inputs = f"""DATA INPUTS:
Patient ID: {patient_info['patient_id']}
Final Prediction: {patient_info['final_prediction_label']}"""

    # Build STAGE 1 section (always shown)
    stage1_contributing = _format_features_list(
        patient_info['risk_increasing_features']
    )
    stage1_mitigating = _format_features_list(
        patient_info['risk_decreasing_features']
    )

    stage1_section = f"""---
STAGE 1: FALL RISK ASSESSMENT
These factors influenced whether the patient was predicted to have falls at all.

Risk-Contributing Factors:
{stage1_contributing}

Risk-Mitigating Factors:
{stage1_mitigating}"""

    # Build STAGE 2 section (only if routed)
    stage2_section = ""
    if patient_info['has_severity_assessment']:
        stage2_contributing = _format_features_list(
            patient_info['severity_increasing_features']
        )
        stage2_mitigating = _format_features_list(
            patient_info['severity_decreasing_features']
        )

        stage2_section = f"""

---
STAGE 2: SEVERITY ASSESSMENT
These factors influenced whether falls were predicted to be mild or moderate.

Severity-Contributing Factors:
{stage2_contributing}

Severity-Mitigating Factors:
{stage2_mitigating}"""

    # Build INSTRUCTIONS section
    stage2_instruction = """
**Stage 2: Fall Severity Assessment**
- Severity-Contributing Factors:
  • [bullet for each factor]
- Severity-Mitigating Factors:
  • [bullet for each factor]
""" if patient_info['has_severity_assessment'] else ""

    overview_template = f"The model classified this patient as {patient_info['final_prediction_label']}."

    instructions = f"""
---
INSTRUCTIONS:
You are a neurologist explaining a machine learning fall-risk prediction for a Parkinson’s disease patient.

IMPORTANT CONTEXT:
- The machine learning model predicts fall risk for a patient with Parkinson's disease.
- The LLM does not make the prediction; it only explains the model output.
- Do not introduce new features or change contribution direction.
- Do not infer additional clinical severity or causal relationships.
- Be clinically neutral and avoid deterministic statements.

TASK:
Write a structured clinical explanation summarizing the model’s fall-risk prediction for this patient. 
The explanation should:
1. Use the provided overview sentence exactly as written.
2. Lists ALL factors organized by section using bullet points.
3. Preserve section order and factor order.
3. End with one concise clinical summary.

BULLET FORMAT:
- Format: [clinical interpretation] ([abbreviated feature name] = [value or label])
- The clinical interpretation should be meaningful and understandable without abbreviations.
- Do NOT repeat the numeric value in the interpretation; the value appears in parentheses.
- Use short/abbreviated feature names in parentheses (e.g., "FOG", "MoCA", "H&Y stage").
- For UPDRS scores: use "UPDRS-I total", "UPDRS-III total" for part totals; use item names like "Postural Stability", "Gait" for individual items (not "UPDRS-III").
- Examples of GOOD bullets:
  • Postural instability present at diagnosis (Postural instability = Yes)
  • Rare freezing episodes (FOG = 1)
  • Moderate gait impairment requiring walking aid (Gait = 3)
  • Preserved cognition (MoCA = 30)
  • No non-motor symptoms (UPDRS-I total = 0)
  • Motor complications present (UPDRS-IV total = 14)
- Examples of BAD bullets (do NOT do this):
  • Yes (Postural instability symptom flag at diagnosis = 1)
  • MDS-UPDRS Part IV total score (MDS-UPDRS Part IV total score = 14)
  • Severe postural instability (UPDRS-III = 4)  // confusing: looks like total but is an item
  • Depressive symptoms score 9 (GDS-15 = 9)  // value "9" is duplicated

VALUE INTERPRETATION:
- Use scale labels when available (e.g., "Rare freezing" for FOG=1, not "Less freezing").
- Do NOT assume mitigating factors have low values or contributing factors have high values.
- For value 0, use "No/Absent/None" (e.g., "No non-motor symptoms" not "Lower non-motor burden").
- Do NOT use comparative words ("Lower", "Less", "Higher") unless comparing to a reference.

CLINICAL SUMMARY FORMAT:                                                                                                                                          
- Summarize the top 2–3 contributing and mitigating factors in one sentence.
- Use cautious language such as "were associated with" and "within this model."
- Focus on overall clinical themes rather than listing all factors.
- Do NOT repeat probabilities or section details.
- Do NOT use comparative age terms such as "older" or "younger."
- Use "no" for 0 values (e.g., "no daytime sleepiness").

OVERVIEW SENTENCE:
{overview_template}

OUTPUT FORMAT:

Clinical Fall Risk Summary for Patient ID: {patient_info['patient_id']}

[Use the overview sentence exactly as provided]

**Stage 1: Fall Risk Assessment**
- Risk-Contributing Factors:
  • [bullet for each factor]
- Risk-Mitigating Factors:
  • [bullet for each factor]

{stage2_instruction}

**Clinical Summary:** [one sentence]"""

    return f"{data_inputs}\n\n{stage1_section}{stage2_section}\n{instructions}"


def generate_explanation(patient_id, debug=None, provider=None, model_name=None, temperature=LLM_TEMPERATURE):
    """Generate a clinical explanation for a patient's fall risk prediction.

    This is the main entry point for generating explanations. It builds the prompt
    from patient data, calls the LLM, and returns both the prompt and explanation.

    Args:
        patient_id: Patient PATNO identifier
        debug: If True, prints full prompt to console. If None, uses config.yaml setting.
        provider: "openrouter" or "google". If None, uses config.yaml setting.
        model_name: Optional LLM model name override
        temperature: LLM temperature setting

    Returns:
        Dictionary with 'prompt', 'explanation', and 'model' keys
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
