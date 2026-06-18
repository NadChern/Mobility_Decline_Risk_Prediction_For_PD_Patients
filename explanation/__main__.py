"""Run the explanation pipeline.

Usage: python -m explanation
"""
from .contract import PATIENT_ID_COLUMN
from .data_loader import (
    DEBUG, GOOGLE_API_KEY, LLM_PROVIDER, OPENROUTER_API_KEY, patient_ids,
)
from .llm import generate_explanation


# Default to the first patient in the test set so the demo always points at a
# PATNO that exists in the current artifacts. Override to inspect another.
patient_id = int(patient_ids[PATIENT_ID_COLUMN].iloc[0])

# Require the key for whichever provider config.yaml selected.
_required_key = GOOGLE_API_KEY if LLM_PROVIDER == "google" else OPENROUTER_API_KEY
_required_env = "GOOGLE_API_KEY" if LLM_PROVIDER == "google" else "OPENROUTER_API_KEY"

if not _required_key:
    print(
        f"{_required_env} is not set in .env, but config.yaml selects "
        f"provider '{LLM_PROVIDER}'.\n"
        f"Add {_required_env} to .env (or switch the provider in config.yaml) "
        "to run LLM explanation generation."
    )
else:
    try:
        result = generate_explanation(patient_id, debug=DEBUG)

        if DEBUG:
            print("=" * 70)
            print("PROMPT GENERATOR")
            print("=" * 70)
            print(result['prompt'])
            print()

        print("=" * 70)
        print("CLINICAL SUMMARY")
        print("=" * 70)
        print(result['explanation'])

    except Exception as exc:
        print(f"Error generating explanation: {exc}")
