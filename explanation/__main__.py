"""Run the explanation pipeline.

Usage: python -m explanation
"""
from .data_loader import DEBUG, OPENROUTER_API_KEY
from .llm import generate_explanation


# Change this PATNO to inspect a different patient.
patient_id = 4022

if not OPENROUTER_API_KEY:
    print(
        "OPENROUTER_API_KEY is not set in .env.\n"
        "Install `python-dotenv`, `langchain`, and `langchain-openrouter`,\n"
        "then add OPENROUTER_API_KEY to .env to run LLM explanation generation."
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
