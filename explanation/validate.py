"""Generate validation dataset for LLM explanations."""
import sys

def _print(msg):
    print(msg)
    sys.stdout.flush()

_print("Loading modules...")
import random
import time
import pandas as pd
from pathlib import Path
_print("Loading data artifacts...")
from .contract import PATIENT_ID_COLUMN
from .data_loader import X_test, patient_ids, y_pred_final
_print("Loading LLM module...")
from .llm import generate_explanation
_print("Ready.\n")


def create_validation_dataset(n_per_category=10, output_path=None, seed=42, max_retries=3):
    """Generate Excel file with patients and explanations by category.

    Args:
        n_per_category: Number of patients to sample per category (default 10)
        output_path: Path for output Excel file (default: validation_results.xlsx in project root)
        seed: Random seed for reproducibility (default 42)
        max_retries: Maximum retries per patient on API error (default 3)

    Returns:
        DataFrame with all patient data and explanations
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent / "validation_results.xlsx"

    output_path = Path(output_path)
    partial_path = output_path.with_suffix(".partial.xlsx")

    _print(f"Output: {output_path}")
    _print(f"Sampling {n_per_category} patients per category (seed={seed})")

    random.seed(seed)

    # Category mapping
    categories = {0: "No falls", 1: "Mild falls", 2: "Moderate falls"}

    # Check for partial results to resume
    processed_ids = set()
    rows = []
    if partial_path.exists():
        existing_df = pd.read_excel(partial_path)
        rows = existing_df.to_dict("records")
        processed_ids = set(existing_df["Patient_ID"].tolist())
        _print(f"Resuming from {len(rows)} previously processed patients")

    for cat_id, cat_label in categories.items():
        # Find patients in this category
        indices = [i for i, pred in enumerate(y_pred_final) if pred == cat_id]

        # Random sample of n patients (or all if fewer)
        selected = random.sample(indices, min(n_per_category, len(indices)))

        for idx in selected:
            patient_id = int(patient_ids.iloc[idx][PATIENT_ID_COLUMN])

            # Skip if already processed
            if patient_id in processed_ids:
                _print(f"Skipping {cat_label}: Patient {patient_id} (already processed)")
                continue

            # Get feature values
            features = X_test.iloc[idx].to_dict()

            # Generate explanation with retry logic
            _print(f"Calling LLM for {cat_label}: Patient {patient_id}...")
            explanation = None
            for attempt in range(max_retries):
                try:
                    result = generate_explanation(patient_id, debug=False)
                    explanation = result["explanation"]
                    break
                except Exception as e:
                    wait_time = 2 ** (attempt + 1)  # Exponential backoff: 2, 4, 8 seconds
                    _print(f"  Error for Patient {patient_id}: {type(e).__name__}: {e}")
                    if attempt < max_retries - 1:
                        _print(f"  Retry {attempt + 1}/{max_retries} in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        _print(f"  Failed after {max_retries} attempts")

            # Add delay between API calls to avoid rate limiting
            time.sleep(1)

            if explanation is None:
                _print(f"FAILED {cat_label}: Patient {patient_id} after {max_retries} retries")
                explanation = "[ERROR: Failed to generate explanation]"

            # Build row
            row = {
                "Category": cat_label,
                "Patient_ID": patient_id,
                **features,
                "LLM_Explanation": explanation,
            }
            rows.append(row)
            processed_ids.add(patient_id)
            _print(f"Processed {cat_label}: Patient {patient_id}")

            # Save partial progress after each patient
            pd.DataFrame(rows).to_excel(partial_path, index=False)

    # Create final DataFrame and save
    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)

    # Clean up partial file
    if partial_path.exists():
        partial_path.unlink()

    _print(f"\nSaved {len(df)} patients to {output_path}")
    return df


if __name__ == "__main__":
    create_validation_dataset()
