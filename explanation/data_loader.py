"""Data loading and configuration for the explanation pipeline."""
import os
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from dotenv import load_dotenv

from .contract import ARTIFACT_FILENAMES, PATIENT_ID_COLUMN

load_dotenv()

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ARTIFACTS_DIR = PROJECT_ROOT / "explanation_artifacts"
FEATURE_MAP_PATH = BASE_DIR / "feature_map.csv"
CONFIG_PATH = BASE_DIR / "config.yaml"


# =============================================================================
# Configuration
# =============================================================================
def _load_config():
    """Load configuration from config.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}\n"
            "Please ensure config.yaml exists in the explanation/ folder."
        )
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

_config = _load_config()

# SHAP settings
COVERAGE_THRESHOLD = _config['shap']['coverage_threshold']

# LLM settings
LLM_PROVIDER = _config['llm']['provider']
LLM_TEMPERATURE = _config['llm']['temperature']
OPENROUTER_MODEL = _config['llm']['openrouter']['model']
GOOGLE_MODEL = _config['llm']['google']['model']

# Evaluation-only synthesis-judge settings. Kept separate from the explanation generator so a
# judge model cannot accidentally replace the production model.
_synthesis_judge_config = _config.get('evaluation', {}).get('synthesis_judges', {})
SYNTHESIS_JUDGE_PROVIDER = _synthesis_judge_config.get('provider', 'openrouter')
SYNTHESIS_JUDGE_TEMPERATURE = _synthesis_judge_config.get('temperature', 0)
SYNTHESIS_JUDGE_MAX_TOKENS = _synthesis_judge_config.get('max_tokens', 2048)
SYNTHESIS_JUDGE_REASONING = _synthesis_judge_config.get('reasoning')
SYNTHESIS_JUDGE_RETEST_PROFILES = _synthesis_judge_config.get('retest_profiles', {})
SYNTHESIS_JUDGE_REPETITIONS_PER_ORDER = _synthesis_judge_config.get(
    'repetitions_per_order', 3
)
SYNTHESIS_JUDGE_REVERSE_ORDER = _synthesis_judge_config.get('reverse_order', True)
SYNTHESIS_JUDGE_MODELS = tuple(
    model['id'] for model in _synthesis_judge_config.get('models', [])
)
SYNTHESIS_JUDGE_MODEL_SETTINGS = {
    model['id']: {key: model[key] for key in ('reasoning', 'max_tokens') if key in model}
    for model in _synthesis_judge_config.get('models', [])
}

# Debug mode
DEBUG = _config.get('debug', False)

# API keys from environment (secrets stay in .env, not config.yaml)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# =============================================================================
# Stage 1 Data Loading
# =============================================================================
shap_values = np.load(ARTIFACTS_DIR / ARTIFACT_FILENAMES['shap_stage1'])
X_test = pd.read_csv(ARTIFACTS_DIR / ARTIFACT_FILENAMES['x_test'])
y_pred_proba = np.load(ARTIFACTS_DIR / ARTIFACT_FILENAMES['y_pred_proba'])
patient_ids = pd.read_csv(ARTIFACTS_DIR / ARTIFACT_FILENAMES['patient_ids'])
feature_map = pd.read_csv(FEATURE_MAP_PATH)
feature_map_lookup = feature_map.set_index('feature_name').to_dict(orient='index')

# Validation checks for Stage 1
if len(patient_ids) != len(X_test):
    raise ValueError("patient_ids_test.csv and X_test.csv row counts do not match.")
if shap_values.shape[0] != len(X_test):
    raise ValueError("shap_values_stage1.npy row count does not match X_test.csv.")
if len(y_pred_proba) != len(X_test):
    raise ValueError("y_pred_proba.npy row count does not match X_test.csv.")


# =============================================================================
# Stage 2 Data Loading
# =============================================================================
STAGE2_AVAILABLE = False
routing_mask = None
shap_values_stage2 = None
y_pred_s2 = None
y_pred_final = None
_stage2_index_map = None


def load_stage2_artifacts():
    """Load Stage 2 artifacts. Raises error if missing."""
    global STAGE2_AVAILABLE, routing_mask, shap_values_stage2
    global y_pred_s2, y_pred_final, _stage2_index_map

    stage2_files = [
        ARTIFACT_FILENAMES['routing_mask'],
        ARTIFACT_FILENAMES['shap_stage2'],
        ARTIFACT_FILENAMES['y_pred_s2'],
        ARTIFACT_FILENAMES['y_pred_final'],
    ]

    # Check if all Stage 2 files exist
    missing = [f for f in stage2_files if not (ARTIFACTS_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Stage 2 artifacts not found: {', '.join(missing)}\n"
            f"Expected location: {ARTIFACTS_DIR}\n\n"
            "To generate the artifacts, run the modelling notebook's export cell\n"
            "(explanation.export.export_explanation_artifacts), which writes all\n"
            "required files into the explanation_artifacts/ folder."
        )

    try:
        routing_mask = np.load(ARTIFACTS_DIR / ARTIFACT_FILENAMES['routing_mask'])
        shap_values_stage2 = np.load(ARTIFACTS_DIR / ARTIFACT_FILENAMES['shap_stage2'])
        y_pred_s2 = np.load(ARTIFACTS_DIR / ARTIFACT_FILENAMES['y_pred_s2'])
        y_pred_final = np.load(ARTIFACTS_DIR / ARTIFACT_FILENAMES['y_pred_final'])

        # Validate Stage 2 artifacts
        if len(routing_mask) != len(X_test):
            raise ValueError("routing_mask.npy length does not match X_test.csv.")
        if routing_mask.sum() != len(shap_values_stage2):
            raise ValueError("shap_values_stage2.npy row count does not match routed patients.")
        if len(y_pred_s2) != routing_mask.sum():
            raise ValueError("y_pred_s2.npy length does not match routed patients.")
        if len(y_pred_final) != len(X_test):
            raise ValueError("y_pred_final.npy length does not match X_test.csv.")

        # Build mapping from test index to Stage 2 index
        routed_indices = np.where(routing_mask)[0]
        _stage2_index_map = {test_idx: s2_idx for s2_idx, test_idx in enumerate(routed_indices)}

        STAGE2_AVAILABLE = True
        return True

    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(
            f"Error loading Stage 2 artifacts: {e}\n\n"
            "The artifact files may be corrupted or have mismatched shapes.\n"
            "Re-run the export cell to regenerate the artifacts."
        ) from e


def get_stage2_index_map():
    """Return the stage 2 index map."""
    return _stage2_index_map


# Load Stage 2 artifacts at module import
load_stage2_artifacts()
