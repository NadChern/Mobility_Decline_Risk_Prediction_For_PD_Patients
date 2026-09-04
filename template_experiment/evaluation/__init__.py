"""Evaluation modules for deterministic-template versus LLM comparisons.

Compatibility aliases preserve imports used by earlier pilot notebooks while new code uses the
shorter, purpose-based module names.
"""

from importlib import import_module
import sys

_ALIASES = {
    "build_gemini_pairs": "build_pilot_dataset",
    "metric_utils": "shared",
    "interpretation_fidelity": "prose_checks",
    "fidelity_comparison": "fidelity",
    "completeness_comparison": "completeness",
    "unsupported_information_comparison": "unsupported_information",
    "structure_comparison": "structure",
    "synthesis_comparison": "synthesis",
    "complexity_scalability": "complexity",
}

for _old_name, _new_name in _ALIASES.items():
    sys.modules[f"{__name__}.{_old_name}"] = import_module(f".{_new_name}", __name__)

sys.modules[f"{__name__}.geval_synthesis_judge"] = import_module(
    "..experimental.llm_judge", __name__
)
sys.modules[f"{__name__}.flexibility_stress_test"] = import_module(
    "..experimental.flexibility_stress_test", __name__
)
