"""Deterministic comparators for the LLM explanation experiment."""

import sys

from . import domain_template as _domain_template
from . import generic_template as _generic_template
from .generic_template import (
    METHOD_NAME,
    generate_template_explanation,
    render_template_explanation,
)
from .domain_template import (
    GROUPED_METHOD_NAME,
    generate_template_grouped_explanation,
    render_template_grouped_explanation,
)

# Compatibility for existing callers while the repository adopts the clearer module names.
sys.modules[f"{__name__}.renderer"] = _generic_template
sys.modules[f"{__name__}.domain_renderer"] = _domain_template

__all__ = [
    "METHOD_NAME",
    "GROUPED_METHOD_NAME",
    "generate_template_explanation",
    "generate_template_grouped_explanation",
    "render_template_explanation",
    "render_template_grouped_explanation",
]
