"""Metric 5: complexity / scalability / flexibility (deterministic, no LLM judge).

Part A — static complexity: how much authored machinery each method needs.
Part B — flexibility: how each method behaves on an UNSEEN feature (compositional generalization /
         systematicity for data-to-text). The LLM's zero-edit rows are CLAIMS that must be confirmed
         by actually generating and re-running the fidelity panel (compositional fidelity) — the
         literature shows neural models also hallucinate on unseen combinations, so we never assume
         it. Those rows are marked accordingly.

Honesty: the prompt is manual logic too, so its instruction length is counted rather than treated
as free. The production map-free prompt does not receive the experiment taxonomy.
"""

import argparse
import inspect
from pathlib import Path

import pandas as pd

from ..taxonomy import CATEGORY_LABELS, FEATURE_CATEGORY

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "template_experiment/outputs/pilot/complexity"


def _sloc(*funcs):
    """Approximate source lines of code: non-blank, non-comment lines of the given functions."""
    total = 0
    for func in funcs:
        for line in inspect.getsource(func).splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                total += 1
    return total


def _interpretation_prompt_words():
    """Word count of the MODEL INTERPRETATION instruction block in the production prompt."""
    from explanation import llm
    source = inspect.getsource(llm._build_prompt)
    start = source.find("- MODEL INTERPRETATION:")
    end = source.find("Clinical Fall Risk Summary", start)
    block = source[start:end] if start != -1 and end != -1 else ""
    return len(block.split())


def static_complexity():
    """Per-method authored-machinery counts. Some are structural facts about the system."""
    from .. import domain_template, generic_template

    generic_sloc = _sloc(generic_template._interpretation, generic_template._join_names)
    grouped_sloc = _sloc(
        domain_template._grouped_interpretation,
        domain_template._grouped_labels,
    )

    rows = [
        {"method": "Generic template", "templates": 1, "grouping_rules": 0,
         "map_entries": 0, "category_labels": 0, "prompts": 0,
         "embedded_category_examples": 0, "interp_sloc": generic_sloc,
         "prompt_words": 0},
        {"method": "Domain template", "templates": 1, "grouping_rules": 1,
         "map_entries": len(FEATURE_CATEGORY), "category_labels": len(CATEGORY_LABELS),
         "prompts": 0, "embedded_category_examples": 0, "interp_sloc": grouped_sloc,
         "prompt_words": 0},
        {"method": "LLM (Gemini)", "templates": 0, "grouping_rules": 0,
         "map_entries": 0, "category_labels": 0, "prompts": 1,
         # Auditable zero: the production map-free prompt supplies no taxonomy labels/mappings.
         "embedded_category_examples": 0, "interp_sloc": 0,
         "prompt_words": _interpretation_prompt_words()},
    ]
    return pd.DataFrame(rows)


def flexibility_unseen_feature():
    """How each method handles a brand-new feature not seen during development.

    'renders' = produces valid output with no code change. 'groups' = places the new feature into a
    higher-level category with no code change. Edit counts are the authored artifacts needed to
    restore grouping.
    """
    rows = [
        {"method": "Generic template",
         "renders_unseen_no_edits": True, "groups_unseen_no_edits": False,
         "edits_to_group_new_feature": "n/a (never groups)",
         "note": "Generic loop renders any factor list; it never groups, so nothing to maintain."},
        {"method": "Domain template",
         "renders_unseen_no_edits": True, "groups_unseen_no_edits": False,
         "edits_to_group_new_feature":
             "1 map entry for an existing category; +1 display label for a new category",
         "note": "Unmapped feature silently falls back to listing; grouping requires new artifacts."},
        {"method": "LLM (Gemini)",
         "renders_unseen_no_edits": True, "groups_unseen_no_edits": "CLAIM — verify",
         "edits_to_group_new_feature": "0 (from the same prompt) — VERIFY compositional fidelity",
         "note": "Must be confirmed by generation + re-running the fidelity panel on the new case; "
                 "neural D2T can hallucinate/omit on unseen combinations."},
    ]
    return pd.DataFrame(rows)


def write_report(static_df, flex_df, output_path):
    def md_table(df):
        header = "| " + " | ".join(df.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
        return "\n".join([header, sep] + body)

    output_path.write_text(
        "# Complexity / scalability / flexibility (Metric 5)\n\n"
        "## Part A — static complexity (authored machinery)\n\n" + md_table(static_df)
        + "\n\nThe prompt is counted as manual logic. It contains grouping instructions but receives "
          "no explicit experiment taxonomy labels or mappings. The honest claim "
          "is that the template's authored burden **grows with the feature/category set**, while the "
          "prompt's is roughly fixed. SLOC is approximate (non-blank, non-comment).\n\n"
          "## Part B — flexibility on an unseen feature (systematicity)\n\n" + md_table(flex_df)
        + "\n\nThe LLM's zero-edit rows are **claims to be confirmed by generation**, not assumptions: "
          "we must generate on the held-out feature/combination and re-run the fidelity panel "
          "(compositional fidelity), because neural data-to-text models also hallucinate or omit on "
          "unseen combinations. Report whatever we find, including cases where the LLM needs no edits "
          "but is less faithful.\n\n"
          "**Next (needs generation + frozen dev/held-out split):** per held-out case, log edits "
          "required, success-without-modification, and compositional fidelity for all three arms.\n",
        encoding="utf-8")


def run(output_dir=DEFAULT_OUTPUT_DIR):
    static_df = static_complexity()
    flex_df = flexibility_unseen_feature()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    static_df.to_csv(output_dir / "static_counts.csv", index=False)
    flex_df.to_csv(output_dir / "unseen_feature_assumptions.csv", index=False)
    write_report(static_df, flex_df, output_dir / "report.md")
    return static_df, flex_df


def main():
    parser = argparse.ArgumentParser(description="Report Metric 5 (complexity/scalability/flexibility).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
