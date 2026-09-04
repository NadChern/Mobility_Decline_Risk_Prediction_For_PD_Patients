"""Build the pilot dataset and regenerate every completed evaluation family."""

import argparse

from . import completeness, complexity, fidelity, structure, unsupported_information
from .build_pilot_dataset import build_pilot_dataset


def run_all(rebuild_dataset=True, source_path=None):
    """Regenerate the matched pilot reports and all completed metric outputs."""
    if rebuild_dataset:
        if source_path is None:
            build_pilot_dataset()
        else:
            build_pilot_dataset(source_path=source_path)
    return {
        "fidelity": fidelity.run(),
        "completeness": completeness.run(),
        "unsupported_information": unsupported_information.run(),
        "structure": structure.run(),
        "complexity": complexity.run(),
    }


def main():
    parser = argparse.ArgumentParser(description="Regenerate the complete template pilot.")
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Reuse outputs/pilot/pilot_reports.csv instead of rebuilding it.",
    )
    parser.add_argument(
        "--source",
        help="Saved LLM generations CSV used to rebuild the paired pilot dataset.",
    )
    args = parser.parse_args()
    if args.skip_dataset and args.source:
        parser.error("--source cannot be combined with --skip-dataset")
    run_all(rebuild_dataset=not args.skip_dataset, source_path=args.source)


if __name__ == "__main__":
    main()
