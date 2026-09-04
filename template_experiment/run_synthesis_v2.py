"""Orchestrate all synthesis-v2 steps that do not require human adjudication."""

import argparse
import time

from .evaluation.map_free_analysis import AUTHOR_WORKSHEET, run as run_map_free
from .evaluation.parser_validation import run as run_parser
from .evaluation.part1_structural import run as run_part1
from .evaluation.reproducibility import run as run_reproducibility
from .evaluation.recheck_dataset_a import OUTPUT_DIR as CHECKER_OUTPUT, RECHECKED_RECORDS, run as run_recheck
from .experimental.flexibility_stress_test import run as run_flexibility
from .experimental.llm_judge import run as run_judges
from .experimental.llm_judge_map_free import run as run_map_free_judges
from .generate_dataset_a import run as run_dataset_a
from .synthesis_protocol import freeze_protocol
from .run_support import progress


def run(generate=False, judge=False, request_timeout=120, judge_max_calls_per_panel=None):
    if request_timeout <= 0:
        raise ValueError("Request timeout must be positive.")
    stages = [
        ("Dataset A: taxonomy-conditioned reports", lambda: run_dataset_a(generate=generate, request_timeout=request_timeout)),
        ("Recheck saved reports with versioned checker (no API calls)", run_recheck),
        ("Parser validation", lambda: run_parser(output_dir=CHECKER_OUTPUT / "parser_validation")),
        ("Part 1 structural analysis / checker status", lambda: run_part1(records_path=RECHECKED_RECORDS, output_dir=CHECKER_OUTPUT / "part1_structural")),
        ("Versioned map-free analysis", lambda: run_map_free(output_dir=CHECKER_OUTPUT / "part2a_map_free", adjudications_path=AUTHOR_WORKSHEET)),
        ("Dataset C: held-out reports", lambda: run_flexibility(generate=generate, request_timeout=request_timeout)),
        ("Part 1 judges" if judge else "Prepare Part 1 judge items (no API calls)", lambda: run_judges(records_path=RECHECKED_RECORDS, output_dir=CHECKER_OUTPUT / "judge_part1_runtime_v3", execute=judge, request_timeout=request_timeout, max_calls=judge_max_calls_per_panel)),
        ("Map-free judges" if judge else "Prepare map-free judge items (no API calls)", lambda: run_map_free_judges(output_dir=CHECKER_OUTPUT / "judge_part2_runtime_v3", execute=judge, worksheet_path=CHECKER_OUTPUT / "part2a_map_free/disagreement_worksheet.csv", request_timeout=request_timeout, max_calls=judge_max_calls_per_panel)),
        ("Check frozen protocol (preserve existing manifest)", freeze_protocol),
        ("Deterministic reproducibility audit (no API calls)", lambda: run_reproducibility(
            output_path=CHECKER_OUTPUT / "provenance/freeze_checksums.csv",
            map_free_output_dir=CHECKER_OUTPUT / "part2a_map_free", parser_output_dir=CHECKER_OUTPUT / "parser_validation")),
    ]
    for index, (name, action) in enumerate(stages, 1):
        started = time.monotonic()
        progress(f"Stage {index}/{len(stages)}: {name}")
        try:
            action()
        except BaseException:
            progress(f"Stage {index}/{len(stages)} stopped: {name}. Saved generation checkpoints are retained.")
            raise
        progress(f"Stage {index}/{len(stages)} finished ({time.monotonic() - started:.1f}s)")
    progress("Pipeline finished. See dataset status reports for contract flags and review requirements.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Call Gemini for Dataset A and C.")
    parser.add_argument("--judge", action="store_true", help="Call both configured OpenRouter judges.")
    parser.add_argument("--request-timeout", type=int, default=120, help="Per-request Gemini/judge timeout in seconds (default: 120).")
    parser.add_argument("--judge-max-calls-per-panel", type=int, help="Cap new calls separately for each judge panel (e.g. 2 permits up to 4 calls total).")
    args = parser.parse_args()
    run(args.generate, args.judge, args.request_timeout, args.judge_max_calls_per_panel)
