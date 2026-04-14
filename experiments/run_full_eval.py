#!/usr/bin/env python3
"""
Full DynaCU-Bench evaluation: 3 models × 2 modes × representative tasks.
Produces the main results table for the paper.
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from experiments.browser_eval import BrowserEvaluator, EvalResult
from dynacubench.tasks_v2 import DynaCUBench, TaskCategory


def main():
    bench = DynaCUBench()

    # Representative task subset: 1E + 1M per category = 20 tasks
    # This is the "quick eval" — full eval runs all 100
    task_ids = [
        # D: Transient UI — core thesis tasks
        "D-E1", "D-E2", "D-E3", "D-M4",
        # E: Audio Alerts
        "E-E1",
        # F: Animation
        "F-E1", "F-E2",
        # H: Sequential
        "H-E1", "H-E3",
        # I: Live Streams
        "I-E1", "I-E2",
        # G: Games (visual-only, control group)
        "G-E3",
    ]
    tasks = [t for t in bench if t.task_id in task_ids]

    models = ["gemini-3-flash", "claude-sonnet-4-6"]
    modes = ["standard", "aoi_full"]

    total_evals = len(models) * len(modes) * len(tasks)
    logger.info(
        "Full eval: %d models x %d modes x %d tasks = %d evaluations",
        len(models), len(modes), len(tasks), total_evals,
    )

    out_dir = Path("results/full_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[EvalResult] = []

    for model_name in models:
        for mode in modes:
            print(f"\n{'=' * 70}")
            print(f"  {model_name} + {mode}")
            print(f"{'=' * 70}")

            evaluator = BrowserEvaluator(
                model_name=model_name,
                observation_mode=mode,
                max_steps=10,
                step_interval_s=2.0,
            )

            for task in tasks:
                logger.info("--- %s (%s) ---", task.task_id, task.difficulty.value)
                result = evaluator.run_task(task)
                all_results.append(result)

                status = "PASS" if result.success else "FAIL"
                print(
                    f"  {status:4s} | {task.task_id:6s} "
                    f"| {result.result_val:25s} "
                    f"| steps={result.steps_taken} "
                    f"| model_ms={result.total_model_latency_ms:.0f}"
                )

            # Save after each model+mode combo
            combo = f"{model_name}_{mode}"
            combo_results = [
                r for r in all_results
                if r.model_name == model_name and r.observation_mode == mode
            ]
            with open(out_dir / f"{combo}.json", "w") as f:
                json.dump([r.to_dict() for r in combo_results], f, indent=2)

    # Save all results
    with open(out_dir / "all_results.json", "w") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)

    # Print summary table
    print_summary(all_results, models, modes)


def print_summary(results: list[EvalResult], models: list[str], modes: list[str]):
    """Print LaTeX-ready summary table."""
    groups = defaultdict(list)
    for r in results:
        groups[(r.model_name, r.observation_mode)].append(r)

    cat_groups = defaultdict(list)
    for r in results:
        cat_short = r.category.split("_")[0]
        cat_groups[(r.model_name, r.observation_mode, cat_short)].append(r)

    print(f"\n{'=' * 70}")
    print("  DynaCU-Bench Results — Main Table")
    print(f"{'=' * 70}\n")

    # Overall accuracy per model × mode
    header = f"{'Model':<22} {'Mode':<16} {'Pass':>5} {'Total':>6} {'Acc':>7} {'AvgSteps':>9} {'AvgLatMs':>9}"
    print(header)
    print("-" * len(header))

    for model in models:
        for mode in modes:
            res = groups[(model, mode)]
            n_pass = sum(1 for r in res if r.success)
            n_total = len(res)
            acc = 100.0 * n_pass / n_total if n_total else 0
            avg_steps = sum(r.steps_taken for r in res) / n_total if n_total else 0
            avg_lat = sum(r.total_model_latency_ms for r in res) / n_total if n_total else 0
            marker = " <--" if mode == "aoi_full" else ""
            print(
                f"{model:<22} {mode:<16} {n_pass:>5} "
                f"{n_total:>6} {acc:>6.1f}% {avg_steps:>9.1f} {avg_lat:>8.0f}ms{marker}"
            )
        print()

    # Per-category breakdown
    categories = sorted(set(r.category.split("_")[0] for r in results))
    print(f"\n{'Category':<12}", end="")
    for model in models:
        for mode in modes:
            label = f"{model[:8]}+{mode[:3]}"
            print(f" {label:>14}", end="")
    print()
    print("-" * (12 + 15 * len(models) * len(modes)))

    for cat in categories:
        print(f"{cat:<12}", end="")
        for model in models:
            for mode in modes:
                res = cat_groups[(model, mode, cat)]
                if res:
                    n_pass = sum(1 for r in res if r.success)
                    print(f" {n_pass}/{len(res):>11}", end="")
                else:
                    print(f" {'--':>14}", end="")
        print()

    # Delta table: standard → aoi_full improvement
    print(f"\n{'=' * 70}")
    print("  Improvement: standard → aoi_full")
    print(f"{'=' * 70}")
    for model in models:
        std_res = groups[(model, "standard")]
        aoi_res = groups[(model, "aoi_full")]
        std_pass = sum(1 for r in std_res if r.success)
        aoi_pass = sum(1 for r in aoi_res if r.success)
        n = len(std_res)
        delta = aoi_pass - std_pass
        print(
            f"  {model}: {std_pass}/{n} → {aoi_pass}/{n} "
            f"(delta: {'+' if delta >= 0 else ''}{delta})"
        )


if __name__ == "__main__":
    main()
