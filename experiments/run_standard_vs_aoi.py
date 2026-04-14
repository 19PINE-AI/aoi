#!/usr/bin/env python3
"""
Standard vs. AOI comparison on dynamic tasks.
Demonstrates the core thesis: screenshot-only agents fail on
timing-dependent tasks; AOI observation fixes this.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

from experiments.browser_eval import BrowserEvaluator
from dynacubench.tasks_v2 import DynaCUBench


def main():
    bench = DynaCUBench()

    # Tasks that test the thesis -- dynamic/transient content
    test_ids = ["D-E1", "D-E3", "D-M4", "F-E2", "H-E1", "I-E2"]
    tasks = [t for t in bench if t.task_id in test_ids]

    if not tasks:
        print("No matching tasks found!")
        return

    results = {}
    for mode in ["standard", "aoi_full"]:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  MODE: {mode}")
        print(sep)

        evaluator = BrowserEvaluator(
            model_name="gemini-2.5-flash",
            observation_mode=mode,
            max_steps=8,
            step_interval_s=2.0,
        )

        results[mode] = []
        for task in tasks:
            result = evaluator.run_task(task)
            results[mode].append(result)
            status = "PASS" if result.success else "FAIL"
            print(
                f"  {status:4s} | {task.task_id:6s} | {result.result_val:25s} "
                f"| steps={result.steps_taken} | {result.total_time_s:.1f}s"
            )

    # Summary comparison
    sep = "=" * 60
    print(f"\n{sep}")
    print("  COMPARISON: standard vs aoi_full")
    print(sep)
    print(f"  {'Task':<8} {'Standard':<12} {'AOI Full':<12} Delta")
    print(f"  {'-' * 40}")
    for i, task in enumerate(tasks):
        std = results["standard"][i]
        aoi = results["aoi_full"][i]
        std_s = "PASS" if std.success else "FAIL"
        aoi_s = "PASS" if aoi.success else "FAIL"
        if aoi.success and not std.success:
            delta = "+AOI"
        elif std.success == aoi.success:
            delta = "same"
        else:
            delta = "-AOI"
        print(f"  {task.task_id:<8} {std_s:<12} {aoi_s:<12} {delta}")

    std_pass = sum(1 for r in results["standard"] if r.success)
    aoi_pass = sum(1 for r in results["aoi_full"] if r.success)
    print(f"\n  Standard: {std_pass}/{len(tasks)}   AOI: {aoi_pass}/{len(tasks)}")


if __name__ == "__main__":
    main()
