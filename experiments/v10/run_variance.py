#!/usr/bin/env python3
"""3-seed variance replication of the Claude Sonnet 4.6 AOI-full headline.

Seed 1 is the existing run in `results/v9_full_100_claude_aoi.json`.
Seeds 2 and 3 are fresh re-runs at the API's default sampling temperature;
because the model output is non-deterministic, two extra runs give us a
realistic variance estimate without requiring API-side seed control.

Each run is resumable: if the output file already has results for some
tasks, they are skipped.

Usage:
    python experiments/v10/run_variance.py --seeds 2 3
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.browser_eval import BrowserEvaluator  # noqa: E402
from dynacubench.tasks_v3 import DynaCUBenchV3, TaskCategory  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("variance")

OUT = PROJECT / "results"


def run_one_seed(seed: int):
    out_file = OUT / f"v10_variance_seed{seed}_claude_aoi.json"
    evaluator = BrowserEvaluator(
        model_name="claude-sonnet-4-6",
        observation_mode="aoi_full",
        max_steps=15,
        step_interval_s=2.0,
    )
    bench = DynaCUBenchV3(html_tasks_dir=PROJECT / "benchmark_env/html_tasks")
    dynamic_tasks = [t for t in bench.tasks if t.category != TaskCategory.S_STATIC]

    results = []
    done_ids = set()
    if out_file.exists():
        try:
            results = json.load(open(out_file))
            done_ids = {r["task_id"] for r in results if r.get("success") is not None}
            log.info("Resume seed %d: %d tasks already done", seed, len(done_ids))
        except Exception:
            results = []

    for i, t in enumerate(dynamic_tasks):
        if t.task_id in done_ids:
            continue
        log.info("[seed=%d] [%d/%d] %s", seed, i + 1, len(dynamic_tasks), t.task_id)
        t0 = time.time()
        try:
            r = evaluator.run_task(t)
            results.append(r.to_dict())
        except Exception as e:
            log.exception("Crashed %s: %s", t.task_id, e)
            results.append({
                "task_id": t.task_id,
                "category": t.category.value,
                "difficulty": t.difficulty.value,
                "model_name": "claude-sonnet-4-6",
                "observation_mode": "aoi_full",
                "success": False,
                "result_val": "exception",
                "steps_taken": 0,
                "total_time_s": time.time() - t0,
                "total_model_latency_ms": 0,
                "total_obs_overhead_ms": 0,
                "steps": [],
                "error": str(e),
            })
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    n_pass = sum(1 for r in results if r.get("success"))
    log.info("DONE seed %d: %d/%d", seed, n_pass, len(results))
    return n_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 3],
                        help="Which seeds to run (1 is the existing v9 run).")
    args = parser.parse_args()

    summary = {}
    for seed in args.seeds:
        summary[seed] = run_one_seed(seed)

    print("=== VARIANCE SUMMARY ===")
    for s, n in summary.items():
        print(f"  seed {s}: {n}/100")


if __name__ == "__main__":
    main()
