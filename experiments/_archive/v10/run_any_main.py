#!/usr/bin/env python3
"""Generic 100-task main eval (standard + aoi_full) for an arbitrary model.

Usage:
    python experiments/v10/run_any_main.py --model gemini-3-flash --tag g3
    python experiments/v10/run_any_main.py --model grok-4.3 --tag grok43
    python experiments/v10/run_any_main.py --model grok-4-fast-reasoning --tag grok4fast

Writes results to:
    results/v10_<tag>_standard.json
    results/v10_<tag>_aoi_full.json
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
log = logging.getLogger("any")

OUT = PROJECT / "results"


def run_one(model: str, mode: str, tag: str):
    out_file = OUT / f"v10_{tag}_{mode}.json"
    evaluator = BrowserEvaluator(
        model_name=model,
        observation_mode=mode,
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
            log.info("Resume %s/%s: %d already done", tag, mode, len(done_ids))
        except Exception:
            results = []

    for i, t in enumerate(dynamic_tasks):
        if t.task_id in done_ids:
            continue
        log.info("[%s/%s] [%d/%d] %s", tag, mode, i + 1, len(dynamic_tasks), t.task_id)
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
                "model_name": model,
                "observation_mode": mode,
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
    log.info("DONE %s/%s: %d/%d", tag, mode, n_pass, len(results))
    return n_pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True, help="Short ID used in result filenames")
    p.add_argument("--modes", default="standard,aoi_full",
                   help="Comma-separated modes")
    args = p.parse_args()

    summary = {}
    for mode in args.modes.split(","):
        summary[mode] = run_one(args.model, mode, args.tag)

    print(f"\n=== {args.tag} ({args.model}) ===")
    for mode, n in summary.items():
        print(f"  {mode}: {n}/100")


if __name__ == "__main__":
    main()
