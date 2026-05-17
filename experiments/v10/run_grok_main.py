#!/usr/bin/env python3
"""Grok-4 100-task main eval (standard + aoi_full).

Requires XAI_API_KEY in environment.

Usage:
    XAI_API_KEY=... python experiments/v10/run_grok_main.py
"""
import json
import logging
import os
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
log = logging.getLogger("grok")

OUT = PROJECT / "results"


def run_one(mode: str):
    if not os.environ.get("XAI_API_KEY") and not os.environ.get("GROK_API_KEY"):
        raise RuntimeError("Set XAI_API_KEY before launching the Grok eval")

    out_file = OUT / f"v10_grok4_{mode}.json"
    evaluator = BrowserEvaluator(
        model_name="grok-4",
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
            log.info("Resume %s: %d tasks already done", mode, len(done_ids))
        except Exception:
            results = []

    for i, t in enumerate(dynamic_tasks):
        if t.task_id in done_ids:
            continue
        log.info("[%s] [%d/%d] %s", mode, i + 1, len(dynamic_tasks), t.task_id)
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
                "model_name": "grok-4",
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
    log.info("DONE grok-4 %s: %d/%d", mode, n_pass, len(results))
    return n_pass


if __name__ == "__main__":
    n_std = run_one("standard")
    n_aoi = run_one("aoi_full")
    print(f"\n=== GROK-4 SUMMARY ===")
    print(f"  standard: {n_std}/100")
    print(f"  aoi_full: {n_aoi}/100")
