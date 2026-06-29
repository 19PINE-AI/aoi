#!/usr/bin/env python3
"""v10c: Three additional 100-task runs (standard + AOI full).

Models:
  - gemini-3-flash         (Gemini 3 Flash sidebar to Gemini 2.5)
  - grok-4.3               (latest Grok-4 series release)
  - grok-4-fast-reasoning  (lower-latency Grok-4 variant; latency-confound check)

Each model writes:
  results/v10c_<short>_standard.json
  results/v10c_<short>_aoi_full.json

Resumable: existing per-task entries are reused.

Usage:
    XAI_API_KEY=...  python experiments/v10/run_newer_models.py [--models gemini-3-flash grok-4.3 grok-4-fast-reasoning]
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
log = logging.getLogger("v10c")

OUT = PROJECT / "results"

SHORT_NAME = {
    "gemini-3-flash":          "gemini3flash",
    "grok-4.3":                "grok43",
    "grok-4-fast-reasoning":   "grok4fast",
}


def run_one(model_name: str, mode: str, max_steps: int = 15):
    short = SHORT_NAME[model_name]
    out_file = OUT / f"v10c_{short}_{mode}.json"
    evaluator = BrowserEvaluator(
        model_name=model_name,
        observation_mode=mode,
        max_steps=max_steps,
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
            log.info("Resume %s/%s: %d done", model_name, mode, len(done_ids))
        except Exception:
            results = []

    for i, t in enumerate(dynamic_tasks):
        if t.task_id in done_ids:
            continue
        log.info("[%s/%s] [%d/%d] %s",
                 model_name, mode, i + 1, len(dynamic_tasks), t.task_id)
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
                "model_name": model_name,
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
    log.info("DONE %s/%s: %d/%d", model_name, mode, n_pass, len(results))
    return n_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["gemini-3-flash", "grok-4.3", "grok-4-fast-reasoning"])
    parser.add_argument("--max-steps", type=int, default=15)
    args = parser.parse_args()

    summary = {}
    for m in args.models:
        if m not in SHORT_NAME:
            log.error("Unknown model %s (allowed: %s)", m, list(SHORT_NAME))
            continue
        log.info("===== %s standard =====", m)
        n_std = run_one(m, "standard", max_steps=args.max_steps)
        log.info("===== %s aoi_full =====", m)
        n_aoi = run_one(m, "aoi_full",  max_steps=args.max_steps)
        summary[m] = (n_std, n_aoi)

    print("\n=== v10c SUMMARY ===")
    for m, (s, a) in summary.items():
        print(f"  {m:30s} standard={s}/100  aoi_full={a}/100  delta=+{a - s}")


if __name__ == "__main__":
    main()
