"""
A1: Keyframe causal probe on Gemini 3 Flash.

Modes (all with audio + scaffold; vary the keyframe channel only):
  aoi_full_max1kf      — cap at 1 keyframe per step
  aoi_full_max2kf      — cap at 2 keyframes per step
  aoi_full_max3kf      — cap at 3 keyframes per step
  aoi_full_max5kf      — cap at 5 keyframes per step (= aoi_full, used as anchor)
  aoi_full_noise_kf    — same image count as ran-cap-5 but each keyframe replaced
                         by random-noise image of identical size (controls for
                         token-budget dilution)
  aoi_full_dup_kf      — keyframes replaced with copies of the post-action
                         screenshot (controls for novel-content vs. just-more-images)
  aoi_full_reorder_kf  — keyframes positioned AFTER the post-action screenshot in
                         the image list (recency manipulation)

Subset: 50 tasks from B, C, D, E, F (10 each) where keyframes are actively produced.
The audio-only categories (A, G, H) produce no keyframes so adding capacity is a
no-op; we exclude those to keep the per-mode signal sharp.  J is excluded because
it's already known to be latency-bound for some models.

Output: results/extensions/a1_g3_<mode>.json (resumable)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path("/home/ubuntu/adaptive-observation-paper")
sys.path.insert(0, str(ROOT))

from experiments.browser_eval import BrowserEvaluator  # noqa: E402
from dynacubench.tasks_v3 import DynaCUBenchV3, TaskCategory  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("a1")

OUT = ROOT / "results/extensions"
OUT.mkdir(parents=True, exist_ok=True)

# Tasks subset: 5 visual-active categories × 10 tasks each = 50
SUBSET_CATS = {
    TaskCategory.B_MEETING,
    TaskCategory.C_VIDEO,
    TaskCategory.D_CAROUSEL,
    TaskCategory.E_DASHBOARD,
    TaskCategory.F_TRANSIENT,
}


def run_mode(model: str, mode: str, tag: str, max_tasks: int | None = None):
    out_file = OUT / f"a1_{tag}_{mode}.json"
    evaluator = BrowserEvaluator(
        model_name=model,
        observation_mode=mode,
        max_steps=15,
        step_interval_s=2.0,
    )
    bench = DynaCUBenchV3(html_tasks_dir=ROOT / "benchmark_env/html_tasks")
    tasks = [t for t in bench.tasks if t.category in SUBSET_CATS]
    if max_tasks:
        tasks = tasks[:max_tasks]

    results = []
    done = set()
    if out_file.exists():
        try:
            results = json.load(open(out_file))
            done = {r["task_id"] for r in results if r.get("success") is not None}
            log.info("Resume %s/%s: %d already done", tag, mode, len(done))
        except Exception:
            results = []

    for i, t in enumerate(tasks):
        if t.task_id in done:
            continue
        log.info("[%s/%s] [%d/%d] %s", tag, mode, i + 1, len(tasks), t.task_id)
        t0 = time.time()
        try:
            r = evaluator.run_task(t)
            results.append(r.to_dict())
        except Exception as e:
            log.exception("Crashed %s: %s", t.task_id, e)
            results.append({
                "task_id": t.task_id, "category": t.category.value,
                "difficulty": t.difficulty.value, "model_name": model,
                "observation_mode": mode, "success": False,
                "result_val": "exception", "steps_taken": 0,
                "total_time_s": time.time() - t0,
                "total_model_latency_ms": 0, "total_obs_overhead_ms": 0,
                "steps": [], "error": str(e),
            })
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    n_pass = sum(1 for r in results if r.get("success"))
    log.info("DONE %s/%s: %d/%d", tag, mode, n_pass, len(results))
    return n_pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemini-3-flash")
    p.add_argument("--tag", default="g3")
    p.add_argument("--modes", default=("aoi_full_max1kf,aoi_full_max2kf,aoi_full_max3kf,"
                                       "aoi_full_max5kf,aoi_full_noise_kf,"
                                       "aoi_full_dup_kf,aoi_full_reorder_kf"))
    p.add_argument("--max-tasks", type=int, default=None)
    args = p.parse_args()

    summary = {}
    for mode in args.modes.split(","):
        summary[mode] = run_mode(args.model, mode, args.tag, args.max_tasks)

    print(f"\n=== A1 keyframe probe ({args.tag}, {args.model}) ===")
    for mode, n in summary.items():
        print(f"  {mode}: {n}/50")
