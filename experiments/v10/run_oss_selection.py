#!/usr/bin/env python3
"""Selection-method ablation on an open-source vision model (Section 6.5).

Runs a single observation_mode (uniform_1fps / pixel_diff / random_keyframes)
on the 50 visual-temporal tasks (categories C, D, E, F, J) where keyframe
selection actually differs.  Used to produce
results/v10_oss_qwen3vl32b_<mode>.json.

This is a thin driver around BrowserEvaluator parameterised so the same
script can produce all three rows of Table 6 by re-invoking with
different --mode / --out values.

Usage:
    python experiments/v10/run_oss_selection.py \
        --model qwen3-vl-32b --mode uniform_1fps \
        --out results/v10_oss_qwen3vl32b_uniform_1fps.json
    python experiments/v10/run_oss_selection.py \
        --model qwen3-vl-32b --mode pixel_diff \
        --out results/v10_oss_qwen3vl32b_pixel_diff.json
    python experiments/v10/run_oss_selection.py \
        --model qwen3-vl-32b --mode random_keyframes \
        --out results/v10_oss_qwen3vl32b_random_keyframes.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.browser_eval import BrowserEvaluator  # noqa: E402
from dynacubench.tasks_v3 import DynaCUBenchV3, TaskCategory  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("oss-sel")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-vl-32b")
    p.add_argument("--mode", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--categories",
        default="C,D,E,F,J",
        help="comma-separated category prefixes (default: visual-temporal cats)",
    )
    p.add_argument("--max-steps", type=int, default=12)
    args = p.parse_args()

    bench = DynaCUBenchV3(html_tasks_dir=PROJECT / "benchmark_env/html_tasks")
    cat_prefixes = tuple(args.categories.split(","))
    tasks = [
        t for t in bench.tasks
        if t.category != TaskCategory.S_STATIC
        and t.task_id.split("-")[0] in cat_prefixes
    ]
    log.info("Visual subset: %d tasks across categories %s", len(tasks), cat_prefixes)

    out = Path(args.out)
    existing = []
    if out.exists():
        try:
            existing = json.load(open(out))
        except Exception:
            existing = []
    done_ids = {r["task_id"] for r in existing if r.get("success") is not None}

    ev = BrowserEvaluator(
        model_name=args.model,
        observation_mode=args.mode,
        max_steps=args.max_steps,
        step_interval_s=2.0,
    )
    results = list(existing)

    for i, t in enumerate(tasks):
        if t.task_id in done_ids:
            continue
        log.info("[%d/%d] %s/%s/%s", i + 1, len(tasks), args.model, args.mode, t.task_id)
        try:
            r = ev.run_task(t)
            results.append(r.to_dict())
        except Exception as e:
            log.exception("Task %s crashed", t.task_id)
            results.append({
                "task_id": t.task_id,
                "category": t.category.value,
                "difficulty": t.difficulty.value,
                "model_name": args.model,
                "observation_mode": args.mode,
                "success": False,
                "result_val": "exception",
                "steps_taken": 0,
                "total_time_s": 0,
                "total_model_latency_ms": 0,
                "total_obs_overhead_ms": 0,
                "steps": [],
                "error": str(e),
            })
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    n_pass = sum(1 for r in results if r.get("success"))
    log.info("DONE %s: %d/%d", args.mode, n_pass, len(results))


if __name__ == "__main__":
    main()
