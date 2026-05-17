#!/usr/bin/env python3
"""Run standard_structured for a single CU model on the 100 dynamic DynaCU tasks.

Used by Section 7.5 (Static-Task Verification / structured_prompt confound)
to produce results/v10_structured_<model>.json.  The standard_structured
mode supplies the AOI's prompt scaffolding (DOM element list + prior-step
text trajectory) but disables all keyframe and audio extraction, so any
gain over Standard isolates the prompt-format effect.

Usage:
    python experiments/v10/run_structured.py \
        --model claude-sonnet-4-6 \
        --out results/v10_structured_claude.json
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
log = logging.getLogger("structured")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-steps", type=int, default=15)
    args = p.parse_args()

    bench = DynaCUBenchV3(html_tasks_dir=PROJECT / "benchmark_env/html_tasks")
    tasks = [t for t in bench.tasks if t.category != TaskCategory.S_STATIC]
    log.info("Loaded %d dynamic tasks", len(tasks))

    out = Path(args.out)
    existing = []
    if out.exists():
        try:
            existing = json.load(open(out))
        except Exception:
            existing = []
    done_ids = {r["task_id"] for r in existing if r.get("success") is not None}
    log.info("Resume: %d already done", len(done_ids))

    ev = BrowserEvaluator(
        model_name=args.model,
        observation_mode="standard_structured",
        max_steps=args.max_steps,
        step_interval_s=2.0,
    )
    results = list(existing)

    for i, t in enumerate(tasks):
        if t.task_id in done_ids:
            continue
        log.info("[%d/%d] %s/%s", i + 1, len(tasks), args.model, t.task_id)
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
                "observation_mode": "standard_structured",
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
    log.info("DONE: %d/%d", n_pass, len(results))


if __name__ == "__main__":
    main()
