"""
A2: Decompose the standard_structured prompt-format effect into sub-components.

The current `standard_structured` mode differs from `standard` by two things in
the observation pipeline:
  (1) The presence of [PAGE ELEMENTS] (DOM element id list) appended to the
      prompt; this is conditional on the mode.
  (2) Optional trajectory wrapping ([CONTEXT — prior steps] block) when prior
      steps had narrations or audio — but in `standard` mode the narrations
      are still generated and the trajectory wrapping still runs.

To cleanly isolate the contribution of each sub-component, we run three modes:

  standard_minimal     — raw prompt: only the task instruction + screenshot.
                         No [CONTEXT]/[NEW] structure, no trajectory wrapping,
                         no [PAGE ELEMENTS].  Lower bound.
  standard_traj_only   — structured trajectory wrapping (= existing `standard`)
                         but explicitly NO [PAGE ELEMENTS].
  standard_pageel_only — minimal prompt + [PAGE ELEMENTS] only.  Isolates the
                         DOM-list effect from the structure effect.
  standard_structured  — already on file: structured trajectory + [PAGE ELEMENTS]
                         (= existing reference).

Models: Claude Sonnet 4.6 (perception-leaning per paper).  Gemini 2.5 Flash if
time permits.

Each on the 100-task DynaCU-Bench dynamic suite.  Resumable.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.browser_eval import BrowserEvaluator  # noqa: E402
from dynacubench.tasks_v3 import DynaCUBenchV3, TaskCategory  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("a2")

OUT = ROOT / "results/extensions"
OUT.mkdir(parents=True, exist_ok=True)


def run_mode(model: str, mode: str, tag: str, max_tasks: int | None = None,
             cats: set | None = None):
    out_file = OUT / f"a2_{tag}_{mode}.json"
    evaluator = BrowserEvaluator(
        model_name=model, observation_mode=mode,
        max_steps=15, step_interval_s=2.0,
    )
    bench = DynaCUBenchV3(html_tasks_dir=ROOT / "benchmark_env/html_tasks")
    tasks = [t for t in bench.tasks if t.category != TaskCategory.S_STATIC]
    if cats:
        tasks = [t for t in tasks if t.category in cats]
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
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--tag", default="claude")
    p.add_argument("--modes",
                   default="standard_minimal,standard_pageel_only")
    p.add_argument("--max-tasks", type=int, default=None)
    args = p.parse_args()

    summary = {}
    for mode in args.modes.split(","):
        summary[mode] = run_mode(args.model, mode, args.tag, args.max_tasks)

    print(f"\n=== A2 ({args.tag}, {args.model}) ===")
    for mode, n in summary.items():
        print(f"  {mode}: {n}")
