"""
B1: Replication of standard vs AOI-full on additional open-source CU models.

Adds two open-source models from the paper's prior list (EvoCUA-32B, Fara-7B):
  - UI-TARS-1.5-7B (ByteDance, the most-cited open-source CU baseline)
  - GLM-4.5V (Zhipu, another open-source vision-language model)
  - Qwen3-VL-32B (Alibaba; complete from 50 to 100 tasks)

All routed through OpenRouter to avoid the local vLLM NVML mismatch blocker.

Output: results/extensions/b1_<model>_<mode>.json (resumable)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path("/home/ubuntu/adaptive-observation-paper")
sys.path.insert(0, str(ROOT))

# Ensure VLLM_API_KEY is set so OpenAI-compatible client passes the OR key
if "OPENROUTER_API_KEY" in os.environ and "VLLM_API_KEY" not in os.environ:
    os.environ["VLLM_API_KEY"] = os.environ["OPENROUTER_API_KEY"]

from experiments.browser_eval import BrowserEvaluator  # noqa: E402
from dynacubench.tasks_v3 import DynaCUBenchV3, TaskCategory  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("b1")

OUT = ROOT / "results/extensions"
OUT.mkdir(parents=True, exist_ok=True)


def run_one(model: str, mode: str, tag: str, max_tasks: int | None = None):
    out_file = OUT / f"b1_{tag}_{mode}.json"
    evaluator = BrowserEvaluator(
        model_name=model, observation_mode=mode,
        max_steps=15, step_interval_s=2.0,
    )
    bench = DynaCUBenchV3(html_tasks_dir=ROOT / "benchmark_env/html_tasks")
    tasks = [t for t in bench.tasks if t.category != TaskCategory.S_STATIC]
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
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--modes", default="standard,aoi_full")
    p.add_argument("--max-tasks", type=int, default=None)
    args = p.parse_args()

    summary = {}
    for mode in args.modes.split(","):
        summary[mode] = run_one(args.model, mode, args.tag, args.max_tasks)

    print(f"\n=== B1 ({args.tag}, {args.model}) ===")
    for mode, n in summary.items():
        print(f"  {mode}: {n}")
