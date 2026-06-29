#!/usr/bin/env python3
"""
Full evaluation: API models × {standard, aoi_full} × 10 easy tasks.
Also supports local models via vLLM and ablation studies.

Usage:
    # API models (standard + AOI):
    python experiments/run_full_eval.py --phase api

    # Local models (after vLLM server is running):
    python experiments/run_full_eval.py --phase local

    # Ablation study (all modes):
    python experiments/run_full_eval.py --phase ablation --model gemini-3-flash
"""
import sys
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.browser_eval import BrowserEvaluator, EvalResult
from dynacubench.tasks_v3 import DynaCUBenchV3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EASY_TASK_IDS = ["A-E1", "B-E1", "C-E1", "D-E1", "E-E1",
                 "F-E1", "G-E1", "H-E1", "I-E1", "J-E1"]

API_MODELS = ["gemini-3-flash", "claude-sonnet-4-6", "gpt-5"]
LOCAL_MODELS = ["fara-7b", "ui-tars-7b"]
CORE_MODES = ["standard", "aoi_full"]
ABLATION_MODES = ["standard", "aoi_visual", "aoi_audio", "aoi_full",
                   "aoi_interactive", "uniform_1fps", "uniform_3fps"]


def run_eval(model_name, mode, task_ids, max_steps=10):
    bench = DynaCUBenchV3(html_tasks_dir=Path("benchmark_env/html_tasks"))
    tasks = [bench.get_task(tid) for tid in task_ids]

    evaluator = BrowserEvaluator(
        model_name=model_name,
        observation_mode=mode,
        max_steps=max_steps,
        step_interval_s=2.0,
    )

    results = []
    for task in tasks:
        logger.info("═══ %s + %s: %s ═══", model_name, mode, task.task_id)
        try:
            result = evaluator.run_task(task)
        except Exception as e:
            logger.error("Task %s failed: %s", task.task_id, e)
            result = EvalResult(
                task_id=task.task_id, category=task.category.value,
                difficulty=task.difficulty.value, model_name=model_name,
                observation_mode=mode, success=False, result_val="exception",
                steps_taken=0, total_time_s=0, total_model_latency_ms=0,
                total_obs_overhead_ms=0, error=str(e),
            )
        results.append(result)
        status = "PASS" if result.success else "FAIL"
        logger.info("  %s: steps=%d, result=%s, time=%.1fs",
                     status, result.steps_taken, result.result_val, result.total_time_s)

    return results


def save_results(results, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    logger.info("Saved %d results to %s", len(results), filename)


def print_summary(all_results):
    groups = defaultdict(list)
    for r in all_results:
        groups[(r.model_name, r.observation_mode)].append(r)

    print(f"\n{'='*80}")
    print("DynaCU-Bench v3 Evaluation Results (Improved AOI Agent v2)")
    print(f"{'='*80}")
    print(f"\n{'Model':<25} {'Mode':<20} {'Pass':>5} {'Total':>6} {'Acc%':>7} {'AvgSteps':>9}")
    print("-" * 72)

    models_seen = []
    for (model, mode), res_list in sorted(groups.items()):
        n_pass = sum(1 for r in res_list if r.success)
        n_total = len(res_list)
        acc = 100 * n_pass / n_total if n_total > 0 else 0
        avg_steps = sum(r.steps_taken for r in res_list) / n_total if n_total else 0
        print(f"{model:<25} {mode:<20} {n_pass:>5} {n_total:>6} {acc:>6.1f}% {avg_steps:>9.1f}")
        if model not in models_seen:
            models_seen.append(model)

    # Per-task breakdown per model
    mode_list = sorted(set(r.observation_mode for r in all_results))

    for model in models_seen:
        print(f"\n--- {model} ---")
        print(f"{'Task':<10}", end="")
        for mode in mode_list:
            print(f"{mode:<18}", end="")
        print()
        task_ids = sorted(set(r.task_id for r in all_results if r.model_name == model))
        for tid in task_ids:
            print(f"{tid:<10}", end="")
            for mode in mode_list:
                res = [r for r in all_results
                       if r.task_id == tid and r.model_name == model and r.observation_mode == mode]
                if res:
                    s = "PASS" if res[0].success else "FAIL"
                    print(f"{s:<18}", end="")
                else:
                    print(f"{'--':<18}", end="")
            print()

    # Delta summary
    print(f"\n{'='*80}")
    print("Improvement: standard -> aoi_full")
    print(f"{'='*80}")
    for model in models_seen:
        std = groups.get((model, "standard"), [])
        aoi = groups.get((model, "aoi_full"), [])
        if std and aoi:
            std_pass = sum(1 for r in std if r.success)
            aoi_pass = sum(1 for r in aoi if r.success)
            delta = aoi_pass - std_pass
            pct = 100 * delta / max(std_pass, 1)
            print(f"  {model}: {std_pass}/{len(std)} -> {aoi_pass}/{len(aoi)} "
                  f"(+{delta} tasks, +{pct:.0f}%)")

    print(f"{'='*80}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="api", choices=["api", "local", "ablation", "all"])
    parser.add_argument("--model", default=None, help="Single model for ablation")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--output-dir", default="results/v2")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    all_results = []

    if args.phase in ("api", "all"):
        for model in API_MODELS:
            for mode in CORE_MODES:
                logger.info("Running %s + %s", model, mode)
                results = run_eval(model, mode, EASY_TASK_IDS, args.max_steps)
                save_results(results, out_dir / f"{model}_{mode}.json")
                all_results.extend(results)

    if args.phase in ("local", "all"):
        for model in LOCAL_MODELS:
            for mode in CORE_MODES:
                logger.info("Running %s + %s", model, mode)
                results = run_eval(model, mode, EASY_TASK_IDS, args.max_steps)
                save_results(results, out_dir / f"{model}_{mode}.json")
                all_results.extend(results)

    if args.phase == "ablation":
        model = args.model or "gemini-3-flash"
        for mode in ABLATION_MODES:
            logger.info("Ablation: %s + %s", model, mode)
            results = run_eval(model, mode, EASY_TASK_IDS, args.max_steps)
            save_results(results, out_dir / f"ablation_{model}_{mode}.json")
            all_results.extend(results)

    if all_results:
        print_summary(all_results)
        save_results(all_results, out_dir / "all_results.json")


if __name__ == "__main__":
    main()
