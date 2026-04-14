"""Run 10-task evaluation: one easy task per category, standard vs aoi_full."""
import sys
import json
import logging
import time
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

def run_eval(model_name: str, mode: str, task_ids: list[str], max_steps: int = 10):
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
        result = evaluator.run_task(task)
        results.append(result)
        status = "PASS" if result.success else "FAIL"
        logger.info("  %s: steps=%d, result=%s, time=%.1fs",
                     status, result.steps_taken, result.result_val, result.total_time_s)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--mode", default="standard")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    task_ids = ["A-E1", "B-E1", "C-E1", "D-E1", "E-E1",
                "F-E1", "G-E1", "H-E1", "I-E1", "J-E1"]

    results = run_eval(args.model, args.mode, task_ids, args.max_steps)

    # Summary
    n_pass = sum(1 for r in results if r.success)
    print(f"\n{'='*60}")
    print(f"Model: {args.model} | Mode: {args.mode}")
    print(f"Passed: {n_pass}/{len(results)}")
    print(f"{'='*60}")
    for r in results:
        s = "PASS" if r.success else "FAIL"
        print(f"  {r.task_id:8s} {r.category:15s} {s:4s} | steps={r.steps_taken:2d} | val={r.result_val}")
    print(f"{'='*60}")

    # Save
    out_file = args.output or f"results/{args.model}_{args.mode}_10tasks.json"
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
