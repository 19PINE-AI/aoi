"""
Run DynaCU-Bench evaluation: 10 easy tasks (one per category).

Supports running all 10 tasks or a filtered subset. Each task runs with
a fresh evaluator state to prevent cross-task interference.

Prerequisites:
  - Whisper service running: python -m aoi.whisper_service --model large-v3
  - PulseAudio virtual devices configured
  - API keys set: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
"""
import sys
import json
import logging
import subprocess
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


def _cleanup_audio():
    """Thorough audio cleanup between tasks to prevent leaking."""
    # Kill any pacat processes injecting audio
    subprocess.run(["pkill", "-f", "pacat.*virtual_speaker"],
                   capture_output=True, timeout=2)
    # Kill any parecord capture processes
    subprocess.run(["pkill", "-f", "parecord.*virtual_speaker"],
                   capture_output=True, timeout=2)
    # Brief pause for PulseAudio to drain
    time.sleep(0.5)


def run_eval(model_name: str, mode: str, task_ids: list[str], max_steps: int = 15,
             output_file: str = None):
    bench = DynaCUBenchV3(html_tasks_dir=Path("benchmark_env/html_tasks"))

    # Resume support: load existing results and skip completed tasks
    existing_results = []
    completed_ids = set()
    if output_file and Path(output_file).exists():
        try:
            with open(output_file) as f:
                existing_data = json.load(f)
            existing_results = existing_data
            completed_ids = {r["task_id"] for r in existing_data}
            logger.info("Resuming: %d tasks already completed, skipping them", len(completed_ids))
        except (json.JSONDecodeError, KeyError):
            logger.warning("Could not parse existing output file, starting fresh")

    remaining_ids = [tid for tid in task_ids if tid not in completed_ids]
    if not remaining_ids:
        logger.info("All %d tasks already completed!", len(task_ids))
        return [EvalResult.from_dict(r) for r in existing_results]

    logger.info("Running %d remaining tasks (of %d total)", len(remaining_ids), len(task_ids))
    tasks = [bench.get_task(tid) for tid in remaining_ids]

    results = []
    for task in tasks:
        # Create a FRESH evaluator per task to prevent any state leaking
        _cleanup_audio()
        evaluator = BrowserEvaluator(
            model_name=model_name,
            observation_mode=mode,
            max_steps=max_steps,
            step_interval_s=2.0,
        )

        logger.info("═══ %s + %s: %s ═══", model_name, mode, task.task_id)
        result = evaluator.run_task(task)
        results.append(result)
        status = "PASS" if result.success else "FAIL"
        logger.info("  %s: steps=%d, result=%s, time=%.1fs",
                     status, result.steps_taken, result.result_val, result.total_time_s)

        # Incremental save after each task
        if output_file:
            all_data = existing_results + [r.to_dict() for r in results]
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(all_data, f, indent=2)

    # Combine existing + new results
    all_results = [EvalResult.from_dict(r) for r in existing_results] + results
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DynaCU-Bench 10-task evaluation")
    parser.add_argument("--model", default="gemini-2.0-flash",
                        help="CU model to evaluate")
    parser.add_argument("--mode", default="standard",
                        help="Observation mode (standard, aoi_full, aoi_visual, aoi_audio)")
    parser.add_argument("--max-steps", type=int, default=15,
                        help="Maximum steps per task")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Specific task IDs to run (e.g. A-E1 G-E1)")
    parser.add_argument("--output", default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    if args.tasks:
        task_ids = args.tasks
    else:
        task_ids = ["A-E1", "B-E1", "C-E1", "D-E1", "E-E1",
                    "F-E1", "G-E1", "H-E1", "I-E1", "J-E1"]

    # Check Whisper service is running for audio modes
    if args.mode in ("aoi_full", "aoi_audio", "aoi_interactive", "aoi_visual_asr"):
        import requests
        try:
            r = requests.get("http://localhost:8786/health", timeout=2)
            if r.status_code != 200:
                print("ERROR: Whisper service not healthy. Start it with:")
                print("  python -m aoi.whisper_service --model large-v3")
                sys.exit(1)
            logger.info("Whisper service: %s", r.json())
        except Exception:
            print("ERROR: Whisper service not running. Start it with:")
            print("  python -m aoi.whisper_service --model large-v3")
            sys.exit(1)

    out_file = args.output or f"results/{args.model}_{args.mode}_10tasks.json"
    results = run_eval(args.model, args.mode, task_ids, args.max_steps, output_file=out_file)

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
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
