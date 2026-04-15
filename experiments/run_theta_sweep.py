"""
CLIP theta sensitivity sweep for Section 7.1.

Runs a subset of visual tasks at different theta values to show
the method is not sensitive to precise threshold tuning.

Uses 30 visual tasks from categories C (video), D (carousel),
E (dashboard), F (transient UI) — 3 difficulty levels each.
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


# Visual tasks (categories C-F) — 40 tasks total
VISUAL_TASK_IDS = [
    "C-E1", "C-E2", "C-E3", "C-M1", "C-M2", "C-M3", "C-M4", "C-H1", "C-H2", "C-H3",
    "D-E1", "D-E2", "D-E3", "D-M1", "D-M2", "D-M3", "D-M4", "D-H1", "D-H2", "D-H3",
    "E-E1", "E-E2", "E-E3", "E-M1", "E-M2", "E-M3", "E-M4", "E-H1", "E-H2", "E-H3",
    "F-E1", "F-E2", "F-E3", "F-M1", "F-M2", "F-M3", "F-M4", "F-H1", "F-H2", "F-H3",
]

THETA_VALUES = [0.02, 0.04, 0.08, 0.12, 0.20, 0.30]


def _cleanup_audio():
    subprocess.run(["pkill", "-f", "pacat.*virtual_speaker"],
                   capture_output=True, timeout=2)
    subprocess.run(["pkill", "-f", "parecord.*virtual_speaker"],
                   capture_output=True, timeout=2)
    time.sleep(0.5)


def run_theta_sweep(
    model_name: str = "claude-sonnet-4-6",
    thetas: list[float] = None,
    task_ids: list[str] = None,
    max_steps: int = 15,
    output_dir: str = "results/theta_sweep",
):
    thetas = thetas or THETA_VALUES
    task_ids = task_ids or VISUAL_TASK_IDS

    bench = DynaCUBenchV3(html_tasks_dir=Path("benchmark_env/html_tasks"))
    tasks = [bench.get_task(tid) for tid in task_ids]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}

    for theta in thetas:
        logger.info("=" * 60)
        logger.info("Running theta=%.3f on %d tasks", theta, len(tasks))
        logger.info("=" * 60)

        results = []
        for task in tasks:
            _cleanup_audio()
            evaluator = BrowserEvaluator(
                model_name=model_name,
                observation_mode="aoi_visual",  # visual only, no audio
                max_steps=max_steps,
                step_interval_s=2.0,
                clip_theta=theta,
            )

            logger.info("═══ theta=%.3f: %s ═══", theta, task.task_id)
            result = evaluator.run_task(task)
            results.append(result)
            status = "PASS" if result.success else "FAIL"
            logger.info("  %s: steps=%d, result=%s, time=%.1fs",
                         status, result.steps_taken, result.result_val, result.total_time_s)

        n_pass = sum(1 for r in results if r.success)
        summary = {
            "theta": theta,
            "passed": n_pass,
            "total": len(results),
            "success_rate": n_pass / len(results),
            "avg_steps": sum(r.steps_taken for r in results) / len(results),
            "avg_time": sum(r.total_time_s for r in results) / len(results),
        }
        all_summaries[str(theta)] = summary
        logger.info("theta=%.3f: %d/%d (%.1f%%)", theta, n_pass, len(results),
                     summary["success_rate"] * 100)

        # Save per-theta results
        out_file = output_dir / f"theta_{theta:.3f}.json"
        with open(out_file, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

    # Save summary
    summary_file = output_dir / "theta_sweep_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Print table
    print(f"\n{'='*60}")
    print(f"CLIP Theta Sweep — {model_name}")
    print(f"{'='*60}")
    print(f"{'Theta':>8} {'Pass':>6} {'Total':>6} {'Rate':>8} {'Avg Steps':>10}")
    print(f"{'-'*40}")
    for theta_str, s in all_summaries.items():
        print(f"{float(theta_str):>8.3f} {s['passed']:>6d} {s['total']:>6d} "
              f"{s['success_rate']*100:>7.1f}% {s['avg_steps']:>10.1f}")
    print(f"{'='*60}")
    print(f"Summary saved to {summary_file}")

    return all_summaries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CLIP theta sensitivity sweep")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--thetas", nargs="+", type=float, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--output-dir", default="results/theta_sweep")
    args = parser.parse_args()

    run_theta_sweep(
        model_name=args.model,
        thetas=args.thetas,
        task_ids=args.tasks,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )
