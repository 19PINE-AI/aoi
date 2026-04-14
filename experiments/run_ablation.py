"""
Main ablation experiment runner.

Evaluates 9 observation configurations across all DynaCU-Bench tasks:
  1. standard         — screenshot only, no audio
  2. uniform_1fps     — 1 FPS uniform frame sampling
  3. uniform_3fps     — 3 FPS uniform sampling
  4. pixel_diff       — pixel-level diff only, no CLIP
  5. aoi_visual_only  — CLIP keyframes, no audio
  6. aoi_visual_asr   — CLIP keyframes + Whisper ASR
  7. aoi_full         — CLIP keyframes + Qwen3/Gemini audio + narration

Outputs:
  results/results_{mode}_{model}.json for each configuration
  results/summary_table.json        — aggregate table
"""

import json
import logging
import sys
import os
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dynacubench import DynaCUBench, TaskEvaluator, TaskCategory
from dynacubench.evaluator import EvaluationResult
from experiments.mock_cu_model import MockStandardCUModel, MockAOICUModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


OBSERVATION_MODES = [
    "standard",
    "uniform_1fps",
    "uniform_3fps",
    "pixel_diff",
    "aoi_visual_only",
    "aoi_visual_asr",
    "aoi_full",
]


def run_full_ablation(
    max_tasks_per_mode: int = None,
    output_dir: Path = Path("results"),
    modes: list = None,
):
    """Run ablation across all modes and save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = DynaCUBench()
    logger.info("Benchmark loaded: %d tasks across %s", len(benchmark), benchmark.summary())

    modes_to_run = modes or OBSERVATION_MODES
    all_summaries = {}

    for mode in modes_to_run:
        logger.info("=" * 60)
        logger.info("Running mode: %s", mode)
        logger.info("=" * 60)

        # Use mode-appropriate mock models
        # (In real experiments, use actual CU models)
        if mode == "standard":
            model = MockStandardCUModel(model_name="mock_standard")
        else:
            model = MockAOICUModel(model_name=f"mock_aoi_{mode}")

        from aoi.agent_loop import AOIConfig
        config = AOIConfig(
            mode=mode,
            post_action_buffer_ms=50,  # Fast for evaluation
            audio_backend="gemini" if mode in ("aoi_visual_asr", "aoi_full") else "none",
        )

        evaluator = TaskEvaluator(
            cu_model=model,
            aoi_config=config,
            observation_mode=mode,
            model_name="mock",
            output_dir=output_dir,
        )

        results = evaluator.evaluate_benchmark(
            benchmark,
            max_tasks=max_tasks_per_mode,
            max_steps_per_task=15,
        )

        filename = f"results_{mode}_mock.json"
        evaluator.save_results(results, filename)

        summary = evaluator.compute_summary(results)
        all_summaries[mode] = summary

        logger.info("Mode %s: success_rate=%.1f%%", mode, summary["success_rate"] * 100)
        for cat, cat_summary in summary.get("by_category", {}).items():
            logger.info(
                "  %s: %.1f%% (%d/%d)",
                cat, cat_summary["success_rate"] * 100,
                cat_summary["success"], cat_summary["total"],
            )

    # Save aggregate summary table
    summary_path = output_dir / "summary_table.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    logger.info("Summary table saved to %s", summary_path)

    return all_summaries


def print_results_table(summaries: dict):
    """Print a formatted results table."""
    from dynacubench.tasks import TaskCategory

    categories = [c.value for c in TaskCategory]
    modes = list(summaries.keys())

    # Header
    print("\n" + "=" * 80)
    print("DynaCU-Bench Results — Success Rate by Observation Mode and Category")
    print("=" * 80)

    col_w = 14
    header = f"{'Mode':<22}" + "".join(f"{cat[:12]:>{col_w}}" for cat in categories) + f"{'OVERALL':>{col_w}}"
    print(header)
    print("-" * len(header))

    for mode in modes:
        s = summaries[mode]
        row = f"{mode:<22}"
        for cat in categories:
            cat_s = s.get("by_category", {}).get(cat, {})
            rate = cat_s.get("success_rate", 0.0)
            row += f"{rate * 100:>{col_w - 1}.1f}%"
        row += f"{s.get('success_rate', 0.0) * 100:>{col_w - 1}.1f}%"
        print(row)

    print("=" * 80)


def print_efficiency_table(summaries: dict):
    """Print efficiency metrics."""
    print("\n" + "=" * 60)
    print("Efficiency Metrics by Mode")
    print("=" * 60)
    print(f"{'Mode':<22}{'Keyframes/Step':>16}{'Audio Ratio':>14}{'Tokens/Task':>14}")
    print("-" * 66)

    for mode, s in summaries.items():
        kf = s.get("avg_keyframes_per_step", 0.0)
        audio = s.get("avg_audio_activation_ratio", 0.0)
        tokens = s.get("avg_tokens_per_task", 0)
        print(f"{mode:<22}{kf:>15.3f} {audio:>13.1%} {tokens:>13.0f}")

    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DynaCU-Bench ablation experiments")
    parser.add_argument("--modes", nargs="+", default=None, help="Observation modes to evaluate")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks per mode (for quick tests)")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    summaries = run_full_ablation(
        max_tasks_per_mode=args.max_tasks,
        output_dir=Path(args.output_dir),
        modes=args.modes,
    )

    print_results_table(summaries)
    print_efficiency_table(summaries)
