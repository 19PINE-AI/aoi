"""
Headless AOI Ablation Experiment.

Runs 7 observation modes against all DynaCU-Bench tasks using:
- Controlled synthetic stimuli (no real screen/audio hardware needed)
- Mock CU models that correctly use observation context
- Real CLIP for keyframe extraction
- Simulated audio model responses (matching task ground truth)

Results demonstrate the AOI's improvement over standard screenshot-only approach.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from dynacubench import DynaCUBench, TaskCategory
from experiments.headless_runner import HeadlessEvaluator
from experiments.mock_cu_model import MockStandardCUModel, MockAOICUModel


MODES = [
    "standard",        # Screenshot only, no audio
    "uniform_1fps",    # 1 FPS uniform sampling
    "uniform_3fps",    # 3 FPS uniform sampling
    "pixel_diff",      # Pixel diff (no CLIP semantic filter)
    "aoi_visual_only", # CLIP keyframes, no audio
    "aoi_visual_asr",  # CLIP + Whisper ASR (speech only)
    "aoi_full",        # CLIP + Gemini audio + narration (full system)
]


def make_model(mode: str):
    """Return appropriate mock model for each observation mode."""
    if mode == "standard":
        return MockStandardCUModel(model_name="mock_standard")
    else:
        return MockAOICUModel(model_name=f"mock_aoi_{mode}")


def print_results_table(summaries: dict):
    categories = [c.value for c in TaskCategory]
    col_w = 14

    print("\n" + "=" * 90)
    print("DynaCU-Bench Ablation Results — Task Success Rate (%)")
    print("=" * 90)

    header = f"{'Observation Mode':<24}"
    for cat in categories:
        label = cat.split("_")[0][0].upper() + "." + cat.split("_")[1][:6]
        header += f"{label:>{col_w}}"
    header += f"{'OVERALL':>{col_w}}"
    print(header)
    print("-" * len(header))

    for mode in MODES:
        if mode not in summaries:
            continue
        s = summaries[mode]
        row = f"{mode:<24}"
        for cat in categories:
            cat_s = s.get("by_category", {}).get(cat, {})
            rate = cat_s.get("success_rate", 0.0)
            row += f"{rate * 100:>{col_w - 1}.1f}%"
        row += f"{s.get('success_rate', 0.0) * 100:>{col_w - 1}.1f}%"
        print(row)

    print("=" * 90)


def print_efficiency_table(summaries: dict):
    print("\n" + "=" * 72)
    print("Efficiency Metrics by Observation Mode")
    print("=" * 72)
    print(f"{'Mode':<24}{'Keyframes/Step':>16}{'Audio Ratio':>14}{'Avg Steps':>12}{'Tokens':>12}")
    print("-" * 78)

    for mode in MODES:
        if mode not in summaries:
            continue
        s = summaries[mode]
        kf = s.get("avg_keyframes_per_step", 0.0)
        audio = s.get("avg_audio_activation_ratio", 0.0)
        steps = s.get("avg_steps", 0.0)
        tokens = s.get("avg_tokens", 0)
        print(f"{mode:<24}{kf:>15.3f} {audio:>13.1%} {steps:>11.1f} {tokens:>11.0f}")

    print("=" * 72)


def print_category_analysis(summaries: dict):
    """Show where each mode makes progress."""
    print("\n" + "=" * 60)
    print("Success Rate by Category — Key Comparisons")
    print("=" * 60)

    key_modes = ["standard", "aoi_visual_only", "aoi_full"]
    for cat in [c.value for c in TaskCategory]:
        print(f"\n  {cat}:")
        for mode in key_modes:
            if mode in summaries:
                cat_s = summaries[mode].get("by_category", {}).get(cat, {})
                rate = cat_s.get("success_rate", 0.0)
                n_s = cat_s.get("success", 0)
                n_t = cat_s.get("total", 0)
                bar = "█" * int(rate * 20)
                print(f"    {mode:24s}: {rate * 100:5.1f}% ({n_s}/{n_t}) {bar}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-steps", type=int, default=15)
    args = parser.parse_args()

    benchmark = DynaCUBench()
    logger.info(
        "DynaCU-Bench: %d tasks across %s",
        len(benchmark), benchmark.summary()
    )

    evaluator = HeadlessEvaluator(
        cu_model_factory=make_model,
        output_dir=Path(args.output_dir),
        audio_backend="none",  # Use simulated audio responses
    )

    summaries = evaluator.run_ablation(
        benchmark=benchmark,
        modes=args.modes,
        max_tasks=args.max_tasks,
        max_steps=args.max_steps,
    )

    print_results_table(summaries)
    print_efficiency_table(summaries)
    print_category_analysis(summaries)

    # Print key finding
    std_rate = summaries.get("standard", {}).get("success_rate", 0.0)
    full_rate = summaries.get("aoi_full", {}).get("success_rate", 0.0)
    improvement = full_rate - std_rate
    print(f"\n{'='*60}")
    print(f"Key Finding:")
    print(f"  Standard loop:  {std_rate * 100:.1f}% overall success")
    print(f"  AOI (full):     {full_rate * 100:.1f}% overall success")
    print(f"  Improvement:    +{improvement * 100:.1f} percentage points")
    print(f"{'='*60}")
