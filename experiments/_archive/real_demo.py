"""
Real AOI Demonstration — uses actual Claude API for CU model inference.

Runs 3 representative tasks (one per key AOI component) with both:
  - Standard (screenshot only)
  - AOI Full (CLIP keyframes + simulated audio + narration)

Demonstrates the concrete observation difference and whether each
configuration can solve the task.
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from dynacubench import DynaCUBench, TaskCategory
from dynacubench.tasks import Task
from experiments.headless_runner import HeadlessTaskRunner, HeadlessStepResult
from aoi.cu_model import ClaudeCUModel


DEMO_TASKS = ["A-001", "B-001", "D-001"]


def print_step(step: HeadlessStepResult, task_id: str, mode: str):
    print(f"\n  [{task_id}/{mode}] Step {step.step_id}:")
    if step.audio_text:
        print(f"    AUDIO: {step.audio_text[:120]}")
    if step.n_keyframes > 0:
        print(f"    KEYFRAMES: {step.n_keyframes} captured")
    print(f"    ACTION: {step.action[:100]}")
    print(f"    NARRATION: {step.narration[:80]}")


def run_task_demo(task: Task, cu_model, mode: str, max_steps: int = 6):
    print(f"\n{'─'*60}")
    print(f"Task {task.task_id} ({task.category.value}) — Mode: {mode}")
    print(f"Instruction: {task.instruction[:100]}")
    print(f"Ground truth: {task.ground_truth}")
    print(f"{'─'*60}")

    runner = HeadlessTaskRunner(
        cu_model=cu_model,
        observation_mode=mode,
        audio_backend="none",  # Use simulated audio (avoids real Gemini calls for demo)
    )

    step_results, success = runner.run_task(task, max_steps=max_steps)

    for step in step_results:
        print_step(step, task.task_id, mode)
        if success and step == step_results[-1]:
            break

    print(f"\n  Result: {'SUCCESS ✓' if success else 'FAILED ✗'}")
    print(f"  Steps taken: {len(step_results)}")
    print(f"  Total keyframes captured: {sum(s.n_keyframes for s in step_results)}")
    print(f"  Audio activations: {sum(1 for s in step_results if s.audio_text)}")

    return success, step_results


def main():
    benchmark = DynaCUBench()
    demo_tasks = {t.task_id: t for t in benchmark}

    # Use real Claude model
    logger.info("Loading Claude model...")
    cu_model = ClaudeCUModel(model="claude-opus-4-6", max_tokens=512)

    modes = ["standard", "aoi_full"]
    results_summary = []

    print("\n" + "=" * 70)
    print("AOI Live Demonstration — Real Claude + Synthetic Task Stimuli")
    print("=" * 70)

    for task_id in DEMO_TASKS:
        if task_id not in demo_tasks:
            logger.warning("Task %s not found", task_id)
            continue

        task = demo_tasks[task_id]
        task_results = {}

        for mode in modes:
            try:
                success, steps = run_task_demo(task, cu_model, mode, max_steps=8)
                task_results[mode] = {
                    "success": success,
                    "steps": len(steps),
                    "keyframes": sum(s.n_keyframes for s in steps),
                    "audio": sum(1 for s in steps if s.audio_text),
                }
            except Exception as e:
                logger.error("Error on task %s mode %s: %s", task_id, mode, e)
                task_results[mode] = {"success": False, "error": str(e)}

        results_summary.append({"task_id": task_id, **task_results})

    # Print comparison
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"{'Task':<10}{'Standard':>20}{'AOI Full':>20}{'Delta':>15}")
    print("-" * 65)

    for r in results_summary:
        std = "✓" if r.get("standard", {}).get("success") else "✗"
        aoi = "✓" if r.get("aoi_full", {}).get("success") else "✗"
        delta = "→ same" if std == aoi else ("→ AOI fixes it!" if std == "✗" and aoi == "✓" else "→ regression?")
        print(f"{r['task_id']:<10}{std:>20}{aoi:>20}{delta:>15}")

    print("=" * 70)

    # Save results
    out_path = Path("results/real_demo.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
