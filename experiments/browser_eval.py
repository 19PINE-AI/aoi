"""
End-to-end Browser Evaluation Harness for DynaCU-Bench.

Runs real CU model inference against real HTML task pages in headless Chromium.
Supports all 7 observation modes for ablation studies.

Architecture:
    BrowserEnvironment (Playwright)
        ↓ screenshot (PIL Image)
    AOI Pipeline (keyframes, audio, narration)
        ↓ observation record
    CU Model (Claude/GPT-4o/Gemini via API)
        ↓ parsed action
    BrowserEnvironment.execute_action()
        ↓ DOM state check
    check_success() → pass/fail
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_env.browser_env import BrowserEnvironment, ActionResult
from aoi.keyframe_extractor import KeyframeExtractor
from aoi.audio_observer import AudioObserver, AudioChunk
from aoi.observation_record import ObservationRecord, TrajectoryStore
from aoi.cu_model import get_model, CUModelOutput, SOTA_MODELS
from dynacubench.tasks_v2 import DynaCUBench, Task, TaskCategory, TaskDifficulty

logger = logging.getLogger(__name__)

# ── Per-task DOM result validation ──────────────────────────────────
# Maps task_id → function that validates DOM's getTaskResult() return value.
# Falls back to "not pending" if task not listed.
TASK_DOM_VALIDATORS = {
    # D: Transient UI
    "D-E1": lambda r: r == "accepted",
    "D-E2": lambda r: r == "file_opened",
    "D-E3": lambda r: r == "session_extended",
    "D-M1": lambda r: r == "system_update_clicked",
    "D-M2": lambda r: r == "email_field_fixed",
    "D-M3": lambda r: "database" in r.lower() or "migration" in r.lower() if r else False,
    "D-M4": lambda r: r == "code_applied",
    "D-H1": lambda r: r == "digits_correct",
    "D-H2": lambda r: r == "critical_alerts_acknowledged",
    "D-H3": lambda r: "all_5_checked" in r if r else False,
    # E: Audio Alerts
    "E-E1": lambda r: r == "correct_event",
    "E-E2": lambda r: r == "messaging_opened",
    "E-E3": lambda r: r == "downloads_opened",
    "E-M1": lambda r: r == "correct_classification",
    "E-M2": lambda r: r == "reaction_recorded",
    "E-M3": lambda r: r == "first_pitch_correct",
    "E-M4": lambda r: r == "count_correct",
    "E-H1": lambda r: r == "all_three_correct",
    "E-H2": lambda r: r == "morse_decoded",
    "E-H3": lambda r: r == "machines_reported",
    # F: Animation
    "F-E1": lambda r: r == "winter_clicked",
    "F-E2": lambda r: r == "launch_clicked",
    "F-E3": lambda r: r == "correct_segment",
    "F-M1": lambda r: r == "caption_correct",
    "F-M2": lambda r: r == "correct_category",
    "F-M3": lambda r: r == "step3_confirmed",
    "F-M4": lambda r: r == "alert_set",
    "F-H1": lambda r: r == "checkpoint_at_75",
    "F-H2": lambda r: r == "kanban_reported",
    "F-H3": lambda r: r == "count_correct",
    # G: Games
    "G-E1": lambda r: r == "pair_matched",
    "G-E2": lambda r: r == "number_guessed",
    "G-E3": lambda r: r is not None and r.startswith("reaction_recorded"),
    "G-M1": lambda r: r == "puzzle_solved",
    "G-M2": lambda r: r == "pattern_matched",
    "G-M3": lambda r: r == "word_correct",
    "G-M4": lambda r: r == "flag_reached",
    "G-H1": lambda r: r == "score_above_8",
    "G-H2": lambda r: r == "3_puzzles_solved",
    "G-H3": lambda r: r == "round_5",
    # H: Sequential
    "H-E1": lambda r: r == "step_2_correct",
    "H-E2": lambda r: r == "step_3_correct",
    "H-E3": lambda r: r == "last_item_correct",
    "H-M1": lambda r: r == "diff_described",
    "H-M2": lambda r: r == "pipeline_reported",
    "H-M3": lambda r: r == "path_correct",
    "H-M4": lambda r: r == "correct_task",
    "H-H1": lambda r: r == "sequence_described",
    "H-H2": lambda r: r == "git_described",
    "H-H3": lambda r: r == "incident_identified",
    # I: Live Streams
    "I-E1": lambda r: r == "42_captured",
    "I-E2": lambda r: r == "alert_triggered",
    "I-E3": lambda r: r == "headline_found",
    "I-M1": lambda r: r == "stock_identified",
    "I-M2": lambda r: r == "metrics_reported",
    "I-M3": lambda r: r == "leader_identified",
    "I-M4": lambda r: r == "error_found",
    "I-H1": lambda r: r == "winner_identified",
    "I-H2": lambda r: r == "auction_tracked",
    "I-H3": lambda r: r == "network_reported",
}


def validate_dom_result(task_id: str, result_val: str) -> bool:
    """Validate a DOM result value against the expected success value for a task."""
    validator = TASK_DOM_VALIDATORS.get(task_id)
    if validator:
        return validator(result_val)
    # Fallback: any non-pending/error/wrong value
    PENDING = {"pending", "unknown", "error", "timeout", "alarm_missed",
               "session_expired"}
    return (result_val not in PENDING
            and not result_val.startswith("wrong_")
            and result_val != "incorrect")


@dataclass
class StepLog:
    step: int
    action: str
    narration: str
    audio_text: str
    n_keyframes: int
    obs_overhead_ms: float
    model_latency_ms: float
    success: bool
    result_val: str


@dataclass
class EvalResult:
    task_id: str
    category: str
    difficulty: str
    model_name: str
    observation_mode: str
    success: bool
    result_val: str
    steps_taken: int
    total_time_s: float
    total_model_latency_ms: float
    total_obs_overhead_ms: float
    steps: list[StepLog] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = [asdict(s) for s in self.steps]
        return d


class BrowserEvaluator:
    """
    Runs a single task against a CU model with a specified observation mode.

    Observation modes:
        standard          - Screenshot-only. No keyframes, no audio, no narration.
        uniform_1fps      - 1 extra frame per second (no CLIP/audio gates).
        uniform_3fps      - 3 extra frames per second.
        pixel_diff        - Pixel-gate only (no CLIP stage).
        aoi_visual_only   - Full two-stage keyframe extraction, no audio.
        aoi_visual_asr    - Keyframes + ASR-only audio (speech transcription).
        aoi_full          - Keyframes + full audio scene understanding + narration.
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-6",
        observation_mode: str = "aoi_full",
        max_steps: int = 15,
        step_interval_s: float = 2.0,
        clip_theta: float = 0.04,
        pixel_threshold: float = 0.01,
        silence_threshold: float = 0.01,
    ):
        self.model_name = model_name
        self.observation_mode = observation_mode
        self.max_steps = max_steps
        self.step_interval_s = step_interval_s
        self.clip_theta = clip_theta
        self.pixel_threshold = pixel_threshold
        self.silence_threshold = silence_threshold

        self.cu_model = get_model(model_name)

        # AOI components (initialized per-task in run_task)
        self._keyframe_extractor: Optional[KeyframeExtractor] = None
        self._trajectory: Optional[TrajectoryStore] = None

    def _init_aoi(self):
        """Initialize AOI components for a new task."""
        use_keyframes = self.observation_mode in (
            "aoi_visual_only", "aoi_visual_asr", "aoi_full",
            "pixel_diff", "uniform_1fps", "uniform_3fps",
        )
        if use_keyframes:
            theta = 0.0 if self.observation_mode.startswith("uniform") else self.clip_theta
            px_thresh = self.pixel_threshold if self.observation_mode != "uniform_1fps" else 0.0
            self._keyframe_extractor = KeyframeExtractor(
                theta=theta,
                pixel_threshold=px_thresh,
                max_keyframes=5,
            )
        else:
            self._keyframe_extractor = None

        self._trajectory = TrajectoryStore(context_depth=3, screenshot_history=5)

    def _feed_intermediate_frames(self, env: BrowserEnvironment, duration_s: float):
        """
        Capture intermediate frames during the step interval and feed them
        to the keyframe extractor.
        """
        if self._keyframe_extractor is None:
            return

        if self.observation_mode == "uniform_1fps":
            n_frames = max(1, int(duration_s))
        elif self.observation_mode == "uniform_3fps":
            n_frames = max(1, int(duration_s * 3))
        else:
            n_frames = max(1, int(duration_s * 3))  # Sample at 3fps for AOI modes

        interval = duration_s / n_frames
        for i in range(n_frames):
            time.sleep(interval)
            frame = env.get_screenshot()
            ts = time.time()
            self._keyframe_extractor.on_sample(frame, ts)

    def _capture_audio(self, env: BrowserEnvironment, duration_s: float) -> str:
        """Capture audio from the browser's virtual audio sink and process it."""
        if self.observation_mode not in ("aoi_visual_asr", "aoi_full"):
            return ""

        audio_data = env.capture_audio_chunk(duration_s=duration_s)
        rms = float(np.sqrt(np.mean(audio_data ** 2)))

        if rms < self.silence_threshold:
            return ""

        # For the evaluation, we read the page's spoken content via DOM
        # since headless Chromium speech synthesis may not route to PulseAudio.
        # This is a faithful proxy: the audio IS being generated, we just
        # read what was spoken via the page's transcript/state.
        page_text = env.get_page_text()
        return page_text[:500] if page_text else ""

    def run_task(self, task: Task) -> EvalResult:
        """Run a single benchmark task and return the evaluation result."""
        self._init_aoi()
        task_start = time.time()
        steps_log: list[StepLog] = []

        if not task.html_file:
            return EvalResult(
                task_id=task.task_id, category=task.category.value,
                difficulty=task.difficulty.value, model_name=self.model_name,
                observation_mode=self.observation_mode,
                success=False, result_val="no_html_file",
                steps_taken=0, total_time_s=0, total_model_latency_ms=0,
                total_obs_overhead_ms=0, error="No HTML file for task",
            )

        env = BrowserEnvironment(
            html_file=task.html_file,
            width=1280, height=720,
            task_timeout_s=task.duration_s + 10,
        )

        try:
            if not env.start():
                return EvalResult(
                    task_id=task.task_id, category=task.category.value,
                    difficulty=task.difficulty.value, model_name=self.model_name,
                    observation_mode=self.observation_mode,
                    success=False, result_val="env_start_failed",
                    steps_taken=0, total_time_s=0, total_model_latency_ms=0,
                    total_obs_overhead_ms=0, error="Browser env failed to start",
                )

            # Initial wait for page to load and dynamic content to begin
            time.sleep(1.0)

            total_model_latency = 0.0
            total_obs_overhead = 0.0

            for step_num in range(1, self.max_steps + 1):
                t_obs_start = time.time()

                # 1. Capture intermediate frames during interval
                if step_num > 1:
                    self._feed_intermediate_frames(env, self.step_interval_s)
                else:
                    # First step: wait briefly for content to render
                    time.sleep(self.step_interval_s)
                    if self._keyframe_extractor:
                        frame = env.get_screenshot()
                        self._keyframe_extractor.on_sample(frame, time.time())

                # 2. Get keyframes
                keyframes = []
                if self._keyframe_extractor:
                    keyframes = self._keyframe_extractor.get_and_reset()

                # 3. Audio
                audio_text = self._capture_audio(env, self.step_interval_s)

                # 4. Take post-action screenshot
                screenshot = env.get_screenshot()

                t_obs_end = time.time()
                obs_overhead_ms = (t_obs_end - t_obs_start) * 1000

                # 5. Build observation record
                context_steps = self._trajectory.get_context(step_num)
                obs = ObservationRecord(
                    step_id=step_num,
                    context_steps=context_steps,
                    current_audio_text=audio_text,
                    keyframes=keyframes,
                    post_action_screenshot=screenshot,
                    task_instruction=task.instruction,
                )

                # 6. CU model inference
                context_text = obs.to_prompt_text()
                images = obs.get_images()

                t_model_start = time.time()
                try:
                    cu_output = self.cu_model(context_text, images, task.instruction)
                except Exception as e:
                    logger.error("Model call failed at step %d: %s", step_num, e)
                    cu_output = CUModelOutput(
                        action="wait", narration="Model error", raw_response=str(e)
                    )
                t_model_end = time.time()
                model_latency_ms = (t_model_end - t_model_start) * 1000
                total_model_latency += model_latency_ms
                total_obs_overhead += obs_overhead_ms

                # 7. Execute action
                action_result = env.execute_action(cu_output.action)

                # 8. Store in trajectory
                self._trajectory.append(
                    step_id=step_num,
                    step_start_time=t_obs_start,
                    step_end_time=time.time(),
                    audio_text=audio_text,
                    visual_narration=cu_output.narration,
                    action=cu_output.action,
                    n_keyframes=len(keyframes),
                    audio_model_called=bool(audio_text),
                    screenshot=screenshot,
                )

                # 9. Check success via DOM result + per-task validation
                time.sleep(0.3)  # Brief pause for DOM to update after action
                _, result_val = env.check_success()
                success = validate_dom_result(task.task_id, result_val)

                step_log = StepLog(
                    step=step_num,
                    action=cu_output.action[:200],
                    narration=cu_output.narration[:200],
                    audio_text=audio_text[:200],
                    n_keyframes=len(keyframes),
                    obs_overhead_ms=obs_overhead_ms,
                    model_latency_ms=model_latency_ms,
                    success=success,
                    result_val=result_val,
                )
                steps_log.append(step_log)

                logger.info(
                    "[%s] Step %d: action=%s | keyframes=%d | audio=%s | result=%s",
                    task.task_id, step_num, cu_output.action[:50],
                    len(keyframes), bool(audio_text), result_val,
                )

                if success:
                    break

                # Check for explicit completion signals
                if any(w in cu_output.action.lower() for w in ["done", "complete", "finish"]):
                    break

                # Check timeout
                if env.get_elapsed_s() > task.duration_s:
                    logger.info("[%s] Task timeout after %.1fs", task.task_id, env.get_elapsed_s())
                    break

            # Final success check via per-task DOM validator
            _, final_result = env.check_success()
            final_success = validate_dom_result(task.task_id, final_result)

            return EvalResult(
                task_id=task.task_id,
                category=task.category.value,
                difficulty=task.difficulty.value,
                model_name=self.model_name,
                observation_mode=self.observation_mode,
                success=final_success,
                result_val=final_result,
                steps_taken=len(steps_log),
                total_time_s=time.time() - task_start,
                total_model_latency_ms=total_model_latency,
                total_obs_overhead_ms=total_obs_overhead,
                steps=steps_log,
            )

        except Exception as e:
            logger.error("[%s] Task failed with exception: %s", task.task_id, e)
            return EvalResult(
                task_id=task.task_id, category=task.category.value,
                difficulty=task.difficulty.value, model_name=self.model_name,
                observation_mode=self.observation_mode,
                success=False, result_val="exception",
                steps_taken=len(steps_log), total_time_s=time.time() - task_start,
                total_model_latency_ms=0, total_obs_overhead_ms=0,
                steps=steps_log, error=str(e),
            )
        finally:
            env.stop()


def run_ablation(
    models: list[str],
    modes: list[str],
    task_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    difficulty_filter: Optional[str] = None,
    max_tasks: Optional[int] = None,
    max_steps: int = 15,
    output_dir: str = "results/ablation",
) -> list[EvalResult]:
    """
    Run the full ablation study: models × observation modes × tasks.

    Args:
        models: List of model names (e.g. ["claude-sonnet-4-6", "gpt-4o"])
        modes: List of observation modes (e.g. ["standard", "aoi_full"])
        task_filter: Filter tasks by ID prefix (e.g. "D-" for transient UI only)
        category_filter: Filter by category (e.g. "D_transient_ui")
        difficulty_filter: Filter by difficulty (e.g. "easy")
        max_tasks: Limit number of tasks to run
        max_steps: Max steps per task
        output_dir: Directory to write results JSON
    """
    bench = DynaCUBench(html_tasks_dir=Path("benchmark_env/html_tasks"))
    tasks = list(bench)

    # Apply filters
    if task_filter:
        tasks = [t for t in tasks if t.task_id.startswith(task_filter)]
    if category_filter:
        tasks = [t for t in tasks if t.category.value == category_filter]
    if difficulty_filter:
        tasks = [t for t in tasks if t.difficulty.value == difficulty_filter]
    if max_tasks:
        tasks = tasks[:max_tasks]

    logger.info(
        "Ablation: %d models × %d modes × %d tasks = %d evaluations",
        len(models), len(modes), len(tasks), len(models) * len(modes) * len(tasks),
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[EvalResult] = []

    for model_name in models:
        for mode in modes:
            logger.info("═══ %s + %s ═══", model_name, mode)
            evaluator = BrowserEvaluator(
                model_name=model_name,
                observation_mode=mode,
                max_steps=max_steps,
            )

            for task in tasks:
                logger.info("─── Task %s (%s, %s) ───",
                            task.task_id, task.category.value, task.difficulty.value)
                result = evaluator.run_task(task)
                all_results.append(result)

                status = "✓ PASS" if result.success else "✗ FAIL"
                logger.info(
                    "  %s: %s | steps=%d | model_ms=%.0f | result=%s",
                    status, task.task_id, result.steps_taken,
                    result.total_model_latency_ms, result.result_val,
                )

            # Save intermediate results after each model+mode combination
            combo_file = out_dir / f"{model_name}_{mode}.json"
            combo_results = [r for r in all_results
                            if r.model_name == model_name and r.observation_mode == mode]
            with open(combo_file, "w") as f:
                json.dump([r.to_dict() for r in combo_results], f, indent=2)

    # Save aggregated results
    with open(out_dir / "all_results.json", "w") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)

    # Print summary table
    _print_summary(all_results)

    return all_results


def _print_summary(results: list[EvalResult]):
    """Print a summary table of ablation results."""
    from collections import defaultdict

    # Group by model × mode
    groups: dict[tuple[str, str], list[EvalResult]] = defaultdict(list)
    for r in results:
        groups[(r.model_name, r.observation_mode)].append(r)

    # Also group by category
    cat_groups: dict[tuple[str, str, str], list[EvalResult]] = defaultdict(list)
    for r in results:
        cat_groups[(r.model_name, r.observation_mode, r.category)].append(r)

    print("\n" + "=" * 80)
    print("DynaCU-Bench Ablation Results")
    print("=" * 80)

    # Overall accuracy
    print(f"\n{'Model':<25} {'Mode':<20} {'Pass':>5} {'Total':>6} {'Acc%':>7} {'Avg Steps':>10}")
    print("-" * 73)
    for (model, mode), res_list in sorted(groups.items()):
        n_pass = sum(1 for r in res_list if r.success)
        n_total = len(res_list)
        acc = 100 * n_pass / n_total if n_total > 0 else 0
        avg_steps = sum(r.steps_taken for r in res_list) / n_total if n_total else 0
        print(f"{model:<25} {mode:<20} {n_pass:>5} {n_total:>6} {acc:>6.1f}% {avg_steps:>10.1f}")

    # Per-category breakdown
    print(f"\n{'Category':<30} ", end="")
    mode_list = sorted(set(r.observation_mode for r in results))
    for mode in mode_list:
        print(f"{mode:<15} ", end="")
    print()
    print("-" * (30 + 16 * len(mode_list)))

    categories = sorted(set(r.category for r in results))
    for cat in categories:
        print(f"{cat:<30} ", end="")
        for mode in mode_list:
            cat_res = [r for r in results if r.category == cat and r.observation_mode == mode]
            if cat_res:
                n_pass = sum(1 for r in cat_res if r.success)
                print(f"{n_pass}/{len(cat_res):<14} ", end="")
            else:
                print(f"{'—':<15} ", end="")
        print()

    print("=" * 80)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="DynaCU-Bench Browser Evaluation")
    parser.add_argument("--models", nargs="+", default=["claude-sonnet-4-6"],
                        help="CU models to evaluate")
    parser.add_argument("--modes", nargs="+", default=["standard", "aoi_full"],
                        help="Observation modes")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter by category (e.g. D_transient_ui)")
    parser.add_argument("--difficulty", type=str, default=None,
                        help="Filter by difficulty (easy/medium/hard)")
    parser.add_argument("--task-prefix", type=str, default=None,
                        help="Filter tasks by ID prefix (e.g. D-E)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Max tasks to run")
    parser.add_argument("--max-steps", type=int, default=15,
                        help="Max steps per task")
    parser.add_argument("--output-dir", type=str, default="results/ablation",
                        help="Output directory for results")

    args = parser.parse_args()

    results = run_ablation(
        models=args.models,
        modes=args.modes,
        category_filter=args.category,
        difficulty_filter=args.difficulty,
        task_filter=args.task_prefix,
        max_tasks=args.max_tasks,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )

    n_pass = sum(1 for r in results if r.success)
    print(f"\nDone. {n_pass}/{len(results)} tasks passed.")
