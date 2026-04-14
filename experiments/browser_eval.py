"""
End-to-end Browser Evaluation Harness for DynaCU-Bench v3.

Runs real CU model inference against real HTML task pages in headless Chromium.
Supports 5 observation modes for ablation studies.

Architecture:
    BrowserEnvironment (Playwright)
        ↓ screenshot (PIL Image)
    AOI Pipeline (keyframes, audio, narration)
        ↓ observation record
    CU Model (Claude/GPT-4o/Gemini via API)
        ↓ parsed action
    BrowserEnvironment.execute_action()
        ↓ DOM state + LLM judge
    evaluate() → EvalOutcome
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
from aoi.audio_pipeline import AudioProcessor, TwoLayerAudio, TTSEngine
from aoi.observation_record import ObservationRecord, TrajectoryStore
from aoi.cu_model import get_model, CUModelOutput, SOTA_MODELS
from dynacubench.tasks_v3 import (
    DynaCUBenchV3, Task, TaskCategory, TaskDifficulty, EvalType,
)
from dynacubench.llm_evaluator import LLMEvaluator, EvalOutcome

logger = logging.getLogger(__name__)

# ── Task evaluation ─────────────────────────────────────────────────
# v3 uses dom_success_value from the task registry + LLM evaluator for
# hybrid/llm tasks.  No more hardcoded per-task validators.

FAIL_VALUES = frozenset({
    "pending", "unknown", "error", "timeout", "alarm_missed",
    "session_expired", "incorrect",
})


def validate_dom_result(task: Task, result_val: str) -> bool:
    """Check DOM result against the task's expected success value."""
    if task.dom_success_value:
        return result_val == task.dom_success_value
    # Fallback: any non-failure value
    return (
        result_val not in FAIL_VALUES
        and not result_val.startswith("wrong_")
    )


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
    # LLM evaluation fields (for hybrid/llm tasks)
    eval_type: str = "dom"
    llm_score: Optional[float] = None
    llm_reason: Optional[str] = None
    final_score: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = [asdict(s) for s in self.steps]
        return d


class BrowserEvaluator:
    """
    Runs a single task against a CU model with a specified observation mode.

    Observation modes (v3):
        standard          - Screenshot-only. No keyframes, no audio.
        aoi_visual         - Keyframes only (3fps CLIP extraction), no audio.
        aoi_audio          - Audio only (PulseAudio → Whisper), no keyframes.
        aoi_full           - Keyframes + audio (full perception, no speak).
        aoi_interactive    - Full perception + speak action (TTS → mic).
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
        self._audio_processor: Optional[AudioProcessor] = None

        # LLM evaluator for hybrid/llm tasks
        self._llm_evaluator = LLMEvaluator()

    def _init_aoi(self):
        """Initialize AOI components for a new task."""
        import subprocess

        # Kill any lingering audio injection from prior task
        if hasattr(self, '_audio_inject_thread') and self._audio_inject_thread is not None:
            self._audio_inject_thread.join(timeout=2)
            self._audio_inject_thread = None
        # Kill any lingering pacat processes from prior audio injection
        subprocess.run(["pkill", "-f", "pacat.*virtual_speaker"],
                       capture_output=True, timeout=2)
        # Brief pause to let PulseAudio drain residual audio
        time.sleep(0.5)

        use_keyframes = self.observation_mode in (
            "aoi_visual", "aoi_visual_only", "aoi_visual_asr", "aoi_full",
            "aoi_interactive", "pixel_diff", "uniform_1fps", "uniform_3fps",
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

        # Audio pipeline: real PulseAudio capture → Whisper ASR
        use_audio = self.observation_mode in ("aoi_visual_asr", "aoi_full", "aoi_audio", "aoi_interactive")
        if use_audio:
            if self._audio_processor is None:
                self._audio_processor = AudioProcessor(
                    layer1_duration_s=5.0,
                    layer2_duration_s=60.0,
                    whisper_model_size="base",
                    silence_threshold=self.silence_threshold,
                )
            else:
                # Reset ring buffer to prevent audio leaking from prior task
                self._audio_processor.reset()
            # Start capture (idempotent)
            self._audio_processor.start()
        else:
            self._audio_processor = None

        self._trajectory = TrajectoryStore(context_depth=5, screenshot_history=5)

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

    def _inject_page_audio(self, env: BrowserEnvironment):
        """
        Extract audio text from the HTML page and play it through PulseAudio
        virtual_speaker so the AOI audio pipeline can capture it.

        Headless Chromium has no speechSynthesis voices, so we simulate
        browser audio output by: page JS → extract text → edge-tts → pacat → virtual_speaker.

        This is functionally equivalent to what would happen if speechSynthesis
        worked: audio flows through PulseAudio to the capture pipeline.
        """
        if self._audio_processor is None:
            return

        try:
            # Extract audio content from page JavaScript
            audio_texts = env._page.evaluate('''() => {
                // Try common variable names used in our task pages
                const texts = [];

                // audioText variable (podcasts, voicemails)
                if (typeof audioText !== 'undefined') {
                    texts.push(audioText);
                }

                // slideAudio array (meetings with slides)
                if (typeof slideAudio !== 'undefined' && Array.isArray(slideAudio)) {
                    texts.push(...slideAudio);
                }

                // Find SpeechSynthesisUtterance text in script elements
                if (texts.length === 0) {
                    const scripts = document.querySelectorAll('script');
                    for (const s of scripts) {
                        const src = s.textContent;
                        // Match SpeechSynthesisUtterance("...") or ('...')
                        // Use separate patterns for double and single quotes
                        // to handle apostrophes inside text correctly
                        const dqMatches = src.matchAll(/SpeechSynthesisUtterance\("([^"]+)"\)/g);
                        for (const m of dqMatches) {
                            if (m[1] && m[1].length > 5) texts.push(m[1]);
                        }
                        const sqMatches = src.matchAll(/SpeechSynthesisUtterance\('([^']+)'\)/g);
                        for (const m of sqMatches) {
                            if (m[1] && m[1].length > 5) texts.push(m[1]);
                        }
                    }
                }

                return texts;
            }''')

            if not audio_texts:
                logger.debug("No audio content found in page")
                return

            # Combine all audio texts
            full_text = " ... ".join(audio_texts)
            # Estimate audio duration (~150 words/min for TTS, ~5 chars/word)
            self._audio_duration_s = len(full_text) / (150 * 5 / 60)
            logger.info("Injecting page audio (%d chars, ~%.1fs): %s",
                        len(full_text), self._audio_duration_s, full_text[:80])

            # Synthesize and play in background thread
            import threading
            def _play_audio():
                try:
                    import subprocess
                    tts = TTSEngine(voice="en-US-GuyNeural")
                    audio_data, sr = tts.synthesize(full_text)
                    if len(audio_data) > 0:
                        # Play through virtual_speaker sink
                        raw_bytes = audio_data.astype(np.float32).tobytes()
                        subprocess.run(
                            ["pacat", "--format=float32le", f"--rate={sr}",
                             "--channels=1", "--device=virtual_speaker", "--raw"],
                            input=raw_bytes, capture_output=True,
                            timeout=len(audio_data) / sr + 10,
                        )
                        logger.info("Page audio injection complete (%.1fs)", len(audio_data) / sr)
                except Exception as e:
                    logger.warning("Audio injection failed: %s", e)

            thread = threading.Thread(target=_play_audio, daemon=True)
            thread.start()
            # Store thread reference so we know audio is playing
            self._audio_inject_thread = thread

        except Exception as e:
            logger.warning("Failed to extract page audio: %s", e)

    def _capture_audio(self, env: BrowserEnvironment, duration_s: float) -> TwoLayerAudio:
        """
        Capture audio via real PulseAudio → Whisper ASR pipeline.

        Returns a TwoLayerAudio object with:
          Layer 1: Recent 3-5s transcript (synced with keyframes)
          Layer 2: Rolling 30-60s transcript with sentence timestamps

        No DOM proxies — audio flows through the real PulseAudio virtual
        speaker sink, exactly as a human would hear it.
        """
        if self._audio_processor is None:
            return TwoLayerAudio(
                layer1_text="", layer1_duration_s=0,
                layer2_segments=[], layer2_duration_s=0,
                capture_end_time=time.time(),
            )

        return self._audio_processor.get_two_layer_audio()

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

        # Gemini 3 Flash uses 0-1000 normalized coordinates
        uses_normalized = "gemini-3" in self.model_name

        # AOI modes need extra time for audio injection, CLIP processing,
        # Whisper ASR, and pre-step buffering (~30s overhead)
        aoi_overhead = 30.0 if self.observation_mode != "standard" else 0.0
        env = BrowserEnvironment(
            html_file=task.html_file,
            width=1280, height=720,
            task_timeout_s=task.duration_s + 10 + aoi_overhead,
            coord_scale_1000=uses_normalized,
        )
        # Attach audio processor to env for speak action support
        if self._audio_processor:
            env._audio_processor = self._audio_processor

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

            # Inject page audio through PulseAudio (since speechSynthesis
            # doesn't work in headless Chromium)
            if self.observation_mode in ("aoi_audio", "aoi_full", "aoi_interactive",
                                         "aoi_visual_asr"):
                self._inject_page_audio(env)
                # Wait proportionally to audio length so agent hears content
                # before its first step. Short audio (< 5s) → 2s wait,
                # long audio (podcast/meeting) → up to 5s head start.
                # Keep this moderate — longer waits eat the step budget.
                audio_wait = getattr(self, '_audio_duration_s', 0.0)
                pre_step_wait = min(max(2.0, audio_wait * 0.3), 5.0)
                logger.info("Pre-step audio buffer: %.1fs wait (audio=%.1fs)",
                            pre_step_wait, audio_wait)
                time.sleep(pre_step_wait)

            total_model_latency = 0.0
            total_obs_overhead = 0.0
            last_action_error: Optional[str] = None  # For error feedback to model

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

                # 3. Audio (real PulseAudio → Whisper pipeline)
                two_layer = self._capture_audio(env, self.step_interval_s)
                audio_text = two_layer.layer1_text
                audio_context = two_layer.format_for_prompt() if two_layer.has_audio else ""

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
                    current_audio_context=audio_context,
                    keyframes=keyframes,
                    post_action_screenshot=screenshot,
                    task_instruction=task.instruction,
                )

                # 6. CU model inference
                context_text = obs.to_prompt_text()

                # Add interactive page elements for AOI modes
                # This gives the model DOM context about available inputs/buttons
                if self.observation_mode != "standard":
                    page_elements = env.get_interactive_elements()
                    if page_elements:
                        context_text += f"\n\n{page_elements}"

                # Add accumulated audio transcript for audio-enabled modes
                # This ensures no spoken information is lost from earlier steps
                if self.observation_mode in ("aoi_full", "aoi_audio",
                                              "aoi_interactive", "aoi_visual_asr"):
                    full_transcript = self._trajectory.get_full_transcript()
                    if full_transcript and step_num > 1:
                        context_text += (
                            f"\n\n[FULL AUDIO HISTORY — all audio heard so far]\n"
                            f"  {full_transcript}"
                        )

                # Inject action error feedback so the model can self-correct
                if last_action_error:
                    context_text += (
                        f"\n\n[ACTION ERROR — previous action failed]\n"
                        f"  Error: {last_action_error}\n"
                        f"  Try a different action. Use click(x, y) with pixel "
                        f"coordinates. To type text, use type(\"text\") or "
                        f"click the input first then type(\"text\"). "
                        f"See [PAGE ELEMENTS] above for valid element IDs."
                    )
                    last_action_error = None

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

                # Track action errors for feedback on next step
                if not action_result.success and action_result.error:
                    last_action_error = action_result.error
                    logger.info("[%s] Action error at step %d: %s",
                                task.task_id, step_num, action_result.error[:100])

                # 8. Store in trajectory
                self._trajectory.append(
                    step_id=step_num,
                    step_start_time=t_obs_start,
                    step_end_time=time.time(),
                    audio_text=audio_text,
                    visual_narration=cu_output.narration,
                    action=cu_output.action,
                    n_keyframes=len(keyframes),
                    audio_model_called=two_layer.has_audio,
                    audio_context=audio_context,
                    screenshot=screenshot,
                )

                # 9. Check success via DOM result + task-defined validation
                time.sleep(0.3)  # Brief pause for DOM to update after action
                _, result_val = env.check_success()
                success = validate_dom_result(task, result_val)

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

                # Check for explicit completion signals (standalone, not embedded)
                act_l = cu_output.action.lower().strip()
                if act_l in ("done", "complete", "finish", "done.", "task complete",
                             "task done", "done(success)", "done(failure)",
                             "stop", "stop()", "stop()."):
                    break

                # Check timeout
                if env.get_elapsed_s() > task.duration_s:
                    logger.info("[%s] Task timeout after %.1fs", task.task_id, env.get_elapsed_s())
                    break

            # Final evaluation: DOM check + optional LLM evaluation
            _, final_result = env.check_success()
            dom_passed = validate_dom_result(task, final_result)

            # Collect agent's typed/spoken responses for LLM evaluation
            agent_response = env.get_page_text()

            # Run LLM evaluation for hybrid/llm tasks
            eval_outcome = self._llm_evaluator.evaluate_task(
                task,
                dom_result=final_result,
                agent_response=agent_response,
            )

            return EvalResult(
                task_id=task.task_id,
                category=task.category.value,
                difficulty=task.difficulty.value,
                model_name=self.model_name,
                observation_mode=self.observation_mode,
                success=eval_outcome.final_passed,
                result_val=final_result,
                steps_taken=len(steps_log),
                total_time_s=time.time() - task_start,
                total_model_latency_ms=total_model_latency,
                total_obs_overhead_ms=total_obs_overhead,
                steps=steps_log,
                eval_type=task.eval_type.value,
                llm_score=eval_outcome.llm_result.score if eval_outcome.llm_result else None,
                llm_reason=eval_outcome.llm_result.reason if eval_outcome.llm_result else None,
                final_score=eval_outcome.final_score,
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
            if self._audio_processor:
                self._audio_processor.stop()


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
    bench = DynaCUBenchV3(html_tasks_dir=Path("benchmark_env/html_tasks"))
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
