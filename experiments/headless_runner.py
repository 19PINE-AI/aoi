"""
Headless Evaluation Runner.

Instead of relying on real screen capture / audio devices, this runner:
1. Generates synthetic stimuli (frames + audio) for each task
2. Explicitly feeds frames to the KeyframeExtractor at the right timestamps
3. Explicitly feeds audio chunks to the AudioObserver
4. Builds the ObservationRecord for each step
5. Calls the CU model and records success

This lets us run the full AOI pipeline in a headless CI/lab environment
and get meaningful comparative results between observation modes.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from aoi.keyframe_extractor import KeyframeExtractor
from aoi.audio_observer import AudioObserver, AudioChunk
from aoi.observation_record import ObservationRecord, TrajectoryStore
from dynacubench.synthetic_media import SyntheticMediaGenerator
from dynacubench.tasks import Task, TaskCategory, DynaCUBench
from dynacubench.evaluator import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class HeadlessStepResult:
    step_id: int
    action: str
    narration: str
    audio_text: str
    n_keyframes: int
    context_text: str   # For debugging


class HeadlessTaskRunner:
    """
    Runs one task through the AOI pipeline without real hardware.

    Simulation fidelity:
    - Frames are served at 3 Hz from pre-rendered slideshow
    - Audio is pre-rendered and segmented into step-sized chunks
    - Each step processes the interval frames + audio
    - CU model receives the real observation record
    """

    def __init__(
        self,
        cu_model,
        observation_mode: str = "aoi_full",
        step_duration_s: float = 3.5,
        audio_overlap_s: float = 0.5,
        clip_theta: float = 0.15,
        audio_backend: str = "none",  # Use "none" for mock; "gemini" for real
        audio_silence_threshold: float = 0.01,
    ):
        self.cu_model = cu_model
        self.observation_mode = observation_mode
        self.step_duration_s = step_duration_s
        self.audio_overlap_s = audio_overlap_s
        self.clip_theta = clip_theta
        self.audio_backend = audio_backend
        self.audio_silence_threshold = audio_silence_threshold

        self.gen = SyntheticMediaGenerator()

        # AOI components — re-created per task
        self._extractor: Optional[KeyframeExtractor] = None
        self._audio_obs: Optional[AudioObserver] = None
        self._trajectory: Optional[TrajectoryStore] = None

    def _init_components(self):
        self._extractor = KeyframeExtractor(
            theta=self.clip_theta,
            pixel_threshold=0.01,
            max_keyframes=5,
        )
        if self.audio_backend != "none":
            self._audio_obs = AudioObserver(
                silence_threshold=self.audio_silence_threshold,
                backend=self.audio_backend,
                overlap_s=self.audio_overlap_s,
            )
        else:
            self._audio_obs = None
        self._trajectory = TrajectoryStore(context_depth=3)

    def _get_frames_for_step(
        self,
        all_frames: list[tuple[float, Image.Image]],
        step_start: float,
        step_end: float,
    ) -> list[tuple[float, Image.Image]]:
        """Return frames within the current step interval."""
        return [(t, img) for t, img in all_frames if step_start <= t < step_end]

    def _get_audio_for_step(
        self,
        audio: np.ndarray,
        sample_rate: int,
        step_start: float,
        step_end: float,
    ) -> np.ndarray:
        """Return audio slice for the current step interval (with overlap prepended)."""
        start_s = max(0.0, step_start - self.audio_overlap_s)
        start_idx = int(start_s * sample_rate)
        end_idx = int(step_end * sample_rate)
        return audio[start_idx:min(end_idx, len(audio))]

    def _simulate_audio_model(
        self,
        audio_slice: np.ndarray,
        sample_rate: int,
        task: Task,
        step_id: int,
    ) -> str:
        """
        Simulate what an audio model would return for a given audio slice.
        Uses task metadata to generate realistic audio descriptions.

        In a real evaluation, this calls the actual Gemini/Whisper model.
        """
        if len(audio_slice) == 0:
            return ""

        # Check RMS of the new (post-overlap) portion
        overlap_samples = int(self.audio_overlap_s * sample_rate)
        new_audio = audio_slice[overlap_samples:]
        if len(new_audio) == 0:
            return ""

        rms = float(np.sqrt(np.mean(new_audio ** 2)))
        if rms < self.audio_silence_threshold:
            return ""  # Silent

        # Return task-specific audio description
        meta = task.metadata

        # Category B: meeting audio
        if task.category == TaskCategory.B_MEETING:
            spoken_url = meta.get("spoken_url", "")
            spoken_number = meta.get("spoken_number", "")
            spoken_code = meta.get("spoken_code", "")
            if spoken_url and step_id >= 2:
                return f"Speaker says: 'Please check the full report at {spoken_url}'"
            if spoken_number and step_id >= 2:
                return f"Speaker announces room number {spoken_number}"
            if spoken_code and step_id >= 2:
                return f"Facilitator says: 'The code word is {spoken_code}'"
            if meta.get("speakers") and step_id >= 3:
                return "Alice: 'I'll prepare the report'. Bob: 'I'll schedule the demo'"
            return "Meeting in progress. Participants discussing agenda."

        # Category D: audio alerts
        if task.category == TaskCategory.D_AUDIO_ALERT:
            alert_type = meta.get("alert_type", "")
            expected = meta.get("expected", "")
            if "calendar" in alert_type:
                return f"Calendar alarm sound. The system alarm is ringing."
            if "notification" in alert_type:
                return "Notification ding sound heard."
            if expected == "critical" or meta.get("alert_frequency") == "high":
                return "High-pitched beep: critical system alert."
            if "timer" in alert_type:
                return "Timer alarm sound. The countdown has ended."
            return "Alert sound detected."

        # Category E: combined
        if task.category == TaskCategory.E_COMBINED:
            narration = meta.get("narration", "")
            verbal = meta.get("verbal_instruction", "")
            if narration and step_id >= 1:
                return f"Presenter says: '{narration}'"
            if verbal and step_id >= 2:
                return f"Presenter instructs: '{verbal}'"
            return "Presenter is speaking and showing content."

        return ""  # No relevant audio for other categories

    def _simulate_keyframe_extraction(
        self,
        step_frames: list[tuple[float, Image.Image]],
        task: Task,
    ) -> list:
        """
        Process frames through the keyframe extractor.
        For modes without CLIP, simulate appropriate behavior.
        """
        if self.observation_mode == "standard":
            return []  # No intermediate frames in standard mode

        if self.observation_mode in ("aoi_full", "aoi_visual_only", "aoi_visual_asr", "pixel_diff"):
            # Feed each frame to the CLIP-based extractor
            for t, frame in step_frames:
                self._extractor.on_sample(frame, t)
            return self._extractor.get_and_reset()

        elif self.observation_mode == "uniform_1fps":
            # Return last frame only (1 FPS)
            if step_frames:
                t, img = step_frames[-1]
                from aoi.keyframe_extractor import Keyframe
                return [Keyframe(t, img, np.zeros(512, dtype=np.float32), 1.0)]
            return []

        elif self.observation_mode == "uniform_3fps":
            # Return up to 3 evenly spaced frames
            from aoi.keyframe_extractor import Keyframe
            if not step_frames:
                return []
            indices = np.linspace(0, len(step_frames) - 1, min(3, len(step_frames)), dtype=int)
            return [Keyframe(step_frames[i][0], step_frames[i][1], np.zeros(512, dtype=np.float32), 1.0)
                    for i in indices]

        return []

    def run_task(self, task: Task, max_steps: int = 15) -> tuple[list[HeadlessStepResult], bool]:
        """Run a task and return (step_results, success)."""
        self._init_components()

        # Generate stimulus
        make_fn = getattr(self.gen, f"make_task_{task.task_id.replace('-', '').lower()}_stimulus", None)
        if make_fn:
            all_frames, audio_data = make_fn()
        else:
            slide = [{"text": task.instruction[:40], "duration_s": task.duration_s}]
            all_frames = self.gen.create_slideshow_frames(slide)
            audio_data = np.zeros(int(task.duration_s * 16000), dtype=np.float32)

        sample_rate = 16000
        all_actions = []
        step_results = []

        for step_id in range(1, max_steps + 1):
            step_start = (step_id - 1) * self.step_duration_s
            step_end = step_id * self.step_duration_s

            # 1. Get frames for this step
            step_frames = self._get_frames_for_step(all_frames, step_start, step_end)

            # 2. Process frames through keyframe extractor
            keyframes = self._simulate_keyframe_extraction(step_frames, task)

            # 3. Get post-action screenshot (last frame in step, or default)
            if step_frames:
                screenshot = step_frames[-1][1]
            elif all_frames:
                screenshot = all_frames[-1][1]
            else:
                screenshot = Image.new("RGB", (1280, 720), (200, 200, 200))

            # 4. Process audio
            audio_text = ""
            if self.observation_mode not in ("standard", "aoi_visual_only"):
                audio_slice = self._get_audio_for_step(
                    audio_data, sample_rate, step_start, step_end
                )
                if self.audio_backend != "none" and self._audio_obs is not None:
                    chunk = AudioChunk(
                        data=audio_slice.astype(np.float32),
                        sample_rate=sample_rate,
                        start_time=step_start,
                        end_time=step_end,
                    )
                    audio_text = self._audio_obs.process(
                        chunk=chunk,
                        prior_transcript=self._trajectory.get_last_audio_text(),
                        new_portion_start=step_start,
                    )
                else:
                    # Simulate what audio model would say
                    audio_text = self._simulate_audio_model(
                        audio_slice, sample_rate, task, step_id
                    )

            # 5. Build observation record
            context_steps = self._trajectory.get_context(step_id)
            obs = ObservationRecord(
                step_id=step_id,
                context_steps=context_steps,
                current_audio_text=audio_text,
                keyframes=keyframes,
                post_action_screenshot=screenshot,
                task_instruction=task.instruction,
            )

            context_text = obs.to_prompt_text()
            images = obs.get_images()

            # 6. Call CU model
            from aoi.cu_model import CUModelOutput
            cu_output = self.cu_model(context_text, images, task.instruction)

            # 7. Generate keyframe narration from visual context
            if keyframes:
                narration = f"Keyframe captured: {len(keyframes)} frame(s) of visual change"
            else:
                narration = cu_output.narration

            # 8. Record step
            self._trajectory.append(
                step_id=step_id,
                step_start_time=step_start,
                step_end_time=step_end,
                audio_text=audio_text,
                visual_narration=narration,
                action=cu_output.action,
                n_keyframes=len(keyframes),
                audio_model_called=bool(audio_text),
                screenshot=screenshot,
            )

            step_results.append(HeadlessStepResult(
                step_id=step_id,
                action=cu_output.action,
                narration=narration,
                audio_text=audio_text,
                n_keyframes=len(keyframes),
                context_text=context_text,
            ))

            all_actions.append(cu_output.action)

            # Check success
            all_actions_text = " | ".join(all_actions)
            if task.success_fn and task.success_fn(all_actions_text):
                break

        all_actions_text = " | ".join(r.action for r in step_results)
        success = task.success_fn(all_actions_text) if task.success_fn else False

        return step_results, success


class HeadlessEvaluator:
    """Evaluates multiple observation modes across DynaCU-Bench tasks in headless mode."""

    def __init__(
        self,
        cu_model_factory,  # callable(mode) -> CU model
        output_dir: Path = Path("results"),
        audio_backend: str = "none",  # "none" for mocks, "gemini" for real
    ):
        self.cu_model_factory = cu_model_factory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_backend = audio_backend

    def run_mode(
        self,
        benchmark: DynaCUBench,
        mode: str,
        max_tasks: Optional[int] = None,
        max_steps: int = 15,
    ) -> list[EvaluationResult]:
        """Run one observation mode across all tasks."""
        tasks = list(benchmark)
        if max_tasks:
            tasks = tasks[:max_tasks]

        model = self.cu_model_factory(mode)
        runner = HeadlessTaskRunner(
            cu_model=model,
            observation_mode=mode,
            audio_backend=self.audio_backend,
        )

        results = []
        for i, task in enumerate(tasks):
            logger.info("[%d/%d] %s (mode=%s)", i + 1, len(tasks), task.task_id, mode)
            t0 = time.time()
            try:
                step_results, success = runner.run_task(task, max_steps=max_steps)
            except Exception as e:
                logger.error("Task %s error: %s", task.task_id, e)
                step_results = []
                success = False

            wall_time = time.time() - t0
            traj_summary = runner._trajectory.summary() if runner._trajectory else {}

            result = EvaluationResult(
                task_id=task.task_id,
                category=task.category.value,
                difficulty=task.difficulty.value,
                success=success,
                agent_output=" | ".join(r.action for r in step_results),
                n_steps=len(step_results),
                n_keyframes_total=traj_summary.get("total_keyframes", 0),
                n_audio_activations=traj_summary.get("audio_steps", 0),
                total_tokens=sum(
                    (len(r.context_text.split()) + len(r.action.split())) for r in step_results
                ),
                wall_time_s=wall_time,
                observation_mode=mode,
                model_name=getattr(model, "model_name", "unknown"),
            )
            results.append(result)
            logger.info(
                "  -> success=%s, steps=%d, keyframes=%d, audio=%d",
                success, len(step_results),
                traj_summary.get("total_keyframes", 0),
                traj_summary.get("audio_steps", 0),
            )

        return results

    def run_ablation(
        self,
        benchmark: DynaCUBench,
        modes: list[str],
        max_tasks: Optional[int] = None,
        max_steps: int = 15,
    ) -> dict:
        """Run full ablation across modes and return summaries."""
        all_summaries = {}

        for mode in modes:
            logger.info("=" * 60)
            logger.info("MODE: %s", mode)
            logger.info("=" * 60)

            results = self.run_mode(benchmark, mode, max_tasks, max_steps)

            # Save per-mode results
            filename = f"headless_{mode}.json"
            path = self.output_dir / filename
            with open(path, "w") as f:
                json.dump([r.to_dict() for r in results], f, indent=2)

            # Compute summary
            total = len(results)
            successful = sum(1 for r in results if r.success)
            by_cat = {}
            for r in results:
                cat = r.category
                by_cat.setdefault(cat, {"total": 0, "success": 0})
                by_cat[cat]["total"] += 1
                by_cat[cat]["success"] += int(r.success)

            total_steps = sum(r.n_steps for r in results)
            summary = {
                "mode": mode,
                "success_rate": successful / total if total else 0.0,
                "n_successful": successful,
                "total_tasks": total,
                "by_category": {
                    cat: {
                        "success_rate": v["success"] / v["total"],
                        **v,
                    }
                    for cat, v in by_cat.items()
                },
                "avg_keyframes_per_step": sum(r.n_keyframes_total for r in results) / max(total_steps, 1),
                "avg_audio_activation_ratio": sum(r.n_audio_activations for r in results) / max(total_steps, 1),
                "avg_steps": total_steps / max(total, 1),
                "avg_tokens": sum(r.total_tokens for r in results) / max(total, 1),
            }
            all_summaries[mode] = summary

            logger.info(
                "Overall: %.1f%% (%d/%d)",
                summary["success_rate"] * 100, successful, total,
            )
            for cat, v in by_cat.items():
                logger.info("  %s: %.1f%%", cat[:20], v["success"] / v["total"] * 100)

        # Save aggregate
        with open(self.output_dir / "headless_summary.json", "w") as f:
            json.dump(all_summaries, f, indent=2)

        return all_summaries
