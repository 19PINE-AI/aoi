"""
AOI-Augmented Agent Loop.

Replaces the standard CU agent loop:
  repeat: screenshot -> CU_model -> execute -> wait

With the AOI-augmented loop:
  repeat:
    keyframes = keyframe_extractor.get_and_reset()
    audio = audio_observer.process(audio_buffer.get_chunk(...))
    context = trajectory.get_context(step_id)
    narration, action = cu_model(context, keyframes + [screenshot], task)
    trajectory.append(step, audio, narration, action)
    execute(action)
    wait(buffer_ms)

The AOI layer is transparent for static, silent tasks (zero overhead, zero extra tokens).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from PIL import Image

from .keyframe_extractor import KeyframeExtractor, Keyframe
from .audio_observer import AudioObserver, AudioBuffer, AudioChunk
from .observation_record import ObservationRecord, TrajectoryStore
from .screen_capture import ScreenCapture
from .cu_model import CUModelOutput

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    step_id: int
    action: str
    narration: str
    audio_text: str
    n_keyframes: int
    observation_overhead_ms: float
    audio_model_called: bool
    token_cost: dict


@dataclass
class AOIConfig:
    """Configuration for the AOI layer."""
    # Keyframe extraction
    clip_theta: float = 0.04  # Calibrated: web UI modals produce ~0.08 cosine distance
    pixel_threshold: float = 0.01
    max_keyframes_per_step: int = 5
    capture_fps: float = 3.0

    # Audio
    silence_threshold: float = 0.01
    audio_overlap_s: float = 3.5
    audio_backend: str = "gemini"  # "gemini", "openai_whisper", "none"

    # Trajectory
    context_depth: int = 3
    screenshot_history: int = 5

    # Timing
    post_action_buffer_ms: float = 500.0  # wait after action for effects to settle

    # Observation mode (for ablation studies)
    mode: str = "aoi_full"  # "standard", "uniform_1fps", "uniform_3fps", "pixel_diff",
                             # "aoi_visual_only", "aoi_visual_asr", "aoi_full"


class AOIAgentLoop:
    """
    Model-agnostic AOI-augmented agent loop.

    Usage:
        loop = AOIAgentLoop(cu_model=my_model, config=AOIConfig())
        loop.start()
        result = loop.run_task("Do X on the screen", max_steps=50)
        loop.stop()
    """

    def __init__(
        self,
        cu_model,
        config: AOIConfig = None,
        execute_action: Optional[Callable[[str], None]] = None,
        take_screenshot: Optional[Callable[[], Image.Image]] = None,
    ):
        self.cu_model = cu_model
        self.config = config or AOIConfig()

        # Allow injection of custom execute/screenshot for testing
        self._execute_action = execute_action or self._default_execute
        self._take_screenshot = take_screenshot or self._default_screenshot

        # AOI components
        self.keyframe_extractor = KeyframeExtractor(
            theta=self.config.clip_theta,
            pixel_threshold=self.config.pixel_threshold,
            max_keyframes=self.config.max_keyframes_per_step,
        )

        self.audio_buffer = AudioBuffer(
            overlap_s=self.config.audio_overlap_s,
        )

        if self.config.audio_backend != "none":
            self.audio_observer = AudioObserver(
                silence_threshold=self.config.silence_threshold,
                backend=self.config.audio_backend,
                overlap_s=self.config.audio_overlap_s,
            )
        else:
            self.audio_observer = None

        self.trajectory = TrajectoryStore(
            context_depth=self.config.context_depth,
            screenshot_history=self.config.screenshot_history,
        )

        self.screen_capture = ScreenCapture(
            fps=self.config.capture_fps,
            on_frame=self._on_screen_frame,
        )

        self._step_id = 0
        self._running = False
        self._step_start_time = time.time()

    def start(self) -> bool:
        """Start background capture threads."""
        self._running = True
        capture_ok = self.screen_capture.start()
        audio_ok = self.audio_buffer.start_capture()
        logger.info(
            "AOI started: screen_capture=%s, audio=%s, mode=%s",
            capture_ok, audio_ok, self.config.mode,
        )
        return capture_ok

    def stop(self):
        """Stop background capture threads."""
        self._running = False
        self.screen_capture.stop()
        self.audio_buffer.stop_capture()

    def _on_screen_frame(self, frame: Image.Image, timestamp: float):
        """Called by ScreenCapture thread for each frame."""
        if self.config.mode == "standard":
            return  # No intermediate frame processing in standard mode

        if self.config.mode in ("aoi_visual_only", "aoi_visual_asr", "aoi_full", "pixel_diff"):
            self.keyframe_extractor.on_sample(frame, timestamp)

        elif self.config.mode in ("uniform_1fps", "uniform_3fps"):
            # For uniform modes, always capture (KeyframeExtractor used with theta=0 effectively)
            from .keyframe_extractor import Keyframe
            import numpy as np
            # Just add every frame (extractor acts as a FIFO buffer)
            self.keyframe_extractor._lock.acquire()
            self.keyframe_extractor._keyframes.append(
                Keyframe(timestamp=timestamp, image=frame.copy(),
                         clip_embedding=np.zeros(512, dtype=np.float32),
                         pixel_change_ratio=1.0)
            )
            self.keyframe_extractor._lock.release()

    def run_step(self, task: str, screenshot: Optional[Image.Image] = None) -> StepResult:
        """Execute one agent step and return the result."""
        step_start = time.time()
        self._step_id += 1
        step_id = self._step_id

        t_obs_start = time.time()

        # 1. Collect keyframes from the previous interval
        keyframes = self.keyframe_extractor.get_and_reset()

        # 2. Process audio for this interval
        audio_text = ""
        audio_called = False
        if self.audio_observer is not None and self.config.mode in ("aoi_visual_asr", "aoi_full"):
            audio_chunk = self.audio_buffer.get_chunk(
                start_time=self._step_start_time,
                end_time=time.time(),
                include_overlap=True,
            )
            prior_transcript = self.trajectory.get_last_audio_text()
            audio_text = self.audio_observer.process(
                chunk=audio_chunk,
                prior_transcript=prior_transcript,
                new_portion_start=self._step_start_time,
            )
            audio_called = audio_text != ""

        # 3. Take post-action screenshot
        if screenshot is None:
            screenshot = self._take_screenshot()

        t_obs_end = time.time()
        obs_overhead_ms = (t_obs_end - t_obs_start) * 1000

        # 4. Build observation record
        context_steps = self.trajectory.get_context(step_id)
        obs_record = ObservationRecord(
            step_id=step_id,
            context_steps=context_steps,
            current_audio_text=audio_text,
            keyframes=keyframes,
            post_action_screenshot=screenshot,
            task_instruction=task,
        )

        # 5. CU model inference (narration + action)
        context_text = obs_record.to_prompt_text()
        images = obs_record.get_images()
        cu_output: CUModelOutput = self.cu_model(context_text, images, task)

        token_cost = obs_record.token_cost_estimate()

        # 6. Persist text record (images NOT stored)
        self.trajectory.append(
            step_id=step_id,
            step_start_time=self._step_start_time,
            step_end_time=time.time(),
            audio_text=audio_text,
            visual_narration=cu_output.narration,
            action=cu_output.action,
            n_keyframes=len(keyframes),
            audio_model_called=audio_called,
            screenshot=screenshot,
        )

        # 7. Execute action
        self._execute_action(cu_output.action)

        # 8. Wait for post-action effects to settle
        time.sleep(self.config.post_action_buffer_ms / 1000.0)
        self._step_start_time = time.time()

        return StepResult(
            step_id=step_id,
            action=cu_output.action,
            narration=cu_output.narration,
            audio_text=audio_text,
            n_keyframes=len(keyframes),
            observation_overhead_ms=obs_overhead_ms,
            audio_model_called=audio_called,
            token_cost=token_cost,
        )

    def run_task(
        self,
        task: str,
        max_steps: int = 50,
        done_fn: Optional[Callable[[StepResult], bool]] = None,
        screenshot_fn: Optional[Callable[[], Image.Image]] = None,
    ) -> list[StepResult]:
        """
        Run the full agent loop for a task.

        Args:
            task: Natural language task instruction
            max_steps: Maximum number of steps
            done_fn: Optional callback to check if task is complete
            screenshot_fn: Optional override for taking screenshots

        Returns:
            List of StepResults for all steps taken
        """
        self._step_start_time = time.time()
        self.trajectory = TrajectoryStore(
            context_depth=self.config.context_depth,
            screenshot_history=self.config.screenshot_history,
        )

        results = []
        for i in range(max_steps):
            screenshot = screenshot_fn() if screenshot_fn else None
            result = self.run_step(task, screenshot=screenshot)
            results.append(result)

            logger.info(
                "Step %d: action=%s, keyframes=%d, audio=%s, overhead=%.1fms",
                result.step_id,
                result.action[:60],
                result.n_keyframes,
                bool(result.audio_text),
                result.observation_overhead_ms,
            )

            if done_fn and done_fn(result):
                logger.info("Task complete after %d steps", i + 1)
                break

            # Check for explicit done action
            if "done" in result.action.lower() or "complete" in result.action.lower():
                break

        return results

    def get_efficiency_stats(self) -> dict:
        """Return AOI efficiency statistics for the current trajectory."""
        traj = self.trajectory.summary()
        kf_stats = self.keyframe_extractor.get_stats()
        audio_stats = self.audio_observer.get_stats() if self.audio_observer else {}
        return {
            "trajectory": traj,
            "keyframe_extractor": kf_stats,
            "audio_observer": audio_stats,
        }

    @staticmethod
    def _default_execute(action: str):
        """Default no-op executor (for testing)."""
        logger.debug("Execute: %s", action)

    @staticmethod
    def _default_screenshot() -> Image.Image:
        """Default screenshot using mss (returns blank if unavailable)."""
        try:
            import mss
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                shot = sct.grab(monitor)
                return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        except Exception:
            return Image.new("RGB", (1280, 720), (200, 200, 200))
