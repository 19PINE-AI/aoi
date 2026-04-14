"""
Observation Record and Trajectory Store.

The ObservationRecord assembles the structured input for the CU model at each step:
  [CONTEXT] — text from recent prior steps (narrations + audio + actions)
  [NEW]     — current audio text + keyframe images + post-action screenshot

The TrajectoryStore maintains the full trajectory with text persistence:
  - Audio transcriptions: persist as text forever
  - Visual narrations: persist as text forever
  - Keyframe images: NOT stored (ephemeral, pruned after step)
  - Post-action screenshots: stored up to history_window images
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image


@dataclass
class StepRecord:
    """Persistent record of a single agent step stored in trajectory."""
    step_id: int
    step_start_time: float
    step_end_time: float
    audio_text: str           # From AudioObserver (persists)
    visual_narration: str     # From CU model's narration output (persists)
    action: str               # Action taken (persists)
    n_keyframes: int          # How many keyframes were presented (metadata only)
    audio_model_called: bool  # Whether audio model was invoked


@dataclass
class ObservationRecord:
    """
    The complete input record assembled for the CU model at one step.
    Contains both text context (from trajectory) and raw images.
    """
    step_id: int
    context_steps: list[StepRecord]       # Prior steps as text only
    current_audio_text: str               # New audio from this step
    keyframes: list                        # list of Keyframe objects (with .image, .timestamp)
    post_action_screenshot: Optional[Image.Image]
    task_instruction: str

    def to_prompt_text(self) -> str:
        """Render the text portion of the observation record."""
        lines = [f"=== Step {self.step_id} Observation ===\n"]

        if self.context_steps:
            lines.append("[CONTEXT — prior steps]\n")
            for step in self.context_steps:
                t_start = step.step_start_time
                t_end = step.step_end_time
                lines.append(f"  Step {step.step_id} (t={t_start:.1f}–{t_end:.1f}s):")
                if step.audio_text:
                    lines.append(f"    AUDIO: {step.audio_text}")
                if step.visual_narration:
                    lines.append(f"    VISUAL: {step.visual_narration}")
                lines.append(f"    ACTION: {step.action}")
                lines.append("")

        lines.append("[NEW — current interval]")
        if self.current_audio_text:
            lines.append(f"  AUDIO: {self.current_audio_text}")

        if self.keyframes:
            for kf in self.keyframes:
                lines.append(f"  [{kf.timestamp:.1f}s] <keyframe image>")

        lines.append(f"  <post-action screenshot>")
        lines.append("")
        lines.append(f"[TASK] {self.task_instruction}")

        return "\n".join(lines)

    def get_images(self) -> list[Image.Image]:
        """Return all images in order: keyframes first, then post-action screenshot."""
        images = [kf.image for kf in self.keyframes]
        if self.post_action_screenshot is not None:
            images.append(self.post_action_screenshot)
        return images

    def token_cost_estimate(self) -> dict:
        """Rough token cost estimate for this observation."""
        text_tokens = len(self.to_prompt_text().split()) * 1.3  # rough approximation
        image_tokens = len(self.get_images()) * 258  # CLIP ViT token count
        return {
            "text_tokens": int(text_tokens),
            "image_tokens": image_tokens,
            "total": int(text_tokens) + image_tokens,
        }


class TrajectoryStore:
    """
    Maintains the agent trajectory with:
    - Persistent text records for all steps
    - Sliding window of post-action screenshots
    - Context depth control (how many prior steps to include)
    """

    def __init__(
        self,
        context_depth: int = 3,          # Prior steps to include as CONTEXT
        screenshot_history: int = 5,     # Post-action screenshots to keep
    ):
        self.context_depth = context_depth
        self.screenshot_history = screenshot_history

        self._steps: list[StepRecord] = []
        self._screenshots: list[tuple[int, Image.Image]] = []  # (step_id, image)

    def append(
        self,
        step_id: int,
        step_start_time: float,
        step_end_time: float,
        audio_text: str,
        visual_narration: str,
        action: str,
        n_keyframes: int,
        audio_model_called: bool,
        screenshot: Optional[Image.Image] = None,
    ) -> StepRecord:
        """Record a completed step."""
        record = StepRecord(
            step_id=step_id,
            step_start_time=step_start_time,
            step_end_time=step_end_time,
            audio_text=audio_text,
            visual_narration=visual_narration,
            action=action,
            n_keyframes=n_keyframes,
            audio_model_called=audio_model_called,
        )
        self._steps.append(record)

        if screenshot is not None:
            self._screenshots.append((step_id, screenshot))
            # Prune old screenshots
            if len(self._screenshots) > self.screenshot_history:
                self._screenshots = self._screenshots[-self.screenshot_history:]

        return record

    def get_context(self, current_step_id: int) -> list[StepRecord]:
        """
        Return the last context_depth steps as text-only context.
        Adaptively returns fewer if there's no dynamic content.
        """
        prior = [s for s in self._steps if s.step_id < current_step_id]

        # Adaptive depth: if no audio/visual activity in recent steps, reduce depth
        recent = prior[-self.context_depth:] if prior else []
        has_dynamic = any(s.audio_text or s.visual_narration != "No visual change." for s in recent)

        if not has_dynamic:
            return []  # Static task — no context needed

        return recent

    def get_last_audio_text(self) -> str:
        """Return the most recent audio transcription for overlap context."""
        for step in reversed(self._steps):
            if step.audio_text:
                return step.audio_text
        return ""

    def get_full_transcript(self) -> str:
        """Return all audio text concatenated (for long-form tasks)."""
        parts = [s.audio_text for s in self._steps if s.audio_text]
        return " ".join(parts)

    def get_full_narration(self) -> str:
        """Return all visual narrations concatenated."""
        parts = [s.visual_narration for s in self._steps if s.visual_narration]
        return " ".join(parts)

    def summary(self) -> dict:
        n = len(self._steps)
        if n == 0:
            return {"steps": 0}
        audio_steps = sum(1 for s in self._steps if s.audio_text)
        keyframe_steps = sum(1 for s in self._steps if s.n_keyframes > 0)
        total_keyframes = sum(s.n_keyframes for s in self._steps)
        return {
            "steps": n,
            "audio_steps": audio_steps,
            "audio_step_ratio": audio_steps / n,
            "keyframe_steps": keyframe_steps,
            "avg_keyframes_per_step": total_keyframes / n,
            "total_keyframes": total_keyframes,
        }
