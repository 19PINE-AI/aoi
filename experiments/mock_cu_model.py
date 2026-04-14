"""
Mock CU models for controlled evaluation experiments.

These simulate different model behaviors:
- Standard model: responds only to screenshots
- AOI-aware model: uses audio + keyframe context
- Oracle model: perfect success (upper bound)
- Blind model: always fails (lower bound)
"""

from __future__ import annotations

import random
import re
from PIL import Image
from aoi.cu_model import CUModelOutput


class MockStandardCUModel:
    """
    Simulates a standard screenshot-only CU agent.
    Has limited ability to detect dynamic content.
    """

    def __init__(self, model_name: str = "mock_standard", success_rate_static: float = 0.8):
        self.model_name = model_name
        self.success_rate_static = success_rate_static

    def __call__(self, context_text: str, images: list[Image.Image], task: str) -> CUModelOutput:
        # Standard model only sees screenshots — no audio context
        # Can only succeed if the answer is visible in the current screenshot
        has_audio_context = "AUDIO:" in context_text and context_text.split("AUDIO:")[1].strip()[:5] != ""

        # For transient UI tasks: occasionally catches the event
        if "cookie" in task.lower():
            if random.random() < 0.15:  # Low chance — must catch it in a screenshot
                return CUModelOutput(
                    action="click Accept All",
                    narration="I see a cookie consent popup and clicked Accept All.",
                    raw_response="NARRATION: Cookie popup visible\nACTION: click Accept All",
                )

        if "download" in task.lower():
            if random.random() < 0.1:
                return CUModelOutput(
                    action="click Open file",
                    narration="Download notification visible.",
                    raw_response="ACTION: click Open file",
                )

        # For audio tasks: always fails (no audio perception)
        if any(kw in task.lower() for kw in ["hear", "sound", "listen", "alarm", "audio", "meeting"]):
            return CUModelOutput(
                action="wait()",
                narration="No visual change.",
                raw_response="NARRATION: No visual change.\nACTION: wait()",
            )

        # For video tasks: occasionally catches something from a screenshot
        if any(kw in task.lower() for kw in ["video", "watch", "tutorial", "demo"]):
            if random.random() < 0.05:  # Very low chance
                return CUModelOutput(
                    action="type 'CloudSync Pro'",
                    narration="Slide visible with product name.",
                    raw_response="ACTION: type 'CloudSync Pro'",
                )
            return CUModelOutput(
                action="wait()",
                narration="No visual change.",
                raw_response="ACTION: wait()",
            )

        return CUModelOutput(
            action="wait()",
            narration="No visual change.",
            raw_response="ACTION: wait()",
        )


class MockAOICUModel:
    """
    Simulates a CU agent equipped with the full AOI (visual + audio).
    Has high success on tasks where audio or keyframe context is available.
    """

    def __init__(self, model_name: str = "mock_aoi", task_knowledge: dict = None):
        self.model_name = model_name
        self.task_knowledge = task_knowledge or {}
        self._step_count = 0

    def __call__(self, context_text: str, images: list[Image.Image], task: str) -> CUModelOutput:
        self._step_count += 1

        # Extract audio context from the observation record
        audio_text = ""
        if "AUDIO:" in context_text:
            try:
                audio_text = context_text.split("AUDIO:")[1].split("\n")[0].strip()
            except Exception:
                pass

        # Extract visual narration context
        visual_context = ""
        if "VISUAL:" in context_text:
            try:
                visual_context = context_text.split("VISUAL:")[1].split("\n")[0].strip()
            except Exception:
                pass

        n_keyframes = context_text.count("<keyframe image>")

        # ── Category A: Video comprehension ──────────────────────
        if "cloudsync pro" in task.lower() or "product name" in task.lower():
            if n_keyframes > 0 or "cloudsync" in visual_context.lower():
                return CUModelOutput(
                    action="type 'CloudSync Pro'",
                    narration="Keyframe shows slide with 'CloudSync Pro' text.",
                    raw_response="NARRATION: Slide shows CloudSync Pro\nACTION: type 'CloudSync Pro'",
                )

        if "count" in task.lower() and "slide" in task.lower():
            if n_keyframes >= 3 or self._step_count > 8:
                return CUModelOutput(
                    action="type '5'",
                    narration="Counted 5 slides from keyframes.",
                    raw_response="ACTION: type '5'",
                )

        # ── Category B: Meeting audio ─────────────────────────────
        if "example.com/report" in audio_text or "example dot com" in audio_text.lower():
            return CUModelOutput(
                action="open_url('https://example.com/report')",
                narration="No visual change.",
                raw_response="ACTION: open_url('https://example.com/report')",
            )

        if "407" in audio_text or "room" in audio_text.lower():
            return CUModelOutput(
                action="type '407'",
                narration="No visual change.",
                raw_response="ACTION: type '407'",
            )

        if "delta-7" in audio_text.lower() or "delta 7" in audio_text.lower():
            return CUModelOutput(
                action="type 'DELTA-7'",
                narration="No visual change.",
                raw_response="ACTION: type 'DELTA-7'",
            )

        # ── Category C: Transient UI ─────────────────────────────
        if n_keyframes > 0:
            if "cookie" in task.lower():
                if "cookie" in visual_context.lower() or "accept" in visual_context.lower():
                    return CUModelOutput(
                        action="click Accept All",
                        narration="Keyframe shows cookie consent popup.",
                        raw_response="NARRATION: Cookie popup captured\nACTION: click Accept All",
                    )
            if "download" in task.lower():
                if "download" in visual_context.lower() or "complete" in visual_context.lower():
                    return CUModelOutput(
                        action="click Open file",
                        narration="Download toast notification captured.",
                        raw_response="ACTION: click Open file",
                    )

        # ── Category D: Audio alerts ─────────────────────────────
        if audio_text:
            if "alarm" in audio_text.lower() or "beep" in audio_text.lower() or "chime" in audio_text.lower():
                if "alarm" in task.lower() or "calendar" in task.lower():
                    return CUModelOutput(
                        action="type 'Team Standup'",
                        narration="No visual change.",
                        raw_response="ACTION: type 'Team Standup'",
                    )
                if "critical" in task.lower() or "error" in task.lower():
                    return CUModelOutput(
                        action="type 'CRITICAL'",
                        narration="No visual change.",
                        raw_response="ACTION: type 'CRITICAL'",
                    )
                if "notification" in task.lower():
                    return CUModelOutput(
                        action="switch_to_messaging_app()",
                        narration="No visual change.",
                        raw_response="ACTION: switch_to_messaging_app()",
                    )

        # ── Category E: Combined ─────────────────────────────────
        if "submit" in task.lower():
            if "submit" in audio_text.lower() or "submit" in visual_context.lower():
                return CUModelOutput(
                    action="click Submit button",
                    narration="Submit button visible.",
                    raw_response="ACTION: click Submit button",
                )

        if "nexus" in audio_text.lower() or "nexus" in visual_context.lower():
            return CUModelOutput(
                action="fill form: company='Nexus Labs', year='2019'",
                narration="Slide shows Nexus Labs; speaker mentioned 2019.",
                raw_response="ACTION: fill form: company='Nexus Labs', year='2019'",
            )

        if "export" in audio_text.lower():
            return CUModelOutput(
                action="click File > Export",
                narration="No visual change.",
                raw_response="ACTION: click File > Export",
            )

        # Default: wait and observe
        return CUModelOutput(
            action="wait()",
            narration="Observing. No actionable content yet.",
            raw_response="ACTION: wait()",
        )


class RealCUModelWrapper:
    """Wraps a real CU model (Claude/GPT/Gemini) with evaluation-friendly interface."""

    def __init__(self, base_model, model_name: str = "claude"):
        self.base_model = base_model
        self.model_name = model_name

    def __call__(self, context_text: str, images: list[Image.Image], task: str) -> CUModelOutput:
        return self.base_model(context_text, images, task)
