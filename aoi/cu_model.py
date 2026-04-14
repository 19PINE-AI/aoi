"""
CU Model Interface — wraps various computer-use LLMs with a unified API.

Each model receives:
  - Text context (trajectory narrations + audio)
  - Images (keyframes + post-action screenshot)
  - Task instruction

Each model returns:
  - action: str (the action to execute)
  - narration: str (visual description for long-term context)
"""

from __future__ import annotations

import base64
import io
import logging
import os
from dataclasses import dataclass
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode()


@dataclass
class CUModelOutput:
    action: str
    narration: str
    raw_response: str


NARRATION_INSTRUCTION = (
    "\n\nIn addition to your action, provide a one-sentence visual narration: "
    "describe any NEW visual information in the current screenshots not already "
    "captured in the audio or prior context. "
    "Format: NARRATION: <text>\\nACTION: <your action>"
)


class ClaudeCUModel:
    """Claude computer-use model via Anthropic API."""

    def __init__(self, model: str = "claude-opus-4-6", max_tokens: int = 1024):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.max_tokens = max_tokens

    def __call__(
        self,
        context_text: str,
        images: list[Image.Image],
        task: str,
    ) -> CUModelOutput:
        content = []

        if context_text.strip():
            content.append({"type": "text", "text": context_text})

        for img in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": pil_to_base64(img),
                },
            })

        content.append({
            "type": "text",
            "text": f"Task: {task}{NARRATION_INSTRUCTION}",
        })

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        raw = response.content[0].text
        return _parse_output(raw)


class OpenAICUModel:
    """OpenAI GPT-4V / computer-use model."""

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 1024):
        import openai
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.max_tokens = max_tokens

    def __call__(
        self,
        context_text: str,
        images: list[Image.Image],
        task: str,
    ) -> CUModelOutput:
        content = []

        if context_text.strip():
            content.append({"type": "text", "text": context_text})

        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{pil_to_base64(img)}"},
            })

        content.append({
            "type": "text",
            "text": f"Task: {task}{NARRATION_INSTRUCTION}",
        })

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        raw = response.choices[0].message.content
        return _parse_output(raw)


class GeminiCUModel:
    """Google Gemini computer-use model (google-genai SDK)."""

    def __init__(self, model: str = "gemini-2.0-flash", max_tokens: int = 1024):
        from google import genai
        from google.genai import types
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self._model_id = model
        self._types = types
        self.max_tokens = max_tokens

    def __call__(
        self,
        context_text: str,
        images: list[Image.Image],
        task: str,
    ) -> CUModelOutput:
        parts = []

        if context_text.strip():
            parts.append(self._types.Part.from_text(text=context_text))

        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            parts.append(self._types.Part.from_bytes(
                data=buf.getvalue(), mime_type="image/png",
            ))

        parts.append(self._types.Part.from_text(
            text=f"Task: {task}{NARRATION_INSTRUCTION}",
        ))

        response = self._client.models.generate_content(
            model=self._model_id,
            contents=[self._types.Content(role="user", parts=parts)],
            config=self._types.GenerateContentConfig(
                max_output_tokens=self.max_tokens,
            ),
        )
        raw = response.text
        return _parse_output(raw)


def _parse_output(raw: str) -> CUModelOutput:
    """Parse model output into (narration, action)."""
    narration = "No visual change."
    action = raw.strip()

    if "NARRATION:" in raw and "ACTION:" in raw:
        try:
            narr_part = raw.split("NARRATION:")[1].split("ACTION:")[0].strip()
            act_part = raw.split("ACTION:")[1].strip()
            narration = narr_part
            action = act_part
        except Exception:
            pass
    elif "NARRATION:" in raw:
        try:
            narration = raw.split("NARRATION:")[1].strip()
        except Exception:
            pass

    return CUModelOutput(action=action, narration=narration, raw_response=raw)


# ── Model registry ────────────────────────────────────────────────────────────
# Current SOTA CU models available via API (April 2026)
SOTA_MODELS = {
    # Anthropic
    "claude-opus-4-6":    ("anthropic", "claude-opus-4-6"),
    "claude-sonnet-4-6":  ("anthropic", "claude-sonnet-4-6"),
    # OpenAI
    "gpt-4o":             ("openai",    "gpt-4o"),
    # Google — use dedicated CU preview model where available
    "gemini-3-flash":     ("google",    "gemini-3-flash-preview"),
    "gemini-2.5-flash":   ("google",    "gemini-2.5-flash"),
    "gemini-cu":          ("google",    "gemini-2.5-computer-use-preview-10-2025"),
    "gemini-2.0-flash":   ("google",    "gemini-2.0-flash"),
}


def get_model(model_name: str) -> "ClaudeCUModel | OpenAICUModel | GeminiCUModel":
    """Factory to get a CU model by name (short alias or full ID)."""
    # Resolve alias
    if model_name in SOTA_MODELS:
        provider, full_id = SOTA_MODELS[model_name]
    else:
        # Infer provider from prefix
        if model_name.startswith("claude"):
            provider, full_id = "anthropic", model_name
        elif model_name.startswith("gpt") or model_name.startswith("o3") or model_name.startswith("o4"):
            provider, full_id = "openai", model_name
        elif model_name.startswith("gemini"):
            provider, full_id = "google", model_name
        else:
            raise ValueError(f"Unknown model: {model_name}. Known: {list(SOTA_MODELS)}")

    if provider == "anthropic":
        return ClaudeCUModel(model=full_id)
    elif provider == "openai":
        return OpenAICUModel(model=full_id)
    elif provider == "google":
        return GeminiCUModel(model=full_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")
