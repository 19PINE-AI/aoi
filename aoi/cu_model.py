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


NARRATION_INSTRUCTION = """

You are a computer-use agent controlling a real browser. You MUST output exactly one executable action.

Available actions:
  click(x, y)              — click at pixel coordinates (e.g. click(640, 360))
  type("text")             — type text into the focused input (auto-finds first visible input)
  fill(#id, "text")        — fill a specific input by its HTML id attribute
  triple_click(x, y)       — select all text at coordinates (for replacing text)
  key(Enter)               — press a keyboard key (Enter, Tab, Escape, ctrl+a, etc.)
  scroll(down, 300)        — scroll the page (up/down, pixels)
  speak("text")            — say text aloud through the microphone
  wait()                   — do nothing, observe the screen more
  done                     — signal task is complete

OBSERVATION SPACE:
- You receive IMAGES: keyframes showing visual changes + current screenshot.
- You may receive AUDIO transcripts: text heard through the browser's audio.
- You may receive PAGE ELEMENTS: list of interactive elements with their IDs and positions.
- You receive CONTEXT: your prior steps' observations and actions.
- You may receive FULL AUDIO HISTORY: all audio heard across all steps.

RULES:
1. Output EXACTLY one action. Do NOT describe what you plan to do.
2. If you hear audio content (podcasts, meetings, voicemails), LISTEN and USE that information.
3. If [PAGE ELEMENTS] shows an input with id="X", use fill(#X, "value") to fill it.
4. If no PAGE ELEMENTS, click the input field first, then type("text").
5. For tasks that require listening to audio: use wait() to let more audio play before acting.
6. For tasks that require spoken responses: type your answer into the text input field.

Also provide a one-sentence visual narration of what is NEW on screen.

Format your response exactly as:
NARRATION: <one sentence about new visual info, or "No visual change.">
ACTION: <exactly one action command>"""


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

        # GPT-5+ uses max_completion_tokens; older models use max_tokens
        token_param = (
            {"max_completion_tokens": self.max_tokens}
            if "gpt-5" in self.model or "o3" in self.model or "o4" in self.model
            else {"max_tokens": self.max_tokens}
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            **token_param,
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
        raw = response.text or ""
        return _parse_output(raw)


class LocalVLLMModel:
    """
    Open-source VLM served via vLLM with OpenAI-compatible API.
    Supports Fara-7B, UI-TARS-1.5, and other Qwen2.5-VL-based CU models.
    """

    # Model-specific system prompts
    FARA_SYSTEM = (
        "You are a computer-use agent. You control a real browser.\n"
        "Output a JSON action: {\"name\":\"computer_use\",\"arguments\":{\"action\":\"<action>\",\"coordinate\":[x,y]}}\n"
        "Actions: left_click, type, key, scroll, terminate.\n"
        "For typing: {\"name\":\"computer_use\",\"arguments\":{\"action\":\"type\",\"text\":\"...\"}}\n"
        "For keys: {\"name\":\"computer_use\",\"arguments\":{\"action\":\"key\",\"key\":\"Enter\"}}\n"
    )

    UI_TARS_SYSTEM = (
        "You are a GUI agent. Perform the task by outputting actions.\n"
        "Action format: click(start_box='(x,y)'), type(text='...'), scroll(direction='down'), done\n"
    )

    def __init__(
        self,
        model: str = "microsoft/Fara-7B",
        base_url: str = "http://localhost:5000/v1",
        max_tokens: int = 1024,
        model_family: str = "fara",  # "fara", "ui_tars", "generic"
    ):
        import openai
        self.client = openai.OpenAI(base_url=base_url, api_key="local")
        self.model = model
        self.max_tokens = max_tokens
        self.model_family = model_family

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

        # Use model-specific system prompt if available
        system = ""
        if self.model_family == "fara":
            system = self.FARA_SYSTEM
        elif self.model_family == "ui_tars":
            system = self.UI_TARS_SYSTEM

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        raw = response.choices[0].message.content
        return self._parse_local_output(raw)

    def _parse_local_output(self, raw: str) -> CUModelOutput:
        """Parse output from local models, handling model-specific formats."""
        import json as json_mod
        import re

        # Try standard NARRATION/ACTION format first
        output = _parse_output(raw)

        # For Fara: parse JSON action format
        if self.model_family == "fara":
            try:
                # Extract JSON from response
                json_match = re.search(r'\{[^{}]*"name"\s*:\s*"computer_use"[^{}]*\}', raw)
                if json_match:
                    action_json = json_mod.loads(json_match.group())
                    args = action_json.get("arguments", {})
                    act_type = args.get("action", "")
                    coord = args.get("coordinate", [])

                    if act_type == "left_click" and coord:
                        output.action = f"click({coord[0]}, {coord[1]})"
                    elif act_type == "type":
                        output.action = f'type("{args.get("text", "")}")'
                    elif act_type == "key":
                        output.action = f'key({args.get("key", "Enter")})'
                    elif act_type == "scroll":
                        output.action = f'scroll(down, 300)'
                    elif act_type == "terminate":
                        output.action = "done"
            except Exception:
                pass

        # For UI-TARS: convert click(start_box='(x,y)') to click(x, y)
        elif self.model_family == "ui_tars":
            box_match = re.search(r"click\(start_box='?\((\d+),\s*(\d+)\)'?\)", output.action)
            if box_match:
                output.action = f"click({box_match.group(1)}, {box_match.group(2)})"

        return output


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
    "gpt-5.4":            ("openai",    "gpt-5.4"),
    "gpt-5":              ("openai",    "gpt-5"),
    "gpt-4o":             ("openai",    "gpt-4o"),
    # Google — use dedicated CU preview model where available
    "gemini-3-flash":     ("google",    "gemini-3-flash-preview"),
    "gemini-2.5-flash":   ("google",    "gemini-2.5-flash"),
    "gemini-cu":          ("google",    "gemini-2.5-computer-use-preview-10-2025"),
    "gemini-2.0-flash":   ("google",    "gemini-2.0-flash"),
}

# Open-source models served locally via vLLM
LOCAL_MODELS = {
    "fara-7b":       ("microsoft/Fara-7B",                "fara",    "http://localhost:5000/v1"),
    "ui-tars-7b":    ("ByteDance-Seed/UI-TARS-1.5-7B",   "ui_tars", "http://localhost:5001/v1"),
    "ui-tars-72b":   ("ByteDance-Seed/UI-TARS-72B-DPO",  "ui_tars", "http://localhost:5002/v1"),
    "opencua-7b":    ("xlangai/OpenCUA-7B",               "generic", "http://localhost:5003/v1"),
    "evocua-32b":    ("meituan/EvoCUA-32B-20260105",      "generic", "http://localhost:5004/v1"),
}


def get_model(model_name: str) -> "ClaudeCUModel | OpenAICUModel | GeminiCUModel | LocalVLLMModel":
    """Factory to get a CU model by name (short alias or full ID)."""
    # Check local models first
    if model_name in LOCAL_MODELS:
        hf_id, family, base_url = LOCAL_MODELS[model_name]
        # Allow override via env var (e.g. VLLM_BASE_URL=http://host:port/v1)
        base_url = os.environ.get("VLLM_BASE_URL", base_url)
        return LocalVLLMModel(model=hf_id, base_url=base_url, model_family=family)

    # Resolve API model alias
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
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"API models: {list(SOTA_MODELS)}. "
                f"Local models: {list(LOCAL_MODELS)}"
            )

    if provider == "anthropic":
        return ClaudeCUModel(model=full_id)
    elif provider == "openai":
        return OpenAICUModel(model=full_id)
    elif provider == "google":
        return GeminiCUModel(model=full_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")
