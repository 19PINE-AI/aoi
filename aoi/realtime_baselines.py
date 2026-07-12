"""
Realtime/Live API baselines for DynaCU-Bench.

We compare the AOI (model-agnostic perception layer over a batch CU model)
against monolithic streaming multimodal models that bundle perception with
reasoning into a single end-to-end model:

    GeminiLiveBaseline    — Google Gemini 2.5/Flash Live API (audio + video + tool use)
    OpenAIRealtimeBaseline — OpenAI Realtime API (audio + tool use, images via context)

These are NOT drop-in CU agents; the public Realtime/Live APIs are voice-first
function-calling models.  The closest fair comparison is:
    1. Open one streaming session per task.
    2. Stream the page audio (PulseAudio capture) into the session.
    3. Periodically send the current screenshot as a multimodal input.
    4. The model emits tool calls; we map them to browser actions.
    5. Continue until the model emits `done` or the task times out.

Both classes implement the same `run_task(task) -> EvalResult` interface
as `BrowserEvaluator`, so they slot into the existing eval harness.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from benchmark_env.browser_env import BrowserEnvironment
from aoi.audio_pipeline import AudioProcessor, TTSEngine
from dynacubench.tasks_v3 import Task
from dynacubench.llm_evaluator import LLMEvaluator

log = logging.getLogger(__name__)


# Identical action grammar to BrowserEvaluator so the browser env executes
# unchanged.  We expose them as JSON-schema function tools.
ACTION_TOOLS = [
    {
        "name": "click",
        "description": "Click at the given pixel coordinates on the page.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "fill",
        "description": "Fill a form input identified by HTML id with the given text.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["id", "text"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text into the currently-focused input.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "select",
        "description": "Choose an option in a <select> dropdown by its value.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["id", "value"],
        },
    },
    {
        "name": "key",
        "description": "Press a keyboard key (e.g. Enter, Tab, Escape).",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "scroll",
        "description": "Scroll the page up or down by a number of pixels.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down"]},
                "pixels": {"type": "integer"},
            },
            "required": ["direction", "pixels"],
        },
    },
    {
        "name": "speak",
        "description": "Say text aloud through the microphone (for tasks that ask the agent to respond verbally).",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "wait",
        "description": "Do nothing this step; observe more screen/audio.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "done",
        "description": "Signal that the task is complete.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def tool_call_to_action(name: str, args: dict) -> str:
    """Translate a function-call into the action string the browser executes."""
    if name == "click":
        return f"click({args.get('x', 0)}, {args.get('y', 0)})"
    if name == "fill":
        return f'fill(#{args.get("id", "").lstrip("#")}, "{args.get("text", "")}")'
    if name == "type_text":
        return f'type("{args.get("text", "")}")'
    if name == "select":
        return f'select(#{args.get("id", "").lstrip("#")}, "{args.get("value", "")}")'
    if name == "key":
        return f'key({args.get("key", "Enter")})'
    if name == "scroll":
        return f'scroll({args.get("direction", "down")}, {args.get("pixels", 300)})'
    if name == "speak":
        return f'speak("{args.get("text", "")}")'
    if name == "wait":
        return "wait()"
    if name == "done":
        return "done"
    return "wait()"


@dataclass
class RealtimeStepLog:
    step: int
    action: str
    tool_call: str
    result_val: str
    elapsed_s: float


@dataclass
class RealtimeEvalResult:
    task_id: str
    category: str
    difficulty: str
    model_name: str
    observation_mode: str
    success: bool
    result_val: str
    steps_taken: int
    total_time_s: float
    steps: list = field(default_factory=list)
    error: Optional[str] = None
    final_score: float = 0.0
    heard_audio: str = ""

    def to_dict(self):
        d = asdict(self)
        d["steps"] = [asdict(s) if hasattr(s, "__dict__") else s for s in self.steps]
        return d


SYSTEM_PROMPT = (
    "You are a computer-use agent driving a real web browser.  You receive screenshots "
    "of the page and audio captured from the page's speaker.  At each step you MUST "
    "call exactly one of the provided tools to control the browser.  Do not narrate; "
    "just call the tool.  When the task is complete, call `done`."
)


# ──────────────────────────────────────────────────────────────────────
# Gemini Live baseline
# ──────────────────────────────────────────────────────────────────────

class GeminiLiveBaseline:
    """Gemini 2.5 Flash Live as a CU agent.

    Per task: open a Live session, stream screenshots and the captured page
    audio, accept tool calls, execute them in Playwright, and continue until
    the model calls `done` or the task times out.

    Implementation notes:
      - The Live API is asyncio-only.  We run the asyncio loop in a worker
        thread so the synchronous Playwright env can block on browser calls.
      - Audio is streamed in 1-second 16-bit PCM @ 16 kHz chunks.
      - Screenshots are sent at 0.5 Hz to balance realism with latency.
    """

    def __init__(self, model: str = "gemini-2.5-flash-native-audio-latest",
                 max_steps: int = 15, step_interval_s: float = 2.0):
        self.model = model
        self.max_steps = max_steps
        self.step_interval_s = step_interval_s
        self._llm_evaluator = LLMEvaluator()

    def run_task(self, task: Task) -> RealtimeEvalResult:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options={"api_version": "v1beta"},
        )

        env = BrowserEnvironment(
            html_file=task.html_file, width=1280, height=720,
            task_timeout_s=task.duration_s + 30,
        )
        env.start()
        time.sleep(1.0)

        # Set up audio capture from PulseAudio
        audio_proc = AudioProcessor(layer1_duration_s=2.0, layer2_duration_s=8.0,
                                    silence_threshold=0.01)
        audio_proc.start()

        # Inject page audio
        try:
            self._inject_page_audio(env, audio_proc)
        except Exception as e:
            log.warning("Audio injection failed: %s", e)

        steps_log = []
        t_start = time.time()
        action_count = [0]
        last_result = ["pending"]
        done_flag = [False]
        last_action_str = [""]

        # Build proper FunctionDeclaration objects from the ACTION_TOOLS dicts.
        # Native-audio Live models require AUDIO modality but still emit tool
        # calls when the user message implies an action — that's the path we use.
        _SCHEMA_TYPE_MAP = {"object":"OBJECT","integer":"INTEGER","string":"STRING","boolean":"BOOLEAN"}
        def _make_schema(d):
            t = _SCHEMA_TYPE_MAP.get((d.get("type") or "string").lower(), "STRING")
            kw = {"type": t}
            if "properties" in d:
                kw["properties"] = {k: _make_schema(v) for k, v in d["properties"].items()}
            if "required" in d:
                kw["required"] = list(d["required"])
            if "enum" in d:
                kw["enum"] = list(d["enum"])
            return types.Schema(**kw)
        fn_decls = [
            types.FunctionDeclaration(
                name=t["name"], description=t.get("description", ""),
                parameters=_make_schema(t.get("parameters") or {"type": "object"}),
            )
            for t in ACTION_TOOLS
        ]
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],  # native-audio model requires audio modality
            tools=[types.Tool(function_declarations=fn_decls)],
            system_instruction=types.Content(
                role="user",
                parts=[types.Part(text=SYSTEM_PROMPT + f"\n\nTASK: {task.instruction}")],
            ),
        )

        # Use a single per-step async send/recv (one short Live session per step).
        # This avoids running Playwright sync calls inside an async-thread, which
        # causes greenlet errors when the worker thread doesn't own the page.
        # Each step is still a real Live session round-trip — the streaming
        # nature is preserved within the step; we just don't keep one open
        # across the entire task.
        err = None

        async def one_step(image_bytes: bytes, audio_text: str, step_num: int):
            try:
                async with client.aio.live.connect(model=self.model, config=config) as session:
                    parts = []
                    if audio_text:
                        parts.append({"text": f"[heard audio over the last {self.step_interval_s:.0f}s]: {audio_text}"})
                    parts.append({"inline_data": {"mime_type": "image/jpeg",
                                                  "data": base64.b64encode(image_bytes).decode()}})
                    parts.append({"text": f"Step {step_num}. Pick exactly one tool call to drive the browser."})
                    await session.send_client_content(
                        turns={"role": "user", "parts": parts},
                        turn_complete=True,
                    )
                    tool_name, tool_args = None, {}
                    async for response in session.receive():
                        if response.tool_call:
                            for fc in response.tool_call.function_calls:
                                tool_name = fc.name
                                tool_args = dict(fc.args) if fc.args else {}
                            break
                        if hasattr(response, "server_content") and response.server_content:
                            if getattr(response.server_content, "turn_complete", False):
                                break
                    return tool_name, tool_args, None
            except Exception as e:
                return None, {}, str(e)

        def run_async(coro_factory):
            """Run an async fn to completion in a fresh thread + loop, so we
            never collide with Playwright's own event loop on the main thread.
            coro_factory: a 0-arg callable that returns a fresh coroutine."""
            result_holder = [None]
            err_holder = [None]
            def _worker():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result_holder[0] = loop.run_until_complete(coro_factory())
                    finally:
                        loop.close()
                except Exception as e:
                    err_holder[0] = str(e)
            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            t.join(timeout=60)
            if err_holder[0]:
                return (None, {}, err_holder[0])
            return result_holder[0] or (None, {}, "no result")

        # Main step loop on the same thread as the Playwright env (sync calls safe here).
        for step_num in range(1, self.max_steps + 1):
            if env.get_elapsed_s() > task.duration_s:
                break

            # 1. Take screenshot (sync, owns Playwright)
            try:
                img = env.get_screenshot()
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                img_bytes = buf.getvalue()
            except Exception as e:
                log.warning("Screenshot failed: %s", e)
                break

            # 2. Snapshot recent audio transcript (Whisper-side state is shared)
            try:
                two_layer = audio_proc.get_two_layer_audio()
                audio_text = two_layer.layer1_text or ""
            except Exception:
                audio_text = ""

            # 3. Round-trip the Live API for one tool call
            tool_name, tool_args, step_err = run_async(
                lambda: one_step(img_bytes, audio_text, step_num)
            )
            if step_err:
                err = step_err
                log.warning("Live step %d error: %s", step_num, step_err[:160])
                # Still log the step so we have a record
                steps_log.append(RealtimeStepLog(
                    step=step_num, action="(error)",
                    tool_call=f"ERROR: {step_err[:120]}",
                    result_val=last_result[0],
                    elapsed_s=time.time() - t_start,
                ))
                # Brief backoff in case of rate limit / transient
                time.sleep(2.0)
                continue

            action = tool_call_to_action(tool_name, tool_args) if tool_name else "wait()"
            last_action_str[0] = action

            # 4. Execute action (sync, owns Playwright)
            try:
                env.execute_action(action)
            except Exception as e:
                log.warning("Execute action failed: %s", e)

            time.sleep(0.3)
            try:
                _, result_val = env.check_success()
            except Exception:
                result_val = last_result[0]
            last_result[0] = result_val

            steps_log.append(RealtimeStepLog(
                step=step_num, action=action[:200],
                tool_call=f"{tool_name}({json.dumps(tool_args, default=str)[:120]})" if tool_name else "(none)",
                result_val=result_val,
                elapsed_s=time.time() - t_start,
            ))
            log.info("[%s gemini-live] step %d: %s -> %s",
                     task.task_id, step_num, action[:60], result_val)

            if tool_name == "done" or action == "done":
                break

        # Final scoring
        try:
            _, final_result = env.check_success()
        except Exception:
            final_result = last_result[0]

        agent_response = ""
        try:
            agent_response = env.get_page_text()
        except Exception:
            pass

        eval_outcome = self._llm_evaluator.evaluate_task(
            task, dom_result=final_result, agent_response=agent_response,
        )

        env.stop()
        audio_proc.stop()

        return RealtimeEvalResult(
            task_id=task.task_id, category=task.category.value,
            difficulty=task.difficulty.value, model_name=self.model,
            observation_mode="gemini_live",
            success=eval_outcome.final_passed,
            result_val=final_result,
            steps_taken=len(steps_log),
            total_time_s=time.time() - t_start,
            steps=steps_log,
            error=err,
            final_score=eval_outcome.final_score,
        )

    def _inject_page_audio(self, env, audio_proc):
        """Same TTS injection as BrowserEvaluator: extract page text → edge-tts → PulseAudio."""
        import subprocess as sp
        try:
            audio_texts = env._page.evaluate('''() => {
                const t = [];
                if (typeof audioText !== "undefined") t.push(audioText);
                if (typeof slideAudio !== "undefined" && Array.isArray(slideAudio)) t.push(...slideAudio);
                return t;
            }''')
            if not audio_texts:
                # fall back to captured speechSynthesis utterances
                for _ in range(40):
                    time.sleep(0.5)
                    cap = env._page.evaluate('window.__capturedUtterances ? window.__capturedUtterances.slice() : []')
                    if cap and len(cap) > 0:
                        audio_texts = cap
                        break
            if not audio_texts:
                return

            full = " ... ".join(audio_texts)
            tts = TTSEngine(voice="en-US-GuyNeural")
            audio, sr = tts.synthesize(full)
            if len(audio) == 0:
                return

            def _play():
                raw = audio.astype(np.float32).tobytes()
                sp.run(
                    ["pacat", "--format=float32le", f"--rate={sr}",
                     "--channels=1", "--device=virtual_speaker", "--raw"],
                    input=raw, capture_output=True,
                    timeout=len(audio) / sr + 10,
                )

            t = threading.Thread(target=_play, daemon=True)
            t.start()
        except Exception as e:
            log.debug("audio injection error: %s", e)


# ──────────────────────────────────────────────────────────────────────
# OpenAI Realtime baseline
# ──────────────────────────────────────────────────────────────────────

class OpenAIRealtimeBaseline:
    """OpenAI Realtime API as a CU agent.

    The Realtime endpoint is voice-first.  When the underlying model
    supports vision inside the same Realtime session (gpt-realtime-2.0 and
    later), we still use chat-completions with the same model id as the
    integration path because the Realtime websocket exposes only the audio
    side; what we want to measure is the multimodal reasoning the model
    brings to a screenshot + audio transcript + tool-call decision.  For
    older models that do not accept images in Realtime (e.g.
    gpt-4o-realtime-preview), we fall back to a chat-completions call with
    the matching non-realtime model id (gpt-4o).

    The `model` argument selects the realtime-family model.  Recognised
    values (May 2026):
      * "gpt-4o"                   — legacy chat-completions stand-in for
                                     gpt-4o-realtime-preview.
      * "gpt-realtime-2.0"         — current OpenAI Realtime model with
                                     in-session vision; we route through
                                     chat completions on the realtime
                                     model id.
    """

    def __init__(self, vision_model: str = "gpt-realtime-2.0",
                 max_steps: int = 15, step_interval_s: float = 2.0):
        self.model = vision_model
        self.max_steps = max_steps
        self.step_interval_s = step_interval_s
        self._llm_evaluator = LLMEvaluator()

    def run_task(self, task: Task) -> RealtimeEvalResult:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        env = BrowserEnvironment(
            html_file=task.html_file, width=1280, height=720,
            task_timeout_s=task.duration_s + 30,
        )
        env.start()
        time.sleep(1.0)

        audio_proc = AudioProcessor(layer1_duration_s=4.0, layer2_duration_s=15.0,
                                    silence_threshold=0.01)
        audio_proc.start()

        try:
            self._inject_page_audio(env, audio_proc)
        except Exception as e:
            log.warning("Audio injection failed: %s", e)

        steps_log = []
        t_start = time.time()
        last_result = "pending"
        prior_tool_calls = []
        full_audio_history = ""
        api_failures = 0          # count steps where the model call raised
        last_api_error = None

        # OpenAI Responses API with function calling
        tools_for_responses = [{"type": "function", "function": t} for t in ACTION_TOOLS]

        for step in range(1, self.max_steps + 1):
            time.sleep(self.step_interval_s)

            # Capture audio segment
            two_layer = audio_proc.get_two_layer_audio()
            if two_layer.layer1_text:
                full_audio_history += " " + two_layer.layer1_text

            # Encode screenshot
            img = env.get_screenshot()
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            user_text = f"Task: {task.instruction}\n"
            if full_audio_history.strip():
                user_text += f"\n[All audio heard so far]: {full_audio_history.strip()}\n"
            user_text += f"\nStep {step} — choose exactly one tool call to drive the browser."

            # Use chat-completions with tools for compatibility.
            # `self.model` selects the realtime-family backbone — gpt-4o for
            # legacy gpt-4o-realtime-preview; gpt-realtime-2.0 for the
            # vision-capable Realtime model released in 2026.
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                                "detail": "auto",
                            }},
                        ]},
                    ],
                    tools=tools_for_responses,
                    tool_choice="required",
                    max_tokens=512,
                )
                msg = resp.choices[0].message
                if msg.tool_calls:
                    tc = msg.tool_calls[0]
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        tool_args = {}
                else:
                    tool_name, tool_args = "wait", {}
            except Exception as e:
                # Do NOT silently degrade to wait(): a bad model id / 404 would
                # otherwise masquerade as a genuine all-wait "0/N" result. Track
                # it so the task is flagged invalid rather than scored as a real 0.
                api_failures += 1
                last_api_error = str(e)
                log.warning("API call failed at step %d: %s", step, e)
                tool_name, tool_args = "wait", {}

            action = tool_call_to_action(tool_name, tool_args)
            env.execute_action(action)

            time.sleep(0.3)
            _, result_val = env.check_success()
            last_result = result_val

            steps_log.append(RealtimeStepLog(
                step=step, action=action[:200],
                tool_call=f"{tool_name}({json.dumps(tool_args, default=str)[:120]})",
                result_val=result_val, elapsed_s=time.time() - t_start,
            ))

            log.info("[%s openai-realtime] step %d: %s -> %s",
                     task.task_id, step, action[:60], result_val)

            if tool_name == "done":
                break
            if env.get_elapsed_s() > task.duration_s:
                break

        try:
            _, final_result = env.check_success()
        except Exception:
            final_result = last_result

        try:
            agent_response = env.get_page_text()
        except Exception:
            agent_response = ""

        eval_outcome = self._llm_evaluator.evaluate_task(
            task, dom_result=final_result, agent_response=agent_response,
        )

        env.stop()
        audio_proc.stop()

        # If every step failed at the API layer, the model never actually drove
        # the browser — this is an invalid run (e.g. wrong/unavailable model id),
        # not a legitimate 0. Surface it so downstream aggregation can exclude it.
        run_error = None
        if steps_log and api_failures >= len(steps_log):
            run_error = (f"INVALID: all {api_failures} model calls failed "
                         f"(model={self.model!r}); last error: {last_api_error}")

        return RealtimeEvalResult(
            task_id=task.task_id, category=task.category.value,
            difficulty=task.difficulty.value, model_name=self.model,
            observation_mode="openai_realtime",
            success=eval_outcome.final_passed,
            result_val=final_result,
            steps_taken=len(steps_log),
            total_time_s=time.time() - t_start,
            steps=steps_log,
            error=run_error,
            final_score=eval_outcome.final_score,
        )

    def _inject_page_audio(self, env, audio_proc):
        """Same TTS injection logic as GeminiLiveBaseline."""
        return GeminiLiveBaseline._inject_page_audio(self, env, audio_proc)


# ──────────────────────────────────────────────────────────────────────
# OpenAI Realtime GA baseline (true websocket, native audio)
# ──────────────────────────────────────────────────────────────────────

def _f32_to_pcm16_24k(audio_f32: "np.ndarray", src_rate: int = 16000) -> bytes:
    """Resample float32 [-1,1] mono audio to 24 kHz and encode as PCM16 LE."""
    if audio_f32 is None or len(audio_f32) == 0:
        return b""
    if src_rate != 24000:
        n_out = int(round(len(audio_f32) * 24000 / src_rate))
        if n_out <= 0:
            return b""
        x_old = np.linspace(0.0, 1.0, num=len(audio_f32), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        audio_f32 = np.interp(x_new, x_old, audio_f32).astype(np.float32)
    clipped = np.clip(audio_f32, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


class OpenAIRealtimeWSBaseline:
    """OpenAI Realtime GA API (gpt-realtime / gpt-realtime-2) as a CU agent.

    Unlike OpenAIRealtimeBaseline (which routed a transcript through
    chat-completions), this connects to the real Realtime websocket and streams
    the page's *native audio* (PCM16 @ 24 kHz) plus periodic screenshots, so the
    model hears the page exactly as the streaming-voice product would. One
    websocket session per task; we drive turns manually (server VAD disabled) and
    execute the returned function calls in the browser.

    API call failures and server `error` events are recorded (never silently
    degraded to wait()), so a misconfigured run is flagged INVALID rather than
    scored as a real 0.
    """

    def __init__(self, model: str = "gpt-realtime-2",
                 max_steps: int = 15, step_interval_s: float = 2.0,
                 provide_page_elements: bool = True,
                 ws_base: str = "wss://api.openai.com/v1/realtime",
                 api_key_env: str = "OPENAI_API_KEY",
                 send_images: bool = True):
        self.model = model
        self.max_steps = max_steps
        self.step_interval_s = step_interval_s
        # When True, hand the model the same [PAGE ELEMENTS] id list the AOI
        # scaffold provides (isolates audio perception). When False, screenshot
        # only — matches the original streaming baselines (model alone).
        self.provide_page_elements = provide_page_elements
        # Provider-agnostic: OpenAI GA Realtime and xAI Grok Voice both speak the
        # GA Realtime protocol. ws_base/api_key_env/send_images switch providers.
        # Grok Voice accepts no image input, so send_images=False there.
        self.ws_base = ws_base
        self.api_key_env = api_key_env
        self.send_images = send_images
        self._llm_evaluator = LLMEvaluator()

    def _ga_tools(self):
        # GA flattens the chat-completions {function:{...}} wrapper.
        return [
            {"type": "function", "name": t["name"],
             "description": t.get("description", ""),
             "parameters": t.get("parameters") or {"type": "object", "properties": {}}}
            for t in ACTION_TOOLS
        ]

    def run_task(self, task: Task) -> RealtimeEvalResult:
        import websocket  # websocket-client (sync), avoids asyncio/greenlet clashes

        url = f"{self.ws_base}?model={self.model}"
        key = os.environ[self.api_key_env]

        env = BrowserEnvironment(
            html_file=task.html_file, width=1280, height=720,
            task_timeout_s=task.duration_s + 30,
        )
        env.start()
        time.sleep(1.0)

        audio_proc = AudioProcessor(layer1_duration_s=self.step_interval_s,
                                    layer2_duration_s=15.0, silence_threshold=0.01)
        audio_proc.start()
        try:
            self._inject_page_audio(env, audio_proc)
        except Exception as e:
            log.warning("Audio injection failed: %s", e)

        steps_log = []
        t_start = time.time()
        last_result = "pending"
        api_failures = 0
        last_api_error = None
        heard = []  # transcripts of page audio the model received, for logging

        ws = None
        last_grab = None  # wall-time of previous audio capture, so no audio is dropped
        try:
            ws = websocket.create_connection(
                url, header=[f"Authorization: Bearer {key}"], timeout=30,
            )
            self._send(ws, {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": SYSTEM_PROMPT + f"\n\nTASK: {task.instruction}",
                    "tools": self._ga_tools(),
                    "tool_choice": "required",
                    "output_modalities": ["text"],
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {"model": "whisper-1"},
                            "turn_detection": None,  # we drive turns manually
                        }
                    },
                },
            })
        except Exception as e:
            api_failures += 1
            last_api_error = f"connect/session.update failed: {e}"
            log.error("Realtime connect failed: %s", e)

        for step in range(1, self.max_steps + 1):
            if env.get_elapsed_s() > task.duration_s or ws is None:
                break
            time.sleep(self.step_interval_s)
            if last_grab is None:
                last_grab = t_start

            # 1. Native audio since the last grab → append + commit (no audio dropped
            #    during response latency). Window capped at the ring-buffer length.
            try:
                window = min(time.time() - last_grab, audio_proc.capture.buffer_duration_s)
                last_grab = time.time()
                audio_f32, _, _ = audio_proc.capture.get_audio(window)
                pcm = _f32_to_pcm16_24k(audio_f32, src_rate=audio_proc.capture.sample_rate)
                if len(pcm) >= 24000 * 2 * 0.12:  # >=120 ms required to commit
                    self._send(ws, {"type": "input_audio_buffer.append",
                                    "audio": base64.b64encode(pcm).decode()})
                    self._send(ws, {"type": "input_audio_buffer.commit"})
            except Exception as e:
                log.warning("audio append failed step %d: %s", step, e)

            # 2. Screenshot (if the provider accepts vision) + instruction as a user turn
            try:
                content = []
                # Give the baseline the same interactive-element id list the AOI
                # scaffold provides, so failures reflect perception, not selector
                # guessing (keeps the comparison favourable to the baseline).
                try:
                    page_elems = env.get_interactive_elements() if self.provide_page_elements else ""
                except Exception:
                    page_elems = ""
                step_text = f"Step {step}. Choose exactly one tool call to drive the browser."
                if page_elems:
                    step_text += "\n" + page_elems
                content.append({"type": "input_text", "text": step_text})
                if self.send_images:  # Grok Voice accepts audio+text only
                    img = env.get_screenshot()
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=70)
                    img_b64 = base64.b64encode(buf.getvalue()).decode()
                    content.append({"type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{img_b64}"})
                self._send(ws, {
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": content},
                })
                self._send(ws, {"type": "response.create"})
            except Exception as e:
                api_failures += 1
                last_api_error = str(e)
                log.warning("response.create failed step %d: %s", step, e)
                steps_log.append(RealtimeStepLog(
                    step=step, action="(error)", tool_call=f"ERROR: {str(e)[:120]}",
                    result_val=last_result, elapsed_s=time.time() - t_start))
                continue

            # 3. Drain events until this response completes; capture the tool call
            tool_name, tool_args, step_err = self._await_tool_call(ws, heard)
            if step_err:
                api_failures += 1
                last_api_error = step_err
                log.warning("Realtime step %d error: %s", step, step_err[:160])
                steps_log.append(RealtimeStepLog(
                    step=step, action="(error)", tool_call=f"ERROR: {step_err[:120]}",
                    result_val=last_result, elapsed_s=time.time() - t_start))
                continue

            action = tool_call_to_action(tool_name, tool_args) if tool_name else "wait()"
            try:
                env.execute_action(action)
            except Exception as e:
                log.warning("execute action failed: %s", e)
            time.sleep(0.3)
            try:
                _, result_val = env.check_success()
            except Exception:
                result_val = last_result
            last_result = result_val

            steps_log.append(RealtimeStepLog(
                step=step, action=action[:200],
                tool_call=f"{tool_name}({json.dumps(tool_args, default=str)[:120]})" if tool_name else "(none)",
                result_val=result_val, elapsed_s=time.time() - t_start))
            log.info("[%s rt-ws %s] step %d: %s -> %s",
                     task.task_id, self.model, step, action[:60], result_val)

            if tool_name == "done" or action == "done":
                break

        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

        try:
            _, final_result = env.check_success()
        except Exception:
            final_result = last_result
        try:
            agent_response = env.get_page_text()
        except Exception:
            agent_response = ""

        eval_outcome = self._llm_evaluator.evaluate_task(
            task, dom_result=final_result, agent_response=agent_response)

        env.stop()
        audio_proc.stop()

        run_error = None
        if not steps_log or (api_failures >= max(len(steps_log), 1)):
            run_error = (f"INVALID: {api_failures} model-call failures "
                         f"(model={self.model!r}); last error: {last_api_error}")

        result = RealtimeEvalResult(
            task_id=task.task_id, category=task.category.value,
            difficulty=task.difficulty.value, model_name=self.model,
            observation_mode="openai_realtime_ws",
            success=eval_outcome.final_passed, result_val=final_result,
            steps_taken=len(steps_log), total_time_s=time.time() - t_start,
            steps=steps_log, error=run_error, final_score=eval_outcome.final_score)
        # Stash what the model heard for the appendix/debugging.
        result.heard_audio = " | ".join(h for h in heard if h)[:2000]
        return result

    def _inject_page_audio(self, env, audio_proc):
        """Same TTS injection as the other baselines (edge-tts → virtual_speaker)."""
        return GeminiLiveBaseline._inject_page_audio(self, env, audio_proc)

    @staticmethod
    def _send(ws, ev):
        ws.send(json.dumps(ev))

    def _await_tool_call(self, ws, heard):
        """Read server events until response.done; return (name, args, error)."""
        name, args_str = None, ""
        deadline = time.time() + 45
        while time.time() < deadline:
            try:
                raw = ws.recv()
            except Exception as e:
                return None, {}, f"recv failed: {e}"
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            t = ev.get("type", "")
            if t == "error":
                return None, {}, "server error: " + json.dumps(ev.get("error", ev))[:300]
            if t == "conversation.item.input_audio_transcription.completed":
                heard.append(ev.get("transcript", ""))
            elif t == "response.function_call_arguments.done":
                name = ev.get("name") or name
                args_str = ev.get("arguments") or args_str
            elif t == "response.output_item.done":
                item = ev.get("item", {})
                if item.get("type") == "function_call":
                    name = item.get("name") or name
                    args_str = item.get("arguments") or args_str
            elif t == "response.done":
                break
        try:
            args = json.loads(args_str) if args_str else {}
        except Exception:
            args = {}
        return name, args, None
