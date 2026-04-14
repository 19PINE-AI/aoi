"""
Real Browser Environment for DynaCU-Bench.

Uses Playwright to:
1. Serve HTML task pages in a real headless Chromium browser
2. Capture real screenshots (what the agent actually sees)
3. Execute CU model actions (click, type, key, navigate)
4. Verify success by querying DOM state — NOT string matching

This replaces the synthetic PIL-image environment with a real browser.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright, Page, Browser

logger = logging.getLogger(__name__)

HTML_TASKS_DIR = Path(__file__).parent / "html_tasks"


@dataclass
class ActionResult:
    success: bool
    action_text: str
    error: Optional[str] = None


class BrowserEnvironment:
    """
    A real Chromium browser environment for one benchmark task.

    Lifecycle:
        env = BrowserEnvironment("C001_cookie_consent.html")
        env.start()
        screenshot = env.get_screenshot()       # PIL Image
        env.execute_action("click 640 600")     # Click the accept button
        result = env.check_success()            # Query DOM state
        env.stop()
    """

    def __init__(
        self,
        html_file: str,
        width: int = 1280,
        height: int = 720,
        task_timeout_s: float = 30.0,
        audio_enabled: bool = True,
        coord_scale_1000: bool = False,
    ):
        self.html_file = html_file
        self.width = width
        self.height = height
        self.task_timeout_s = task_timeout_s
        self.audio_enabled = audio_enabled
        self.coord_scale_1000 = coord_scale_1000

        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        self._task_start_time: Optional[float] = None

        # Audio capture
        self._audio_frames: list[np.ndarray] = []
        self._audio_lock = threading.Lock()

        # Audio pipeline (set externally by evaluation harness)
        self._audio_processor = None

        # Action history
        self._actions: list[ActionResult] = []

    def start(self) -> bool:
        """Start the browser and load the task page."""
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--autoplay-policy=no-user-gesture-required",
                    "--disable-web-security",
                    "--allow-file-access-from-files",
                ],
            )
            context = self._browser.new_context(
                viewport={"width": self.width, "height": self.height},
                permissions=["microphone"] if self.audio_enabled else [],
            )
            self._page = context.new_page()

            # Monkey-patch SpeechSynthesis to capture utterance text
            # (headless Chromium has no TTS voices, so we intercept and
            # replay via PulseAudio in the evaluator)
            self._page.add_init_script("""
                window.__capturedUtterances = [];
                const _origSpeak = speechSynthesis.speak.bind(speechSynthesis);
                speechSynthesis.speak = function(utterance) {
                    if (utterance && utterance.text) {
                        window.__capturedUtterances.push(utterance.text);
                    }
                    // Fire onend callback quickly so sequential utterances chain
                    // fast — we only need to capture text, not actually play audio
                    if (utterance && utterance.onend) {
                        setTimeout(() => utterance.onend(new Event('end')), 50);
                    }
                };
            """)

            # Load the HTML task file
            html_path = HTML_TASKS_DIR / self.html_file
            if html_path.exists():
                self._page.goto(f"file://{html_path.absolute()}")
            else:
                raise FileNotFoundError(f"Task HTML not found: {html_path}")

            self._task_start_time = time.time()
            logger.info("Browser environment started: %s", self.html_file)
            return True

        except Exception as e:
            logger.error("Failed to start browser: %s", e)
            return False

    def stop(self):
        """Stop the browser."""
        if self._page:
            self._page.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        logger.info("Browser environment stopped")

    def get_screenshot(self) -> Image.Image:
        """Capture and return the current browser state as a PIL Image."""
        try:
            png_bytes = self._page.screenshot(type="png", full_page=False)
            return Image.open(io.BytesIO(png_bytes)).convert("RGB")
        except Exception as e:
            logger.warning("Screenshot failed: %s", e)
            return Image.new("RGB", (self.width, self.height), (200, 200, 200))

    def get_elapsed_s(self) -> float:
        if self._task_start_time is None:
            return 0.0
        return time.time() - self._task_start_time

    def get_page_text(self) -> str:
        """Get visible text content of the page."""
        try:
            return self._page.evaluate("document.body.innerText") or ""
        except:
            return ""

    def get_dom_result(self) -> dict:
        """
        Query the page for task result state.
        Returns dict with 'result' key set by the page's JavaScript.
        """
        try:
            result = self._page.evaluate("window.getTaskResult ? window.getTaskResult() : 'unknown'")
            return {"result": result}
        except Exception as e:
            return {"result": "error", "error": str(e)}

    def get_interactive_elements(self) -> str:
        """
        Extract visible interactive elements from the page with their IDs,
        types, labels, and bounding boxes. This provides DOM context to help
        the agent know what inputs/buttons exist and how to target them.
        """
        try:
            elements = self._page.evaluate('''() => {
                const results = [];
                const selectors = 'input, textarea, button, select, [contenteditable="true"], a[href]';
                const els = document.querySelectorAll(selectors);
                for (const el of els) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) continue;
                    if (window.getComputedStyle(el).display === 'none') continue;
                    if (window.getComputedStyle(el).visibility === 'hidden') continue;

                    const info = {
                        tag: el.tagName.toLowerCase(),
                        type: el.type || '',
                        id: el.id || '',
                        name: el.name || '',
                        placeholder: el.placeholder || '',
                        value: (el.value || '').substring(0, 50),
                        text: (el.textContent || '').trim().substring(0, 50),
                        x: Math.round(rect.x + rect.width / 2),
                        y: Math.round(rect.y + rect.height / 2),
                    };

                    // Get associated label
                    if (el.id) {
                        const label = document.querySelector('label[for="' + el.id + '"]');
                        if (label) info.label = label.textContent.trim().substring(0, 50);
                    }

                    results.push(info);
                }
                return results;
            }''')

            if not elements:
                return ""

            lines = ["[PAGE ELEMENTS — interactive]"]
            for el in elements:
                parts = [el['tag']]
                if el.get('type'):
                    parts.append(f"type={el['type']}")
                if el.get('id'):
                    parts.append(f"id=\"{el['id']}\"")
                if el.get('label'):
                    parts.append(f"label=\"{el['label']}\"")
                elif el.get('placeholder'):
                    parts.append(f"placeholder=\"{el['placeholder']}\"")
                elif el.get('text') and el['tag'] == 'button':
                    parts.append(f"text=\"{el['text']}\"")
                if el.get('value'):
                    parts.append(f"value=\"{el['value']}\"")
                pos = f"at ({el['x']},{el['y']})"
                lines.append(f"  {' '.join(parts)} {pos}")

            return "\n".join(lines)

        except Exception as e:
            logger.debug("get_interactive_elements failed: %s", e)
            return ""

    def execute_action(self, action_text: str) -> ActionResult:
        """
        Execute a CU model action string.

        Supported action formats:
          click <x> <y>                 — mouse click at coordinates
          click_element <selector>      — click CSS selector
          type <text>                   — type text (into focused element)
          fill <selector> <text>        — fill input field
          key <combo>                   — keyboard shortcut (e.g. 'Enter', 'ctrl+a')
          navigate <url>                — navigate browser to URL
          scroll <direction> <amount>   — scroll page
          wait <seconds>                — pause
          speak <text>                  — TTS audio output via virtual microphone
          js <code>                     — run JavaScript on page (for testing)
        """
        try:
            action = action_text.strip()
            action_lower = action.lower()

            # --- Compound actions: semicolons, "and click", "then type" ---
            # Split on ";" or " and then " / " then " / " and " + action verb
            if ';' in action:
                parts = [p.strip() for p in action.split(';') if p.strip()]
                if len(parts) >= 2:
                    results = []
                    for p in parts:
                        r = self.execute_action(p)
                        results.append(r)
                        time.sleep(0.3)
                    return ActionResult(
                        any(r.success for r in results),
                        " → ".join(p.strip()[:30] for p in parts),
                    )
            compound_m = re.search(
                r'(.+?)\s+(?:and\s+then|then|and)\s+((?:click|type|enter|fill|submit)\s.+)$',
                action, re.IGNORECASE,
            )
            if compound_m:
                first_part = compound_m.group(1).strip()
                second_part = compound_m.group(2).strip()
                r1 = self.execute_action(first_part)
                time.sleep(0.3)
                r2 = self.execute_action(second_part)
                return ActionResult(
                    r1.success or r2.success,
                    f"{first_part} → {second_part}",
                )

            # --- speak "text" / speak("text") — TTS → virtual microphone injection ---
            m = re.match(r'speak\s*\(\s*["\'](.+?)["\']\s*\)', action, re.IGNORECASE)
            if not m:
                m = re.match(r'speak\s+["\'](.+?)["\']', action, re.IGNORECASE)
            if not m:
                m = re.match(r'speak\s+(.+)', action, re.IGNORECASE)
            if m and 'speak' in action_lower[:10]:
                text = m.group(1).strip().strip("'\"")
                if self._audio_processor:
                    success = self._audio_processor.speak(text)
                    return ActionResult(success, f"speak '{text[:50]}'")
                else:
                    logger.warning("speak action called but no audio_processor attached")
                    return ActionResult(False, action_text, error="No audio processor")

            # --- type_text_at(x, y, "text") — click at coords, then type ---
            m = re.match(r'type_text_at\s*\(\s*(\d+)[,\s]+(\d+)[,\s]+["\']([^"\']+)["\']', action, re.IGNORECASE)
            if m:
                x, y = int(m.group(1)), int(m.group(2))
                text = m.group(3)
                if self.coord_scale_1000:
                    x = int(x * self.width / 1000)
                    y = int(y * self.height / 1000)
                self._page.mouse.click(x, y)
                time.sleep(0.2)
                self._page.keyboard.type(text)
                return ActionResult(True, f"type_text_at({x},{y},'{text}')")

            # --- triple_click / double_click at x, y ---
            m = re.match(r'(triple|double)_?click\s*\(?x?(\d+)[,\s]+y?(\d+)\)?', action_lower)
            if m:
                click_type, x, y = m.group(1), int(m.group(2)), int(m.group(3))
                if self.coord_scale_1000:
                    x = int(x * self.width / 1000)
                    y = int(y * self.height / 1000)
                count = 3 if click_type == 'triple' else 2
                self._page.mouse.click(x, y, click_count=count)
                return ActionResult(True, action_text)

            # --- click x y / click(x, y) / click(x,y) / click x500 y610 ---
            # Also handles Gemini 3 format: click(point=[611, 452])
            m = re.match(r'click\s*\(?x?(\d+)[,\s]+y?(\d+)\)?', action_lower)
            if not m:
                m = re.search(r'click\s*\(?\s*point\s*=\s*\[(\d+)[,\s]+(\d+)\]', action_lower)
            if not m:
                # JSON-style: click {"point": [x, y]}
                m = re.search(r'click\s*\{[^}]*"point"\s*:\s*\[(\d+)[,\s]+(\d+)\]', action_lower)
            if not m:
                # Fallback: any "click" followed by two numbers somewhere in text
                m = re.search(r'click[^0-9]*(\d{2,4})[,\s]+(\d{2,4})', action_lower)
            if m:
                x, y = int(m.group(1)), int(m.group(2))
                # Scale normalized coordinates (0-1000) to viewport pixels.
                # Gemini 3 Flash outputs coordinates in a 1000x1000 normalized
                # space. The flag coord_scale_1000 is set by the evaluator.
                if self.coord_scale_1000:
                    x = int(x * self.width / 1000)
                    y = int(y * self.height / 1000)
                self._page.mouse.click(x, y)
                return ActionResult(True, action_text)

            # --- click_element CSS selector ---
            m = re.match(r'click_element\s+(.+)', action, re.IGNORECASE)
            if m:
                sel = m.group(1).strip()
                self._page.click(sel, timeout=3000)
                return ActionResult(True, action_text)

            # --- fill selector text / fill(selector, "text") ---
            m = re.match(r'fill\s*\(\s*(\S+?)\s*,\s*(.+?)\s*\)', action, re.IGNORECASE)
            if not m:
                m = re.match(r'fill\s+(\S+)\s+(.+)', action, re.IGNORECASE)
            if m:
                sel, text = m.group(1), m.group(2).strip().strip("'\"")
                self._page.fill(sel, text)
                return ActionResult(True, action_text)

            # --- type "X" into Y / type("X") / enter "X" ---
            # Handles both: type "text" and type("text")
            m = re.match(r'(?:type|enter|input)\s*\(\s*["\']([^"\']+)["\']\s*(?:,\s*[^)]+)?\s*\)', action, re.IGNORECASE)
            if not m:
                m = re.match(r'(?:type|enter|input)\s+["\']([^"\']+)["\']', action, re.IGNORECASE)
            if m:
                text = m.group(1).strip()
                # If a contenteditable element is focused (agent clicked it), type there
                try:
                    focused_ce = self._page.evaluate(
                        "document.activeElement?.getAttribute('contenteditable') === 'true'")
                    if focused_ce:
                        self._page.keyboard.type(text)
                        return ActionResult(True, f"typed into focused contenteditable: '{text}'")
                except:
                    pass
                # Find the first visible text input and fill it
                for sel in ['input[type="text"]:visible', 'input:not([type]):visible',
                            'textarea:visible', 'input[type="text"]', 'textarea', 'input']:
                    try:
                        loc = self._page.locator(sel).first
                        if loc.is_visible():
                            loc.fill(text)
                            return ActionResult(True, f"filled input with '{text}'")
                    except:
                        continue
                # Try contenteditable elements (e.g. Google Docs-like editors)
                try:
                    ce = self._page.locator('[contenteditable="true"]:visible').first
                    if ce.is_visible():
                        ce.click()
                        time.sleep(0.1)
                        self._page.keyboard.type(text)
                        return ActionResult(True, f"typed into contenteditable: '{text}'")
                except:
                    pass
                # Fallback: focus first input, then type
                try:
                    self._page.locator('input').first.focus()
                    self._page.keyboard.type(text)
                    return ActionResult(True, f"typed '{text}'")
                except:
                    pass

            # --- type text (simple, no quotes) ---
            m = re.match(r'type\s+(.+?)(?:\s+(?:into|in|on)\s+.+)?$', action, re.IGNORECASE)
            if m:
                text = m.group(1).strip().strip("'\"")
                if text and len(text) < 200:
                    # If a contenteditable element is focused, type there
                    try:
                        focused_ce = self._page.evaluate(
                            "document.activeElement?.getAttribute('contenteditable') === 'true'")
                        if focused_ce:
                            self._page.keyboard.type(text)
                            return ActionResult(True, f"typed into focused contenteditable: '{text}'")
                    except:
                        pass
                    # Try to fill the first visible input
                    try:
                        inp = self._page.locator('input[type="text"]:visible, textarea:visible').first
                        inp.fill(text)
                        return ActionResult(True, f"filled '{text}'")
                    except:
                        self._page.keyboard.type(text)
                        return ActionResult(True, action_text)

            # --- key combo ---
            m = re.match(r'key[\s(]+(.+?)[\s)]*$', action, re.IGNORECASE)
            if m:
                combo = m.group(1).strip()
                self._page.keyboard.press(combo)
                return ActionResult(True, action_text)

            # --- navigate url ---
            m = re.match(r'(?:navigate|open_url|goto)\s+["\']?(.+?)["\']?\s*$', action, re.IGNORECASE)
            if m:
                url = m.group(1).strip()
                if not url.startswith('http'):
                    url = 'https://' + url
                # For evaluation, record navigation intent without actually leaving the task page
                self._page.evaluate(
                    f"document.getElementById('url-input') && (document.getElementById('url-input').value = '{url}')"
                )
                return ActionResult(True, action_text)

            # --- wait N / wait(N) / wait() (standalone only) ---
            m = re.match(r'wait\s*\(?\s*(\d+(?:\.\d+)?)?\s*\)?\s*$', action_lower)
            if m:
                if m.group(1):
                    val = float(m.group(1))
                    secs = val / 1000.0 if val > 100 else val
                else:
                    secs = 0.5
                time.sleep(min(secs, 2.0))  # Cap at 2s per wait
                return ActionResult(True, action_text)

            # --- js code ---
            m = re.match(r'js\s+(.+)', action, re.IGNORECASE)
            if m:
                code = m.group(1)
                result = self._page.evaluate(code)
                return ActionResult(True, f"js: {result}")

            # --- scroll ---
            m = re.match(r'scroll\s+(up|down)\s+(\d+)', action_lower)
            if m:
                direction = m.group(1)
                amount = int(m.group(2))
                delta = -amount if direction == 'up' else amount
                self._page.mouse.wheel(0, delta)
                return ActionResult(True, action_text)

            # --- click "button text" / click button "text" ---
            # Handle natural language: Click "Accept All", click the "Shop Now" button, etc.
            m = re.search(r'["\']([^"\']+)["\']', action)
            if m and "click" in action_lower:
                btn_text = m.group(1)
                for role in ["button", "link"]:
                    try:
                        loc = self._page.get_by_role(role, name=re.compile(
                            re.escape(btn_text), re.IGNORECASE))
                        if loc.count() > 0:
                            loc.first.click()
                            return ActionResult(True, f"clicked {role} '{btn_text}'")
                    except:
                        pass
                # Also try get_by_text as a fallback
                try:
                    loc = self._page.get_by_text(re.compile(
                        re.escape(btn_text), re.IGNORECASE))
                    if loc.count() > 0:
                        loc.first.click()
                        return ActionResult(True, f"clicked text '{btn_text}'")
                except:
                    pass

            # --- Do nothing / wait / observe / pause / narration actions ---
            # Must come BEFORE keyword fallback to prevent "wait for X then click Capture"
            # from triggering the "capture" keyword.
            if any(w in action_lower for w in ["do nothing", "observe", "no action",
                                                "pause", "narration:"]):
                time.sleep(0.5)
                return ActionResult(True, action_text)
            if re.match(r'(?:wait|i.ll wait|i need to wait|since|i can see|i will|i am|no)\s+', action_lower):
                time.sleep(0.5)
                return ActionResult(True, action_text)
            if action_lower in ("wait", "wait.", "wait()", "no"):
                time.sleep(0.5)
                return ActionResult(True, action_text)

            # --- Generic keyword fallback ---
            keywords = ['accept', 'open', 'submit', 'install', 'approve',
                        'dismiss', 'close', 'launch', 'capture', 'alert',
                        'record', 'stay', 'navigate', 'shop', 'confirm',
                        'acknowledge', 'download', 'start']
            for kw in keywords:
                if kw in action_lower:
                    for role in ["button", "link"]:
                        try:
                            locator = self._page.get_by_role(
                                role, name=re.compile(kw, re.IGNORECASE))
                            if locator.count() > 0:
                                locator.first.click()
                                return ActionResult(True, f"clicked {role} matching '{kw}'")
                        except:
                            pass

            return ActionResult(False, action_text, error=f"Unrecognized action: {action[:80]}")

        except Exception as e:
            logger.warning("Action failed (%s): %s", action_text[:50], e)
            return ActionResult(False, action_text, error=str(e))

    def check_success(self, expected_result: str | None = None) -> tuple[bool, str]:
        """
        Verify task success by querying DOM state via window.getTaskResult().
        Returns (success: bool, result_value: str).

        If expected_result is provided, checks for exact match.
        Otherwise, treats any non-pending/non-error value as success.
        """
        dom_result = self.get_dom_result()
        result_val = dom_result.get("result", "unknown")

        # Known failure / incomplete values
        PENDING = {"pending", "unknown", "error", "timeout", "alarm_missed",
                   "session_expired", "wrong_event", "wrong_url", "wrong_word",
                   "wrong_step", "wrong_pair", "premature_alert", "game_over",
                   "fell"}

        if expected_result:
            success = result_val == expected_result
        else:
            success = result_val not in PENDING and not result_val.startswith("wrong_")
        return success, result_val

    def capture_audio_chunk(
        self,
        duration_s: float = 3.5,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Capture audio from virtual PulseAudio sink.
        Returns float32 numpy array of audio samples.
        """
        try:
            import subprocess
            result = subprocess.run(
                [
                    "parecord",
                    "--format=float32le",
                    f"--rate={sample_rate}",
                    "--channels=1",
                    f"--device=virtual_speaker.monitor",
                    f"--duration={duration_s}",
                ],
                capture_output=True,
                timeout=duration_s + 2,
            )
            if result.returncode == 0 and result.stdout:
                return np.frombuffer(result.stdout, dtype=np.float32)
        except Exception as e:
            logger.debug("Audio capture failed: %s", e)
        return np.zeros(int(duration_s * sample_rate), dtype=np.float32)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
