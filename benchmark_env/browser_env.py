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
    ):
        self.html_file = html_file
        self.width = width
        self.height = height
        self.task_timeout_s = task_timeout_s
        self.audio_enabled = audio_enabled

        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        self._task_start_time: Optional[float] = None

        # Audio capture
        self._audio_frames: list[np.ndarray] = []
        self._audio_lock = threading.Lock()

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
          js <code>                     — run JavaScript on page (for testing)
        """
        try:
            action = action_text.strip()
            action_lower = action.lower()

            # --- click x y / click(x, y) / click(x,y) / click x500 y610 ---
            m = re.match(r'click\s*\(?x?(\d+)[,\s]+y?(\d+)\)?', action_lower)
            if m:
                x, y = int(m.group(1)), int(m.group(2))
                self._page.mouse.click(x, y)
                return ActionResult(True, action_text)

            # --- click_element CSS selector ---
            m = re.match(r'click_element\s+(.+)', action, re.IGNORECASE)
            if m:
                sel = m.group(1).strip()
                self._page.click(sel, timeout=3000)
                return ActionResult(True, action_text)

            # --- fill selector text ---
            m = re.match(r'fill\s+(\S+)\s+(.+)', action, re.IGNORECASE)
            if m:
                sel, text = m.group(1), m.group(2).strip().strip("'\"")
                self._page.fill(sel, text)
                return ActionResult(True, action_text)

            # --- type "X" into Y / type "X" in the input / enter "X" ---
            # Natural language: Type "Configuration" into the text input
            m = re.match(r'(?:type|enter|input)\s+["\']([^"\']+)["\']', action, re.IGNORECASE)
            if m:
                text = m.group(1).strip()
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
                if text and len(text) < 100:
                    # Try to fill the first visible input
                    try:
                        inp = self._page.locator('input[type="text"]:visible, textarea:visible').first
                        inp.fill(text)
                        return ActionResult(True, f"filled '{text}'")
                    except:
                        self._page.keyboard.type(text)
                        return ActionResult(True, action_text)

            # --- key combo ---
            m = re.match(r'key\s+(.+)', action, re.IGNORECASE)
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

            # --- wait (standalone only, not "wait for..." or "wait and...") ---
            m = re.match(r'wait(?:\s+(\d+(?:\.\d+)?))?\s*$', action_lower)
            if m:
                secs = float(m.group(1)) if m.group(1) else 0.5
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

            # --- Do nothing / wait / observe / pause / narration actions ---
            if any(w in action_lower for w in ["do nothing", "wait", "observe", "no action",
                                                "pause", "narration:"]):
                time.sleep(0.5)
                return ActionResult(True, action_text)

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
