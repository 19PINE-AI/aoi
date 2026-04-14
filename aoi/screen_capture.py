"""
Screen capture module — captures frames from the display at a target FPS.
Supports both headless (Xvfb + virtual display) and native environments.
"""

from __future__ import annotations

import threading
import time
import logging
from typing import Optional, Callable
from PIL import Image

logger = logging.getLogger(__name__)


class ScreenCapture:
    """
    Background screen capture thread. Calls on_frame(image, timestamp) for each frame.

    Supports:
    - mss: fast cross-platform screen capture (native displays)
    - PIL/ImageGrab: fallback
    - Synthetic frame injection: for testing without a real display
    """

    def __init__(
        self,
        fps: float = 3.0,
        on_frame: Optional[Callable[[Image.Image, float], None]] = None,
        monitor_index: int = 1,  # mss monitor index (1 = primary)
        resize: Optional[tuple[int, int]] = None,  # Resize captured frames
    ):
        self.fps = fps
        self.on_frame = on_frame
        self.monitor_index = monitor_index
        self.resize = resize
        self._interval = 1.0 / fps

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._synthetic_frames: list[tuple[Image.Image, float]] = []
        self._synthetic_lock = threading.Lock()
        self._use_synthetic = False

    def start(self) -> bool:
        """Start capture thread. Returns True if real capture is available."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return not self._use_synthetic

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def inject_frame(self, image: Image.Image, timestamp: Optional[float] = None):
        """Inject a synthetic frame (for testing)."""
        if timestamp is None:
            timestamp = time.time()
        with self._synthetic_lock:
            self._synthetic_frames.append((image, timestamp))

    def _capture_loop(self):
        try:
            import mss
            with mss.mss() as sct:
                monitor = sct.monitors[self.monitor_index] if self.monitor_index < len(sct.monitors) else sct.monitors[0]
                logger.info("Screen capture started: %dx%d @ %.1f fps", monitor["width"], monitor["height"], self.fps)
                while not self._stop_event.is_set():
                    t0 = time.time()
                    try:
                        screenshot = sct.grab(monitor)
                        frame = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                        if self.resize:
                            frame = frame.resize(self.resize, Image.BILINEAR)
                        if self.on_frame:
                            self.on_frame(frame, t0)
                    except Exception as e:
                        logger.debug("Frame capture error: %s", e)

                    elapsed = time.time() - t0
                    sleep_time = max(0.0, self._interval - elapsed)
                    time.sleep(sleep_time)
        except Exception as e:
            logger.warning("Real screen capture unavailable (%s), using synthetic mode", e)
            self._use_synthetic = True
            self._synthetic_loop()

    def _synthetic_loop(self):
        """Serve injected synthetic frames for headless testing."""
        logger.info("Screen capture: synthetic mode (headless environment)")
        while not self._stop_event.is_set():
            time.sleep(self._interval)
            with self._synthetic_lock:
                if self._synthetic_frames:
                    frame, ts = self._synthetic_frames.pop(0)
                    if self.on_frame:
                        self.on_frame(frame, ts)


def create_test_frame(width: int = 1280, height: int = 720, color: tuple = (200, 200, 200), text: str = "") -> Image.Image:
    """Create a simple test frame for headless testing."""
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGB", (width, height), color)
    if text:
        draw = ImageDraw.Draw(img)
        draw.text((width // 4, height // 2), text, fill=(50, 50, 50))
    return img
