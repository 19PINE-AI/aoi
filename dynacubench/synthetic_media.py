"""
Synthetic Media Generator — creates reproducible test stimuli for DynaCU-Bench.

Generates:
- Slideshow videos (for Category A and E tasks)
- Audio speech (TTS for Category B tasks)
- Notification tones (for Category D tasks)
- Transient UI frames (for Category C tasks) served as timed image sequences
"""

from __future__ import annotations

import math
import struct
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SyntheticMediaGenerator:
    """Generates synthetic video and audio stimuli for headless testing."""

    def __init__(self, output_dir: Path = Path("media")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    # Video generation
    # ─────────────────────────────────────────────────────────────

    def create_slideshow_frames(
        self,
        slides: list[dict],
        fps: float = 3.0,
        width: int = 1280,
        height: int = 720,
    ) -> list[tuple[float, Image.Image]]:
        """
        Create a list of (timestamp, frame) pairs for a slideshow.

        Args:
            slides: List of dicts with keys: 'text', 'duration_s', 'bg_color'
            fps: Frames per second
            width, height: Frame dimensions

        Returns:
            List of (timestamp, PIL.Image) pairs
        """
        frames = []
        t = 0.0

        for i, slide in enumerate(slides):
            duration_s = slide.get("duration_s", 5.0)
            bg_color = slide.get("bg_color", (240, 240, 255))
            text = slide.get("text", f"Slide {i + 1}")
            sub_text = slide.get("sub_text", "")
            n_frames = max(1, int(duration_s * fps))

            for _ in range(n_frames):
                frame = self._render_slide(text, sub_text, width, height, bg_color, slide_num=i + 1)
                frames.append((t, frame))
                t += 1.0 / fps

        return frames

    def _render_slide(
        self,
        title: str,
        sub_text: str,
        width: int,
        height: int,
        bg_color: tuple,
        slide_num: int = 1,
    ) -> Image.Image:
        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        # Title
        draw.rectangle([40, 40, width - 40, 140], fill=(255, 255, 255), outline=(100, 100, 200), width=2)
        draw.text((width // 2, 90), title, fill=(30, 30, 120), anchor="mm")

        # Sub-text
        if sub_text:
            draw.text((width // 2, height // 2), sub_text, fill=(60, 60, 60), anchor="mm")

        # Slide number indicator
        draw.text((width - 60, height - 40), f"Slide {slide_num}", fill=(150, 150, 150))

        return img

    def create_product_demo_video(self, product_name: str = "CloudSync Pro", duration_s: float = 15.0) -> list[tuple[float, Image.Image]]:
        """Category A-001: Product demo video with product name on screen."""
        slides = [
            {"text": "Product Demo", "duration_s": 3.0, "bg_color": (220, 235, 255)},
            {"text": product_name, "sub_text": "The future of cloud sync", "duration_s": 5.0, "bg_color": (200, 255, 220)},
            {"text": "Key Features", "sub_text": "Fast • Secure • Reliable", "duration_s": 4.0, "bg_color": (255, 240, 200)},
            {"text": "Thank you!", "duration_s": 3.0, "bg_color": (240, 220, 255)},
        ]
        return self.create_slideshow_frames(slides)

    def create_counting_slideshow(self, n_slides: int = 5) -> list[tuple[float, Image.Image]]:
        """Category A-002: Numbered slides for counting task."""
        slides = [
            {"text": f"Slide {i + 1} of {n_slides}", "sub_text": f"Section {i + 1}", "duration_s": 5.0,
             "bg_color": (200 + i * 10, 220, 240 - i * 5)}
            for i in range(n_slides)
        ]
        return self.create_slideshow_frames(slides)

    def create_transient_ui_sequence(
        self,
        base_frame: Image.Image,
        popup_frames: list[tuple[float, float, Image.Image]],  # (appear_at, dismiss_at, popup_image)
        total_duration_s: float = 10.0,
        fps: float = 3.0,
    ) -> list[tuple[float, Image.Image]]:
        """
        Create a sequence of frames where popups appear and disappear.
        Simulates transient UI events (Category C).
        """
        frames = []
        n_frames = int(total_duration_s * fps)

        for i in range(n_frames):
            t = i / fps
            frame = base_frame.copy()

            # Overlay any active popups
            for appear_at, dismiss_at, popup_img in popup_frames:
                if appear_at <= t < dismiss_at:
                    # Center the popup on the frame
                    pw, ph = popup_img.size
                    fw, fh = frame.size
                    x = (fw - pw) // 2
                    y = (fh - ph) // 2
                    frame.paste(popup_img, (x, y))

            frames.append((t, frame))

        return frames

    def create_popup(
        self,
        text: str,
        button_text: str = "Accept",
        width: int = 400,
        height: int = 200,
        bg_color: tuple = (255, 250, 230),
    ) -> Image.Image:
        """Create a popup/dialog image."""
        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        # Border
        draw.rectangle([2, 2, width - 3, height - 3], outline=(100, 100, 100), width=2)

        # Text
        draw.text((width // 2, height // 3), text, fill=(30, 30, 30), anchor="mm")

        # Button
        btn_x = width // 2 - 60
        draw.rectangle([btn_x, height - 60, btn_x + 120, height - 20], fill=(70, 130, 200))
        draw.text((width // 2, height - 40), button_text, fill=(255, 255, 255), anchor="mm")

        return img

    # ─────────────────────────────────────────────────────────────
    # Audio generation
    # ─────────────────────────────────────────────────────────────

    def generate_tone(
        self,
        frequency: float,
        duration_s: float,
        sample_rate: int = 16000,
        amplitude: float = 0.3,
        fade_ms: float = 10.0,
    ) -> np.ndarray:
        """Generate a pure tone as numpy array (float32)."""
        n = int(duration_s * sample_rate)
        t = np.linspace(0, duration_s, n, endpoint=False)
        wave_data = amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Fade in/out to avoid clicks
        fade_samples = int(fade_ms * sample_rate / 1000)
        if fade_samples > 0 and 2 * fade_samples < n:
            fade = np.linspace(0, 1, fade_samples)
            wave_data[:fade_samples] *= fade
            wave_data[-fade_samples:] *= fade[::-1]

        return wave_data

    def generate_notification_ding(self, sample_rate: int = 16000) -> np.ndarray:
        """High-pitched notification ding (Category D: notification sound)."""
        # Two-tone ding: 880Hz then 1760Hz
        tone1 = self.generate_tone(880, 0.15, sample_rate, amplitude=0.4)
        silence = np.zeros(int(0.05 * sample_rate), dtype=np.float32)
        tone2 = self.generate_tone(1760, 0.2, sample_rate, amplitude=0.3)
        return np.concatenate([tone1, silence, tone2])

    def generate_calendar_alarm(self, sample_rate: int = 16000) -> np.ndarray:
        """Calendar alarm sound — repeating tones (Category D-001)."""
        single_beep = self.generate_tone(660, 0.3, sample_rate, amplitude=0.5)
        silence = np.zeros(int(0.15 * sample_rate), dtype=np.float32)
        return np.concatenate([single_beep, silence] * 3)

    def generate_error_beep(self, sample_rate: int = 16000) -> np.ndarray:
        """High-pitched error beep (critical alert)."""
        return self.generate_tone(2000, 0.5, sample_rate, amplitude=0.6)

    def generate_warning_chime(self, sample_rate: int = 16000) -> np.ndarray:
        """Low, gentle warning chime."""
        return self.generate_tone(440, 0.4, sample_rate, amplitude=0.3)

    def generate_tts_speech(self, text: str, sample_rate: int = 16000) -> np.ndarray:
        """
        Generate synthetic speech audio.
        Falls back to a simple frequency-modulated signal if TTS unavailable.
        In production, would use a real TTS engine.
        """
        try:
            # Try using pyttsx3 or gTTS if available
            return self._tts_with_gtts(text, sample_rate)
        except Exception:
            # Fallback: generate a plausible-duration silence + tone pattern
            return self._tts_fallback(text, sample_rate)

    def _tts_with_gtts(self, text: str, sample_rate: int) -> np.ndarray:
        """Use gTTS for text-to-speech."""
        from gtts import gTTS
        import io
        import soundfile as sf

        buf = io.BytesIO()
        tts = gTTS(text=text, lang="en", slow=False)
        tts.write_to_fp(buf)
        buf.seek(0)

        # gTTS outputs MP3, convert to WAV
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(buf.read())
            mp3_path = f.name

        wav_path = mp3_path.replace(".mp3", ".wav")
        subprocess.run(
            ["ffmpeg", "-i", mp3_path, "-ar", str(sample_rate), "-ac", "1", "-y", wav_path],
            capture_output=True,
        )

        import os
        os.unlink(mp3_path)

        data, _ = sf.read(wav_path)
        os.unlink(wav_path)
        return data.astype(np.float32)

    def _tts_fallback(self, text: str, sample_rate: int) -> np.ndarray:
        """
        Simple fallback: generate ~2.5 words/second of tone pattern
        (enough to test RMS gating and timing, not actual speech).
        """
        words = len(text.split())
        duration_s = max(1.0, words / 2.5)
        # Frequency-modulated signal resembling speech cadence
        t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
        # Fundamental at 150Hz (male voice) with harmonics
        signal = (
            0.2 * np.sin(2 * np.pi * 150 * t) +
            0.1 * np.sin(2 * np.pi * 300 * t) +
            0.05 * np.sin(2 * np.pi * 600 * t)
        ).astype(np.float32)

        # Simulate word-level pauses (every ~0.4 seconds, 50ms silence)
        pause_samples = int(0.05 * sample_rate)
        word_samples = int(0.4 * sample_rate)
        for i in range(0, len(signal) - word_samples, word_samples):
            if i + word_samples + pause_samples < len(signal):
                signal[i + word_samples:i + word_samples + pause_samples] = 0

        return signal

    def save_audio_wav(self, data: np.ndarray, path: Path, sample_rate: int = 16000):
        """Save numpy audio as WAV file."""
        import soundfile as sf
        sf.write(str(path), data, sample_rate)
        return path

    def create_audio_with_events(
        self,
        silence_duration_s: float,
        event_audio: np.ndarray,
        total_duration_s: float,
        event_at_s: float,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Create audio track with a specific event at a specified time."""
        total_samples = int(total_duration_s * sample_rate)
        audio = np.zeros(total_samples, dtype=np.float32)

        event_start = int(event_at_s * sample_rate)
        event_end = min(event_start + len(event_audio), total_samples)
        audio[event_start:event_end] = event_audio[:event_end - event_start]

        return audio

    # ─────────────────────────────────────────────────────────────
    # Task-specific generators
    # ─────────────────────────────────────────────────────────────

    def make_task_a001_stimulus(self) -> tuple[list, np.ndarray]:
        """A-001: Product demo video + silence."""
        frames = self.create_product_demo_video("CloudSync Pro", 15.0)
        audio = np.zeros(int(15.0 * 16000), dtype=np.float32)  # silent
        return frames, audio

    def make_task_b001_stimulus(self) -> tuple[list, np.ndarray]:
        """B-001: Meeting audio with spoken URL."""
        speech = self.generate_tts_speech(
            "Good morning everyone. Please check the full report at example dot com slash report. "
            "We'll be reviewing the Q3 results today."
        )
        # Pad with silence before and after
        pre_silence = np.zeros(int(3.0 * 16000), dtype=np.float32)
        post_silence = np.zeros(int(5.0 * 16000), dtype=np.float32)
        audio = np.concatenate([pre_silence, speech, post_silence])
        # Static base frame (looking at meeting interface)
        frames = self.create_slideshow_frames([
            {"text": "Team Meeting", "sub_text": "Recording in progress...", "duration_s": 20.0,
             "bg_color": (240, 240, 240)}
        ])
        return frames, audio

    def make_task_c001_stimulus(self) -> tuple[list, np.ndarray]:
        """C-001: Cookie consent popup appears at t=2s, auto-dismisses at t=6s."""
        base = Image.new("RGB", (1280, 720), (245, 245, 245))
        draw = ImageDraw.Draw(base)
        draw.text((640, 360), "Website Content Here", fill=(100, 100, 100), anchor="mm")

        popup = self.create_popup("This website uses cookies.\nDo you accept?", "Accept All")
        frames = self.create_transient_ui_sequence(
            base_frame=base,
            popup_frames=[(2.0, 6.0, popup)],
            total_duration_s=10.0,
        )
        audio = np.zeros(int(10.0 * 16000), dtype=np.float32)
        return frames, audio

    def make_task_d001_stimulus(self) -> tuple[list, np.ndarray]:
        """D-001: Calendar alarm sounds at t=10s."""
        alarm = self.generate_calendar_alarm()
        audio = self.create_audio_with_events(
            silence_duration_s=10.0,
            event_audio=alarm,
            total_duration_s=20.0,
            event_at_s=10.0,
        )
        # Static work environment frame
        frames = self.create_slideshow_frames([
            {"text": "Document Editor", "sub_text": "Working on quarterly report...",
             "duration_s": 20.0, "bg_color": (250, 250, 250)}
        ])
        return frames, audio

    def make_task_d003_stimulus(self) -> tuple[list, np.ndarray]:
        """D-003: High-pitched beep (critical error) at t=8s."""
        error_beep = self.generate_error_beep()
        audio = self.create_audio_with_events(
            silence_duration_s=8.0,
            event_audio=error_beep,
            total_duration_s=25.0,
            event_at_s=8.0,
        )
        frames = self.create_slideshow_frames([
            {"text": "System Dashboard", "sub_text": "All systems nominal",
             "duration_s": 25.0, "bg_color": (245, 255, 245)}
        ])
        return frames, audio
