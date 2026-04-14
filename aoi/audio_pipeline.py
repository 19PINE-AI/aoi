"""
Real Audio Pipeline for DynaCU-Bench v3.

Replaces all DOM-based audio proxies (window._spokenContent) with a real
PulseAudio capture → Whisper ASR pipeline, and provides TTS → mic injection
for the agent's `speak` action.

Architecture:
  Browser audio out → virtual_speaker sink → parecord → ring buffer → Whisper → text
  Agent speak text → edge-tts → WAV → pacat → virtual_mic sink → browser mic in

Two-layer audio representation:
  Layer 1: Recent 3-5s of audio synced with current keyframes
  Layer 2: Continuous 30-60s rolling transcript with sentence-level timestamps
"""

from __future__ import annotations

import asyncio
import io
import logging
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TimestampedSegment:
    """A transcribed segment with start/end timestamps."""
    text: str
    start_s: float   # seconds since capture started
    end_s: float
    is_recent: bool = False  # True if within the Layer 1 window


@dataclass
class TwoLayerAudio:
    """
    Two-layer audio representation for the observation record.

    Layer 1 (recent): Audio from the last 3-5 seconds, synced with keyframes.
                      Short, high-relevance text for immediate context.
    Layer 2 (context): Rolling 30-60s transcript with sentence timestamps.
                       Provides broader conversational/audio context.
    """
    layer1_text: str                          # Recent 3-5s transcript
    layer1_duration_s: float                  # How many seconds Layer 1 covers
    layer2_segments: list[TimestampedSegment]  # Full transcript with timestamps
    layer2_duration_s: float                  # How many seconds Layer 2 covers
    capture_end_time: float                   # Absolute time of capture end

    def format_for_prompt(self) -> str:
        """Format two-layer audio for inclusion in the agent prompt."""
        lines = []

        if self.layer1_text:
            lines.append(f"[AUDIO — recent {self.layer1_duration_s:.0f}s]")
            lines.append(f"  {self.layer1_text}")

        if self.layer2_segments:
            lines.append(f"[AUDIO CONTEXT — last {self.layer2_duration_s:.0f}s transcript]")
            for seg in self.layer2_segments:
                marker = " *" if seg.is_recent else ""
                lines.append(
                    f"  [{seg.start_s:.1f}s–{seg.end_s:.1f}s]{marker} {seg.text}"
                )

        return "\n".join(lines) if lines else ""

    @property
    def has_audio(self) -> bool:
        return bool(self.layer1_text) or bool(self.layer2_segments)


# ---------------------------------------------------------------------------
# PulseAudio device management
# ---------------------------------------------------------------------------

class PulseAudioManager:
    """Manages virtual PulseAudio sinks and sources for the benchmark."""

    SPEAKER_SINK = "virtual_speaker"
    MIC_SINK = "virtual_mic"
    MIC_SOURCE = "virtual_mic_source"

    @staticmethod
    def ensure_devices() -> bool:
        """Ensure all required virtual audio devices exist. Returns True if ready."""
        try:
            # Check existing sinks
            result = subprocess.run(
                ["pactl", "list", "short", "sinks"],
                capture_output=True, text=True, timeout=5,
            )
            existing_sinks = result.stdout

            # Create speaker sink if missing
            if PulseAudioManager.SPEAKER_SINK not in existing_sinks:
                subprocess.run([
                    "pactl", "load-module", "module-null-sink",
                    f"sink_name={PulseAudioManager.SPEAKER_SINK}",
                    "sink_properties=device.description=Virtual_Speaker",
                    "rate=44100", "channels=2", "format=s16le",
                ], capture_output=True, timeout=5)
                logger.info("Created virtual_speaker sink")

            # Create mic sink if missing
            if PulseAudioManager.MIC_SINK not in existing_sinks:
                subprocess.run([
                    "pactl", "load-module", "module-null-sink",
                    f"sink_name={PulseAudioManager.MIC_SINK}",
                    "sink_properties=device.description=Virtual_Microphone",
                    "rate=44100", "channels=1", "format=s16le",
                ], capture_output=True, timeout=5)
                logger.info("Created virtual_mic sink")

            # Check sources for remap
            result = subprocess.run(
                ["pactl", "list", "short", "sources"],
                capture_output=True, text=True, timeout=5,
            )
            if PulseAudioManager.MIC_SOURCE not in result.stdout:
                subprocess.run([
                    "pactl", "load-module", "module-remap-source",
                    f"master={PulseAudioManager.MIC_SINK}.monitor",
                    f"source_name={PulseAudioManager.MIC_SOURCE}",
                    "source_properties=device.description=Virtual_Microphone_Input",
                ], capture_output=True, timeout=5)
                logger.info("Created virtual_mic_source remap")

            # Set virtual_mic_source as default source (browsers pick this up)
            subprocess.run(
                ["pactl", "set-default-source", PulseAudioManager.MIC_SOURCE],
                capture_output=True, timeout=5,
            )

            logger.info("PulseAudio devices ready")
            return True

        except Exception as e:
            logger.error("PulseAudio setup failed: %s", e)
            return False

    @staticmethod
    def route_browser_audio(browser_pid: Optional[int] = None):
        """Route browser audio output to virtual_speaker sink."""
        try:
            # Move all sink inputs to virtual_speaker
            result = subprocess.run(
                ["pactl", "list", "short", "sink-inputs"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                input_id = line.split()[0]
                subprocess.run(
                    ["pactl", "move-sink-input", input_id,
                     PulseAudioManager.SPEAKER_SINK],
                    capture_output=True, timeout=5,
                )
                logger.debug("Routed sink-input %s → virtual_speaker", input_id)
        except Exception as e:
            logger.warning("Failed to route browser audio: %s", e)


# ---------------------------------------------------------------------------
# Continuous audio capture with ring buffer
# ---------------------------------------------------------------------------

class ContinuousAudioCapture:
    """
    Captures audio continuously from virtual_speaker.monitor into a 60-second
    ring buffer. Runs parecord in a background thread.
    """

    def __init__(
        self,
        device: str = "virtual_speaker.monitor",
        sample_rate: int = SAMPLE_RATE,
        buffer_duration_s: float = 60.0,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.buffer_duration_s = buffer_duration_s
        self.max_samples = int(buffer_duration_s * sample_rate)

        self._lock = threading.Lock()
        self._buffer = np.zeros(self.max_samples, dtype=np.float32)
        self._write_pos = 0
        self._total_written = 0
        self._capture_start: Optional[float] = None

        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> bool:
        """Start continuous audio capture in a background thread."""
        if self._running:
            return True

        try:
            self._capture_start = time.time()
            self._process = subprocess.Popen(
                [
                    "parecord",
                    "--format=float32le",
                    f"--rate={self.sample_rate}",
                    "--channels=1",
                    f"--device={self.device}",
                    "--raw",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._running = True
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()
            logger.info(
                "Continuous audio capture started (device=%s, rate=%d, buffer=%.0fs)",
                self.device, self.sample_rate, self.buffer_duration_s,
            )
            return True
        except Exception as e:
            logger.error("Failed to start audio capture: %s", e)
            return False

    def stop(self):
        """Stop capture and clean up."""
        self._running = False
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        logger.info("Audio capture stopped (total samples: %d)", self._total_written)

    def _reader_loop(self):
        """Read raw float32 samples from parecord stdout into ring buffer."""
        CHUNK_BYTES = self.sample_rate * 4  # 1 second of float32 data
        while self._running and self._process and self._process.poll() is None:
            try:
                raw = self._process.stdout.read(CHUNK_BYTES)
                if not raw:
                    break
                samples = np.frombuffer(raw, dtype=np.float32)
                self._write_to_buffer(samples)
            except Exception as e:
                if self._running:
                    logger.warning("Audio reader error: %s", e)
                break

    def _write_to_buffer(self, data: np.ndarray):
        """Write samples into the ring buffer (thread-safe)."""
        with self._lock:
            n = len(data)
            if n >= self.max_samples:
                data = data[-self.max_samples:]
                n = self.max_samples

            end = self._write_pos + n
            if end <= self.max_samples:
                self._buffer[self._write_pos:end] = data
            else:
                first = self.max_samples - self._write_pos
                self._buffer[self._write_pos:] = data[:first]
                self._buffer[:n - first] = data[first:]

            self._write_pos = end % self.max_samples
            self._total_written += n

    def get_audio(self, last_n_seconds: float) -> tuple[np.ndarray, float, float]:
        """
        Retrieve the last N seconds of audio from the ring buffer.

        Returns:
            (audio_data, start_time, end_time) — audio_data is float32 numpy array.
            Times are absolute (seconds since epoch).
        """
        with self._lock:
            n_samples = min(
                int(last_n_seconds * self.sample_rate),
                self._total_written,
                self.max_samples,
            )
            if n_samples == 0:
                now = time.time()
                return np.zeros(0, dtype=np.float32), now, now

            # Read backwards from write position
            start_idx = (self._write_pos - n_samples) % self.max_samples
            if start_idx < self._write_pos:
                data = self._buffer[start_idx:self._write_pos].copy()
            else:
                data = np.concatenate([
                    self._buffer[start_idx:],
                    self._buffer[:self._write_pos],
                ]).copy()

            now = time.time()
            duration = len(data) / self.sample_rate
            return data, now - duration, now

    def get_rms(self, last_n_seconds: float = 1.0) -> float:
        """Get RMS energy of the last N seconds (for silence gating)."""
        audio, _, _ = self.get_audio(last_n_seconds)
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))

    def reset(self):
        """Clear the ring buffer for a fresh task (avoids audio leaking between tasks)."""
        with self._lock:
            self._buffer[:] = 0.0
            self._write_pos = 0
            self._total_written = 0
            self._capture_start = time.time()
        logger.info("Audio ring buffer reset")

    @property
    def elapsed_s(self) -> float:
        if self._capture_start is None:
            return 0.0
        return time.time() - self._capture_start


# ---------------------------------------------------------------------------
# Whisper ASR (local, via faster-whisper)
# ---------------------------------------------------------------------------

class WhisperTranscriber:
    """
    Local speech-to-text using faster-whisper (CTranslate2 backend).
    Singleton model loading to avoid per-task overhead.
    """

    _shared_model = None
    _shared_model_size = None
    _lock = threading.Lock()

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self._model = None

    def _ensure_model(self):
        """Load model (shared singleton across instances)."""
        with WhisperTranscriber._lock:
            if (WhisperTranscriber._shared_model is not None
                    and WhisperTranscriber._shared_model_size == self.model_size):
                self._model = WhisperTranscriber._shared_model
                return

            from faster_whisper import WhisperModel
            logger.info("Loading faster-whisper model: %s (device=%s)", self.model_size, self.device)
            t0 = time.time()
            model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="int8" if self.device == "cpu" else "float16",
            )
            elapsed = time.time() - t0
            logger.info("Whisper model loaded in %.1fs", elapsed)

            WhisperTranscriber._shared_model = model
            WhisperTranscriber._shared_model_size = self.model_size
            self._model = model

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> list[TimestampedSegment]:
        """
        Transcribe audio to timestamped segments.

        Args:
            audio: float32 numpy array
            sample_rate: sample rate of audio

        Returns:
            List of TimestampedSegment with word-level or segment-level timestamps.
        """
        if len(audio) == 0:
            return []

        # Silence gate: skip if RMS is very low
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            return []

        self._ensure_model()

        # faster-whisper expects float32 at 16kHz
        if sample_rate != 16000:
            # Resample
            import scipy.signal
            audio = scipy.signal.resample(
                audio, int(len(audio) * 16000 / sample_rate)
            ).astype(np.float32)

        segments, info = self._model.transcribe(
            audio,
            beam_size=1,          # Fast greedy decoding
            best_of=1,
            language="en",
            vad_filter=True,      # Skip silence
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
        )

        results = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                results.append(TimestampedSegment(
                    text=text,
                    start_s=seg.start,
                    end_s=seg.end,
                ))

        return results

    def transcribe_text_only(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        """Transcribe and return plain text (no timestamps)."""
        segments = self.transcribe(audio, sample_rate)
        return " ".join(seg.text for seg in segments)


# ---------------------------------------------------------------------------
# TTS Engine (edge-tts)
# ---------------------------------------------------------------------------

class TTSEngine:
    """
    Text-to-speech using edge-tts (Microsoft Edge TTS, high quality, no API key).
    Produces WAV audio that can be injected into the virtual microphone.
    """

    def __init__(self, voice: str = "en-US-GuyNeural"):
        self.voice = voice

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Convert text to audio.

        Returns:
            (audio_data, sample_rate) — audio_data is float32 numpy array.
        """
        import edge_tts

        # edge-tts is async — run in a dedicated thread to avoid
        # conflicts with Playwright's event loop
        result_holder = [None]

        def _run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result_holder[0] = loop.run_until_complete(self._synthesize_async(text))
            finally:
                loop.close()

        thread = threading.Thread(target=_run_in_thread)
        thread.start()
        thread.join(timeout=30)
        audio_bytes = result_holder[0]

        if not audio_bytes:
            return np.zeros(0, dtype=np.float32), SAMPLE_RATE

        # Decode MP3 to numpy (edge-tts outputs MP3)
        return self._decode_mp3(audio_bytes)

    async def _synthesize_async(self, text: str) -> bytes:
        """Async TTS synthesis."""
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks)

    @staticmethod
    def _decode_mp3(mp3_bytes: bytes) -> tuple[np.ndarray, int]:
        """Decode MP3 bytes to float32 numpy array using soundfile or av."""
        try:
            # Try av (installed with faster-whisper)
            import av
            container = av.open(io.BytesIO(mp3_bytes))
            audio_stream = container.streams.audio[0]
            resampler = av.audio.resampler.AudioResampler(
                format="s16", layout="mono", rate=SAMPLE_RATE,
            )
            frames = []
            for frame in container.decode(audio_stream):
                resampled = resampler.resample(frame)
                for r in resampled:
                    arr = r.to_ndarray().flatten()
                    frames.append(arr.astype(np.float32) / 32768.0)
            container.close()
            if frames:
                return np.concatenate(frames), SAMPLE_RATE
        except Exception as e:
            logger.warning("av decode failed: %s, trying soundfile", e)

        # Fallback: write to temp file and use ffmpeg
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(mp3_bytes)
                mp3_path = f.name

            wav_path = mp3_path.replace(".mp3", ".wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", mp3_path, "-ar", str(SAMPLE_RATE),
                 "-ac", "1", "-f", "wav", wav_path],
                capture_output=True, timeout=10,
            )

            import soundfile as sf
            data, sr = sf.read(wav_path, dtype="float32")
            Path(mp3_path).unlink(missing_ok=True)
            Path(wav_path).unlink(missing_ok=True)
            return data, sr
        except Exception as e:
            logger.error("TTS decode failed: %s", e)
            return np.zeros(0, dtype=np.float32), SAMPLE_RATE


# ---------------------------------------------------------------------------
# Microphone injection (TTS → virtual_mic)
# ---------------------------------------------------------------------------

class MicInjector:
    """
    Injects audio into the virtual microphone sink so the browser
    receives it as microphone input (for the agent's `speak` action).
    """

    def __init__(self, sink: str = "virtual_mic", sample_rate: int = SAMPLE_RATE):
        self.sink = sink
        self.sample_rate = sample_rate

    def inject(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bool:
        """
        Play audio into the virtual_mic sink using pacat.

        Args:
            audio: float32 numpy array
            sample_rate: sample rate of audio

        Returns:
            True if injection succeeded.
        """
        if len(audio) == 0:
            return False

        # Convert to raw bytes (float32le)
        raw_bytes = audio.astype(np.float32).tobytes()

        try:
            result = subprocess.run(
                [
                    "pacat",
                    "--format=float32le",
                    f"--rate={sample_rate}",
                    "--channels=1",
                    f"--device={self.sink}",
                    "--raw",
                ],
                input=raw_bytes,
                capture_output=True,
                timeout=len(audio) / sample_rate + 5,
            )
            if result.returncode == 0:
                duration = len(audio) / sample_rate
                logger.info("Injected %.1fs audio into %s", duration, self.sink)
                return True
            else:
                logger.warning("pacat failed: %s", result.stderr.decode()[:200])
                return False
        except Exception as e:
            logger.error("Mic injection failed: %s", e)
            return False


# ---------------------------------------------------------------------------
# Two-Layer Audio Processor
# ---------------------------------------------------------------------------

class AudioProcessor:
    """
    Orchestrates the two-layer audio pipeline:

    1. Captures audio from PulseAudio ring buffer
    2. Runs faster-whisper ASR on the captured audio
    3. Produces TwoLayerAudio with:
       - Layer 1: Recent 3-5s transcript (synced with keyframes)
       - Layer 2: Rolling 30-60s transcript with segment timestamps

    Also handles the `speak` action via TTS + mic injection.
    """

    def __init__(
        self,
        layer1_duration_s: float = 5.0,
        layer2_duration_s: float = 60.0,
        whisper_model_size: str = "base",
        tts_voice: str = "en-US-GuyNeural",
        silence_threshold: float = 0.005,
    ):
        self.layer1_duration_s = layer1_duration_s
        self.layer2_duration_s = layer2_duration_s
        self.silence_threshold = silence_threshold

        # Components
        self.capture = ContinuousAudioCapture(
            buffer_duration_s=layer2_duration_s + 10,  # slight headroom
        )
        self.transcriber = WhisperTranscriber(model_size=whisper_model_size)
        self.tts = TTSEngine(voice=tts_voice)
        self.mic_injector = MicInjector()

        # Rolling transcript cache for Layer 2
        self._layer2_cache: list[TimestampedSegment] = []
        self._last_l2_transcribe_time: float = 0.0
        self._l2_min_interval_s: float = 5.0  # Don't re-transcribe L2 more than every 5s

        # Stats
        self.stats = {
            "l1_calls": 0,
            "l2_calls": 0,
            "speak_calls": 0,
            "total_asr_ms": 0.0,
            "total_tts_ms": 0.0,
        }

    def start(self) -> bool:
        """Start the audio capture pipeline."""
        PulseAudioManager.ensure_devices()
        return self.capture.start()

    def stop(self):
        """Stop the audio capture pipeline."""
        self.capture.stop()

    def reset(self):
        """Reset audio state for a new task (prevents cross-task audio leaking)."""
        self.capture.reset()
        self._layer2_cache.clear()
        self._last_l2_transcribe_time = 0.0
        logger.info("AudioProcessor reset for new task")

    def get_two_layer_audio(self) -> TwoLayerAudio:
        """
        Produce a TwoLayerAudio snapshot for the current agent step.

        Layer 1: Transcribe last layer1_duration_s seconds
        Layer 2: Transcribe last layer2_duration_s seconds (cached, refreshed periodically)
        """
        now = time.time()

        # --- Layer 1: Recent audio ---
        l1_audio, l1_start, l1_end = self.capture.get_audio(self.layer1_duration_s)
        l1_rms = float(np.sqrt(np.mean(l1_audio ** 2))) if len(l1_audio) > 0 else 0.0

        l1_text = ""
        if l1_rms >= self.silence_threshold and len(l1_audio) > 0:
            t0 = time.time()
            l1_text = self.transcriber.transcribe_text_only(l1_audio)
            asr_ms = (time.time() - t0) * 1000
            self.stats["total_asr_ms"] += asr_ms
            self.stats["l1_calls"] += 1
            logger.debug("Layer 1 ASR: %.0fms, text='%s'", asr_ms, l1_text[:80])

        # --- Layer 2: Rolling context ---
        # Re-transcribe Layer 2 if enough time has passed
        if now - self._last_l2_transcribe_time >= self._l2_min_interval_s:
            l2_audio, l2_start, l2_end = self.capture.get_audio(self.layer2_duration_s)
            l2_rms = float(np.sqrt(np.mean(l2_audio ** 2))) if len(l2_audio) > 0 else 0.0

            if l2_rms >= self.silence_threshold and len(l2_audio) > 0:
                t0 = time.time()
                segments = self.transcriber.transcribe(l2_audio)

                # Mark segments that fall within the Layer 1 window
                l1_boundary = self.capture.elapsed_s - self.layer1_duration_s
                for seg in segments:
                    seg.is_recent = seg.start_s >= l1_boundary

                self._layer2_cache = segments
                asr_ms = (time.time() - t0) * 1000
                self.stats["total_asr_ms"] += asr_ms
                self.stats["l2_calls"] += 1
                logger.debug("Layer 2 ASR: %.0fms, %d segments", asr_ms, len(segments))

            self._last_l2_transcribe_time = now

        return TwoLayerAudio(
            layer1_text=l1_text,
            layer1_duration_s=self.layer1_duration_s,
            layer2_segments=list(self._layer2_cache),
            layer2_duration_s=self.layer2_duration_s,
            capture_end_time=now,
        )

    def speak(self, text: str) -> bool:
        """
        Agent speak action: convert text to audio via TTS, inject into virtual mic.

        Returns True if the audio was successfully injected.
        """
        if not text.strip():
            return False

        t0 = time.time()
        audio, sr = self.tts.synthesize(text)
        tts_ms = (time.time() - t0) * 1000
        self.stats["total_tts_ms"] += tts_ms
        self.stats["speak_calls"] += 1

        if len(audio) == 0:
            logger.warning("TTS produced empty audio for: '%s'", text[:50])
            return False

        logger.info(
            "TTS: %.0fms, %.1fs audio for '%s'",
            tts_ms, len(audio) / sr, text[:50],
        )

        return self.mic_injector.inject(audio, sr)

    def get_stats(self) -> dict:
        return dict(self.stats)
