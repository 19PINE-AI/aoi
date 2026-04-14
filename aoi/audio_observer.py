"""
Volume-Gated Audio Observer — RMS energy gate + multimodal audio scene understanding.

Design goals (from paper §3.3):
- ~0ms overhead when silent (>90% of typical desktop work)
- Full audio scene understanding (speech + non-speech) via Gemini/Claude multimodal API
- Overlapping window for boundary continuity across agent step boundaries
- Supports both API-based inference (Gemini 2.0 Flash) and local Whisper fallback
"""

from __future__ import annotations

import io
import logging
import time
import threading
import base64
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # Hz — standard for speech models
CHANNELS = 1


@dataclass
class AudioChunk:
    """A segment of captured audio."""
    data: np.ndarray        # float32, shape (n_samples,), range [-1, 1]
    sample_rate: int
    start_time: float       # seconds since epoch
    end_time: float         # seconds since epoch

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def rms_energy(self) -> float:
        if len(self.data) == 0:
            return 0.0
        return float(np.sqrt(np.mean(self.data ** 2)))


class AudioBuffer:
    """
    Ring buffer that captures audio continuously from a virtual/system audio device.
    The agent loop calls get_chunk() to retrieve the latest interval.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        max_duration_s: float = 30.0,  # ring buffer size
        overlap_s: float = 3.5,        # overlap carried forward for boundary handling
    ):
        self.sample_rate = sample_rate
        self.overlap_s = overlap_s
        self.max_samples = int(max_duration_s * sample_rate)

        self._lock = threading.Lock()
        self._buffer = np.zeros(self.max_samples, dtype=np.float32)
        self._write_pos = 0
        self._total_samples = 0
        self._capture_start_time = time.time()
        self._stream = None

    def start_capture(self, device: Optional[str] = None) -> bool:
        """
        Start audio capture. Returns True if successful.
        Falls back to synthetic silence if no audio device is available
        (common in headless VM environments).
        """
        try:
            import sounddevice as sd

            def _callback(indata, frames, time_info, status):
                if status:
                    logger.debug("Audio status: %s", status)
                mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
                self._write(mono.astype(np.float32))

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=_callback,
                device=device,
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
            )
            self._stream.start()
            logger.info("Audio capture started (device=%s, rate=%d Hz)", device, self.sample_rate)
            return True
        except Exception as e:
            logger.warning("Audio capture unavailable: %s — using synthetic silence", e)
            return False

    def stop_capture(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _write(self, data: np.ndarray):
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
            self._total_samples += n

    def get_chunk(
        self,
        start_time: float,
        end_time: float,
        include_overlap: bool = True,
    ) -> AudioChunk:
        """
        Retrieve audio between start_time and end_time.
        If include_overlap, prepend overlap_s of prior audio for boundary continuity.
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._capture_start_time
            total_buffered = min(self._total_samples, self.max_samples)

            def time_to_sample(t: float) -> int:
                # Convert absolute time to sample index
                samples_ago = int((now - t) * self.sample_rate)
                return max(0, min(total_buffered, samples_ago))

            s_end = time_to_sample(end_time)
            s_start = time_to_sample(start_time)

            if include_overlap:
                s_start = time_to_sample(start_time - self.overlap_s)

            # Extract from ring buffer (handle wrap-around)
            n_samples = s_start - s_end  # samples_ago is inverted
            if n_samples <= 0:
                data = np.zeros(0, dtype=np.float32)
            else:
                # Read from ring buffer backwards from write_pos
                indices = np.arange(s_end, s_start)
                read_positions = (self._write_pos - 1 - indices) % self.max_samples
                data = self._buffer[read_positions[::-1]]

        return AudioChunk(
            data=data,
            sample_rate=self.sample_rate,
            start_time=start_time - (self.overlap_s if include_overlap else 0),
            end_time=end_time,
        )

    def inject_synthetic(self, data: np.ndarray, duration_s: float = None):
        """Inject audio for testing (e.g. synthetic speech or tones)."""
        self._write(data.astype(np.float32))


class AudioObserver:
    """
    Volume-gated multimodal audio scene observer.

    Processing pipeline:
    1. Compute RMS energy of chunk's new portion
    2. If below silence_threshold, return "" (zero model cost)
    3. Otherwise, call multimodal audio model (Gemini or local Whisper)
       with overlapping audio + prior transcript for boundary continuity
    """

    def __init__(
        self,
        silence_threshold: float = 0.01,   # RMS threshold below which audio is "silent"
        backend: str = "gemini",            # "gemini", "openai_whisper", or "whisper_local"
        model_name: str = "gemini-2.0-flash",
        overlap_s: float = 3.5,
    ):
        self.silence_threshold = silence_threshold
        self.backend = backend
        self.model_name = model_name
        self.overlap_s = overlap_s

        self._client = None
        self.stats = {
            "chunks_processed": 0,
            "silent_chunks": 0,
            "model_calls": 0,
            "total_model_ms": 0.0,
        }

    def _init_client(self):
        if self._client is not None:
            return
        if self.backend == "gemini":
            import google.generativeai as genai
            import os
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self._client = genai.GenerativeModel(self.model_name)
            logger.info("AudioObserver: Gemini client initialized (%s)", self.model_name)
        elif self.backend in ("openai_whisper", "whisper_local"):
            import openai
            import os
            self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            logger.info("AudioObserver: OpenAI Whisper client initialized")

    def _chunk_to_wav_bytes(self, chunk: AudioChunk) -> bytes:
        """Convert numpy audio to WAV bytes for API upload."""
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, chunk.data, chunk.sample_rate, format="WAV", subtype="FLOAT")
        buf.seek(0)
        return buf.read()

    def _call_gemini(self, wav_bytes: bytes, prior_transcript: str) -> str:
        """Call Gemini with audio data for multimodal scene understanding."""
        import google.generativeai as genai

        audio_part = {"mime_type": "audio/wav", "data": base64.b64encode(wav_bytes).decode()}

        prompt = (
            f"You are observing audio from a computer screen recording. "
            f"Describe EVERYTHING you hear: speech content, speaker changes, "
            f"notification sounds, alert beeps, music, system sounds, and any other audio events. "
            f"Be concise but complete.\n\n"
        )
        if prior_transcript:
            prompt += f"Prior context (do not repeat): {prior_transcript[-200:]}\n\n"
        prompt += (
            "Describe only NEW audio not already in the prior context. "
            "If there is speech, transcribe it. If there are non-speech sounds, describe them. "
            "If it's truly silent, say '[silent]'."
        )

        response = self._client.generate_content([prompt, audio_part])
        return response.text.strip()

    def _call_whisper(self, wav_bytes: bytes) -> str:
        """Fallback: OpenAI Whisper (speech only, no non-speech sounds)."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name
        try:
            with open(tmp_path, "rb") as f:
                result = self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text",
                )
            return result.strip()
        finally:
            os.unlink(tmp_path)

    def _call_anthropic_audio(self, wav_bytes: bytes, prior_transcript: str) -> str:
        """Call Claude with audio via base64 for scene understanding."""
        import anthropic, os
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        audio_b64 = base64.b64encode(wav_bytes).decode()
        prompt = (
            "You are observing audio from a computer screen. "
            "Describe all audio events: speech (transcribe it), notification sounds, "
            "alert beeps, and any other sounds. Be concise."
        )
        if prior_transcript:
            prompt += f"\nPrior context: {prior_transcript[-200:]}\nDescribe only NEW audio."

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "audio/wav",
                            "data": audio_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text.strip()

    def process(
        self,
        chunk: AudioChunk,
        prior_transcript: str = "",
        new_portion_start: Optional[float] = None,
    ) -> str:
        """
        Process an audio chunk and return a text description.

        Args:
            chunk: AudioChunk including overlap portion
            prior_transcript: Text from previous step for continuity
            new_portion_start: Start time of the NEW portion (after overlap).
                               Used to compute RMS gate on new audio only.

        Returns:
            Text description of audio scene, or "" if silent.
        """
        self.stats["chunks_processed"] += 1

        # Compute RMS on the new (non-overlap) portion only
        if new_portion_start is not None and len(chunk.data) > 0:
            overlap_samples = int(self.overlap_s * chunk.sample_rate)
            new_data = chunk.data[overlap_samples:]
        else:
            new_data = chunk.data

        rms = float(np.sqrt(np.mean(new_data ** 2))) if len(new_data) > 0 else 0.0

        if rms < self.silence_threshold:
            self.stats["silent_chunks"] += 1
            return ""

        # Audio is present — call the model
        self._init_client()
        t0 = time.time()

        try:
            wav_bytes = self._chunk_to_wav_bytes(chunk)
            if self.backend == "gemini":
                result = self._call_gemini(wav_bytes, prior_transcript)
            elif self.backend in ("openai_whisper", "whisper_local"):
                result = self._call_whisper(wav_bytes)
            else:
                result = self._call_gemini(wav_bytes, prior_transcript)  # default
        except Exception as e:
            logger.error("Audio model call failed: %s", e)
            result = f"[audio processing error: {e}]"

        elapsed_ms = (time.time() - t0) * 1000
        self.stats["model_calls"] += 1
        self.stats["total_model_ms"] += elapsed_ms
        logger.debug("Audio model call: %.1fms, result: %s", elapsed_ms, result[:80])

        # Filter out "[silent]" model responses
        if "[silent]" in result.lower():
            return ""

        return result

    def get_stats(self) -> dict:
        return dict(self.stats)
