"""
Persistent Whisper ASR Service — runs on GPU, serves transcriptions via HTTP.

This service keeps Whisper large-v3 loaded on GPU permanently and accepts
audio data via HTTP POST. This avoids:
  1. Model loading delay (~5-10s cold start per process)
  2. VRAM duplication when multiple eval processes load the model
  3. In-process model overhead interfering with the eval loop

Usage:
    # Start the service (once, keep running):
    python -m aoi.whisper_service --model large-v3 --port 8786

    # Transcribe audio (from any process):
    curl -X POST http://localhost:8786/transcribe \
         -F "audio=@audio.raw" -F "sample_rate=16000"
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper ASR Service")

# Global model reference
_model = None
_model_size = None


def load_model(model_size: str = "large-v3", device: str = "cuda"):
    """Load the Whisper model into GPU memory."""
    global _model, _model_size
    from faster_whisper import WhisperModel

    logger.info("Loading faster-whisper model: %s (device=%s)", model_size, device)
    t0 = time.time()
    _model = WhisperModel(
        model_size,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
    )
    _model_size = model_size
    elapsed = time.time() - t0
    logger.info("Whisper model loaded in %.1fs", elapsed)


@dataclass
class TranscribeSegment:
    text: str
    start_s: float
    end_s: float


def _transcribe_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> list[TranscribeSegment]:
    """Transcribe audio array to timestamped segments."""
    if len(audio) == 0:
        return []

    # Silence gate
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.005:
        return []

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import scipy.signal
        audio = scipy.signal.resample(
            audio, int(len(audio) * 16000 / sample_rate)
        ).astype(np.float32)

    segments, info = _model.transcribe(
        audio,
        beam_size=1,
        best_of=1,
        language="en",
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=300,
            speech_pad_ms=200,
        ),
    )

    results = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            results.append(TranscribeSegment(
                text=text,
                start_s=seg.start,
                end_s=seg.end,
            ))

    return results


@app.get("/health")
async def health():
    return {"status": "ok", "model": _model_size, "model_loaded": _model is not None}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    sample_rate: int = Form(16000),
    text_only: bool = Form(False),
):
    """
    Transcribe raw float32 audio data.

    Args:
        audio: Raw float32le audio file
        sample_rate: Sample rate of the audio (default 16000)
        text_only: If True, return only the concatenated text

    Returns:
        JSON with segments (text, start_s, end_s) or plain text
    """
    t0 = time.time()

    raw_bytes = await audio.read()
    audio_array = np.frombuffer(raw_bytes, dtype=np.float32)

    if len(audio_array) == 0:
        return JSONResponse({"segments": [], "text": "", "duration_ms": 0})

    segments = _transcribe_audio(audio_array, sample_rate)
    elapsed_ms = (time.time() - t0) * 1000

    if text_only:
        text = " ".join(s.text for s in segments)
        return JSONResponse({
            "text": text,
            "duration_ms": round(elapsed_ms, 1),
        })

    return JSONResponse({
        "segments": [
            {"text": s.text, "start_s": round(s.start_s, 2), "end_s": round(s.end_s, 2)}
            for s in segments
        ],
        "text": " ".join(s.text for s in segments),
        "duration_ms": round(elapsed_ms, 1),
    })


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Whisper ASR Service")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8786, help="Port to listen on")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    load_model(args.model, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
