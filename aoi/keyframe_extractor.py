"""
Adaptive Keyframe Extractor — Stage 1: pixel gate, Stage 2: CLIP semantic distance.

Design goals (from paper §3.2):
- Sub-1ms cost when screen is static (pixel gate short-circuits)
- ~5-10ms amortized per sample with CLIP on GPU when screen changes
- Suppress periodic noise (spinners, cursors, looping ads) via CLIP semantic stability
- Capture semantically meaningful transitions (dialogs, page navigation, video scene cuts)
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Optional
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Keyframe:
    timestamp: float          # seconds since epoch
    image: Image.Image        # PIL image
    clip_embedding: np.ndarray  # shape (512,), float32
    pixel_change_ratio: float   # ratio that triggered capture


class KeyframeExtractor:
    """
    Two-stage adaptive keyframe extractor.

    Stage 1 — Pixel gate: if < pixel_threshold fraction of pixels changed,
               skip entirely (< 1 ms CPU cost).
    Stage 2 — CLIP distance: if cosine distance to anchor embedding < theta,
               suppress (periodic noise filter). If >= theta, emit keyframe
               and reanchor.

    Thread-safe: on_sample() can be called from a background capture thread
    while get_and_reset() is called from the agent loop thread.
    """

    def __init__(
        self,
        device: str = "cuda",
        theta: float = 0.04,  # Calibrated on DynaCU-Bench: web UI changes ~0.05-0.08
        pixel_threshold: float = 0.01,
        max_keyframes: int = 5,
        sample_size: tuple[int, int] = (224, 224),  # CLIP input size
    ):
        self.theta = theta
        self.pixel_threshold = pixel_threshold
        self.max_keyframes = max_keyframes
        self.sample_size = sample_size
        self.device = device

        # Lazy-load CLIP so module import is fast
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_lock = threading.Lock()

        # State (protected by _lock)
        self._lock = threading.Lock()
        self._anchor_emb: Optional[np.ndarray] = None
        self._last_gray: Optional[np.ndarray] = None  # downsized grayscale for fast pixel diff
        self._keyframes: list[Keyframe] = []

        # Metrics
        self.stats = {
            "samples_total": 0,
            "pixel_gate_passed": 0,
            "clip_gate_passed": 0,
            "keyframes_emitted": 0,
        }

    def _load_clip(self):
        if self._clip_model is not None:
            return
        with self._clip_lock:
            if self._clip_model is not None:
                return
            import clip
            import torch
            model, preprocess = clip.load("ViT-B/16", device=self.device)
            model.eval()
            self._clip_model = model
            self._clip_preprocess = preprocess
            logger.info("CLIP ViT-B/16 loaded on %s", self.device)

    def _encode_clip(self, image: Image.Image) -> np.ndarray:
        import torch
        self._load_clip()
        tensor = self._clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._clip_model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0].astype(np.float32)

    @staticmethod
    def _pixel_change_ratio(current: np.ndarray, previous: np.ndarray) -> float:
        """Fraction of pixels that changed by more than 10 grayscale units."""
        diff = np.abs(current.astype(np.int16) - previous.astype(np.int16))
        return float(np.mean(diff > 10))

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """1 - cosine_similarity for unit-normalized vectors."""
        return float(1.0 - np.dot(a, b))

    def on_sample(self, frame: Image.Image, timestamp: Optional[float] = None) -> None:
        """
        Process a new screen sample. Call this from the capture thread at ~3 Hz.
        Thread-safe.
        """
        if timestamp is None:
            timestamp = time.time()

        # Downsample for fast pixel comparison (64x64 grayscale)
        small = np.array(frame.convert("L").resize((64, 64), Image.BILINEAR))

        with self._lock:
            self.stats["samples_total"] += 1

            # --- Stage 1: Pixel gate ---
            if self._last_gray is not None:
                ratio = self._pixel_change_ratio(small, self._last_gray)
                if ratio < self.pixel_threshold:
                    # Screen unchanged — skip CLIP entirely
                    return
            else:
                ratio = 1.0  # First frame, always pass

            self._last_gray = small
            self.stats["pixel_gate_passed"] += 1

            # --- Stage 2: CLIP semantic distance ---
            emb = self._encode_clip(frame)

            if self._anchor_emb is None:
                # Bootstrap anchor on first semantically non-trivial frame
                self._anchor_emb = emb
                return

            dist = self._cosine_distance(emb, self._anchor_emb)
            if dist < self.theta:
                # Semantically similar to anchor (spinner, cursor, repeated frame)
                return

            # Semantic change detected — emit keyframe, reanchor
            self.stats["clip_gate_passed"] += 1
            self.stats["keyframes_emitted"] += 1
            self._anchor_emb = emb
            self._keyframes.append(
                Keyframe(
                    timestamp=timestamp,
                    image=frame.copy(),
                    clip_embedding=emb,
                    pixel_change_ratio=ratio,
                )
            )

    def get_and_reset(self) -> list[Keyframe]:
        """
        Return collected keyframes (up to max_keyframes) and clear the buffer.
        Call this from the agent loop thread at each step.
        """
        with self._lock:
            result = self._keyframes[-self.max_keyframes :]
            self._keyframes.clear()
            return result

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self.stats)

    def reset_anchor(self) -> None:
        """Force anchor reset (e.g. after agent navigates to a new page)."""
        with self._lock:
            self._anchor_emb = None
            self._last_gray = None
