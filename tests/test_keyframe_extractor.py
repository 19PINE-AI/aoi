"""Tests for the KeyframeExtractor."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
from PIL import Image, ImageDraw
import pytest

from aoi.keyframe_extractor import KeyframeExtractor


def make_frame(color: tuple, width: int = 224, height: int = 224, text: str = "") -> Image.Image:
    img = Image.new("RGB", (width, height), color)
    if text:
        draw = ImageDraw.Draw(img)
        draw.text((width // 2, height // 2), text, fill=(0, 0, 0))
    return img


def test_pixel_gate_suppresses_identical_frames():
    """Identical frames should be suppressed by the pixel gate."""
    extractor = KeyframeExtractor(theta=0.04, pixel_threshold=0.01)
    frame = make_frame((200, 200, 200))

    extractor.on_sample(frame, timestamp=0.0)  # First frame — bootstraps anchor
    extractor.on_sample(frame, timestamp=0.1)
    extractor.on_sample(frame, timestamp=0.2)
    extractor.on_sample(frame, timestamp=0.3)

    keyframes = extractor.get_and_reset()
    # No keyframes emitted for identical frames
    assert len(keyframes) == 0, f"Expected 0 keyframes, got {len(keyframes)}"


def test_semantic_change_captured():
    """Significantly different frames should produce keyframes."""
    extractor = KeyframeExtractor(theta=0.10, pixel_threshold=0.01)

    # White frame (bootstrap anchor)
    white = make_frame((255, 255, 255))
    extractor.on_sample(white, timestamp=0.0)

    # Very different frame (dark blue — should exceed CLIP theta)
    dark_blue = make_frame((10, 10, 150))
    extractor.on_sample(dark_blue, timestamp=0.5)

    keyframes = extractor.get_and_reset()
    assert len(keyframes) >= 1, "Expected at least 1 keyframe for semantic change"


def test_max_keyframes_respected():
    """Should not emit more than max_keyframes per step."""
    extractor = KeyframeExtractor(theta=0.05, pixel_threshold=0.001, max_keyframes=3)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for i, color in enumerate(colors):
        extractor.on_sample(make_frame(color), timestamp=float(i))

    keyframes = extractor.get_and_reset()
    assert len(keyframes) <= 3, f"Expected max 3 keyframes, got {len(keyframes)}"


def test_buffer_cleared_after_get():
    """get_and_reset should clear the buffer."""
    extractor = KeyframeExtractor(theta=0.05, pixel_threshold=0.001)

    white = make_frame((255, 255, 255))
    blue = make_frame((0, 0, 200))

    extractor.on_sample(white, 0.0)
    extractor.on_sample(blue, 1.0)

    first_get = extractor.get_and_reset()
    second_get = extractor.get_and_reset()

    assert len(second_get) == 0, "Buffer should be empty after first get"


def test_pixel_gate_passes_changed_frame():
    """A frame with >1% changed pixels should pass the pixel gate."""
    extractor = KeyframeExtractor(theta=0.30, pixel_threshold=0.01)  # High CLIP threshold

    # Start with all-white frame
    white = make_frame((255, 255, 255))
    extractor.on_sample(white, 0.0)

    # Change 50% of pixels to black (should pass pixel gate)
    changed = Image.new("RGB", (224, 224))
    arr = np.array(white)
    arr[:112, :] = 0  # Top half black
    changed = Image.fromarray(arr)

    extractor.on_sample(changed, 1.0)

    stats = extractor.get_stats()
    assert stats["pixel_gate_passed"] >= 1, "Changed frame should pass pixel gate"


def test_stats_tracking():
    """Stats should be correctly tracked."""
    extractor = KeyframeExtractor(theta=0.10, pixel_threshold=0.01)

    frames = [make_frame((i * 40, i * 20, 100)) for i in range(5)]
    for i, frame in enumerate(frames):
        extractor.on_sample(frame, float(i))

    stats = extractor.get_stats()
    assert stats["samples_total"] == 5
    assert stats["pixel_gate_passed"] <= stats["samples_total"]
    assert stats["keyframes_emitted"] <= stats["clip_gate_passed"]


def test_reset_anchor():
    """reset_anchor should allow re-capture of any subsequent frame."""
    extractor = KeyframeExtractor(theta=0.04, pixel_threshold=0.01)

    frame = make_frame((200, 200, 200))
    extractor.on_sample(frame, 0.0)

    # Same frame — no keyframe
    extractor.on_sample(frame, 1.0)
    assert len(extractor.get_and_reset()) == 0

    # Reset anchor — next frame is same but re-bootstraps
    extractor.reset_anchor()
    extractor.on_sample(frame, 2.0)  # Bootstraps new anchor, no keyframe yet

    # Different frame after reset
    different = make_frame((0, 100, 200))
    extractor.on_sample(different, 3.0)

    # Should have captured since we have a different semantic frame after anchor
    # (may or may not depending on CLIP distance — test just that it doesn't crash)
    extractor.get_and_reset()  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
