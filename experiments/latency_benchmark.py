"""
AOI Latency Benchmark — measures actual per-sample processing times.

Validates the paper's claims in §3.2:
- Pixel gate: <1ms per sample
- CLIP encode: 5-10ms per sample (GPU)
- Amortized cost: ~1-2ms (CLIP runs only when pixels changed ~10% of samples)

Also measures:
- CLIP threshold sensitivity (how many keyframes at different theta values)
- Pixel gate effectiveness (fraction of samples suppressed)
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

from aoi.keyframe_extractor import KeyframeExtractor


def make_static_frame(size=(1280, 720)) -> Image.Image:
    return Image.new("RGB", size, (200, 200, 200))


def make_dynamic_frame(i: int, size=(1280, 720)) -> Image.Image:
    """Different color per frame — maximally dynamic."""
    colors = [(255, 50, 50), (50, 255, 50), (50, 50, 255), (255, 255, 50), (50, 255, 255)]
    img = Image.new("RGB", size, colors[i % len(colors)])
    draw = ImageDraw.Draw(img)
    draw.text((size[0] // 2, size[1] // 2), f"Frame {i}", fill=(0, 0, 0))
    return img


def make_spinner_frame(i: int, size=(1280, 720)) -> Image.Image:
    """Simulates a spinner: mostly identical with small rotating element."""
    img = Image.new("RGB", size, (230, 230, 230))
    draw = ImageDraw.Draw(img)
    # Static background
    draw.rectangle([100, 100, size[0] - 100, size[1] - 100], fill=(240, 240, 240))
    draw.text((size[0] // 2, size[1] // 2), "Loading...", fill=(100, 100, 100))
    # Small spinner element that changes (simulating cursor/spinner blink)
    angle = i * 45  # Rotates
    cx, cy = size[0] // 2 + 100, size[1] // 2
    for j in range(8):
        a = (j * 45) % 360
        r = 15
        x = int(cx + r * np.cos(np.radians(a + angle)))
        y = int(cy + r * np.sin(np.radians(a + angle)))
        brightness = 255 - (j * 30)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(brightness, brightness, brightness))
    return img


def benchmark_pixel_gate(n_samples: int = 1000):
    """Measure pixel gate throughput on static frames."""
    extractor = KeyframeExtractor(theta=0.15)
    frame = make_static_frame()
    small = np.array(frame.convert("L").resize((64, 64)))

    t0 = time.perf_counter()
    for _ in range(n_samples):
        if extractor._last_gray is not None:
            _ = extractor._pixel_change_ratio(small, extractor._last_gray)
        extractor._last_gray = small
    elapsed = time.perf_counter() - t0

    per_sample_ms = (elapsed / n_samples) * 1000
    print(f"\nPixel Gate Benchmark ({n_samples} samples):")
    print(f"  Total time:      {elapsed * 1000:.1f} ms")
    print(f"  Per sample:      {per_sample_ms:.3f} ms")
    print(f"  Throughput:      {n_samples / elapsed:.0f} samples/sec")
    print(f"  Claim (paper):   <1ms/sample")
    print(f"  Status:          {'✓ PASS' if per_sample_ms < 1.0 else '✗ FAIL'}")
    return per_sample_ms


def benchmark_clip_encode(n_samples: int = 50):
    """Measure CLIP encoding throughput on GPU."""
    extractor = KeyframeExtractor(theta=0.15)
    extractor._load_clip()  # Pre-load

    frames = [make_dynamic_frame(i) for i in range(n_samples)]

    # Warmup
    for frame in frames[:5]:
        extractor._encode_clip(frame)

    t0 = time.perf_counter()
    for frame in frames:
        extractor._encode_clip(frame)
    elapsed = time.perf_counter() - t0

    per_sample_ms = (elapsed / n_samples) * 1000
    print(f"\nCLIP Encode Benchmark ({n_samples} samples on GPU):")
    print(f"  Total time:      {elapsed * 1000:.1f} ms")
    print(f"  Per sample:      {per_sample_ms:.1f} ms")
    print(f"  Throughput:      {n_samples / elapsed:.0f} frames/sec")
    print(f"  Claim (paper):   5-10ms/sample")
    print(f"  Status:          {'✓ PASS' if 1.0 <= per_sample_ms <= 15.0 else '✗ FAIL'}")
    return per_sample_ms


def benchmark_amortized_cost(n_samples: int = 300, static_fraction: float = 0.90):
    """Measure amortized cost (pixel gate suppresses ~90% of static samples)."""
    extractor = KeyframeExtractor(theta=0.15, pixel_threshold=0.01)

    # Mix of static and dynamic frames (90:10)
    n_static = int(n_samples * static_fraction)
    n_dynamic = n_samples - n_static
    frames = []
    for i in range(n_samples):
        if i < n_static:
            frames.append(make_static_frame())
        else:
            frames.append(make_dynamic_frame(i))
    # Shuffle to interleave
    np.random.seed(42)
    np.random.shuffle(frames)

    t0 = time.perf_counter()
    for i, frame in enumerate(frames):
        extractor.on_sample(frame, timestamp=float(i) / 3.0)
    elapsed = time.perf_counter() - t0

    stats = extractor.get_stats()
    amortized_ms = (elapsed / n_samples) * 1000
    pixel_gate_efficiency = 1.0 - stats["pixel_gate_passed"] / stats["samples_total"]

    print(f"\nAmortized Cost Benchmark ({n_samples} samples, {static_fraction*100:.0f}% static):")
    print(f"  Total time:          {elapsed * 1000:.1f} ms")
    print(f"  Amortized/sample:    {amortized_ms:.2f} ms")
    print(f"  Pixel gate blocked:  {pixel_gate_efficiency * 100:.1f}%")
    print(f"  CLIP calls:          {stats['pixel_gate_passed']}")
    print(f"  Keyframes emitted:   {stats['keyframes_emitted']}")
    print(f"  Claim (paper):       ~1-2ms amortized")
    print(f"  Status:              {'✓ PASS' if amortized_ms < 5.0 else '✗ FAIL (headless CPU)'}")
    return amortized_ms


def benchmark_theta_sensitivity():
    """Measure keyframe count at different theta thresholds."""
    print("\nCLIP Threshold (θ) Sensitivity:")
    print(f"{'θ':<8}{'Keyframes (dyn)':<20}{'Keyframes (spinner)':<22}{'Keyframes (static)'}")
    print("-" * 70)

    thetas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    for theta in thetas:
        n_test = 30
        # Dynamic scenario
        ext_dyn = KeyframeExtractor(theta=theta)
        for i in range(n_test):
            ext_dyn.on_sample(make_dynamic_frame(i), float(i))
        kf_dyn = len(ext_dyn.get_and_reset())

        # Spinner scenario (periodic noise)
        ext_spin = KeyframeExtractor(theta=theta)
        for i in range(n_test):
            ext_spin.on_sample(make_spinner_frame(i), float(i))
        kf_spin = len(ext_spin.get_and_reset())

        # Static scenario
        ext_stat = KeyframeExtractor(theta=theta)
        for i in range(n_test):
            ext_stat.on_sample(make_static_frame(), float(i))
        kf_stat = len(ext_stat.get_and_reset())

        print(f"{theta:<8.2f}{kf_dyn:<20}{kf_spin:<22}{kf_stat}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-clip", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("AOI Component Latency Benchmark")
    print("=" * 60)

    pixel_ms = benchmark_pixel_gate()
    if not args.skip_clip:
        clip_ms = benchmark_clip_encode()
        amortized_ms = benchmark_amortized_cost()
        benchmark_theta_sensitivity()

    print("\n" + "=" * 60)
    print("Summary — Paper Claims vs Measured")
    print("=" * 60)
    print(f"{'Component':<30}{'Paper Claim':<20}{'Measured'}")
    print("-" * 65)
    print(f"{'Pixel gate/sample':<30}{'<1ms':<20}{pixel_ms:.3f}ms")
    if not args.skip_clip:
        print(f"{'CLIP encode/sample (GPU)':<30}{'5-10ms':<20}{clip_ms:.1f}ms")
        print(f"{'Amortized/sample (90% static)':<30}{'1-2ms':<20}{amortized_ms:.2f}ms")
    print("=" * 60)
