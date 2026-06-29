#!/usr/bin/env python3
"""Theta sensitivity sweep across multiple task types."""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING)

from benchmark_env.browser_env import BrowserEnvironment
from aoi.keyframe_extractor import KeyframeExtractor

test_tasks = [
    ("D_transient_session_warning.html", "Modal overlay"),
    ("F_anim_carousel.html", "Carousel slides"),
    ("C002_download_toast.html", "Download toast"),
    ("D_transient_flash_sale.html", "Flash sale banner"),
    ("I_stream_counter.html", "Live counter"),
]

thetas = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]

header = "Task                           "
for t in thetas:
    header += f" t={t:.2f}"
print(header)
print("-" * len(header))

for html_file, label in test_tasks:
    env = BrowserEnvironment(html_file)
    env.start()

    frames = []
    for i in range(12):
        time.sleep(0.85)
        frames.append(env.get_screenshot())
    env.stop()

    row = f"{label:<30s} "
    for theta in thetas:
        kfe = KeyframeExtractor(theta=theta, pixel_threshold=0.01, max_keyframes=20)
        for i, frame in enumerate(frames):
            kfe.on_sample(frame, i * 0.85)
        keyframes = kfe.get_and_reset()
        row += f"  {len(keyframes):>4d}"
    print(row)
