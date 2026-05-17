#!/usr/bin/env python3
"""Compute McNemar tests + 95% Wilson CIs on v9 results."""
import json, math, os, sys
from pathlib import Path

RESULTS = Path("/home/ubuntu/adaptive-observation-paper/results")

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0, centre - half), min(1, centre + half)

def mcnemar_exact(b, c):
    """Exact mid-p McNemar's test on discordant pairs.
       b = success in A but not in B, c = success in B but not in A."""
    n = b + c
    if n == 0:
        return 1.0
    # Two-sided exact binomial p-value
    k = min(b, c)
    from math import comb
    p = sum(comb(n, i) for i in range(0, k+1)) / (2**n)
    return min(1.0, 2*p)

def load(name):
    return json.load(open(RESULTS / name))

def task_outcomes(results):
    """Return dict task_id -> bool(success)."""
    return {r["task_id"]: bool(r.get("success")) for r in results}

# --- Main per-model: Standard vs AOI full ---
models = [
    ("Claude 4.6",  "v9_full_100_claude_standard.json",       "v9_full_100_claude_aoi.json"),
    ("GPT-5.4",     "v9_full_100_gpt54_standard.json",        "v9_full_100_gpt54_aoi.json"),
    ("Gemini 2.5",  "v9_full_100_gemini25flash_standard.json","v9_full_100_gemini25flash_aoi.json"),
    ("EvoCUA-32B",  "v9_full_100_evocua32b_standard.json",    "v9_full_100_evocua32b_aoi.json"),
    ("Fara-7B",     "v9_full_100_fara7b_standard.json",       "v9_full_100_fara7b_aoi.json"),
]

print("=== Standard vs AOI-full per model (paired McNemar, 95% Wilson CIs) ===")
print(f"{'Model':12s} {'Std':>6s} {'95% CI':>14s} {'AOI':>6s} {'95% CI':>14s} {'Δ':>6s} {'b':>4s} {'c':>4s} {'p (mid)':>10s}")
for name, std_f, aoi_f in models:
    sa = task_outcomes(load(std_f))
    sb = task_outcomes(load(aoi_f))
    keys = set(sa) & set(sb)
    n = len(keys)
    ka = sum(sa[k] for k in keys)
    kb = sum(sb[k] for k in keys)
    b = sum(1 for k in keys if sa[k] and not sb[k])  # std-only success
    c = sum(1 for k in keys if not sa[k] and sb[k])  # aoi-only success
    p = mcnemar_exact(b, c)
    lo_a, hi_a = wilson_ci(ka, n)
    lo_b, hi_b = wilson_ci(kb, n)
    print(f"{name:12s} {ka:>3d}/{n:>2d}  [{lo_a*100:4.1f},{hi_a*100:4.1f}]  {kb:>3d}/{n:>2d}  [{lo_b*100:4.1f},{hi_b*100:4.1f}]  {kb-ka:+5d}  {b:>3d}  {c:>3d}   {p:8.2e}")

# --- Ablation chain on Claude ---
print("\n=== Claude ablation chain (each row vs immediately previous) ===")
chain = [
    ("Standard",         "v9_full_100_claude_standard.json"),
    ("Uniform 1 FPS",    "v9_full_100_claude_uniform_1fps.json"),
    ("Uniform 3 FPS",    "v9_full_100_claude_uniform_3fps.json"),
    ("Pixel-diff",       "v9_full_100_claude_pixel_diff.json"),
    ("Random keyframes", "v9_full_100_claude_random_keyframes.json"),
    ("AOI visual only",  "v9_full_100_claude_aoi_visual.json"),
    ("AOI visual+ASR",   "v9_full_100_claude_aoi_visual_asr.json"),
    ("AOI full",         "v9_full_100_claude_aoi.json"),
]
prev = None
for name, f in chain:
    sa = task_outcomes(load(f))
    keys = sorted(sa.keys())
    n = len(keys)
    ka = sum(sa[k] for k in keys)
    lo, hi = wilson_ci(ka, n)
    if prev is not None:
        prev_name, prev_outcomes = prev
        b = sum(1 for k in keys if prev_outcomes.get(k) and not sa[k])
        c = sum(1 for k in keys if not prev_outcomes.get(k) and sa[k])
        p = mcnemar_exact(b, c)
        print(f"  {name:18s} {ka:>3d}/{n}  [{lo*100:4.1f},{hi*100:4.1f}]   vs {prev_name:18s} b={b} c={c} p={p:.2e}")
    else:
        print(f"  {name:18s} {ka:>3d}/{n}  [{lo*100:4.1f},{hi*100:4.1f}]")
    prev = (name, sa)

# --- Selection method convergence (vs Standard) ---
print("\n=== Selection methods compared to AOI visual only (do they differ?) ===")
ref = task_outcomes(load("v9_full_100_claude_aoi_visual.json"))
for name, f in [
    ("Uniform 1 FPS",    "v9_full_100_claude_uniform_1fps.json"),
    ("Uniform 3 FPS",    "v9_full_100_claude_uniform_3fps.json"),
    ("Pixel-diff",       "v9_full_100_claude_pixel_diff.json"),
    ("Random keyframes", "v9_full_100_claude_random_keyframes.json"),
]:
    sa = task_outcomes(load(f))
    keys = set(sa) & set(ref)
    b = sum(1 for k in keys if sa[k] and not ref[k])
    c = sum(1 for k in keys if not sa[k] and ref[k])
    p = mcnemar_exact(b, c)
    print(f"  {name:18s} vs AOI-visual: b={b} c={c} p={p:.3f}")
