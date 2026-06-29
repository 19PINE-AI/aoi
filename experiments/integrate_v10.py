#!/usr/bin/env python3
"""Integrate v10 experiment results: print summary and McNemar p-values.

Produces the values needed to fill the [pending] / [running] cells in the
paper.
"""
import json, math, sys
from pathlib import Path
from collections import Counter

R = Path(__file__).resolve().parents[1] / "results"

def load(name):
    p = R / name
    if not p.exists():
        return None
    return json.load(open(p))

def by_id(rs):
    return {r["task_id"]: bool(r.get("success")) for r in rs}

def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k+1)) / (2**n)
    return min(1.0, 2*p)

def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z*z/n
    ctr = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0, ctr-half)*100, min(1, ctr+half)*100

def diff_breakdown(rs):
    pass_d = Counter(); total_d = Counter()
    for r in rs:
        d = r["difficulty"]
        total_d[d] += 1
        if r.get("success"):
            pass_d[d] += 1
    return pass_d, total_d

def steps_kf_audio(rs):
    n_steps = sum(r.get("steps_taken", 0) for r in rs)
    n_kf = sum(sum(s.get("n_keyframes", 0) for s in r.get("steps", []))
               for r in rs)
    n_audio = sum(sum(1 for s in r.get("steps", []) if s.get("audio_text",""))
                  for r in rs)
    return n_steps, n_kf, n_audio

print("=" * 70)
print("Static-50 verification")
print("=" * 70)
for name in ("v10_static50_claude_standard.json",
             "v10_static50_claude_aoi.json"):
    rs = load(name)
    if not rs:
        print(f"  {name}: not yet available")
        continue
    n = len(rs); n_pass = sum(1 for r in rs if r.get("success"))
    lo, hi = wilson(n_pass, n)
    avg_steps = sum(r.get("steps_taken", 0) for r in rs) / max(n, 1)
    n_steps, n_kf, n_audio = steps_kf_audio(rs)
    print(f"  {name}: {n_pass}/{n} ({100*n_pass/n:.1f}%, 95% CI [{lo:.1f}, {hi:.1f}])")
    print(f"    avg steps = {avg_steps:.2f}, total keyframes = {n_kf}, audio steps = {n_audio}")
    pd, td = diff_breakdown(rs)
    for d in ("easy", "medium", "hard"):
        print(f"    {d}: {pd[d]}/{td[d]}")

print()
print("=" * 70)
print("Narration-discarded ablation (Claude)")
print("=" * 70)
nd = load("v10_narration_discarded_claude.json")
aoi = load("v9_full_100_claude_aoi.json")
asr = load("v9_full_100_claude_aoi_visual_asr.json")
if nd and aoi and asr:
    nd_pass = sum(1 for r in nd if r.get("success"))
    aoi_pass = sum(1 for r in aoi if r.get("success"))
    asr_pass = sum(1 for r in asr if r.get("success"))
    n_nd = len(nd); n_aoi = len(aoi); n_asr = len(asr)

    nd_d = by_id(nd); aoi_d = by_id(aoi); asr_d = by_id(asr)
    common_aoi = set(nd_d) & set(aoi_d)
    common_asr = set(nd_d) & set(asr_d)
    b1 = sum(1 for k in common_aoi if not nd_d[k] and aoi_d[k])
    c1 = sum(1 for k in common_aoi if nd_d[k] and not aoi_d[k])
    p1 = mcnemar(c1, b1)
    b2 = sum(1 for k in common_asr if not asr_d[k] and nd_d[k])
    c2 = sum(1 for k in common_asr if asr_d[k] and not nd_d[k])
    p2 = mcnemar(b2, c2)

    print(f"  AOI visual+ASR (no narration generated):  {asr_pass}/{n_asr}")
    print(f"  AOI full narration-discarded:              {nd_pass}/{n_nd}")
    print(f"  AOI full (narration retained):             {aoi_pass}/{n_aoi}")
    print()
    print(f"  Narration discarded vs AOI full:           p = {p1:.2e}")
    print(f"  Narration discarded vs AOI visual+ASR:     p = {p2:.2e}")
    print()
    if abs(nd_pass - asr_pass) <= 5:
        print("  Interpretation: narration-discarded ≈ visual+ASR")
        print("                  → +18pp gain is from PERSISTENT MEMORY, not CoT.")
    elif abs(nd_pass - aoi_pass) <= 5:
        print("  Interpretation: narration-discarded ≈ AOI full")
        print("                  → +18pp gain is from inference-time CoT.")
    else:
        print("  Interpretation: narration-discarded sits between the two")
        print("                  → both CoT and persistent memory contribute.")
else:
    print("  Narration-discarded eval not yet complete.")
    if not nd: print("    waiting for: v10_narration_discarded_claude.json")
    print()
