#!/usr/bin/env python3
"""Keyframe-in-context decomposition: aoi_full vs aoi_audio per model.

aoi_audio = scaffold + ASR + narration, NO keyframes.
aoi_full  = aoi_audio + inter-step keyframe images.
So (aoi_full - aoi_audio) is the marginal contribution of the keyframe
IMAGES, measured in the full deployment context (audio + narration present).

Run after the aoi_audio runs land:
    python experiments/analyze_keyframe_context.py
"""
import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"
CATS = list("ABCDEFGHIJ")

# (model label, aoi_full file, aoi_audio file)
MODELS = [
    ("Claude Sonnet 4.6", "v9_full_100_claude_aoi.json",        "v10_claudeaudio_aoi_audio.json"),
    ("Gemini 2.5 Flash",  "v9_full_100_gemini25flash_aoi.json", "v10_gem25audio_aoi_audio.json"),
    ("GPT-5.4 (OpenRouter)", "v10_gpt54or_aoi_full.json",       "v10_gpt54or_aoi_audio.json"),
    ("Gemini 3 Flash",    "v10c_gemini3flash_aoi_full.json",    "v12_g3flash_aoi_audio.json"),
]


def load(fn):
    p = RES / fn
    return json.loads(p.read_text()) if p.exists() else None


def percat(recs):
    d = {c: 0 for c in CATS}
    for r in recs:
        if r.get("success"):
            d[r["task_id"].split("-")[0]] += 1
    return d


def total(recs):
    return sum(1 for r in recs if r.get("success"))


print(f"{'Model':22} {'aoi_audio':>9} {'aoi_full':>8} {'kf Δ':>6}   per-category kf Δ (full−audio)")
print("-" * 100)
rows = []
for label, full_f, audio_f in MODELS:
    full, audio = load(full_f), load(audio_f)
    if full is None or audio is None:
        print(f"{label:22}  MISSING ({'full' if full is None else ''} {'audio' if audio is None else ''})"
              f"  n_full={len(full) if full else 0} n_audio={len(audio) if audio else 0}")
        continue
    if len(audio) < 100:
        print(f"{label:22}  aoi_audio INCOMPLETE: {len(audio)}/100")
    tf, ta = total(full), total(audio)
    pf, pa = percat(full), percat(audio)
    catdelta = "  ".join(f"{c}{pf[c]-pa[c]:+d}" for c in CATS)
    print(f"{label:22} {ta:>9} {tf:>8} {tf-ta:>+6}   {catdelta}")
    rows.append({"model": label, "aoi_audio": ta, "aoi_full": tf, "kf_delta": tf - ta,
                 "per_cat_delta": {c: pf[c] - pa[c] for c in CATS}})

out = RES / "keyframe_context_summary.json"
out.write_text(json.dumps(rows, indent=2))
print(f"\nWrote {out}")
