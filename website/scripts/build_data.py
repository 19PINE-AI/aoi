#!/usr/bin/env python3
"""Extract paper results into JSON files consumed by the website.

Reads the raw evaluation result files in results/ and writes compact JSON
to website/public/data/.  All numbers are computed from the same files the
paper tables are built from — nothing is hand-entered.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "website" / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)

CATEGORIES = {
    "A_podcast": ("A", "Podcast / Audio"),
    "B_meeting": ("B", "Video Conference"),
    "C_video": ("C", "Video Watching"),
    "D_carousel": ("D", "Carousel / Rotating"),
    "E_dashboard": ("E", "Live Dashboard"),
    "F_transient": ("F", "Transient Notifs"),
    "G_phone": ("G", "Voice / Phone In"),
    "H_interview": ("H", "Voice Interview"),
    "I_collab": ("I", "Collab Editing"),
    "J_game": ("J", "Interactive Game"),
    "S_static": ("S", "Static Baseline"),
}


def load(name):
    return json.loads((RES / name).read_text())


def passed(r):
    fs = r.get("final_score")
    if fs is not None:
        return fs >= 0.5
    return bool(r.get("success"))


def rate(records):
    n = sum(1 for r in records if passed(r))
    return {"pass": n, "total": len(records), "rate": round(100.0 * n / max(1, len(records)), 1)}


def per_category(records):
    by = defaultdict(list)
    for r in records:
        by[r["category"]].append(r)
    return {c: rate(v) for c, v in sorted(by.items())}


def per_difficulty(records):
    by = defaultdict(list)
    for r in records:
        by[r["difficulty"]].append(r)
    return {d: rate(v) for d, v in by.items()}


def summarize(name, label, model, mode, group=None):
    recs = load(name)
    return {
        "file": name,
        "label": label,
        "model": model,
        "mode": mode,
        "group": group,
        **rate(recs),
        "per_category": per_category(recs),
        "per_difficulty": per_difficulty(recs),
    }


# ── 1. Main results: six models, standard vs aoi_full (v9 full 100) ──
MAIN = [
    ("Claude Sonnet 4.6", "v9_full_100_claude_standard.json", "v9_full_100_claude_aoi.json"),
    ("GPT-5.4", "v9_full_100_gpt54_standard.json", "v9_full_100_gpt54_aoi.json"),
    ("Gemini 2.5 Flash", "v9_full_100_gemini25flash_standard.json", "v9_full_100_gemini25flash_aoi.json"),
    ("Grok-4", "v10_grok4_standard.json", "v10_grok4_aoi_full.json"),
    ("EvoCUA-32B", "v9_full_100_evocua32b_standard.json", "v9_full_100_evocua32b_aoi.json"),
    ("Fara-7B", "v9_full_100_fara7b_standard.json", "v9_full_100_fara7b_aoi.json"),
]
main_results = []
for model, std_f, aoi_f in MAIN:
    std, aoi = load(std_f), load(aoi_f)
    main_results.append({
        "model": model,
        "standard": rate(std),
        "aoi_full": rate(aoi),
        "delta": round(rate(aoi)["rate"] - rate(std)["rate"], 1),
        "per_category": {
            "standard": per_category(std),
            "aoi_full": per_category(aoi),
        },
        "per_difficulty": {
            "standard": per_difficulty(std),
            "aoi_full": per_difficulty(aoi),
        },
    })

# ── 2. Ablation tiers (Claude Sonnet 4.6, v9 full 100) ──
ABLATION = [
    ("Standard (screenshot-only)", "v9_full_100_claude_standard.json", "standard"),
    ("Pixel-diff keyframes", "v9_full_100_claude_pixel_diff.json", "pixel_diff"),
    ("Uniform 1 fps", "v9_full_100_claude_uniform_1fps.json", "uniform_1fps"),
    ("Uniform 3 fps", "v9_full_100_claude_uniform_3fps.json", "uniform_3fps"),
    ("Random keyframes", "v9_full_100_claude_random_keyframes.json", "random_keyframes"),
    ("+ CLIP keyframes (AOI visual)", "v9_full_100_claude_aoi_visual.json", "aoi_visual"),
    ("+ ASR (visual + audio)", "v9_full_100_claude_aoi_visual_asr.json", "aoi_visual_asr"),
    ("+ Narration (AOI full)", "v9_full_100_claude_aoi.json", "aoi_full"),
]
ablation = [summarize(f, label, "Claude Sonnet 4.6", mode) for label, f, mode in ABLATION]

# ── 3. Selection-method ablation on open-source Qwen3-VL-32B ──
OSS_SEL = [
    ("Uniform 1 fps", "v10_oss_qwen3vl32b_uniform_1fps.json"),
    ("Random keyframes", "v10_oss_qwen3vl32b_random_keyframes.json"),
    ("Pixel-diff", "v10_oss_qwen3vl32b_pixel_diff.json"),
]
oss_selection = [summarize(f, label, "Qwen3-VL-32B", "selection") for label, f in OSS_SEL]

# ── 4. Theta sweep ──
theta_sweep = []
for f in sorted((RES / "theta_sweep").glob("theta_0*.json")):
    recs = json.loads(f.read_text())
    theta = float(f.stem.split("_")[1])
    r = rate(recs)
    avg_kf = None
    n_kf = [s.get("n_keyframes", 0) for rec in recs for s in rec.get("steps", [])]
    if n_kf:
        avg_kf = round(sum(n_kf) / len(n_kf), 2)
    theta_sweep.append({"theta": theta, **r, "avg_keyframes_per_step": avg_kf})

# ── 5. Streaming baselines (12-task audio subset) ──
subset_openai = load("v10_subset_openai_realtime_v2.json")
subset_gemini = load("v10_subset_gemini_live.json")
subset_ids = [r["task_id"] for r in subset_openai]
claude_aoi = load("v9_full_100_claude_aoi.json")
aoi_on_subset = [r for r in claude_aoi if r["task_id"] in subset_ids]
streaming = [
    {"system": "AOI full (Claude Sonnet 4.6)", **rate(aoi_on_subset)},
    {"system": "OpenAI Realtime", **rate(subset_openai)},
    {"system": "Gemini Live", **rate(subset_gemini)},
]

# ── 6. Newer models / Gemini-3 four-way decomposition ──
fourway = [
    {"mode": "standard", "label": "Standard", **rate(load("v10c_gemini3flash_standard.json"))},
    {"mode": "standard_audio", "label": "Standard + audio", **rate(load("v12_g3flash_standard_audio.json"))},
    {"mode": "aoi_audio", "label": "AOI audio (no keyframes)", **rate(load("v12_g3flash_aoi_audio.json"))},
    {"mode": "aoi_full", "label": "AOI full (with keyframes)", **rate(load("v10c_gemini3flash_aoi_full.json"))},
]
newer = []
for model, std_f, aoi_f in [
    ("Gemini 3 Flash", "v10c_gemini3flash_standard.json", "v10c_gemini3flash_aoi_full.json"),
    ("Grok-4.3", "v10c_grok43_standard.json", "v10c_grok43_aoi_full.json"),
    ("Grok-4-fast", "v10c_grok4fast_standard.json", "v10c_grok4fast_aoi_full.json"),
]:
    std, aoi = load(std_f), load(aoi_f)
    newer.append({"model": model, "standard": rate(std), "aoi_full": rate(aoi),
                  "delta": round(rate(aoi)["rate"] - rate(std)["rate"], 1)})

# ── 7. Static-50 no-degradation check ──
static50 = [
    {"mode": "standard", **rate(load("v10_static50_claude_standard.json"))},
    {"mode": "aoi_full", **rate(load("v10_static50_claude_aoi.json"))},
]

# ── 8. Variance (3 seeds, Claude aoi_full) ──
seeds = [
    {"seed": 1, **rate(load("v9_full_100_claude_aoi.json"))},
    {"seed": 2, **rate(load("v10_variance_seed2_claude_aoi.json"))},
    {"seed": 3, **rate(load("v10_variance_seed3_claude_aoi.json"))},
]

# ── 9. Open-source replication (extensions B1) ──
oss_repl = []
for model, std_f, aoi_f in [
    ("Qwen3-VL-30B-A3B", "extensions/b1_qwen3-vl-30b_standard.json", "extensions/b1_qwen3-vl-30b_aoi_full.json"),
    ("Qwen3-VL-235B-A22B", "extensions/b1_qwen3-vl-235b_standard.json", "extensions/b1_qwen3-vl-235b_aoi_full.json"),
]:
    std, aoi = load(std_f), load(aoi_f)
    oss_repl.append({"model": model, "standard": rate(std), "aoi_full": rate(aoi),
                     "delta": round(rate(aoi)["rate"] - rate(std)["rate"], 1)})

# ── 10. Prompt-format decomposition (extensions A2 + structured runs) ──
prompt_decomp = [
    {"mode": "standard_minimal", "label": "Minimal prompt", **rate(load("extensions/a2_claude_standard_minimal.json"))},
    {"mode": "standard", "label": "Standard", **rate(load("v9_full_100_claude_standard.json"))},
    {"mode": "standard_pageel_only", "label": "+ page elements", **rate(load("extensions/a2_claude_standard_pageel_only.json"))},
    {"mode": "standard_structured", "label": "+ structured scaffold", **rate(load("v10_structured_claude.json"))},
    {"mode": "aoi_full", "label": "AOI full", **rate(load("v9_full_100_claude_aoi.json"))},
]

# ── 11. Narration ablation ──
narration = [
    {"mode": "aoi_visual_asr", "label": "No narration", **rate(load("v9_full_100_claude_aoi_visual_asr.json"))},
    {"mode": "narration_discarded", "label": "Narration generated, then discarded", **rate(load("v10_narration_discarded_claude.json"))},
    {"mode": "aoi_full", "label": "Narration persists (AOI full)", **rate(load("v9_full_100_claude_aoi.json"))},
]

# ── 12. Keyframe-in-context: keyframe images' marginal value WITH narration present ──
# aoi_audio = scaffold + ASR + narration, NO keyframes; aoi_full adds keyframes.
# (aoi_full - aoi_audio) is the keyframe-image contribution in the deployed context.
KF_CONTEXT = [
    ("Claude Sonnet 4.6", "v9_full_100_claude_aoi.json",        "v10_claudeaudio_aoi_audio.json"),
    ("Gemini 2.5 Flash",  "v9_full_100_gemini25flash_aoi.json", "v10_gem25audio_aoi_audio.json"),
    # GPT-5.4 measured via OpenRouter (both modes, same adapter) because the
    # direct OpenAI key lost model.request scope; the OpenAI-direct
    # v10_gpt54audio_aoi_audio.INVALID_401.json run was a 401 failure (0/100).
    ("GPT-5.4",           "v10_gpt54or_aoi_full.json",          "v10_gpt54or_aoi_audio.json"),
    ("Gemini 3 Flash",    "v10c_gemini3flash_aoi_full.json",    "v12_g3flash_aoi_audio.json"),
]
keyframe_context = []
for model, full_f, audio_f in KF_CONTEXT:
    if not (RES / full_f).exists() or not (RES / audio_f).exists():
        continue
    full, audio = load(full_f), load(audio_f)
    if len(audio) < 100:
        continue
    rf, ra = rate(full), rate(audio)
    pcf, pca = per_category(full), per_category(audio)
    per_cat_delta = {c: pcf[c]["pass"] - pca.get(c, {"pass": 0}).get("pass", 0) for c in pcf}
    keyframe_context.append({
        "model": model,
        "aoi_audio": ra, "aoi_full": rf,
        "kf_delta": round(rf["rate"] - ra["rate"], 1),
        "per_category_delta": per_cat_delta,
    })

results = {
    "main_results": main_results,
    "ablation": ablation,
    "oss_selection": oss_selection,
    "theta_sweep": theta_sweep,
    "streaming": streaming,
    "gemini3_fourway": fourway,
    "newer_models": newer,
    "static50": static50,
    "seeds": seeds,
    "oss_replication": oss_repl,
    "prompt_decomposition": prompt_decomp,
    "narration_ablation": narration,
    "keyframe_context": keyframe_context,
    "categories": CATEGORIES,
}
(OUT / "results.json").write_text(json.dumps(results, indent=1))
print("wrote results.json")

# ── Trajectories: full step data for the explorer ──
TRAJ_RUNS = [
    ("claude_aoi", "Claude Sonnet 4.6 — AOI full", "v9_full_100_claude_aoi.json"),
    ("claude_standard", "Claude Sonnet 4.6 — Standard", "v9_full_100_claude_standard.json"),
    ("gpt54_aoi", "GPT-5.4 — AOI full", "v9_full_100_gpt54_aoi.json"),
    ("gpt54_standard", "GPT-5.4 — Standard", "v9_full_100_gpt54_standard.json"),
    ("gemini3_aoi_audio", "Gemini 3 Flash — AOI audio", "v12_g3flash_aoi_audio.json"),
    ("gemini3_aoi_full", "Gemini 3 Flash — AOI full", "v10c_gemini3flash_aoi_full.json"),
    ("fara7b_aoi", "Fara-7B — AOI full", "v9_full_100_fara7b_aoi.json"),
    ("evocua_aoi", "EvoCUA-32B — AOI full", "v9_full_100_evocua32b_aoi.json"),
]
runs_index = []
for run_id, label, fname in TRAJ_RUNS:
    recs = load(fname)
    out = []
    for r in recs:
        out.append({
            "task_id": r["task_id"],
            "category": r["category"],
            "difficulty": r["difficulty"],
            "success": passed(r),
            "steps_taken": r.get("steps_taken"),
            "total_time_s": round(r.get("total_time_s") or 0, 1),
            "error": r.get("error"),
            "steps": [
                {
                    "step": s.get("step"),
                    "action": s.get("action"),
                    "narration": s.get("narration") or "",
                    "audio_text": s.get("audio_text") or "",
                    "n_keyframes": s.get("n_keyframes", 0),
                }
                for s in r.get("steps", [])
            ],
        })
    (OUT / f"traj_{run_id}.json").write_text(json.dumps(out))
    runs_index.append({"id": run_id, "label": label, **rate(recs)})
    print(f"wrote traj_{run_id}.json ({len(out)} tasks)")
(OUT / "runs.json").write_text(json.dumps(runs_index, indent=1))

# ── Task catalog from dynacubench/tasks_v3.py ──
sys.path.insert(0, str(ROOT))
from dynacubench.tasks_v3 import DynaCUBenchV3  # noqa: E402

catalog = [t.to_dict() for t in DynaCUBenchV3()]
(OUT / "tasks.json").write_text(json.dumps(catalog, indent=1))
print(f"wrote tasks.json ({len(catalog)} tasks)")
