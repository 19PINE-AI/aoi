"""
A1 per-task drilldown: identify on which specific tasks each keyframe variant
helps or hurts.

For each task in the 50-task subset, line up the outcome under:
  - max1kf, max3kf, max5kf (= existing aoi_full), noise_kf, dup_kf, reorder_kf
  - aoi_audio (no keyframes, from existing v12 run)

The matrix reveals patterns:
  * Tasks where ALL keyframe variants succeed → keyframes don't matter
  * Tasks where only aoi_audio succeeds → keyframes hurt
  * Tasks where only high-budget keyframes succeed → keyframes help
  * Tasks where noise_kf ≈ max5kf but dup_kf ≠ max5kf → dilution mechanism
  * Tasks where dup_kf ≈ max5kf but noise_kf ≠ max5kf → distraction mechanism
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "results/extensions"

MODES = ["aoi_full_max1kf", "aoi_full_max2kf", "aoi_full_max3kf",
         "aoi_full_noise_kf", "aoi_full_dup_kf", "aoi_full_reorder_kf"]


def load_outcomes(path: Path) -> dict[str, bool]:
    if not path.exists(): return {}
    try:
        d = json.load(open(path))
        return {r["task_id"]: bool(r.get("success")) for r in d}
    except Exception:
        return {}


def main():
    # Probe: all A1 modes
    probe_outcomes = {m: load_outcomes(EXT / f"a1_g3_{m}.json") for m in MODES}
    # Anchor: existing aoi_full (= max5kf)
    aoi_full_outcomes = load_outcomes(ROOT / "results/v10c_gemini3flash_aoi_full.json")
    aoi_audio_outcomes = load_outcomes(ROOT / "results/v12_g3flash_aoi_audio.json")

    # Subset: B, C, D, E, F categories
    SUBSET_CATS = {"B_meeting", "C_video", "D_carousel", "E_dashboard", "F_transient"}
    # Get list of task IDs in the subset
    try:
        ref_data = json.load(open(ROOT / "results/v10c_gemini3flash_aoi_full.json"))
        subset_tids = sorted(r["task_id"] for r in ref_data if r["category"] in SUBSET_CATS)
    except Exception:
        subset_tids = []

    print(f"\n{'task':<8} ", end="")
    cols = MODES + ["aoi_full(=max5)", "aoi_audio"]
    for m in cols:
        print(f"{m[:14]:<15}", end="")
    print()
    print("-" * (8 + 15 * len(cols)))

    pattern_counts = defaultdict(int)
    for tid in subset_tids:
        row_vals = []
        for m in MODES:
            v = probe_outcomes[m].get(tid)
            row_vals.append("✓" if v is True else ("✗" if v is False else "-"))
        row_vals.append("✓" if aoi_full_outcomes.get(tid) else "✗")
        row_vals.append("✓" if aoi_audio_outcomes.get(tid) else "✗")

        print(f"{tid:<8} ", end="")
        for v in row_vals:
            print(f"{v:<15}", end="")
        print()

        # Pattern: tuple of (kf_helps, dilution, distraction) signatures
        # Compare aoi_audio (no kf) vs max5kf (kf with novel content) vs
        # noise_kf (kf-budget with noise) vs dup_kf (kf-budget with dup of screenshot)
        audio_v = aoi_audio_outcomes.get(tid, None)
        max5_v = aoi_full_outcomes.get(tid, None)
        if audio_v is None or max5_v is None: continue
        if audio_v == max5_v:
            pattern_counts["kf_neutral"] += 1
        elif audio_v and not max5_v:
            pattern_counts["kf_hurts"] += 1
        else:
            pattern_counts["kf_helps"] += 1

    print("\n=== Audio (no kf) vs Max5kf pattern counts ===")
    for k, v in pattern_counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
