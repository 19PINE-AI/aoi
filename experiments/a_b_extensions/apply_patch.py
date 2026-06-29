"""
Apply v12 paper patches: read concrete numbers from
results/extensions/SUMMARY.json and substitute {{...}} placeholders in
paper/v12_patch.tex.template, writing the filled LaTeX to
paper/v12_patch.tex.

Use after experiments finish:
    python3 experiments/a_b_extensions/aggregate_results.py
    python3 experiments/a_b_extensions/apply_patch.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "results/extensions"
TPL = ROOT / "paper/v12_patch.tex.template"
OUT = ROOT / "paper/v12_patch.tex"


def fmt_p(p) -> str:
    if p is None: return "--"
    if isinstance(p, str): return p
    if p < 1e-3: return f"$<10^{{-3}}$"
    if p < 1e-2: return f"$<10^{{-2}}$"
    return f"{p:.3f}"


def fmt_rate(r) -> str:
    if r is None: return "--"
    return f"{r:.1f}\\%"


def fmt_int(v) -> str:
    if v is None: return "--"
    return str(v)


def main():
    summary = json.load(open(EXT / "SUMMARY.json"))
    a1 = summary.get("a1", {}).get("rows", {})
    a2 = summary.get("a2", {}).get("rows", {})
    b1 = summary.get("b1", {})

    subs = {}

    # A1: rows are aoi_full_max{N}kf, aoi_full_noise_kf, aoi_full_dup_kf, aoi_full_reorder_kf
    for mode_key, prefix in [("aoi_full_max1kf", "max1"),
                              ("aoi_full_max2kf", "max2"),
                              ("aoi_full_max3kf", "max3"),
                              ("aoi_full_max5kf_ref", "max5"),
                              ("aoi_full_noise_kf", "noise"),
                              ("aoi_full_dup_kf", "dup"),
                              ("aoi_full_reorder_kf", "reorder"),
                              ("aoi_audio_ref", "audio")]:
        r = a1.get(mode_key, {})
        subs[f"{prefix}_pass"] = fmt_int(r.get("pass"))
        subs[f"{prefix}_rate"] = fmt_rate(r.get("rate"))

    # A2: rows are standard_minimal, standard_pageel_only, plus refs
    for mode_key, prefix in [("standard_minimal", "min"),
                              ("standard_pageel_only", "pageel")]:
        r = a2.get(mode_key, {})
        subs[f"{prefix}_pass"] = fmt_int(r.get("pass"))
        subs[f"{prefix}_rate"] = fmt_rate(r.get("rate"))

    # B1: keys are qwen3-vl-235b, qwen3-vl-30b, ...
    for tag, prefix in [("qwen3-vl-235b", "q235b"),
                         ("qwen3-vl-30b", "q30b")]:
        data = b1.get(tag, {})
        std = data.get("standard", {})
        aoi = data.get("aoi_full", {})
        mc = data.get("mcnemar", {})
        subs[f"{prefix}_std"] = fmt_int(std.get("pass"))
        subs[f"{prefix}_aoi"] = fmt_int(aoi.get("pass"))
        delta = (aoi.get("rate", 0) - std.get("rate", 0)) if std and aoi else None
        subs[f"{prefix}_delta"] = (f"$+${delta:.1f}" if delta else "--")
        subs[f"{prefix}_p"] = fmt_p(mc.get("p"))

    # Interpretation placeholders — to fill manually based on numbers
    interp_placeholders = [
        "primary_mechanism", "noise_or_dup", "matches_or_diverges",
        "dilution_or_content", "shows_monotonic_or_not",
        "differs_or_not", "position_matters_or_not",
        "matches_or_not", "matches_or_not2",
        "strengthen_or_qualify", "magnitude",
    ]
    for k in interp_placeholders:
        if k not in subs:
            subs[k] = f"[{k}]"  # leave visible placeholder for manual fill

    tpl = TPL.read_text()
    # Substitute {{key}} with values
    def repl(m):
        key = m.group(1)
        return subs.get(key, f"{{{{{key}}}}}")  # keep placeholder if missing
    out = re.sub(r"\{\{(\w+)\}\}", repl, tpl)
    OUT.write_text(out)
    print(f"Wrote: {OUT}")
    # Report unresolved placeholders
    remaining = re.findall(r"\{\{(\w+)\}\}", out)
    if remaining:
        print(f"Unresolved placeholders ({len(remaining)}): {set(remaining)}")
    else:
        print("All placeholders resolved.")


if __name__ == "__main__":
    main()
