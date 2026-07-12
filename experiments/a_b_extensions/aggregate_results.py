"""
Aggregate results from A1, A2, A3, B1 extensions and emit tables for the paper.

Produces:
  results/extensions/SUMMARY.md         human-readable summary
  results/extensions/SUMMARY.json       machine-readable totals + breakdowns
  results/extensions/a1_per_category.json  per-category breakdowns for A1
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from collections import defaultdict
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "results/extensions"


def load_run(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.load(open(path))
    except Exception:
        return []


def mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Return (b, c, p) for paired binary outcomes a and b.

    b = # tasks pass in A but not B; c = # tasks pass in B but not A.
    Uses scipy binomtest if available; else 2 * min(binomial(b+c, b, 0.5),
    binomial(b+c, c, 0.5)).
    """
    assert len(a) == len(b)
    bb = sum(1 for x, y in zip(a, b) if x and not y)
    cc = sum(1 for x, y in zip(a, b) if y and not x)
    n = bb + cc
    if n == 0:
        return bb, cc, 1.0
    try:
        from scipy.stats import binomtest
        res = binomtest(min(bb, cc), n, 0.5, alternative="two-sided")
        return bb, cc, res.pvalue
    except Exception:
        # Mid-p approximation
        from math import comb
        p_total = sum(comb(n, k) for k in range(min(bb, cc) + 1)) / (2 ** n)
        return bb, cc, min(1.0, 2 * p_total)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5) / denom
    return (max(0.0, center - half) * 100, min(1.0, center + half) * 100)


def index_by_task(runs: list[dict]) -> dict[str, dict]:
    return {r["task_id"]: r for r in runs}


def aligned_outcomes(a: dict[str, dict], b: dict[str, dict]) -> tuple[list, list, list]:
    """Return aligned (task_ids, a_success, b_success) over the intersection."""
    common = sorted(set(a) & set(b))
    av = [bool(a[t].get("success")) for t in common]
    bv = [bool(b[t].get("success")) for t in common]
    return common, av, bv


# ── A1: keyframe causal probe on Gemini 3 ──
def aggregate_a1() -> dict:
    modes = ["aoi_full_max1kf", "aoi_full_max2kf", "aoi_full_max3kf",
             "aoi_full_noise_kf", "aoi_full_dup_kf", "aoi_full_reorder_kf"]
    rows = {}
    by_cat = {}
    for m in modes:
        f = EXT / f"a1_g3_{m}.json"
        d = load_run(f)
        if not d:
            continue
        idx = index_by_task(d)
        n_pass = sum(1 for r in d if r.get("success"))
        n_total = len(d)
        low, high = wilson_ci(n_pass, n_total)
        rows[m] = {
            "pass": n_pass, "total": n_total,
            "rate": 100 * n_pass / max(n_total, 1),
            "ci_low": low, "ci_high": high,
        }
        per_cat = defaultdict(lambda: [0, 0])
        for r in d:
            cat = r.get("category", "?")
            per_cat[cat][1] += 1
            if r.get("success"): per_cat[cat][0] += 1
        by_cat[m] = {c: f"{v[0]}/{v[1]}" for c, v in sorted(per_cat.items())}

    # Reference: use existing v10c_gemini3flash_aoi_full.json restricted to the
    # same 50-task subset (B/C/D/E/F).
    ref = load_run(ROOT / "results/v10c_gemini3flash_aoi_full.json")
    ref_idx = index_by_task(ref)
    subset_cats = {"B_meeting", "C_video", "D_carousel", "E_dashboard", "F_transient"}
    subset_ids = [t for t, r in ref_idx.items() if r.get("category") in subset_cats]
    if subset_ids:
        n_pass = sum(1 for t in subset_ids if ref_idx[t].get("success"))
        low, high = wilson_ci(n_pass, len(subset_ids))
        rows["aoi_full_max5kf_ref"] = {
            "pass": n_pass, "total": len(subset_ids),
            "rate": 100 * n_pass / len(subset_ids),
            "ci_low": low, "ci_high": high,
            "note": "Reference from v10c_gemini3flash_aoi_full.json (same subset)",
        }
        # also use aoi_audio reference (no keyframes)
        # v12_g3flash_aoi_audio.json contains the full 100-task aoi_audio run
        # already used in the paper's Section 5.5 4-way decomposition.
        aoi_audio = load_run(ROOT / "results/v12_g3flash_aoi_audio.json")
        if not aoi_audio:
            aoi_audio = load_run(ROOT / "results/v10c_gemini3flash_aoi_audio.json")
        if aoi_audio:
            aa = index_by_task(aoi_audio)
            common = [t for t in subset_ids if t in aa]
            n_pass = sum(1 for t in common if aa[t].get("success"))
            low, high = wilson_ci(n_pass, len(common))
            rows["aoi_audio_ref"] = {
                "pass": n_pass, "total": len(common),
                "rate": 100 * n_pass / max(len(common), 1),
                "ci_low": low, "ci_high": high,
                "note": "Reference: aoi_audio (no keyframes)",
            }

    # Pairwise McNemar tests between modes that share task IDs
    # Mainly: max1kf vs max3kf, max3kf vs max5kf, max5kf vs noise, max5kf vs dup
    mcnemar = {}
    a1_results = {m: index_by_task(load_run(EXT / f"a1_g3_{m}.json")) for m in modes}
    pairs = [
        ("aoi_full_max1kf", "aoi_full_max3kf"),
        ("aoi_full_max1kf", "aoi_full_noise_kf"),
        ("aoi_full_max3kf", "aoi_full_noise_kf"),
        ("aoi_full_max3kf", "aoi_full_dup_kf"),
        ("aoi_full_noise_kf", "aoi_full_dup_kf"),
        ("aoi_full_max3kf", "aoi_full_reorder_kf"),
    ]
    for a, b in pairs:
        ai, bi = a1_results.get(a, {}), a1_results.get(b, {})
        if not ai or not bi: continue
        _, av, bv = aligned_outcomes(ai, bi)
        if not av: continue
        bb, cc, p = mcnemar_exact(av, bv)
        mcnemar[f"{a} vs {b}"] = {
            "b_only_a": bb, "c_only_b": cc, "p": p, "n_common": len(av),
        }

    return {"rows": rows, "mcnemar": mcnemar, "by_category": by_cat}


# ── A2: prompt-format decomposition on Claude ──
def aggregate_a2() -> dict:
    modes = ["standard_minimal", "standard_pageel_only"]
    rows = {}
    for m in modes:
        f = EXT / f"a2_claude_{m}.json"
        d = load_run(f)
        if not d:
            continue
        n_pass = sum(1 for r in d if r.get("success"))
        n_total = len(d)
        low, high = wilson_ci(n_pass, n_total)
        rows[m] = {
            "pass": n_pass, "total": n_total,
            "rate": 100 * n_pass / max(n_total, 1),
            "ci_low": low, "ci_high": high,
        }

    # References from existing Claude data
    refs = {
        "standard": ROOT / "results/v9_full_100_claude_standard.json",
        "standard_structured": ROOT / "results/v10_structured_claude.json",
        "aoi_full": ROOT / "results/v9_full_100_claude_aoi.json",
    }
    for name, path in refs.items():
        d = load_run(path)
        if not d:
            continue
        n_pass = sum(1 for r in d if r.get("success"))
        n_total = len(d)
        low, high = wilson_ci(n_pass, n_total)
        rows[name + "_ref"] = {
            "pass": n_pass, "total": n_total,
            "rate": 100 * n_pass / max(n_total, 1),
            "ci_low": low, "ci_high": high,
        }

    # Pairwise McNemar tests
    a2_idx = {m: index_by_task(load_run(EXT / f"a2_claude_{m}.json")) for m in modes}
    a2_idx["standard"] = index_by_task(load_run(refs["standard"]))
    a2_idx["standard_structured"] = index_by_task(load_run(refs["standard_structured"]))

    pairs = [
        ("standard_minimal", "standard"),
        ("standard_minimal", "standard_pageel_only"),
        ("standard_pageel_only", "standard_structured"),
        ("standard", "standard_structured"),
    ]
    mcnemar = {}
    for a, b in pairs:
        ai, bi = a2_idx.get(a, {}), a2_idx.get(b, {})
        if not ai or not bi: continue
        _, av, bv = aligned_outcomes(ai, bi)
        if not av: continue
        bb, cc, p = mcnemar_exact(av, bv)
        mcnemar[f"{a} vs {b}"] = {
            "b_only_a": bb, "c_only_b": cc, "p": p, "n_common": len(av),
        }
    return {"rows": rows, "mcnemar": mcnemar}


# ── A3: existing narration audit ──
def aggregate_a3() -> dict:
    f = EXT / "a3_narration_audit.json"
    if not f.exists():
        return {}
    return json.load(open(f)).get("summary", {})


# ── B1: open-source model replication ──
def aggregate_b1() -> dict:
    out = {}
    for tag in ["qwen3-vl-235b", "qwen3-vl-30b", "qwen25-vl-72b"]:
        modes_data = {}
        for mode in ["standard", "aoi_full"]:
            f = EXT / f"b1_{tag}_{mode}.json"
            d = load_run(f)
            if not d:
                continue
            n_pass = sum(1 for r in d if r.get("success"))
            n_total = len(d)
            low, high = wilson_ci(n_pass, n_total)
            modes_data[mode] = {
                "pass": n_pass, "total": n_total,
                "rate": 100 * n_pass / max(n_total, 1),
                "ci_low": low, "ci_high": high,
            }
        if modes_data:
            # McNemar within model
            std = index_by_task(load_run(EXT / f"b1_{tag}_standard.json"))
            aoi = index_by_task(load_run(EXT / f"b1_{tag}_aoi_full.json"))
            if std and aoi:
                _, av, bv = aligned_outcomes(std, aoi)
                bb, cc, p = mcnemar_exact(av, bv)
                modes_data["mcnemar"] = {
                    "standard_only_pass": bb, "aoi_only_pass": cc,
                    "p": p, "n_common": len(av),
                }
            out[tag] = modes_data
    return out


def emit_summary():
    a1 = aggregate_a1()
    a2 = aggregate_a2()
    a3 = aggregate_a3()
    b1 = aggregate_b1()

    summary = {"a1": a1, "a2": a2, "a3": a3, "b1": b1}
    with open(EXT / "SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown report
    lines = ["# Extensions Summary — A/B Experiments\n"]

    lines.append("## A1: Keyframe Causal Probe on Gemini 3 Flash\n")
    lines.append("Subset: 50 tasks from B, C, D, E, F (visual-active categories).\n")
    lines.append("| Mode | Pass/Total | Rate | 95% Wilson CI |")
    lines.append("|------|-----------|------|----------------|")
    for mode in ["aoi_full_max1kf", "aoi_full_max2kf", "aoi_full_max3kf",
                 "aoi_full_max5kf_ref", "aoi_full_noise_kf", "aoi_full_dup_kf",
                 "aoi_full_reorder_kf", "aoi_audio_ref"]:
        r = a1.get("rows", {}).get(mode)
        if not r:
            lines.append(f"| {mode} | (pending) | -- | -- |")
            continue
        lines.append(f"| {mode} | {r['pass']}/{r['total']} | {r['rate']:.1f}% | [{r['ci_low']:.1f}, {r['ci_high']:.1f}] |")

    if a1.get("mcnemar"):
        lines.append("\n**Pairwise McNemar p-values:**")
        for k, v in a1["mcnemar"].items():
            lines.append(f"- {k}: b={v['b_only_a']}, c={v['c_only_b']}, p={v['p']:.3g}, n={v['n_common']}")

    if a1.get("by_category"):
        lines.append("\n**Per-category breakdown:**")
        cats = sorted({c for d in a1["by_category"].values() for c in d})
        header = "| Mode | " + " | ".join(cats) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(cats) + 1))
        for mode, cat_data in a1["by_category"].items():
            row = f"| {mode} | " + " | ".join(cat_data.get(c, "-") for c in cats) + " |"
            lines.append(row)

    lines.append("\n## A2: Prompt-Format Decomposition (Claude Sonnet 4.6)\n")
    lines.append("| Mode | Pass/Total | Rate | 95% Wilson CI |")
    lines.append("|------|-----------|------|----------------|")
    for mode in ["standard_minimal", "standard_ref", "standard_pageel_only",
                 "standard_structured_ref", "aoi_full_ref"]:
        r = a2.get("rows", {}).get(mode)
        if not r:
            lines.append(f"| {mode} | (pending) | -- | -- |")
            continue
        lines.append(f"| {mode} | {r['pass']}/{r['total']} | {r['rate']:.1f}% | [{r['ci_low']:.1f}, {r['ci_high']:.1f}] |")
    if a2.get("mcnemar"):
        lines.append("\n**Pairwise McNemar p-values:**")
        for k, v in a2["mcnemar"].items():
            lines.append(f"- {k}: b={v['b_only_a']}, c={v['c_only_b']}, p={v['p']:.3g}, n={v['n_common']}")

    lines.append("\n## A3: Narration Content Quality Audit (Claude Sonnet 4.6)\n")
    if a3:
        lines.append(f"- Audited memory-load tasks: {a3.get('n_audited', 0)}")
        for k, v in a3.get("label_counts", {}).items():
            pct = 100 * v / max(a3["n_audited"], 1)
            lines.append(f"- {k}: {v} ({pct:.1f}%)")
        lines.append("\nDetailed JSON: results/extensions/a3_narration_audit.json")

    lines.append("\n## B1: Open-Source Model Replication\n")
    lines.append("| Model | Standard | AOI full | Δ | McNemar p |")
    lines.append("|-------|----------|---------|----|-----------|")
    for tag, data in b1.items():
        std = data.get("standard", {})
        aoi = data.get("aoi_full", {})
        mc = data.get("mcnemar", {})
        delta = aoi.get("rate", 0) - std.get("rate", 0) if std and aoi else None
        p_str = f"{mc['p']:.3g}" if isinstance(mc.get("p"), float) else "-"
        delta_str = f"+{delta:.1f}pp" if delta is not None else "-"
        lines.append(
            f"| {tag} | "
            f"{std.get('pass', '-')}/{std.get('total', '-')} | "
            f"{aoi.get('pass', '-')}/{aoi.get('total', '-')} | "
            f"{delta_str} | "
            f"{p_str} |"
        )

    with open(EXT / "SUMMARY.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    emit_summary()
