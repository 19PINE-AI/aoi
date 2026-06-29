#!/usr/bin/env python3
"""
Patch [pending]/[running]/[deferred] cells in main.tex with the latest
v10 numbers from results/.

Idempotent: each call replaces the latest known value, never duplicates.
"""
import json, math, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper/main.tex"
R = ROOT / "results"

def load(name):
    p = R / name
    return json.load(open(p)) if p.exists() else None

def main():
    text = TEX.read_text()
    changed = False

    # ── Static-50 ─────────────────────────────────────────────────────
    std = load("v10_static50_claude_standard.json")
    aoi = load("v10_static50_claude_aoi.json")

    if std and len(std) == 50:
        n_pass = sum(1 for r in std if r.get("success"))
        avg_steps = sum(r.get("steps_taken", 0) for r in std) / 50
        line = f"Standard & {n_pass} \\scriptsize({100*n_pass/50:.1f}\\%) & {avg_steps:.2f} & 0 & 0 \\\\"
        text2 = re.sub(
            r"Standard & 43 \\scriptsize\(86\.0\\%\) & 3\.26 & 0 & 0 \\\\",
            line, text)
        if text2 != text:
            text = text2; changed = True; print(f"Static Standard: {n_pass}/50, avg_steps={avg_steps:.2f}")

    if aoi and len(aoi) == 50:
        n_pass = sum(1 for r in aoi if r.get("success"))
        avg_steps = sum(r.get("steps_taken", 0) for r in aoi) / 50
        total_kf = sum(sum(s.get("n_keyframes", 0) for s in r.get("steps", []))
                       for r in aoi)
        total_audio = sum(sum(1 for s in r.get("steps", []) if s.get("audio_text",""))
                          for r in aoi)
        line = f"AOI full & {n_pass} \\scriptsize({100*n_pass/50:.1f}\\%) & {avg_steps:.2f} & {total_kf} & {total_audio} \\\\"
        text2 = re.sub(
            r"AOI full & \\texttt\{\[running\]\} & \\texttt\{\[running\]\} & 0 & 0 \\\\",
            line, text)
        if text2 != text:
            text = text2; changed = True
            print(f"Static AOI: {n_pass}/50, avg_steps={avg_steps:.2f}, KF={total_kf}, audio={total_audio}")

    # ── Narration-discarded ───────────────────────────────────────────
    nd = load("v10_narration_discarded_claude.json")
    if nd and len(nd) == 100:
        n_pass = sum(1 for r in nd if r.get("success"))
        # McNemar tests vs aoi_full and aoi_visual_asr
        from math import comb
        def by_id(rs): return {r["task_id"]: bool(r.get("success")) for r in rs}
        def mcnemar(b, c):
            n = b + c
            if n == 0: return 1.0
            k = min(b, c)
            p = sum(comb(n, i) for i in range(0, k+1)) / (2**n)
            return min(1.0, 2*p)

        full = by_id(load("v9_full_100_claude_aoi.json"))
        asr = by_id(load("v9_full_100_claude_aoi_visual_asr.json"))
        nd_d = by_id(nd)

        common_full = set(nd_d) & set(full)
        b_f = sum(1 for k in common_full if not nd_d[k] and full[k])
        c_f = sum(1 for k in common_full if nd_d[k] and not full[k])
        p_full = mcnemar(b_f, c_f)

        common_asr = set(nd_d) & set(asr)
        b_a = sum(1 for k in common_asr if not asr[k] and nd_d[k])
        c_a = sum(1 for k in common_asr if asr[k] and not nd_d[k])
        p_asr = mcnemar(b_a, c_a)

        # Patch the narration ablation table
        line = (
            f"AOI full \\emph{{narration discarded}} (CoT preserved)      "
            f"& {n_pass} & ${p_full:.2e}$".replace("e-0", r"\times10^{-").replace("e+0", r"\times10^{")
        )
        # Simpler: directly inject readable values
        line = (
            f"AOI full \\emph{{narration discarded}} (CoT preserved)      "
            f"& {n_pass} & $p={p_full:.2e}$ \\\\")
        text2 = re.sub(
            r"AOI full \\emph\{narration discarded\} \(CoT preserved\)      & \\texttt\{\[pending\]\} & \\texttt\{\[pending\]\} \\\\",
            line, text)
        if text2 != text:
            text = text2; changed = True
            print(f"Narration discarded: {n_pass}/100")
            print(f"  vs AOI full       : p = {p_full:.2e}")
            print(f"  vs AOI visual+ASR : p = {p_asr:.2e}")

            # Also patch the "Section ... shows result" prose
            # Determine direction
            asr_pass = sum(asr.values())
            full_pass = sum(full.values())
            if abs(n_pass - asr_pass) <= abs(n_pass - full_pass):
                # closer to ASR → CoT-only effect rejected
                interp = (
                    f"The narration-discarded score is {n_pass}\\,\\%, near the visual+ASR baseline "
                    f"of {asr_pass}\\,\\% and ${100*(full_pass-n_pass)/full_pass:.0f}$\\,\\% below the full-AOI "
                    f"score of {full_pass}\\,\\% (McNemar $p={p_full:.2e}$ vs.\\ full).  "
                    f"The +18\\,pp benefit therefore comes almost entirely from \\emph{{persistent text "
                    f"memory across the trajectory}}, not from inference-time chain-of-thought."
                )
            else:
                interp = (
                    f"The narration-discarded score is {n_pass}\\,\\%, close to the full-AOI score "
                    f"of {full_pass}\\,\\% and well above the visual+ASR baseline of {asr_pass}\\,\\%.  "
                    f"This indicates the +18\\,pp benefit is predominantly from \\emph{{inference-time "
                    f"chain-of-thought}} during narration generation, with persistent memory contributing "
                    f"a smaller share."
                )
            text2 = re.sub(
                r"Section~\\ref\{sec:narration_discard\} reports the result\.",
                "Section~\\\\ref{sec:narration_discard} reports the result; in brief: " + interp,
                text)
            if text2 != text:
                text = text2

    if changed:
        TEX.write_text(text)
        print("paper/main.tex updated.")
    else:
        print("No changes (either nothing finished, or already up to date).")

if __name__ == "__main__":
    main()
