#!/usr/bin/env python3
"""Fill paper placeholders from v10 result JSONs.

Reads the v10_*.json result files and substitutes placeholder strings
(e.g. STRUCT_CLAUDE, OSS_UNIFORM, SAN_OAI_C) in paper/main.tex with the
actual numbers.  Idempotent: re-running after an additional eval lands
will only fill in placeholders that still resolve.

Usage:
    python experiments/v10/fill_placeholders.py
"""
import json
from math import comb
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
R = PROJECT / "results"
P = PROJECT / "paper" / "main.tex"


def load(name):
    fp = R / name
    if not fp.exists():
        return None
    return json.load(open(fp))


def total(rows):
    return sum(1 for r in rows if r.get("success"))


def mcnemar(rows1, rows2):
    if not rows1 or not rows2:
        return None
    by1 = {r["task_id"]: r["success"] for r in rows1}
    by2 = {r["task_id"]: r["success"] for r in rows2}
    common = set(by1) & set(by2)
    b = sum(1 for t in common if by1[t] and not by2[t])
    c = sum(1 for t in common if by2[t] and not by1[t])
    n = b + c
    if n == 0:
        return 1.0
    k_obs = min(b, c)
    pmid = 0.5 * (comb(n, k_obs) / 2 ** n)
    for k in range(k_obs):
        pmid += comb(n, k) / 2 ** n
    return min(2 * pmid, 1.0)


def n_cloud(rows):
    return total(rows) if (rows and len(rows) >= 100) else None


def n_oss(rows):
    return total(rows) if (rows and len(rows) >= 50) else None


def n_san(rows):
    return total(rows) if (rows and len(rows) >= 5) else None


def main():
    sc = load("v10_structured_claude.json")
    sg = load("v10_structured_gpt54.json")
    sm = load("v10_structured_gemini25.json")
    ou = load("v10_oss_qwen3vl32b_uniform_1fps.json")
    op = load("v10_oss_qwen3vl32b_pixel_diff.json")
    orand = load("v10_oss_qwen3vl32b_random_keyframes.json")
    nso = load("v10_sanity_openai_realtime.json")
    nsg = load("v10_sanity_gemini_live.json")

    claude_sstd = n_cloud(sc)
    gpt_sstd = n_cloud(sg)
    gemini_sstd = n_cloud(sm)
    oss_uni = n_oss(ou)
    oss_pix = n_oss(op)
    oss_rand = n_oss(orand)
    san_oai = n_san(nso)
    san_gli = n_san(nsg)

    def adapter_exercised(d):
        if not d or len(d) < 5:
            return None
        n = 0
        for r in d.values():
            steps = r.get("steps", [])
            any_action = any(
                s.get("action") and s.get("action") not in ("", None) for s in steps
            )
            if any_action or r.get("result_val") not in (None, "", "env_start_failed"):
                n += 1
        return n

    san_oai_ex = adapter_exercised({r["task_id"]: r for r in nso}) if nso else None
    san_gli_ex = adapter_exercised({r["task_id"]: r for r in nsg}) if nsg else None

    p_oss_1 = (
        mcnemar(ou, op)
        if (oss_uni is not None and oss_pix is not None)
        else None
    )
    p_oss_2 = (
        mcnemar(op, orand)
        if (oss_pix is not None and oss_rand is not None)
        else None
    )

    print(f"cloud_struct: claude={claude_sstd} gpt={gpt_sstd} gemini={gemini_sstd}")
    print(f"oss: uniform={oss_uni} pixel={oss_pix} random={oss_rand}")
    print(f"oss_pairwise: p_oss_1={p_oss_1} p_oss_2={p_oss_2}")
    print(f"sanity: oai={san_oai} ex={san_oai_ex}; gli={san_gli} ex={san_gli_ex}")

    tex = P.read_text()

    DYNA_BASELINE = {"claude": 38, "gpt": 37, "gemini": 21}
    DYNA_AOI = {"claude": 82, "gpt": 57, "gemini": 69}

    def fill_struct_inplace(model_key, sstd, tex_in):
        base = DYNA_BASELINE[model_key]
        aoi = DYNA_AOI[model_key]
        if sstd is None:
            return tex_in
        pfmt = sstd - base
        perc = aoi - sstd
        pfmt_str = f"$+${pfmt}" if pfmt >= 0 else f"$-${abs(pfmt)}"
        perc_str = f"$+${perc}" if perc >= 0 else f"$-${abs(perc)}"
        tex_in = tex_in.replace(f"STRUCT\\_{model_key.upper()}", str(sstd))
        tex_in = tex_in.replace(f"PFMT\\_{model_key.upper()}", pfmt_str)
        tex_in = tex_in.replace(f"PRC\\_{model_key.upper()}", perc_str)
        return tex_in

    # IMPORTANT: replace longer placeholders first to avoid prefix-collision
    # bugs (e.g. STRUCT\_GEMINI is a prefix of STRUCT\_GEMINI3; replacing
    # GEMINI first corrupts the GEMINI3 placeholder).  The v10c Gemini 3
    # row uses a distinct token (GEMTHREE) to remove the ambiguity entirely;
    # the comment is here for future contributors who add new keys.
    tex = fill_struct_inplace("claude", claude_sstd, tex)
    tex = fill_struct_inplace("gpt", gpt_sstd, tex)
    tex = fill_struct_inplace("gemini", gemini_sstd, tex)

    if oss_uni is not None:
        tex = tex.replace("OSS\\_UNIFORM", str(oss_uni))
    if oss_pix is not None:
        tex = tex.replace("OSS\\_PIXDIFF", str(oss_pix))
    if oss_rand is not None:
        tex = tex.replace("OSS\\_RANDOM", str(oss_rand))
    if p_oss_1 is not None:
        tex = tex.replace("OSS\\_P1", f"{p_oss_1:.2f}")
    if p_oss_2 is not None:
        tex = tex.replace("OSS\\_P2", f"{p_oss_2:.2f}")

    if san_oai is not None:
        tex = tex.replace("SANITY\\_OPENAI", str(san_oai))
    if san_gli is not None:
        tex = tex.replace("SANITY\\_GEMINI", str(san_gli))
    if san_gli_ex is not None:
        tex = tex.replace("SAN\\_GEM\\_ADAPT", str(san_gli_ex))

    if nso and len(nso) >= 5:
        by_oai = {r["task_id"]: r for r in nso}
        for tid, key in [
            ("C-E1", "C"),
            ("E-E1", "E"),
            ("F-E1", "F1"),
            ("F-E2", "F2"),
            ("I-E1", "I"),
        ]:
            r = by_oai.get(tid)
            cell = "\\checkmark" if (r and r.get("success")) else "\\ding{55}"
            tex = tex.replace(f"SAN\\_OAI\\_{key}", cell)
    if nsg and len(nsg) >= 5:
        by_gli = {r["task_id"]: r for r in nsg}
        for tid, key in [
            ("C-E1", "C"),
            ("E-E1", "E"),
            ("F-E1", "F1"),
            ("F-E2", "F2"),
            ("I-E1", "I"),
        ]:
            r = by_gli.get(tid)
            cell = "\\checkmark" if (r and r.get("success")) else "\\ding{55}"
            tex = tex.replace(f"SAN\\_GEM\\_{key}", cell)

    # ── v10b additions ─────────────────────────────────────────────
    # Variance (3-seed Claude AOI-full)
    seed1 = load("v9_full_100_claude_aoi.json")
    seed2 = load("v10_variance_seed2_claude_aoi.json")
    seed3 = load("v10_variance_seed3_claude_aoi.json")
    seed_scores = [n_cloud(s) for s in (seed1, seed2, seed3)]
    if all(s is not None for s in seed_scores):
        import statistics
        mean = statistics.mean(seed_scores)
        std = statistics.stdev(seed_scores) if len(seed_scores) > 1 else 0.0
        tex = tex.replace("\\texttt{[s2]}", str(seed_scores[1]))
        tex = tex.replace("\\texttt{[s3]}", str(seed_scores[2]))
        tex = tex.replace("\\texttt{[mean]}", f"{mean:.1f}")
        tex = tex.replace("\\texttt{[std]}", f"{std:.1f}")
        tex = tex.replace("\\texttt{[STD\\_PP]}", f"{std:.1f}")
        tex = tex.replace("\\texttt{[MIN\\_PP]}", f"{min(seed_scores):.0f}")

    # Grok-4 main eval
    g_std = load("v10_grok4_standard.json")
    g_aoi = load("v10_grok4_aoi_full.json")
    if n_cloud(g_std) is not None:
        tex = tex.replace("\\texttt{[grok-std]}", str(n_cloud(g_std)))
    if n_cloud(g_aoi) is not None:
        tex = tex.replace("\\texttt{[grok-aoi]}", str(n_cloud(g_aoi)))
    if n_cloud(g_std) is not None and n_cloud(g_aoi) is not None:
        delta = n_cloud(g_aoi) - n_cloud(g_std)
        tex = tex.replace("\\texttt{[grok-delta]}", f"+{delta}")
        p = mcnemar(g_std, g_aoi)
        if p is not None:
            tex = tex.replace("\\texttt{[grok-p]}", f"{p:.1e}" if p < 0.01 else f"{p:.2f}")

    # gpt-realtime-2.0 audio subset
    # NOTE: gpt-realtime-2.0 was attempted but is only available on the GA API
    # tier (our project has beta tier).  Row removed from the streaming table;
    # see Section 7.6 backbone-choice note.
    rt2 = load("v10_subset_openai_realtime_v2.json")
    if rt2 and len(rt2) >= 12:
        by_cat = {"A": 0, "B": 0, "G": 0, "H": 0}
        for r in rt2:
            cat = r["task_id"].split("-")[0]
            if cat in by_cat and r.get("success"):
                by_cat[cat] += 1
        total_rt2 = sum(by_cat.values())
        tex = tex.replace("\\texttt{[A]}", str(by_cat["A"]))
        tex = tex.replace("\\texttt{[B]}", str(by_cat["B"]))
        tex = tex.replace("\\texttt{[G]}", str(by_cat["G"]))
        tex = tex.replace("\\texttt{[H]}", str(by_cat["H"]))
        tex = tex.replace("\\texttt{[Total]}", str(total_rt2))

    # v10c rows: Gemini 3 Flash + Grok-4.3 + Grok-4-fast-reasoning
    # File naming: experiments/v10/run_newer_models.py writes
    #     results/v10c_<short>_<mode>.json
    # where <short> is gemini3flash / grok43 / grok4fast.
    # Placeholder prefixes in paper/main.tex are G3F / GROK43 / GROK4F.
    # Replacement order: longer placeholders first so e.g. G3F_DELTA does not
    # get partially substituted by an earlier G3F_STD rule.
    for tag, ph_prefix in [
        ("gemini3flash", "G3F"),
        ("grok43",       "GROK43"),
        ("grok4fast",    "GROK4F"),
    ]:
        std_rows = load(f"v10c_{tag}_standard.json")
        aoi_rows = load(f"v10c_{tag}_aoi_full.json")
        n_std = n_cloud(std_rows) if std_rows else None
        n_aoi = n_cloud(aoi_rows) if aoi_rows else None

        if n_std is not None and n_aoi is not None:
            delta = n_aoi - n_std
            p = mcnemar(std_rows, aoi_rows)
            tex = tex.replace(
                f"{ph_prefix}\\_DELTA",
                f"$+${delta}" if delta >= 0 else f"$-${abs(delta)}",
            )
            tex = tex.replace(
                f"{ph_prefix}_DELTA",
                f"$+${delta}" if delta >= 0 else f"$-${abs(delta)}",
            )
            if p is not None:
                p_str = f"{p:.1e}" if p < 0.01 else f"{p:.2f}"
                tex = tex.replace(f"{ph_prefix}\\_P", p_str)
                tex = tex.replace(f"{ph_prefix}_P", p_str)
        if n_std is not None:
            tex = tex.replace(f"{ph_prefix}\\_STD", str(n_std))
            tex = tex.replace(f"{ph_prefix}_STD", str(n_std))
        if n_aoi is not None:
            tex = tex.replace(f"{ph_prefix}\\_AOI", str(n_aoi))
            tex = tex.replace(f"{ph_prefix}_AOI", str(n_aoi))

    # Gemini 3 standard_structured (v10c follow-up).
    # Token name "GEMTHREE" rather than "GEMINI3" so the substring
    # STRUCT_GEMINI (used for Gemini 2.5) does not corrupt this placeholder
    # via prefix collision.  In v10c the result is hard-coded directly in
    # the .tex (29 / -7 / +16) because the eval is complete; this block
    # remains for re-runs and consistency with future iterations.
    g3_struct = load("v10c_structured_gemini3.json")
    s = n_cloud(g3_struct) if g3_struct else None
    if s is not None:
        # Gemini 3 Flash standard = 36, AOI full = 45 from the v10c main run.
        base_g3, aoi_g3 = 36, 45
        pfmt = s - base_g3
        perc = aoi_g3 - s
        tex = tex.replace("GEMTHREE\\_STRUCT", str(s))
        tex = tex.replace(
            "GEMTHREE\\_PFMT",
            f"$+${pfmt}" if pfmt >= 0 else f"$-${abs(pfmt)}",
        )
        tex = tex.replace(
            "GEMTHREE\\_PRC",
            f"$+${perc}" if perc >= 0 else f"$-${abs(perc)}",
        )

    # OSS standard_structured rows in Table 6 (Fara + EvoCUA)
    DYNA_BASELINE_OSS = {"fara": 17, "evocua": 18}
    DYNA_AOI_OSS = {"fara": 34, "evocua": 55}
    for key, name in [("fara", "Fara"), ("evocua", "EvoCUA")]:
        rows = load(f"v10_structured_{key}.json")
        s = n_cloud(rows) if rows else None
        if s is not None:
            base = DYNA_BASELINE_OSS[key]
            aoi = DYNA_AOI_OSS[key]
            pfmt = s - base
            perc = aoi - s
            tex = tex.replace(f"STRUCT\\_{key.upper()}", str(s))
            tex = tex.replace(
                f"PFMT\\_{key.upper()}",
                f"$+${pfmt}" if pfmt >= 0 else f"$-${abs(pfmt)}",
            )
            tex = tex.replace(
                f"PRC\\_{key.upper()}",
                f"$+${perc}" if perc >= 0 else f"$-${abs(perc)}",
            )

    P.write_text(tex)
    print("=== Wrote main.tex ===")


if __name__ == "__main__":
    main()
