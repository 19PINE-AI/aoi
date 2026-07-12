"""
Generate LaTeX tables from experimental results.
Produces the tables used in the paper.
"""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

# V9 full 100-task results
V9_FILES = {
    "Claude Sonnet 4.6 Standard": "v9_full_100_claude_standard.json",
    "Claude Sonnet 4.6 Uniform 1FPS": "v9_full_100_claude_uniform_1fps.json",
    "Claude Sonnet 4.6 Uniform 3FPS": "v9_full_100_claude_uniform_3fps.json",
    "Claude Sonnet 4.6 Pixel-diff": "v9_full_100_claude_pixel_diff.json",
    "Claude Sonnet 4.6 Random KF": "v9_full_100_claude_random_keyframes.json",
    "Claude Sonnet 4.6 AOI Visual": "v9_full_100_claude_aoi_visual.json",
    "Claude Sonnet 4.6 AOI Visual+ASR": "v9_full_100_claude_aoi_visual_asr.json",
    "Claude Sonnet 4.6 AOI Full": "v9_full_100_claude_aoi.json",
    "GPT-5.4 Standard": "v9_full_100_gpt54_standard.json",
    "GPT-5.4 AOI Full": "v9_full_100_gpt54_aoi.json",
    "EvoCUA-32B Standard": "v9_full_100_evocua32b_standard.json",
    "EvoCUA-32B AOI Full": "v9_full_100_evocua32b_aoi.json",
    "Fara-7B Standard": "v9_full_100_fara7b_standard.json",
    "Fara-7B AOI Full": "v9_full_100_fara7b_aoi.json",
}

# Gemini 2.5 Flash results (added after main experiments)
GEMINI_FILES = {
    "Gemini 2.5 Flash Standard": "v9_full_100_gemini25flash_standard.json",
    "Gemini 2.5 Flash AOI Full": "v9_full_100_gemini25flash_aoi.json",
}

STATIC_FILES = {
    "Claude Standard (static)": "static_claude_standard.json",
    "Claude AOI Full (static)": "static_claude_aoi.json",
}

CATS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def load_results(filename):
    with open(RESULTS_DIR / filename) as f:
        return json.load(f)


def per_category(data):
    cats = {}
    for r in data:
        cat = r["task_id"][0]
        if cat not in cats:
            cats[cat] = {"pass": 0, "total": 0}
        cats[cat]["total"] += 1
        if r["success"]:
            cats[cat]["pass"] += 1
    return cats


def per_difficulty(data):
    diffs = {"E": {"pass": 0, "total": 0}, "M": {"pass": 0, "total": 0},
             "H": {"pass": 0, "total": 0}}
    for r in data:
        d = r["task_id"].split("-")[1][0]
        diffs[d]["total"] += 1
        if r["success"]:
            diffs[d]["pass"] += 1
    return diffs


def main():
    print("=" * 80)
    print("TABLE 2: Main Results")
    print("=" * 80)

    for name, fname in V9_FILES.items():
        data = load_results(fname)
        passed = sum(1 for r in data if r["success"])
        total = len(data)
        print(f"{name:<40} {passed:>3}/{total}")

    print("\n" + "=" * 80)
    print("TABLE 3: Per-Category Results")
    print("=" * 80)
    header = f"{'Config':<40}" + "".join(f"{c:>5}" for c in CATS) + f"{'Tot':>6}"
    print(header)
    for name, fname in V9_FILES.items():
        data = load_results(fname)
        cats = per_category(data)
        row = f"{name:<40}"
        for c in CATS:
            if c in cats:
                row += f"{cats[c]['pass']:>5}"
            else:
                row += f"{'--':>5}"
        passed = sum(1 for r in data if r["success"])
        row += f"{passed:>6}"
        print(row)

    print("\n" + "=" * 80)
    print("TABLE 4: Difficulty Breakdown")
    print("=" * 80)
    for name in ["Claude Sonnet 4.6 Standard", "Claude Sonnet 4.6 AOI Full",
                  "GPT-5.4 Standard", "GPT-5.4 AOI Full",
                  "EvoCUA-32B Standard", "EvoCUA-32B AOI Full",
                  "Fara-7B Standard", "Fara-7B AOI Full"]:
        data = load_results(V9_FILES[name])
        d = per_difficulty(data)
        print(f"{name:<40} E={d['E']['pass']:>2}/{d['E']['total']}  "
              f"M={d['M']['pass']:>2}/{d['M']['total']}  "
              f"H={d['H']['pass']:>2}/{d['H']['total']}")

    print("\n" + "=" * 80)
    print("TABLE 6: Theta Sweep")
    print("=" * 80)
    for f in sorted((RESULTS_DIR / "theta_sweep").glob("theta_0.*.json")):
        data = load_results(f"theta_sweep/{f.name}")
        theta = f.stem.replace("theta_", "")
        passed = sum(1 for r in data if r["success"])
        avg_steps = sum(r["steps_taken"] for r in data) / len(data)
        print(f"theta={theta}  {passed}/40  ({passed/40*100:.1f}%)  "
              f"avg_steps={avg_steps:.1f}")

    print("\n" + "=" * 80)
    print("TABLE 7: Efficiency Stats")
    print("=" * 80)
    for name in ["Claude Sonnet 4.6 Standard", "Claude Sonnet 4.6 AOI Full",
                  "GPT-5.4 Standard", "GPT-5.4 AOI Full",
                  "EvoCUA-32B Standard", "EvoCUA-32B AOI Full",
                  "Fara-7B Standard", "Fara-7B AOI Full"]:
        data = load_results(V9_FILES[name])
        n = len(data)
        avg_steps = sum(r["steps_taken"] for r in data) / n
        avg_time = sum(r["total_time_s"] for r in data) / n
        avg_model = sum(r.get("total_model_latency_ms", 0) for r in data) / n / 1000
        avg_obs = sum(r.get("total_obs_overhead_ms", 0) for r in data) / n / 1000
        print(f"{name:<40} steps={avg_steps:>5.1f}  time={avg_time:>6.1f}s  "
              f"model={avg_model:>5.1f}s  obs={avg_obs:>5.1f}s")


    # Gemini results (if available)
    print("\n" + "=" * 80)
    print("GEMINI 2.5 Flash Results")
    print("=" * 80)
    for name, fname in GEMINI_FILES.items():
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            data = load_results(fname)
            passed = sum(1 for r in data if r["success"])
            total = len(data)
            print(f"{name:<40} {passed:>3}/{total}")
            cats = per_category(data)
            row = f"  per-cat: "
            for c in CATS:
                if c in cats:
                    row += f"{c}={cats[c]['pass']}/{cats[c]['total']} "
            print(row)
        else:
            print(f"{name:<40} (not yet available)")

    # Static verification
    print("\n" + "=" * 80)
    print("STATIC VERIFICATION")
    print("=" * 80)
    for name, fname in STATIC_FILES.items():
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            data = load_results(fname)
            passed = sum(1 for r in data if r["success"])
            total = len(data)
            avg_steps = sum(r["steps_taken"] for r in data) / total
            print(f"{name:<40} {passed:>3}/{total}  avg_steps={avg_steps:.1f}")
        else:
            print(f"{name:<40} (not yet available)")


if __name__ == "__main__":
    main()
