"""
A3: Narration content quality audit.

Goal: For tasks where AOI-full passes but narration-discarded fails,
determine whether the narrations in the AOI-full trajectory contained
the load-bearing fact (memory-hit) or whether the agent succeeded by
some other means (incidental).

Method:
  1. Compare AOI-full vs narration-discarded on 100 dynamic tasks.
  2. Identify the "memory-load" set: tasks where AOI-full passed and
     narration-discarded failed. This is the +8pp signal that should
     be explained by persistent memory.
  3. For each such task, extract the narrations from prior steps in the
     AOI-full trajectory (i.e., narrations that would be visible as
     persistent text memory to the step that produced the winning action).
  4. Run an LLM judge that, given (task instruction, narrations from prior
     steps, the winning action), classifies each task as:
       - "memory_hit": narrations contain the fact the action depends on
       - "incidental": narrations don't contain the fact (the +8 is
         coming from elsewhere)
       - "ambiguous": cannot determine
  5. Output rates + per-task detail.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/ubuntu/adaptive-observation-paper")
sys.path.insert(0, str(ROOT))

AOI_FULL = ROOT / "results/v9_full_100_claude_aoi.json"
NARR_DISC = ROOT / "results/v10_narration_discarded_claude.json"
OUT = ROOT / "results/extensions/a3_narration_audit.json"


def load_runs(path: Path) -> dict[str, dict]:
    """Return task_id → result dict (last successful or last attempt)."""
    data = json.load(open(path))
    if isinstance(data, dict):
        data = list(data.values())
    by_id: dict[str, dict] = {}
    for r in data:
        tid = r["task_id"]
        # If multiple entries per task, keep the last (current)
        by_id[tid] = r
    return by_id


def extract_narrations_before_action(steps: list[dict], winning_step_idx: int) -> list[str]:
    """Return narrations from steps 0..winning_step_idx-1 (memory available at winning step)."""
    narrs = []
    for i, s in enumerate(steps):
        if i >= winning_step_idx:
            break
        n = s.get("narration", "") or ""
        if n and n.strip() and n.strip().lower() not in ("no visual change.", "no visual change", "n/a"):
            narrs.append(f"Step {i+1}: {n.strip()}")
    return narrs


def find_winning_step(steps: list[dict]) -> int:
    """Index (0-based) of the step where the task succeeded."""
    for i, s in enumerate(steps):
        if s.get("success", False):
            return i
    # If never succeeds (shouldn't happen for our filter), use last
    return len(steps) - 1


def call_judge(task_id: str, instruction: str, narrations: list[str], winning_step: dict) -> dict:
    """LLM judge classifies whether narrations contained the load-bearing fact."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    narr_text = "\n".join(narrations) if narrations else "(no prior narrations)"
    audio = (winning_step.get("audio_text", "") or "")[:300]
    action = winning_step.get("action", "")
    narr_at_action = winning_step.get("narration", "")

    prompt = f"""You are auditing whether a CU agent's persistent text memory of past visual observations contained the fact that made its winning action correct.

TASK INSTRUCTION:
  {instruction}

NARRATIONS FROM PRIOR STEPS (persistent text memory available to the agent at the moment of the winning action):
{narr_text}

WINNING STEP DETAILS:
  Action taken: {action}
  Audio captured AT this step (not persistent prior memory): {audio}
  Narration generated AT this step: {narr_at_action}

QUESTION: Did the persistent text memory from prior steps (the narrations above) contain the specific fact, value, or content that the winning action depends on?

Classify into one of:
  memory_hit    — narrations from prior steps explicitly contain the answer / fact / target value / instruction used in the winning action.
  cot_only      — narrations from prior steps do NOT contain the answer; the agent succeeded due to in-context info available at the winning step (current audio, current screenshot, narration generated at the same step). The +8pp would be a chain-of-thought effect, not memory.
  incidental    — the agent succeeded but the connection to narration is unclear; the prior narrations are generic ("a slide is shown"), not the specific fact.
  ambiguous     — cannot tell from this data.

Respond with ONLY a JSON object: {{"label": "<one of memory_hit | cot_only | incidental | ambiguous>", "reason": "<one sentence>"}}"""

    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    # Strip code fences
    if raw.startswith("```"):
        raw = "\n".join(line for line in raw.split("\n") if not line.startswith("```"))
    try:
        out = json.loads(raw)
    except Exception:
        # Best-effort recovery
        out = {"label": "ambiguous", "reason": f"parse_error: {raw[:200]}"}
    return out


def main():
    full = load_runs(AOI_FULL)
    nd = load_runs(NARR_DISC)

    # Sanity
    common = set(full) & set(nd)
    full_only_pass = [t for t in common if full[t]["success"] and not nd[t]["success"]]
    nd_only_pass = [t for t in common if not full[t]["success"] and nd[t]["success"]]
    both_pass = [t for t in common if full[t]["success"] and nd[t]["success"]]
    both_fail = [t for t in common if not full[t]["success"] and not nd[t]["success"]]

    print(f"Coverage: {len(common)} task IDs in common")
    print(f"  AOI-full PASS, ND FAIL: {len(full_only_pass)}  ← the +8pp memory signal")
    print(f"  AOI-full FAIL, ND PASS: {len(nd_only_pass)}")
    print(f"  Both PASS: {len(both_pass)}")
    print(f"  Both FAIL: {len(both_fail)}")

    # Load instructions
    from dynacubench.tasks_v3 import DynaCUBenchV3
    bench = DynaCUBenchV3(html_tasks_dir=Path("benchmark_env/html_tasks"))
    instructions = {t.task_id: t.instruction for t in bench}

    audit_results = []
    cat_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    print(f"\nAuditing {len(full_only_pass)} memory-load tasks...")
    for i, tid in enumerate(sorted(full_only_pass)):
        result = full[tid]
        steps = result.get("steps", [])
        if not steps:
            continue
        winning_idx = find_winning_step(steps)
        narrations = extract_narrations_before_action(steps, winning_idx)
        instr = instructions.get(tid, "(unknown)")
        winning_step = steps[winning_idx]

        try:
            judgement = call_judge(tid, instr, narrations, winning_step)
        except Exception as e:
            judgement = {"label": "ambiguous", "reason": f"call_failed: {e}"}

        cat = result.get("category", "?")
        cat_counts[cat][judgement["label"]] += 1

        audit_results.append({
            "task_id": tid,
            "category": cat,
            "difficulty": result.get("difficulty", "?"),
            "n_prior_narrations": len(narrations),
            "winning_step_idx": winning_idx,
            "winning_action": winning_step.get("action", ""),
            "narrations_before_action": narrations,
            "winning_step_narration": winning_step.get("narration", ""),
            "winning_step_audio": (winning_step.get("audio_text", "") or "")[:300],
            "judgement": judgement,
        })

        print(f"[{i+1:2d}/{len(full_only_pass)}] {tid:<8} {judgement['label']:<14} {judgement['reason'][:80]}")

    # Aggregate
    label_counts = defaultdict(int)
    for r in audit_results:
        label_counts[r["judgement"]["label"]] += 1

    summary = {
        "n_audited": len(audit_results),
        "label_counts": dict(label_counts),
        "by_category": {k: dict(v) for k, v in cat_counts.items()},
    }

    out_obj = {
        "summary": summary,
        "audit_results": audit_results,
        "coverage": {
            "full_only_pass": full_only_pass,
            "nd_only_pass": nd_only_pass,
            "both_pass_count": len(both_pass),
            "both_fail_count": len(both_fail),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out_obj, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"  Audited: {summary['n_audited']}")
    for lbl, n in label_counts.items():
        pct = 100 * n / max(summary["n_audited"], 1)
        print(f"  {lbl:<14}: {n} ({pct:.1f}%)")
    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()
