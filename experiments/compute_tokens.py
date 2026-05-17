#!/usr/bin/env python3
"""Estimate token usage and dollar cost per task across configurations.

Token estimates are derived from the actual eval logs:
  - Each step produces a context_text (text portion)
  - Each step has 0..K keyframe images + 1 post-action screenshot
  - Each image is approximated at 258 tokens (Anthropic vision token approximation)
  - Audio transcripts inline as text

Pricing (May 2026, public API list prices):
  - Claude Sonnet 4.6:        $3 / Mtok input,  $15 / Mtok output
  - GPT-5.4:                  $5 / Mtok input,  $15 / Mtok output
  - Gemini 2.5 Flash:         $0.30/ Mtok input, $2.50 / Mtok output
"""
import json, math
from pathlib import Path

RESULTS = Path("/home/ubuntu/adaptive-observation-paper/results")

PRICING = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet":      (3.0, 15.0),
    "gpt-5.4":            (5.0, 15.0),
    "gpt-54":             (5.0, 15.0),
    "gemini-2.5-flash":   (0.30, 2.50),
    "gemini-25-flash":    (0.30, 2.50),
    "evocua-32b":         (0.0, 0.0),  # local
    "fara-7b":            (0.0, 0.0),  # local
}

def price(model):
    m = model.lower()
    for k, p in PRICING.items():
        if k in m:
            return p
    return (0.0, 0.0)

def tokens_per_step(step):
    text = step.get("narration", "") + step.get("audio_text", "") + step.get("action", "")
    text_tokens = max(int(len(text.split()) * 1.3), 0)
    image_tokens = (1 + step.get("n_keyframes", 0)) * 258
    return text_tokens, image_tokens

def aggregate(file):
    results = json.load(open(file))
    total_input = 0
    total_output = 0
    n_tasks = 0
    n_steps_total = 0
    for r in results:
        steps = r.get("steps", [])
        n_steps_total += len(steps)
        n_tasks += 1
        # Output tokens: ~narration + action per step (~50 tokens approx)
        # Input tokens: cumulative context grows with trajectory
        cum_text_input = 0
        for i, s in enumerate(steps):
            tt, it = tokens_per_step(s)
            # Input for this step: prior trajectory text (cumulative) + current obs
            # Approximate trajectory growth: assume ~80 text tokens per prior step + current images
            prior_step_text = sum(int(len(steps[j].get("narration","").split() + steps[j].get("audio_text","").split()) * 1.3) for j in range(i))
            input_tokens = prior_step_text + tt + it + 200  # 200 = system prompt + task instr
            output_tokens = max(int(len((s.get("narration","") + s.get("action","")).split()) * 1.3), 30)
            total_input += input_tokens
            total_output += output_tokens
    return total_input, total_output, n_tasks, n_steps_total

if __name__ == "__main__":
    print(f"{'File':52s} {'In tokens':>14s} {'Out tokens':>12s} {'$ / 100 tasks':>14s}")
    files = [
        ("v9_full_100_claude_standard.json", "claude-sonnet-4-6"),
        ("v9_full_100_claude_aoi.json", "claude-sonnet-4-6"),
        ("v9_full_100_claude_uniform_1fps.json", "claude-sonnet-4-6"),
        ("v9_full_100_claude_pixel_diff.json", "claude-sonnet-4-6"),
        ("v9_full_100_claude_random_keyframes.json", "claude-sonnet-4-6"),
        ("v9_full_100_claude_aoi_visual.json", "claude-sonnet-4-6"),
        ("v9_full_100_claude_aoi_visual_asr.json", "claude-sonnet-4-6"),
        ("v9_full_100_gpt54_standard.json", "gpt-5.4"),
        ("v9_full_100_gpt54_aoi.json", "gpt-5.4"),
        ("v9_full_100_gemini25flash_standard.json", "gemini-2.5-flash"),
        ("v9_full_100_gemini25flash_aoi.json", "gemini-2.5-flash"),
        ("v9_full_100_evocua32b_standard.json", "evocua-32b"),
        ("v9_full_100_evocua32b_aoi.json", "evocua-32b"),
        ("v9_full_100_fara7b_standard.json", "fara-7b"),
        ("v9_full_100_fara7b_aoi.json", "fara-7b"),
    ]
    for f, model in files:
        path = RESULTS / f
        if not path.exists():
            continue
        ti, to, nt, ns = aggregate(path)
        pin, pout = price(model)
        cost = (ti / 1e6) * pin + (to / 1e6) * pout
        avg_tokens_per_task = (ti + to) / max(nt, 1)
        print(f"{f:52s} {ti:>14,d} {to:>12,d} {cost:>13.2f}   (~{avg_tokens_per_task/1000:.1f}k tok/task, {ns/nt:.1f} steps/task)")
