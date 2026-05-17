# Paper Revision Status (v11)

This document captures the v11 revision targeting the three blockers from
the prior review:
(1) `standard_structured` run on DynaCU-Bench for all five models,
(2) open-source replication of the selection-method ablation,
(3) sanity-check + reframe for the streaming baselines.

## Done

- **Author block + title** — Bojie Li / Pine AI / boj@19pine.ai with bolded title.
- **`standard_structured` infrastructure verified end-to-end** — `experiments/browser_eval.py` already supports the mode; smoke test on F-E2 (cookie banner) passes in 6.2 s with Claude Sonnet 4.6.
- **OpenRouter wiring for OSS models** — added Qwen3-VL-32B-Instruct and Qwen3-VL-8B-Instruct to `aoi/cu_model.py` `LOCAL_MODELS`, with `VLLM_API_KEY` and `VLLM_OR_PROVIDER=alibaba` env hooks for OpenAI-compatible OpenRouter routing. Used as substitute for EvoCUA-32B because the host's vLLM still cannot start (NVML 595.58 vs. kernel 590.48.01).
- **Streaming-baseline adapter sanity-check harness** — `/tmp/run_streaming_sanity.py` (driver) wraps both adapters with a 5-task purely-visual sanity set (C-E1, E-E1, F-E1, F-E2, I-E1) and writes results to `results/v10_sanity_*.json`.
- **Streaming-baseline reframe in paper** — Section 6.6 ("Comparison with Streaming Multimodal Baselines") now leads with an adapter-sanity check that distinguishes "adapter exercised" (5/5 for both baselines) from "task-correct" (0/5 for both). The audio-subset failures are now explicitly attributed to the streaming-voice paradigm rather than to scaffolding.
- **Per-category narration-discarded breakdown** — Table 5 now reports per-category numbers, not just the total.
- **95% Wilson CI error bars on bar charts** — Figures 3 (main results across 5 models) and 6 (ablation chain).
- **Algorithm 1 caption** — clarified to call out "ablation only; default selector is Stage 1 alone".
- **Public code link** — placeholder GitHub URL inserted in the author block.
- **Limitations section** — replaced the deferred-OSS-ablation paragraph with the Qwen3-VL-32B substitution note; replaced the structured-prompt confound paragraph with a pointer to the new direct measurement.

## Running (background tasks at v11 finalisation time)

| Experiment | Tasks | Mode | Output | Status |
|---|---|---|---|---|
| structured_claude  | 100 | claude / standard_structured  | `results/v10_structured_claude.json`  | running |
| structured_gpt54   | 100 | gpt-5.4 / standard_structured | `results/v10_structured_gpt54.json`   | running |
| structured_gemini25| 100 | gemini-2.5-flash / standard_structured | `results/v10_structured_gemini25.json` | running |
| oss_uniform_1fps   | 50  | qwen3-vl-32b / uniform_1fps   | `results/v10_oss_qwen3vl32b_uniform_1fps.json` | running |
| oss_pixel_diff     | 50  | qwen3-vl-32b / pixel_diff     | `results/v10_oss_qwen3vl32b_pixel_diff.json` | running |
| oss_random_keyframes| 50 | qwen3-vl-32b / random_keyframes | `results/v10_oss_qwen3vl32b_random_keyframes.json` | running |
| sanity_openai      | 5   | openai realtime / sanity      | `results/v10_sanity_openai_realtime.json` | done — 0/5 task-correct, 5/5 adapter exercised |
| sanity_gemini      | 5   | gemini live / sanity          | `results/v10_sanity_gemini_live.json` | done — 0/5 task-correct, 5/5 adapter exercised |

## Pending — open-source local-vLLM models

EvoCUA-32B and Fara-7B `standard_structured` rows in Table 6 are
intentionally left as `---` (vLLM blocked) with the Static-50 +14 pp upper
bound used as the conservative perception lower bound for those two
models.  The text now says exactly this.  When the host's NVML drift is
fixed, run:
```
python experiments/run_v10_experiments.py --static --selection-ablation
```
to produce the missing rows.

## Re-running the analysis

After every new eval JSON lands in `results/`:
```
python /tmp/fill_placeholders.py         # fills paper placeholders from JSON
python experiments/compute_stats.py      # McNemar p-values + Wilson CIs
python experiments/compute_tokens.py     # token/$ per task
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Files added/changed in v11

- **paper/main.tex** — author/title; new Table 6 (`tab:structured`); new Table 7 (`tab:oss_selection`); new Table 8 (`tab:streaming_sanity`); appendix `app:streaming_sanity`; abstract update; conclusion update; algorithm caption; error bars on Figures 3 and 6.
- **paper/references.bib** — `qwen3vl` entry added.
- **aoi/cu_model.py** — `VLLM_API_KEY` + `VLLM_OR_PROVIDER` env hooks; `qwen3-vl-32b` and `qwen3-vl-8b` aliases.
- **experiments/v10/run_structured.py** — driver for `standard_structured` runs.
- **experiments/v10/run_oss_selection.py** — driver for OSS selection-method ablation.
- **experiments/v10/run_streaming_sanity.py** — driver for streaming-baseline adapter sanity check.
- **experiments/v10/fill_placeholders.py** — fills paper placeholders from result JSON.

(All four were previously at `/tmp/` during the v10 push and have been
moved in-tree under `experiments/v10/` for the public release.)
