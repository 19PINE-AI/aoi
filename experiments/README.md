# Experiments

Evaluation harness and analysis scripts. All paths are relative to the repo root;
run scripts from there (e.g. `python -m experiments.browser_eval ...`).

## Main harness

| Script | Purpose |
|--------|---------|
| **`browser_eval.py`** | The evaluation harness. Runs CU models over DynaCU-Bench under any observation mode and writes per-task JSON to `results/`. Run as `python -m experiments.browser_eval --models <m> --modes standard aoi_full --output-dir results/<run>`. |

Key flags: `--models` (aliases from `aoi/cu_model.py`), `--modes` (`standard`,
`standard_structured`, `standard_audio`, `aoi_visual`, `aoi_audio`, `aoi_full`,
`aoi_interactive`, plus selection ablations `uniform_1fps` / `uniform_3fps` /
`pixel_diff` / `random_keyframes` / `aoi_visual_asr`, A1 keyframe probes
`aoi_full_{max1kf,max2kf,max3kf,max5kf,noise_kf,dup_kf,reorder_kf}`, and A2 prompt
probes `standard_{minimal,traj_only,pageel_only}`), `--category`, `--difficulty`,
`--task-prefix`, `--max-tasks`, `--max-steps` (default 15), `--output-dir`.

## Analysis (regenerate paper tables from `results/`)

| Script | Output |
|--------|--------|
| `compute_stats.py` | McNemar significance tests + Wilson confidence intervals for the main table and ablations. |
| `compute_tokens.py` | Token counts and dollar-cost accounting (efficiency table). |
| `analyze_keyframe_context.py` | Keyframe-in-context marginal value (AOI-full vs AOI-audio) per model. |
| `latency_benchmark.py` | Observation-overhead / latency measurements. |

These read committed files in `results/` (see [`results/README.md`](../results/README.md))
and need no API keys.

## Serving local models

| Script | Purpose |
|--------|---------|
| `serve_local_model.sh <alias>` | Start a vLLM OpenAI-compatible server for an open-source model (Fara-7B, UI-TARS, EvoCUA, Qwen3-VL, …). Override the interpreter with `PYTHON=...`. |
| `launch_evocua.sh` | Convenience launcher for EvoCUA-32B. |

## Appendix experiments — `a_b_extensions/`

| Script | Appendix |
|--------|----------|
| `a1_keyframe_probe.py`, `a1_per_task_drilldown.py` | A1 — keyframe causal probe on Gemini-3 (budget / noise / duplicate / reorder variants). |
| `a2_prompt_decomp.py` | A2 — prompt-format decomposition (minimal → +trajectory → +page-elements → +scaffold). |
| `a3_narration_audit.py` | A3 — LLM audit of whether persisted narrations carry load-bearing facts. |
| `b1_open_source_replication.py` (+ `b1_runner.sh`) | B1 — independent open-source replication (Qwen3-VL via OpenRouter). |
| `aggregate_results.py`, `apply_patch.py` | Roll up extension results / patch paper placeholders. |

## Author-workflow & one-off runners — `_archive/`

These produced specific runs during the study and are kept for provenance only. They
are **not** part of the end-user reproduction path, may reference archived result
files, and are not guaranteed to run as-is. They live under `_archive/`:

- `update_paper.py`, `integrate_v10.py`, `generate_tables.py` — patch numbers / emit
  LaTeX into `paper/main.tex` (superseded by `compute_stats.py` + the website's
  `build_data.py`).
- `run_standard_vs_aoi.py`, `run_ablation.py`, `run_theta_sweep.py` / `theta_sweep.py`,
  `run_oss_selection.py`, `run_realtime_subset.py`, `run_v10_experiments.py` — focused
  drivers around `browser_eval`.
- `v10/` — the current-gen rerun batch: streaming baselines (`run_realtime_ws*.py`,
  `run_grok_voice*.py`, `run_streaming_sanity.py`), newer-model evals
  (`run_newer_models.py`, `run_any_main.py`, `run_grok_main.py`), variance
  (`run_variance.py`), structured-prompt isolation (`run_structured.py`), and shell
  orchestrators.
- `run_full_eval.py`, `run_10task_eval.py`, `run_headless_ablation.py`,
  `headless_runner.py`, `mock_cu_model.py`, `real_demo.py` — early development
  harnesses, superseded by `browser_eval.py`. `mock_cu_model.py` (offline fixture
  model) and `headless_runner.py` (synthetic no-browser runner) are imported by the
  other archived harnesses.
