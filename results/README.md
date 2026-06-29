# DynaCU-Bench evaluation results

Raw per-task evaluation records. Every number in the paper and on the website
is computed from these files — nothing is hand-entered. Each file is a JSON
list of per-task records: `task_id`, `category`, `difficulty`, `success` /
`final_score`, `steps_taken`, `total_time_s`, and a `steps` array (per step:
`action`, `narration`, `audio_text`, `n_keyframes`, latency).

A task counts as passed when `final_score >= 0.5` (falling back to `success`).

## Naming scheme

| Prefix | Meaning |
|--------|---------|
| `v9_full_100_*`   | Canonical 100-task run (DynaCU-Bench: 10 categories × 10). Claude ablation ladder + the main GPT-5.4 / Gemini-2.5 / EvoCUA / Fara runs. |
| `v10_*`           | Follow-on experiments on the same task set: Grok-4, Static-50, variance seeds, structured-prompt control, narration-discarded, audio-only, OSS selection, real-content, and the streaming `subset_*` runs. |
| `v10c_*`          | Re-run batch for current-gen closed models: Gemini-3 Flash, Grok-4.3, Grok-4-fast. |
| `v12_*`           | Gemini-3 audio-mode runs for the four-way decomposition. |
| `extensions/`     | Appendix experiments: `a1_*` keyframe causal probe, `a2_*` prompt decomposition, `a3_*` narration audit, `b1_*` Qwen3-VL open-source replication. |
| `theta_sweep/`    | CLIP threshold (θ) calibration sweep, θ = 0.02 … 0.30. |
| `subset_*`        | 12-task spoken-content subset (3 each: Podcast, Meeting, Phone, Interview) used for the streaming baselines. |
| `_archive/`       | Superseded development runs (early 10-task and per-category iterations, `full_eval*`, `headless_*`). Kept for provenance; **not** used by any reproduction script. |

## Canonical files → paper table / figure / website block

Generated/consumed by `website/scripts/build_data.py` and
`experiments/compute_stats.py` (significance), `experiments/compute_tokens.py`
(cost), `experiments/analyze_keyframe_context.py`.

| Paper artifact | Files |
|----------------|-------|
| **Table 1 — main results, 9 models** | `v9_full_100_{claude,gpt54,gemini25flash,evocua32b,fara7b}_{standard,aoi}.json`, `v10_grok4_{standard,aoi_full}.json`, `v10c_{gemini3flash,grok43,grok4fast}_{standard,aoi_full}.json` |
| **Component ablation (Claude ladder)** | `v9_full_100_claude_{standard,pixel_diff,uniform_1fps,uniform_3fps,random_keyframes,aoi_visual,aoi_visual_asr,aoi}.json` |
| **Selection invariance (Qwen3-VL-32B)** | `v10_oss_qwen3vl32b_{uniform_1fps,random_keyframes,pixel_diff}.json` |
| **θ sensitivity** | `theta_sweep/theta_0.0*.json` |
| **Table 5 — streaming baselines** | `v10_subset_{gemini_live,openai_realtime,grok_voice,grok_voice_noscaffold,openai_realtime_ws,openai_realtime_ws_noscaffold}.json` (+ Claude AOI subset of `v9_full_100_claude_aoi.json`) |
| **Gemini-3 four-way decomposition** | `v10c_gemini3flash_standard.json`, `v12_g3flash_standard_audio.json`, `v12_g3flash_aoi_audio.json`, `v10c_gemini3flash_aoi_full.json` |
| **Static-50 no-degradation** | `v10_static50_claude_{standard,aoi}.json` |
| **Variance (3 seeds)** | `v9_full_100_claude_aoi.json`, `v10_variance_seed{2,3}_claude_aoi.json` |
| **Prompt-format decomposition** | `extensions/a2_claude_standard_{minimal,pageel_only}.json`, `v10_structured_claude.json` |
| **Narration ablation** | `v9_full_100_claude_aoi_visual_asr.json`, `v10_narration_discarded_claude.json`, `v9_full_100_claude_aoi.json` |
| **Keyframe-in-context (per model)** | `v10_claudeaudio_aoi_audio.json`, `v10_gem25audio_aoi_audio.json`, `v10_gpt54or_{aoi_full,aoi_audio}.json`, `v12_g3flash_aoi_audio.json` |
| **Open-source replication (Qwen3-VL)** | `extensions/b1_qwen3-vl-{30b,235b}_{standard,aoi_full}.json` |
| **Gate activity (per-step)** | `v9_full_100_claude_aoi.json` |
| **DynaCU-Real-Local** | `v10_realcontent_claude_{standard,aoi}.json` |
| **Appendix: keyframe causal probe (Gemini-3)** | `extensions/a1_g3_*.json` |
| **Appendix: narration audit** | `extensions/a3_narration_audit.json` |

## Regenerating tables from these files

```bash
python experiments/compute_stats.py          # McNemar tests + Wilson CIs
python experiments/compute_tokens.py         # token + cost accounting
python experiments/analyze_keyframe_context.py
python website/scripts/build_data.py         # → website/public/data/*.json
```

To produce the results from scratch instead of regenerating from these files,
see the reproduction section of the top-level `README.md`.
