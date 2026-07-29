# Anonymous Code/Data Appendix

We generate this archive from the live project files with
`supplementary/make_anonymous_snapshot.ps1`. We copy the files instead of
maintaining a second implementation, so the project tree remains our source of
truth.

## Included material

| Archive path | Source path | Purpose |
|---|---|---|
| `aoi/` | `aoi/` | AOI implementation, model adapters, audio processing, and streaming adapters |
| `benchmark_env/` | `benchmark_env/` | Browser environment and task assets |
| `dynacubench/` | `dynacubench/` | Benchmark definitions, generators, and evaluators |
| `experiments/` | `experiments/` | Maintained experiment runners and analysis scripts |
| `results/` | `results/` | Raw per-task records used for paper tables and figures |
| `tests/` | `tests/` | Automated tests |
| `requirements.txt` | `requirements.txt` | Python dependencies |
| `.env.example` | `.env.example` | Placeholder-only environment configuration |
| `PROJECT_README.md` | `README.md` | An anonymized copy of the project documentation |

We record the SHA-256 digest of every included file in
`FILE_MANIFEST_SHA256.txt`. We include the maintained implementation and the raw
records used for our reported analyses. `ANONYMOUS_LICENSE.txt` provides the
code and data terms for this review artifact.

## Primary reproduction paths

Install the dependencies and configure API endpoints using `.env.example`.
The maintained evaluation entry point is `experiments/browser_eval.py`.
Existing raw records can be analyzed without API access:

```bash
python experiments/compute_stats.py
python experiments/compute_tokens.py
python experiments/analyze_keyframe_context.py
```

`results/README.md` maps submitted results to the corresponding tables,
figures, ablations, and diagnostic analyses.

## Paper-to-code map

| Paper section | Implementation |
|---|---|
| “Inter-Step Keyframe Capture” (sec:keyframes); Algorithm 1 | `aoi/keyframe_extractor.py` |
| “Volume-Gated Audio Observation” (sec:audio) | `aoi/audio_pipeline.py`, `aoi/whisper_service.py` |
| “Visual Narration for Long-Term Context” (sec:narration); “Observation Record” (sec:obsrecord) | `aoi/observation_record.py`, `aoi/cu_model.py` |
| “Design Principles” (sec:design-principles) | `dynacubench/tasks_v3.py`, `benchmark_env/` |
| “Evaluation Protocol” (sec:evaluation-protocol) | `dynacubench/llm_evaluator.py`, `experiments/browser_eval.py` |
| “Setup” (sec:setup); “Main Results” (sec:main-results) | `experiments/browser_eval.py`, `experiments/compute_stats.py` |
| “Comparison with Streaming Multimodal Baselines” (sec:streaming) | `aoi/realtime_baselines.py` |

## Known reproducibility limits

We used provider-default sampling for hosted model calls and did not supply API
seed arguments. We repeated the headline configuration across three seeds, but
we did not retain their values or seed-setting procedure in the included run
metadata. We also did not retain the CPU model, exact operating-system release,
CUDA/driver versions, a complete dependency lock, every hosted model snapshot,
or the ASR launch device in the run metadata.
