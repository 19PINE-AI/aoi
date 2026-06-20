# Agent-Computer Observation Interface (AOI) & DynaCU-Bench

A model-agnostic **perception layer** that gives existing computer-use (CU) agents
adaptive visual and audio perception of *dynamic* screen content — video, animations,
transient UI events, meetings, notifications, and spoken instructions — with **zero
retraining**, plus **DynaCU-Bench**, a benchmark of dynamic browser tasks that
screenshot-only agents cannot solve.

This repository accompanies the paper *"Agent-Computer Observation Interfaces Enable
Dynamic Computer Use"* (Bojie Li, Pine AI). The source for the paper is in
[`paper/main.tex`](paper/main.tex).

## What problem does this solve?

Every current CU agent ties observation to action: one screenshot per action step,
3–5 s apart, and no audio. Between screenshots the agent is blind and deaf. The AOI
**decouples observation (continuous, adaptive) from action (discrete)**: it sits
between the environment and any image-based CU model, watches the full interval
between agent steps, and converts the screen and audio streams into the sparse images
and text the model already accepts.

It has three gated components, each preceded by a sub-millisecond gate that produces
nothing on static, silent content:

1. **Inter-step keyframe capture** — continuous ~3 Hz screen sampling with pixel-change
   gating (optional CLIP semantic filtering); 0–5 keyframe images per step.
2. **Volume-gated audio observation** — RMS-energy gate followed by Whisper large-v3
   speech transcription (speech only).
3. **Visual narration** — the CU model itself narrates new visual content as a
   side-output each step; the text persists in the trajectory after images are pruned.

Across eight CU models (closed and open-source, 7B to frontier scale) the AOI yields
**+17 to +48 pp** on DynaCU-Bench with no retraining. See the paper for the full
results, ablations, per-model component analysis, and streaming-baseline comparison.

## Repository layout

| Path | Contents |
|------|----------|
| [`aoi/`](aoi/) | The perception layer: keyframe extractor, audio observer/Whisper service, observation record, agent loop, CU-model adapters, streaming baselines. |
| [`dynacubench/`](dynacubench/) | Benchmark definitions, task generators, DOM/LLM/hybrid evaluators, Static-50 and DynaCU-Real-Local builders. |
| [`benchmark_env/`](benchmark_env/) | Playwright browser environment and the self-contained HTML task files (categories A–J, plus static and v2 backups). |
| [`experiments/`](experiments/) | Evaluation harness (`browser_eval.py`), ablation/probe scripts, table generation, statistics, token accounting. |
| [`results/`](results/) | Raw per-run evaluation JSON used to produce the paper tables. |
| [`paper/`](paper/) | LaTeX source, figures, and the compiled PDF. |
| [`docs/`](docs/benchmark_design.md) | DynaCU-Bench design document. |
| [`tests/`](tests/) | Unit tests for the keyframe extractor, observation record, and benchmark. |

## DynaCU-Bench

100 dynamic browser tasks across 10 categories (10 tasks each: 3 easy, 4 medium,
3 hard), spanning three capability axes — audio perception (A), visual-temporal
perception (V), and real-time interaction (I):

| Cat. | Domain | Axes |
|------|--------|------|
| A | Podcast comprehension | A |
| B | Meeting participation | A+V+I |
| C | Video / screencast | V |
| D | Carousel / animation | V |
| E | Live dashboard | V |
| F | Transient UI events | V |
| G | Phone / voice call | A+I |
| H | Interview / voice | A+I |
| I | Collaborative editing | V+I |
| J | Browser games | V+I |

Each task is a self-contained HTML file (audio via the Web Speech API / edge-TTS,
logic and evaluation embedded). Companion sets: **Static-50** (no-degradation control
on purely static pages) and **DynaCU-Real-Local** (cross-engine ASR robustness with
`espeak` audio and real `asciinema` screencasts).

## Quick start

```bash
# System deps: PulseAudio (virtual audio I/O), a Chromium for Playwright
pip install -r requirements.txt        # playwright, openai-clip, faster-whisper, vllm, ...
playwright install chromium

# Run the benchmark: standard vs. AOI-full on Claude Sonnet 4.6
python -m experiments.browser_eval \
    --models claude-sonnet-4-6 \
    --modes standard aoi_full \
    --output-dir results/run1
```

Key flags: `--models` (one or more CU models), `--modes` (`standard`, `aoi_full`,
`standard_structured`, `aoi_audio`, `aoi_visual`, selection variants, keyframe probes —
see the mode glossary in the paper), `--category`/`--difficulty` to subset tasks,
`--max-steps` (default 15). Cloud models use their native APIs (set the relevant API
keys via environment variables); local models are served via vLLM. The Whisper ASR
service runs separately on CPU (`aoi/whisper_service.py`).

See [`docs/benchmark_design.md`](docs/benchmark_design.md) for the full benchmark
rationale and [`paper/main.tex`](paper/main.tex) for methodology and results.

## Citation

```bibtex
@article{li2026aoi,
  title  = {Agent-Computer Observation Interfaces Enable Dynamic Computer Use},
  author = {Li, Bojie},
  year   = {2026}
}
```
