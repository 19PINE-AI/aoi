# Agent-Computer Observation Interface (AOI) & DynaCU-Bench

A model-agnostic **perception layer** that gives existing computer-use (CU) agents
adaptive visual and audio perception of *dynamic* screen content — video, animations,
transient UI events, meetings, notifications, and spoken instructions — with **zero
retraining**, plus **DynaCU-Bench**, a benchmark of dynamic browser tasks that
screenshot-only agents cannot solve.

> Across **nine CU models** (closed and open-source, 7B to frontier scale) the AOI
> yields **+17 to +48 pp** on DynaCU-Bench with no retraining — Gemini 3 Flash the
> lone exception, where keyframe-token dilution means components must be selected
> per model.

📄 **Paper:** [`paper/main.pdf`](paper/main.pdf) · 🌐 **Website:** https://01.me/research/aoi
· 📊 **Results map:** [`results/README.md`](results/README.md)

Authors: **Bojie Li** (Pine AI) and **Noah Shi** (University of Washington).

---

## Contents

- [What problem does this solve?](#what-problem-does-this-solve)
- [Repository layout](#repository-layout)
- [DynaCU-Bench](#dynacu-bench)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproducing the paper](#reproducing-the-paper)
- [Tests](#tests)
- [License & citation](#license--citation)

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
   gating (optional CLIP semantic filtering, θ = 0.04); 0–5 keyframe images per step.
2. **Volume-gated audio observation** — RMS-energy gate followed by Whisper large-v3
   speech transcription (speech only).
3. **Visual narration** — the CU model itself narrates new visual content as a
   side-output each step; the text persists in the trajectory after images are pruned.

Because the gates cost under a millisecond and pass nothing on unchanged, silent
content, AOI adds perception only where the screen is actually dynamic and reduces to
the standard loop everywhere else (~60% of steps are fully idle). See the paper for the
full results, ablations, per-model component analysis, and streaming-baseline comparison.

## Repository layout

| Path | Contents |
|------|----------|
| [`aoi/`](aoi/) | The perception layer: keyframe extractor, audio pipeline / Whisper service, observation record, agent loop, CU-model adapters, streaming baselines. |
| [`dynacubench/`](dynacubench/) | Benchmark definitions (`tasks_v3.py` → `DynaCUBenchV3`), DOM/LLM/hybrid evaluators, and the Static-50 and DynaCU-Real-Local generators. |
| [`benchmark_env/`](benchmark_env/) | Playwright browser environment and the self-contained HTML task files (categories A–J, the static control, and real-content assets). |
| [`experiments/`](experiments/) | Evaluation harness (`browser_eval.py`) and analysis scripts (stats, tables, token accounting, ablations). See [`experiments/README.md`](experiments/README.md). |
| [`results/`](results/) | Raw per-task evaluation JSON. See [`results/README.md`](results/README.md) for the file → paper-table map. |
| [`paper/`](paper/) | LaTeX source, figures, and the compiled PDF. |
| [`website/`](website/) | The interactive results site (React + Vite); `scripts/build_data.py` derives its data from `results/`. |
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
logic and evaluation embedded). Scoring is a deterministic page-state check for 93
tasks, a check + LLM rubric for 6, and LLM-judge only for 1. Companion sets:
**Static-50** (no-degradation control on purely static pages) and **DynaCU-Real-Local**
(cross-engine ASR robustness with `espeak` audio and real `asciinema` screencasts).

## Installation

```bash
# System deps: PulseAudio (virtual audio I/O for audio modes), ffmpeg, espeak
# (for DynaCU-Real-Local), and a Chromium for Playwright.
pip install -r requirements.txt        # playwright, openai-clip, faster-whisper, edge-tts, ...
playwright install chromium
```

**API keys** — copy [`.env.example`](.env.example) to `.env` and fill in the keys for
the models you intend to run (`.env` is gitignored). Only the relevant ones are needed:

| Variable | Used for |
|----------|----------|
| `ANTHROPIC_API_KEY` | Claude |
| `OPENAI_API_KEY` | GPT-5.4 / GPT-4o / OpenAI Realtime |
| `GEMINI_API_KEY` | Gemini 2.5 / 3 Flash, Gemini Live, LLM judge |
| `XAI_API_KEY` | Grok-4 / 4.3 / 4-fast / Grok Voice |
| `OPENROUTER_API_KEY` (+ `OPENAI_BASE_URL`) | Open-source models (Qwen3-VL, GPT-5.4-or) via OpenRouter |
| `VLLM_API_KEY` | Locally served models (vLLM) |

Local open-source models (Fara-7B, EvoCUA-32B, UI-TARS, …) are served with vLLM; the
audio modes additionally need the Whisper microservice (below). The unit tests need
none of the above.

## Quick start

```bash
# 1. Start the Whisper ASR microservice (needed for any audio / aoi_full mode).
#    Defaults to GPU + large-v3; pass --device cpu to run on CPU.
python -m aoi.whisper_service --port 8786            # add --device cpu if no GPU

# 2. Run the benchmark: standard loop vs. AOI-full on Claude Sonnet 4.6.
python -m experiments.browser_eval \
    --models claude-sonnet-4-6 \
    --modes standard aoi_full \
    --output-dir results/run1
```

`browser_eval` flags: `--models` (one or more CU models), `--modes`
(`standard`, `aoi_full`, `standard_structured`, `aoi_audio`, `aoi_visual`, selection
variants, keyframe probes — see the mode glossary in the paper and
`experiments/README.md`), `--category` / `--difficulty` / `--task-prefix` /
`--max-tasks` to subset, `--max-steps` (default 15). Closed models use their native
APIs; local models are served via vLLM (`experiments/serve_local_model.sh <alias>`).

## Reproducing the paper

There are two paths. To **regenerate the paper tables from the committed results**
(no eval re-run, no API keys):

```bash
python experiments/compute_stats.py          # McNemar tests + Wilson confidence intervals
python experiments/compute_tokens.py         # token + dollar cost accounting
python experiments/analyze_keyframe_context.py
python website/scripts/build_data.py         # → website/public/data/*.json (drives the site)
```

[`results/README.md`](results/README.md) maps every committed result file to the table
or figure it produces.

To **reproduce a result from scratch**:

1. Start the Whisper service (and, for open-source models, a vLLM server).
2. Run `python -m experiments.browser_eval --models <model> --modes standard aoi_full
   --output-dir results/<run>`.
3. Point the analysis scripts above at your run directory.

Accepted `--models` aliases include `claude-sonnet-4-6`, `gpt-5.4`, `gemini-2.5-flash`,
`gemini-3-flash`, `grok-4`, `grok-4.3`, `grok-4-fast-reasoning`, `evocua-32b`,
`fara-7b`, and `qwen3-vl-*` (full list in `aoi/cu_model.py`). The benchmark task suite
is regenerated by `dynacubench/static_task_generator.py` (Static-50) and
`dynacubench/build_realcontent_local.py` (DynaCU-Real-Local).

See [`docs/benchmark_design.md`](docs/benchmark_design.md) for the full benchmark
rationale and [`paper/main.pdf`](paper/main.pdf) for methodology and results.

## Tests

```bash
python -m pytest tests/ -q        # keyframe extractor, observation record, benchmark
```

The unit tests exercise the keyframe extractor, observation record, and benchmark
definitions only — no browser, Whisper service, or API keys required.

## License & citation

- **Code** (`aoi/`, `dynacubench/`, `experiments/`, `tests/`, `website/`) — MIT, see [`LICENSE`](LICENSE).
- **Benchmark data, results, figures, and paper** — CC BY 4.0, see [`LICENSE-DATA`](LICENSE-DATA).

```bibtex
@article{li2026aoi,
  title  = {Agent-Computer Observation Interfaces Enable Dynamic Computer Use},
  author = {Li, Bojie and Shi, Noah},
  year   = {2026}
}
```
