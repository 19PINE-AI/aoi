# Remaining Reproducibility Information

This internal handoff file contains only checklist items that are not currently
`yes` and could become `yes` if additional information is recovered. The paper is
the primary source of truth. Current repository files may be older, newer, or
incomplete and must not silently override the paper.

For every recovered fact, write the operative value or procedure in author voice
in the paper itself. Do not point readers to the checklist, repository, appendix,
or this file for information required to interpret or reproduce a claim.

## 4.2 Development ranges and selection criteria — current answer: `partial`

### What we already have

- We report a final 3 Hz screen-sampling rate.
- We compare 64 x 64 grayscale frames and define a changed pixel using an
  intensity difference greater than 10/255.
- We report a 1% pixel-change skip threshold, a 0.04 CLIP cosine-distance
  threshold, and a 3% large-change capture override.
- We report the CLIP calibration candidates
  {0.001, 0.01, 0.04, 0.05, 0.08, 0.10, 0.15}, the transient-dialog calibration
  procedure, and the observed distances used to select 0.04.
- We report the pixel-change observations supporting the 1% skip threshold.
- We report the later task-level CLIP sensitivity sweep
  {0.02, 0.04, 0.08, 0.12, 0.20, 0.30} and uniform controls at 1 and 3 FPS.
  The paper does not present either as a development-time search.
- We report final values for the keyframe cap, trajectory depth, audio gates and
  windows, Whisper decoding, VAD timing, task limits, model output cap, local
  context, and statistical procedure.
- We state that temperature, top-p, and API seed arguments were omitted in favor
  of provider/runtime defaults.

### What we still need and where it goes

For each item below, first determine whether it was fixed once, inherited from a
model/library recommendation, constrained by the benchmark/runtime, or selected
after trying alternatives. If it was selected, recover the complete candidate
set or range, number of trials, development tasks or split, selection criterion,
whether test tasks were inspected, and whether selection was global or
model-specific.

| Item | Information still needed | Paper destination |
|---|---|---|
| Screen sampling rate: 3 Hz | Development treatment; if tuned, all rates tried and the accuracy/latency criterion | **“Inter-Step Keyframe Capture”** (`sec:keyframes`), beside the 3 Hz value |
| Pixel representation: 64 x 64 grayscale | Whether resolution and color conversion were fixed or compared; if compared, candidates and criterion | **“Inter-Step Keyframe Capture”** (`sec:keyframes`), beside the comparison definition |
| Per-pixel threshold: 10/255 | Whether fixed or tuned; if tuned, candidates, calibration data, and criterion | **“Inter-Step Keyframe Capture”** (`sec:keyframes`), beside the changed-pixel definition |
| Pixel skip threshold: 1% | Complete candidate set if alternatives were tried; confirmation that calibration was separate from reported test tasks; complete selection rule | **“Inter-Step Keyframe Capture”** (`sec:keyframes`), in the calibration explanation |
| CLIP threshold: 0.04 | Number of calibration trials; whether reported benchmark tasks were inspected; whether capture/rejection behavior was the complete criterion or latency/success also entered | **“Inter-Step Keyframe Capture”** (`sec:keyframes`), in the existing threshold-selection sentence |
| CLIP preprocessing | Confirm the submitted resize, crop, normalization, and precision. The current 224 x 224 path is only a lead. If preprocessing was chosen rather than required by the model, recover its development treatment | Operative pipeline in **“Implementation”** (`sec:implementation`); selection treatment in **“Inter-Step Keyframe Capture”** (`sec:keyframes`) |
| Large-change override: 3% | Whether fixed or tuned; if tuned, candidates, calibration tasks, measurements, and criterion | **“Inter-Step Keyframe Capture”** (`sec:keyframes`), beside the 3% override |
| Maximum keyframes: five | Original basis for choosing five, or a direct statement that it was fixed without search. A repository artifact suggests a later {1, 2, 3, 5} probe, but the paper does not report it and it must not be treated as development evidence without contemporaneous confirmation | **“Inter-Step Keyframe Capture”** (`sec:keyframes`), beside the cap |
| Stored trajectory depth: five records | Whether fixed, constrained, inherited, or tuned; if tuned, depths tried and accuracy/latency/token criterion | **“Visual Narration for Long-Term Context”** (`sec:narration`), beside the five-record limit |
| RMS speech gate: 0.01 | Whether fixed or tuned; if tuned, thresholds, audio calibration set, false-positive/false-negative criterion | **“Volume-Gated Audio Observation”** (`sec:audio`), beside the RMS threshold |
| Audio buffer: 70 s | Whether fixed or tuned; if tuned, lengths and continuity/latency/memory criterion | **“Volume-Gated Audio Observation”** (`sec:audio`), in the buffer description |
| Transcript windows: 5 s recent and 60 s rolling | Whether fixed or tuned; if tuned, window candidates and criterion | **“Volume-Gated Audio Observation”** (`sec:audio`), beside the window values |
| Transcript refresh: at most every 5 s | Whether fixed or tuned; if tuned, intervals and latency/cost/accuracy criterion | **“Volume-Gated Audio Observation”** (`sec:audio`), beside the refresh interval |
| Whisper beam size 1 and best-of 1 | Whether inherited, fixed for latency, or selected; if selected, decoding configurations and metric | **“Volume-Gated Audio Observation”** (`sec:audio`), in the Whisper configuration sentence |
| Whisper language: English | Confirm that English was fixed by benchmark scope rather than selected after a search | **“Volume-Gated Audio Observation”** (`sec:audio`), beside the language setting |
| VAD: 300 ms silence and 200 ms padding | Whether defaults, fixed choices, or tuned; if tuned, candidates and segmentation criterion | **“Volume-Gated Audio Observation”** (`sec:audio`), beside the VAD settings |
| Agent step cap: 15 | Whether inherited from benchmark design, fixed before evaluation, or selected; if selected, candidate caps and completion/cost criterion | **“Evaluation Protocol”** (`sec:evaluation-protocol`), beside the termination rule |
| Agent step interval | Recover the submitted interval or per-model rule. The current 2.0 s default requires contemporaneous confirmation. If tuned, report candidates and criterion | **“Evaluation Protocol”** (`sec:evaluation-protocol`), beside the timing rules |
| Browser viewport | Confirm the submitted viewport. The current 1280 x 720 value requires contemporaneous confirmation. If selected, report alternatives and criterion | **“Evaluation Protocol”** (`sec:evaluation-protocol`) or **“Implementation”** (`sec:implementation`) |
| Initial page-settle wait | Recover the submitted value or state that no explicit wait was set. The current 1.0 s value requires confirmation. If tuned, report candidates and criterion | **“Evaluation Protocol”** (`sec:evaluation-protocol`) |
| Post-action DOM-settle wait | Recover the submitted value or rule. The current 0.3 s value requires confirmation. If tuned, report candidates and criterion | **“Evaluation Protocol”** (`sec:evaluation-protocol`) |
| Base time margin: stimulus + 10 s | Why 10 s was chosen; whether alternatives were tried; if so, candidates and timeout/fairness criterion | **“Evaluation Protocol”** (`sec:evaluation-protocol`), beside the wall-clock formula |
| Additional non-standard-mode allowance: 30 s | Why 30 s was chosen; whether alternatives or measured overhead informed it; if selected, candidates and criterion | **“Evaluation Protocol”** (`sec:evaluation-protocol`), beside the additional allowance |
| Judge threshold: 0.5 | Whether specified before evaluation or calibrated; if calibrated, candidate thresholds, labeled development examples, and criterion | **“Evaluation Protocol”** (`sec:evaluation-protocol`), beside the pass rule |
| Non-streaming output cap: 1,024 | Whether fixed, provider-recommended, or selected; if selected, caps and truncation/cost criterion | **“Setup”** (`sec:setup`), beside the output cap |
| Local context: 4,096 | Whether a runtime/model constraint, fixed resource choice, or tuned; if tuned, lengths and success/memory criterion | **“Setup”** (`sec:setup`), beside the context value |
| Temperature, top-p, API seed arguments | Confirm that these were never set or varied for every model/run. If any exception exists, recover its values, range, and selection procedure | **“Setup”** (`sec:setup`); exceptions belong in the relevant sensitivity or streaming subsection |
| Statistical constants | Confirm that Wilson z=1.96, family-wise alpha=0.05, the six planned comparisons, and Holm correction were specified before examining results | **“Main Results”** (`sec:main-results`), in **Statistical protocol** |

### To change 4.2 to `yes`

The paper must state the development treatment of every final parameter. Every
parameter actually varied must include its range/candidates and selection
criterion. Every untuned parameter should be described directly, for example:
“we fixed X before evaluation and did not search alternatives.”

## 4.7 Random-seed procedure — current answer: `no`

### What we already have

- We report three headline results of 82%, 78%, and 76%, with mean 78.7% and
  standard deviation 3.1 percentage points.
- The paper calls these three seeded runs.
- We state that API seed arguments were omitted and provider/runtime defaults
  were used.
- Available repository files expose possible Python, NumPy, browser-JavaScript,
  task-order, and TTS randomness, but they may not represent the submitted runs.
- NumPy seed 42 in the latency benchmark is unrelated to the task-success runs
  and must not be used as evidence for them.

### What we still need and where it goes

| Randomness source | Information still needed | Paper destination |
|---|---|---|
| Three headline seeds | Literal seed values; whether one master seed or component seeds were used; mapping from each run to its seed | **“Main Results”** (`sec:main-results`), in the **Variance** paragraph |
| Seed derivation | Exact formula for deriving component/task seeds from a master seed, including ordering and reset points | **“Evaluation Protocol”** (`sec:evaluation-protocol`) |
| Hosted-model/API randomness | Reconcile omitted API seed arguments with “three seeds.” State whether randomness was controlled by provider seed, sampling settings, another component, or only repeated calls; state directly if hosted decoding could not be seeded | General procedure in **“Setup”** (`sec:setup`); headline application in **“Main Results”** (`sec:main-results`) |
| Python `random` | Whether `random.seed(...)` was called; literal value/formula; initialization point; per-task versus per-run behavior | **“Evaluation Protocol”** (`sec:evaluation-protocol`); control-specific exception in **“Perception Channel Contributions”** (`sec:analysis`) if needed |
| NumPy RNG | Whether `np.random.seed` or `Generator` was used; exact value/formula; initialization timing; whether noise controls shared the run seed | **“Evaluation Protocol”** (`sec:evaluation-protocol`) and, if applicable, **“Perception Channel Contributions”** (`sec:analysis`) |
| PyTorch/CUDA RNG | Whether `torch.manual_seed`, CUDA seed calls, deterministic algorithms, or cuDNN controls were used; exact values; nondeterministic operations left enabled | **“Implementation”** (`sec:implementation`) or the seed paragraph in **“Evaluation Protocol”** (`sec:evaluation-protocol`) |
| Browser JavaScript | Whether benchmark-page RNG was seeded or overridden; injection mechanism; page regeneration procedure; affected task families | **“Evaluation Protocol”** (`sec:evaluation-protocol`) |
| TTS | Whether stimuli were cached or regenerated; determinism for fixed text/voice; voice and generation settings; seed if supported | Stimulus procedure in **“Evaluation Protocol”** (`sec:evaluation-protocol`); software details in **“Implementation”** (`sec:implementation`) |
| Task order | Exact order or ordering rule for each run; whether shuffled; seed/permutation procedure; reset behavior between tasks | **“Evaluation Protocol”** (`sec:evaluation-protocol`) |
| Environment state | Browser profile/cache/cookie reset, benchmark-server reset, audio-device reset, and whether state crossed tasks or seeds | **“Evaluation Protocol”** (`sec:evaluation-protocol`) |
| Seed logging | Where seeds were recorded and how an exact rerun retrieves them | **“Evaluation Protocol”** (`sec:evaluation-protocol`) |

### To change 4.7 to `yes`

The paper must give the literal seeds and a procedure sufficient to reproduce
their effect on every relevant randomness source. A generic statement that seeds
were set is insufficient. If the three evaluations were independent repetitions
rather than controlled seeded runs, say so directly and keep 4.7 as `no`.

## 4.8 Computing infrastructure — current answer: `partial`

### What we already have

- One Linux host.
- NVIDIA RTX PRO 6000 Blackwell GPU with 96 GB VRAM.
- 192 GB system RAM.
- Python 3.11.
- Playwright 1.49 with headless Chromium.
- PulseAudio 16.1 with `parecord` and `pacat`.
- ffmpeg and edge-TTS.
- OpenAI CLIP ViT-B/16.
- Whisper large-v3 through faster-whisper/CTranslate2.
- vLLM 0.19.0 and a 4,096-token local context.
- CLIP and local-model inference used the GPU; cloud models used HTTP APIs.
- We directly disclose that several original-environment fields were not
  preserved.

### What we still need and where it goes

Unless noted otherwise, all recovered infrastructure information belongs in
**“Implementation”** (`sec:implementation`). Use the original experimental host,
container, scheduler logs, package export, or another contemporaneous record. Do
not substitute values from the current workstation without confirming that it
was the submitted environment.

| Item | Information still needed | Paper destination |
|---|---|---|
| GPU configuration | Exact GPU count; confirmation that all local runs used the same GPU; MIG or multi-GPU configuration if applicable | **“Implementation”** (`sec:implementation`) |
| CPU | Exact model, socket count, physical/logical core count, and any allocation restriction | **“Implementation”** (`sec:implementation`) |
| RAM | Confirm whether 192 GB was installed or allocated memory | **“Implementation”** (`sec:implementation`) |
| Operating system | Distribution, release/version, and architecture | **“Implementation”** (`sec:implementation`) |
| NVIDIA environment | Driver version and CUDA runtime/toolkit version | **“Implementation”** (`sec:implementation`) |
| Python | Exact patch version and environment manager/container if recoverable | **“Implementation”** (`sec:implementation`) |
| PyTorch stack | PyTorch version and CUDA build; torchvision/torchaudio versions if used | **“Implementation”** (`sec:implementation`) |
| Numerical stack | NumPy and SciPy versions used for submitted analyses | **“Implementation”** (`sec:implementation`) |
| Browser | Exact Chromium revision and materially relevant launch flags | **“Implementation”** (`sec:implementation`) |
| Audio system | Exact virtual-device configuration if it affects replication | **“Implementation”** (`sec:implementation`) |
| ffmpeg | Version/build and operative encoding/resampling flags not already stated | **“Implementation”** (`sec:implementation`) |
| edge-TTS | Package version | **“Implementation”** (`sec:implementation`); voice and synthesis procedure in **“Evaluation Protocol”** (`sec:evaluation-protocol`) |
| CLIP | Package or repository commit, preprocessing implementation, precision, and device | **“Implementation”** (`sec:implementation`) |
| ASR runtime | `faster-whisper` and CTranslate2 versions; actual submitted device and precision; model revision/cache identity | **“Implementation”** (`sec:implementation`); decoding values remain in **“Volume-Gated Audio Observation”** (`sec:audio`) |
| vLLM/local serving | Exact shared launch configuration, tensor parallelism, memory-utilization and scheduler settings, and nonstandard build information | Shared details in **“Implementation”** (`sec:implementation`); model-specific values in **“Setup”** (`sec:setup`) |
| API clients | Exact Anthropic/OpenAI/Google/xAI/OpenRouter library names and versions, or confirmation that raw HTTP was used | **“Implementation”** (`sec:implementation`) |
| Environment capture | Any retained lockfile, package export, container image/digest, scheduler environment, or commit identifying the submitted environment | **“Implementation”** (`sec:implementation`); retain the limitation if unavailable |

### To change 4.8 to `yes`

The paper must include the GPU and CPU models, memory, operating system, and
versions of all relevant software/framework components. If any required field
cannot be recovered, state that limitation directly and retain `partial`.

## 4.13 Final model and algorithm parameters — current answer: `partial`

### What we already have

- Exact public model identifiers for the primary models, Qwen replications,
  sensitivity models, and streaming models.
- A shared 1,024-token cap for non-streaming responses.
- A 4,096-token local-model context.
- Temperature, top-p, and API seed arguments omitted in favor of
  provider/runtime defaults.
- Fara served in float16.
- A direct limitation that EvoCUA dtype/quantization was not recorded
  consistently.
- Final keyframe, audio, context, task-budget, judge-threshold, and statistical
  values stated in the paper.
- A direct limitation that several hosted snapshot/routing and streaming-cap
  fields were not retained.

### What we still need and where it goes

| Model or parameter family | Information still needed | Paper destination |
|---|---|---|
| Claude Sonnet 4.6 | Hosted snapshot/revision/date; computer-use beta/version flags; provider image resizing/count limits; any settings outside the recorded request | **“Setup”** (`sec:setup`) |
| GPT-5.4 | Snapshot/revision; computer-use/tool configuration; reasoning-effort setting if applicable; provider image handling | **“Setup”** (`sec:setup`) |
| Gemini 2.5 Flash | Snapshot/revision; computer-use/tool, safety, and generation settings outside recorded arguments; image limits | **“Setup”** (`sec:setup`) |
| Grok-4 | Snapshot/revision; tool/computer-use settings; provider image handling and other material external defaults | **“Setup”** (`sec:setup`) |
| EvoCUA-32B | Model revision/commit; submitted dtype and quantization; complete vLLM launch command including GPU-memory utilization, trust-remote-code, tensor parallelism, scheduler, and image limits | **“Setup”** (`sec:setup`); shared runtime version in **“Implementation”** (`sec:implementation`) |
| Fara-7B | Model revision/commit; complete vLLM launch configuration; confirmation of no quantization; model-specific image/token settings | **“Setup”** (`sec:setup`) |
| Qwen3-VL-235B | Routed provider, resolved snapshot/revision, routing policy, and provider image/context limits | **“Setup”** (`sec:setup`), in the Qwen replication disclosure |
| Qwen3-VL-30B | Routed provider, resolved snapshot/revision, routing policy, and provider image/context limits | **“Setup”** (`sec:setup`), in the Qwen replication disclosure |
| Gemini 3 Flash | Exact preview snapshot/date and provider-side settings/image limits | **“Model Sensitivity Study”** (`sec:model_sensitivity`) |
| Grok-4.3 | Exact snapshot/revision and provider-side settings/image limits | **“Model Sensitivity Study”** (`sec:model_sensitivity`) |
| Grok-4-fast-reasoning | Snapshot/revision, reasoning-mode controls, and provider-side settings/image limits | **“Model Sensitivity Study”** (`sec:model_sensitivity`) |
| OpenAI Realtime | Hosted snapshot; response/output cap; session configuration; audio/image formats and cadence; tool-choice/VAD settings; truncation policy | **“Comparison with Streaming Multimodal Baselines”** (`sec:streaming`) |
| Gemini Live | Resolved snapshot/date; response/output cap; session configuration; audio/image formats and cadence; tool and generation settings | **“Comparison with Streaming Multimodal Baselines”** (`sec:streaming`) |
| LLM judge | Exact Gemini 2.0 Flash identifier/snapshot; prompt/rubric availability; output cap, temperature, top-p, seed, safety, and parsing settings if set; otherwise confirmation that each was omitted | **“Evaluation Protocol”** (`sec:evaluation-protocol`) |
| Keyframe algorithm | Confirm that all reported final values apply to every main configuration and identify any exception | **“Inter-Step Keyframe Capture”** (`sec:keyframes`) |
| CLIP preprocessing | Submitted resize/crop/normalization and precision; current 224 x 224 implementation requires historical confirmation | **“Implementation”** (`sec:implementation`) or **“Inter-Step Keyframe Capture”** (`sec:keyframes`) |
| Audio algorithm | Submitted ASR device/precision and any configuration-specific differences | **“Volume-Gated Audio Observation”** (`sec:audio`); device/precision in **“Implementation”** (`sec:implementation`) |
| TTS stimuli | Submitted voice identifier and synthesis options; whether audio was generated once and reused. `en-US-GuyNeural` is only the current maintained default until confirmed | **“Evaluation Protocol”** (`sec:evaluation-protocol`); package version in **“Implementation”** (`sec:implementation`) |
| Narration/context | Exact narration instruction/template; maximum narration length if operative; configuration-specific retention behavior | **“Visual Narration for Long-Term Context”** (`sec:narration`) and **“Observation Record”** (`sec:obsrecord`) |
| Screenshot retention | Submitted screenshot-history/image-pruning count and whether it varied by mode; current value five requires historical confirmation | **“Visual Narration for Long-Term Context”** (`sec:narration`) and **“Observation Record”** (`sec:obsrecord`) |
| Observation modes | Contemporaneous component flags, prompts, history, and image/audio retention for every named mode; especially the operative distinction between `aoi_visual_asr` and `aoi_full` | Definitions in **“Setup”** (`sec:setup`); ablation distinctions in **“Perception Channel Contributions”** (`sec:analysis`) |
| Browser/timing controls | Submitted viewport, step interval, initial page wait, post-action DOM wait, and model-specific timing exceptions; current 1280 x 720, 2.0 s, 1.0 s, and 0.3 s values require confirmation | **“Evaluation Protocol”** (`sec:evaluation-protocol`) |
| Action/tool formatting | Submitted action prompt/schema and coordinate convention for each model family, including normalized-coordinate or function-calling settings | **“Setup”** (`sec:setup`); shared grammar may also appear in **“Background and Motivation”** (`sec:background`) |
| Task termination | Confirmation that no model-specific step/time exceptions existed beyond those stated; exact value for every exception | **“Evaluation Protocol”** (`sec:evaluation-protocol`) or the relevant model-specific subsection |

The current repository contains conflicting or potentially stale launch and mode
configurations. Request contemporaneous provider metadata, submitted-run logs,
scheduler commands, or the exact submitted code revision. In particular, do not
reinterpret the paper's `aoi_visual_asr` versus `aoi_full` result from the current
harness; recover the operative submitted distinction.

### To change 4.13 to `yes`

The paper must state every operative final model, serving, observation, timing,
prompt/schema, and provider-side parameter. “Provider default” is sufficient only
when we truly did not set the parameter and the provider/model snapshot is
identified well enough to recover that default. If any operative parameter remains
unknown, state that limitation directly and retain `partial`.
