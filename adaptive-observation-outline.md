# The Agent Observation Interface: Adaptive Multimodal Perception for Computer-Use Agents

## Paper Outline

---

## Core Thesis

The bottleneck in computer-use agents is not reasoning --- it is observation. Current CU agents observe the world through periodic static screenshots taken every 3--5 seconds and have no audio perception at all, making them incapable of handling dynamic visual content (video playback, animations, transient UI events) or any audio-driven interaction (meetings, notifications, alerts). Just as SWE-agent (Yang et al., NeurIPS 2024) showed that the agent-computer *action* interface dramatically affects agent performance, we show that the *observation* interface is equally critical and equally underexplored. We introduce the **Agent Observation Interface (AOI)**, a lightweight, model-agnostic perception layer that sits between continuous screen/audio streams and any existing image-based CU model. The AOI has three components, all scaling to zero overhead for static, silent desktop work: (1) a two-stage adaptive keyframe extractor (pixel gate + CLIP semantic distance) that captures frames only when the screen changes semantically, (2) volume-gated multimodal audio scene understanding (Qwen3 Omni) that transcribes both speech and non-speech sounds only when audio is present, and (3) CU-model-generated visual narration that converts ephemeral keyframe images into persistent text for long-term context. With zero model retraining, the AOI enables existing CU agents to handle video meetings, YouTube content, animated web pages, audio alerts, and transient UI events they previously could not attempt. On DynaCU-Bench, a new benchmark of 200+ tasks requiring dynamic perception, five diverse CU models equipped with the AOI achieve 40--60% success where they previously score near zero, while maintaining identical performance on standard static benchmarks.

---

## 1. Introduction

### 1.1 The Success of Computer-Use Agents

Computer-use agents have advanced remarkably. OSWorld scores have climbed from 12.24% (GPT-4V, April 2024) to 72.7% (Claude Opus 4.6), surpassing the human baseline of 72.36% within roughly two years. Models like Claude Computer Use (Anthropic), OpenAI CUA/Operator, UI-TARS (ByteDance), Surfer 2 (H Company), OpenCUA (HKU/xlang-ai), and Fara-7B (Microsoft) can operate real desktops: filling forms, navigating websites, editing documents, and using professional software.

### 1.2 The Blind Spot: Dynamic Content

All these agents share a fundamental architectural limitation. They observe the world through **discrete static screenshots** taken every 3--5 seconds (determined by model inference latency + action execution + a post-action buffer). Between screenshots, the agent is completely blind and entirely deaf. This makes current CU agents incapable of:

- Understanding video content playing on screen (YouTube, tutorials, media)
- Catching transient UI events that appear and auto-dismiss within one observation interval (cookie consent dialogs, toast notifications, download completion popups)
- Hearing anything at all: meeting speech, notification sounds, system alerts, error beeps
- Perceiving animations that convey information (loading progress, carousel content, transitioning slides)

Quantitative evidence of the gap:

- GUI-World (Chen et al., ICLR 2025): video LLMs fail on all temporal GUI tasks across a 12K-annotation benchmark spanning 6 scenarios
- VideoWebArena (Jang et al., ICLR 2025): best model achieves 13.3% vs. human 73.9% on video-dependent web tasks; long-context models actually perform *worse* with video tutorials than without (5--10% decrease)
- VideoGameBench (Zhang et al., 2025): even with games paused during inference (Lite mode), Gemini 3.1 Pro completes only 1.6% of games; in real-time mode, only 0.48%
- CUA-Suite (ServiceNow, 2026): analysis of 55 hours of 30fps human desktop recordings across 87 applications demonstrates the richness of temporal dynamics (mouse movements, animations, audio feedback) that screenshot-based agents entirely miss

### 1.3 Why Existing Approaches Fall Short

**Retraining CU models on video data.** Expensive, model-specific, and no one has demonstrated this successfully at scale. The training data (paired video observations with computer-use actions) barely exists, though recent efforts (VideoAgentTrek, Watch and Learn) have begun mining tutorial videos for trajectory data.

**Using video-native models (Gemini).** Gemini's standard video understanding captures frames at 1 FPS, encodes each independently through a ViT-based visual encoder into 258 tokens, and relies on its long context window for temporal reasoning. This is fundamentally frame sampling + brute-force context length, processing all frames regardless of whether content has changed, wasting tokens on redundant static frames.

**Native real-time multimodal models (Gemini 3.1 Flash Live, Project Astra).** Google's Gemini 3.1 Flash Live (March 2026) processes continuous audio/video/text streams in real time via WebSockets. This works --- but locks users into a single model provider, offers no selective perception (all audio/video is processed regardless of relevance), and requires API migration for every existing CU deployment. Our approach is complementary: a lightweight external layer that gives *any* CU model adaptive dynamic perception.

**General computer control frameworks (Cradle).** Cradle (BAAI, ICML 2025) includes an information gathering module with consecutive frame difference analysis for dynamic content. However, it uses pixel-level differencing without semantic filtering (prone to false positives from cursors, spinners, ads), has no audio perception, and its frame processing is tightly coupled to its specific agent architecture rather than being a model-agnostic layer.

### 1.4 Our Approach

The bottleneck is not the agent model --- it is the **observation interface**. Current agents conflate two concerns: *when to look* and *when to act*. We decouple them. Observation runs continuously and adaptively in the background; action remains discrete. The bridge between them is a lightweight observation layer that converts continuous screen and audio streams into the sparse images and text that existing CU models already accept.

### 1.5 Contributions

1. We identify the **observation interface** as a critical but overlooked bottleneck in computer-use agents, complementary to the action interface studied by SWE-agent.
2. We introduce the **Agent Observation Interface (AOI)**, a model-agnostic perception layer with three components: two-stage adaptive keyframe extraction (pixel gate + CLIP semantic distance), volume-gated Qwen3 Omni audio scene understanding, and CU-model-generated visual narration for persistent long-term context. All three scale to zero overhead when the environment is static and silent.
3. We introduce **DynaCU-Bench**, a benchmark of 200+ computer-use tasks across five categories that specifically require dynamic visual and/or audio perception --- tasks unsolvable from static screenshots alone --- with human performance baselines.
4. We demonstrate that five existing CU models (UI-TARS, OpenCUA, Qwen3-VL, Claude CU, OpenAI CUA), equipped with the AOI and **zero retraining**, achieve 40--60% success on dynamic tasks where they previously score near zero, while maintaining identical performance on standard static benchmarks (OSWorld, WebArena).

---

## 2. Background: The CU Agent Loop

All current CU agents follow the same loop:

```
repeat:
    s_t <- screenshot()
    a_t <- CU_model(s_t, trajectory)
    execute(a_t)
    wait(buffer_ms)    # e.g., 500ms for action effect to settle
```

One screenshot per action step. The interval between observations is determined by model inference time + action execution + post-action buffer, typically 3--5 seconds total. Everything that occurs between screenshots --- visual changes, audio events, transient UI elements --- is invisible to the agent.

Some models support multi-image input through screenshot history: UI-TARS (up to 5 images), OpenCUA (up to 3), Qwen-VL (up to 4), Claude CU (~10). But all images are post-action screenshots from prior steps. None captures what happens *between* steps.

Four categories of content that this loop cannot capture:

| Category | Example | Why Missed |
|---|---|---|
| Transient UI events | Toast notifications, cookie consent dialogs that auto-dismiss | Appear and vanish within one observation interval |
| Continuous visual media | YouTube video, animated tutorials, presentation slides | Information encoded in temporal visual change |
| Audio events | Meeting speech, notification dings, error alerts, system chimes | No audio capture at all |
| Periodic visual noise | Loading spinners, blinking cursors, cycling ad banners | Should be ignored; triggers false positives in naive detection |

---

## 3. System Design

The AOI sits between the environment and any existing CU model. It observes the entire interval between agent steps and provides the CU model with additional keyframes, audio descriptions, and accumulated text context --- all in standard image + text format that every CU model already accepts.

For static, silent tasks (the common case), the AOI produces nothing --- behavior is identical to the standard loop. For dynamic tasks, the CU model receives strictly more information. No retraining is needed.

### 3.1 Architecture Overview

The AOI has three components, governed by one principle --- **scales to zero**: every component has a fast gate that short-circuits processing when there is nothing to perceive.

```
                    ┌──────────────────────────────┐
  Screen stream ──> │  Adaptive Keyframe Extractor  │──> keyframe images
   (~3 Hz)          │  (pixel gate + CLIP distance)  │    (0--5 per step)
                    └──────────────────────────────┘
                    ┌──────────────────────────────┐
  Audio stream ───> │  Volume-Gated Audio Observer   │──> audio description
   (continuous)     │  (RMS gate + Qwen3 Omni)       │    (text, per step)
                    └──────────────────────────────┘
                    ┌──────────────────────────────┐
  Prior steps ────> │  Visual Narration Context       │──> text context
   (trajectory)     │  (accumulated narrations)       │    (from prior steps)
                    └──────────────────────────────┘
                              │
                              v
                    ┌──────────────────────────────┐
                    │  Observation Record             │──> images + text
                    │  (standard CU model input)      │    to CU model
                    └──────────────────────────────┘
```

### 3.2 Adaptive Keyframe Extraction

**The problem.** Between agent steps, the screen may change (a video plays, a dialog appears, a slide transitions) or stay static. Naive approaches fail: uniform high-FPS sampling wastes model input slots on redundant frames; pixel-level differencing triggers on spinners, cursors, and ads that carry no useful information.

**Two-stage filtering.**

*Stage 1: Pixel gate.* Compute the ratio of changed pixels between the current sample and the last captured keyframe. If below threshold (1% of pixels), skip entirely. This catches ~90% of samples where nothing moved on screen. Cost: <1ms on CPU.

*Stage 2: CLIP semantic distance.* When pixels did change, encode the current frame with CLIP-ViT-B/16 and compute cosine distance to the last keyframe's CLIP embedding. Capture a new keyframe only when the distance exceeds threshold theta. Reset the anchor embedding to the new keyframe.

**Why CLIP works as a unified filter:**

- **Periodic noise (spinners, cursors, looping ads):** Pixels change but CLIP embedding remains stable because the semantic content is identical across loop iterations. Near-zero CLIP distance.
- **Semantically meaningful changes (dialog appearing, page navigation, video scene cut):** These alter semantic content and produce large CLIP embedding shifts, reliably exceeding the threshold.
- **No separate heuristics needed.** CLIP absorbs region segmentation, periodicity detection, and motion classification into a single distance metric.

**Algorithm:**

```python
class KeyframeExtractor:
    def __init__(self, clip_model, theta=0.15, max_keyframes=5):
        self.clip = clip_model
        self.theta = theta
        self.max_keyframes = max_keyframes
        self.anchor_emb = None
        self.last_raw = None
        self.keyframes = []  # list of (timestamp, image) pairs

    def on_sample(self, frame, timestamp):
        # Stage 1: Pixel gate
        if self.last_raw is not None:
            if pixel_change_ratio(frame, self.last_raw) < 0.01:
                return
        self.last_raw = frame

        # Stage 2: CLIP semantic distance
        emb = self.clip.encode(frame)
        if self.anchor_emb is None:
            self.anchor_emb = emb
            return

        if 1 - cosine_similarity(emb, self.anchor_emb) > self.theta:
            self.keyframes.append((timestamp, frame))
            self.anchor_emb = emb  # recenter anchor

    def get_and_reset(self):
        result = self.keyframes[-self.max_keyframes:]
        self.keyframes.clear()
        return result
```

**Computational cost:**

- Pixel gate: <1ms per sample (CPU only)
- CLIP-ViT-B/16 encode: 5--10ms per sample (GPU), runs only when pixels changed (~10% of samples)
- Cosine similarity: <0.1ms
- Amortized cost per sample: ~1--2ms
- At 3 Hz sampling rate: ~3--6ms total observation overhead per second

### 3.3 Audio Scene Understanding

**The problem.** Current CU agents have zero audio perception. They cannot hear meeting speech, notification sounds, error alerts, or any other audio event. Pure ASR models (Whisper) transcribe speech but produce nothing for non-speech audio:

| Audio Event | Whisper Output | Agent Needs |
|---|---|---|
| Notification ding | (nothing) | "A notification sound was heard" |
| Meeting join chime | (nothing) | "Someone joined the meeting" |
| Error alert beep | (nothing) | "An error alert sounded" |
| System alarm | (nothing) | "A calendar alarm is ringing" |

A multimodal audio model (Qwen3 Omni) describes the full audio scene: speech content, environmental sounds, and audio events --- all as natural language text.

**Volume gate.** In most computer use (file editing, form filling, silent web browsing), there is no audio at all. Computing RMS energy of the audio buffer is sub-millisecond. If below the silence threshold, skip the audio model call entirely. Cost: ~0ms for the majority of steps.

```python
class AudioObserver:
    def __init__(self, audio_model, silence_threshold):
        self.model = audio_model  # Qwen3 Omni
        self.silence_threshold = silence_threshold

    def process(self, audio_chunk, prior_transcript):
        if rms_energy(audio_chunk) < self.silence_threshold:
            return ""  # silent, skip model call, cost ~0ms

        return self.model.describe(
            audio=audio_chunk,
            context=prior_transcript
        )
```

**The boundary problem.** Agent step boundaries are determined by model inference latency, not by speech content. A 3.5-second audio chunk captures ~8--9 words at normal speaking rate (~2.5 words/second). This almost always falls mid-sentence:

```
Actual speech stream:
  "...strategy focuses on three key areas. First, expanding into the..."

Step N chunk (t=38.5 to t=42.0):
  Raw audio: "...tegy focuses on three key ar—"
                                             ^
                              arbitrary boundary, mid-word
```

**Overlapping window with prior transcript.** To resolve boundary artifacts, send the audio model: (1) raw audio with ~3.5 seconds of overlap from the previous interval, so the model hears complete words at the boundary, and (2) the prior transcript as text context. The overlap provides acoustic continuity; the transcript provides semantic continuity.

```
Input to audio model:
  Raw audio:      [t_{N-1} ........ t_{N+1}]
                   ^ overlap         ^ current interval end

  Text context:   "Prior transcript ended with:
                   '...strategy focuses on three key'"

  Instruction:    "Continue transcription from where the prior
                   transcript ended. Cover only the new portion.
                   Describe any non-speech sounds."

Output:           "key areas. First, expanding"
```

**Timing and the critical path.** The audio chunk must include the post-action buffer period because this captures action feedback sounds (click confirmations, error beeps, success chimes). The audio model processing (200--300ms when sound is present) is on the critical path: the audio chunk is not complete until after the post-action buffer, and the CU model needs the audio description before reasoning about the next step.

**Cumulative overhead:**

| Scenario | % Steps with Audio | Overhead per 50-Step Task |
|---|---|---|
| Silent desktop work | ~0% | ~0s |
| Web browsing (occasional audio ads) | ~5% | ~0.5--0.75s |
| Video meeting | ~80% | ~8--12s |
| YouTube watching | ~90% | ~9--13.5s |

### 3.4 Visual Narration for Long-Term Context

**The problem.** CU models retain only 3--5 recent images in context. Keyframes from earlier steps are pruned. For tasks spanning many steps (watching a 10-minute video = ~170 steps), the agent loses all visual memory of earlier content. Audio transcriptions persist naturally as text in the trajectory, but visual observations have no analogous persistence.

**The mechanism.** At each step, the CU model outputs both an action and a brief visual narration --- a text description of what is visually new in the current keyframes. This narration is stored in the trajectory and persists indefinitely, even after the corresponding images are pruned.

This is generated in the same inference call that produces the action --- no additional model call. The only cost is a few extra output tokens per step:

```
In addition to your action, briefly describe any new visual
information in the current screenshots not already captured
by the audio. If nothing visually changed, say "No visual change."
Keep it under two sentences.
```

Key design points:
- **No separate captioner model.** Unlike LLoVi (Zhang et al., EMNLP 2024) which uses dedicated captioners (LaViLa, BLIP2), narration is a side-output of the CU model's existing inference call. Zero additional compute.
- **Task-relevant narration.** The CU model narrates what it sees *in the context of its current task* --- selective, goal-directed description rather than generic captioning.
- **Persistence for agent loops.** Narration accumulates across an open-ended agent trajectory where images are pruned from context.

**Realistic example.** Agent watching a recorded presentation. Each step spans ~3.5 seconds:

```
Step 12 (t ~= 38.5--42.0s):
  Keyframes captured: 0 (slide unchanged, CLIP distance below theta)
  Audio (Qwen3 Omni): "focuses on three key areas."
  -> CU model output:
     narration: "Same slide."
     action: wait()

Step 13 (t ~= 42.0--45.5s):
  Keyframes captured: 1 at t=43.2s (new slide detected by CLIP)
  Audio: "First, expanding into"
  -> CU model output:
     narration: "New slide: map of Europe, western countries in blue."
     action: wait()

Step 14 (t ~= 45.5--49.0s):
  Keyframes captured: 0 (same slide)
  Audio: "the European market. We see"
  -> CU model output:
     narration: "Same map slide."
     action: wait()

Step 15 (t ~= 49.0--52.5s):
  Keyframes captured: 0 (same slide)
  Audio: "thirty percent opportunity there."
  -> CU model output:
     narration: "No visual change."
     action: wait()
```

At step 170 (end of the video), the agent reads back through the full trajectory and can reconstruct: the presenter discussed European expansion (steps 13--15) while showing a map of Europe. The audio fragments, read in sequence, form coherent speech. The visual narrations mark when and what changed on screen.

### 3.5 The Observation Record

At each step, the CU model receives a structured document combining text context from recent prior steps and new raw observations from the current interval.

```
=== Step N Observation ===

[CONTEXT -- text from recent prior steps, no images]

  Step N-2 (t ~= X--Y):
    AUDIO: "<transcription text>"
    VISUAL: "<narration text from CU model>"
    ACTION: <action taken>

  Step N-1 (t ~= Y--Z):
    AUDIO: "<transcription text>"
    VISUAL: "<narration text from CU model>"
    ACTION: <action taken>

[NEW -- current interval]

    AUDIO: "<transcription from audio model>"
    <timestamp> [IMAGE: keyframe]      <- raw image, if any captured
    ...
    <timestamp> [IMAGE: post-action screenshot]  <- always present

[TASK] <task instruction>
```

Design points:

- **CONTEXT** is pure text: narrations + transcriptions + actions from previous steps. No images (already pruned). Provides continuity.
- **NEW** contains the current audio transcription (text), any keyframe images captured during this interval (raw images), and the post-action screenshot.
- Keyframes in the NEW section are raw images without descriptions --- the CU model has not seen them yet.
- Context depth is adaptive: 0 prior steps for static desktop work, 2--3 steps for video/meeting watching.

### 3.6 Integration with CU Models

**Image input mapping.** Keyframes occupy the multi-image slots CU models already provide for screenshot history:

| Model | Max Images | Slot Usage |
|---|---|---|
| UI-TARS v1.5/v2 | 5 | Up to 5 keyframes |
| OpenCUA | 3 | Up to 3 keyframes |
| Qwen2.5-VL / Qwen3-VL CU | 4 | Up to 4 keyframes |
| Claude Computer Use | ~10 | Keyframes in conversation history |
| Single-image models (Aguvis, etc.) | 1 | Most recent keyframe only |

When no keyframes were captured (the common case), only the post-action screenshot is sent --- identical to the standard loop.

**Audio and narration mapping.** Both audio descriptions and visual narrations are natural language text, prepended to the observation context. All CU models accept text input. No model modification required.

**Token cost:**

| Component | Tokens Added (per step) | When |
|---|---|---|
| Audio transcription text | ~20--50 tokens | Only when audio is present |
| Visual narration context (2--3 prior steps) | ~30--80 tokens | Only when prior steps had dynamic content |
| Keyframe images | ~258 tokens/image | Only when keyframes captured (avg 0.1--0.3/step) |
| Post-action screenshot | ~258 tokens | Always (same as standard loop) |

For static, silent tasks: zero additional tokens. For a video meeting task: ~50--130 additional text tokens per step, plus occasional keyframe images.

**Complete agent step:**

```python
def agent_step(step_n):
    # 1. Keyframe extraction (fast, ~10ms total)
    keyframes = keyframe_extractor.get_and_reset()

    # 2. Audio processing (0ms if silent, 200-300ms if sound present)
    audio_chunk = audio_buffer.get(
        start=prev_interval_start,   # overlap for word boundaries
        end=current_time             # includes post-action buffer
    )
    if rms_energy(audio_chunk.new_portion) > silence_threshold:
        audio_text = audio_model(
            audio=audio_chunk,
            prior_transcript=trajectory.last_audio_text()
        )
    else:
        audio_text = ""

    # 3. Build observation record
    context_text = trajectory.get_recent_text(n_steps=2)
    images = keyframes + [post_action_screenshot]

    # 4. CU model inference (single call -> narration + action)
    narration, action = cu_model(
        context=context_text,
        images=images,
        audio=audio_text,
        task=task_instruction,
        trajectory=trajectory
    )

    # 5. Store text for future context (persists after images pruned)
    trajectory.append(
        step=step_n,
        audio_text=audio_text,          # persists
        visual_narration=narration,     # persists
        action=action                   # persists
    )
    # Keyframe images are NOT stored long-term; pruned naturally

    # 6. Execute
    execute(action)
    wait(buffer_ms)
```

---

## 4. Implementation

### 4.1 System Architecture

[TODO: Describe the concrete implementation]

- Language and framework (Python, built on top of OSWorld infrastructure)
- Screen capture mechanism: virtual framebuffer (Xvfb) in VM environments, platform-specific APIs (X11/XComposite, macOS ScreenCaptureKit, Windows DXGI) for native deployment
- Audio capture: PulseAudio loopback module in Linux VMs, system audio loopback on macOS/Windows
- Process architecture: keyframe extractor runs as a separate thread with a shared ring buffer; audio observer runs as a separate thread; main thread runs the CU agent loop and assembles observation records

### 4.2 Model Serving

- CLIP-ViT-B/16: loaded locally on GPU, shared across the keyframe extractor (single model instance, ~150MB VRAM)
- Qwen3 Omni: [local GPU serving via vLLM / API call --- specify which and latency implications]
- CU models: each model uses its native API or local serving setup (details per model in Appendix)

### 4.3 Deployment Footprint

- Lines of code for the AOI layer (excluding CU models and benchmarking infrastructure)
- Additional GPU memory required beyond the CU model itself
- Supported platforms and OS versions tested

---

## 5. DynaCU-Bench

### 5.1 Design Principles

- Tasks must **require** dynamic visual or audio perception --- they should be unsolvable from static screenshots alone
- Represent realistic computer-use scenarios, not artificial constructs
- Reproducible: use recorded video/audio playback in controlled VM environments (extending OSWorld infrastructure)
- Cover all four dynamic content categories from Section 2
- Include human performance baselines to calibrate the ceiling
- 200+ tasks with difficulty stratification (easy/medium/hard) within each category
- Clearly differentiated from existing benchmarks: GUI-World evaluates video LLMs but not CU agents; VideoWebArena tests video-conditioned web tasks but not audio or transient UI events; CUA-Suite provides training data but not a CU agent evaluation protocol

### 5.2 Task Categories

**Category A: Video Comprehension (50+ tasks).** The agent watches video content and must act on what it saw.

- "Watch this tutorial video, then replicate the demonstrated steps in the application"
- "A product demo video is playing. After it ends, fill in the form with the product name and price shown"
- Difficulty levels: Easy (single fact extraction), Medium (multi-step procedure), Hard (temporal reasoning across video segments)

**Category B: Meeting / Live Audio (40+ tasks).** The agent listens and acts on spoken information.

- "Join this recorded video call. When the presenter mentions a URL, open it in the browser"
- "Listen to this conference talk. When it ends, type a one-paragraph summary"
- Difficulty levels: Easy (single spoken fact), Medium (multi-speaker dialogue), Hard (long meeting with action items)

**Category C: Transient UI Events (40+ tasks).** The agent must notice and react to ephemeral visual events.

- "Browse this website. Accept the cookie consent dialog when it appears" (dialog auto-dismisses after 5 seconds)
- "A file is downloading. When the download-complete notification appears, open the downloaded file"
- Difficulty levels: Easy (single transient event), Medium (multiple events, some distractors), Hard (timed sequence of events)

**Category D: Audio Alerts (30+ tasks).** The agent must react to non-speech sounds.

- "Work on this document. When you hear the calendar alarm, open the calendar and report the event"
- "Monitor this dashboard. When an alert sound plays, note the time"
- Difficulty levels: Easy (single alert), Medium (distinguish between alert types), Hard (multiple alerts requiring different responses)

**Category E: Combined Multimodal (40+ tasks).** Requires both visual and audio dynamic perception.

- "Watch this narrated tutorial and follow along step by step"
- "Join this screen-sharing session. The presenter will verbally instruct you to click specific elements"
- Difficulty levels: Easy (audio+visual aligned), Medium (audio and visual carry different information), Hard (complex multimodal reasoning)

### 5.3 Evaluation Protocol

- Each task executes in a real VM (Ubuntu/Windows/macOS)
- Video/audio content is pre-recorded for reproducibility
- Success criteria are execution-based: file content matches expected output, correct element was clicked, correct text was typed, etc.
- **Human baseline**: each task completed by 3 human annotators to establish ceiling performance
- Metrics:
  - Task success rate (primary)
  - Observation efficiency: keyframes captured per step, audio model activations per step
  - End-to-end task completion time (including observation overhead)
  - Token cost: additional tokens consumed by the AOI per task

### 5.4 Benchmark Availability

DynaCU-Bench will be publicly released with:
- Complete task definitions, VM images, and evaluation scripts
- Pre-recorded video/audio stimuli for reproducibility
- Human baseline results
- Hosted on [platform] with documentation

---

## 6. Evaluation

### 6.1 Setup

**CU models tested** (covering open-source and closed-source, single-image and multi-image):

- UI-TARS-72B (open-source, multi-image up to 5)
- OpenCUA-72B (open-source, multi-image up to 3)
- Qwen2.5-VL-72B CU (open-source, multi-image up to 4)
- Claude Sonnet Computer Use (closed-source, multi-image)
- OpenAI CUA (closed-source, multi-image)

**Observation configurations:**

1. **Standard**: one screenshot per step, no audio (current paradigm baseline)
2. **Uniform-1FPS**: capture at fixed 1 FPS, feed most recent N frames to model
3. **Uniform-3FPS**: capture at fixed 3 FPS, feed most recent N frames
4. **Pixel-diff only**: capture when pixel change ratio exceeds threshold, no CLIP filtering
5. **Random keyframes (matched budget)**: randomly sample the same number of frames as our method, to test whether CLIP selection provides better frames or just more frames
6. **Gemini 3.1 Pro (native video)**: Gemini with native video input at 1 FPS, representing the brute-force video model approach
7. **AOI (visual only)**: CLIP-based adaptive keyframes, no audio, no narration
8. **AOI (visual + ASR)**: CLIP keyframes + Whisper speech-only transcription
9. **AOI (full)**: CLIP keyframes + Qwen3 Omni audio scene understanding + visual narration

### 6.2 Main Results on DynaCU-Bench

Table: task success rate across all models x observation configurations x task categories.

Expected findings:

- **Standard** (screenshot loop): ~0% on video comprehension, meeting, and audio tasks. May partially succeed on transient UI tasks if timing is coincidentally favorable.
- **Uniform-1FPS / Uniform-3FPS**: moderate improvement on visual tasks but wastes model input slots on redundant frames; no improvement on audio tasks.
- **Pixel-diff only**: overcaptures due to ads, spinners, cursor blinks triggering false keyframes. Slightly better than uniform but noisier.
- **Random keyframes (matched budget)**: demonstrates CLIP selection provides better frames, not just more frames.
- **Gemini native video**: comparable or better on some visual tasks due to long context, but much higher token cost, no audio perception, no selective filtering.
- **AOI (visual only)**: strong improvement on visual dynamic tasks. CLIP correctly filters periodic noise.
- **AOI (visual + ASR)**: adds gains on meeting tasks (speech captured). Zero improvement on audio alert tasks (Whisper ignores non-speech sounds).
- **AOI (full)**: additional gains specifically on non-speech audio tasks, isolating the contribution of multimodal audio understanding over speech-only ASR.

### 6.3 Comparison with Gemini Native Video

Dedicated analysis comparing AOI-equipped CU models against Gemini with native video input:

- **Task success rate**: AOI-equipped models vs. Gemini on each DynaCU-Bench category
- **Token efficiency**: tokens consumed per task (Gemini processes all frames; AOI is selective)
- **Audio perception**: Gemini 3.1 Pro standard video mode has no audio processing; Gemini 3.1 Flash Live does. Compare both.
- **Model flexibility**: AOI works with any CU model; Gemini approach locks to one provider

### 6.4 Static Benchmark Verification

Run on OSWorld and WebArena with the AOI attached. Expected: identical performance to the standard loop. The observation layer produces zero keyframes and skips audio processing when the screen is static and silent.

### 6.5 Efficiency Analysis

Average per-step observation statistics by task category:

| Task Category | Avg Keyframes/Step | % Steps with Audio | Avg Overhead/Step | Additional Tokens/Step |
|---|---|---|---|---|
| Video Comprehension | 0.1--0.3 | ~90% | ~200ms | ~80--150 |
| Meeting Participation | 0.05--0.1 | ~80% | ~180ms | ~50--100 |
| Transient UI Events | 0.1--0.2 | ~5% | ~10ms | ~10--30 |
| Audio Alerts | ~0 | ~30% | ~70ms | ~20--50 |
| Static tasks (OSWorld) | ~0 | ~0% | ~0ms | 0 |

---

## 7. Analysis

### 7.1 CLIP Threshold Sensitivity

Sweep theta from 0.05 to 0.35. Plot task success rate and average keyframes per step. Expected: a broad plateau around theta in [0.12, 0.20] where accuracy is stable and keyframe count varies moderately. The method is not sensitive to precise threshold tuning.

### 7.2 Keyframe Selection: CLIP vs. Alternatives

Compare keyframe selection methods at matched frame budgets:

| Keyframe Method | Avg Keyframes/Step | Success (Visual Tasks) | False Positives |
|---|---|---|---|
| Uniform 1 FPS | Fixed | Baseline | N/A (all frames) |
| Pixel diff only | High (overcaptures) | Moderate | High (spinners, cursors) |
| Random (matched budget) | Matched | Lower | N/A |
| CLIP distance (ours) | Low (adaptive) | Highest | Low |

Demonstrate that CLIP filtering is critical for suppressing periodic noise while capturing semantically meaningful changes. Random selection at the same frame budget performs worse, confirming CLIP provides better frames, not just more.

### 7.3 Audio: Whisper vs. Qwen3 Omni

| Audio Method | Meeting Tasks | Audio Alert Tasks |
|---|---|---|
| No audio | ~0% | ~0% |
| Whisper (speech only) | Moderate | ~0% |
| Qwen3 Omni (full scene) | High | Significant |

The gap concentrates on tasks involving non-speech audio events. For speech-only tasks, they perform comparably. This justifies multimodal audio understanding over pure ASR.

### 7.4 Visual Narration Ablation

Compare with and without visual narration on long video comprehension tasks. Without narration, the agent loses visual context from early steps and performance degrades on tasks requiring recall of earlier visual content.

Break down by task length: narration matters more for longer tasks (>50 steps) and less for short tasks (<10 steps).

### 7.5 Failure Modes

Systematic analysis of when the AOI fails:

- **Fast visual transitions (<300ms)**: Events entirely between samples are missed.
- **CLIP false negatives**: Semantically different but visually similar screens (e.g., two spreadsheets with similar layouts, text-heavy pages with small but critical changes).
- **Visual narration hallucinations**: The CU model may misread numbers or hallucinate UI elements. These errors are permanently encoded in the trajectory.
- **Audio masking**: Non-speech sounds drowned out by concurrent speech or background noise.

---

## 8. Limitations

### 8.1 Action Latency Is Unsolved

The observation layer extends perception but does not accelerate decision-making. The CU model still requires 1--5 seconds per inference step. This makes the approach unsuitable for real-time video games, fast-paced interactive applications, or any task where delayed action leads to immediate failure. Solving action latency is an orthogonal problem requiring faster models, predictive action policies, or domain-specific control loops.

### 8.2 Temporal Resolution Floor

Sampling at ~3 Hz means events shorter than ~300ms may fall entirely between samples and be missed. Fine-grained motion properties (rotation direction, gesture trajectory, speed of movement) cannot be reliably captured from sparse keyframes. For human-paced computer use (meetings, web browsing, presentations), 300ms resolution is sufficient. For high-speed visual content, information is irretrievably lost.

### 8.3 Visual Narration Quality Is Bounded by the CU Model

Long-term visual memory depends on the CU model producing accurate text descriptions. Errors --- a misread number, a hallucinated UI element, an omitted detail --- are permanently encoded in the trajectory. Quality may be lower for complex visual content: dense charts, small text, multi-element diagrams.

### 8.4 Not a Replacement for Native Multimodal Models

For applications where a single model provider is acceptable and cost is secondary to capability, native real-time multimodal models (Gemini 3.1 Flash Live) may provide a simpler and potentially more capable solution. The AOI's advantages --- model-agnosticism, zero overhead on static tasks, no retraining --- are most valuable when users need to preserve their existing CU model choice, minimize overhead for mixed static/dynamic workloads, or deploy in open-source settings.

---

## 9. Related Work

### 9.1 Computer-Use Agents

Claude Computer Use (Anthropic), OpenAI CUA/Operator, UI-TARS v1.5/v2 (ByteDance), Surfer 2 (H Company), OpenCUA (HKU/xlang-ai), Qwen2.5-VL/Qwen3-VL (Alibaba), Fara-7B (Microsoft). All operate through the screenshot-reason-act loop described in Section 2. None addresses dynamic visual content or audio perception. Distinct from grounding-only models (ShowUI, SeeClick, OS-Atlas, UGround) that accept a single image and output coordinates.

### 9.2 Real-Time Multimodal Models

Gemini 3.1 Flash Live (Google, 2026) processes continuous audio/video/text streams in real time, achieving 90.8% on ComplexFuncBench Audio. Project Astra (Google DeepMind) extends this into a full multimodal assistant. These monolithic approaches embed continuous perception into a specific model. Our work is complementary: model-agnostic adaptive perception as an external layer.

### 9.3 Video Understanding for GUI

GUI-World (Chen et al., ICLR 2025): 12K+ GUI video annotations; video LLMs fall short on all temporal GUI tasks. VideoWebArena (Jang et al., ICLR 2025): long-context models perform worse with video tutorials than without. VideoGUI (Lin et al., NeurIPS 2024): GPT-4o performs poorly on visual-centric GUI planning. CUA-Suite (ServiceNow, 2026): 55 hours of 30fps video across 87 applications. These benchmarks identify the problem but do not propose solutions that work with existing CU agents.

Related efforts mine video for training data: VideoAgentTrek (2025), Watch and Learn (Song et al., 2025), Liu et al. (2025). These address training/in-context learning, not runtime observation.

### 9.4 Keyframe Selection

VideoTree (Wang et al., CVPR 2025): CLIP-based hierarchical frame selection. VideoAgent (Wang et al., ECCV 2024): LLM-driven iterative frame selection. SlowFast-LLaVA (Xu et al., Apple, 2024): dual-resolution without learned selection. Adaptive Keyframe Sampling (Tang et al., CVPR 2025): plug-and-play selection under fixed token budgets. Key finding: for short segments (<1 minute), simple methods (uniform sampling, CLIP distance) are near-optimal. All prior work targets offline video QA; our contribution is real-time observation filtering for CU agents, where the two-stage design is critical for <10ms latency.

### 9.5 Multimodal Audio Understanding

Pure ASR (Whisper, wav2vec) produces nothing for non-speech audio. Multimodal audio models (Qwen2-Audio, Qwen3 Omni, Qwen3.5-Omni) understand the full audio scene. No prior work addresses audio perception in GUI agents.

### 9.6 Long-Form Video Understanding via Text

LLoVi (Zhang et al., EMNLP 2024): caption-then-reason pipeline competitive with direct visual processing for long-form video QA. MA-LMM (He et al., CVPR 2024): streaming video with compressed memory banks. Our visual narration follows the caption-then-reason principle but narrations are generated by the CU model itself (no separate captioner), are task-relevant, and serve as persistence across an open-ended agent trajectory.

### 9.7 Adaptive Middleware for GUI Agents

Adaptive VLM Routing (AVR, 2026): lightweight routing layer reducing inference costs by 78%. Shares the "adaptive middleware" pattern with our work but operates on the action/reasoning channel rather than observation. Cradle (BAAI, ICML 2025): frame difference analysis for games, but pixel-level without semantic filtering and tightly coupled to its architecture.

World models (WebDreamer, CUWM, MobileDreamer) predict future states rather than observe current dynamics --- orthogonal and potentially complementary.

---

## 10. Conclusion

Computer-use agents cannot perceive dynamic content because their observation is limited to periodic static screenshots and they have no audio perception. We show this is an **observation problem, not a model problem**, and introduce the **Agent Observation Interface (AOI)** --- a model-agnostic perception layer complementary to the action interface studied by SWE-agent. The AOI's three components --- CLIP-based adaptive keyframe extraction, volume-gated multimodal audio scene understanding, and CU-model-generated visual narration --- are transparent for static, silent tasks (zero overhead), adaptive for dynamic content (capturing keyframes and audio descriptions only when semantically warranted), and compatible with any existing image-based CU model without retraining. On DynaCU-Bench, five diverse CU agents equipped with the AOI achieve 40--60% success on dynamic tasks where they previously score near zero, while maintaining identical performance on standard static benchmarks.

---

## Appendix

- **A**: Full DynaCU-Bench task list, evaluation criteria, and human baseline results
- **B**: Implementation details --- CLIP model variant, theta calibration procedure, volume gate threshold calibration, audio overlap duration, audio model prompt template
- **C**: Per-model detailed results tables
- **D**: Qualitative examples of CLIP-based keyframe selection across different dynamic content types
- **E**: Complete multi-step trajectory examples for video watching tasks, showing accumulated narration + audio over 20+ steps
- **F**: Latency profiling across hardware configurations
- **G**: Audio fragment resolution examples
- **H**: Token cost breakdown per model per task category
- **I**: Failure case analysis with qualitative examples
- **J**: Comparison with Gemini 3.1 Flash Live on representative tasks
- **K**: Agent step timeline visualization
- **L**: Additional ablations: volume gate effectiveness, audio overlap quality, context depth, multi-image vs. single-image

---

## References (Key Citations)

### Computer-Use Agents
- Anthropic. Claude Computer Use. 2024.
- OpenAI. Computer-Using Agent (CUA). 2025.
- ByteDance. UI-TARS: Pioneering Automated GUI Interaction with Native Agents. 2025.
- H Company. Surfer 2. 2025.
- HKU / xlang-ai. OpenCUA. 2025.
- Microsoft. Fara-7B. 2025.
- Alibaba. Qwen2.5-VL / Qwen3-VL Computer Use. 2025.

### GUI Video Understanding
- Chen et al. GUI-World: A Dataset for GUI-Oriented Multimodal LLM-based Agents. ICLR 2025.
- Jang et al. VideoWebArena: Evaluating Long Context Multimodal Agents with Video Tasks. ICLR 2025.
- Lin et al. VideoGUI: A Benchmark for GUI Automation from Instructional Videos. NeurIPS 2024.
- Zhang et al. VideoGameBench: Can LMMs Play Video Games? 2025.
- ServiceNow. CUA-Suite. 2026.

### Real-Time Multimodal Models
- Google. Gemini 3.1 Flash Live. 2026.
- Google DeepMind. Project Astra. 2024--2026.

### Keyframe Selection
- Wang et al. VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos. CVPR 2025.
- Wang et al. VideoAgent: Long-form Video Understanding with Large Language Model as Agent. ECCV 2024.
- Xu et al. SlowFast-LLaVA: A Strong Training-Free Baseline for Video Large Language Models. Apple, 2024.
- Tang et al. Adaptive Keyframe Sampling for Long Video Understanding. CVPR 2025.

### Audio Understanding
- Alibaba. Qwen3 Omni / Qwen3.5-Omni Technical Report. 2025--2026.
- Radford et al. Robust Speech Recognition via Large-Scale Weak Supervision (Whisper). 2023.

### Long-Form Video and Text
- Zhang et al. LLoVi: A Simple LLM Framework for Long-Range Video Question-Answering. EMNLP 2024.
- He et al. MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding. CVPR 2024.

### Agent Interfaces and Frameworks
- Yang et al. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. NeurIPS 2024.
- BAAI. Cradle: Empowering Foundation Agents Towards General Computer Control. ICML 2025.

### World Models and Adaptive Middleware
- Gu et al. WebDreamer: Is Your LLM Secretly a World Model of the Internet? 2024.
- Guan et al. CUWM: Computer Use World Model. Microsoft, 2026.
- Adaptive VLM Routing for Computer Use Agents. 2026.

### Video Data Mining for CU Agents
- VideoAgentTrek. 2025.
- Song et al. Watch and Learn: Learning to Use Computers from Online Videos. 2025.
- Liu et al. Learning from Online Videos at Inference Time for Computer-Use Agents. 2025.
