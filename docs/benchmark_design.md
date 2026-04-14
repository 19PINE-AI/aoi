# DynaCU-Bench: Benchmark Design Document

## 1. Motivation

Current computer-use (CU) benchmarks — OSWorld, WebArena, VisualWebArena — evaluate agents on **static** browser tasks: fill a form, navigate a website, extract information from a page. These tasks can be solved from a single screenshot per action step. The agent takes a screenshot, reasons about it, acts, and repeats.

Real computer use is not static. Humans routinely:

- **Listen** to podcasts, meetings, and phone calls while using a browser
- **Watch** videos, screencasts, and live presentations with changing visual content
- **React** to transient notifications, error messages, and live dashboards
- **Speak** in interviews, voice calls, and collaborative sessions
- **Play** browser-based games requiring real-time visual tracking

None of these scenarios are captured by existing benchmarks because they require capabilities that current CU agents lack: **continuous audio perception**, **visual change tracking between screenshots**, and **real-time bidirectional interaction**.

DynaCU-Bench fills this gap. It is designed to evaluate the dynamic perception and interaction capabilities that the Agent Observation Interface (AOI) enables.

## 2. Design Methodology

### 2.1 Three Capability Axes

Every task in DynaCU-Bench targets one or more of three capability axes that distinguish dynamic computer use from static computer use:

| Axis | Description | Current CU agents |
|------|-------------|-------------------|
| **(a) Audio perception** | Extract information from spoken content (podcasts, meetings, phone calls) | Cannot do this. No audio channel. |
| **(b) Visual-temporal perception** | Track visual changes occurring between screenshot intervals (carousel slides, transient errors, video frames, dashboard updates) | Cannot do this. One screenshot per step (~5s interval). |
| **(c) Real-time interaction** | Respond within tight timing windows, carry out multi-turn exchanges, produce audio output | Limited. 5-10s latency per action. No audio output. |

### 2.2 Difficulty Progression

Tasks are organized into three difficulty tiers based on which axes they require:

| Tier | Axes required | Description | Expected standard agent performance | Expected AOI agent performance |
|------|--------------|-------------|-------------------------------------|-------------------------------|
| **Easy** | (a) or (b) alone | Passive observation, then act at leisure | ~0% (missing modality) | 60-80% |
| **Medium** | (a)+(b) together, or single axis under time pressure | Multi-modal observation, or observation with moderate interaction | ~0% | 30-60% |
| **Hard** | (a)+(b)+(c) | Real-time multi-modal interaction: perceive, reason, and respond under time pressure | ~0% | 10-30% |

The hard tier intentionally exceeds what current AOI-equipped agents can reliably do. This establishes a performance ceiling that maps the frontier for future work, ensuring the benchmark remains useful as agent capabilities improve.

### 2.3 Grounding in Realistic Scenarios

Every task must correspond to a **real activity that humans perform in a browser**. We reject synthetic test patterns (e.g., "a code flashes for 1.5 seconds") in favor of scenarios grounded in actual use:

- Listening to a podcast on Spotify
- Attending a Google Meet presentation
- Watching a YouTube tutorial
- Monitoring a Grafana dashboard
- Receiving a phone call in the browser
- Playing a browser-based game
- Collaborating on a Google Doc while on a call

The question "would a human actually do this in a browser?" must be answered affirmatively for every task.

### 2.4 No Reward Hacking

The evaluation harness must not provide agents with shortcuts that bypass the intended perception channel:

- **No `window._spokenContent` or DOM-based audio proxies.** Audio must flow through real speaker output, be captured via the audio pipeline, and be transcribed by ASR. If the ASR mishears, that is part of the task difficulty — just as a human might mishear in a noisy call.
- **No hidden DOM attributes exposing visual content.** Transient visual information must be captured via real screenshots/keyframes, not read from JavaScript variables.
- **No evaluation-only APIs.** The only special hook is `window.getTaskResult()` for verifying task completion state (did the agent click the right button, type the correct answer). This function checks DOM state — it does not expose task content.

The agent must perceive the computer exactly as a human does: through the screen and speakers. It must act exactly as a human does: through keyboard, mouse, and microphone.

## 3. Evaluation Harness Architecture

### 3.1 Browser Environment

- **Headless Chromium** via Playwright, running inside an Xvfb virtual display
- **Viewport**: 1280x720 pixels
- **Permissions**: microphone access granted (for tasks requiring agent speech input)
- **No network access to external sites** — all task pages are served locally as HTML files

### 3.2 Audio Infrastructure

Audio is the critical differentiator. The harness implements a full-duplex audio pipeline using PulseAudio virtual devices:

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (Chromium)                     │
│                                                          │
│  Page audio output ──► default audio sink                │
│  getUserMedia(mic) ◄── default audio source              │
└──────────┬───────────────────────────────┬───────────────┘
           │                               │
           ▼                               ▲
┌──────────────────────┐       ┌──────────────────────┐
│  PulseAudio Virtual  │       │  PulseAudio Virtual   │
│  Speaker Sink        │       │  Microphone Source     │
│  (monitor output)    │       │  (pipe source)         │
└──────────┬───────────┘       └───────────┬───────────┘
           │                               ▲
           ▼                               │
┌──────────────────────┐       ┌──────────────────────┐
│  parecord capture    │       │  pacat / ffmpeg       │
│  → WAV buffer        │       │  ← WAV from TTS      │
└──────────┬───────────┘       └───────────┬───────────┘
           │                               ▲
           ▼                               │
┌──────────────────────┐       ┌──────────────────────┐
│  Whisper ASR         │       │  TTS Engine           │
│  (speech → text)     │       │  (text → speech)      │
│  e.g., whisper-1,    │       │  e.g., Piper, Coqui,  │
│  faster-whisper      │       │  or cloud TTS API     │
└──────────┬───────────┘       └───────────┬───────────┘
           │                               ▲
           ▼                               │
┌──────────────────────────────────────────────────────────┐
│                      AOI Pipeline                        │
│                                                          │
│  Audio transcript ──► Observation Record ◄── Keyframes   │
│                              │                           │
│                              ▼                           │
│                     CU Model (LLM)                       │
│                              │                           │
│                              ▼                           │
│                   Action: click/type/speak                │
│                              │                           │
│              ┌───────────────┼───────────────┐           │
│              ▼               ▼               ▼           │
│          Mouse/KB      Keyboard          TTS → Mic       │
│          actions       typing            (speak action)  │
└──────────────────────────────────────────────────────────┘
```

**Speaker capture path (perception):**
1. Browser page generates audio via SpeechSynthesis, Web Audio API, or `<audio>` elements
2. Audio routes to PulseAudio virtual speaker sink
3. `parecord` captures from the sink's monitor output into a **rolling ring buffer** (last 60 seconds of raw audio, 16kHz float32)
4. Audio is transcribed and delivered to the agent in **two layers** (see §3.2.1)

**Microphone injection path (production):**
1. Agent's CU model outputs a `speak "text"` action
2. TTS engine converts text to audio waveform
3. Audio is injected into PulseAudio virtual microphone source via `pacat` or `ffmpeg`
4. Browser receives the audio via `getUserMedia({ audio: true })` — the page hears the agent "speak"

**Key properties:**
- No DOM-based shortcuts. Audio flows through real system audio channels.
- The same pipeline works for any page that produces or consumes audio.
- Latency target: < 2 seconds end-to-end (capture → transcribe → agent → TTS → inject).

### 3.2.1 Two-Layer Audio Representation

Audio is presented to the agent in two layers that mirror human perception. Humans can precisely associate what was *just* said with what they are currently seeing, but for older context they recall a coherent narrative without exact frame-by-frame correspondence.

**Layer 1 — Recent audio marker (last inter-screenshot interval, ~3-5s)**

This layer captures what was spoken during the most recent screenshot interval. It is tightly coupled with the keyframes from that interval, enabling the agent to correlate audio with visual changes.

```
recent_audio:
  transcript: "...and marketing is finalized at 1.4 million."
  start_t: 45.2s          # wall-clock time since task start
  end_t: 48.7s
  associated_keyframes:    # visual changes during this audio window
    - keyframe at 46.1s (slide transitioned to budget chart)
    - keyframe at 47.8s (presenter highlighted a bar)
```

This is produced by transcribing only the audio from the last inter-screenshot interval (~3-5 seconds). It is short enough that Whisper can transcribe it in < 500ms.

**Layer 2 — Long-horizon audio context (last 30-60s)**

This layer provides a continuous, uninterrupted transcription of the last 30-60 seconds of audio. It preserves sentence structure, argument flow, and speaker turns. It is **not** fragmented by screenshot boundaries.

```
audio_context:
  transcript: >
    Good morning everyone. Let me walk through the Q3 budget.
    Engineering is at 2.1 million, which is on track. Sales came
    in at 800K. For marketing, after our discussion last week,
    we've set the budget at 1.4 million, down from the initial
    request of 1.8 million. Operations is holding steady at 650K.
    And marketing is finalized at 1.4 million.
  start_t: 18.7s
  end_t: 48.7s
  sentence_timestamps:
    - {t: 18.7, text: "Good morning everyone."}
    - {t: 20.1, text: "Let me walk through the Q3 budget."}
    - {t: 23.4, text: "Engineering is at 2.1 million, which is on track."}
    - {t: 28.2, text: "Sales came in at 800K."}
    - {t: 31.0, text: "For marketing, after our discussion last week..."}
    - {t: 38.5, text: "Operations is holding steady at 650K."}
    - {t: 42.9, text: "And marketing is finalized at 1.4 million."}
```

This is produced by transcribing the full 60-second rolling audio buffer. Two approaches are supported:

1. **Rolling Whisper**: Transcribe the buffer with faster-whisper (supports 30s chunks natively; two passes for 60s). Self-contained, ~1-2s latency on GPU. Word-level timestamps available via Whisper's `--word_timestamps` mode.

2. **Streaming multimodal**: Feed the audio buffer (optionally with recent keyframes) to a multimodal model such as Gemini 1.5 Pro. Produces higher-quality transcripts with better speaker diarization and can incorporate visual context. Higher API cost and latency.

**Why two layers?**

Fragmenting audio into per-screenshot chunks destroys coherence. A sentence like "The budget was cut to 1.4 million because of the Q2 overspend" might span two screenshot intervals. Chopping it produces two fragments ("The budget was cut to" | "1.4 million because of the Q2 overspend") that are each harder to understand.

The two-layer design avoids this:
- Layer 1 provides tight audio-visual coupling for the *most recent* interval only (where it matters)
- Layer 2 provides coherent narrative context for *everything before* (where flow matters more than frame-level correlation)

**Timestamps** are included at sentence-level granularity in Layer 2, enabling the agent to reason about temporal ordering ("the speaker mentioned the deadline *before* showing the timeline slide") without requiring exact frame-level alignment.

### 3.3 Visual Observation

The AOI keyframe extractor runs in parallel with the agent loop:

1. **Capture thread** samples the browser at 3 fps (every 333ms)
2. **Stage 1 — Pixel gate**: if < 1% of pixels changed, skip (< 1ms cost)
3. **Stage 2 — CLIP semantic distance**: if cosine distance to anchor < 0.04, suppress (noise filter). Otherwise, emit keyframe and reanchor.
4. Keyframes are included in the agent's observation record as additional images

Standard mode provides only one screenshot per agent step (~5 seconds). AOI mode provides all captured keyframes between steps, enabling the agent to see transient visual events.

### 3.4 Observation Record (What the Agent Receives)

At each step, the agent receives an observation record combining all perception channels:

```
Observation Record — Step N (t = 48.7s)
═══════════════════════════════════════════════════════════

[AUDIO CONTEXT — last 60 seconds]
(18.7s) Good morning everyone.
(20.1s) Let me walk through the Q3 budget.
(23.4s) Engineering is at 2.1 million, which is on track.
(28.2s) Sales came in at 800K.
(31.0s) For marketing, after our discussion last week,
        we've set the budget at 1.4 million, down from
        the initial request of 1.8 million.
(38.5s) Operations is holding steady at 650K.
(42.9s) And marketing is finalized at 1.4 million.

[RECENT AUDIO — last 3.5 seconds, synced with visual]
(45.2s–48.7s) "And marketing is finalized at 1.4 million."
  ↳ During this audio: 1 keyframe captured (slide changed)

[VISUAL — current screenshot]
  → screenshot_step_N.png (1280x720)

[VISUAL — keyframes captured since last step]
  → keyframe_46.1s.png (slide transitioned to budget chart)
  → keyframe_47.8s.png (presenter highlighted a bar)

[TASK]
  "What is the finalized marketing budget?"

[PRIOR ACTIONS]
  Step N-1: wait 3s
  Step N-2: click(640, 400)
```

This structure gives the agent:
- **Coherent audio narrative** (Layer 2) for understanding context
- **Precise recent audio** (Layer 1) for correlating with the latest visual changes
- **Current screenshot** for spatial reasoning and action targeting
- **Keyframes** for visual events that occurred between screenshots
- **Action history** for continuity across steps

### 3.5 Agent Action Space

The agent can perform these actions:

| Action | Description |
|--------|-------------|
| `click <x> <y>` | Mouse click at pixel coordinates |
| `type "<text>"` | Type text via keyboard |
| `key <combo>` | Keyboard shortcut (e.g., Enter, Ctrl+A) |
| `scroll <direction> <amount>` | Scroll the page |
| `speak "<text>"` | Generate audio via TTS and inject into browser microphone |
| `wait <seconds>` | Pause before next observation |
| `done` | Signal task completion |

The `speak` action enables the agent to participate in voice interactions (Categories G, H, I). The harness converts the text to audio via TTS and injects it into the browser's microphone input through PulseAudio.

### 3.6 Evaluation Modes

| Mode | Screenshots | Keyframes | Audio capture | Audio output |
|------|:-----------:|:---------:|:-------------:|:------------:|
| `standard` | 1 per step | none | none | none |
| `aoi_visual` | 1 per step | yes (3fps) | none | none |
| `aoi_audio` | 1 per step | none | yes (Whisper) | none |
| `aoi_full` | 1 per step | yes (3fps) | yes (Whisper) | none |
| `aoi_interactive` | 1 per step | yes (3fps) | yes (Whisper) | yes (TTS) |

The ablation across modes isolates the contribution of each AOI component:
- `standard` vs `aoi_visual`: measures the value of keyframe extraction (visual-temporal perception)
- `standard` vs `aoi_audio`: measures the value of audio perception alone
- `aoi_visual` vs `aoi_full`: measures the marginal value of adding audio to visual observation
- `aoi_full` vs `aoi_interactive`: measures the value of audio output (speaking capability)

## 4. Task Categories

### Category A: Podcast / Audio Content
**Capability axis**: (a) audio perception
**Scenario**: Agent opens a podcast player page. An episode auto-plays via the browser's audio output. The page shows a player UI (artwork, progress bar, controls) but no transcript or show notes containing the answer. The agent must listen to the spoken content and answer a factual question.

| Difficulty | Description | Example |
|-----------|-------------|---------|
| Easy | Single fact extraction from short audio | "What price was mentioned in the product review?" |
| Medium | Two facts from a longer segment | "What was the drug name AND the efficacy rate?" |
| Hard | Must pause at the right moment and type what was heard, or answer a question requiring inference across the full audio | "Summarize the three main arguments presented" |

**Why standard agents fail**: No audio channel. The page shows only a play button and progress bar — no content is visible.

**Why AOI agents succeed**: Audio capture → Whisper transcription → agent receives the spoken content as text.

---

### Category B: Video Conference / Meeting
**Capability axis**: (a) audio + (b) visual-temporal
**Scenario**: Agent joins a simulated Google Meet / Zoom call. A presenter speaks while sharing slides. Slides auto-advance and are gone once passed. Some information is only spoken (not on slides), some is only on slides (not spoken). Answering requires perceiving both modalities.

| Difficulty | Description | Example |
|-----------|-------------|---------|
| Easy | Answer about something only spoken (slides provide context but not the answer) | "What is the new launch date?" (spoken by presenter, not on any slide) |
| Medium | Answer requires correlating audio with a specific slide that was shown briefly | "What was the East region's sales figure?" (slide showed blank, speaker filled in verbally) |
| Hard | Presenter asks the agent a question and waits for a response in chat or via audio within a time window | "Based on slide 2, what should we prioritize?" (agent must have seen slide 2 AND respond in time) |

**Why standard agents fail**: Miss auto-advancing slides (visual-temporal), miss spoken content (audio).

**Why AOI agents succeed**: Keyframes capture each slide transition; audio capture gets the presenter's speech.

---

### Category C: Video / Screencast Watching
**Capability axis**: (b) visual-temporal, some (a) audio
**Scenario**: Agent watches a YouTube-style tutorial or product demo. The video content is simulated with auto-advancing slides, terminal output, or animated demonstrations. Narration provides additional context via audio.

| Difficulty | Description | Example |
|-----------|-------------|---------|
| Easy | Identify a value shown in the video content | "What command was run in the terminal demo?" |
| Medium | Track a sequence of steps demonstrated | "List the three configuration steps shown" |
| Hard | Follow along with a tutorial and replicate steps in a side panel | "The video shows how to configure the settings — do it in the form below" |

**Why standard agents fail**: Video content changes continuously; a single screenshot captures only one frame of many.

**Why AOI agents succeed**: Keyframe extractor captures each visual transition in the video content.

---

### Category D: Carousel / Rotating Content
**Capability axis**: (b) visual-temporal
**Scenario**: Agent visits a web page with auto-rotating content — product carousels, sliding testimonials, cycling banners, news tickers. Information is spread across multiple slides/cards that rotate automatically. Understanding the full content requires observing the rotation over time.

| Difficulty | Description | Example |
|-----------|-------------|---------|
| Easy | Extract information from a specific carousel slide | "What promo code is shown on the winter sale banner?" |
| Medium | Compare information across multiple slides | "Which product has the lowest price across all carousel items?" |
| Hard | Track rapidly cycling content and aggregate | "How many testimonials mention 'excellent service'?" |

**Why standard agents fail**: A single screenshot captures only whichever slide is currently showing. Other slides have already rotated away or haven't appeared yet.

**Why AOI agents succeed**: Keyframes capture each slide as it appears, building a complete picture of the rotating content.

---

### Category E: Live Dashboard / Monitoring
**Capability axis**: (b) visual-temporal + (c) real-time interaction
**Scenario**: Agent monitors a live operations dashboard (server metrics, stock prices, IoT sensors, application logs). Values update continuously on screen.

| Difficulty | Description | Example |
|-----------|-------------|---------|
| Easy | Report a metric value that appeared briefly | "What was the peak CPU usage in the last 30 seconds?" |
| Medium | Detect an anomaly and take action | "Click the alert button when temperature exceeds 37 degrees" |
| Hard | Triage multiple simultaneous alerts, acknowledging each within a time window | "Acknowledge all critical alerts within 5 seconds of appearance" |

**Why standard agents fail**: Metric values change between screenshots. Brief spikes, threshold crossings, and transient alerts are missed.

**Why AOI agents succeed**: Continuous keyframe capture detects visual changes in the dashboard as they happen.

---

### Category F: Transient Errors & Notifications
**Capability axis**: (b) visual-temporal + (c) real-time interaction
**Scenario**: Agent interacts with a web application where transient UI elements appear briefly — error toasts after form submission, notification badges, cookie consent banners that auto-dismiss, download completion toasts, session expiration warnings. These elements contain critical information or require timely action.

| Difficulty | Description | Example |
|-----------|-------------|---------|
| Easy | Capture information from a toast/notification that appears for a few seconds | "What was the error code shown after form submission?" |
| Medium | React to a transient element before it auto-dismisses | "Click 'Accept' on the cookie consent banner before it disappears" |
| Hard | Handle a chain of transient events — error appears, agent must read it, fix the form, resubmit, and catch the success confirmation | "Fix the validation error and confirm the success message" |

**Why standard agents fail**: Transient elements appear and disappear between screenshot intervals. By the time the next screenshot is taken, the toast/error is gone.

**Why AOI agents succeed**: Keyframe extractor detects the visual change when a transient element appears and captures it.

---

### Category G: Voice / Phone Interaction (Inbound)
**Capability axis**: (a) audio perception + (c) real-time interaction
**Scenario**: Agent receives a simulated phone call or voice assistant interaction in the browser. The caller speaks and expects the agent to respond — either by typing into a form, clicking buttons, or speaking back.

| Difficulty | Description | Example |
|-----------|-------------|---------|
| Easy | Listen to a voicemail and type the callback number | "Enter the phone number from the voicemail" |
| Medium | Interactive voice menu: caller speaks options, agent must listen and respond | "The system says 'Press 1 for sales, 2 for support' — click the right option" |
| Hard | Multi-turn phone conversation where the caller asks questions and expects typed or spoken responses within a time window | "The caller asks your order number, then your shipping preference — respond to each" |

**Why standard agents fail**: No audio perception. The phone UI shows a call timer and participant info but not what is being said.

**Why AOI agents succeed**: Audio capture transcribes the caller's speech, enabling the agent to understand and respond.

---

### Category H: Voice Interview / Audio Output (Outbound)
**Capability axis**: (a) audio perception + (c) real-time interaction (audio output)
**Scenario**: Agent participates in a mock interview conducted via a browser-based video call. The interviewer asks questions and expects the agent to respond by speaking (via the `speak` action, which generates audio through TTS and injects it into the browser's microphone input). The page listens via `getUserMedia` and processes the agent's spoken responses.

| Difficulty | Description | Example |
|-----------|-------------|---------|
| Easy | Interviewer asks a simple factual question; agent must speak the answer | "What is the capital of France?" → agent speaks "Paris" |
| Medium | Interviewer asks about information displayed on screen; agent must read and speak | "What is the revenue figure shown in the report?" → agent reads the page and speaks the number |
| Hard | Interviewer shows a video/presentation, asks questions about it, and expects spoken responses within a time window | "Based on the demo you just watched, what was the main feature?" → agent must have perceived the visual content AND speak the answer |

**Why standard agents fail**: Cannot produce audio output. Cannot perceive the interviewer's audio questions.

**Why AOI agents succeed (partially)**: Can perceive audio. The `speak` action with TTS enables audio output. Hard tasks remain challenging due to the multi-modal perception + generation requirement.

**Note**: This category tests a capability frontier. The `speak` action requires the harness to support TTS → microphone injection. Even AOI-equipped agents may struggle with the latency and coordination required for natural conversation.

---

### Category I: Collaborative Editing (Shared Document + Phone Call)
**Capability axis**: (a) audio + (b) visual-temporal + (c) real-time interaction
**Scenario**: Agent and a simulated human collaborator both work on a shared document (Google Docs style) while on a phone call. The collaborator makes edits that appear in real time and discusses changes verbally. The agent must observe the changes, listen to instructions, and make its own edits in response.

| Difficulty | Description | Example |
|-----------|-------------|---------|
| Easy | Collaborator dictates text via audio; agent types it into the document | "Type the following paragraph..." |
| Medium | Collaborator says "fix the number in section 2" while editing section 1; agent must find and fix the right section | Audio instruction references visual content |
| Hard | Full back-and-forth: collaborator edits, discusses, asks questions, and the agent must respond with both edits and speech simultaneously | Real-time collaborative work session |

**Why standard agents fail**: Cannot hear the collaborator's instructions (audio). Miss the collaborator's real-time edits (visual-temporal). Cannot respond fast enough for interactive collaboration.

**Why AOI agents succeed (partially)**: Audio perception captures instructions. Keyframes capture edit changes. But the coordination of listening + observing + editing + speaking simultaneously remains extremely challenging.

---

### Category J: Interactive Game / Real-time Response
**Capability axis**: (b) visual-temporal + (c) real-time interaction
**Scenario**: Agent plays a browser-based game requiring visual tracking and timed responses. Games are simple enough that the rules are self-evident from the UI but require continuous observation and timely action.

| Difficulty | Description | Example |
|-----------|-------------|---------|
| Easy | Click a target that appears on screen, generous timing (~5 second window) | "Click the mole when it pops up" |
| Medium | Pattern-following: watch a sequence, then reproduce it | "Watch the Simon Says pattern and repeat it" |
| Hard | Continuous real-time gameplay requiring sub-second decisions | "Navigate the character through obstacles" or "Catch falling objects" |

**Why standard agents fail**: Game state changes continuously. A screenshot every 5 seconds misses most game events. Reaction times of 5-10 seconds are too slow for gameplay.

**Why AOI agents succeed (partially)**: Keyframe capture at 3fps tracks game state changes. But agent reaction time (model inference latency) may still be too slow for hard tasks requiring sub-second responses.

---

## 5. Task Count and Distribution

Each category contains 10 tasks:
- 3 Easy
- 4 Medium
- 3 Hard

Total: **10 categories x 10 tasks = 100 tasks**

Distribution by primary modality requirement:

| Primary modality | Categories | Task count |
|-----------------|------------|------------|
| Audio perception | A, G, H | 30 |
| Visual-temporal perception | C, D, E, F, J | 50 |
| Audio + visual combined | B, I | 20 |

Distribution by interaction requirement:

| Interaction level | Categories | Task count |
|------------------|------------|------------|
| Passive observation → act at leisure | A, C, D (easy) | ~25 |
| Observation + timed response | B, E, F, G (easy/medium) | ~35 |
| Real-time multi-turn interaction | G, H, I, J (medium/hard) | ~40 |

## 6. Success Verification

Each HTML task page implements a `window.getTaskResult()` function that returns the current task state by querying DOM state:

- `"pending"` — task not yet completed
- `"<success_value>"` — task-specific success indicator (e.g., `"price_correct"`, `"code_entered"`)
- `"timeout"` — task time limit exceeded
- `"<failure_value>"` — specific failure mode (e.g., `"wrong_answer"`, `"missed_deadline"`)

This function checks observable DOM state (input field values, clicked buttons, page state) — it does **not** expose task content that should come through audio or visual observation channels.

## 7. Evaluation Metrics

### Primary metric: Task Success Rate
Percentage of tasks completed successfully, broken down by:
- Category (A-J)
- Difficulty tier (Easy / Medium / Hard)
- Observation mode (standard / aoi_visual / aoi_audio / aoi_full / aoi_interactive)
- Model (Claude, Gemini, GPT-4o)

### Secondary metrics:
- **Steps to completion**: How many agent steps were needed
- **Time to completion**: Wall-clock time from task start to success
- **Observation overhead**: Time spent on perception (keyframe extraction, ASR) vs. model inference
- **Audio output quality**: For Category H tasks, how often the page correctly receives the agent's spoken response

### Key comparisons:
1. **standard vs. aoi_full**: The headline number. How much does AOI improve task success?
2. **aoi_visual vs. aoi_audio vs. aoi_full**: Ablation showing the contribution of each modality
3. **Easy vs. Medium vs. Hard**: Performance degradation curve showing where current capabilities break down
4. **Cross-model comparison**: Do different LLMs benefit equally from AOI?
