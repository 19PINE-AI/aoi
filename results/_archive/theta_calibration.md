# CLIP Theta Calibration Results

## Measurement Setup
- HTML tasks rendered in headless Chromium (Playwright) at 1280x720
- CLIP ViT-B/16 on RTX PRO 6000 Blackwell (sm_120)
- Frames captured at 1.25 fps (800ms intervals)
- Task: D_transient_session_warning.html (modal overlay appears/disappears)

## CLIP Cosine Distance Measurements

| Frame Transition     | Cosine Distance | Visual Change |
|---------------------|----------------|---------------|
| Static (no change)  | 0.0006         | None (noise)  |
| Modal appears       | 0.0805         | Session warning overlay |
| Modal disappears    | 0.0796         | Overlay removed |
| First frame vs any  | 0.0286-0.0805  | Varying with modal state |

## Pixel Change Ratios (64x64 grayscale)

| Frame Transition     | Pixel Change Ratio | Threshold Met (0.01)? |
|---------------------|-------------------|----------------------|
| Static              | 0.0000            | No                    |
| Modal appears       | 0.9304            | Yes                   |
| Modal disappears    | 0.9314            | Yes                   |
| Small UI shift      | 0.0015            | No                    |

## Theta Selection

- **Original theta (0.15)**: Zero keyframes detected. Too aggressive for web UI changes.
- **Calibrated theta (0.04)**: 2 keyframes detected (modal appear + disappear). Correctly captures meaningful visual changes while filtering static noise (0.0006 << 0.04).

### Theta Sensitivity

| Theta | Keyframes Detected | Notes |
|-------|-------------------|-------|
| 0.15  | 0                 | Misses all web UI changes |
| 0.10  | 0                 | Still too aggressive |
| 0.08  | 1                 | Captures only the larger transition |
| 0.05  | 2                 | Captures both modal events |
| 0.04  | 2                 | Optimal — captures events, filters noise |
| 0.01  | 3                 | Starts capturing minor rendering noise |
| 0.001 | 10                | Captures everything (uniform sampling) |

**Selected: theta = 0.04** — captures web UI state changes (modals, toasts, slide transitions) while filtering anti-aliasing noise and static frame noise.
