# Extensions Summary — A/B Experiments

## A1: Keyframe Causal Probe on Gemini 3 Flash

Subset: 50 tasks from B, C, D, E, F (visual-active categories).

| Mode | Pass/Total | Rate | 95% Wilson CI |
|------|-----------|------|----------------|
| aoi_full_max1kf | 20/50 | 40.0% | [27.6, 53.8] |
| aoi_full_max2kf | 18/50 | 36.0% | [24.1, 49.9] |
| aoi_full_max3kf | 20/50 | 40.0% | [27.6, 53.8] |
| aoi_full_max5kf_ref | 18/50 | 36.0% | [24.1, 49.9] |
| aoi_full_noise_kf | 17/50 | 34.0% | [22.4, 47.8] |
| aoi_full_dup_kf | 20/50 | 40.0% | [27.6, 53.8] |
| aoi_full_reorder_kf | 22/50 | 44.0% | [31.2, 57.7] |
| aoi_audio_ref | 24/50 | 48.0% | [34.8, 61.5] |

**Pairwise McNemar p-values:**
- aoi_full_max1kf vs aoi_full_max3kf: b=3, c=3, p=1, n=50
- aoi_full_max1kf vs aoi_full_noise_kf: b=6, c=3, p=0.508, n=50
- aoi_full_max3kf vs aoi_full_noise_kf: b=5, c=2, p=0.453, n=50
- aoi_full_max3kf vs aoi_full_dup_kf: b=3, c=3, p=1, n=50
- aoi_full_noise_kf vs aoi_full_dup_kf: b=1, c=4, p=0.375, n=50
- aoi_full_max3kf vs aoi_full_reorder_kf: b=1, c=3, p=0.625, n=50

**Per-category breakdown:**
| Mode | B_meeting | C_video | D_carousel | E_dashboard | F_transient |
|---|---|---|---|---|---|
| aoi_full_max1kf | 6/10 | 8/10 | 4/10 | 1/10 | 1/10 |
| aoi_full_max2kf | 5/10 | 6/10 | 6/10 | 1/10 | 0/10 |
| aoi_full_max3kf | 6/10 | 9/10 | 4/10 | 1/10 | 0/10 |
| aoi_full_noise_kf | 6/10 | 6/10 | 5/10 | 0/10 | 0/10 |
| aoi_full_dup_kf | 7/10 | 7/10 | 5/10 | 1/10 | 0/10 |
| aoi_full_reorder_kf | 7/10 | 9/10 | 5/10 | 1/10 | 0/10 |

## A2: Prompt-Format Decomposition (Claude Sonnet 4.6)

| Mode | Pass/Total | Rate | 95% Wilson CI |
|------|-----------|------|----------------|
| standard_minimal | 16/100 | 16.0% | [10.1, 24.4] |
| standard_ref | 38/100 | 38.0% | [29.1, 47.8] |
| standard_pageel_only | 46/100 | 46.0% | [36.6, 55.7] |
| standard_structured_ref | 57/100 | 57.0% | [47.2, 66.3] |
| aoi_full_ref | 82/100 | 82.0% | [73.3, 88.3] |

**Pairwise McNemar p-values:**
- standard_minimal vs standard: b=0, c=22, p=4.77e-07, n=100
- standard_minimal vs standard_pageel_only: b=1, c=31, p=1.54e-08, n=100
- standard_pageel_only vs standard_structured: b=3, c=14, p=0.0127, n=100
- standard vs standard_structured: b=3, c=22, p=0.000157, n=100

## A3: Narration Content Quality Audit (Claude Sonnet 4.6)

- Audited memory-load tasks: 10
- memory_hit: 2 (20.0%)
- cot_only: 8 (80.0%)

Detailed JSON: results/extensions/a3_narration_audit.json

## B1: Open-Source Model Replication

| Model | Standard | AOI full | Δ | McNemar p |
|-------|----------|---------|----|-----------|
| Qwen3-VL-235B-A22B-Instruct | 22/100 | 64/100 | +42 | 4.46e-10 (b=4, c=46) |
| Qwen3-VL-30B-A3B-Instruct | 18/100 | 42/100 | +24 | 1.82e-04 (b=8, c=32) |
