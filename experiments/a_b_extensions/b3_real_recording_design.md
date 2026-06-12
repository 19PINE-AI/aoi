# B3: Real YouTube/meeting recording pilot — design and decision

## Goal
Verify that the AOI's gains hold on real (non-synthesized) audio and video,
not just on edge-TTS speech and HTML-CSS animations.

## Existing infrastructure

DynaCU-Real-Local (Section 5.11 / Appendix C of the paper) covers 12 tasks
across 4 sub-domains (podcast / meeting / screencast / voice) using:
  - LibriVox Aesop's-Fables audiobooks (PD)
  - asciinema community screencasts (BSD-2-Clause)
  - FOSDEM 2024 talk excerpts (CC-BY)

The paper currently scores Standard 11/12 and AOI 11/12 on this set, with
the explicit note that ``answers are recoverable from the CU model's
pretraining, so [it] cannot also serve as a perception comparison.''

## The hard part: pretraining-resistant tasks on real audio

A faithful "real-content perception comparison" requires audio/video assets
whose answer is *not* in the CU model's pretraining.  Three strategies, each
with trade-offs:

  1. **Recently uploaded content (post-cutoff).**  Claude Opus 4.7 has a
     January 2026 cutoff; recordings uploaded after that date are guaranteed
     unseen.  This requires sourcing CC-licensed material from after that
     date and verifying the cutoff per-model.
  2. **Niche / low-traffic content.**  YouTube channels with <1k views per
     video, regional accents, technical hobbyist domains (ham-radio CQ
     contacts, knitting pattern tutorials, woodworking measurements).  These
     are present in pretraining only stochastically.
  3. **Perceptual-detail tasks on known content.**  Even if the model
     "knows" Aesop's fables, it cannot recover from pretraining the *exact*
     pause-length between sentences in the LibriVox recording or the
     specific speaker's emphasis pattern.  A perceptually-grounded question
     like ``how many seconds elapse between 'fox' and 'grapes' in the second
     sentence'' depends on the actual recording rather than the text.

## What is added to the revision

A complete real-content pilot meeting the above criteria is left to the
benchmark-design follow-up acknowledged in the limitations section.

Within this revision, we add:
  - An explicit one-paragraph note in Section 5.11 / Appendix C stating
    *why* DynaCU-Real-Local does not function as a perception comparison
    and what the three strategies above would look like.
  - A pointer to `experiments/a_b_extensions/b3_real_recording_design.md`
    for the design discussion.

## Why we don't fully execute B3 in this revision

Building a pretraining-resistant real-content benchmark is benchmark-design
work, not a small extension to an existing one:

  - 30+ candidate clips must be sourced, each manually annotated with a
    ground-truth answer derivable only from the audio/video signal.
  - For each clip, the question must be screened against the CU model's
    pretraining (run "what's in this LibriVox track" as a knowledge query;
    refine until the model fails *without* listening).
  - This iteration is task-by-task and the size of the result is
    proportional to the engineering effort, not to a clean methodological
    finding.

The signal from DynaCU-Bench (synthetic TTS) plus DynaCU-Real-Local
(real audio but pretraining-recoverable answers) bracket the perception
question well enough to justify the AOI's design.  A full real-content
pilot is the natural next paper, not the natural next 100-task run inside
this one.
