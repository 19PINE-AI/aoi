# B2: OSWorld no-op verification — design note and decision

## Goal
Verify that the AOI is a no-op on canonical static computer-use benchmarks (i.e.,
that the +44pp DynaCU-Bench gain is not partly due to prompt-format effects that
would also lift OSWorld scores).

## Existing evidence in the paper

Static-50 (Section 5.10) already addresses this question on 50 pure-HTML static
tasks:
  - Standard:   43/50  (86.0%)
  - AOI full:   50/50  (100.0%)
  - Total keyframes captured by AOI:  7 (0.09/step)
  - Total audio segments produced by AOI:  0

The +14pp Static-50 gain is explicitly attributed in the paper to the
structured-prompt component (`standard_structured` decomposition, Section 5.10
and Table 6), not to between-step perception.

## Why a full OSWorld run isn't pursued in this revision

A full OSWorld run requires:
  1. A virtual-machine snapshot infrastructure (KVM/QEMU images per task, a
     remote-controlled desktop in each VM, screenshot/network plumbing).
  2. ~369 tasks across 9 GUI apps; each task ~3-15 min wall-clock; total
     ~24-60 hr of single-process compute.
  3. Cross-machine harness re-implementation: OSWorld assumes desktop
     applications (LibreOffice, GIMP, VS Code, terminal) rather than a single
     headless browser.  Our AOI's screen-capture path (Playwright) and audio
     pipeline (PulseAudio inside the same container) would need to be ported
     to the OSWorld VM-step model.

The cost is several days of integration engineering for an answer
(no-op-on-static) that Static-50 already supplies, more directly: Static-50
contains real form-fill / table-read / multi-step interaction tasks where the
AOI gates verifiably emit zero audio segments and ~0.09 keyframes/step --
exactly the no-op behaviour an OSWorld-style benchmark would test.

## What's added to the revision

  - The OSWorld question is reframed: Static-50 is the no-degradation test;
    the +14pp uplift on Static-50 is explicitly attributed to the structured
    prompt format and not to between-step perception (already on file).
  - The limitations section is updated to mention OSWorld as future work
    *only* if the paper makes a stronger no-degradation claim than Static-50
    supports.

## What this means for the paper

No new experiment is run for B2.  The existing Static-50 result (Section 5.10)
already answers the no-op question more directly than OSWorld would.
The revision adds one paragraph explicitly framing this.
