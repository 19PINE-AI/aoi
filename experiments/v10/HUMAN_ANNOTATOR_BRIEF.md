# Human Annotator Brief — DynaCU-Bench Calibration Subset

## Why we are running this

The DynaCU-Bench paper reports task-success numbers for six computer-use
(CU) agents.  Reviewers will reasonably ask: "is 82% near-ceiling or
near-floor for these tasks?"  Without a human baseline we cannot
answer.  This study collects a human reference rate on a sampled subset
of DynaCU-Bench so the paper can quote a calibration figure (e.g.
"human reference: 93%; AOI-full Claude: 82%").

This is a calibration exercise, not a competition.  We are not asking
annotators to beat the agent; we are asking how a competent computer
user does on the same fixed task set.

## Annotator profile we are recruiting

- Comfortable using a web browser daily.
- Native or fluent English (most spoken audio is in English).
- Willing to use headphones for the audio tasks.
- ~90 minutes per session.

We need 2-3 annotators each running the full 20-task subset.  Hourly rate:
TBD by user; we suggest at least the local prevailing rate for short
remote piecework.

## What the annotator does

For each of the 20 tasks:
1. Open the task's HTML file in a fresh Chromium window (we provide a
   script).
2. Read the task instruction shown at the top of the page.
3. Solve the task by interacting with the page (clicking, typing,
   listening to audio, etc.) within the task's time budget (Easy 90 s,
   Medium 150 s, Hard 240 s).
4. When the page shows a "submitted / accepted" indicator (or the
   annotator believes the task is done), close the tab and mark
   pass/fail on a paper or digital form provided in `tally.csv`.
5. Move to the next task.  Annotators may not re-attempt a task they
   marked done.

Annotators are explicitly **allowed**:
- to take notes during audio playback;
- to re-open a task **once** if they accidentally closed it before
  submitting;
- to skip a task and mark it as failed; we want a realistic refusal rate.

Annotators are explicitly **not allowed**:
- to use any AI assistant (no copilot, no GPT, no Claude, no Grok);
- to consult external search engines for task content;
- to share or discuss tasks with other annotators during the session.

## Sampled subset (20 tasks: 2 per category × 10 categories)

The subset is fixed across annotators so we can compute inter-annotator
agreement.

| Category | Easy | Medium |
|---|---|---|
| A — Podcast | A-E1, A-M1 |
| B — Meeting | B-E1, B-M2 |
| C — Video | C-E1, C-M1 |
| D — Carousel | D-E1, D-M1 |
| E — Dashboard | E-E1, E-M1 |
| F — Transient UI | F-E1, F-M1 |
| G — Phone | G-E1, G-M1 |
| H — Interview | H-E1, H-M1 |
| I — Collab | I-E1, I-M1 |
| J — Games | J-E1, J-M1 |

We deliberately omit Hard tasks: collecting a human ceiling on hard
multi-step tasks is a much larger study; for the v10 paper we just want
a credible calibration number on Easy and Medium.

## Scoring

Each task has a DOM-side success check (or LLM judge, per task).  After
the annotator marks pass/fail in `tally.csv`, the harness runs the same
success check used by the agent runs.  The annotator's self-mark is for
sanity; the official score is the harness's.

A task is "annotator-passed" if both the annotator self-marks pass and
the harness check confirms.

## Inter-annotator agreement

We will report Cohen's kappa on the per-task pass/fail across annotators.
Anything below kappa = 0.6 means the task definition is ambiguous and
we will flag it in the limitations section.

## What we report in the paper

A single sentence in §6.2 or §6.7 of the form:

> A human reference rate, collected by [N] annotators on a fixed 20-task
> subset (2 per category × 10 categories, Easy and Medium only) under
> the same time budgets and submission protocol, is [X]% ± [Y]%, vs.\
> Claude Sonnet 4.6 + AOI full at [matching subset score]%.

That's it.  We are not building an annotation dataset; we just want a
single calibration number.

## Logistics

- The annotation harness lives at
  `experiments/v10/run_human_subset.py` (TBD: launcher that opens each
  task HTML in turn, starts a timer, and records the post-submit DOM
  state).
- Each annotator gets a unique annotator ID; results go into
  `results/v10_human_<annotatorid>.json`.
- An aggregation script `experiments/v10/aggregate_human.py` (TBD)
  produces the calibration number and the kappa figure.

## Ethics / consent

Annotators sign a one-page consent form covering: (1) they may stop at
any time, (2) we record only their pass/fail mark and total time, (3)
their browsing history is not retained, (4) they will be paid in full
even if they fail or skip tasks.  Template at `consent_form.md` (TBD).

## Open items for the user to fill in

1. Hourly rate.
2. Recruitment channel (Mechanical Turk?  Internal team?  Friends/family
   for pilot?).
3. Whether to run a 5-task pilot first to refine the protocol before
   running the full 20.
4. Sign-off on the omitted-hard-tasks decision.
