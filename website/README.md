# AOI Paper Website

Static React site presenting the paper **"Agent-Computer Observation Interfaces Enable
Dynamic Computer Use"**: interactive result visualizations, recorded agent trajectory
videos, a step-by-step trajectory explorer, and the live-playable DynaCU-Bench task suite.

## Develop / build

```bash
npm install
npm run dev        # local dev server
npm run build      # static site → dist/  (fully self-contained, host anywhere)
```

The build uses relative asset paths (`base: './'`), so `dist/` works from any
sub-path on static hosting (GitHub Pages, S3, nginx).

## Regenerating content

All site content is derived from the repo's raw evaluation outputs:

- **`scripts/build_data.py`** — recomputes every chart/table JSON in
  `public/data/` from `results/*.json` plus the task catalog from
  `dynacubench/tasks_v3.py`. Run with plain `python3`.
- **`scripts/record_trajectories.py`** — replays logged agent trajectories
  (actions, timing, narration, audio captions) on the original
  `benchmark_env/html_tasks/` pages with Playwright and records the videos in
  `public/videos/` (webm → mp4 + poster via ffmpeg). The soundtrack is
  reconstructed: page speechSynthesis utterances are captured with timestamps
  and re-rendered with edge-tts (GuyNeural — the same voice the eval harness
  played into the virtual speaker), then muxed into the mp4; agent `speak()`
  output is voiced with AriaNeural. Run with the project Python:
  `python scripts/record_trajectories.py`.
- **`public/tasks/`** — verbatim copy of `benchmark_env/html_tasks/` +
  `realcontent_assets/` so tasks run live in the browser.
- **`public/paper/aoi-paper.pdf`** — copy of `paper/main.pdf`.

## Site sections

| Section | Content |
|---|---|
| Hero | Title, abstract summary, headline numbers (computed from data) |
| Architecture | Three-component AOI diagram |
| Results | Main 6-model chart, per-category heatmap, ablations, θ sweep, streaming baselines, Gemini-3 four-way decomposition, controls |
| Recordings | Trajectory replay videos — AOI vs standard pairs + per-category gallery |
| Trajectories | Explorer over 8 full evaluation runs (100 tasks each) |
| Benchmark | All 150 DynaCU-Bench tasks, playable live in a modal |
