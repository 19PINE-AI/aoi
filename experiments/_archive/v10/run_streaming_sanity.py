#!/usr/bin/env python3
"""Adapter sanity-check for the streaming baselines (Section 7.6).

Runs both Gemini Live and OpenAI Realtime on a small purely-visual sanity
set where no audio comprehension is required.  If the baselines pass at
least one task each, the adapter is functional and the 0/12 + 3/12
audio-subset scores reflect actual model limitations rather than
adapter bugs.

The chosen sanity set is C-E1, E-E1, F-E1, F-E2, I-E1 (one per category
across C, E, I; two from F to cover both transient-banner subtypes).

Usage:
    python experiments/v10/run_streaming_sanity.py
"""
import json
import logging
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from dynacubench.tasks_v3 import DynaCUBenchV3  # noqa: E402
from aoi.realtime_baselines import GeminiLiveBaseline, OpenAIRealtimeBaseline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sanity")

SANITY_IDS = ["F-E2", "F-E1", "E-E1", "I-E1", "C-E1"]


def main():
    bench = DynaCUBenchV3(html_tasks_dir=PROJECT / "benchmark_env/html_tasks")
    sanity = [t for t in bench.tasks if t.task_id in SANITY_IDS]
    log.info("Sanity tasks: %s", [t.task_id for t in sanity])

    results_dir = PROJECT / "results"
    results_dir.mkdir(exist_ok=True)
    summary = {}

    for label, cls, kwargs in [
        ("openai_realtime", OpenAIRealtimeBaseline, {"max_steps": 8}),
        ("gemini_live", GeminiLiveBaseline, {"max_steps": 8}),
    ]:
        out_file = results_dir / f"v10_sanity_{label}.json"
        existing = []
        if out_file.exists():
            try:
                existing = json.load(open(out_file))
            except Exception:
                existing = []
        done_ids = {r["task_id"] for r in existing if r.get("success") is not None}
        rows = list(existing)
        log.info("=== %s ===", label)
        try:
            ev = cls(**kwargs)
        except Exception as e:
            log.exception("init failed: %s", e)
            continue
        for t in sanity:
            if t.task_id in done_ids:
                continue
            log.info("[%s] %s", label, t.task_id)
            try:
                r = ev.run_task(t)
                d = r.to_dict() if hasattr(r, "to_dict") else {
                    "task_id": getattr(r, "task_id", t.task_id),
                    "success": getattr(r, "success", False),
                    "result_val": getattr(r, "result_val", ""),
                    "steps_taken": getattr(r, "steps_taken", 0),
                }
            except Exception as e:
                log.exception("crash: %s", e)
                d = {"task_id": t.task_id, "success": False, "error": str(e)}
            rows.append(d)
            with open(out_file, "w") as f:
                json.dump(rows, f, indent=2)
        n_pass = sum(1 for r in rows if r.get("success"))
        log.info("DONE %s: %d/%d", label, n_pass, len(rows))
        summary[label] = (n_pass, len(rows))

    print("=== SUMMARY ===")
    for k, (np_, nt) in summary.items():
        print(f"  {k}: {np_}/{nt}")


if __name__ == "__main__":
    main()
