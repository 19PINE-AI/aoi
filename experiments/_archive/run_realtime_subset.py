#!/usr/bin/env python3
"""Run a Realtime baseline on a specific task subset for the paper comparison.

Designed for the Section 7 streaming-baselines comparison: a small but
balanced cross-category subset that any reasonable eval budget can finish.
"""
import argparse, json, sys, time, logging
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from dynacubench.tasks_v3 import DynaCUBenchV3

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("rt-subset")

# A balanced 12-task subset across the audio-heavy categories.
# Each category gets 3 tasks: 2 easy + 1 medium.
SUBSET_IDS = [
    "A-E1","A-E2","A-M1",   # podcast
    "B-E1","B-E2","B-M1",   # meeting
    "G-E1","G-E2","G-M1",   # phone
    "H-E1","H-E2","H-M1",   # interview
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, choices=["openai_realtime","gemini_live"])
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    bench = DynaCUBenchV3(html_tasks_dir=Path("benchmark_env/html_tasks"))
    tasks = [bench.get_task(t) for t in SUBSET_IDS if bench.get_task(t)]
    log.info("Subset: %d tasks", len(tasks))

    from aoi.realtime_baselines import OpenAIRealtimeBaseline, GeminiLiveBaseline
    cls = OpenAIRealtimeBaseline if args.baseline == "openai_realtime" else GeminiLiveBaseline
    evaluator = cls(max_steps=args.max_steps)

    out_file = Path(args.out or f"results/v10_subset_{args.baseline}.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    results = []
    if out_file.exists():
        try:
            results = json.load(open(out_file))
        except Exception:
            results = []
    done_ids = {r.get("task_id") for r in results}

    for i, t in enumerate(tasks):
        if t.task_id in done_ids:
            log.info("Skip %s (already done)", t.task_id)
            continue
        log.info("[%d/%d] %s", i+1, len(tasks), t.task_id)
        t0 = time.time()
        try:
            r = evaluator.run_task(t)
            results.append(r.to_dict())
        except Exception as e:
            log.exception("Crashed %s: %s", t.task_id, e)
            results.append({"task_id": t.task_id, "success": False, "error": str(e),
                            "total_time_s": time.time() - t0})
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    n_pass = sum(1 for r in results if r.get("success"))
    log.info("DONE %s: %d/%d", args.baseline, n_pass, len(results))

if __name__ == "__main__":
    main()
