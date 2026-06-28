#!/usr/bin/env python3
"""OpenAI Realtime GA (gpt-realtime-2, true websocket + native audio) on the
12-task audio subset and the 5-task visual sanity set.

This supersedes run_realtime_v2.py, whose chat-completions path 404'd on the
realtime model id and produced an all-wait() 0/12 artifact. Here we stream the
page's native audio over the GA Realtime websocket and provide the same
[PAGE ELEMENTS] id list the AOI agents get, so results reflect perception.

Writes:
  results/v10_subset_openai_realtime_ws.json   (audio subset, 12 tasks)
  results/v10_sanity_openai_realtime_ws.json   (visual sanity, 5 tasks)
"""
import json, logging, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from dynacubench.tasks_v3 import DynaCUBenchV3
from aoi.realtime_baselines import OpenAIRealtimeWSBaseline

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("rt-ws")

OUT = PROJECT / "results"
MODEL = "gpt-realtime-2"

# Same 12-task audio subset as run_realtime_subset.py (Pod/Meet/Phone/Intv ×3).
SUBSET_IDS = ["A-E1","A-E2","A-M1", "B-E1","B-E2","B-M1",
              "G-E1","G-E2","G-M1", "H-E1","H-E2","H-M1"]
# Same 5 visual-sanity tasks as run_realtime_v2.py.
SANITY_IDS = ["C-E1", "E-E1", "F-E1", "F-E2", "I-E1"]


def run_subset(label, ids, out_name, max_steps):
    out_file = OUT / out_name
    bench = DynaCUBenchV3(html_tasks_dir=PROJECT / "benchmark_env/html_tasks")
    tasks = [bench.get_task(i) for i in ids if bench.get_task(i)]

    results = []
    if out_file.exists():
        try:
            results = json.load(open(out_file))
        except Exception:
            results = []
    done = {r["task_id"] for r in results if r.get("error") is None and "success" in r}

    ev = OpenAIRealtimeWSBaseline(model=MODEL, max_steps=max_steps)
    for i, t in enumerate(tasks):
        if t.task_id in done:
            log.info("[%s] skip %s (done)", label, t.task_id); continue
        log.info("[%s] [%d/%d] %s", label, i + 1, len(tasks), t.task_id)
        t0 = time.time()
        try:
            r = ev.run_task(t)
            d = r.to_dict()
        except Exception as e:
            log.exception("crash %s", t.task_id)
            d = {"task_id": t.task_id, "success": False, "error": f"CRASH: {e}"}
        d["wall_s"] = round(time.time() - t0, 1)
        results = [x for x in results if x.get("task_id") != t.task_id] + [d]
        json.dump(results, open(out_file, "w"), indent=2)

    valid = [r for r in results if r.get("error") is None]
    npass = sum(1 for r in valid if r.get("success"))
    log.info("DONE %s: %d/%d valid (%d invalid)",
             label, npass, len(valid), len(results) - len(valid))
    return npass, len(valid)


def main():
    ns, nv_s = run_subset("sanity", SANITY_IDS, "v10_sanity_openai_realtime_ws.json", 8)
    na, nv_a = run_subset("audio", SUBSET_IDS, "v10_subset_openai_realtime_ws.json", 12)
    print("\n=== gpt-realtime-2 (GA websocket, native audio) ===")
    print(f"  visual sanity: {ns}/{nv_s}")
    print(f"  audio subset : {na}/{nv_a}")


if __name__ == "__main__":
    main()
