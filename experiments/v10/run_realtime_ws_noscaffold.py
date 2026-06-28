#!/usr/bin/env python3
"""gpt-realtime-2 on the 12-task audio subset WITHOUT the [PAGE ELEMENTS]
scaffold (screenshot + native audio only) — matches the original streaming
baselines' information access. Disentangles "new model" from "DOM scaffold".

Writes results/v10_subset_openai_realtime_ws_noscaffold.json
"""
import json, logging, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))
from dynacubench.tasks_v3 import DynaCUBenchV3
from aoi.realtime_baselines import OpenAIRealtimeWSBaseline

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("rt-ws-ns")

SUBSET_IDS = ["A-E1","A-E2","A-M1", "B-E1","B-E2","B-M1",
              "G-E1","G-E2","G-M1", "H-E1","H-E2","H-M1"]
OUT = PROJECT / "results" / "v10_subset_openai_realtime_ws_noscaffold.json"

bench = DynaCUBenchV3(html_tasks_dir=PROJECT / "benchmark_env/html_tasks")
ev = OpenAIRealtimeWSBaseline(model="gpt-realtime-2", max_steps=12,
                             provide_page_elements=False)
results = json.load(open(OUT)) if OUT.exists() else []
done = {r["task_id"] for r in results if r.get("error") is None and "success" in r}
for i, tid in enumerate(SUBSET_IDS):
    if tid in done:
        continue
    t = bench.get_task(tid)
    log.info("[%d/%d] %s", i + 1, len(SUBSET_IDS), tid)
    try:
        d = ev.run_task(t).to_dict()
    except Exception as e:
        log.exception("crash %s", tid)
        d = {"task_id": tid, "success": False, "error": f"CRASH: {e}"}
    results = [x for x in results if x.get("task_id") != tid] + [d]
    json.dump(results, open(OUT, "w"), indent=2)

valid = [r for r in results if r.get("error") is None]
npass = sum(1 for r in valid if r.get("success"))
print(f"\n=== gpt-realtime-2 NO-SCAFFOLD audio subset: {npass}/{len(valid)} ===")
