#!/usr/bin/env python3
"""Grok Voice (grok-voice-think-fast) on the 12-task audio subset, NO scaffold.

Grok Voice has no vision, so "no scaffold" = native audio only, with no page
representation at all (no screenshot, no [PAGE ELEMENTS] text). This is the
most information-starved condition and parallels gpt-realtime-2 "alone".
Writes results/v10_subset_grok_voice_noscaffold.json
"""
import json, logging, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))
from dynacubench.tasks_v3 import DynaCUBenchV3
from aoi.realtime_baselines import OpenAIRealtimeWSBaseline

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("grok-voice-ns")

SUBSET_IDS = ["A-E1","A-E2","A-M1", "B-E1","B-E2","B-M1",
              "G-E1","G-E2","G-M1", "H-E1","H-E2","H-M1"]
OUT = PROJECT / "results" / "v10_subset_grok_voice_noscaffold.json"

bench = DynaCUBenchV3(html_tasks_dir=PROJECT / "benchmark_env/html_tasks")
ev = OpenAIRealtimeWSBaseline(
    model="grok-voice-latest", max_steps=12,
    provide_page_elements=False, send_images=False,   # audio only
    ws_base="wss://api.x.ai/v1/realtime", api_key_env="XAI_API_KEY")

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
from collections import defaultdict
cat = defaultdict(lambda: [0, 0]); order = {"A": "Pod", "B": "Meet", "G": "Phone", "H": "Intv"}
for r in valid:
    c = order.get(r["task_id"][0], "?"); cat[c][1] += 1; cat[c][0] += int(bool(r.get("success")))
print(f"\n=== grok-voice NO-SCAFFOLD audio subset: {npass}/{len(valid)} ===")
print("  " + "  ".join(f"{c} {cat[c][0]}/{cat[c][1]}" for c in ["Pod", "Meet", "Phone", "Intv"]))
print("  passed:", [r["task_id"] for r in valid if r.get("success")])
