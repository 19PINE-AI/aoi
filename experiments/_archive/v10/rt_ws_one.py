#!/usr/bin/env python3
"""Single-task integration smoke test for the GA Realtime websocket adapter.

Runs one task end-to-end and dumps the step log + what the model heard, so we
can confirm native audio + tool-calling work before running the full subset.
"""
import json, logging, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

from dynacubench.tasks_v3 import DynaCUBenchV3
from aoi.realtime_baselines import OpenAIRealtimeWSBaseline

TASK_ID = sys.argv[1] if len(sys.argv) > 1 else "A-E1"

bench = DynaCUBenchV3(html_tasks_dir=PROJECT / "benchmark_env/html_tasks")
task = bench.get_task(TASK_ID)
print(f"TASK {task.task_id} [{task.category.value}/{task.difficulty.value}]: {task.instruction}")

ev = OpenAIRealtimeWSBaseline(model="gpt-realtime-2", max_steps=12)
r = ev.run_task(task)

print("\n===== RESULT =====")
print("success    :", r.success)
print("final_score:", r.final_score)
print("result_val :", r.result_val)
print("steps      :", r.steps_taken)
print("error      :", r.error)
print("heard_audio:", (r.heard_audio or "")[:600])
print("\n--- steps ---")
for s in r.steps:
    sd = s if isinstance(s, dict) else s.__dict__
    print(f"  {sd['step']:>2} | {sd['tool_call']:<32} | {sd['result_val']}")
