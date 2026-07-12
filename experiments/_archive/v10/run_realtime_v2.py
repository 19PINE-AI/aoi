#!/usr/bin/env python3
"""OpenAI Realtime with gpt-realtime-2.0 on the 12-task audio subset and
the 5-task visual sanity set.

Re-runs both subsets with the new (vision-capable) Realtime model and
writes results to:
  - results/v10_subset_openai_realtime_v2.json   (audio subset, 12 tasks)
  - results/v10_sanity_openai_realtime_v2.json   (visual sanity, 5 tasks)

Usage:
    python experiments/v10/run_realtime_v2.py
"""
import json
import logging
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from dynacubench.tasks_v3 import DynaCUBenchV3, TaskCategory  # noqa: E402
from aoi.realtime_baselines import OpenAIRealtimeBaseline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rt2")

OUT = PROJECT / "results"

SANITY_IDS = ["C-E1", "E-E1", "F-E1", "F-E2", "I-E1"]


def run_subset(label: str, task_filter, out_file_name: str, max_steps: int = 12):
    out_file = OUT / out_file_name
    bench = DynaCUBenchV3(html_tasks_dir=PROJECT / "benchmark_env/html_tasks")
    tasks = [t for t in bench.tasks if task_filter(t)]

    existing = []
    if out_file.exists():
        try:
            existing = json.load(open(out_file))
        except Exception:
            existing = []
    done_ids = {r["task_id"] for r in existing if r.get("success") is not None}

    evaluator = OpenAIRealtimeBaseline(vision_model="gpt-realtime-2.0",
                                       max_steps=max_steps)
    results = list(existing)

    for i, t in enumerate(tasks):
        if t.task_id in done_ids:
            continue
        log.info("[%s] [%d/%d] %s", label, i + 1, len(tasks), t.task_id)
        try:
            r = evaluator.run_task(t)
            results.append(r.to_dict() if hasattr(r, "to_dict") else r)
        except Exception as e:
            log.exception("Crash %s: %s", t.task_id, e)
            results.append({"task_id": t.task_id, "success": False,
                            "error": str(e)})
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    n_pass = sum(1 for r in results if r.get("success"))
    log.info("DONE %s: %d/%d", label, n_pass, len(results))
    return n_pass


def main():
    n_sanity = run_subset(
        "sanity",
        lambda t: t.task_id in SANITY_IDS,
        "v10_sanity_openai_realtime_v2.json",
        max_steps=8,
    )
    n_subset = run_subset(
        "audio_subset",
        lambda t: (
            t.category != TaskCategory.S_STATIC
            and t.task_id.split("-")[0] in {"A", "B", "G", "H"}
            and t.difficulty.value in ("easy",)  # 3 easy per category = 12
        ),
        "v10_subset_openai_realtime_v2.json",
        max_steps=15,
    )
    print(f"\n=== GPT-REALTIME-2.0 SUMMARY ===")
    print(f"  visual sanity (5 tasks):  {n_sanity}/5")
    print(f"  audio subset (~12 tasks): {n_subset}/~12")


if __name__ == "__main__":
    main()
