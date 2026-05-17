#!/usr/bin/env python3
"""
v10 paper-improvement experiments.

Three new evaluations:
  1. Static-50: Claude on the expanded 50-task static baseline (standard + aoi_full).
     Verifies the AOI does not degrade purely-static work and that the gates suppress
     observation correctly.

  2. Narration-discarded ablation: Claude on the 100 dynamic tasks with the model
     still generating narrations each step (so any chain-of-thought-while-acting
     benefit still applies), but the narrations are dropped from the trajectory.
     Resolves the confound between persistent-text-memory and inference-time CoT.

  3. (Optional, --selection-ablation) EvoCUA-32B + pixel_diff/random_keyframes
     to test whether the selection-method-doesn't-matter finding holds for an
     open-source model.

Run:
  python experiments/run_v10_experiments.py --static --narration-discard
  python experiments/run_v10_experiments.py --selection-ablation
"""
import argparse, json, logging, sys, time
from pathlib import Path
from typing import Optional

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.browser_eval import BrowserEvaluator, EvalResult
from dynacubench.tasks_v3 import DynaCUBenchV3, TaskCategory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("v10")

OUT = PROJECT / "results"


def run_one(model: str, mode: str, tasks, output_file: Path, max_steps: int = 15,
            step_interval_s: float = 2.0):
    log.info("=" * 70)
    log.info("Run: model=%s mode=%s n_tasks=%d -> %s",
             model, mode, len(tasks), output_file.name)
    log.info("=" * 70)

    evaluator = BrowserEvaluator(
        model_name=model,
        observation_mode=mode,
        max_steps=max_steps,
        step_interval_s=step_interval_s,
    )

    # Resume support: skip tasks already completed.
    done_ids = set()
    if output_file.exists():
        try:
            previous = json.load(open(output_file))
            done_ids = {r["task_id"] for r in previous if r.get("success") is not None}
            log.info("Resume: %d tasks already completed", len(done_ids))
        except Exception:
            previous = []
    else:
        previous = []

    results = list(previous)

    for i, task in enumerate(tasks):
        if task.task_id in done_ids:
            continue
        log.info("[%d/%d] %s (%s)", i + 1, len(tasks), task.task_id, task.difficulty.value)
        t0 = time.time()
        try:
            r = evaluator.run_task(task)
            results.append(r.to_dict())
        except Exception as e:
            log.exception("Task %s crashed: %s", task.task_id, e)
            results.append({
                "task_id": task.task_id,
                "category": task.category.value,
                "difficulty": task.difficulty.value,
                "model_name": model,
                "observation_mode": mode,
                "success": False,
                "result_val": "exception",
                "steps_taken": 0,
                "total_time_s": time.time() - t0,
                "total_model_latency_ms": 0,
                "total_obs_overhead_ms": 0,
                "steps": [],
                "error": str(e),
            })

        # Persist after every task so the run is resumable
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    n_pass = sum(1 for r in results if r.get("success"))
    log.info("DONE %s/%s: %d/%d passed", model, mode, n_pass, len(results))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true",
                        help="Run static-50 baseline (Claude std + AOI full)")
    parser.add_argument("--narration-discard", action="store_true",
                        help="Run aoi_full_no_narration_memory ablation on 100 dynamic tasks (Claude)")
    parser.add_argument("--selection-ablation", action="store_true",
                        help="Run pixel_diff + random_keyframes on EvoCUA-32B (100 tasks)")
    parser.add_argument("--realtime-gemini", action="store_true",
                        help="Run Gemini Live baseline on audio-heavy subset (A,B,G,H)")
    parser.add_argument("--realtime-openai", action="store_true",
                        help="Run OpenAI Realtime baseline on audio-heavy subset (A,B,G,H)")
    parser.add_argument("--realtime-tasks", default="A,B,G,H",
                        help="Comma-separated category prefixes for Realtime evals")
    parser.add_argument("--realcontent", action="store_true",
                        help="Run on the 12-task DynaCU-Real validation set")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--max-steps", type=int, default=15)
    args = parser.parse_args()

    bench = DynaCUBenchV3(html_tasks_dir=Path("benchmark_env/html_tasks"))
    static_tasks = [t for t in bench.tasks if t.category == TaskCategory.S_STATIC]
    dynamic_tasks = [t for t in bench.tasks if t.category != TaskCategory.S_STATIC]

    if args.static:
        run_one(args.model, "standard", static_tasks,
                OUT / "v10_static50_claude_standard.json", max_steps=args.max_steps)
        run_one(args.model, "aoi_full", static_tasks,
                OUT / "v10_static50_claude_aoi.json", max_steps=args.max_steps)

    if args.narration_discard:
        run_one(args.model, "aoi_full_no_narration_memory", dynamic_tasks,
                OUT / "v10_narration_discarded_claude.json", max_steps=args.max_steps)

    if args.selection_ablation:
        for mode in ("pixel_diff", "random_keyframes"):
            run_one("evocua-32b", mode, dynamic_tasks,
                    OUT / f"v10_evocua_{mode}.json", max_steps=args.max_steps)

    if args.realtime_gemini or args.realtime_openai:
        from aoi.realtime_baselines import GeminiLiveBaseline, OpenAIRealtimeBaseline
        prefixes = tuple(args.realtime_tasks.split(","))
        rt_tasks = [t for t in dynamic_tasks if t.task_id.split("-")[0] in prefixes]
        log.info("Realtime subset: %d tasks (%s)", len(rt_tasks), prefixes)

        for use, cls, fname in [
            (args.realtime_gemini,  GeminiLiveBaseline,    "v10_gemini_live.json"),
            (args.realtime_openai,  OpenAIRealtimeBaseline,"v10_openai_realtime.json"),
        ]:
            if not use:
                continue
            evaluator = cls(max_steps=args.max_steps)
            results = []
            out_file = OUT / fname
            if out_file.exists():
                results = json.load(open(out_file))
                done_ids = {r["task_id"] for r in results}
                rt_remaining = [t for t in rt_tasks if t.task_id not in done_ids]
            else:
                rt_remaining = rt_tasks
            for i, t in enumerate(rt_remaining):
                log.info("[%s] [%d/%d] %s", fname, i+1, len(rt_remaining), t.task_id)
                try:
                    r = evaluator.run_task(t)
                    results.append(r.to_dict())
                except Exception as e:
                    log.exception("Crashed: %s", e)
                    results.append({"task_id": t.task_id, "success": False,
                                    "error": str(e)})
                with open(out_file, "w") as f:
                    json.dump(results, f, indent=2)
            log.info("DONE %s: %d results", fname, len(results))

    if args.realcontent:
        # Real-content (DynaCU-Real-Local) tasks have IDs starting with 'R_'
        # and are not registered in tasks_v3.py.  Auto-register from disk.
        from dynacubench.tasks_v3 import Task, TaskDifficulty, EvalType
        rc_dir = PROJECT / "benchmark_env" / "html_tasks"
        rc_files = sorted(rc_dir.glob("R_*.html"))
        rc_tasks = []
        # Each task's getTaskResult() returns "<key>_correct" on success or
        # "<x>_typed"/"no_answer"/"incorrect" otherwise.  We pass the task-
        # specific success value below.
        success_map = {
            "R_pod1_aesop_fox.html":         "fox_correct",
            "R_pod2_aesop_lion.html":        "mouse_correct",
            "R_pod3_aesop_ant.html":         "ant_correct",
            "R_meet1_python.html":           "dynamic_correct",
            "R_meet2_postgres.html":         "mvcc_correct",
            "R_meet3_rust.html":             "tokio_correct",
            "R_cast1_git.html":              "clone_correct",
            "R_cast2_pip.html":              "pkg_typed",
            "R_cast3_npm.html":              "answer_typed",
            "R_voice1_yesno.html":           "answer_typed",
            "R_voice2_directions.html":      "direction_typed",
            "R_voice3_appointment.html":     "day_typed",
        }
        for f in rc_files:
            rc_tasks.append(Task(
                task_id=f.stem,
                category=TaskCategory.S_STATIC,  # category enum is required; results read R_* prefix
                difficulty=TaskDifficulty.MEDIUM,
                instruction=("Listen to / watch the on-page content and answer the question. "
                             "Type your answer into the text box and click Submit."),
                ground_truth="see_html",
                html_file=f.name,
                eval_type=EvalType.DOM,
                axes=[],
                dom_success_value=success_map.get(f.name),
                duration_s=90.0,
            ))
        run_one(args.model, "standard", rc_tasks,
                OUT / "v10_realcontent_claude_standard.json", max_steps=10,
                step_interval_s=3.0)
        run_one(args.model, "aoi_full", rc_tasks,
                OUT / "v10_realcontent_claude_aoi.json", max_steps=10,
                step_interval_s=3.0)


if __name__ == "__main__":
    main()
