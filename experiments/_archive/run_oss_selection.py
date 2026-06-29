#!/usr/bin/env python3
"""Selection-method ablation on an open-source model.

Tests whether the convergence finding (uniform / pixel-diff / random / CLIP all
score similarly) holds for an open-source model, not just Claude.

We use the *visual* categories (C, D, E, F, J = 50 tasks) because selection
methods only differ on visual content.  Audio-only categories would all score
the same regardless of selection because no keyframes are captured.

Original plan was EvoCUA-32B, but vLLM cannot start on the current host due
to an NVML driver/library version mismatch (NVML 595.58 vs torch's loaded
library), which blocks vLLM's CUDA platform inference even after
monkey-patching workarounds.  We substitute Qwen3-8B (already running on
port 8001 from before the NVML drift).  The convergence claim does not
depend on absolute scores, only on whether the selection methods cluster.
"""
import argparse, json, logging, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from experiments.browser_eval import BrowserEvaluator
from dynacubench.tasks_v3 import DynaCUBenchV3, TaskCategory

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("oss-sel")

OUT = PROJECT / "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--modes", default="pixel_diff,random_keyframes",
                   help="comma-separated list of modes to evaluate")
    p.add_argument("--categories", default="C,D,E,F,J",
                   help="comma-separated category prefixes (visual-temporal cats)")
    args = p.parse_args()

    bench = DynaCUBenchV3(html_tasks_dir=Path("benchmark_env/html_tasks"))
    cat_prefixes = tuple(args.categories.split(","))
    tasks = [t for t in bench.tasks
             if t.category != TaskCategory.S_STATIC
             and t.task_id.split("-")[0] in cat_prefixes]
    log.info("Visual subset: %d tasks across categories %s",
             len(tasks), cat_prefixes)

    for mode in args.modes.split(","):
        out_file = OUT / f"v10_oss_{args.model}_{mode}.json"
        evaluator = BrowserEvaluator(
            model_name=args.model, observation_mode=mode,
            max_steps=args.max_steps, step_interval_s=2.0,
        )
        results = []
        if out_file.exists():
            try:
                results = json.load(open(out_file))
            except Exception:
                pass
        done = {r.get("task_id") for r in results}

        log.info("=" * 70)
        log.info("Running %s + %s on %d remaining tasks",
                 args.model, mode, len(tasks) - len(done))
        log.info("=" * 70)
        for i, t in enumerate(tasks):
            if t.task_id in done:
                continue
            log.info("[%s] [%d/%d] %s", out_file.name, i+1, len(tasks), t.task_id)
            try:
                r = evaluator.run_task(t)
                results.append(r.to_dict())
            except Exception as e:
                log.exception("Crashed: %s", e)
                results.append({"task_id": t.task_id, "success": False, "error": str(e)})
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)
        n_pass = sum(1 for r in results if r.get("success"))
        log.info("DONE %s/%s: %d/%d", args.model, mode, n_pass, len(results))


if __name__ == "__main__":
    main()
