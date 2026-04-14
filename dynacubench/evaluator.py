"""
DynaCU-Bench Evaluation Harness.

Runs agent loops against benchmark tasks and collects metrics:
- Task success rate (primary)
- Observation efficiency (keyframes/step, audio activations/step)
- Additional tokens consumed by AOI per task
- End-to-end task completion time
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from .tasks import Task, TaskCategory, TaskDifficulty

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    task_id: str
    category: str
    difficulty: str
    success: bool
    agent_output: str
    n_steps: int
    n_keyframes_total: int
    n_audio_activations: int
    total_tokens: int
    wall_time_s: float
    observation_mode: str
    model_name: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "success": self.success,
            "agent_output": self.agent_output[:200],
            "n_steps": self.n_steps,
            "n_keyframes_total": self.n_keyframes_total,
            "n_audio_activations": self.n_audio_activations,
            "total_tokens": self.total_tokens,
            "wall_time_s": self.wall_time_s,
            "observation_mode": self.observation_mode,
            "model_name": self.model_name,
            "error": self.error,
        }


class SimulatedEnvironment:
    """
    A simulated task environment for headless evaluation.

    Serves synthetic frames and audio for a task stimulus,
    and checks for success based on agent actions.
    """

    def __init__(self, task: Task, media_generator):
        self.task = task
        self.media_gen = media_generator
        self._frames: list[tuple[float, Image.Image]] = []
        self._audio: Optional[object] = None
        self._frame_idx = 0
        self._start_time = None
        self._agent_actions: list[str] = []

        self._setup()

    def _setup(self):
        """Load or generate the task stimulus."""
        from .synthetic_media import SyntheticMediaGenerator
        gen = self.media_gen if self.media_gen else SyntheticMediaGenerator()

        # Generate stimulus based on task
        make_fn = getattr(gen, f"make_task_{self.task.task_id.replace('-', '').lower()}_stimulus", None)
        if make_fn:
            self._frames, audio_data = make_fn()
            if audio_data is not None:
                self._audio_data = audio_data
        else:
            # Default: static frame for the duration
            slide = [{"text": self.task.instruction[:40], "duration_s": self.task.duration_s,
                      "bg_color": (240, 240, 240)}]
            self._frames = gen.create_slideshow_frames(slide)
            self._audio_data = None

    def get_screenshot(self) -> Image.Image:
        """Return the current frame (simulating screen capture)."""
        if not self._frames:
            return Image.new("RGB", (1280, 720), (200, 200, 200))

        # Advance frame based on elapsed time
        if self._start_time is None:
            self._start_time = time.time()

        elapsed = time.time() - self._start_time
        # Find the appropriate frame for current time
        for i, (t, frame) in enumerate(self._frames):
            if t >= elapsed:
                return frame.copy()
        return self._frames[-1][1].copy()

    def inject_audio_to_buffer(self, audio_buffer) -> None:
        """Pre-load audio data into the AOI audio buffer."""
        if self._audio_data is not None and audio_buffer is not None:
            audio_buffer.inject_synthetic(self._audio_data)

    def record_action(self, action: str) -> bool:
        """Record an agent action and check for success."""
        self._agent_actions.append(action)
        if self.task.success_fn:
            return self.task.success_fn(action)
        return self.task.ground_truth in action

    def get_all_actions_text(self) -> str:
        return " | ".join(self._agent_actions)

    def check_success(self) -> bool:
        """Check overall task success based on all actions taken."""
        all_actions = self.get_all_actions_text()
        if self.task.success_fn:
            return self.task.success_fn(all_actions)
        return str(self.task.ground_truth) in all_actions


class TaskEvaluator:
    """
    Runs evaluation of an agent (with or without AOI) across benchmark tasks.
    """

    def __init__(
        self,
        cu_model,
        aoi_config=None,
        observation_mode: str = "aoi_full",
        model_name: str = "unknown",
        output_dir: Path = Path("results"),
    ):
        self.cu_model = cu_model
        self.aoi_config = aoi_config
        self.observation_mode = observation_mode
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_task(self, task: Task, max_steps: int = 20) -> EvaluationResult:
        """Evaluate a single task and return the result."""
        from .synthetic_media import SyntheticMediaGenerator
        from aoi.agent_loop import AOIAgentLoop, AOIConfig

        gen = SyntheticMediaGenerator()
        env = SimulatedEnvironment(task, gen)

        # Set up the AOI loop
        config = self.aoi_config or AOIConfig(mode=self.observation_mode)
        config.mode = self.observation_mode
        config.audio_backend = "none" if self.observation_mode == "standard" else config.audio_backend
        config.post_action_buffer_ms = 100  # Speed up for evaluation

        loop = AOIAgentLoop(
            cu_model=self.cu_model,
            config=config,
            execute_action=env.record_action,
            take_screenshot=env.get_screenshot,
        )
        loop.start()

        # Pre-load audio
        env.inject_audio_to_buffer(loop.audio_buffer)

        t_start = time.time()
        error = None
        results = []

        try:
            def done_fn(step_result):
                return env.check_success()

            results = loop.run_task(
                task=task.instruction,
                max_steps=max_steps,
                done_fn=done_fn,
                screenshot_fn=env.get_screenshot,
            )
        except Exception as e:
            logger.error("Task %s failed: %s", task.task_id, e)
            error = str(e)
        finally:
            loop.stop()

        wall_time = time.time() - t_start
        success = env.check_success()

        stats = loop.get_efficiency_stats()
        n_keyframes = stats["trajectory"].get("total_keyframes", 0)
        n_audio = stats["trajectory"].get("audio_steps", 0)
        total_tokens = sum(r.token_cost.get("total", 0) for r in results)

        return EvaluationResult(
            task_id=task.task_id,
            category=task.category.value,
            difficulty=task.difficulty.value,
            success=success,
            agent_output=env.get_all_actions_text(),
            n_steps=len(results),
            n_keyframes_total=n_keyframes,
            n_audio_activations=n_audio,
            total_tokens=total_tokens,
            wall_time_s=wall_time,
            observation_mode=self.observation_mode,
            model_name=self.model_name,
            error=error,
        )

    def evaluate_benchmark(
        self,
        benchmark,
        max_tasks: Optional[int] = None,
        max_steps_per_task: int = 20,
        categories: Optional[list] = None,
    ) -> list[EvaluationResult]:
        """Run evaluation across all (or filtered) benchmark tasks."""
        tasks = list(benchmark)
        if categories:
            tasks = [t for t in tasks if t.category in categories]
        if max_tasks:
            tasks = tasks[:max_tasks]

        results = []
        for i, task in enumerate(tasks):
            logger.info(
                "[%d/%d] Evaluating %s (%s/%s) with mode=%s",
                i + 1, len(tasks), task.task_id,
                task.category.value, task.difficulty.value,
                self.observation_mode,
            )
            result = self.evaluate_task(task, max_steps=max_steps_per_task)
            results.append(result)
            logger.info("  -> success=%s, steps=%d, keyframes=%d, audio=%d",
                        result.success, result.n_steps, result.n_keyframes_total, result.n_audio_activations)

        return results

    def save_results(self, results: list[EvaluationResult], filename: str = "results.json"):
        path = self.output_dir / filename
        data = [r.to_dict() for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Results saved to %s", path)
        return path

    def compute_summary(self, results: list[EvaluationResult]) -> dict:
        """Compute aggregate metrics from evaluation results."""
        if not results:
            return {}

        total = len(results)
        successful = sum(1 for r in results if r.success)

        # By category
        by_cat = {}
        for result in results:
            cat = result.category
            if cat not in by_cat:
                by_cat[cat] = {"total": 0, "success": 0}
            by_cat[cat]["total"] += 1
            by_cat[cat]["success"] += int(result.success)

        # By difficulty
        by_diff = {}
        for result in results:
            diff = result.difficulty
            if diff not in by_diff:
                by_diff[diff] = {"total": 0, "success": 0}
            by_diff[diff]["total"] += 1
            by_diff[diff]["success"] += int(result.success)

        avg_keyframes_per_step = 0.0
        avg_audio_ratio = 0.0
        total_steps = sum(r.n_steps for r in results)
        if total_steps > 0:
            total_kf = sum(r.n_keyframes_total for r in results)
            total_audio = sum(r.n_audio_activations for r in results)
            avg_keyframes_per_step = total_kf / total_steps
            avg_audio_ratio = total_audio / total_steps

        return {
            "model": results[0].model_name if results else "unknown",
            "mode": results[0].observation_mode if results else "unknown",
            "total_tasks": total,
            "success_rate": successful / total if total > 0 else 0.0,
            "n_successful": successful,
            "by_category": {
                cat: {"success_rate": v["success"] / v["total"], **v}
                for cat, v in by_cat.items()
            },
            "by_difficulty": {
                diff: {"success_rate": v["success"] / v["total"], **v}
                for diff, v in by_diff.items()
            },
            "avg_keyframes_per_step": avg_keyframes_per_step,
            "avg_audio_activation_ratio": avg_audio_ratio,
            "avg_steps_per_task": total_steps / total if total > 0 else 0,
            "avg_tokens_per_task": sum(r.total_tokens for r in results) / total if total > 0 else 0,
        }
