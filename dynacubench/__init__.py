"""
DynaCU-Bench: A benchmark of computer-use tasks requiring dynamic visual and/or audio perception.

Task Categories:
  A: Video Comprehension (tasks requiring watching video content)
  B: Meeting / Live Audio (tasks requiring listening to speech)
  C: Transient UI Events (tasks requiring catching ephemeral UI elements)
  D: Audio Alerts (tasks requiring detecting non-speech sounds)
  E: Combined Multimodal (tasks requiring both visual and audio dynamic perception)
"""

from .tasks import Task, TaskCategory, TaskDifficulty, DynaCUBench
from .synthetic_media import SyntheticMediaGenerator
from .evaluator import TaskEvaluator, EvaluationResult

__all__ = [
    "Task", "TaskCategory", "TaskDifficulty", "DynaCUBench",
    "SyntheticMediaGenerator", "TaskEvaluator", "EvaluationResult",
]
