"""
Agent Observation Interface (AOI)

A model-agnostic perception layer that augments computer-use agents
with adaptive dynamic visual and audio perception.
"""

from .keyframe_extractor import KeyframeExtractor
from .audio_observer import AudioObserver
from .observation_record import ObservationRecord, TrajectoryStore
from .agent_loop import AOIAgentLoop

__all__ = [
    "KeyframeExtractor",
    "AudioObserver",
    "ObservationRecord",
    "TrajectoryStore",
    "AOIAgentLoop",
]
