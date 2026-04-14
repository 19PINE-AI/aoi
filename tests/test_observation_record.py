"""Tests for ObservationRecord and TrajectoryStore."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from PIL import Image
from aoi.observation_record import ObservationRecord, TrajectoryStore, StepRecord


def make_step_record(step_id: int, audio: str = "", narration: str = "No visual change.", action: str = "wait()") -> StepRecord:
    return StepRecord(
        step_id=step_id,
        step_start_time=float(step_id * 3.5),
        step_end_time=float((step_id + 1) * 3.5),
        audio_text=audio,
        visual_narration=narration,
        action=action,
        n_keyframes=0,
        audio_model_called=bool(audio),
    )


def test_observation_record_text_format():
    """ObservationRecord.to_prompt_text() should include all sections."""
    context = [
        make_step_record(1, audio="Hello world", narration="Slide visible"),
        make_step_record(2, audio="Second step", narration="New slide"),
    ]
    obs = ObservationRecord(
        step_id=3,
        context_steps=context,
        current_audio_text="Third step audio",
        keyframes=[],
        post_action_screenshot=Image.new("RGB", (100, 100)),
        task_instruction="Watch the video",
    )

    text = obs.to_prompt_text()
    assert "Step 3" in text
    assert "CONTEXT" in text
    assert "Hello world" in text
    assert "Slide visible" in text
    assert "Third step audio" in text
    assert "Watch the video" in text


def test_observation_record_empty_context():
    """Empty context should produce minimal output without CONTEXT section."""
    obs = ObservationRecord(
        step_id=1,
        context_steps=[],
        current_audio_text="",
        keyframes=[],
        post_action_screenshot=Image.new("RGB", (100, 100)),
        task_instruction="Type hello",
    )
    text = obs.to_prompt_text()
    assert "Step 1" in text
    assert "Type hello" in text


def test_trajectory_store_appends():
    """TrajectoryStore should accumulate step records."""
    store = TrajectoryStore(context_depth=3)

    for i in range(5):
        store.append(
            step_id=i + 1,
            step_start_time=float(i * 3.5),
            step_end_time=float((i + 1) * 3.5),
            audio_text=f"Audio step {i}" if i % 2 == 0 else "",
            visual_narration="Slide visible" if i == 2 else "No visual change.",
            action=f"action_{i}",
            n_keyframes=1 if i == 2 else 0,
            audio_model_called=i % 2 == 0,
        )

    assert len(store._steps) == 5


def test_trajectory_context_depth():
    """get_context should return at most context_depth steps."""
    store = TrajectoryStore(context_depth=2)

    for i in range(5):
        store.append(
            step_id=i + 1,
            step_start_time=float(i),
            step_end_time=float(i + 1),
            audio_text=f"audio {i}",  # dynamic content to trigger context
            visual_narration="change",
            action="wait",
            n_keyframes=0,
            audio_model_called=True,
        )

    context = store.get_context(current_step_id=6)
    assert len(context) <= 2


def test_trajectory_no_context_for_static():
    """Static tasks (no audio, no visual change) should get zero context."""
    store = TrajectoryStore(context_depth=3)

    for i in range(5):
        store.append(
            step_id=i + 1,
            step_start_time=float(i),
            step_end_time=float(i + 1),
            audio_text="",           # no audio
            visual_narration="No visual change.",  # no visual change
            action="wait",
            n_keyframes=0,
            audio_model_called=False,
        )

    context = store.get_context(current_step_id=6)
    assert len(context) == 0, "Static task should not carry context overhead"


def test_trajectory_last_audio_text():
    """get_last_audio_text should return the most recent audio."""
    store = TrajectoryStore()
    store.append(1, 0.0, 1.0, "First audio", "no change", "wait", 0, True)
    store.append(2, 1.0, 2.0, "", "no change", "wait", 0, False)
    store.append(3, 2.0, 3.0, "Latest audio", "no change", "wait", 0, True)

    assert store.get_last_audio_text() == "Latest audio"


def test_trajectory_summary():
    """summary() should return correct aggregate metrics."""
    store = TrajectoryStore()

    store.append(1, 0.0, 1.0, "audio", "change", "act", 2, True)
    store.append(2, 1.0, 2.0, "", "no change", "wait", 0, False)
    store.append(3, 2.0, 3.0, "audio2", "change2", "act2", 1, True)

    s = store.summary()
    assert s["steps"] == 3
    assert s["audio_steps"] == 2
    assert abs(s["audio_step_ratio"] - 2 / 3) < 0.01
    assert s["total_keyframes"] == 3
    assert abs(s["avg_keyframes_per_step"] - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
