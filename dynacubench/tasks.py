"""
DynaCU-Bench Task Definitions.

Each task specifies:
- category and difficulty
- stimulus: pre-recorded video/audio, synthetic UI event, or mixed
- instruction: what the agent must do
- success_fn: deterministic check of task completion
- ground_truth: expected output or action
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Any
from pathlib import Path


class TaskCategory(Enum):
    A_VIDEO = "A_video_comprehension"
    B_MEETING = "B_meeting_audio"
    C_TRANSIENT_UI = "C_transient_ui"
    D_AUDIO_ALERT = "D_audio_alert"
    E_COMBINED = "E_combined_multimodal"


class TaskDifficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class Task:
    task_id: str
    category: TaskCategory
    difficulty: TaskDifficulty
    instruction: str                      # Natural language task for the agent
    ground_truth: Any                     # Expected answer/action
    success_fn: Optional[Callable] = None # Callable(agent_output) -> bool
    stimulus_video: Optional[Path] = None # Pre-recorded video path
    stimulus_audio: Optional[Path] = None # Pre-recorded audio path
    duration_s: float = 30.0             # Expected duration of stimulus
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "instruction": self.instruction,
            "ground_truth": str(self.ground_truth),
            "duration_s": self.duration_s,
        }


class DynaCUBench:
    """
    The full DynaCU-Bench benchmark.
    Loads task definitions and manages evaluation runs.
    """

    def __init__(self, media_dir: Path = Path("media")):
        self.media_dir = media_dir
        self.tasks: list[Task] = []
        self._register_all_tasks()

    def _register_all_tasks(self):
        """Register all benchmark tasks."""
        self.tasks.extend(self._category_a_tasks())
        self.tasks.extend(self._category_b_tasks())
        self.tasks.extend(self._category_c_tasks())
        self.tasks.extend(self._category_d_tasks())
        self.tasks.extend(self._category_e_tasks())

    # ─────────────────────────────────────────────────────────────
    # Category A: Video Comprehension
    # ─────────────────────────────────────────────────────────────

    def _category_a_tasks(self) -> list[Task]:
        tasks = []

        # A-001 EASY: Extract a single fact from a short video
        tasks.append(Task(
            task_id="A-001",
            category=TaskCategory.A_VIDEO,
            difficulty=TaskDifficulty.EASY,
            instruction=(
                "A product demo video is playing on screen. "
                "After it finishes, type the product name shown in the video into the text box."
            ),
            ground_truth="CloudSync Pro",
            success_fn=lambda output: "cloudsync pro" in output.lower(),
            duration_s=15.0,
            metadata={"synthetic": True, "text_on_screen": "CloudSync Pro"},
        ))

        # A-002 EASY: Count slides in a presentation video
        tasks.append(Task(
            task_id="A-002",
            category=TaskCategory.A_VIDEO,
            difficulty=TaskDifficulty.EASY,
            instruction=(
                "A slide presentation is playing. Count the total number of slides "
                "and enter the count into the text field."
            ),
            ground_truth="5",
            success_fn=lambda output: "5" in output,
            duration_s=25.0,
            metadata={"synthetic": True, "n_slides": 5},
        ))

        # A-003 MEDIUM: Extract multiple facts
        tasks.append(Task(
            task_id="A-003",
            category=TaskCategory.A_VIDEO,
            difficulty=TaskDifficulty.MEDIUM,
            instruction=(
                "Watch the tutorial video. Then fill in the form: "
                "Product name, price, and release date as shown in the video."
            ),
            ground_truth={"name": "DataVault 3.0", "price": "$299", "date": "March 2026"},
            success_fn=lambda output: all(
                k in output.lower() for k in ["datavault", "299", "march"]
            ),
            duration_s=30.0,
            metadata={"synthetic": True},
        ))

        # A-004 MEDIUM: Follow demonstrated steps
        tasks.append(Task(
            task_id="A-004",
            category=TaskCategory.A_VIDEO,
            difficulty=TaskDifficulty.MEDIUM,
            instruction=(
                "Watch the tutorial video that demonstrates how to create a new folder "
                "and rename it to 'reports'. Then perform the same action."
            ),
            ground_truth="folder_named_reports_exists",
            success_fn=lambda output: "reports" in output.lower(),
            duration_s=20.0,
            metadata={"synthetic": True, "action_sequence": ["create_folder", "rename_reports"]},
        ))

        # A-005 HARD: Temporal reasoning across video segments
        tasks.append(Task(
            task_id="A-005",
            category=TaskCategory.A_VIDEO,
            difficulty=TaskDifficulty.HARD,
            instruction=(
                "Watch the recorded presentation. The presenter shows three quarterly "
                "revenue figures across three slides. Calculate the total and type it."
            ),
            ground_truth="$2.4M",
            success_fn=lambda output: "2.4" in output or "2400" in output,
            duration_s=45.0,
            metadata={"synthetic": True, "values": [700000, 900000, 800000]},
        ))

        # A-006 HARD: Long video with multiple visual events
        tasks.append(Task(
            task_id="A-006",
            category=TaskCategory.A_VIDEO,
            difficulty=TaskDifficulty.HARD,
            instruction=(
                "A 60-second screen recording shows a developer navigating through a codebase. "
                "After the video, answer: how many files were opened? Type the number."
            ),
            ground_truth="4",
            success_fn=lambda output: "4" in output,
            duration_s=60.0,
            metadata={"synthetic": True, "n_file_openings": 4},
        ))

        return tasks

    # ─────────────────────────────────────────────────────────────
    # Category B: Meeting / Live Audio
    # ─────────────────────────────────────────────────────────────

    def _category_b_tasks(self) -> list[Task]:
        tasks = []

        # B-001 EASY: Extract a spoken URL
        tasks.append(Task(
            task_id="B-001",
            category=TaskCategory.B_MEETING,
            difficulty=TaskDifficulty.EASY,
            instruction=(
                "A recorded meeting is playing. When the speaker mentions a website URL, "
                "open it in the browser."
            ),
            ground_truth="https://example.com/report",
            success_fn=lambda output: "example.com/report" in output.lower(),
            duration_s=20.0,
            metadata={"synthetic": True, "spoken_url": "example.com/report"},
        ))

        # B-002 EASY: Extract a spoken number
        tasks.append(Task(
            task_id="B-002",
            category=TaskCategory.B_MEETING,
            difficulty=TaskDifficulty.EASY,
            instruction=(
                "Listen to the meeting recording. The speaker announces a meeting room number. "
                "Type the room number into the text field."
            ),
            ground_truth="407",
            success_fn=lambda output: "407" in output,
            duration_s=15.0,
            metadata={"synthetic": True, "spoken_number": "407"},
        ))

        # B-003 MEDIUM: Multi-speaker, extract action items
        tasks.append(Task(
            task_id="B-003",
            category=TaskCategory.B_MEETING,
            difficulty=TaskDifficulty.MEDIUM,
            instruction=(
                "Listen to the team meeting. After it ends, list the action items "
                "mentioned by each speaker in the notes field."
            ),
            ground_truth={"alice": "prepare report", "bob": "schedule demo"},
            success_fn=lambda output: "report" in output.lower() and "demo" in output.lower(),
            duration_s=40.0,
            metadata={"synthetic": True, "speakers": ["alice", "bob"]},
        ))

        # B-004 MEDIUM: React to a spoken instruction
        tasks.append(Task(
            task_id="B-004",
            category=TaskCategory.B_MEETING,
            difficulty=TaskDifficulty.MEDIUM,
            instruction=(
                "You are in a meeting. The facilitator will instruct you to type a specific "
                "code word into the chat. Listen and type it when instructed."
            ),
            ground_truth="DELTA-7",
            success_fn=lambda output: "delta-7" in output.lower() or "delta7" in output.lower(),
            duration_s=25.0,
            metadata={"synthetic": True, "spoken_code": "DELTA-7"},
        ))

        # B-005 HARD: Long meeting with multiple instructions
        tasks.append(Task(
            task_id="B-005",
            category=TaskCategory.B_MEETING,
            difficulty=TaskDifficulty.HARD,
            instruction=(
                "Listen to the 90-second product planning meeting. "
                "At the end, create a summary document with: meeting topic, decisions made, "
                "and next steps, each as a separate paragraph."
            ),
            ground_truth={"topic": "Q3 roadmap", "decision": "delay feature X", "next_step": "user research"},
            success_fn=lambda output: all(
                k in output.lower() for k in ["q3", "delay", "research"]
            ),
            duration_s=90.0,
            metadata={"synthetic": True},
        ))

        return tasks

    # ─────────────────────────────────────────────────────────────
    # Category C: Transient UI Events
    # ─────────────────────────────────────────────────────────────

    def _category_c_tasks(self) -> list[Task]:
        tasks = []

        # C-001 EASY: Cookie consent dialog (auto-dismisses after 4 seconds)
        tasks.append(Task(
            task_id="C-001",
            category=TaskCategory.C_TRANSIENT_UI,
            difficulty=TaskDifficulty.EASY,
            instruction=(
                "You are browsing a website. Accept the cookie consent dialog when it appears. "
                "The dialog will auto-dismiss after 4 seconds if not clicked."
            ),
            ground_truth="cookie_accepted",
            success_fn=lambda output: "accept" in output.lower() or "cookie" in output.lower(),
            duration_s=10.0,
            metadata={"synthetic": True, "dismiss_after_s": 4.0, "event_type": "cookie_consent"},
        ))

        # C-002 EASY: Download completion toast
        tasks.append(Task(
            task_id="C-002",
            category=TaskCategory.C_TRANSIENT_UI,
            difficulty=TaskDifficulty.EASY,
            instruction=(
                "A file download is in progress. When the download-complete notification "
                "appears, click 'Open file'. The notification disappears after 3 seconds."
            ),
            ground_truth="file_opened",
            success_fn=lambda output: "open" in output.lower(),
            duration_s=15.0,
            metadata={"synthetic": True, "dismiss_after_s": 3.0, "event_type": "download_toast"},
        ))

        # C-003 MEDIUM: Multiple transient events with distractor
        tasks.append(Task(
            task_id="C-003",
            category=TaskCategory.C_TRANSIENT_UI,
            difficulty=TaskDifficulty.MEDIUM,
            instruction=(
                "Browse this page. There will be two popups: a newsletter signup and an "
                "important system update notification. Dismiss the newsletter and click "
                "'Install now' on the system update. Both auto-dismiss after 5 seconds."
            ),
            ground_truth="system_update_clicked",
            success_fn=lambda output: "install" in output.lower() or "update" in output.lower(),
            duration_s=20.0,
            metadata={"synthetic": True, "events": ["newsletter", "system_update"]},
        ))

        # C-004 MEDIUM: Timed form validation message
        tasks.append(Task(
            task_id="C-004",
            category=TaskCategory.C_TRANSIENT_UI,
            difficulty=TaskDifficulty.MEDIUM,
            instruction=(
                "Submit the form. A validation error message will briefly appear (3 seconds) "
                "identifying which field is incorrect. Fix the field and resubmit."
            ),
            ground_truth="email_field_fixed",
            success_fn=lambda output: "email" in output.lower(),
            duration_s=15.0,
            metadata={"synthetic": True, "error_field": "email", "error_duration_s": 3.0},
        ))

        # C-005 HARD: Rapid sequence of events requiring ordering
        tasks.append(Task(
            task_id="C-005",
            category=TaskCategory.C_TRANSIENT_UI,
            difficulty=TaskDifficulty.HARD,
            instruction=(
                "Watch for three sequential notifications that each contain a code digit "
                "(each visible for 2 seconds, appearing 3 seconds apart). "
                "After all three, type the 3-digit code in the order they appeared."
            ),
            ground_truth="392",
            success_fn=lambda output: "392" in output,
            duration_s=15.0,
            metadata={"synthetic": True, "digits": [3, 9, 2], "visible_s": 2.0, "interval_s": 3.0},
        ))

        return tasks

    # ─────────────────────────────────────────────────────────────
    # Category D: Audio Alerts
    # ─────────────────────────────────────────────────────────────

    def _category_d_tasks(self) -> list[Task]:
        tasks = []

        # D-001 EASY: Single calendar alarm
        tasks.append(Task(
            task_id="D-001",
            category=TaskCategory.D_AUDIO_ALERT,
            difficulty=TaskDifficulty.EASY,
            instruction=(
                "Work on the document. When you hear a calendar alarm, open the calendar "
                "app and note the event title in the text field."
            ),
            ground_truth="Team Standup",
            success_fn=lambda output: "standup" in output.lower() or "team" in output.lower(),
            duration_s=20.0,
            metadata={"synthetic": True, "alert_type": "calendar", "event": "Team Standup"},
        ))

        # D-002 EASY: Notification ding
        tasks.append(Task(
            task_id="D-002",
            category=TaskCategory.D_AUDIO_ALERT,
            difficulty=TaskDifficulty.EASY,
            instruction=(
                "When you hear a notification sound, switch to the messaging app "
                "and read the new message."
            ),
            ground_truth="notification_heard",
            success_fn=lambda output: "message" in output.lower() or "switch" in output.lower(),
            duration_s=15.0,
            metadata={"synthetic": True, "alert_type": "notification_ding"},
        ))

        # D-003 MEDIUM: Distinguish between two alert types
        tasks.append(Task(
            task_id="D-003",
            category=TaskCategory.D_AUDIO_ALERT,
            difficulty=TaskDifficulty.MEDIUM,
            instruction=(
                "Monitor the dashboard. A high-pitched beep indicates a critical error; "
                "a low chime indicates a warning. When you hear either, type 'CRITICAL' "
                "or 'WARNING' in the status field."
            ),
            ground_truth="CRITICAL",
            success_fn=lambda output: "critical" in output.lower(),
            duration_s=25.0,
            metadata={"synthetic": True, "alert_frequency": "high", "expected": "critical"},
        ))

        # D-004 MEDIUM: Timer expiry sound
        tasks.append(Task(
            task_id="D-004",
            category=TaskCategory.D_AUDIO_ALERT,
            difficulty=TaskDifficulty.MEDIUM,
            instruction=(
                "A countdown timer is running. When the timer alarm sounds, "
                "click the 'Record time' button."
            ),
            ground_truth="button_clicked",
            success_fn=lambda output: "record" in output.lower() or "click" in output.lower(),
            duration_s=30.0,
            metadata={"synthetic": True, "alert_type": "timer_alarm"},
        ))

        # D-005 HARD: Multiple alerts requiring different responses
        tasks.append(Task(
            task_id="D-005",
            category=TaskCategory.D_AUDIO_ALERT,
            difficulty=TaskDifficulty.HARD,
            instruction=(
                "Monitor the system. Three different alerts will sound over 60 seconds: "
                "a beep (dismiss), a chime (click approve), and an alarm (click emergency stop). "
                "Respond correctly to each."
            ),
            ground_truth="all_three_correct",
            success_fn=lambda output: all(
                k in output.lower() for k in ["dismiss", "approve", "stop"]
            ),
            duration_s=60.0,
            metadata={"synthetic": True, "n_alerts": 3},
        ))

        return tasks

    # ─────────────────────────────────────────────────────────────
    # Category E: Combined Multimodal
    # ─────────────────────────────────────────────────────────────

    def _category_e_tasks(self) -> list[Task]:
        tasks = []

        # E-001 EASY: Narrated tutorial, follow one step
        tasks.append(Task(
            task_id="E-001",
            category=TaskCategory.E_COMBINED,
            difficulty=TaskDifficulty.EASY,
            instruction=(
                "Watch the narrated tutorial. The presenter will show and verbally describe "
                "how to click a button. Click that same button."
            ),
            ground_truth="submit_button_clicked",
            success_fn=lambda output: "submit" in output.lower() or "click" in output.lower(),
            duration_s=15.0,
            metadata={"synthetic": True, "narration": "click the Submit button"},
        ))

        # E-002 EASY: Audio + visual fact extraction
        tasks.append(Task(
            task_id="E-002",
            category=TaskCategory.E_COMBINED,
            difficulty=TaskDifficulty.EASY,
            instruction=(
                "A presenter shows a slide while speaking. The slide has the company name; "
                "the speaker mentions the founding year. Fill in both fields."
            ),
            ground_truth={"company": "Nexus Labs", "year": "2019"},
            success_fn=lambda output: "nexus" in output.lower() and "2019" in output,
            duration_s=20.0,
            metadata={"synthetic": True},
        ))

        # E-003 MEDIUM: Verbal instruction pointing to visual element
        tasks.append(Task(
            task_id="E-003",
            category=TaskCategory.E_COMBINED,
            difficulty=TaskDifficulty.MEDIUM,
            instruction=(
                "In a screen-sharing session, the presenter will verbally tell you to "
                "click a specific menu item. The screen shows the application. "
                "Follow the verbal instruction."
            ),
            ground_truth="file_export_clicked",
            success_fn=lambda output: "export" in output.lower(),
            duration_s=25.0,
            metadata={"synthetic": True, "verbal_instruction": "click File then Export"},
        ))

        # E-004 MEDIUM: Visual + audio carry different info, both needed
        tasks.append(Task(
            task_id="E-004",
            category=TaskCategory.E_COMBINED,
            difficulty=TaskDifficulty.MEDIUM,
            instruction=(
                "Watch a recorded demo. The screen shows a chart; the narrator explains "
                "what the chart shows and gives a recommendation. "
                "Summarize both the visual data and the recommendation."
            ),
            ground_truth={"visual": "Q4 revenue up 23%", "recommendation": "increase marketing budget"},
            success_fn=lambda output: "23" in output and "marketing" in output.lower(),
            duration_s=30.0,
            metadata={"synthetic": True},
        ))

        # E-005 HARD: Complex multimodal, long task
        tasks.append(Task(
            task_id="E-005",
            category=TaskCategory.E_COMBINED,
            difficulty=TaskDifficulty.HARD,
            instruction=(
                "Watch a 90-second tutorial where the instructor demonstrates a multi-step "
                "data analysis workflow, narrating each step. Reproduce all steps "
                "in the application currently open."
            ),
            ground_truth="workflow_reproduced",
            success_fn=lambda output: len(output) > 50,  # any substantial action taken
            duration_s=90.0,
            metadata={"synthetic": True, "n_steps": 5},
        ))

        return tasks

    def get_by_category(self, category: TaskCategory) -> list[Task]:
        return [t for t in self.tasks if t.category == category]

    def get_by_difficulty(self, difficulty: TaskDifficulty) -> list[Task]:
        return [t for t in self.tasks if t.difficulty == difficulty]

    def summary(self) -> dict:
        return {
            cat.value: len(self.get_by_category(cat))
            for cat in TaskCategory
        }

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)
