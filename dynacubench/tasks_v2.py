"""
DynaCU-Bench v2 — 10 Categories × 10 Tasks = 100 Total

Each category targets a distinct perceptual challenge that
screenshot-only agents cannot solve.

Structure per category: 3 Easy + 4 Medium + 3 Hard = 10 tasks
Total: 100 tasks across 10 categories.

Categories:
  A: Video Comprehension       — extract content from pre-recorded video
  B: Meeting Speech            — multi-speaker audio, structured extraction
  C: Podcast / Long-form Audio — audio-primary, no on-screen transcript
  D: Transient UI Events       — ephemeral visual elements (auto-dismiss)
  E: Non-speech Audio Alerts   — system sounds: beeps, chimes, alarms
  F: Continuous Animation      — UI-generated motion: carousels, charts
  G: Web Games / Interactive   — real-time visual reaction to game state
  H: Sequential State Transitions — ordered visual changes over time
  I: Live Data Streams         — continuously updating data feeds
  J: Combined Multimodal       — tasks requiring multiple categories
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


class TaskCategory(Enum):
    A_VIDEO         = "A_video_comprehension"
    B_MEETING       = "B_meeting_speech"
    C_PODCAST       = "C_podcast_audio"
    D_TRANSIENT_UI  = "D_transient_ui"
    E_AUDIO_ALERT   = "E_audio_alert"
    F_ANIMATION     = "F_continuous_animation"
    G_GAME          = "G_web_game"
    H_SEQUENTIAL    = "H_sequential_transitions"
    I_LIVESTREAM    = "I_live_data_stream"
    J_COMBINED      = "J_combined_multimodal"


class TaskDifficulty(Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


@dataclass
class Task:
    task_id: str
    category: TaskCategory
    difficulty: TaskDifficulty
    instruction: str
    ground_truth: Any
    success_fn: Optional[Callable] = None
    html_file: Optional[str] = None          # Real HTML task file
    stimulus_audio: Optional[Path] = None    # Pre-rendered audio file
    stimulus_video: Optional[Path] = None    # Pre-rendered video file
    duration_s: float = 30.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "instruction": self.instruction,
            "ground_truth": str(self.ground_truth),
            "duration_s": self.duration_s,
            "html_file": self.html_file,
        }


class DynaCUBench:
    """DynaCU-Bench v2: 100 tasks across 10 perceptual categories."""

    def __init__(self, html_tasks_dir: Path = Path("benchmark_env/html_tasks")):
        self.html_tasks_dir = html_tasks_dir
        self.tasks: list[Task] = []
        self._register_all()

    def _register_all(self):
        for cat_fn in [
            self._cat_a, self._cat_b, self._cat_c, self._cat_d, self._cat_e,
            self._cat_f, self._cat_g, self._cat_h, self._cat_i, self._cat_j,
        ]:
            self.tasks.extend(cat_fn())

    # ══════════════════════════════════════════════════════════════
    # A: VIDEO COMPREHENSION (3E + 4M + 3H = 10)
    # ══════════════════════════════════════════════════════════════
    def _cat_a(self) -> list[Task]:
        return [
            # EASY ──────────────────────────────────────────────
            Task("A-E1", TaskCategory.A_VIDEO, TaskDifficulty.EASY,
                 "A product demo video plays. After it ends, type the product name shown.",
                 "CloudSync Pro",
                 success_fn=lambda o: "cloudsync" in o.lower(),
                 html_file="A_video_product_demo.html", duration_s=15.0),
            Task("A-E2", TaskCategory.A_VIDEO, TaskDifficulty.EASY,
                 "A slide deck plays. Count the total number of slides and enter the count.",
                 "5",
                 success_fn=lambda o: "5" in o,
                 html_file="A_video_slide_count.html", duration_s=25.0),
            Task("A-E3", TaskCategory.A_VIDEO, TaskDifficulty.EASY,
                 "Watch the video. What color is the main logo shown? Type the color.",
                 "blue",
                 success_fn=lambda o: "blue" in o.lower(),
                 html_file="A_video_logo_color.html", duration_s=12.0),
            # MEDIUM ─────────────────────────────────────────────
            Task("A-M1", TaskCategory.A_VIDEO, TaskDifficulty.MEDIUM,
                 "Watch the tutorial and fill in: product name, price, and release date as shown.",
                 {"name": "DataVault 3.0", "price": "$299", "date": "March 2026"},
                 success_fn=lambda o: "datavault" in o.lower() and "299" in o,
                 html_file="A_video_multi_fact.html", duration_s=30.0),
            Task("A-M2", TaskCategory.A_VIDEO, TaskDifficulty.MEDIUM,
                 "Watch the demo. It shows a 3-step installation process. Summarize the three steps.",
                 "download, install, configure",
                 success_fn=lambda o: sum(w in o.lower() for w in ["download","install","config"]) >= 2,
                 html_file="A_video_steps.html", duration_s=35.0),
            Task("A-M3", TaskCategory.A_VIDEO, TaskDifficulty.MEDIUM,
                 "A coding tutorial plays. After it ends, write the function name demonstrated.",
                 "process_data",
                 success_fn=lambda o: "process" in o.lower() and "data" in o.lower(),
                 html_file="A_video_code_tutorial.html", duration_s=40.0),
            Task("A-M4", TaskCategory.A_VIDEO, TaskDifficulty.MEDIUM,
                 "Watch the animated explainer. Which two countries are highlighted on the map?",
                 "Germany and France",
                 success_fn=lambda o: "germany" in o.lower() or "france" in o.lower(),
                 html_file="A_video_map_animation.html", duration_s=30.0),
            # HARD ───────────────────────────────────────────────
            Task("A-H1", TaskCategory.A_VIDEO, TaskDifficulty.HARD,
                 "Watch the recorded presentation. Three quarterly revenue figures appear across slides. Calculate the total.",
                 "$2.4M",
                 success_fn=lambda o: "2.4" in o or "2400" in o,
                 html_file="A_video_revenue_calc.html", duration_s=45.0),
            Task("A-H2", TaskCategory.A_VIDEO, TaskDifficulty.HARD,
                 "A 60-second screen recording shows a developer navigating code. How many files were opened? Type the number.",
                 "4",
                 success_fn=lambda o: "4" in o,
                 html_file="A_video_file_count.html", duration_s=65.0),
            Task("A-H3", TaskCategory.A_VIDEO, TaskDifficulty.HARD,
                 "Watch the video demo end-to-end. Then replicate the 5-step workflow shown in the application open on screen.",
                 "workflow_reproduced",
                 success_fn=lambda o: len(o) > 30,
                 html_file="A_video_workflow.html", duration_s=90.0),
        ]

    # ══════════════════════════════════════════════════════════════
    # B: MEETING SPEECH (3E + 4M + 3H = 10)
    # ══════════════════════════════════════════════════════════════
    def _cat_b(self) -> list[Task]:
        return [
            Task("B-E1", TaskCategory.B_MEETING, TaskDifficulty.EASY,
                 "A recorded meeting plays. When the speaker mentions a URL, open it in the browser.",
                 "example.com/report",
                 success_fn=lambda o: "example.com" in o.lower(),
                 html_file="B001_meeting_audio.html", duration_s=20.0),
            Task("B-E2", TaskCategory.B_MEETING, TaskDifficulty.EASY,
                 "Listen to the meeting. The host announces a room number. Type it in the field.",
                 "407",
                 success_fn=lambda o: "407" in o,
                 html_file="B_meeting_room_number.html", duration_s=15.0),
            Task("B-E3", TaskCategory.B_MEETING, TaskDifficulty.EASY,
                 "A meeting recording plays. Someone mentions a date. Enter it in the date field.",
                 "March 15",
                 success_fn=lambda o: "march" in o.lower() or "15" in o,
                 html_file="B_meeting_date.html", duration_s=18.0),
            # MEDIUM
            Task("B-M1", TaskCategory.B_MEETING, TaskDifficulty.MEDIUM,
                 "Listen to the team meeting. List the action items each speaker mentions.",
                 {"alice": "prepare report", "bob": "schedule demo"},
                 success_fn=lambda o: "report" in o.lower() and "demo" in o.lower(),
                 html_file="B_meeting_action_items.html", duration_s=40.0),
            Task("B-M2", TaskCategory.B_MEETING, TaskDifficulty.MEDIUM,
                 "The facilitator will say a code word. Type it when instructed.",
                 "DELTA-7",
                 success_fn=lambda o: "delta" in o.lower(),
                 html_file="B_meeting_codeword.html", duration_s=25.0),
            Task("B-M3", TaskCategory.B_MEETING, TaskDifficulty.MEDIUM,
                 "Listen to the product review. Which three features received positive feedback?",
                 "speed, reliability, design",
                 success_fn=lambda o: sum(w in o.lower() for w in ["speed","reliab","design"]) >= 2,
                 html_file="B_meeting_product_review.html", duration_s=45.0),
            Task("B-M4", TaskCategory.B_MEETING, TaskDifficulty.MEDIUM,
                 "Two speakers debate a decision. Which option did they ultimately agree on?",
                 "Option B",
                 success_fn=lambda o: "option b" in o.lower() or "option-b" in o.lower(),
                 html_file="B_meeting_decision.html", duration_s=50.0),
            # HARD
            Task("B-H1", TaskCategory.B_MEETING, TaskDifficulty.HARD,
                 "Listen to the 90-second planning meeting. Create a summary with topic, decisions, and next steps.",
                 {"topic": "Q3 roadmap", "decision": "delay feature X", "next": "user research"},
                 success_fn=lambda o: all(k in o.lower() for k in ["q3", "delay", "research"]),
                 html_file="B_meeting_long_summary.html", duration_s=95.0),
            Task("B-H2", TaskCategory.B_MEETING, TaskDifficulty.HARD,
                 "Three speakers discuss a budget proposal with four line items. Enter each item and its approved amount.",
                 "4 line items",
                 success_fn=lambda o: len(o) > 40,
                 html_file="B_meeting_budget.html", duration_s=80.0),
            Task("B-H3", TaskCategory.B_MEETING, TaskDifficulty.HARD,
                 "A technical call covers three bug reports. For each, note the bug ID and severity (P1/P2/P3).",
                 "3 bug reports",
                 success_fn=lambda o: len(o) > 30 and ("p1" in o.lower() or "p2" in o.lower()),
                 html_file="B_meeting_bugs.html", duration_s=70.0),
        ]

    # ══════════════════════════════════════════════════════════════
    # C: PODCAST / LONG-FORM AUDIO (3E + 4M + 3H = 10)
    # Audio-primary: no transcript shown on screen, agent must listen
    # ══════════════════════════════════════════════════════════════
    def _cat_c(self) -> list[Task]:
        return [
            Task("C-E1", TaskCategory.C_PODCAST, TaskDifficulty.EASY,
                 "A podcast clip plays (no transcript). What is the topic of the episode? Type it.",
                 "artificial intelligence",
                 success_fn=lambda o: "ai" in o.lower() or "artificial" in o.lower() or "intelligence" in o.lower(),
                 html_file="C_podcast_topic.html", duration_s=20.0),
            Task("C-E2", TaskCategory.C_PODCAST, TaskDifficulty.EASY,
                 "Listen to the audio clip. The speaker names one specific tool. Type the tool name.",
                 "PyTorch",
                 success_fn=lambda o: "pytorch" in o.lower() or "torch" in o.lower(),
                 html_file="C_podcast_tool.html", duration_s=18.0),
            Task("C-E3", TaskCategory.C_PODCAST, TaskDifficulty.EASY,
                 "An audio instruction clip plays. Follow the instruction (type the word you hear).",
                 "serendipity",
                 success_fn=lambda o: "serendipity" in o.lower(),
                 html_file="C_podcast_instruction.html", duration_s=15.0),
            # MEDIUM
            Task("C-M1", TaskCategory.C_PODCAST, TaskDifficulty.MEDIUM,
                 "Listen to the 60-second podcast segment. Summarize the two main arguments made.",
                 "2 arguments",
                 success_fn=lambda o: len(o) > 40,
                 html_file="C_podcast_arguments.html", duration_s=65.0),
            Task("C-M2", TaskCategory.C_PODCAST, TaskDifficulty.MEDIUM,
                 "A voice memo plays. Extract and fill in: sender name, subject, and requested deadline.",
                 "Alice, budget review, Friday",
                 success_fn=lambda o: "alice" in o.lower() and ("friday" in o.lower() or "budget" in o.lower()),
                 html_file="C_podcast_voicememo.html", duration_s=30.0),
            Task("C-M3", TaskCategory.C_PODCAST, TaskDifficulty.MEDIUM,
                 "An audio interview plays. The guest mentions 3 companies. List all three.",
                 "Google, Apple, Microsoft",
                 success_fn=lambda o: sum(c in o for c in ["Google","Apple","Microsoft"]) >= 2,
                 html_file="C_podcast_companies.html", duration_s=45.0),
            Task("C-M4", TaskCategory.C_PODCAST, TaskDifficulty.MEDIUM,
                 "Listen to the product announcement audio. What is the launch date and starting price?",
                 {"date": "Q2 2026", "price": "$49"},
                 success_fn=lambda o: "49" in o and "2026" in o,
                 html_file="C_podcast_announcement.html", duration_s=35.0),
            # HARD
            Task("C-H1", TaskCategory.C_PODCAST, TaskDifficulty.HARD,
                 "Listen to the 2-minute technical podcast. List the 4 technical concepts explained, in order.",
                 "4 concepts in order",
                 success_fn=lambda o: len(o) > 50,
                 html_file="C_podcast_technical_long.html", duration_s=125.0),
            Task("C-H2", TaskCategory.C_PODCAST, TaskDifficulty.HARD,
                 "A multi-speaker debate audio plays. Score each speaker's argument 1-5 and explain why.",
                 "scored debate",
                 success_fn=lambda o: any(str(i) in o for i in range(1, 6)),
                 html_file="C_podcast_debate.html", duration_s=90.0),
            Task("C-H3", TaskCategory.C_PODCAST, TaskDifficulty.HARD,
                 "Earnings call audio plays. Extract: revenue, gross margin, guidance, and one key risk mentioned.",
                 "4 financial metrics",
                 success_fn=lambda o: len(o) > 60 and ("%" in o or "$" in o),
                 html_file="C_podcast_earnings.html", duration_s=120.0),
        ]

    # ══════════════════════════════════════════════════════════════
    # D: TRANSIENT UI EVENTS (3E + 4M + 3H = 10)
    # Ephemeral elements: appear briefly, auto-dismiss
    # ══════════════════════════════════════════════════════════════
    def _cat_d(self) -> list[Task]:
        return [
            Task("D-E1", TaskCategory.D_TRANSIENT_UI, TaskDifficulty.EASY,
                 "Browse the website. Accept the cookie consent dialog when it appears. It auto-dismisses after 4s.",
                 "accepted",
                 success_fn=lambda o: "accept" in o.lower() or "cookie" in o.lower(),
                 html_file="C001_cookie_consent.html", duration_s=15.0),
            Task("D-E2", TaskCategory.D_TRANSIENT_UI, TaskDifficulty.EASY,
                 "A file is downloading. When the download-complete toast appears, click 'Open file'. Disappears after 3s.",
                 "file_opened",
                 success_fn=lambda o: "open" in o.lower(),
                 html_file="C002_download_toast.html", duration_s=15.0),
            Task("D-E3", TaskCategory.D_TRANSIENT_UI, TaskDifficulty.EASY,
                 "A 'session expiring' warning will pop up. Click 'Stay logged in' before it auto-dismisses (4s).",
                 "session_extended",
                 success_fn=lambda o: "stay" in o.lower() or "session" in o.lower(),
                 html_file="D_transient_session_warning.html", duration_s=10.0),
            # MEDIUM
            Task("D-M1", TaskCategory.D_TRANSIENT_UI, TaskDifficulty.MEDIUM,
                 "Two popups will appear: newsletter signup and system update. Dismiss newsletter, click 'Install' on update. Both disappear after 5s.",
                 "system_update_clicked",
                 success_fn=lambda o: "install" in o.lower() or "update" in o.lower(),
                 html_file="D_transient_two_popups.html", duration_s=20.0),
            Task("D-M2", TaskCategory.D_TRANSIENT_UI, TaskDifficulty.MEDIUM,
                 "Submit the form. A validation error briefly appears (3s) identifying the wrong field. Fix it and resubmit.",
                 "email_field_fixed",
                 success_fn=lambda o: "email" in o.lower() or "fix" in o.lower(),
                 html_file="D_transient_validation_error.html", duration_s=15.0),
            Task("D-M3", TaskCategory.D_TRANSIENT_UI, TaskDifficulty.MEDIUM,
                 "A progress banner updates 3 times (each visible 2s) showing step completions. Note the final step name.",
                 "database migration",
                 success_fn=lambda o: "database" in o.lower() or "migration" in o.lower(),
                 html_file="D_transient_progress_banner.html", duration_s=20.0),
            Task("D-M4", TaskCategory.D_TRANSIENT_UI, TaskDifficulty.MEDIUM,
                 "A flash sale countdown appears briefly (3s) showing a discount code. Enter the code in the checkout field.",
                 "FLASH25",
                 success_fn=lambda o: "flash25" in o.lower() or "flash" in o.lower(),
                 html_file="D_transient_flash_sale.html", duration_s=15.0),
            # HARD
            Task("D-H1", TaskCategory.D_TRANSIENT_UI, TaskDifficulty.HARD,
                 "Three notifications appear sequentially, each showing a digit (2s visible, 3s apart). Type the 3-digit code in order.",
                 "392",
                 success_fn=lambda o: "392" in o,
                 html_file="D_transient_digit_sequence.html", duration_s=15.0),
            Task("D-H2", TaskCategory.D_TRANSIENT_UI, TaskDifficulty.HARD,
                 "Five transient alerts appear over 30s; three are 'info' (ignore) and two are 'critical' (acknowledge). Count and click only the critical ones.",
                 "2_critical_acknowledged",
                 success_fn=lambda o: "2" in o or "critical" in o.lower(),
                 html_file="D_transient_filter_alerts.html", duration_s=35.0),
            Task("D-H3", TaskCategory.D_TRANSIENT_UI, TaskDifficulty.HARD,
                 "A multi-step onboarding checklist appears (5 items, each visible 2s). Check each item off before it disappears.",
                 "all_5_checked",
                 success_fn=lambda o: "5" in o or "all" in o.lower() or "check" in o.lower(),
                 html_file="D_transient_checklist.html", duration_s=20.0),
        ]

    # ══════════════════════════════════════════════════════════════
    # E: NON-SPEECH AUDIO ALERTS (3E + 4M + 3H = 10)
    # ══════════════════════════════════════════════════════════════
    def _cat_e(self) -> list[Task]:
        return [
            Task("E-E1", TaskCategory.E_AUDIO_ALERT, TaskDifficulty.EASY,
                 "Work on the document. When you hear a calendar alarm, open the calendar and note the event title.",
                 "Team Standup",
                 success_fn=lambda o: "standup" in o.lower() or "stand" in o.lower(),
                 html_file="D001_audio_alert.html", duration_s=20.0),
            Task("E-E2", TaskCategory.E_AUDIO_ALERT, TaskDifficulty.EASY,
                 "When you hear a notification ding, switch to the messaging app.",
                 "notification_heard",
                 success_fn=lambda o: "message" in o.lower() or "switch" in o.lower(),
                 html_file="E_alert_notification_ding.html", duration_s=15.0),
            Task("E-E3", TaskCategory.E_AUDIO_ALERT, TaskDifficulty.EASY,
                 "A download completes with a chime. Click 'Open Downloads' when you hear it.",
                 "downloads_opened",
                 success_fn=lambda o: "download" in o.lower() or "open" in o.lower(),
                 html_file="E_alert_download_chime.html", duration_s=15.0),
            # MEDIUM
            Task("E-M1", TaskCategory.E_AUDIO_ALERT, TaskDifficulty.MEDIUM,
                 "Monitor the dashboard. A high-pitched beep = critical error; low chime = warning. Classify and type 'CRITICAL' or 'WARNING'.",
                 "CRITICAL",
                 success_fn=lambda o: "critical" in o.lower(),
                 html_file="E_alert_classify_sounds.html", duration_s=25.0),
            Task("E-M2", TaskCategory.E_AUDIO_ALERT, TaskDifficulty.MEDIUM,
                 "A countdown timer is running. Click 'Record time' immediately when the timer alarm sounds.",
                 "button_clicked",
                 success_fn=lambda o: "record" in o.lower() or "click" in o.lower(),
                 html_file="E_alert_timer.html", duration_s=30.0),
            Task("E-M3", TaskCategory.E_AUDIO_ALERT, TaskDifficulty.MEDIUM,
                 "Three emails arrive (each with a different ding pitch: high/medium/low). Note which arrived first (high/medium/low).",
                 "high",
                 success_fn=lambda o: "high" in o.lower(),
                 html_file="E_alert_pitch_ordering.html", duration_s=25.0),
            Task("E-M4", TaskCategory.E_AUDIO_ALERT, TaskDifficulty.MEDIUM,
                 "A build process plays sounds for pass (ding) and fail (buzz). How many tests passed? Count the dings.",
                 "3",
                 success_fn=lambda o: "3" in o,
                 html_file="E_alert_count_dings.html", duration_s=20.0),
            # HARD
            Task("E-H1", TaskCategory.E_AUDIO_ALERT, TaskDifficulty.HARD,
                 "Three alerts sound over 60s: beep (dismiss), chime (approve), alarm (emergency stop). Respond correctly to each.",
                 "all_three_correct",
                 success_fn=lambda o: all(k in o.lower() for k in ["dismiss", "approv", "stop"]),
                 html_file="E_alert_three_responses.html", duration_s=65.0),
            Task("E-H2", TaskCategory.E_AUDIO_ALERT, TaskDifficulty.HARD,
                 "A Morse-code-like beep sequence encodes a letter. Decode it and type the letter.",
                 "S",
                 success_fn=lambda o: "s" in o.lower(),
                 html_file="E_alert_morse.html", duration_s=30.0),
            Task("E-H3", TaskCategory.E_AUDIO_ALERT, TaskDifficulty.HARD,
                 "Five machines report status via beeps (1 beep=ok, 2=warning, 3=fail). Report status of each machine.",
                 "5_machines_reported",
                 success_fn=lambda o: len(o) > 30,
                 html_file="E_alert_multi_machine.html", duration_s=40.0),
        ]

    # ══════════════════════════════════════════════════════════════
    # F: CONTINUOUS ANIMATION (3E + 4M + 3H = 10)
    # UI-generated motion: carousels, animated charts, loading states
    # ══════════════════════════════════════════════════════════════
    def _cat_f(self) -> list[Task]:
        return [
            Task("F-E1", TaskCategory.F_ANIMATION, TaskDifficulty.EASY,
                 "An image carousel rotates. When the 'Winter Sale' slide appears, click 'Shop Now'.",
                 "shop_now_clicked",
                 success_fn=lambda o: "shop" in o.lower() or "winter" in o.lower(),
                 html_file="F_anim_carousel.html", duration_s=20.0),
            Task("F-E2", TaskCategory.F_ANIMATION, TaskDifficulty.EASY,
                 "A loading progress bar fills. When it reaches 100%, click 'Launch'.",
                 "launch_clicked",
                 success_fn=lambda o: "launch" in o.lower() or "100" in o,
                 html_file="F_anim_progress_bar.html", duration_s=15.0),
            Task("F-E3", TaskCategory.F_ANIMATION, TaskDifficulty.EASY,
                 "An animated pie chart cycles through 4 segments. What is the largest segment label?",
                 "Revenue",
                 success_fn=lambda o: "revenue" in o.lower(),
                 html_file="F_anim_pie_chart.html", duration_s=18.0),
            # MEDIUM
            Task("F-M1", TaskCategory.F_ANIMATION, TaskDifficulty.MEDIUM,
                 "A photo gallery auto-advances every 3 seconds. Note the caption of the 3rd image shown.",
                 "Mountain Summit 2024",
                 success_fn=lambda o: "mountain" in o.lower() or "summit" in o.lower(),
                 html_file="F_anim_gallery.html", duration_s=15.0),
            Task("F-M2", TaskCategory.F_ANIMATION, TaskDifficulty.MEDIUM,
                 "An animated bar chart updates every 2 seconds. Which category reaches the highest value?",
                 "Technology",
                 success_fn=lambda o: "tech" in o.lower(),
                 html_file="F_anim_bar_chart.html", duration_s=20.0),
            Task("F-M3", TaskCategory.F_ANIMATION, TaskDifficulty.MEDIUM,
                 "A CSS animation shows 5 steps of a workflow. Click 'Confirm' when Step 3 is highlighted.",
                 "step3_confirmed",
                 success_fn=lambda o: "confirm" in o.lower() or "step 3" in o.lower(),
                 html_file="F_anim_workflow_steps.html", duration_s=25.0),
            Task("F-M4", TaskCategory.F_ANIMATION, TaskDifficulty.MEDIUM,
                 "A live ticker animates stock prices. When AAPL price drops below $180, click 'Alert me'.",
                 "alert_set",
                 success_fn=lambda o: "alert" in o.lower() or "aapl" in o.lower(),
                 html_file="F_anim_stock_ticker.html", duration_s=30.0),
            # HARD
            Task("F-H1", TaskCategory.F_ANIMATION, TaskDifficulty.HARD,
                 "A circular progress gauge animates from 0 to 100%. Click 'Save checkpoint' at exactly 75%.",
                 "checkpoint_at_75",
                 success_fn=lambda o: "checkpoint" in o.lower() or "75" in o,
                 html_file="F_anim_gauge.html", duration_s=20.0),
            Task("F-H2", TaskCategory.F_ANIMATION, TaskDifficulty.HARD,
                 "5 items in a kanban board animate between states (To Do → In Progress → Done). Report final state of each.",
                 "5_states_reported",
                 success_fn=lambda o: len(o) > 40,
                 html_file="F_anim_kanban.html", duration_s=35.0),
            Task("F-H3", TaskCategory.F_ANIMATION, TaskDifficulty.HARD,
                 "An animated network graph shows nodes connecting and disconnecting. After 30s, how many nodes are connected?",
                 "7",
                 success_fn=lambda o: "7" in o,
                 html_file="F_anim_network_graph.html", duration_s=35.0),
        ]

    # ══════════════════════════════════════════════════════════════
    # G: WEB GAMES / INTERACTIVE (3E + 4M + 3H = 10)
    # Real-time visual reaction to game state
    # ══════════════════════════════════════════════════════════════
    def _cat_g(self) -> list[Task]:
        return [
            Task("G-E1", TaskCategory.G_GAME, TaskDifficulty.EASY,
                 "A memory card game: find and click the matching pair of cards shown briefly at the start.",
                 "pair_matched",
                 success_fn=lambda o: "match" in o.lower() or "pair" in o.lower(),
                 html_file="G_game_memory_cards.html", duration_s=20.0),
            Task("G-E2", TaskCategory.G_GAME, TaskDifficulty.EASY,
                 "A number guessing game: guess the number the AI is thinking of (you get 3 tries, shown on screen).",
                 "number_guessed",
                 success_fn=lambda o: "correct" in o.lower() or "guess" in o.lower(),
                 html_file="G_game_number_guess.html", duration_s=25.0),
            Task("G-E3", TaskCategory.G_GAME, TaskDifficulty.EASY,
                 "A simple reaction test: click the green button as fast as possible when it appears.",
                 "reaction_recorded",
                 success_fn=lambda o: "click" in o.lower() or "ms" in o.lower(),
                 html_file="G_game_reaction_test.html", duration_s=15.0),
            # MEDIUM
            Task("G-M1", TaskCategory.G_GAME, TaskDifficulty.MEDIUM,
                 "A sliding puzzle: arrange 8 tiles in order (1-8). Solve it and click 'Complete'.",
                 "puzzle_solved",
                 success_fn=lambda o: "complet" in o.lower() or "solv" in o.lower(),
                 html_file="G_game_sliding_puzzle.html", duration_s=60.0),
            Task("G-M2", TaskCategory.G_GAME, TaskDifficulty.MEDIUM,
                 "A color-matching game: click tiles to match the shown pattern within 10 seconds.",
                 "pattern_matched",
                 success_fn=lambda o: "match" in o.lower() or "pattern" in o.lower(),
                 html_file="G_game_color_match.html", duration_s=20.0),
            Task("G-M3", TaskCategory.G_GAME, TaskDifficulty.MEDIUM,
                 "A word scramble: unscramble the letters shown and type the word.",
                 "PYTHON",
                 success_fn=lambda o: "python" in o.lower(),
                 html_file="G_game_word_scramble.html", duration_s=20.0),
            Task("G-M4", TaskCategory.G_GAME, TaskDifficulty.MEDIUM,
                 "A simple platformer game: navigate the character to the flag without falling (keyboard controls shown).",
                 "flag_reached",
                 success_fn=lambda o: "flag" in o.lower() or "reached" in o.lower() or "win" in o.lower(),
                 html_file="G_game_platformer.html", duration_s=60.0),
            # HARD
            Task("G-H1", TaskCategory.G_GAME, TaskDifficulty.HARD,
                 "A fast-paced whack-a-mole game: click moles as they appear for 20 seconds. Score above 8 to pass.",
                 "score_above_8",
                 success_fn=lambda o: any(str(n) in o for n in range(9, 20)),
                 html_file="G_game_whack_a_mole.html", duration_s=30.0),
            Task("G-H2", TaskCategory.G_GAME, TaskDifficulty.HARD,
                 "A logic puzzle game: solve 3 constraint-satisfaction puzzles (each shown sequentially).",
                 "3_puzzles_solved",
                 success_fn=lambda o: "3" in o or "all" in o.lower(),
                 html_file="G_game_logic_puzzles.html", duration_s=90.0),
            Task("G-H3", TaskCategory.G_GAME, TaskDifficulty.HARD,
                 "A Simon Says sequence game: watch the flashing color sequence and repeat it. Sequence length increases each round. Reach round 5.",
                 "round_5",
                 success_fn=lambda o: "5" in o or "round" in o.lower(),
                 html_file="G_game_simon_says.html", duration_s=60.0),
        ]

    # ══════════════════════════════════════════════════════════════
    # H: SEQUENTIAL STATE TRANSITIONS (3E + 4M + 3H = 10)
    # Understanding ORDER of visual changes across time
    # ══════════════════════════════════════════════════════════════
    def _cat_h(self) -> list[Task]:
        return [
            Task("H-E1", TaskCategory.H_SEQUENTIAL, TaskDifficulty.EASY,
                 "A 3-step wizard runs automatically. After it completes, type the name of Step 2.",
                 "Configuration",
                 success_fn=lambda o: "config" in o.lower(),
                 html_file="H_seq_wizard.html", duration_s=20.0),
            Task("H-E2", TaskCategory.H_SEQUENTIAL, TaskDifficulty.EASY,
                 "An animated tutorial shows 4 steps. What action is performed in step 3?",
                 "Save file",
                 success_fn=lambda o: "save" in o.lower() or "file" in o.lower(),
                 html_file="H_seq_tutorial_steps.html", duration_s=20.0),
            Task("H-E3", TaskCategory.H_SEQUENTIAL, TaskDifficulty.EASY,
                 "A checklist processes items one by one. Which item completes last?",
                 "Deploy",
                 success_fn=lambda o: "deploy" in o.lower(),
                 html_file="H_seq_checklist_order.html", duration_s=18.0),
            # MEDIUM
            Task("H-M1", TaskCategory.H_SEQUENTIAL, TaskDifficulty.MEDIUM,
                 "An animated diff viewer shows 3 code changes applied in sequence. Describe the change that happens second.",
                 "variable renamed",
                 success_fn=lambda o: "rename" in o.lower() or "variable" in o.lower(),
                 html_file="H_seq_code_diff.html", duration_s=30.0),
            Task("H-M2", TaskCategory.H_SEQUENTIAL, TaskDifficulty.MEDIUM,
                 "A deployment pipeline runs: 5 stages animate from pending → running → pass/fail. Report the final status of each.",
                 "5_stages",
                 success_fn=lambda o: len(o) > 40,
                 html_file="H_seq_pipeline.html", duration_s=35.0),
            Task("H-M3", TaskCategory.H_SEQUENTIAL, TaskDifficulty.MEDIUM,
                 "An animated flowchart walks through a decision tree (3 branches shown). Which path was taken?",
                 "Path B",
                 success_fn=lambda o: "path b" in o.lower() or "branch b" in o.lower(),
                 html_file="H_seq_decision_tree.html", duration_s=25.0),
            Task("H-M4", TaskCategory.H_SEQUENTIAL, TaskDifficulty.MEDIUM,
                 "A Gantt chart animates showing task start/end times. Which task has the longest duration?",
                 "Task C",
                 success_fn=lambda o: "task c" in o.lower() or "c" in o.lower(),
                 html_file="H_seq_gantt.html", duration_s=20.0),
            # HARD
            Task("H-H1", TaskCategory.H_SEQUENTIAL, TaskDifficulty.HARD,
                 "A state machine animates through 8 states with transitions. Draw/describe the complete transition sequence.",
                 "8_states",
                 success_fn=lambda o: len(o) > 50,
                 html_file="H_seq_state_machine.html", duration_s=40.0),
            Task("H-H2", TaskCategory.H_SEQUENTIAL, TaskDifficulty.HARD,
                 "A git graph animates commits, branches, and merges over 30s. Describe the final branch structure.",
                 "main and feature branch",
                 success_fn=lambda o: "branch" in o.lower() or "merge" in o.lower(),
                 html_file="H_seq_git_graph.html", duration_s=35.0),
            Task("H-H3", TaskCategory.H_SEQUENTIAL, TaskDifficulty.HARD,
                 "An event log replays a security incident in real-time (7 events). Identify the moment of compromise.",
                 "event_4",
                 success_fn=lambda o: "4" in o or "event" in o.lower(),
                 html_file="H_seq_security_log.html", duration_s=50.0),
        ]

    # ══════════════════════════════════════════════════════════════
    # I: LIVE DATA STREAMS (3E + 4M + 3H = 10)
    # Continuously updating data: tickers, dashboards, counters
    # ══════════════════════════════════════════════════════════════
    def _cat_i(self) -> list[Task]:
        return [
            Task("I-E1", TaskCategory.I_LIVESTREAM, TaskDifficulty.EASY,
                 "A live counter counts up. Click 'Capture' when it reaches exactly 42.",
                 "42_captured",
                 success_fn=lambda o: "42" in o or "capture" in o.lower(),
                 html_file="I_stream_counter.html", duration_s=15.0),
            Task("I-E2", TaskCategory.I_LIVESTREAM, TaskDifficulty.EASY,
                 "A live temperature feed updates every 2s. When it exceeds 37°C, click 'Alert'.",
                 "alert_triggered",
                 success_fn=lambda o: "alert" in o.lower() or "37" in o,
                 html_file="I_stream_temperature.html", duration_s=20.0),
            Task("I-E3", TaskCategory.I_LIVESTREAM, TaskDifficulty.EASY,
                 "A news ticker scrolls. Identify and type the headline that mentions 'breakthrough'.",
                 "AI breakthrough",
                 success_fn=lambda o: "breakthrough" in o.lower() or "ai" in o.lower(),
                 html_file="I_stream_news_ticker.html", duration_s=25.0),
            # MEDIUM
            Task("I-M1", TaskCategory.I_LIVESTREAM, TaskDifficulty.MEDIUM,
                 "A live stock dashboard updates every 3s. When any stock drops more than 5%, note its ticker.",
                 "NVDA",
                 success_fn=lambda o: "nvda" in o.upper() or "nvidia" in o.lower(),
                 html_file="I_stream_stocks.html", duration_s=30.0),
            Task("I-M2", TaskCategory.I_LIVESTREAM, TaskDifficulty.MEDIUM,
                 "A real-time server metrics dashboard runs. Report: peak CPU%, peak memory%, and which service caused it.",
                 "metrics_reported",
                 success_fn=lambda o: "%" in o and len(o) > 20,
                 html_file="I_stream_server_metrics.html", duration_s=35.0),
            Task("I-M3", TaskCategory.I_LIVESTREAM, TaskDifficulty.MEDIUM,
                 "A live sales leaderboard updates every 2s. After 20 seconds, which salesperson is in first place?",
                 "Sarah",
                 success_fn=lambda o: "sarah" in o.lower(),
                 html_file="I_stream_leaderboard.html", duration_s=25.0),
            Task("I-M4", TaskCategory.I_LIVESTREAM, TaskDifficulty.MEDIUM,
                 "A live log stream shows application events. When an ERROR level event appears, copy its message.",
                 "database connection failed",
                 success_fn=lambda o: "database" in o.lower() or "error" in o.lower(),
                 html_file="I_stream_log_viewer.html", duration_s=25.0),
            # HARD
            Task("I-H1", TaskCategory.I_LIVESTREAM, TaskDifficulty.HARD,
                 "A live A/B test dashboard runs for 30s. When statistical significance is reached, record which variant won.",
                 "variant_b_won",
                 success_fn=lambda o: "b" in o.lower() or "variant" in o.lower(),
                 html_file="I_stream_ab_test.html", duration_s=35.0),
            Task("I-H2", TaskCategory.I_LIVESTREAM, TaskDifficulty.HARD,
                 "A real-time auction page: 5 bids placed in 30s. Track the highest bidder at each bid.",
                 "5_bids_tracked",
                 success_fn=lambda o: len(o) > 40,
                 html_file="I_stream_auction.html", duration_s=35.0),
            Task("I-H3", TaskCategory.I_LIVESTREAM, TaskDifficulty.HARD,
                 "A live network topology map updates every 5s as nodes go up/down. Report final status of all 6 nodes.",
                 "6_nodes",
                 success_fn=lambda o: len(o) > 40,
                 html_file="I_stream_network_map.html", duration_s=45.0),
        ]

    # ══════════════════════════════════════════════════════════════
    # J: COMBINED MULTIMODAL (3E + 4M + 3H = 10)
    # Requires ≥2 perception channels simultaneously
    # ══════════════════════════════════════════════════════════════
    def _cat_j(self) -> list[Task]:
        return [
            Task("J-E1", TaskCategory.J_COMBINED, TaskDifficulty.EASY,
                 "A narrated video: presenter shows and verbally names a button. Click that button.",
                 "submit_clicked",
                 success_fn=lambda o: "submit" in o.lower() or "click" in o.lower(),
                 html_file="J_comb_narrated_click.html", duration_s=15.0),
            Task("J-E2", TaskCategory.J_COMBINED, TaskDifficulty.EASY,
                 "Slide shows company name; speaker mentions founding year. Fill both fields.",
                 {"company": "Nexus Labs", "year": "2019"},
                 success_fn=lambda o: "nexus" in o.lower() and "2019" in o,
                 html_file="J_comb_slide_audio.html", duration_s=20.0),
            Task("J-E3", TaskCategory.J_COMBINED, TaskDifficulty.EASY,
                 "A podcast plays while an animated chart updates. Report both the chart's peak value and the topic discussed.",
                 "combined",
                 success_fn=lambda o: len(o) > 20,
                 html_file="J_comb_podcast_chart.html", duration_s=30.0),
            # MEDIUM
            Task("J-M1", TaskCategory.J_COMBINED, TaskDifficulty.MEDIUM,
                 "Screen-share session: presenter verbally instructs you to click a specific menu item. Follow it.",
                 "file_export_clicked",
                 success_fn=lambda o: "export" in o.lower() or "file" in o.lower(),
                 html_file="J_comb_verbal_instruction.html", duration_s=25.0),
            Task("J-M2", TaskCategory.J_COMBINED, TaskDifficulty.MEDIUM,
                 "A transient popup appears AND an alarm sounds simultaneously. Handle both: dismiss popup, click Stop alarm.",
                 "both_handled",
                 success_fn=lambda o: "dismiss" in o.lower() or "stop" in o.lower(),
                 html_file="J_comb_popup_and_alarm.html", duration_s=20.0),
            Task("J-M3", TaskCategory.J_COMBINED, TaskDifficulty.MEDIUM,
                 "An animated chart updates while a speaker narrates it. Report the peak shown AND the conclusion mentioned.",
                 "chart_and_conclusion",
                 success_fn=lambda o: len(o) > 30,
                 html_file="J_comb_chart_narration.html", duration_s=35.0),
            Task("J-M4", TaskCategory.J_COMBINED, TaskDifficulty.MEDIUM,
                 "A game level ends with a victory fanfare AND a score popup. Note the score shown and the sound heard.",
                 "score_and_sound",
                 success_fn=lambda o: len(o) > 20 and ("score" in o.lower() or "sound" in o.lower()),
                 html_file="J_comb_game_result.html", duration_s=20.0),
            # HARD
            Task("J-H1", TaskCategory.J_COMBINED, TaskDifficulty.HARD,
                 "A narrated 90s tutorial demonstrates a 5-step data workflow. Reproduce all steps in the live app.",
                 "workflow_done",
                 success_fn=lambda o: len(o) > 50,
                 html_file="J_comb_narrated_tutorial.html", duration_s=95.0),
            Task("J-H2", TaskCategory.J_COMBINED, TaskDifficulty.HARD,
                 "A live dashboard + audio feed: 3 alerts fire (visual badge + audio ding). Triage each by severity shown visually and tone heard.",
                 "3_triaged",
                 success_fn=lambda o: len(o) > 40 and "3" in o,
                 html_file="J_comb_alert_triage.html", duration_s=45.0),
            Task("J-H3", TaskCategory.J_COMBINED, TaskDifficulty.HARD,
                 "A podcast debate plays while a real-time voting poll updates visually. At the end, report who won the debate (audio) and what % voted for them (visual).",
                 "debate_winner_and_percent",
                 success_fn=lambda o: "%" in o and len(o) > 20,
                 html_file="J_comb_debate_poll.html", duration_s=90.0),
        ]

    # ── Utility ────────────────────────────────────────────────────

    def get_by_category(self, category: TaskCategory) -> list[Task]:
        return [t for t in self.tasks if t.category == category]

    def get_by_difficulty(self, difficulty: TaskDifficulty) -> list[Task]:
        return [t for t in self.tasks if t.difficulty == difficulty]

    def summary(self) -> dict:
        return {cat.value: len(self.get_by_category(cat)) for cat in TaskCategory}

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)
