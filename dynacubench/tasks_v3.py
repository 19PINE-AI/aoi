"""
DynaCU-Bench v3 — 11 Categories (10 dynamic + 1 static baseline) = 110 Total

Redesigned benchmark grounded in realistic browser activities.
Every task passes the test: "Would a human actually do this in a browser?"

Real audio pipeline (PulseAudio + Whisper ASR), no DOM-based proxies.
Two-layer audio representation, speak action for voice output.

Categories:
  A: Podcast / Audio Content       — audio perception
  B: Video Conference / Meeting    — audio + visual-temporal
  C: Video / Screencast Watching   — visual-temporal (+ some audio)
  D: Carousel / Rotating Content   — visual-temporal
  E: Live Dashboard / Monitoring   — visual-temporal + real-time
  F: Transient Errors & Notifs     — visual-temporal + real-time
  G: Voice / Phone (Inbound)       — audio + real-time
  H: Voice Interview (Outbound)    — audio + real-time + audio output
  I: Collaborative Editing         — audio + visual-temporal + real-time
  J: Interactive Game              — visual-temporal + real-time
  S: Static Baseline               — no dynamic content (zero-overhead verification)

Difficulty tiers:
  Easy   — single axis (a or b alone), passive observe then act
  Medium — two axes or single under time pressure
  Hard   — all three axes (a+b+c), real-time multi-modal

Evaluation types:
  dom    — window.getTaskResult() returns deterministic success value
  llm    — LLM judge scores agent's typed/spoken response against rubric
  hybrid — DOM gate (did agent act?) + LLM quality (was the answer good?)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class TaskCategory(Enum):
    A_PODCAST       = "A_podcast"
    B_MEETING       = "B_meeting"
    C_VIDEO         = "C_video"
    D_CAROUSEL      = "D_carousel"
    E_DASHBOARD     = "E_dashboard"
    F_TRANSIENT     = "F_transient"
    G_PHONE         = "G_phone"
    H_INTERVIEW     = "H_interview"
    I_COLLAB        = "I_collab"
    J_GAME          = "J_game"
    S_STATIC        = "S_static"


class TaskDifficulty(Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class EvalType(Enum):
    DOM    = "dom"       # Deterministic DOM-state check
    LLM    = "llm"       # LLM judge evaluates agent response
    HYBRID = "hybrid"    # DOM gate + LLM quality check


class CapabilityAxis(Enum):
    AUDIO    = "audio"       # (a) audio perception
    VISUAL   = "visual"      # (b) visual-temporal perception
    INTERACT = "interact"    # (c) real-time interaction


@dataclass
class LLMRubric:
    """Evaluation rubric for LLM-judged tasks."""
    criteria: str           # What the LLM judge should evaluate
    key_facts: list[str]    # Facts that must appear for full credit
    scoring: str            # "binary" (pass/fail) or "partial" (0-1 scale)
    max_score: float = 1.0

    def to_judge_prompt(self, agent_response: str) -> str:
        """Build the prompt for the LLM judge."""
        facts_str = "\n".join(f"  - {f}" for f in self.key_facts)
        return (
            f"You are evaluating an AI agent's response to a browser task.\n\n"
            f"EVALUATION CRITERIA:\n{self.criteria}\n\n"
            f"KEY FACTS (the response must include these for full credit):\n{facts_str}\n\n"
            f"AGENT'S RESPONSE:\n\"{agent_response}\"\n\n"
            f"SCORING MODE: {self.scoring}\n"
            f"{'Score 1 if all key facts present and correct, 0 otherwise.' if self.scoring == 'binary' else 'Score 0.0-1.0 based on fraction of key facts present and correct.'}\n\n"
            f"Respond with ONLY a JSON object: {{\"score\": <number>, \"reason\": \"<brief explanation>\"}}"
        )


@dataclass
class Task:
    task_id: str
    category: TaskCategory
    difficulty: TaskDifficulty
    instruction: str
    ground_truth: Any                        # Expected answer / state
    html_file: str                           # HTML file in benchmark_env/html_tasks/
    eval_type: EvalType = EvalType.DOM       # How to evaluate
    axes: list[CapabilityAxis] = field(default_factory=list)  # Required capabilities
    dom_success_value: Optional[str] = None  # Expected getTaskResult() return
    llm_rubric: Optional[LLMRubric] = None  # For LLM/hybrid evaluation
    duration_s: float = 30.0                 # Task timeout
    requires_audio_out: bool = False         # Needs speak action
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "instruction": self.instruction,
            "ground_truth": str(self.ground_truth),
            "eval_type": self.eval_type.value,
            "axes": [a.value for a in self.axes],
            "duration_s": self.duration_s,
            "html_file": self.html_file,
            "requires_audio_out": self.requires_audio_out,
        }


# ══════════════════════════════════════════════════════════════════════
# Task definitions: 11 categories, 110 tasks
# ══════════════════════════════════════════════════════════════════════

def _cat_a() -> list[Task]:
    """Category A: Podcast / Audio Content — audio perception."""
    return [
        # ── EASY (3) ────────────────────────────────────────────────
        Task(
            "A-E1", TaskCategory.A_PODCAST, TaskDifficulty.EASY,
            "A podcast episode is playing. Listen and type the guest's name into the text field.",
            "Dr. Sarah Chen",
            html_file="A_E1_podcast_guest_name.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="guest_name_correct",
            duration_s=90.0,
        ),
        Task(
            "A-E2", TaskCategory.A_PODCAST, TaskDifficulty.EASY,
            "Listen to the product review podcast. What price did the reviewer mention? Type the price.",
            "$49.99",
            html_file="A_E2_podcast_price.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="price_correct",
            duration_s=90.0,
        ),
        Task(
            "A-E3", TaskCategory.A_PODCAST, TaskDifficulty.EASY,
            "A news briefing is playing. What city was mentioned as the location of the event? Type it.",
            "Stockholm",
            html_file="A_E3_podcast_city.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="city_correct",
            duration_s=90.0,
        ),
        # ── MEDIUM (4) ─────────────────────────────────────────────
        Task(
            "A-M1", TaskCategory.A_PODCAST, TaskDifficulty.MEDIUM,
            "Listen to the podcast segment. The host mentions a book title AND the author. Type both into the form fields.",
            {"title": "The Alignment Problem", "author": "Brian Christian"},
            html_file="A_M1_podcast_book.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="both_correct",
            duration_s=150.0,
        ),
        Task(
            "A-M2", TaskCategory.A_PODCAST, TaskDifficulty.MEDIUM,
            "A science podcast discusses a drug trial. Listen for the drug name AND the efficacy percentage. Enter both.",
            {"drug": "Revantix", "efficacy": "73%"},
            html_file="A_M2_podcast_drug_trial.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="both_correct",
            duration_s=150.0,
        ),
        Task(
            "A-M3", TaskCategory.A_PODCAST, TaskDifficulty.MEDIUM,
            "The podcast host reads three headlines. Type all three into the corresponding fields in order.",
            ["Senate passes climate bill", "Tech layoffs reach 50,000", "Mars rover finds ice"],
            html_file="A_M3_podcast_headlines.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="all_three_correct",
            duration_s=150.0,
        ),
        Task(
            "A-M4", TaskCategory.A_PODCAST, TaskDifficulty.MEDIUM,
            "An interview podcast is playing. Listen for the year the company was founded and the number of employees. Fill in both fields.",
            {"year": "2018", "employees": "340"},
            html_file="A_M4_podcast_company_facts.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="both_correct",
            duration_s=150.0,
        ),
        # ── HARD (3) ────────────────────────────────────────────────
        Task(
            "A-H1", TaskCategory.A_PODCAST, TaskDifficulty.HARD,
            "Listen to the full debate segment. Summarize the three main arguments presented by the speaker in the text area.",
            "renewable energy cost reduction, grid reliability concerns, job transition programs",
            html_file="A_H1_podcast_debate_summary.html",
            eval_type=EvalType.LLM,
            axes=[CapabilityAxis.AUDIO],
            llm_rubric=LLMRubric(
                criteria="The agent must summarize three distinct arguments from a spoken debate segment.",
                key_facts=["renewable energy costs", "grid reliability", "job transition"],
                scoring="partial",
            ),
            duration_s=240.0,
        ),
        Task(
            "A-H2", TaskCategory.A_PODCAST, TaskDifficulty.HARD,
            "A financial podcast discusses quarterly results for three companies. Fill in the revenue figure for each company in the table.",
            {"Acme Corp": "$12.4M", "Bolt Inc": "$8.7M", "Cipher Ltd": "$21.1M"},
            html_file="A_H2_podcast_financials.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="all_revenues_correct",
            duration_s=240.0,
        ),
        Task(
            "A-H3", TaskCategory.A_PODCAST, TaskDifficulty.HARD,
            "A podcast presents a recipe with 5 ingredients and quantities. Pause the podcast and enter each ingredient with its quantity in the shopping list.",
            ["2 cups flour", "3 eggs", "200ml milk", "50g butter", "1 tsp vanilla"],
            html_file="A_H3_podcast_recipe.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.INTERACT],
            dom_success_value="all_ingredients_correct",
            duration_s=240.0,
        ),
    ]


def _cat_b() -> list[Task]:
    """Category B: Video Conference / Meeting — audio + visual-temporal."""
    return [
        # ── EASY (3) ────────────────────────────────────────────────
        Task(
            "B-E1", TaskCategory.B_MEETING, TaskDifficulty.EASY,
            "You joined a team meeting. The presenter verbally announces the new launch date, which is NOT shown on any slide. Type the date.",
            "April 28th",
            html_file="B_E1_meeting_launch_date.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="date_correct",
            duration_s=90.0,
        ),
        Task(
            "B-E2", TaskCategory.B_MEETING, TaskDifficulty.EASY,
            "A meeting presentation is running. The speaker mentions a room number for the next meeting. Type it.",
            "407",
            html_file="B_E2_meeting_room.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="room_correct",
            duration_s=90.0,
        ),
        Task(
            "B-E3", TaskCategory.B_MEETING, TaskDifficulty.EASY,
            "During the meeting, a slide shows a chart title but the Y-axis label is only mentioned verbally. Type the Y-axis label.",
            "Revenue in Millions USD",
            html_file="B_E3_meeting_axis_label.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL],
            dom_success_value="label_correct",
            duration_s=90.0,
        ),
        # ── MEDIUM (4) ─────────────────────────────────────────────
        Task(
            "B-M1", TaskCategory.B_MEETING, TaskDifficulty.MEDIUM,
            "Watch the slide deck and listen to the presenter. Slide 3 shows a table with a blank cell — the presenter fills it in verbally. Type the value.",
            "14.6%",
            html_file="B_M1_meeting_blank_cell.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL],
            dom_success_value="value_correct",
            duration_s=150.0,
        ),
        Task(
            "B-M2", TaskCategory.B_MEETING, TaskDifficulty.MEDIUM,
            "The presenter shows 4 slides. After the presentation, enter how many slides contained a bar chart.",
            "2",
            html_file="B_M2_meeting_chart_count.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="count_correct",
            duration_s=150.0,
        ),
        Task(
            "B-M3", TaskCategory.B_MEETING, TaskDifficulty.MEDIUM,
            "During the meeting, the speaker assigns action items to team members. Fill in who is responsible for each action item listed on screen.",
            {"design review": "Alice", "API migration": "Bob", "docs update": "Carol"},
            html_file="B_M3_meeting_action_items.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL],
            dom_success_value="all_assigned",
            duration_s=150.0,
        ),
        Task(
            "B-M4", TaskCategory.B_MEETING, TaskDifficulty.MEDIUM,
            "A budget review meeting is running. The presenter shows department budgets on slides and verbally corrects one figure. Identify which department was corrected and enter the new amount.",
            {"department": "Marketing", "amount": "$1.4M"},
            html_file="B_M4_meeting_budget_correction.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL],
            dom_success_value="correction_correct",
            duration_s=150.0,
        ),
        # ── HARD (3) ────────────────────────────────────────────────
        Task(
            "B-H1", TaskCategory.B_MEETING, TaskDifficulty.HARD,
            "The presenter shows slides and asks you a question verbally: 'Based on slide 2, which region should we prioritize?' Look at slide 2's data and type your answer in the chat box before the timer runs out.",
            "APAC",
            html_file="B_H1_meeting_question.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="answer_correct",
            duration_s=240.0,
        ),
        Task(
            "B-H2", TaskCategory.B_MEETING, TaskDifficulty.HARD,
            "Watch all 6 meeting slides and listen to the commentary. After the presentation, fill in the meeting minutes form: decisions made, next steps, and owner for each.",
            "meeting_minutes",
            html_file="B_H2_meeting_minutes.html",
            eval_type=EvalType.HYBRID,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL],
            dom_success_value="minutes_submitted",
            llm_rubric=LLMRubric(
                criteria="The agent must produce meeting minutes capturing decisions, next steps, and owners discussed across slides and audio.",
                key_facts=["revenue grew to $15.1M", "mobile app v2 in June", "45 new hires priority backend and ML", "Acme Corp expanded 3x"],
                scoring="partial",
            ),
            duration_s=240.0,
        ),
        Task(
            "B-H3", TaskCategory.B_MEETING, TaskDifficulty.HARD,
            "A live meeting simulation: the presenter asks two questions at different times during the slides. Answer each in the chat within 10 seconds of being asked.",
            "both_answered",
            html_file="B_H3_meeting_live_qa.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="both_answered_in_time",
            duration_s=240.0,
        ),
    ]


def _cat_c() -> list[Task]:
    """Category C: Video / Screencast Watching — visual-temporal."""
    return [
        # ── EASY (3) ────────────────────────────────────────────────
        Task(
            "C-E1", TaskCategory.C_VIDEO, TaskDifficulty.EASY,
            "A terminal recording plays showing someone running commands. What was the first command executed? Type it.",
            "git clone https://github.com/example/repo.git",
            html_file="C_E1_screencast_command.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="command_correct",
            duration_s=90.0,
        ),
        Task(
            "C-E2", TaskCategory.C_VIDEO, TaskDifficulty.EASY,
            "A product demo video auto-plays. What is the product name shown in the title screen? Type it.",
            "CloudSync Pro",
            html_file="C_E2_video_product_name.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="name_correct",
            duration_s=90.0,
        ),
        Task(
            "C-E3", TaskCategory.C_VIDEO, TaskDifficulty.EASY,
            "Watch the tutorial video. How many steps are shown in the setup process? Enter the number.",
            "4",
            html_file="C_E3_video_step_count.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="count_correct",
            duration_s=90.0,
        ),
        # ── MEDIUM (4) ─────────────────────────────────────────────
        Task(
            "C-M1", TaskCategory.C_VIDEO, TaskDifficulty.MEDIUM,
            "A coding tutorial plays. Watch the code being written and type the function name that was defined.",
            "process_data",
            html_file="C_M1_screencast_function.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="function_correct",
            duration_s=150.0,
        ),
        Task(
            "C-M2", TaskCategory.C_VIDEO, TaskDifficulty.MEDIUM,
            "A video demo shows navigating through 3 settings pages. List the title of each page in order.",
            ["General", "Security", "Notifications"],
            html_file="C_M2_video_page_titles.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="all_titles_correct",
            duration_s=150.0,
        ),
        Task(
            "C-M3", TaskCategory.C_VIDEO, TaskDifficulty.MEDIUM,
            "Watch the screencast. The presenter opens a config file and changes a port number. Type both the old and new port numbers.",
            {"old": "3000", "new": "8080"},
            html_file="C_M3_screencast_port_change.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="ports_correct",
            duration_s=150.0,
        ),
        Task(
            "C-M4", TaskCategory.C_VIDEO, TaskDifficulty.MEDIUM,
            "A narrated tutorial plays with audio explaining each visual step. The narrator mentions a shortcut key not shown on screen. Type the shortcut.",
            "Ctrl+Shift+P",
            html_file="C_M4_video_shortcut.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.AUDIO],
            dom_success_value="shortcut_correct",
            duration_s=150.0,
        ),
        # ── HARD (3) ────────────────────────────────────────────────
        Task(
            "C-H1", TaskCategory.C_VIDEO, TaskDifficulty.HARD,
            "A video tutorial demonstrates a 5-step configuration process. Replicate each step in the settings panel on the right side of the page.",
            "workflow_replicated",
            html_file="C_H1_video_replicate.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="all_5_steps_correct",
            duration_s=240.0,
        ),
        Task(
            "C-H2", TaskCategory.C_VIDEO, TaskDifficulty.HARD,
            "Watch the recorded code review. The reviewer highlights 3 bugs across different files. Describe each bug in the report fields.",
            "three_bugs",
            html_file="C_H2_video_code_review.html",
            eval_type=EvalType.HYBRID,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.AUDIO],
            dom_success_value="report_submitted",
            llm_rubric=LLMRubric(
                criteria="Agent must describe 3 bugs highlighted in a code review video.",
                key_facts=["off-by-one in loop index", "null pointer dereference", "SQL injection in query builder"],
                scoring="partial",
            ),
            duration_s=240.0,
        ),
        Task(
            "C-H3", TaskCategory.C_VIDEO, TaskDifficulty.HARD,
            "A screencast shows a developer debugging. They open 4 files and place breakpoints. List each filename and line number where a breakpoint was set.",
            {"app.py:42": True, "utils.py:17": True, "config.py:8": True, "main.py:31": True},
            html_file="C_H3_video_breakpoints.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="all_breakpoints_correct",
            duration_s=240.0,
        ),
    ]


def _cat_d() -> list[Task]:
    """Category D: Carousel / Rotating Content — visual-temporal."""
    return [
        # ── EASY (3) ────────────────────────────────────────────────
        Task(
            "D-E1", TaskCategory.D_CAROUSEL, TaskDifficulty.EASY,
            "A product carousel rotates through 4 items. Find the item with the lowest price and click 'Buy' on it.",
            "Wireless Earbuds - $29.99",
            html_file="D_E1_carousel_cheapest.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="cheapest_bought",
            duration_s=90.0,
        ),
        Task(
            "D-E2", TaskCategory.D_CAROUSEL, TaskDifficulty.EASY,
            "A testimonial carousel rotates. How many testimonials mention the word 'excellent'? Type the count.",
            "2",
            html_file="D_E2_carousel_testimonials.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="count_correct",
            duration_s=90.0,
        ),
        Task(
            "D-E3", TaskCategory.D_CAROUSEL, TaskDifficulty.EASY,
            "A news ticker scrolls headlines across the top of the page. Click the headline about technology.",
            "tech_headline_clicked",
            html_file="D_E3_ticker_headline.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="tech_headline_clicked",
            duration_s=90.0,
        ),
        # ── MEDIUM (4) ─────────────────────────────────────────────
        Task(
            "D-M1", TaskCategory.D_CAROUSEL, TaskDifficulty.MEDIUM,
            "A hero banner rotates every 4 seconds showing promotional offers. Find and enter the promo code from the winter sale banner.",
            "WINTER25",
            html_file="D_M1_carousel_promo_code.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="code_correct",
            duration_s=150.0,
        ),
        Task(
            "D-M2", TaskCategory.D_CAROUSEL, TaskDifficulty.MEDIUM,
            "An image gallery auto-rotates. Count the total number of unique images shown and enter the count.",
            "6",
            html_file="D_M2_carousel_image_count.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="count_correct",
            duration_s=150.0,
        ),
        Task(
            "D-M3", TaskCategory.D_CAROUSEL, TaskDifficulty.MEDIUM,
            "A product carousel shows items with ratings. Which product has the highest star rating? Type its name.",
            "Premium Headphones",
            html_file="D_M3_carousel_highest_rated.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="product_correct",
            duration_s=150.0,
        ),
        Task(
            "D-M4", TaskCategory.D_CAROUSEL, TaskDifficulty.MEDIUM,
            "A sliding banner shows 5 team members with their roles. Find the CTO and enter their name.",
            "David Park",
            html_file="D_M4_carousel_team_cto.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="name_correct",
            duration_s=150.0,
        ),
        # ── HARD (3) ────────────────────────────────────────────────
        Task(
            "D-H1", TaskCategory.D_CAROUSEL, TaskDifficulty.HARD,
            "A rapidly rotating carousel (2s per slide) shows 8 products with prices. Calculate the total cost of all products and enter it.",
            "$847.92",
            html_file="D_H1_carousel_total_price.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="total_correct",
            duration_s=240.0,
        ),
        Task(
            "D-H2", TaskCategory.D_CAROUSEL, TaskDifficulty.HARD,
            "Two carousels rotate independently — one shows features, the other shows pricing tiers. Match each feature to its pricing tier and fill in the table.",
            "features_matched",
            html_file="D_H2_carousel_dual_match.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="all_matched",
            duration_s=240.0,
        ),
        Task(
            "D-H3", TaskCategory.D_CAROUSEL, TaskDifficulty.HARD,
            "A news feed carousel shows 10 stories cycling rapidly. A breaking news alert appears on one specific slide for 3 seconds. Click the breaking news banner when it appears.",
            "breaking_clicked",
            html_file="D_H3_carousel_breaking_news.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="breaking_clicked",
            duration_s=240.0,
        ),
    ]


def _cat_e() -> list[Task]:
    """Category E: Live Dashboard / Monitoring — visual-temporal + real-time."""
    return [
        # ── EASY (3) ────────────────────────────────────────────────
        Task(
            "E-E1", TaskCategory.E_DASHBOARD, TaskDifficulty.EASY,
            "A server dashboard shows CPU usage updating every 2 seconds. When CPU exceeds 90%, click the 'Alert' button. Wait and watch.",
            "alert_triggered",
            html_file="E_E1_dashboard_cpu_alert.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="alert_triggered",
            duration_s=90.0,
        ),
        Task(
            "E-E2", TaskCategory.E_DASHBOARD, TaskDifficulty.EASY,
            "A stock ticker dashboard shows live prices. What is the highest price reached by ACME stock? Type it after observing for at least 20 seconds.",
            "142.50",
            html_file="E_E2_dashboard_stock_peak.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="peak_correct",
            duration_s=90.0,
        ),
        Task(
            "E-E3", TaskCategory.E_DASHBOARD, TaskDifficulty.EASY,
            "An IoT sensor dashboard updates temperature readings. When temperature drops below 32F, type 'FREEZE WARNING' in the alert field.",
            "FREEZE WARNING",
            html_file="E_E3_dashboard_temp_warning.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="warning_correct",
            duration_s=90.0,
        ),
        # ── MEDIUM (4) ─────────────────────────────────────────────
        Task(
            "E-M1", TaskCategory.E_DASHBOARD, TaskDifficulty.MEDIUM,
            "A network monitoring dashboard shows packet loss for 4 servers. Identify which server had the highest packet loss spike and enter its name.",
            "web-server-03",
            html_file="E_M1_dashboard_packet_loss.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="server_correct",
            duration_s=150.0,
        ),
        Task(
            "E-M2", TaskCategory.E_DASHBOARD, TaskDifficulty.MEDIUM,
            "A real-time log viewer scrolls application logs. Count the number of ERROR-level entries that appear in 30 seconds and enter the count.",
            "7",
            html_file="E_M2_dashboard_error_count.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="count_correct",
            duration_s=150.0,
        ),
        Task(
            "E-M3", TaskCategory.E_DASHBOARD, TaskDifficulty.MEDIUM,
            "A sales dashboard shows revenue by region updating live. When the East region overtakes West in total revenue, click the 'Flip Alert' button.",
            "flip_detected",
            html_file="E_M3_dashboard_revenue_flip.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="flip_detected",
            duration_s=150.0,
        ),
        Task(
            "E-M4", TaskCategory.E_DASHBOARD, TaskDifficulty.MEDIUM,
            "An application performance dashboard shows 3 metrics: latency, throughput, and error rate. When ANY metric goes red (critical), click the corresponding 'Acknowledge' button.",
            "acknowledged",
            html_file="E_M4_dashboard_acknowledge.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="critical_acknowledged",
            duration_s=150.0,
        ),
        # ── HARD (3) ────────────────────────────────────────────────
        Task(
            "E-H1", TaskCategory.E_DASHBOARD, TaskDifficulty.HARD,
            "A NOC dashboard shows 6 servers. Three will go critical at different times. Acknowledge each within 5 seconds of it going red. All 3 must be acknowledged.",
            "all_acknowledged",
            html_file="E_H1_dashboard_triage.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="all_3_acknowledged",
            duration_s=240.0,
        ),
        Task(
            "E-H2", TaskCategory.E_DASHBOARD, TaskDifficulty.HARD,
            "A multi-panel dashboard: network, CPU, memory, and disk all update live. Write a one-paragraph status report based on the observed data trends.",
            "status_report",
            html_file="E_H2_dashboard_report.html",
            eval_type=EvalType.HYBRID,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="report_submitted",
            llm_rubric=LLMRubric(
                criteria="Agent must write a status report reflecting observed dashboard trends.",
                key_facts=["CPU spiked above 80%", "memory usage stable around 60%", "disk I/O increasing", "network latency normal"],
                scoring="partial",
            ),
            duration_s=240.0,
        ),
        Task(
            "E-H3", TaskCategory.E_DASHBOARD, TaskDifficulty.HARD,
            "A dashboard with audio alerts: a beep sounds when any metric crosses a threshold, and the specific metric is announced verbally. Acknowledge the correct metric each time. Handle 3 alerts.",
            "all_alerts_handled",
            html_file="E_H3_dashboard_audio_alert.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.AUDIO, CapabilityAxis.INTERACT],
            dom_success_value="all_3_alerts_correct",
            duration_s=240.0,
        ),
    ]


def _cat_f() -> list[Task]:
    """Category F: Transient Errors & Notifications — visual-temporal + real-time."""
    return [
        # ── EASY (3) ────────────────────────────────────────────────
        Task(
            "F-E1", TaskCategory.F_TRANSIENT, TaskDifficulty.EASY,
            "Submit the form. An error toast will appear briefly at the top. Read the error message and fix the email field, then resubmit.",
            "form_accepted",
            html_file="F_E1_toast_error_fix.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="form_accepted",
            duration_s=90.0,
        ),
        Task(
            "F-E2", TaskCategory.F_TRANSIENT, TaskDifficulty.EASY,
            "A cookie consent banner appears at the bottom. It auto-dismisses in 5 seconds. Click 'Accept All' before it disappears.",
            "cookies_accepted",
            html_file="F_E2_cookie_consent.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="cookies_accepted",
            duration_s=90.0,
        ),
        Task(
            "F-E3", TaskCategory.F_TRANSIENT, TaskDifficulty.EASY,
            "A download starts automatically. A completion toast shows the filename briefly. Type the filename that was downloaded.",
            "quarterly_report_Q3.pdf",
            html_file="F_E3_download_toast.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="filename_correct",
            duration_s=90.0,
        ),
        # ── MEDIUM (4) ─────────────────────────────────────────────
        Task(
            "F-M1", TaskCategory.F_TRANSIENT, TaskDifficulty.MEDIUM,
            "A session expiration warning appears with a countdown timer. Click 'Extend Session' before the timer reaches zero.",
            "session_extended",
            html_file="F_M1_session_warning.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="session_extended",
            duration_s=150.0,
        ),
        Task(
            "F-M2", TaskCategory.F_TRANSIENT, TaskDifficulty.MEDIUM,
            "Fill out a multi-field form. After submission, a validation toast briefly shows which field has an error. Fix only that field and resubmit.",
            "form_valid",
            html_file="F_M2_validation_toast.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="form_valid",
            duration_s=150.0,
        ),
        Task(
            "F-M3", TaskCategory.F_TRANSIENT, TaskDifficulty.MEDIUM,
            "A web app shows notification badges that appear and fade. Count how many notifications appeared in total during the 30-second observation period.",
            "5",
            html_file="F_M3_notification_count.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="count_correct",
            duration_s=150.0,
        ),
        Task(
            "F-M4", TaskCategory.F_TRANSIENT, TaskDifficulty.MEDIUM,
            "A 'flash sale' pop-up appears for 4 seconds with a discount code. Enter the code in the checkout field before the sale ends.",
            "FLASH40",
            html_file="F_M4_flash_sale.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="code_applied",
            duration_s=150.0,
        ),
        # ── HARD (3) ────────────────────────────────────────────────
        Task(
            "F-H1", TaskCategory.F_TRANSIENT, TaskDifficulty.HARD,
            "Submit a form. The first toast shows an error with a code (e.g., ERR-4021). Fix the issue, resubmit, and read the success toast's confirmation number. Enter the confirmation number.",
            "CNF-78234",
            html_file="F_H1_error_chain.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="confirmation_correct",
            duration_s=240.0,
        ),
        Task(
            "F-H2", TaskCategory.F_TRANSIENT, TaskDifficulty.HARD,
            "A deployment dashboard shows 3 deploy stages running. Toast notifications appear briefly when each stage succeeds or fails. Record the final status of each stage.",
            {"build": "success", "test": "failed", "deploy": "skipped"},
            html_file="F_H2_deploy_stages.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="all_statuses_correct",
            duration_s=240.0,
        ),
        Task(
            "F-H3", TaskCategory.F_TRANSIENT, TaskDifficulty.HARD,
            "Multiple error toasts will appear in sequence, each showing a different error code. The page asks for the THIRD error code. Enter it.",
            "ERR-7712",
            html_file="F_H3_sequential_toasts.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="third_code_correct",
            duration_s=240.0,
        ),
    ]


def _cat_g() -> list[Task]:
    """Category G: Voice / Phone Interaction (Inbound) — audio + real-time."""
    return [
        # ── EASY (3) ────────────────────────────────────────────────
        Task(
            "G-E1", TaskCategory.G_PHONE, TaskDifficulty.EASY,
            "You have a voicemail. Listen to it and type the callback number into the phone field.",
            "555-0147",
            html_file="G_E1_voicemail_number.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="number_correct",
            duration_s=90.0,
        ),
        Task(
            "G-E2", TaskCategory.G_PHONE, TaskDifficulty.EASY,
            "A recorded phone message plays. The caller leaves their name. Type the caller's name.",
            "Jennifer Martinez",
            html_file="G_E2_voicemail_name.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="name_correct",
            duration_s=90.0,
        ),
        Task(
            "G-E3", TaskCategory.G_PHONE, TaskDifficulty.EASY,
            "A phone system plays an automated message with business hours. Enter the closing time.",
            "6:30 PM",
            html_file="G_E3_phone_hours.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="time_correct",
            duration_s=90.0,
        ),
        # ── MEDIUM (4) ─────────────────────────────────────────────
        Task(
            "G-M1", TaskCategory.G_PHONE, TaskDifficulty.MEDIUM,
            "An IVR system plays: 'Press 1 for Sales, Press 2 for Support, Press 3 for Billing'. You need Support. Click the correct button.",
            "support",
            html_file="G_M1_ivr_menu.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.INTERACT],
            dom_success_value="support_selected",
            duration_s=150.0,
        ),
        Task(
            "G-M2", TaskCategory.G_PHONE, TaskDifficulty.MEDIUM,
            "A caller dictates an address over the phone. Type the full address into the form.",
            "742 Evergreen Terrace, Springfield, IL 62704",
            html_file="G_M2_phone_address.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="address_correct",
            duration_s=150.0,
        ),
        Task(
            "G-M3", TaskCategory.G_PHONE, TaskDifficulty.MEDIUM,
            "A customer calls and describes a problem: their order number and the issue. Fill in both fields from what you hear.",
            {"order": "ORD-98234", "issue": "wrong item shipped"},
            html_file="G_M3_phone_complaint.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="both_fields_correct",
            duration_s=150.0,
        ),
        Task(
            "G-M4", TaskCategory.G_PHONE, TaskDifficulty.MEDIUM,
            "An automated phone system reads a verification code digit by digit. Type the 6-digit code.",
            "847293",
            html_file="G_M4_phone_verification.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="code_correct",
            duration_s=150.0,
        ),
        # ── HARD (3) ────────────────────────────────────────────────
        Task(
            "G-H1", TaskCategory.G_PHONE, TaskDifficulty.HARD,
            "A multi-turn phone call: the caller first asks for your account number (shown on screen), then asks for your zip code (also shown). Respond to each question by typing in the response field within 10 seconds.",
            "both_responded",
            html_file="G_H1_phone_multi_turn.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="both_responded_in_time",
            duration_s=240.0,
        ),
        Task(
            "G-H2", TaskCategory.G_PHONE, TaskDifficulty.HARD,
            "A conference call with 2 speakers. Each speaker gives a different data point. One is mentioned only by Speaker A, the other only by Speaker B. Enter both values attributed to the correct speaker.",
            {"speaker_a": "Q3 target: 15%", "speaker_b": "budget: $2.1M"},
            html_file="G_H2_phone_multi_speaker.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="both_attributed_correct",
            duration_s=240.0,
        ),
        Task(
            "G-H3", TaskCategory.G_PHONE, TaskDifficulty.HARD,
            "A caller reads a list of 5 items with prices. The phone UI shows a shopping cart. Add each item and price as the caller says them, keeping up in real time.",
            "all_5_items_added",
            html_file="G_H3_phone_shopping_list.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.INTERACT],
            dom_success_value="all_5_correct",
            duration_s=240.0,
        ),
    ]


def _cat_h() -> list[Task]:
    """Category H: Voice Interview / Audio Output (Outbound) — audio output required."""
    return [
        # ── EASY (3) ────────────────────────────────────────────────
        Task(
            "H-E1", TaskCategory.H_INTERVIEW, TaskDifficulty.EASY,
            "An interviewer asks: 'What is the capital of France?' Speak your answer using the speak action.",
            "Paris",
            html_file="H_E1_interview_capital.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.INTERACT],
            dom_success_value="answer_heard_correct",
            requires_audio_out=True,
            duration_s=90.0,
        ),
        Task(
            "H-E2", TaskCategory.H_INTERVIEW, TaskDifficulty.EASY,
            "The interviewer asks: 'Please state your name as shown on the screen.' Read the name displayed on the page and speak it.",
            "Alexander Thompson",
            html_file="H_E2_interview_name.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="name_spoken_correct",
            requires_audio_out=True,
            duration_s=90.0,
        ),
        Task(
            "H-E3", TaskCategory.H_INTERVIEW, TaskDifficulty.EASY,
            "The system says: 'To verify your identity, please say the 4-digit PIN shown on screen.' Read the PIN aloud.",
            "7294",
            html_file="H_E3_interview_pin.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="pin_verified",
            requires_audio_out=True,
            duration_s=90.0,
        ),
        # ── MEDIUM (4) ─────────────────────────────────────────────
        Task(
            "H-M1", TaskCategory.H_INTERVIEW, TaskDifficulty.MEDIUM,
            "The interviewer asks: 'What is the total revenue shown in the report on screen?' Read the financial report and speak the total.",
            "$4.2 million",
            html_file="H_M1_interview_revenue.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="revenue_spoken_correct",
            requires_audio_out=True,
            duration_s=150.0,
        ),
        Task(
            "H-M2", TaskCategory.H_INTERVIEW, TaskDifficulty.MEDIUM,
            "A voice agent asks two questions sequentially. Answer each by speaking. Q1: 'What year is shown?' Q2: 'What color is the header?'",
            {"year": "2026", "color": "blue"},
            html_file="H_M2_interview_two_questions.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="both_answers_correct",
            requires_audio_out=True,
            duration_s=150.0,
        ),
        Task(
            "H-M3", TaskCategory.H_INTERVIEW, TaskDifficulty.MEDIUM,
            "An automated phone system asks: 'Please say your order number.' The order number is displayed on the page. Speak it clearly.",
            "ORD-44821",
            html_file="H_M3_interview_order.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="order_verified",
            requires_audio_out=True,
            duration_s=150.0,
        ),
        Task(
            "H-M4", TaskCategory.H_INTERVIEW, TaskDifficulty.MEDIUM,
            "A language assessment asks you to repeat a sentence read aloud by the system. Listen, then speak the same sentence.",
            "The quick brown fox jumps over the lazy dog",
            html_file="H_M4_interview_repeat.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.INTERACT],
            dom_success_value="sentence_matched",
            requires_audio_out=True,
            duration_s=150.0,
        ),
        # ── HARD (3) ────────────────────────────────────────────────
        Task(
            "H-H1", TaskCategory.H_INTERVIEW, TaskDifficulty.HARD,
            "A job interview simulation. The interviewer asks: 'Describe the project shown on screen in your own words.' You must read the project details and speak a coherent 2-3 sentence description.",
            "project_description",
            html_file="H_H1_interview_describe.html",
            eval_type=EvalType.HYBRID,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="response_heard",
            llm_rubric=LLMRubric(
                criteria="Agent must speak a coherent description of a project based on on-screen details.",
                key_facts=["project name: DataPipeline v2", "Python and Apache Kafka", "real-time data processing"],
                scoring="partial",
            ),
            requires_audio_out=True,
            duration_s=240.0,
        ),
        Task(
            "H-H2", TaskCategory.H_INTERVIEW, TaskDifficulty.HARD,
            "A verbal math test: the system speaks a math problem, and the answer is partially shown on screen. Combine both to compute and speak the result.",
            "156",
            html_file="H_H2_interview_math.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="answer_correct",
            requires_audio_out=True,
            duration_s=240.0,
        ),
        Task(
            "H-H3", TaskCategory.H_INTERVIEW, TaskDifficulty.HARD,
            "A multi-round interview: the system asks 3 questions about data shown on a dashboard that updates between questions. Answer each by speaking within 15 seconds.",
            "all_3_answered",
            html_file="H_H3_interview_dashboard.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="all_3_correct",
            requires_audio_out=True,
            duration_s=240.0,
        ),
    ]


def _cat_i() -> list[Task]:
    """Category I: Collaborative Editing — audio + visual-temporal + real-time."""
    return [
        # ── EASY (3) ────────────────────────────────────────────────
        Task(
            "I-E1", TaskCategory.I_COLLAB, TaskDifficulty.EASY,
            "A collaborator dictates a sentence over audio. Type what they say into the shared document.",
            "The deadline for the proposal is next Friday",
            html_file="I_E1_collab_dictation.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO],
            dom_success_value="text_correct",
            duration_s=90.0,
        ),
        Task(
            "I-E2", TaskCategory.I_COLLAB, TaskDifficulty.EASY,
            "A collaborator is editing the document in real time. They add a new heading. Type the heading they added into the response field.",
            "Q4 Objectives",
            html_file="I_E2_collab_observe_heading.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL],
            dom_success_value="heading_correct",
            duration_s=90.0,
        ),
        Task(
            "I-E3", TaskCategory.I_COLLAB, TaskDifficulty.EASY,
            "A collaborator says 'delete the last paragraph' over audio. The document is visible. Remove the last paragraph.",
            "paragraph_deleted",
            html_file="I_E3_collab_delete.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.INTERACT],
            dom_success_value="last_paragraph_deleted",
            duration_s=90.0,
        ),
        # ── MEDIUM (4) ─────────────────────────────────────────────
        Task(
            "I-M1", TaskCategory.I_COLLAB, TaskDifficulty.MEDIUM,
            "A collaborator verbally says 'change the number in section 2 to 450'. Find section 2 in the document and update the number.",
            "number_updated",
            html_file="I_M1_collab_edit_number.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="correct_value_in_section2",
            duration_s=150.0,
        ),
        Task(
            "I-M2", TaskCategory.I_COLLAB, TaskDifficulty.MEDIUM,
            "The collaborator edits section 1 visually (you see the cursor moving) while verbally telling you to edit section 3. Make the edit they describe in section 3, not section 1.",
            "correct_section_edited",
            html_file="I_M2_collab_parallel_edit.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="section3_updated",
            duration_s=150.0,
        ),
        Task(
            "I-M3", TaskCategory.I_COLLAB, TaskDifficulty.MEDIUM,
            "A collaborator pastes a table into the doc (visible) and says 'the total in the last row is wrong — fix it.' Calculate the correct total and update the cell.",
            "total_fixed",
            html_file="I_M3_collab_fix_total.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="total_correct",
            duration_s=150.0,
        ),
        Task(
            "I-M4", TaskCategory.I_COLLAB, TaskDifficulty.MEDIUM,
            "While you observe real-time edits to a document, the collaborator verbally asks: 'What did I just change?' Type what was changed.",
            "edit_described",
            html_file="I_M4_collab_describe_change.html",
            eval_type=EvalType.HYBRID,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL],
            dom_success_value="response_submitted",
            llm_rubric=LLMRubric(
                criteria="Agent must describe a real-time edit observed visually.",
                key_facts=["changed title from 'Draft' to 'Final Report'", "updated the date"],
                scoring="partial",
            ),
            duration_s=150.0,
        ),
        # ── HARD (3) ────────────────────────────────────────────────
        Task(
            "I-H1", TaskCategory.I_COLLAB, TaskDifficulty.HARD,
            "Collaborative session: the partner dictates 3 bullet points via audio while simultaneously editing formatting on screen. Type all 3 bullet points in the document.",
            "3_bullets_entered",
            html_file="I_H1_collab_3_bullets.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="all_3_bullets_correct",
            duration_s=240.0,
        ),
        Task(
            "I-H2", TaskCategory.I_COLLAB, TaskDifficulty.HARD,
            "A full collaborative editing session: partner makes 2 edits visually and gives 2 verbal instructions. Execute all 4 changes correctly.",
            "all_changes",
            html_file="I_H2_collab_full_session.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="all_4_changes_correct",
            duration_s=240.0,
        ),
        Task(
            "I-H3", TaskCategory.I_COLLAB, TaskDifficulty.HARD,
            "Collaborator asks via audio: 'Read me the summary paragraph.' You must read the visible summary and speak it back. Then they say 'Now add a conclusion.' Add a conclusion paragraph.",
            "read_and_edit",
            html_file="I_H3_collab_read_and_edit.html",
            eval_type=EvalType.HYBRID,
            axes=[CapabilityAxis.AUDIO, CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="conclusion_added",
            llm_rubric=LLMRubric(
                criteria="Agent must first speak the summary visible on screen, then add a conclusion paragraph.",
                key_facts=["spoke the summary text", "added a conclusion paragraph"],
                scoring="binary",
            ),
            requires_audio_out=True,
            duration_s=240.0,
        ),
    ]


def _cat_j() -> list[Task]:
    """Category J: Interactive Game / Real-time Response — visual-temporal + real-time."""
    return [
        # ── EASY (3) ────────────────────────────────────────────────
        Task(
            "J-E1", TaskCategory.J_GAME, TaskDifficulty.EASY,
            "A whack-a-mole game. Moles pop up one at a time and stay for 5 seconds. Click at least 3 moles to win.",
            "3_moles_hit",
            html_file="J_E1_game_whack_mole.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="score_3_or_more",
            duration_s=90.0,
        ),
        Task(
            "J-E2", TaskCategory.J_GAME, TaskDifficulty.EASY,
            "A reaction time test. When the screen turns green, click as fast as you can. Complete 3 rounds.",
            "3_rounds_complete",
            html_file="J_E2_game_reaction.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="3_rounds_complete",
            duration_s=90.0,
        ),
        Task(
            "J-E3", TaskCategory.J_GAME, TaskDifficulty.EASY,
            "A color matching game. A target color is shown at the top. Click the matching color swatch when it appears among rotating options.",
            "color_matched",
            html_file="J_E3_game_color_match.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="correct_color_clicked",
            duration_s=90.0,
        ),
        # ── MEDIUM (4) ─────────────────────────────────────────────
        Task(
            "J-M1", TaskCategory.J_GAME, TaskDifficulty.MEDIUM,
            "Simon Says: watch a sequence of 4 colored button flashes, then repeat the pattern by clicking the buttons in the same order.",
            "pattern_correct",
            html_file="J_M1_game_simon.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="pattern_correct",
            duration_s=150.0,
        ),
        Task(
            "J-M2", TaskCategory.J_GAME, TaskDifficulty.MEDIUM,
            "A memory card game. Cards flip briefly then face down. Find and click 3 matching pairs.",
            "3_pairs_found",
            html_file="J_M2_game_memory.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="3_pairs_matched",
            duration_s=150.0,
        ),
        Task(
            "J-M3", TaskCategory.J_GAME, TaskDifficulty.MEDIUM,
            "A number puzzle: numbers 1-9 appear in random positions and stay for 3 seconds, then hide. Click them in ascending order from memory.",
            "sequence_correct",
            html_file="J_M3_game_number_memory.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="ascending_correct",
            duration_s=150.0,
        ),
        Task(
            "J-M4", TaskCategory.J_GAME, TaskDifficulty.MEDIUM,
            "A typing speed game. Words appear on screen and scroll down. Type each word before it reaches the bottom. Successfully type 5 words.",
            "5_words_typed",
            html_file="J_M4_game_typing.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="5_words_correct",
            duration_s=150.0,
        ),
        # ── HARD (3) ────────────────────────────────────────────────
        Task(
            "J-H1", TaskCategory.J_GAME, TaskDifficulty.HARD,
            "A platformer mini-game: guide the character by clicking platforms as they appear. Survive for 30 seconds without falling.",
            "survived_30s",
            html_file="J_H1_game_platformer.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="survived_30s",
            duration_s=240.0,
        ),
        Task(
            "J-H2", TaskCategory.J_GAME, TaskDifficulty.HARD,
            "An advanced Simon Says: watch a 6-step pattern with 6 different colors, then reproduce it. You have 2 attempts.",
            "pattern_6_correct",
            html_file="J_H2_game_simon_hard.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="pattern_correct",
            duration_s=240.0,
        ),
        Task(
            "J-H3", TaskCategory.J_GAME, TaskDifficulty.HARD,
            "A dual-task game: track a moving target with clicks while simultaneously typing numbers that flash on screen. Score 5 hits AND 3 correct numbers.",
            "dual_task_complete",
            html_file="J_H3_game_dual_task.html",
            eval_type=EvalType.DOM,
            axes=[CapabilityAxis.VISUAL, CapabilityAxis.INTERACT],
            dom_success_value="hits_5_numbers_3",
            duration_s=240.0,
        ),
    ]


def _cat_s() -> list[Task]:
    """Category S: Static — zero-overhead baseline (no dynamic content)."""
    return [
        Task(
            "S-E1", TaskCategory.S_STATIC, TaskDifficulty.EASY,
            "Read the green heading on the page and type its exact text into the answer box. Click Submit.",
            "Quarterly Revenue Report",
            html_file="S_E1_read_heading.html",
            eval_type=EvalType.DOM,
            axes=[],
            dom_success_value="heading_correct",
            duration_s=30.0,
        ),
        Task(
            "S-E2", TaskCategory.S_STATIC, TaskDifficulty.EASY,
            "Click the button labeled 'Confirm Order'.",
            "Confirm Order",
            html_file="S_E2_click_button.html",
            eval_type=EvalType.DOM,
            axes=[],
            dom_success_value="button_correct",
            duration_s=30.0,
        ),
        Task(
            "S-E3", TaskCategory.S_STATIC, TaskDifficulty.EASY,
            "From the shipping method dropdown, select 'Express Shipping (2-3 days)' and click Submit.",
            "express",
            html_file="S_E3_select_dropdown.html",
            eval_type=EvalType.DOM,
            axes=[],
            dom_success_value="dropdown_correct",
            duration_s=30.0,
        ),
        Task(
            "S-E4", TaskCategory.S_STATIC, TaskDifficulty.EASY,
            "Fill in the contact form with Name: 'Jane Peterson' and Email: 'jane.peterson@example.com'. Click Submit.",
            {"name": "Jane Peterson", "email": "jane.peterson@example.com"},
            html_file="S_E4_contact_form.html",
            eval_type=EvalType.DOM,
            axes=[],
            dom_success_value="form_correct",
            duration_s=30.0,
        ),
        Task(
            "S-E5", TaskCategory.S_STATIC, TaskDifficulty.EASY,
            "Check the 'I agree to the terms and conditions' checkbox and click Submit.",
            "checkbox_submitted",
            html_file="S_E5_checkbox_submit.html",
            eval_type=EvalType.DOM,
            axes=[],
            dom_success_value="checkbox_submitted",
            duration_s=30.0,
        ),
        Task(
            "S-E6", TaskCategory.S_STATIC, TaskDifficulty.EASY,
            "Look at the product table. Find the price of 'Wireless Mouse' and type it into the answer box. Click Submit.",
            "$34.50",
            html_file="S_E6_read_table.html",
            eval_type=EvalType.DOM,
            axes=[],
            dom_success_value="table_correct",
            duration_s=30.0,
        ),
        Task(
            "S-E7", TaskCategory.S_STATIC, TaskDifficulty.EASY,
            "Click the link labeled 'Account Settings' from the navigation menu.",
            "Account Settings",
            html_file="S_E7_click_link.html",
            eval_type=EvalType.DOM,
            axes=[],
            dom_success_value="link_correct",
            duration_s=30.0,
        ),
        Task(
            "S-E8", TaskCategory.S_STATIC, TaskDifficulty.EASY,
            "Count the number of items in the shopping list and type the count into the answer box. Click Submit.",
            "7",
            html_file="S_E8_count_items.html",
            eval_type=EvalType.DOM,
            axes=[],
            dom_success_value="count_correct",
            duration_s=30.0,
        ),
        Task(
            "S-E9", TaskCategory.S_STATIC, TaskDifficulty.EASY,
            "Read the reference code in the Source field and type the exact same text into the Destination field. Click Submit.",
            "TXN-2024-08-ALPHA",
            html_file="S_E9_copy_text.html",
            eval_type=EvalType.DOM,
            axes=[],
            dom_success_value="copy_correct",
            duration_s=30.0,
        ),
        Task(
            "S-E10", TaskCategory.S_STATIC, TaskDifficulty.EASY,
            "Read the paragraph about the Greenfield Community Library. Answer the question: Who founded it? Type the answer and click Submit.",
            "Margaret Thornton",
            html_file="S_E10_read_paragraph.html",
            eval_type=EvalType.DOM,
            axes=[],
            dom_success_value="answer_correct",
            duration_s=30.0,
        ),
        # Easy 11-22 (12 added)
        Task("S-E11", TaskCategory.S_STATIC, TaskDifficulty.EASY,
             "Read the customer contact card and type the phone number exactly as shown into the answer box. Click Submit.",
             "(415) 555-0182", html_file="S_E11_read_phone.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="phone_correct", duration_s=30.0),
        Task("S-E12", TaskCategory.S_STATIC, TaskDifficulty.EASY,
             "Choose the most relevant category for this support ticket and click Submit.",
             "billing", html_file="S_E12_radio_category.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="category_correct", duration_s=30.0),
        Task("S-E13", TaskCategory.S_STATIC, TaskDifficulty.EASY,
             "Read the display settings table and type the refresh rate (number only) into the answer box. Click Submit.",
             "144", html_file="S_E13_read_setting.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="refresh_correct", duration_s=30.0),
        Task("S-E14", TaskCategory.S_STATIC, TaskDifficulty.EASY,
             "Sum the two prices in the order summary and type the total dollar amount into the answer box. Click Submit.",
             "49.75", html_file="S_E14_sum_prices.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="sum_correct", duration_s=30.0),
        Task("S-E15", TaskCategory.S_STATIC, TaskDifficulty.EASY,
             "Select exactly the toppings: mushroom, olive, pepper. Click Place Order.",
             "mushroom_olive_pepper", html_file="S_E15_multi_select.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="toppings_correct", duration_s=30.0),
        Task("S-E16", TaskCategory.S_STATIC, TaskDifficulty.EASY,
             "Click the button labelled '5'.",
             "5", html_file="S_E16_button_grid.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="button5_correct", duration_s=30.0),
        Task("S-E17", TaskCategory.S_STATIC, TaskDifficulty.EASY,
             "Read the user profile and type the email address into the answer box. Click Submit.",
             "ariel.nguyen@northpine.io", html_file="S_E17_find_email.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="email_correct", duration_s=30.0),
        Task("S-E18", TaskCategory.S_STATIC, TaskDifficulty.EASY,
             "Set the quantity to 4 and click Add to Cart.",
             "4", html_file="S_E18_increment_qty.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="qty_correct", duration_s=30.0),
        Task("S-E19", TaskCategory.S_STATIC, TaskDifficulty.EASY,
             "Read the pricing table and type the name of the cheapest plan into the answer box. Click Submit.",
             "Lite", html_file="S_E19_cheapest.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="cheapest_correct", duration_s=30.0),
        Task("S-E20", TaskCategory.S_STATIC, TaskDifficulty.EASY,
             "Find Brassroot Co. in the founding-year list and type its founding year into the answer box. Click Submit.",
             "2015", html_file="S_E20_match_year.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="year_correct", duration_s=30.0),
        Task("S-E21", TaskCategory.S_STATIC, TaskDifficulty.EASY,
             "Use the dropdown to select 'teal' and click Confirm.",
             "teal", html_file="S_E21_select_color.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="color_correct", duration_s=30.0),
        Task("S-E22", TaskCategory.S_STATIC, TaskDifficulty.EASY,
             "Read the project notice and type the procurement deadline date (e.g. 'June 14') into the answer box. Click Submit.",
             "June 14", html_file="S_E22_find_date.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="date_correct", duration_s=30.0),
        # Medium 1-16 (16 added)
        Task("S-M1", TaskCategory.S_STATIC, TaskDifficulty.MEDIUM,
             "Read the shipping rate table and type the rate (number only) for Standard shipping in zone B. Click Submit.",
             "11", html_file="S_M1_conditional_fill.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="rate_correct", duration_s=60.0),
        Task("S-M2", TaskCategory.S_STATIC, TaskDifficulty.MEDIUM,
             "Fill in the shipping address: 240 Maple Ave, Portland, OR 97205. Click Save Address.",
             "240 Maple Ave Portland OR 97205", html_file="S_M2_address_form.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="address_correct", duration_s=60.0),
        Task("S-M3", TaskCategory.S_STATIC, TaskDifficulty.MEDIUM,
             "Calculate 7.5% sales tax on a $80.00 subtotal and enter the tax amount in dollars. Click Submit.",
             "6.00", html_file="S_M3_calculate_tax.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="tax_correct", duration_s=60.0),
        Task("S-M4", TaskCategory.S_STATIC, TaskDifficulty.MEDIUM,
             "Find the engineer with the longest tenure in the table and type their full name. Click Submit.",
             "Ana Roth", html_file="S_M4_filter_employees.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="employee_correct", duration_s=60.0),
        Task("S-M5", TaskCategory.S_STATIC, TaskDifficulty.MEDIUM,
             "Set a password that meets all stated requirements and click Set Password.",
             "valid_password", html_file="S_M5_password.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="password_valid", duration_s=60.0),
        Task("S-M6", TaskCategory.S_STATIC, TaskDifficulty.MEDIUM,
             "Enable two-factor authentication via the Authenticator app option, then confirm.",
             "app", html_file="S_M6_two_factor.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="twofa_correct", duration_s=60.0),
        Task("S-M7", TaskCategory.S_STATIC, TaskDifficulty.MEDIUM,
             "Compute the invoice subtotal as the sum of qty × unit price across all rows. Type the dollar total. Click Submit.",
             "52.25", html_file="S_M7_invoice_total.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="total_correct", duration_s=60.0),
        Task("S-M8", TaskCategory.S_STATIC, TaskDifficulty.MEDIUM,
             "Apply the requested setting changes (enable Two-Step Login and Marketing Emails; disable Activity Sharing) and click Save Settings.",
             "settings_correct", html_file="S_M8_settings_toggle.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="settings_correct", duration_s=60.0),
        Task("S-M9", TaskCategory.S_STATIC, TaskDifficulty.MEDIUM,
             "Schedule a meeting on 2026-06-15 at 14:30 with attendee mariah@firm.io. Click Schedule.",
             "scheduled", html_file="S_M9_schedule_meeting.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="scheduled_correct", duration_s=60.0),
        Task("S-M10", TaskCategory.S_STATIC, TaskDifficulty.MEDIUM,
             "Apply the refund policy to compute the restocking fee in cents on a $200.00 purchase at day 22. Type the answer and click Submit.",
             "5000", html_file="S_M10_policy_reasoning.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="fee_correct", duration_s=60.0),
        Task("S-M11", TaskCategory.S_STATIC, TaskDifficulty.MEDIUM,
             "Find the reference number in the email transcript and type it (uppercase) into the answer box. Click Submit.",
             "RX-7B4K-29CT", html_file="S_M11_find_code.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="code_correct", duration_s=60.0),
        Task("S-M12", TaskCategory.S_STATIC, TaskDifficulty.MEDIUM,
             "Apply the loan eligibility rules to the given applicant and answer 'yes' or 'no'. Click Submit.",
             "no", html_file="S_M12_eligibility.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="eligibility_correct", duration_s=60.0),
        Task("S-M13", TaskCategory.S_STATIC, TaskDifficulty.MEDIUM,
             "Convert 7.5 kilometers to meters; type the integer answer. Click Submit.",
             "7500", html_file="S_M13_unit_conversion.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="conversion_correct", duration_s=60.0),
        Task("S-M14", TaskCategory.S_STATIC, TaskDifficulty.MEDIUM,
             "Find the highest-scoring player on the leaderboard and type the username. Click Submit.",
             "player_yellow", html_file="S_M14_leaderboard.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="winner_correct", duration_s=60.0),
        Task("S-M15", TaskCategory.S_STATIC, TaskDifficulty.MEDIUM,
             "Check the box next to every stock with closing price greater than $100. Click Submit.",
             "above_threshold", html_file="S_M15_above_threshold.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="threshold_correct", duration_s=60.0),
        Task("S-M16", TaskCategory.S_STATIC, TaskDifficulty.MEDIUM,
             "Match the user query to the most relevant tag, then click Submit.",
             "api-credentials", html_file="S_M16_choose_match.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="match_correct", duration_s=60.0),
        # Hard 1-12 (12 added)
        Task("S-H1", TaskCategory.S_STATIC, TaskDifficulty.HARD,
             "Apply the discount, tax, and shipping fee in order to compute the final invoice amount; type the dollar total. Click Submit.",
             "469.92", html_file="S_H1_invoice_workflow.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="final_correct", duration_s=120.0),
        Task("S-H2", TaskCategory.S_STATIC, TaskDifficulty.HARD,
             "Choose the shortest legal route given today's day and the cargo type. Type the route name. Click Submit.",
             "R-Alpha", html_file="S_H2_dispatch.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="route_correct", duration_s=120.0),
        Task("S-H3", TaskCategory.S_STATIC, TaskDifficulty.HARD,
             "Decode the ROT-13 message and type the plaintext (uppercase). Click Submit.",
             "HELLO WORLD", html_file="S_H3_decode_message.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="decode_correct", duration_s=120.0),
        Task("S-H4", TaskCategory.S_STATIC, TaskDifficulty.HARD,
             "Select a subset of items whose prices sum to exactly $50, then click Submit.",
             "budget_50", html_file="S_H4_budget_select.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="budget_correct", duration_s=120.0),
        Task("S-H5", TaskCategory.S_STATIC, TaskDifficulty.HARD,
             "Compose a reply to the email: To = sender's address, Subject begins with 'Re:', body must mention INV-9821. Click Send Reply.",
             "reply_correct", html_file="S_H5_compose_reply.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="reply_correct", duration_s=120.0),
        Task("S-H6", TaskCategory.S_STATIC, TaskDifficulty.HARD,
             "Identify the next number in the sequence and type it. Click Submit.",
             "42", html_file="S_H6_sequence.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="sequence_correct", duration_s=120.0),
        Task("S-H7", TaskCategory.S_STATIC, TaskDifficulty.HARD,
             "Apply the triage rules and type the ticket ID to address first. Click Submit.",
             "T-004", html_file="S_H7_sort_priority.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="triage_correct", duration_s=120.0),
        Task("S-H8", TaskCategory.S_STATIC, TaskDifficulty.HARD,
             "Convert $250 USD to JPY using the chained rates and type the integer answer. Click Submit.",
             "32660", html_file="S_H8_currency.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="currency_correct", duration_s=120.0),
        Task("S-H9", TaskCategory.S_STATIC, TaskDifficulty.HARD,
             "Identify the target field that maps from the source field 'dob' and type its name. Click Submit.",
             "birthdate", html_file="S_H9_schema_map.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="map_correct", duration_s=120.0),
        Task("S-H10", TaskCategory.S_STATIC, TaskDifficulty.HARD,
             "Compute the median of the given numbers and type the integer answer. Click Submit.",
             "9", html_file="S_H10_median.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="median_correct", duration_s=120.0),
        Task("S-H11", TaskCategory.S_STATIC, TaskDifficulty.HARD,
             "Find the shortest path from A to D in the directed graph and type the total distance. Click Submit.",
             "6", html_file="S_H11_path.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="path_correct", duration_s=120.0),
        Task("S-H12", TaskCategory.S_STATIC, TaskDifficulty.HARD,
             "Read the audit log and type the number of distinct users that performed a delete action. Click Submit.",
             "2", html_file="S_H12_audit.html", eval_type=EvalType.DOM,
             axes=[], dom_success_value="audit_correct", duration_s=120.0),
    ]


# ══════════════════════════════════════════════════════════════════════
# Benchmark entry point
# ══════════════════════════════════════════════════════════════════════

class DynaCUBenchV3:
    """DynaCU-Bench v3: 110 tasks across 11 categories (10 dynamic + 1 static baseline)."""

    def __init__(self, html_tasks_dir: Path = Path("benchmark_env/html_tasks")):
        self.html_tasks_dir = html_tasks_dir
        self.tasks: list[Task] = []
        self._register_all()

    def _register_all(self):
        for cat_fn in [
            _cat_a, _cat_b, _cat_c, _cat_d, _cat_e,
            _cat_f, _cat_g, _cat_h, _cat_i, _cat_j,
            _cat_s,
        ]:
            self.tasks.extend(cat_fn())

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self):
        return len(self.tasks)

    def get_task(self, task_id: str) -> Optional[Task]:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    def filter(
        self,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        eval_type: Optional[str] = None,
        axis: Optional[str] = None,
        requires_audio_out: Optional[bool] = None,
    ) -> list[Task]:
        """Filter tasks by any combination of criteria."""
        result = list(self.tasks)
        if category:
            result = [t for t in result if t.category.value == category or t.task_id.startswith(category)]
        if difficulty:
            result = [t for t in result if t.difficulty.value == difficulty]
        if eval_type:
            result = [t for t in result if t.eval_type.value == eval_type]
        if axis:
            ax = CapabilityAxis(axis)
            result = [t for t in result if ax in t.axes]
        if requires_audio_out is not None:
            result = [t for t in result if t.requires_audio_out == requires_audio_out]
        return result

    def summary(self) -> dict:
        """Return a summary of the benchmark."""
        from collections import Counter
        cats = Counter(t.category.value for t in self.tasks)
        diffs = Counter(t.difficulty.value for t in self.tasks)
        evals = Counter(t.eval_type.value for t in self.tasks)
        axes_count = Counter()
        for t in self.tasks:
            for a in t.axes:
                axes_count[a.value] += 1
        return {
            "total_tasks": len(self.tasks),
            "categories": dict(cats),
            "difficulties": dict(diffs),
            "eval_types": dict(evals),
            "capability_axes": dict(axes_count),
            "audio_output_tasks": sum(1 for t in self.tasks if t.requires_audio_out),
        }
