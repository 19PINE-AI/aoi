"""
LLM-based Evaluator for DynaCU-Bench v3.

For tasks with eval_type LLM or HYBRID, a judge model evaluates the
agent's response against a rubric.  DOM-based checks are handled
directly by the harness; this module only handles the "soft" evaluation.

Three evaluation modes:
  dom    — window.getTaskResult() == expected value  (handled by harness)
  llm    — LLM judge scores agent output against rubric
  hybrid — DOM gate (did agent act?) AND LLM quality (was the answer good?)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from dynacubench.tasks_v3 import Task, EvalType, LLMRubric

logger = logging.getLogger(__name__)


@dataclass
class LLMJudgeResult:
    """Result from the LLM judge."""
    score: float          # 0.0 - 1.0
    reason: str           # Brief explanation
    raw_response: str     # Full judge model output
    model_used: str       # Which judge model was used
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.score >= 0.5


@dataclass
class EvalOutcome:
    """Full evaluation outcome combining DOM and LLM checks."""
    task_id: str
    eval_type: str          # dom / llm / hybrid
    dom_passed: Optional[bool] = None
    dom_result: Optional[str] = None
    llm_result: Optional[LLMJudgeResult] = None
    final_score: float = 0.0
    final_passed: bool = False

    def to_dict(self) -> dict:
        d = {
            "task_id": self.task_id,
            "eval_type": self.eval_type,
            "final_score": self.final_score,
            "final_passed": self.final_passed,
        }
        if self.dom_passed is not None:
            d["dom_passed"] = self.dom_passed
            d["dom_result"] = self.dom_result
        if self.llm_result is not None:
            d["llm_score"] = self.llm_result.score
            d["llm_reason"] = self.llm_result.reason
            d["llm_model"] = self.llm_result.model_used
        return d


class LLMEvaluator:
    """
    LLM-based evaluator that judges agent responses against task rubrics.

    Supports multiple backend models for judging. Default: Gemini 2.0 Flash
    (cheapest, fastest for evaluation). Falls back to GPT-4o-mini.
    """

    def __init__(self, judge_model: str = "gemini-2.0-flash"):
        self.judge_model = judge_model
        self._client = None

    def _init_client(self):
        if self._client is not None:
            return

        if "gemini" in self.judge_model:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
            self._client = genai.GenerativeModel(self.judge_model)
            logger.info("LLM Evaluator: Gemini judge initialized (%s)", self.judge_model)

        elif "gpt" in self.judge_model:
            import openai
            self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            logger.info("LLM Evaluator: OpenAI judge initialized (%s)", self.judge_model)

        elif "claude" in self.judge_model:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
            logger.info("LLM Evaluator: Anthropic judge initialized (%s)", self.judge_model)

        else:
            raise ValueError(f"Unknown judge model: {self.judge_model}")

    def judge(self, rubric: LLMRubric, agent_response: str) -> LLMJudgeResult:
        """
        Run the LLM judge on an agent's response.

        Args:
            rubric: The evaluation rubric for this task
            agent_response: What the agent typed/spoke/submitted

        Returns:
            LLMJudgeResult with score and reasoning
        """
        if not agent_response or not agent_response.strip():
            return LLMJudgeResult(
                score=0.0,
                reason="Agent produced no response.",
                raw_response="",
                model_used=self.judge_model,
            )

        self._init_client()
        prompt = rubric.to_judge_prompt(agent_response)

        try:
            raw = self._call_model(prompt)
            score, reason = self._parse_judge_response(raw)
            return LLMJudgeResult(
                score=score,
                reason=reason,
                raw_response=raw,
                model_used=self.judge_model,
            )
        except Exception as e:
            logger.error("LLM judge failed: %s", e)
            return LLMJudgeResult(
                score=0.0,
                reason="Judge call failed",
                raw_response="",
                model_used=self.judge_model,
                error=str(e),
            )

    def _call_model(self, prompt: str) -> str:
        """Call the judge model and return raw text response."""
        if "gemini" in self.judge_model:
            response = self._client.generate_content(prompt)
            return response.text

        elif "gpt" in self.judge_model:
            response = self._client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0,
            )
            return response.choices[0].message.content

        elif "claude" in self.judge_model:
            response = self._client.messages.create(
                model=self.judge_model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

    @staticmethod
    def _parse_judge_response(raw: str) -> tuple[float, str]:
        """Parse JSON response from judge: {"score": 0.8, "reason": "..."}"""
        # Try to extract JSON from the response
        json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                score = float(data.get("score", 0))
                reason = data.get("reason", "No reason given")
                return max(0.0, min(1.0, score)), reason
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: look for a number
        num_match = re.search(r'(\d+\.?\d*)', raw)
        if num_match:
            score = float(num_match.group(1))
            if score > 1:
                score = score / 100.0  # Handle percentage-style responses
            return max(0.0, min(1.0, score)), raw[:200]

        return 0.0, f"Could not parse judge response: {raw[:200]}"

    def evaluate_task(
        self,
        task: Task,
        dom_result: Optional[str] = None,
        agent_response: Optional[str] = None,
    ) -> EvalOutcome:
        """
        Full evaluation for a task, combining DOM and LLM checks as needed.

        Args:
            task: The task definition
            dom_result: Value from window.getTaskResult()
            agent_response: Text the agent typed/spoke (for LLM evaluation)

        Returns:
            EvalOutcome with final score and pass/fail
        """
        outcome = EvalOutcome(
            task_id=task.task_id,
            eval_type=task.eval_type.value,
        )

        # ── DOM evaluation ──────────────────────────────────────────
        if task.eval_type in (EvalType.DOM, EvalType.HYBRID):
            outcome.dom_result = dom_result
            if task.dom_success_value:
                outcome.dom_passed = (dom_result == task.dom_success_value)
            else:
                # Generic: not pending/error
                FAIL_VALS = {"pending", "unknown", "error", "timeout"}
                outcome.dom_passed = (
                    dom_result is not None
                    and dom_result not in FAIL_VALS
                    and not dom_result.startswith("wrong_")
                )

        # ── LLM evaluation ─────────────────────────────────────────
        if task.eval_type in (EvalType.LLM, EvalType.HYBRID) and task.llm_rubric:
            response_text = agent_response or ""
            outcome.llm_result = self.judge(task.llm_rubric, response_text)

        # ── Compute final score ─────────────────────────────────────
        if task.eval_type == EvalType.DOM:
            outcome.final_score = 1.0 if outcome.dom_passed else 0.0
            outcome.final_passed = outcome.dom_passed or False

        elif task.eval_type == EvalType.LLM:
            if outcome.llm_result:
                outcome.final_score = outcome.llm_result.score
                outcome.final_passed = outcome.llm_result.passed
            else:
                outcome.final_score = 0.0
                outcome.final_passed = False

        elif task.eval_type == EvalType.HYBRID:
            # Both must pass: DOM gate AND LLM quality
            dom_ok = outcome.dom_passed or False
            llm_score = outcome.llm_result.score if outcome.llm_result else 0.0

            if dom_ok:
                # DOM passed — final score is the LLM quality score
                outcome.final_score = llm_score
                outcome.final_passed = llm_score >= 0.5
            else:
                # DOM failed — agent didn't even complete the basic action
                outcome.final_score = 0.0
                outcome.final_passed = False

        return outcome
