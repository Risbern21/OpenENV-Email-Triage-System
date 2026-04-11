"""Deterministic email-triage grader used by `/grader`.

Score breakdown (matches EmailTriageObservation fields):
  label_score     — max 0.60  (classification / intent / reply correctness)
  reasoning_score — max 0.30  (non-empty, keyword-rich justification)
  format_score    — max 0.10  (confidence value in [0.0, 1.0] + action_type present)

Difficulty ceilings applied to total:
  easy   → 0.90
  medium → 0.80
  hard   → 0.70
"""

from __future__ import annotations

from typing import Any

from models import EmailTriageAction
# ---------------------------------------------------------------------------
# You must supply this — import your TASK_REGISTRY from tasks.py
# ---------------------------------------------------------------------------
from tasks import TASK_REGISTRY

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _difficulty_ceiling(difficulty: str) -> float:
    return {"easy": 0.90, "medium": 0.80, "hard": 0.70}.get(difficulty.lower(), 0.90)


def _normalize_label(raw: Any, fallback: str = "") -> str:
    return str(raw or fallback).strip().lower()


# ---------------------------------------------------------------------------
# Per-component scorers
# ---------------------------------------------------------------------------


def _score_label(action: EmailTriageAction, task: dict[str, Any]) -> float:
    """
    0.60 points for a correct label at the current stage.

    The TASK_REGISTRY entry is expected to carry ground-truth keys:
      true_classification, true_intent, true_reply
    OR a 'data_corpus' list containing dictionaries with these keys.
    """
    stage_to_truth_key = {
        "classification": "expected_classification",
        "intent": "expected_intent",
        "reply": "expected_reply",
    }

    # Infer current stage from action_type
    stage = _normalize_label(action.action_type)
    truth_key = stage_to_truth_key.get(stage)
    if not truth_key:
        return 0.0

    predicted = _normalize_label(action.content)
    
    # Get all possible truths (from root or corpus)
    possible_truths = []
    if truth_key in task:
        possible_truths.append(_normalize_label(task[truth_key]))
    
    if "data_corpus" in task:
        for item in task["data_corpus"]:
            if truth_key in item:
                possible_truths.append(_normalize_label(item[truth_key]))

    if not possible_truths:
        # No ground truth available — award partial credit
        return 0.30

    if predicted in possible_truths:
        return 0.60

    # Partial credit: predicted is a substring of any truth or vice-versa
    for truth in possible_truths:
        if predicted and (predicted in truth or truth in predicted):
            return 0.30

    return 0.0


def _score_reasoning(action: EmailTriageAction) -> float:
    """
    0.30 points for a non-trivial reasoning string.

    Heuristic tiers:
      0.30 — reasoning is ≥ 20 chars and references at least one signal word
      0.15 — reasoning is present but thin (< 20 chars or no signal words)
      0.00 — empty
    """
    SIGNAL_WORDS = {
        # classification signals
        "spam",
        "important",
        "support",
        "promotional",
        "newsletter",
        # intent signals
        "pricing",
        "complaint",
        "booking",
        "inquiry",
        "refund",
        "feedback",
        # reply signals
        "apologize",
        "confirm",
        "provide",
        "assist",
        "resolve",
        # generic quality signals
        "because",
        "therefore",
        "indicates",
        "suggests",
        "based on",
        "evidence",
        "confidence",
        "pattern",
        "keyword",
    }

    text = str(action.reasoning or "").strip()
    if not text:
        return 0.0

    has_signal = any(w in text.lower() for w in SIGNAL_WORDS)

    if len(text) >= 20 and has_signal:
        return 0.30
    return 0.15


def _score_format(action: EmailTriageAction) -> float:
    """
    0.10 points for well-formed metadata:
      0.05 — confidence is a float in (0, 1) exclusive
      0.05 — action_type is one of the known valid values
    """
    VALID_ACTION_TYPES = {"classification", "intent", "reply"}

    score = 0.0

    try:
        conf = float(action.confidence)
        if 0.0 < conf < 1.0:
            score += 0.05
    except (TypeError, ValueError):
        pass

    if str(action.action_type or "").strip().lower() in VALID_ACTION_TYPES:
        score += 0.05

    return score


# ---------------------------------------------------------------------------
# Public entry point (called by app.py)
# ---------------------------------------------------------------------------


def grade(action: EmailTriageAction, task_id: str) -> float:
    """
    Returns a float in [0.0, 1.0].
    Returns 0.0 for unknown task_ids or on any unexpected error.
    """
    if task_id not in TASK_REGISTRY:
        return 0.0

    try:
        task = TASK_REGISTRY[task_id]
        difficulty = str(task.get("difficulty", "easy")).lower()

        label_score = _score_label(action, task)  # max 0.60
        reasoning_score = _score_reasoning(action)  # max 0.30
        format_score = _score_format(action)  # max 0.10

        raw_total = label_score + reasoning_score + format_score
        ceiling = _difficulty_ceiling(difficulty)
        total = _clamp01(raw_total) * ceiling

        return round(total, 4)

    except Exception:
        return 0.0

def grade_report(action: EmailTriageAction, task_id: str) -> dict[str, Any]:
    """
    Returns a detailed report including component scores and metadata.
    Matches the structure expected by the OpenEnv validator for grading endpoints.
    """
    if task_id not in TASK_REGISTRY:
        return {
            "task_id": task_id,
            "score": 0.0,
            "error": f"Unknown task_id: {task_id}"
        }

    task = TASK_REGISTRY[task_id]
    difficulty = str(task.get("difficulty", "easy")).lower()

    label_score = _score_label(action, task)
    reasoning_score = _score_reasoning(action)
    format_score = _score_format(action)
    score = grade(action=action, task_id=task_id)

    return {
        "task_id": task_id,
        "score": score,
        "passed": 1 if score > 0.5 else 0,
        "total": 1,
        "metric": "email_triage_env",
        "label_score": round(label_score, 4),
        "reasoning_score": round(reasoning_score, 4),
        "format_score": round(format_score, 4),
        "difficulty": difficulty,
        "ceiling": _difficulty_ceiling(difficulty),
    }
