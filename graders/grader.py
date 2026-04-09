"""
grader.py — Email Triage Grader

Adapted from the clip-labelling grader to work with EmailTriageEnvironment.
Grades agent actions across the three sequential stages:
  classification → intent → reply

Reward decomposition per step
──────────────────────────────
  format_score    (max 0.10)  valid fields present and well-formed
  label_score     (max 0.60)  correctness with per-(episode, stage) noise
  reasoning_score (max 0.30)  quality of the agent's reasoning text

Per-step difficulty ceiling
────────────────────────────
  easy   → 0.90
  medium → 0.80
  hard   → 0.70

Noise
──────
Deterministic noise seeded by hash(episode_id + stage + expected_answer)
prevents the agent from using label_score >= threshold as a binary oracle.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Union

from models import EmailTriageAction, EmailTriageState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CLASSIFICATIONS = {
    "spam",
    "ham",
    "promotional",
    "urgent",
    "newsletter",
    "important",
    "support",
}
VALID_INTENTS = {
    "complaint",
    "inquiry",
    "feedback",
    "request",
    "notification",
    "phishing",
    "advertisement",
    "support",
    "unsubscribe",
    "other",
    "pricing",
    "booking",
}

DIFFICULTY_STEP_CEILING: dict[str, float] = {
    "easy": 0.90,
    "medium": 0.80,
    "hard": 0.70,
}

NOISE_AMPLITUDE = 0.08

# Snake_case tokens considered legitimate reasoning vocabulary (not hallucinations).
VALID_REASONING_TOKENS = {
    "subject_line",
    "sender_domain",
    "reply_to",
    "call_to_action",
    "urgency_cue",
    "personal_greeting",
    "unsubscribe_link",
    "spam_trigger",
    "html_only",
    "plain_text",
    "mailing_list",
    "return_path",
    "action_type",  # field name from EmailTriageAction
}

FEATURE_TOKEN_RE = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")

# Keywords the agent should reference in reasoning, by stage.
STAGE_FEATURES: dict[str, list[str]] = {
    "classification": [
        "subject",
        "sender",
        "domain",
        "link",
        "attachment",
        "greeting",
        "urgency",
        "promotion",
        "unsubscribe",
    ],
    "intent": [
        "tone",
        "request",
        "complaint",
        "question",
        "action",
        "sentiment",
        "purpose",
        "goal",
        "directive",
    ],
    "reply": [
        "acknowledge",
        "resolution",
        "apology",
        "follow",
        "confirm",
        "address",
        "solution",
        "escalate",
        "clarif",
        "next step",
    ],
}

POSITIVE_WORDS = frozenset(
    [
        "clear",
        "obvious",
        "definite",
        "strong",
        "high",
        "evident",
        "explicit",
        "confirms",
        "indicates",
        "matches",
        "typical",
    ]
)
NEGATIVE_WORDS = frozenset(
    [
        "suspicious",
        "missing",
        "absent",
        "weak",
        "unusual",
        "inconsistent",
        "lacks",
        "spam",
        "phish",
        "mislead",
    ]
)
MIXED_WORDS = frozenset(
    [
        "borderline",
        "ambiguous",
        "unclear",
        "mixed",
        "could be",
        "possibly",
        "uncertain",
        "partial",
        "tradeoff",
        "conflicting",
    ]
)


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------


@dataclass
class Reward:
    total: float
    format_score: float
    label_score: float
    reasoning_score: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(action: Union[EmailTriageAction, dict[str, Any]]) -> dict[str, Any]:
    """Coerce an EmailTriageAction or raw dict into a plain dict."""
    if isinstance(action, EmailTriageAction):
        return {
            "action_type": action.action_type,
            "content": action.content,
            "reasoning": action.reasoning,
            "confidence": action.confidence,
        }
    return {
        "action_type": str(action.get("action_type", "")),
        "content": str(action.get("content", "")).strip(),
        "reasoning": str(action.get("reasoning", "")).strip(),
        "confidence": float(action.get("confidence", 0.5)),
    }


def _label_noise(episode_id: str, stage: str, expected: str) -> float:
    """Deterministic noise in [-NOISE_AMPLITUDE, +NOISE_AMPLITUDE]."""
    key = f"{episode_id}::{stage}::{expected}".encode()
    digest = hashlib.sha256(key).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFF_FFFF
    return NOISE_AMPLITUDE * (2.0 * frac - 1.0)


def _partial_base(difficulty: str | None) -> float:
    d = (difficulty or "easy").lower()
    if d == "hard":
        return 0.15
    if d == "medium":
        return 0.20
    return 0.25


# ---------------------------------------------------------------------------
# Format score
# ---------------------------------------------------------------------------


def _score_format(action: dict[str, Any], stage: str) -> float:
    """0.10 when the action is structurally valid; 0.0 otherwise.

    Checks:
      - action_type matches the current stage
      - content is non-empty
      - reasoning is non-empty
      - confidence is a float in [0.0, 1.0]
      - content value is within the stage vocabulary (classification / intent)
    """
    content = str(action.get("content", "")).strip()
    reasoning = str(action.get("reasoning", "")).strip()
    action_type = str(action.get("action_type", "")).strip()
    confidence = action.get("confidence", -1.0)

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return 0.0

    if not content or not reasoning:
        return 0.0
    if not (0.0 <= confidence <= 1.0):
        return 0.0
    if action_type != stage:
        return 0.0
    if stage == "classification" and content.lower() not in VALID_CLASSIFICATIONS:
        return 0.0
    if stage == "intent" and content.lower() not in VALID_INTENTS:
        return 0.0

    return 0.10


# ---------------------------------------------------------------------------
# Label score
# ---------------------------------------------------------------------------


def _score_label(
    action: dict[str, Any],
    stage: str,
    state: EmailTriageState,
    difficulty: str | None = None,
) -> float:
    """Score correctness with deterministic per-(episode, stage) noise.

    classification / intent
    ───────────────────────
      Correct → max(0.40, 0.60 + noise)
      Wrong   → 0.00

    reply
    ─────
      Strong match (substring either direction) → max(0.40, 0.60 + noise)
      Partial match (shared keywords > 3 chars) → partial_base + noise * 0.5
      No match                                  → 0.00
    """
    content = str(action.get("content", "")).strip().lower()
    episode_id = str(state.episode_id or "")

    if stage == "classification":
        expected = state.true_classification.strip().lower()
        noise = _label_noise(episode_id, stage, expected)
        return max(0.40, 0.60 + noise) if content == expected else 0.0

    elif stage == "intent":
        expected = state.true_intent.strip().lower()
        noise = _label_noise(episode_id, stage, expected)
        return max(0.40, 0.60 + noise) if content == expected else 0.0

    elif stage == "reply":
        expected = state.true_reply.strip().lower()
        noise = _label_noise(episode_id, stage, expected)
        if content in expected or expected in content:
            return max(0.40, 0.60 + noise)
        pred_words = {w for w in content.split() if len(w) > 3}
        expt_words = {w for w in expected.split() if len(w) > 3}
        if pred_words & expt_words:
            return max(0.0, _partial_base(difficulty) + noise * 0.5)
        return 0.0

    return 0.0


# ---------------------------------------------------------------------------
# Reasoning score
# ---------------------------------------------------------------------------


def _infer_polarity(stage: str, content: str) -> str:
    c = content.lower()
    if stage == "classification":
        return (
            "negative"
            if c in ("spam", "promotional")
            else "positive" if c in ("ham", "important") else "mixed"
        )
    if stage == "intent":
        return (
            "negative"
            if c in ("complaint", "phishing")
            else "mixed" if c in ("feedback", "notification") else "positive"
        )
    return "mixed"


def _contains_directional_cue(reasoning: str, feature: str, polarity: str) -> bool:
    text = reasoning.lower()
    if feature.lower() not in text:
        return False
    words = set(re.split(r"\W+", text))
    if polarity == "positive":
        return bool(POSITIVE_WORDS & words)
    if polarity == "negative":
        return bool(NEGATIVE_WORDS & words)
    return bool(MIXED_WORDS & set(text.split()))


def _score_reasoning(
    action: dict[str, Any],
    stage: str,
    difficulty: str | None = None,
) -> float:
    """Score reasoning quality with difficulty-adjusted thresholds.

    Sub-score 1 — Feature mentions (max 0.10)
      ≥ 2 stage-relevant features → 0.10
      1 feature                   → 0.04 (easy) | 0.03 (medium) | 0.00 (hard)

    Sub-score 2 — Directional cues (max 0.10)
      easy   : ≥ 1 match → 0.10; else >50 chars → 0.03
      medium : ≥ 1 match required → 0.10
      hard   : ≥ 2 matches → 0.10; 1 match → 0.03

    Sub-score 3 — Hallucination + quality (max 0.10)
      easy/medium : 0 hallucinated tokens → 0.10; 1 → 0.04
      hard        : 0 tokens AND >50 chars → 0.10; 0 tokens only → 0.04
    """
    reasoning = str(action.get("reasoning", "")).strip()
    content = str(action.get("content", "")).strip()
    diff = (difficulty or "easy").lower()
    lower = reasoning.lower()
    score = 0.0

    features = STAGE_FEATURES.get(stage, [])
    polarity = _infer_polarity(stage, content)

    # ── Sub-score 1: Feature mentions ─────────────────────────────────────
    mentioned = sum(1 for kw in features if kw in lower)
    if mentioned >= 2:
        score += 0.10
    elif mentioned == 1:
        score += {"hard": 0.00, "medium": 0.03}.get(diff, 0.04)

    # ── Sub-score 2: Directional cues ─────────────────────────────────────
    dir_matches = sum(
        1
        for kw in features
        if kw in lower and _contains_directional_cue(reasoning, kw, polarity)
    )

    if diff == "hard":
        score += 0.10 if dir_matches >= 2 else (0.03 if dir_matches == 1 else 0.0)
    elif diff == "medium":
        score += 0.10 if dir_matches >= 1 else 0.0
    else:
        score += 0.10 if dir_matches >= 1 else (0.03 if len(reasoning) > 50 else 0.0)

    # ── Sub-score 3: Hallucination + quality ──────────────────────────────
    hallucinated = [
        tok
        for tok in FEATURE_TOKEN_RE.findall(lower)
        if tok not in VALID_REASONING_TOKENS
    ]

    if diff == "hard":
        score += (
            0.10
            if (not hallucinated and len(reasoning) > 50)
            else (0.04 if not hallucinated else 0.0)
        )
    else:
        score += 0.10 if not hallucinated else (0.04 if len(hallucinated) == 1 else 0.0)

    return min(max(score, 0.0), 0.30)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def grade(
    action: Union[EmailTriageAction, dict[str, Any]],
    stage: str,
    state: EmailTriageState,
    difficulty: str | None = None,
) -> Reward:
    """Grade a single agent action within an EmailTriageEnvironment step.

    Parameters
    ──────────
    action     : EmailTriageAction instance or equivalent dict
    stage      : "classification" | "intent" | "reply"
    state      : EmailTriageState (ground-truth lookup + noise seeding)
    difficulty : "easy" | "medium" | "hard"  (falls back to state.difficulty)

    Returns
    ───────
    Reward(total, format_score, label_score, reasoning_score)
    """
    diff = difficulty or getattr(state, "difficulty", "easy")
    payload = _normalize(action)

    format_score = _score_format(payload, stage)
    label_score = _score_label(payload, stage, state, difficulty=diff)
    reasoning_score = _score_reasoning(payload, stage, difficulty=diff)

    raw_total = format_score + label_score + reasoning_score
    ceiling = DIFFICULTY_STEP_CEILING.get(diff.lower(), 1.0)
    total = min(raw_total, ceiling)

    return Reward(
        total=round(min(max(total, 0.0), 1.0), 6),
        format_score=round(format_score, 6),
        label_score=round(label_score, 6),
        reasoning_score=round(reasoning_score, 6),
    )


def score(
    action: Union[EmailTriageAction, dict[str, Any]],
    stage: str,
    state: EmailTriageState,
    difficulty: str | None = None,
) -> float:
    """Convenience wrapper — returns only the scalar total reward."""
    return float(grade(action, stage, state, difficulty=difficulty).total)
