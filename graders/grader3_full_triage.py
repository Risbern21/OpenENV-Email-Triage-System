"""
Grader 3: Full Email Triage Pipeline

Evaluates the complete triage: classification + intent + reply.
Each sub-score uses the same logic as your existing grade_* methods.
Final score = weighted average of all three.

Returns a score between 0.0 and 1.0.
"""

from typing import Any, Dict


def _grade_classification(predicted: str, expected: str) -> float:
    return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0


def _grade_intent(predicted: str, expected: str) -> float:
    predicted = predicted.strip().lower()
    expected = expected.strip().lower()
    if predicted == expected:
        return 1.0
    elif predicted in expected or expected in predicted:
        return 0.7
    return 0.0


def _grade_reply(response: str) -> float:
    score = 0.0
    response = response.lower()

    if len(response) > 30:
        score += 0.3

    if any(word in response for word in ["thank", "regards", "assist", "help"]):
        score += 0.3

    if any(word in response for word in ["issue", "request", "order", "support"]):
        score += 0.4

    return min(score, 1.0)


def grade(episode_state: Dict[str, Any], actions: list) -> float:
    """
    Args:
        episode_state: dict with keys:
            - 'true_classification': ground truth label
            - 'true_intent': ground truth intent
        actions: list of {'stage': str, 'output': str} dicts

    Returns:
        float between 0.0 and 1.0 (weighted average of all three stages)
    """
    true_classification = episode_state.get("true_classification", "")
    true_intent = episode_state.get("true_intent", "")

    action_map = {a["stage"]: a["output"] for a in actions if "stage" in a}

    # Score each stage (0.0 if stage was skipped)
    classification_score = _grade_classification(
        action_map.get("classification", ""), true_classification
    )
    intent_score = _grade_intent(
        action_map.get("intent", ""), true_intent
    )
    reply_score = _grade_reply(
        action_map.get("reply", "")
    )

    # Weighted average: classification 30%, intent 30%, reply 40%
    final_score = (
        0.30 * classification_score +
        0.30 * intent_score +
        0.40 * reply_score
    )

    return round(final_score, 4)


if __name__ == "__main__":
    state = {
        "true_classification": "support",
        "true_intent": "complaint",
    }
    actions = [
        {"stage": "classification", "output": "support"},
        {"stage": "intent", "output": "complaint"},
        {"stage": "reply", "output": "We will help you resolve this issue and assist you shortly."},
    ]
    print("Full pipeline score:", grade(state, actions))  # should be close to 1.0
