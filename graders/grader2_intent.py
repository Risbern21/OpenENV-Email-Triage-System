"""
Grader 2: Email Intent Detection

Evaluates whether the agent correctly identified the email's intent.
Uses partial-match scoring (same logic as your existing grade_intent method).
Returns a score between 0.0 and 1.0.
"""

from typing import Any, Dict


def grade(episode_state: Dict[str, Any], actions: list) -> float:
    """
    Args:
        episode_state: dict containing at minimum:
            - 'true_intent': the ground truth intent string
        actions: list of action dicts taken by the agent.
                 Each dict has at least {'stage': str, 'output': str}

    Returns:
        float between 0.0 and 1.0
          1.0  → exact match
          0.7  → partial match (one contains the other)
          0.0  → no match
    """
    true_intent = episode_state.get("true_intent", "").strip().lower()

    intent_action = next(
        (a for a in actions if a.get("stage") == "intent"), None
    )

    if intent_action is None:
        return 0.0

    predicted = intent_action.get("output", "").strip().lower()

    if predicted == true_intent:
        return 1.0
    elif predicted in true_intent or true_intent in predicted:
        return 0.7
    else:
        return 0.0


if __name__ == "__main__":
    state = {"true_intent": "pricing inquiry"}

    # Exact match
    actions = [{"stage": "intent", "output": "pricing inquiry"}]
    print("Exact match:", grade(state, actions))       # 1.0

    # Partial match
    actions2 = [{"stage": "intent", "output": "pricing"}]
    print("Partial match:", grade(state, actions2))    # 0.7

    # No match
    actions3 = [{"stage": "intent", "output": "complaint"}]
    print("No match:", grade(state, actions3))         # 0.0
