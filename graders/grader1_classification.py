"""
Grader 1: Email Classification

Evaluates whether the agent correctly classified the email.
Returns a score between 0.0 and 1.0.

Called by the OpenEnv pipeline after Task 1 completes.
"""

from typing import Any, Dict


def grade(episode_state: Dict[str, Any], actions: list) -> float:
    """
    Args:
        episode_state: dict containing at minimum:
            - 'true_classification': the ground truth label
        actions: list of action dicts taken by the agent.
                 Each dict has at least {'stage': str, 'output': str}

    Returns:
        float between 0.0 and 1.0
    """
    true_classification = episode_state.get("true_classification", "").strip().lower()

    # Find the classification stage action
    classification_action = next(
        (a for a in actions if a.get("stage") == "classification"), None
    )

    if classification_action is None:
        return 0.0

    predicted = classification_action.get("output", "").strip().lower()

    return 1.0 if predicted == true_classification else 0.0


# Allow running directly to verify grader works
if __name__ == "__main__":
    # Example: correct prediction
    state = {"true_classification": "spam"}
    actions = [{"stage": "classification", "output": "spam"}]
    print("Correct prediction score:", grade(state, actions))   # expected: 1.0

    # Example: wrong prediction
    actions2 = [{"stage": "classification", "output": "important"}]
    print("Wrong prediction score:", grade(state, actions2))    # expected: 0.0
