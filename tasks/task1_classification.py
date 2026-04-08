"""
Task 1: Email Classification
The agent must correctly classify an email as 'spam', 'important', or 'support'.
"""

TASK_ID = "task1_classification"
TASK_NAME = "Email Classification"
TASK_DESCRIPTION = "Classify the given email into one of: spam, important, support"

# Fixed seed so the pipeline always gets the same email for reproducibility
SEED = 42

TASK_CONFIG = {
    "task_id": TASK_ID,
    "seed": SEED,
    "stage": "classification",
    "expected_stage_sequence": ["classification"],
    "description": TASK_DESCRIPTION,
}
