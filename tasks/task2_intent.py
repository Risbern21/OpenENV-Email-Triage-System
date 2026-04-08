"""
Task 2: Email Intent Detection
The agent must identify the intent of an email:
e.g., 'pricing inquiry', 'complaint', 'booking', 'promotion'
"""

TASK_ID = "task2_intent"
TASK_NAME = "Email Intent Detection"
TASK_DESCRIPTION = "Identify the intent of the email: pricing inquiry, complaint, booking, or promotion"

SEED = 42

TASK_CONFIG = {
    "task_id": TASK_ID,
    "seed": SEED,
    "stage": "intent",
    "expected_stage_sequence": ["classification", "intent"],
    "description": TASK_DESCRIPTION,
}
