"""
Task 3: Full Email Triage Pipeline
The agent must complete all three stages: classify → detect intent → draft reply.
All three stages are scored and averaged for a final score.
"""

TASK_ID = "task3_full_triage"
TASK_NAME = "Full Email Triage Pipeline"
TASK_DESCRIPTION = (
    "Complete the full triage pipeline: classify the email, detect its intent, "
    "and draft an appropriate reply. All stages are scored."
)

SEED = 42

TASK_CONFIG = {
    "task_id": TASK_ID,
    "seed": SEED,
    "stage": "full_pipeline",
    "expected_stage_sequence": ["classification", "intent", "reply"],
    "description": TASK_DESCRIPTION,
}
