from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskInfo(BaseModel):
    """Returned by /tasks endpoint."""

    task_id: str
    difficulty: str
    description: str
    action_schema: dict
    grader: str


class EmailTriageAction(BaseModel):
    """
    Agent performs an action depending on stage:
    - classification → spam / important / support
    - intent         → pricing / complaint / booking / etc
    - reply          → generated text

    reasoning and confidence are used by the grader to compute
    reasoning_score and format_score respectively.
    """

    action_type: str  # "classification" | "intent" | "reply"
    content: str
    reasoning: str = ""  # grader uses this for reasoning_score
    confidence: float = 0.5  # grader uses this for format_score; must be in [0.0, 1.0]
    metadata: Dict[str, Any] = {}


class EmailTriageObservation(BaseModel):
    """
    What the agent sees after each step.
    Grader breakdown fields are populated on every graded step.
    """

    done: bool
    reward: float

    email_text: str
    current_stage: str  # classification → intent → reply
    history: List[Dict[str, str]]  # past actions

    message: str  # feedback
    metadata: Dict[str, Any] = {}

    # ── Grader breakdown (populated after each step) ──────────────────────
    format_score: float = 0.0  # max 0.10
    label_score: float = 0.0  # max 0.60
    reasoning_score: float = 0.0  # max 0.30


class EmailTriageState(BaseModel):
    """
    Internal environment state.
    difficulty drives grader ceilings: easy=0.90 | medium=0.80 | hard=0.70
    """

    episode_id: Optional[str] = None
    step_count: int = 0

    task_id: str

    email_text: str = ""
    true_classification: str = ""
    true_intent: str = ""
    true_reply: str = ""

    current_stage: str = "classification"
    difficulty: str = "easy"  # "easy" | "medium" | "hard"


print("Types defined: EmailTriageAction, EmailTriageObservation, EmailTriageState")
