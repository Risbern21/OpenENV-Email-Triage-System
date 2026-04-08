from typing import List, Optional, Dict, Any
from pydantic import BaseModel

# These would normally go in models.py

class EmailTriageAction(BaseModel):
    """
    Agent performs an action depending on stage:
    - classification → spam / important / support
    - intent → pricing / complaint / booking / etc
    - reply → generated text
    """
    action_type: str   # "classification" | "intent" | "reply"
    content: str       
    metadata: Dict[str, Any] = {}



class EmailTriageObservation(BaseModel):
    """
    What the agent sees after each step
    """
    done: bool
    reward: float

    email_text: str
    current_stage: str   # classification → intent → reply
    history: List[Dict[str, str]]  # past actions

    message: str   # feedback
    metadata: Dict[str, Any] = {}



class EmailTriageState(BaseModel):
    """
    Internal environment state
    """
    episode_id: Optional[str] = None
    step_count: int = 0

    email_text: str = ""
    true_classification: str = ""
    true_intent: str = ""
    true_reply: str = ""

    current_stage: str = "classification"


print("Types defined: EmailTriageAction, EmailTriageObservation, EmailTriageState")