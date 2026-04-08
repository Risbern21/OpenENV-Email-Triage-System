import random
import uuid
from typing import Dict, List, Any
from models import EmailTriageState, EmailTriageObservation, EmailTriageAction

#  GLOBAL episode storage (survives across requests)
_episodes: Dict[str, EmailTriageState] = {}
_histories: Dict[str, List[Dict]] = {}

class EmailTriageEnvironment:
    EMAILS = [
        {
            "email": "Hi, I would like to know the pricing for your premium plan.",
            "classification": "important",
            "intent": "pricing inquiry",
            "reply": "Thank you for your interest. We will share pricing details shortly."
        },
        {
            "email": "My order hasn’t arrived yet. It’s been over a week.",
            "classification": "support",
            "intent": "complaint",
            "reply": "We’re sorry for the delay. Our support team will resolve this soon."
        },
        {
            "email": "Can I book a demo for your product next week?",
            "classification": "important",
            "intent": "booking",
            "reply": "Sure, we will schedule a demo and confirm shortly."
        },
        {
            "email": "Congratulations! You’ve won a free iPhone. Click here now!",
            "classification": "spam",
            "intent": "promotion",
            "reply": "This appears to be spam. Please avoid clicking suspicious links."
        },
        {
            "email": "The app keeps crashing when I try to open it.",
            "classification": "support",
            "intent": "complaint",
            "reply": "We're sorry for the issue. Our technical team will assist you shortly."
        },
        {
            "email": "Do you offer discounts for students?",
            "classification": "important",
            "intent": "pricing inquiry",
            "reply": "We will share information about available discounts soon."
        },
        {
            "email": "Please schedule a meeting for project discussion.",
            "classification": "important",
            "intent": "booking",
            "reply": "We will arrange a meeting and confirm the schedule shortly."
        },
        {
            "email": "Limited time offer! Get 90% off on all products!",
            "classification": "spam",
            "intent": "promotion",
            "reply": "This is likely spam. Please ignore such messages."
        },
        {
            "email": "I forgot my password and cannot log in.",
            "classification": "support",
            "intent": "complaint",
            "reply": "We will help you reset your password shortly."
        },
        {
            "email": "Is there a free trial available for your service?",
            "classification": "important",
            "intent": "pricing inquiry",
            "reply": "Yes, we will share details about the free trial shortly."
        },
        {
            "email": "Book a slot for consultation tomorrow.",
            "classification": "important",
            "intent": "booking",
            "reply": "We will confirm your consultation slot soon."
        },
        {
            "email": "Why was my payment declined? Please help.",
            "classification": "support",
            "intent": "complaint",
            "reply": "We’re sorry for the inconvenience. Our team will assist you shortly."
        }
    ]

    def __init__(self):
        self.episode_id = None

    def reset(self, seed=None, episode_id=None) -> EmailTriageObservation:
        global _episodes, _histories
        
        if seed is not None:
            random.seed(seed)
        
        episode_id = episode_id or str(uuid.uuid4())
        email = random.choice(self.EMAILS)
        
        
        _episodes[episode_id] = EmailTriageState(
            episode_id=episode_id, step_count=0,
            email_text=email["email"],
            true_classification=email["classification"],
            true_intent=email["intent"],
            true_reply=email["reply"],
            current_stage="classification"
        )
        _histories[episode_id] = []
        
        self.episode_id = episode_id
        
        state = _episodes[episode_id]
        return EmailTriageObservation(
            done=False, reward=0.0,
            email_text=state.email_text,
            current_stage=state.current_stage,
            history=[],
            message="Start with classification",
            metadata={"episode_id": episode_id}
        )

    def step(self, action: EmailTriageAction) -> EmailTriageObservation:
        global _episodes, _histories
        
       
        episode_id = self.episode_id or list(_episodes.keys())[-1] if _episodes else None
        
        if not episode_id or episode_id not in _episodes:
            return EmailTriageObservation(
                done=True, reward=0.0, email_text="", current_stage="error",
                history=[], message="No active episode", metadata={}
            )
        
        state = _episodes[episode_id]
        history = _histories[episode_id]
        
        state.step_count += 1
        stage = state.current_stage
        reward, message = 0.0, ""
        
        # Classification
        if stage == "classification":
            if action.content == state.true_classification:
                reward, message = 0.3, "Correct classification"
            else:
                message = "Wrong classification"
            history.append({"stage": stage, "output": action.content})
            state.current_stage = "intent"
            
        # Intent  
        elif stage == "intent":
            if action.content == state.true_intent:
                reward, message = 0.3, "Correct intent"
            else:
                message = "Wrong intent"
            history.append({"stage": stage, "output": action.content})
            state.current_stage = "reply"
            
        # Reply
        elif stage == "reply":
            if action.content.lower() in state.true_reply.lower():
                reward, message = 0.4, "Good reply"
            else:
                reward, message = 0.2, "Acceptable reply"
            history.append({"stage": stage, "output": action.content})
            _histories[episode_id] = history
            
            return EmailTriageObservation(
                done=True, reward=reward,
                email_text=state.email_text, current_stage="done",
                history=history, message=message, metadata={"episode_id": episode_id}
            )
        
        _histories[episode_id] = history
        
        return EmailTriageObservation(
            done=False, reward=reward,
            email_text=state.email_text, current_stage=state.current_stage,
            history=history, message=message, metadata={"episode_id": episode_id}
        )

    async def step_async(self, action: EmailTriageAction, **kwargs) -> EmailTriageObservation:
        return self.step(action)
    
    async def reset_async(self, seed=None, episode_id=None) -> EmailTriageObservation:
        return self.reset(seed, episode_id)
    
    @property
    def state(self) -> EmailTriageState:
        global _episodes
        return _episodes.get(self.episode_id)
    
    def close(self) -> None:
        pass

print("EmailTriageEnvironment defined.")