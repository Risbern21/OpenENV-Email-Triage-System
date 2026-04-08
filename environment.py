import random
import uuid
from typing import Any, Dict, List

from models import EmailTriageAction, EmailTriageObservation, EmailTriageState
from tasks import TASK_REGISTRY

#  GLOBAL episode storage (survives across requests)
_episodes: Dict[str, EmailTriageState] = {}
_histories: Dict[str, List[Dict]] = {}


def load_task(task_id: str):
    randIdx = random.uniform(0, len(TASK_REGISTRY[task_id]["data_corpus"]))

    return TASK_REGISTRY[task_id]["data_corpus"][randIdx]


class EmailTriageEnvironment:
    def __init__(self):
        self.episode_id = None

    def reset(self, seed=None, episode_id=None) -> EmailTriageObservation:
        global _episodes, _histories

        if seed is not None:
            random.seed(seed)

        episode_id = episode_id or str(uuid.uuid4())
        email = load_task(episode_id)

        _episodes[episode_id] = EmailTriageState(
            episode_id=episode_id,
            step_count=0,
            email_text=email["email"],
            true_classification=email["classification"],
            true_intent=email["intent"],
            true_reply=email["reply"],
            current_stage="classification",
        )
        _histories[episode_id] = []

        self.episode_id = episode_id

        state = _episodes[episode_id]
        return EmailTriageObservation(
            done=False,
            reward=0.0,
            email_text=state.email_text,
            current_stage=state.current_stage,
            history=[],
            message="Start with classification",
            metadata={"episode_id": episode_id},
        )

    def step(self, action: EmailTriageAction) -> EmailTriageObservation:
        global _episodes, _histories

        episode_id = (
            self.episode_id or list(_episodes.keys())[-1] if _episodes else None
        )

        if not episode_id or episode_id not in _episodes:
            return EmailTriageObservation(
                done=True,
                reward=0.0,
                email_text="",
                current_stage="error",
                history=[],
                message="No active episode",
                metadata={},
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
                done=True,
                reward=reward,
                email_text=state.email_text,
                current_stage="done",
                history=history,
                message=message,
                metadata={"episode_id": episode_id},
            )

        _histories[episode_id] = history

        return EmailTriageObservation(
            done=False,
            reward=reward,
            email_text=state.email_text,
            current_stage=state.current_stage,
            history=history,
            message=message,
            metadata={"episode_id": episode_id},
        )

    async def step_async(
        self, action: EmailTriageAction, **kwargs
    ) -> EmailTriageObservation:
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
