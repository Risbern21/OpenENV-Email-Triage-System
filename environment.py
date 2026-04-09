import random
import uuid
from typing import Any, Dict, List

from graders.grader import grade
from models import EmailTriageAction, EmailTriageObservation, EmailTriageState
from tasks import TASK_REGISTRY

#  GLOBAL episode storage (survives across requests)
_episodes: Dict[str, EmailTriageState] = {}
_histories: Dict[str, List[Dict]] = {}


def load_task(task_id: str) -> Dict[str, Any]:
    corpus = TASK_REGISTRY[task_id]["data_corpus"]
    return corpus[random.randint(0, len(corpus) - 1)]


class EmailTriageEnvironment:
    def __init__(self, difficulty: str = "easy"):
        """
        Args:
            difficulty: "easy" | "medium" | "hard"
                        Controls grader ceilings and reasoning thresholds.
        """
        self.episode_id = None
        self.task_id = ""
        self.difficulty = difficulty

    def reset(self, seed=None, episode_id=None) -> EmailTriageObservation:
        global _episodes, _histories

        if seed is not None:
            random.seed(seed)

        episode_id = episode_id or str(uuid.uuid4())
        email = load_task(task_id=self.task_id)

        _episodes[episode_id] = EmailTriageState(
            episode_id=episode_id,
            step_count=0,
            task_id=self.task_id,
            email_text=email["email_text"],
            true_classification=email["expected_classification"],
            true_intent=email["expected_intent"],
            true_reply=email["expected_reply"],
            current_stage="classification",
            difficulty=self.difficulty,
        )
        _histories[episode_id] = []
        self.episode_id = episode_id

        return EmailTriageObservation(
            done=False,
            reward=0.0,
            email_text=email["email_text"],
            current_stage="classification",
            history=[],
            message="Start with classification",
            metadata={"episode_id": episode_id, "difficulty": self.difficulty},
        )

    def step(self, action: EmailTriageAction) -> EmailTriageObservation:
        global _episodes, _histories

        episode_id = self.episode_id or (
            list(_episodes.keys())[-1] if _episodes else None
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
        stage = state.current_stage

        state.step_count += 1

        # ── Grade the action ───────────────────────────────────────────────
        reward_obj = grade(action, stage, state, difficulty=state.difficulty)

        history.append(
            {
                "stage": stage,
                "output": action.content,
                "reasoning": action.reasoning,
                "reward": str(round(reward_obj.total, 4)),
                "format_score": str(round(reward_obj.format_score, 4)),
                "label_score": str(round(reward_obj.label_score, 4)),
                "reasoning_score": str(round(reward_obj.reasoning_score, 4)),
            }
        )
        _histories[episode_id] = history

        # ── Build feedback message ─────────────────────────────────────────
        message = _build_message(stage, action, state, reward_obj)

        # ── Advance stage or finish ────────────────────────────────────────
        next_stage_map = {
            "classification": "intent",
            "intent": "reply",
        }

        if stage == "reply":
            # Terminal step
            return EmailTriageObservation(
                done=True,
                reward=reward_obj.total,
                email_text=state.email_text,
                current_stage="done",
                history=history,
                message=message,
                metadata={"episode_id": episode_id},
                format_score=reward_obj.format_score,
                label_score=reward_obj.label_score,
                reasoning_score=reward_obj.reasoning_score,
            )

        state.current_stage = next_stage_map[stage]

        return EmailTriageObservation(
            done=False,
            reward=reward_obj.total,
            email_text=state.email_text,
            current_stage=state.current_stage,
            history=history,
            message=message,
            metadata={"episode_id": episode_id},
            format_score=reward_obj.format_score,
            label_score=reward_obj.label_score,
            reasoning_score=reward_obj.reasoning_score,
        )

    async def step_async(
        self, action: EmailTriageAction, **kwargs
    ) -> EmailTriageObservation:
        return self.step(action)

    async def reset_async(self, seed=None, episode_id=None) -> EmailTriageObservation:
        return self.reset(seed, episode_id)

    @property
    def state(self) -> EmailTriageState:
        return _episodes.get(self.episode_id)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_message(
    stage: str,
    action: EmailTriageAction,
    state: EmailTriageState,
    reward_obj: Any,
) -> str:
    """Human-readable feedback summarising what the grader found."""
    parts = [
        f"[{stage}] total={reward_obj.total:.3f}",
        f"format={reward_obj.format_score:.2f}",
        f"label={reward_obj.label_score:.2f}",
        f"reasoning={reward_obj.reasoning_score:.2f}",
    ]

    # Directional hint (does not reveal ground truth)
    if reward_obj.label_score >= 0.40:
        parts.append("✓ label correct")
    else:
        parts.append("✗ label incorrect")

    if reward_obj.format_score == 0.0:
        parts.append("(check action_type / confidence / vocabulary)")

    return " | ".join(parts)


print("EmailTriageEnvironment defined.")
