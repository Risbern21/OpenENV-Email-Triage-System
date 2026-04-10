import random
import uuid
from typing import Any, Dict, List, Optional

from models import EmailTriageAction, EmailTriageObservation, EmailTriageState
from tasks import TASK_REGISTRY

from .grader import grade

#  GLOBAL episode storage (survives across requests)
_episodes: Dict[str, EmailTriageState] = {}
_histories: Dict[str, List[Dict]] = {}

# Pick a stable default task so the server never gets an empty task_id
_DEFAULT_TASK_ID = next(iter(TASK_REGISTRY))


def load_task(task_id: str) -> Dict[str, Any]:
    corpus = TASK_REGISTRY[task_id]["data_corpus"]
    return corpus[random.randint(0, len(corpus) - 1)]


class EmailTriageEnvironment:
    def __init__(self, difficulty: str = "easy", task_id: Optional[str] = None):
        """
        Args:
            difficulty: "easy" | "medium" | "hard"
            task_id:    Optional task to pin this instance to.
                        Falls back to the first key in TASK_REGISTRY when
                        the server doesn't supply one.
        """
        self.episode_id: Optional[str] = None
        self.task_id: str = task_id or ""
        self.difficulty: str = difficulty

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_task_id(self, task_id: Optional[str] = None) -> str:
        """
        Priority order:
          1. Explicit task_id passed to reset()
          2. self.task_id set at construction time
          3. _DEFAULT_TASK_ID (first key in TASK_REGISTRY)
        Raises ValueError if the resolved id is not in the registry.
        """
        resolved = task_id or self.task_id or _DEFAULT_TASK_ID
        if resolved not in TASK_REGISTRY:
            raise ValueError(
                f"task_id '{resolved}' not found in TASK_REGISTRY. "
                f"Available: {list(TASK_REGISTRY.keys())}"
            )
        return resolved

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed=None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> EmailTriageObservation:
        global _episodes, _histories

        if seed is not None:
            random.seed(seed)

        resolved_task_id = self._resolve_task_id(task_id)
        resolved_difficulty = difficulty or self.difficulty

        # Keep instance state in sync so step() works without re-passing task_id
        self.task_id = resolved_task_id
        self.difficulty = resolved_difficulty

        episode_id = episode_id or str(uuid.uuid4())
        email = load_task(task_id=resolved_task_id)

        _episodes[episode_id] = EmailTriageState(
            episode_id=episode_id,
            step_count=0,
            task_id=resolved_task_id,
            email_text=email["email_text"],
            true_classification=email["expected_classification"],
            true_intent=email["expected_intent"],
            true_reply=email["expected_reply"],
            current_stage="classification",
            difficulty=resolved_difficulty,
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
            metadata={
                "episode_id": episode_id,
                "task_id": resolved_task_id,
                "difficulty": resolved_difficulty,
            },
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
                message="No active episode. Call /reset first.",
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

        message = _build_message(stage, action, state, reward_obj)

        next_stage_map = {
            "classification": "intent",
            "intent": "reply",
        }

        if stage == "reply":
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

    async def reset_async(
        self,
        seed=None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        **kwargs,
    ) -> EmailTriageObservation:
        """
        Async wrapper used by the openenv HTTP server.
        Accepts task_id and difficulty forwarded from the server's reset request
        so the environment is never stuck with an empty task_id.
        """
        return self.reset(
            seed=seed,
            episode_id=episode_id,
            task_id=task_id,
            difficulty=difficulty,
        )

    # ------------------------------------------------------------------

    @property
    def state(self) -> Optional[EmailTriageState]:
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
    parts = [
        f"[{stage}] total={reward_obj.total:.3f}",
        f"format={reward_obj.format_score:.2f}",
        f"label={reward_obj.label_score:.2f}",
        f"reasoning={reward_obj.reasoning_score:.2f}",
    ]

    if reward_obj.label_score >= 0.40:
        parts.append("✓ label correct")
    else:
        parts.append("✗ label incorrect")

    if reward_obj.format_score == 0.0:
        parts.append("(check action_type / confidence / vocabulary)")

    return " | ".join(parts)


print("EmailTriageEnvironment defined.")
