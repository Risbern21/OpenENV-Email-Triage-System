from typing import Any

import uvicorn
from fastapi import HTTPException, Query
from openenv.core.env_server import create_fastapi_app

from models import EmailTriageAction, EmailTriageObservation
from tasks import TASK_REGISTRY

from .env import EmailTriageEnvironment
from .grader import (_difficulty_ceiling, _score_format, _score_label,
                     _score_reasoning, grade)

app = create_fastapi_app(
    EmailTriageEnvironment, EmailTriageAction, EmailTriageObservation
)


@app.post("/grader")
def get_grader_score(task_id: str, action: EmailTriageAction) -> dict[str, Any]:
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

    task = TASK_REGISTRY[task_id]
    difficulty = str(task.get("difficulty", "easy")).lower()

    label_score = _score_label(action, task)
    reasoning_score = _score_reasoning(action)
    format_score = _score_format(action)
    score = grade(action=action, task_id=task_id)

    return {
        "task_id": task_id,
        "score": score,
        "passed": 1 if score > 0.5 else 0,
        "total": 1,
        "metric": "email_triage_env",
        "label_score": round(label_score, 4),
        "reasoning_score": round(reasoning_score, 4),
        "format_score": round(format_score, 4),
        "difficulty": difficulty,
        "ceiling": _difficulty_ceiling(difficulty),
    }


def _action_from_params(
    action_type: str,
    content: str,
    reasoning: str,
    confidence: float,
) -> EmailTriageAction:
    """Build an EmailTriageAction from query parameters."""
    return EmailTriageAction(
        action_type=action_type,
        content=content,
        reasoning=reasoning,
        confidence=confidence,
    )


@app.get("/grade/task_easy")
def grade_task_easy(
    action_type: str = Query(default="classification"),
    content: str = Query(default=""),
    reasoning: str = Query(default=""),
    confidence: float = Query(default=0.5, ge=0.0, le=1.0),
):
    action = _action_from_params(action_type, content, reasoning, confidence)
    score = grade(action=action, task_id="task_easy")
    return {"score": score, "reward": score}


@app.get("/grade/task_medium")
def grade_task_medium(
    action_type: str = Query(default="classification"),
    content: str = Query(default=""),
    reasoning: str = Query(default=""),
    confidence: float = Query(default=0.5, ge=0.0, le=1.0),
):
    action = _action_from_params(action_type, content, reasoning, confidence)
    score = grade(action=action, task_id="task_medium")
    return {"score": score, "reward": score}


@app.get("/grade/task_hard")
def grade_task_hard(
    action_type: str = Query(default="classification"),
    content: str = Query(default=""),
    reasoning: str = Query(default=""),
    confidence: float = Query(default=0.5, ge=0.0, le=1.0),
):
    action = _action_from_params(action_type, content, reasoning, confidence)
    score = grade(action=action, task_id="task_hard")
    return {"score": score, "reward": score}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
