from typing import Any, Optional

import uvicorn
from fastapi import HTTPException, Query
from openenv.core.env_server import create_fastapi_app

from models import EmailTriageAction, EmailTriageObservation, TaskInfo
from tasks import TASK_REGISTRY

from .env import EmailTriageEnvironment
from .grader import (_difficulty_ceiling, _score_format, _score_label,
                     _score_reasoning, grade, grade_report)

app = create_fastapi_app(
    EmailTriageEnvironment, EmailTriageAction, EmailTriageObservation
)

TASK_SURFACE_DESCRIPTIONS = {
    "task_easy": "triage easy level difficulty emails (classify,intent and reply)",
    "task_medium": "triage medium level difficulty emails (classify,intent and reply)",
    "task_hard": "triage hard level difficulty emails (classify,intent and reply)",
}


def _surface_task_description(task_id: str, fallback: str) -> str:
    return TASK_SURFACE_DESCRIPTIONS.get(task_id, fallback)


@app.get("/tasks", response_model=list[TaskInfo])
def list_tasks() -> list[TaskInfo]:
    return [
        TaskInfo(
            task_id=task_id,
            difficulty=task["difficulty"],
            description=_surface_task_description(task_id, task["description"]),
            action_schema=EmailTriageAction.model_json_schema(),
            grader=f"/grade/{task_id}",
        )
        for task_id, task in TASK_REGISTRY.items()
    ]


@app.post("/grader")
def get_grader_score(task_id: str, action: EmailTriageAction) -> dict[str, Any]:
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

    return grade_report(action, task_id)


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
@app.post("/grade/task_easy")
def grade_task_easy(
    action: Optional[EmailTriageAction] = None,
    action_type: str = Query(default="classification"),
    content: str = Query(default=""),
    reasoning: str = Query(default=""),
    confidence: float = Query(default=0.5, ge=0.0, le=1.0),
):
    if action is None:
        action = _action_from_params(action_type, content, reasoning, confidence)
    return grade_report(action, "task_easy")


@app.get("/grade/task_medium")
@app.post("/grade/task_medium")
def grade_task_medium(
    action: Optional[EmailTriageAction] = None,
    action_type: str = Query(default="classification"),
    content: str = Query(default=""),
    reasoning: str = Query(default=""),
    confidence: float = Query(default=0.5, ge=0.0, le=1.0),
):
    if action is None:
        action = _action_from_params(action_type, content, reasoning, confidence)
    return grade_report(action, "task_medium")


@app.get("/grade/task_hard")
@app.post("/grade/task_hard")
def grade_task_hard(
    action: Optional[EmailTriageAction] = None,
    action_type: str = Query(default="classification"),
    content: str = Query(default=""),
    reasoning: str = Query(default=""),
    confidence: float = Query(default=0.5, ge=0.0, le=1.0),
):
    if action is None:
        action = _action_from_params(action_type, content, reasoning, confidence)
    return grade_report(action, "task_hard")


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
