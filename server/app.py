from typing import Any

import uvicorn
from fastapi import HTTPException
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
        # Breakdown mirrors EmailTriageObservation fields
        "label_score": round(label_score, 4),
        "reasoning_score": round(reasoning_score, 4),
        "format_score": round(format_score, 4),
        "difficulty": difficulty,
        "ceiling": _difficulty_ceiling(difficulty),
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
