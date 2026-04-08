from typing import Any

import uvicorn
from fastapi import HTTPException
from openenv.core.env_server import create_fastapi_app

from environment import EmailTriageEnvironment
from graders.grader import grade
from models import EmailTriageAction, EmailTriageObservation
from tasks import TASK_REGISTRY

app = create_fastapi_app(
    EmailTriageEnvironment, EmailTriageAction, EmailTriageObservation
)


@app.post("/grader")
def get_grader_score(task_id: str, action: EmailTriageAction) -> dict[str, Any]:
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    score = grade(task_id)
    return {
        "task_id": task_id,
        "score": score,
        "passed": 1 if score > 0.5 else 0,
        "total": 1,
        "metric": "clip_quality_alignment",
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
