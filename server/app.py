import uvicorn
from openenv.core.env_server import create_fastapi_app

from environment import EmailTriageEnvironment
from models import EmailTriageAction, EmailTriageObservation

app = create_fastapi_app(
    EmailTriageEnvironment, EmailTriageAction, EmailTriageObservation
)


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
