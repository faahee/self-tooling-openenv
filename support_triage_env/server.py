from __future__ import annotations

from fastapi import FastAPI

from .env import SupportTriageEnvironment
from .models import ResetRequest, SupportTriageAction
from .tasks import TASKS

app = FastAPI(title="support-triage-openenv", version="0.1.0")
ENV = SupportTriageEnvironment()


@app.get("/")
async def root() -> dict:
    return {"name": "support-triage-openenv", "status": "ok", "tasks": list(TASKS)}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/reset")
async def reset(body: ResetRequest | None = None):
    task_name = body.task_name if body else None
    return ENV.reset(task_name).model_dump()


@app.post("/step")
async def step(action: SupportTriageAction):
    return ENV.step(action).model_dump()


@app.get("/state")
async def state():
    return ENV.state().model_dump()


@app.get("/tasks")
async def tasks():
    return {
        name: {
            "difficulty": task.difficulty,
            "objective": task.objective,
            "max_steps": task.max_steps,
        }
        for name, task in TASKS.items()
    }
