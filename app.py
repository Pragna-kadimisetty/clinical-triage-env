"""
FastAPI server for ClinicalTriageEnv.
Endpoints: POST /reset, POST /step, GET /state, GET /tasks, GET /
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import TriageAction, ClinicalObservation, ClinicalState
from environment import ClinicalTriageEnvironment

app = FastAPI(
    title="ClinicalTriageEnv",
    description="OpenEnv-compatible ICU triage environment. Agents allocate scarce hospital resources to patients with varying severity under uncertainty.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_env: Optional[ClinicalTriageEnvironment] = None


class ResetRequest(BaseModel):
    task_id: str = "task1"
    seed: int = 42


class StepResponse(BaseModel):
    observation: ClinicalObservation
    reward: float
    done: bool
    info: dict = {}


@app.get("/")
def root():
    return {
        "status": "ok",
        "env": "ClinicalTriageEnv",
        "version": "1.0.0",
        "tasks": ["task1", "task2", "task3"],
    }


@app.post("/reset", response_model=ClinicalObservation)
def reset(req: ResetRequest):
    global _env
    _env = ClinicalTriageEnvironment(task_id=req.task_id, seed=req.seed)
    return _env.reset()


@app.post("/step", response_model=StepResponse)
def step(action: TriageAction):
    global _env
    if _env is None:
        raise HTTPException(400, "Call /reset first.")
    obs, reward, done, info = _env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=ClinicalState)
def state():
    global _env
    if _env is None:
        raise HTTPException(400, "Call /reset first.")
    return _env.state


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "task1",
                "name": "Single-Resource Triage",
                "difficulty": "easy",
                "description": "Triage 6 patients with severity hints shown. Adequate resources available.",
                "patients": 6,
                "severity_hints_visible": True,
                "resource_pressure": "low",
            },
            {
                "id": "task2",
                "name": "Multi-Resource Triage Under Pressure",
                "difficulty": "medium",
                "description": "Triage 9 patients. No severity hints. Moderate resource scarcity. Deceptive cases included.",
                "patients": 9,
                "severity_hints_visible": False,
                "resource_pressure": "medium",
            },
            {
                "id": "task3",
                "name": "Mass Casualty Surge",
                "difficulty": "hard",
                "description": "Triage 12 patients during surge. No hints. All ICU beds full. Ventilators exhausted. Ethical trade-offs required. Rationale graded.",
                "patients": 12,
                "severity_hints_visible": False,
                "resource_pressure": "critical",
            },
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)