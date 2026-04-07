"""
FastAPI server for ClinicalTriageEnv.
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict

# UPDATED IMPORTS: Pointing to the new package folder
from clinical_triage.models import TriageAction, ClinicalObservation, ClinicalState
from clinical_triage.environment import ClinicalTriageEnvironment

app = FastAPI(
    title="ClinicalTriageEnv",
    description="OpenEnv-compatible ICU triage environment.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

_env: Optional[ClinicalTriageEnvironment] = None

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task1"
    seed: Optional[int] = 42

class StepResult(BaseModel):
    observation: ClinicalObservation
    reward: float
    done: bool
    info: Dict = {}

@app.get("/")
def root():
    return {
        "status": "ok",
        "env": "clinical-triage-env",
        "version": "1.1.0",
        "tasks": ["task1", "task2", "task3"]
    }

@app.post("/reset", response_model=StepResult)
def reset(req: Optional[ResetRequest] = None):
    global _env
    t_id = getattr(req, "task_id", "task1") if req else "task1"
    s = getattr(req, "seed", 42) if req else 42
    
    _env = ClinicalTriageEnvironment(task_id=t_id, seed=s)
    obs = _env.reset()
    
    return StepResult(
        observation=obs,
        reward=0.0,
        done=False,
        info={}
    )

@app.post("/step", response_model=StepResult)
def step(action: TriageAction):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    
    obs, reward, done, info = _env.step(action)
    
    return StepResult(
        observation=obs,
        reward=float(reward),
        done=bool(done),
        info=info
    )

@app.get("/state", response_model=ClinicalState)
def state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return _env.state

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "task1", "name": "Single-Resource Triage", "difficulty": "easy"},
            {"id": "task2", "name": "Multi-Resource Triage", "difficulty": "medium"},
            {"id": "task3", "name": "Mass Casualty Surge", "difficulty": "hard"}
        ]
    }

# This main function is what the [project.scripts] calls
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()