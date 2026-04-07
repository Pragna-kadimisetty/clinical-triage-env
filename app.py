"""
FastAPI server for ClinicalTriageEnv.
Endpoints: POST /reset, POST /step, GET /state, GET /tasks, GET /
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict

from models import TriageAction, ClinicalObservation, ClinicalState
from environment import ClinicalTriageEnvironment

app = FastAPI(
    title="ClinicalTriageEnv",
    description="OpenEnv-compatible ICU triage environment.",
    version="1.1.0",
)

# Enable CORS for Hugging Face Spaces and external agent access
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Global environment instance
_env: Optional[ClinicalTriageEnvironment] = None

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task1"
    seed: Optional[int] = 42

class StepResult(BaseModel):
    """Standard OpenEnv step response structure required by validator."""
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
        "tasks": ["task1", "task2", "task3"],
        "interface": "OpenEnv-Compliant"
    }

@app.post("/reset", response_model=StepResult)
def reset(req: Optional[ResetRequest] = None):
    global _env
    
    # Extract values safely to prevent 422 errors
    t_id = getattr(req, "task_id", "task1") if req else "task1"
    s = getattr(req, "seed", 42) if req else 42
    
    _env = ClinicalTriageEnvironment(task_id=t_id, seed=s)
    obs = _env.reset()
    
    # OpenEnv Validate often expects a StepResult structure even for reset
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
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
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
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)