import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict

# These imports work perfectly now that the files are in the clinical_triage/ folder
from clinical_triage.models import TriageAction, ClinicalObservation
from clinical_triage.environment import ClinicalTriageEnvironment

app = FastAPI(title="ClinicalTriageEnv", version="1.1.0")

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
    return {"status": "ok", "interface": "OpenEnv-Compliant"}

@app.post("/reset", response_model=StepResult)
def reset(req: Optional[ResetRequest] = None):
    global _env
    # Safe handling of request parameters
    t_id = getattr(req, "task_id", "task1") if req else "task1"
    s = getattr(req, "seed", 42) if req else 42
    
    _env = ClinicalTriageEnvironment(task_id=t_id, seed=s)
    obs = _env.reset()
    return StepResult(observation=obs, reward=0.0, done=False)

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
        info=info if info is not None else {}
    )

# This is the entry point for the 'clinical-triage-server' command
def main():
    # Note: "server.app:app" tells uvicorn to look in the server folder, app.py file, for the 'app' object
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()