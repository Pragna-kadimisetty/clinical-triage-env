"""
Typed Pydantic models for ClinicalTriageEnv.
All OpenEnv-required models: Action, Observation, State, Reward.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum


# ------------------------------------------------------------------ #
# Enumerations                                                         #
# ------------------------------------------------------------------ #

class TriageDecision(str, Enum):
    ADMIT_ICU = "admit_icu"           # Admit to ICU (high resource)
    ADMIT_WARD = "admit_ward"         # Admit to general ward (low resource)
    STABILIZE = "stabilize"           # Stabilize and reassess next step
    DISCHARGE = "discharge"           # Discharge / no admission needed
    PALLIATIVE = "palliative"         # Comfort care only (resource-sparing)
    DEFER = "defer"                   # Wait — no action this step


class ResourceType(str, Enum):
    ICU_BED = "icu_bed"
    VENTILATOR = "ventilator"
    WARD_BED = "ward_bed"
    SPECIALIST = "specialist"


# ------------------------------------------------------------------ #
# Action                                                               #
# ------------------------------------------------------------------ #

class TriageAction(BaseModel):
    """
    Action submitted by the agent for a single patient.
    In multi-patient steps, submit one action per step (agent sees one patient at a time).
    """
    patient_id: str = Field(..., description="ID of patient being triaged")
    decision: TriageDecision = Field(..., description="Triage decision for this patient")
    resource_allocation: Dict[ResourceType, int] = Field(
        default_factory=dict,
        description="Specific resources to allocate (e.g. {icu_bed: 1, ventilator: 1})"
    )
    rationale: Optional[str] = Field(
        None,
        description="Optional reasoning (used for grading quality in task3)"
    )
    priority_override: Optional[int] = Field(
        None, ge=1, le=10,
        description="Override patient queue priority 1=highest, 10=lowest"
    )


# ------------------------------------------------------------------ #
# Observation                                                          #
# ------------------------------------------------------------------ #

class PatientVitals(BaseModel):
    heart_rate: int = Field(..., description="BPM")
    systolic_bp: int = Field(..., description="mmHg")
    spo2: float = Field(..., description="Oxygen saturation 0-100%")
    respiratory_rate: int = Field(..., description="Breaths per minute")
    gcs: int = Field(..., ge=3, le=15, description="Glasgow Coma Scale 3-15")
    temperature: float = Field(..., description="°C")
    lactate: Optional[float] = Field(None, description="Lactate mmol/L, if available")


class PatientInfo(BaseModel):
    patient_id: str
    age: int
    chief_complaint: str
    vitals: PatientVitals
    history: str
    time_since_arrival_minutes: int
    deteriorating: bool = Field(False, description="True if vitals worsening since last step")
    severity_hint: Optional[str] = Field(
        None,
        description="Partial hint in easy mode. None in hard mode (hidden state)."
    )


class ResourceState(BaseModel):
    icu_beds_total: int
    icu_beds_used: int
    ventilators_total: int
    ventilators_used: int
    ward_beds_total: int
    ward_beds_used: int
    specialists_available: int


class ClinicalObservation(BaseModel):
    """Full observation returned after reset() and each step()."""
    task_id: str
    step: int
    max_steps: int
    current_patient: PatientInfo
    resource_state: ResourceState
    queue_length: int = Field(..., description="Patients still waiting")
    episode_reward_so_far: float
    last_step_reward: float
    last_step_feedback: str
    done: bool
    info: Dict = Field(default_factory=dict)


# ------------------------------------------------------------------ #
# State (internal snapshot)                                           #
# ------------------------------------------------------------------ #

class ClinicalState(BaseModel):
    """Full internal state snapshot for state() endpoint."""
    episode_id: str
    task_id: str
    step: int
    total_patients_seen: int
    admitted_icu: int
    admitted_ward: int
    discharged: int
    deceased: int
    resource_state: ResourceState
    cumulative_reward: float
    patient_queue_ids: List[str]


# ------------------------------------------------------------------ #
# Reward breakdown (returned in info)                                 #
# ------------------------------------------------------------------ #

class RewardBreakdown(BaseModel):
    survival_reward: float = 0.0
    resource_efficiency: float = 0.0
    equity_penalty: float = 0.0
    delay_penalty: float = 0.0
    invalid_action_penalty: float = 0.0
    rationale_bonus: float = 0.0
    total: float = 0.0