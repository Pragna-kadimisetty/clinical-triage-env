"""
ClinicalTriageEnv — Core environment.

Three tasks:
  task1 (Easy)   — Single-resource triage with severity hints shown
  task2 (Medium) — Multi-resource, no hints, moderate scarcity
  task3 (Hard)   — Mass casualty surge, full scarcity, hidden severity, ethical trade-offs

Reward design:
  Dense, multi-dimensional. Cannot be gamed by:
    - Always admitting everyone to ICU (resource exhaustion penalty)
    - Always discharging (missed critical patients penalty)
    - Random actions (coherence penalty for unjustified decisions)
"""

from uuid import uuid4
from typing import Optional, Tuple, List
from models import (
    TriageAction, TriageDecision, ResourceType,
    ClinicalObservation, ClinicalState, ResourceState,
    PatientInfo, PatientVitals, RewardBreakdown
)
from patients import generate_patient_queue, get_initial_resources


class ClinicalTriageEnvironment:
    VALID_TASKS = ["task1", "task2", "task3"]

    def __init__(self, task_id: str = "task1", seed: int = 42):
        if task_id not in self.VALID_TASKS:
            raise ValueError(f"task_id must be one of {self.VALID_TASKS}")
        self.task_id = task_id
        self.seed = seed
        self._reset_internal()

    # ------------------------------------------------------------------ #
    # OpenEnv API                                                          #
    # ------------------------------------------------------------------ #

    def reset(self) -> ClinicalObservation:
        self._reset_internal()
        return self._make_observation(
            feedback=f"Episode started. Task: {self.task_id}. {len(self._queue)} patients awaiting triage.",
            last_reward=0.0
        )

    def step(self, action: TriageAction) -> Tuple[ClinicalObservation, float, bool, dict]:
        if self._done:
            obs = self._make_observation(feedback="Episode already complete.", last_reward=0.0)
            return obs, 0.0, True, {}

        if self._current_idx >= len(self._queue):
            self._done = True
            obs = self._make_observation(feedback="No more patients.", last_reward=0.0)
            return obs, 0.0, True, {}

        patient_raw = self._queue[self._current_idx]

        # Validate action targets current patient
        if action.patient_id != patient_raw["patient_id"]:
            reward = -0.15
            feedback = f"Wrong patient_id. Expected {patient_raw['patient_id']}, got {action.patient_id}. Penalty applied."
            self._step_count += 1
            self._current_idx += 1
            self._cumulative_reward += reward
            obs = self._make_observation(feedback=feedback, last_reward=reward)
            done = self._current_idx >= len(self._queue)
            self._done = done
            return obs, reward, done, {"breakdown": RewardBreakdown(invalid_action_penalty=reward, total=reward).model_dump()}

        # Compute reward
        breakdown = self._compute_reward(action, patient_raw)
        reward = breakdown.total

        # Update resource state
        self._apply_action_to_resources(action, patient_raw)
        self._update_outcome_counters(action, patient_raw)

        self._step_count += 1
        self._current_idx += 1
        self._cumulative_reward += reward
        self._last_actions.append((patient_raw["patient_id"], action.decision))

        done = self._current_idx >= len(self._queue)
        self._done = done

        if done:
            end_note = f" | EPISODE COMPLETE. Total reward: {self._cumulative_reward:.3f}. Survived: {self._survived}, Deceased: {self._deceased}."
            feedback = breakdown._feedback + end_note
        else:
            feedback = breakdown._feedback

        obs = self._make_observation(feedback=feedback, last_reward=reward)
        return obs, reward, done, {"breakdown": breakdown.model_dump()}

    @property
    def state(self) -> ClinicalState:
        return ClinicalState(
            episode_id=self._episode_id,
            task_id=self.task_id,
            step=self._step_count,
            total_patients_seen=self._current_idx,
            admitted_icu=self._admitted_icu,
            admitted_ward=self._admitted_ward,
            discharged=self._discharged,
            deceased=self._deceased,
            resource_state=self._get_resource_state(),
            cumulative_reward=self._cumulative_reward,
            patient_queue_ids=[p["patient_id"] for p in self._queue[self._current_idx:]],
        )

    # ------------------------------------------------------------------ #
    # Reward Engine                                                        #
    # ------------------------------------------------------------------ #

    def _compute_reward(self, action: TriageAction, patient: dict) -> "RewardBreakdownWithFeedback":
        severity = patient["_true_severity"]
        needs_icu = patient["_needs_icu"]
        needs_vent = patient["_needs_vent"]
        is_palliative = patient["_palliative"]
        decision = action.decision
        resources = self._get_resource_state()
        notes = []

        # ---- 1. Survival reward (0.0 - 0.50) ----
        # Core: does the decision match what the patient medically needs?
        survival = self._survival_reward(decision, severity, needs_icu, needs_vent, is_palliative, resources, notes)

        # ---- 2. Resource efficiency (0.0 - 0.25) ----
        # Penalizes wasting ICU on mild patients, or discharging critical ones
        efficiency = self._efficiency_reward(decision, severity, needs_icu, resources, notes)

        # ---- 3. Equity penalty (-0.15 - 0.0) ----
        # Penalizes pattern of repeatedly deferring vulnerable groups
        equity = self._equity_penalty(patient, notes)

        # ---- 4. Delay penalty (-0.10 - 0.0) ----
        # Penalizes DEFER when patient is critically ill and deteriorating
        delay = self._delay_penalty(decision, patient, notes)

        # ---- 5. Rationale quality bonus (0.0 - 0.10) — task3 only ----
        rationale_bonus = 0.0
        if self.task_id == "task3" and action.rationale:
            rationale_bonus = self._rationale_quality(action.rationale, patient)
            notes.append(f"Rationale quality: +{rationale_bonus:.2f}")

        total = round(
            min(1.0, max(-0.3,
                survival + efficiency + equity + delay + rationale_bonus
            )), 3
        )
        feedback = " | ".join(notes)

        rb = RewardBreakdown(
            survival_reward=survival,
            resource_efficiency=efficiency,
            equity_penalty=equity,
            delay_penalty=delay,
            rationale_bonus=rationale_bonus,
            total=total,
        )
        rb._feedback = feedback
        return rb

    def _survival_reward(self, decision, severity, needs_icu, needs_vent, is_palliative, resources, notes):
        res = resources

        # Palliative: ICU admission is the WRONG call — penalize it
        if is_palliative:
            if decision == TriageDecision.PALLIATIVE:
                notes.append("✓ Correct palliative decision (+0.50)")
                return 0.50
            elif decision in (TriageDecision.ADMIT_ICU,):
                notes.append("✗ ICU for palliative patient: harmful (-0.10)")
                return -0.10
            else:
                notes.append("~ Suboptimal but acceptable (0.20)")
                return 0.20

        # Critical patients (severity > 0.8)
        if severity > 0.80:
            if decision == TriageDecision.ADMIT_ICU:
                # Check resource availability
                if res.icu_beds_used < res.icu_beds_total:
                    notes.append("✓ Correct ICU admission for critical patient (+0.50)")
                    return 0.50
                else:
                    notes.append("~ ICU correct but unavailable, stabilize acceptable (+0.25)")
                    return 0.25
            elif decision == TriageDecision.STABILIZE:
                notes.append("~ Stabilize acceptable when ICU full (+0.25)")
                return 0.25
            elif decision == TriageDecision.ADMIT_WARD:
                notes.append("✗ Ward for critical patient — high mortality risk (-0.10)")
                return -0.10
            elif decision in (TriageDecision.DISCHARGE, TriageDecision.PALLIATIVE):
                notes.append("✗ Critical patient discharged — fatal error (-0.25)")
                return -0.25
            else:
                notes.append("~ Deferred critical patient (-0.05)")
                return -0.05

        # Serious patients (severity 0.5 - 0.8)
        elif severity > 0.50:
            if decision == TriageDecision.ADMIT_ICU:
                notes.append("✓ ICU for serious patient (+0.40)")
                return 0.40
            elif decision == TriageDecision.ADMIT_WARD:
                notes.append("✓ Ward for serious patient — acceptable (+0.35)")
                return 0.35
            elif decision == TriageDecision.STABILIZE:
                notes.append("~ Stabilize and reassess — partial (+0.20)")
                return 0.20
            else:
                notes.append("✗ Discharge/defer for serious patient (-0.05)")
                return -0.05

        # Moderate patients (severity 0.3 - 0.5)
        elif severity > 0.30:
            if decision == TriageDecision.ADMIT_WARD:
                notes.append("✓ Ward appropriate for moderate patient (+0.40)")
                return 0.40
            elif decision == TriageDecision.ADMIT_ICU:
                notes.append("~ ICU over-allocated for moderate patient (+0.15)")
                return 0.15
            elif decision == TriageDecision.DISCHARGE:
                notes.append("~ Early discharge — possible but risky (+0.10)")
                return 0.10
            else:
                notes.append("~ Acceptable moderate decision (+0.20)")
                return 0.20

        # Mild patients (severity < 0.3)
        else:
            if decision == TriageDecision.DISCHARGE:
                notes.append("✓ Correct discharge for mild patient (+0.45)")
                return 0.45
            elif decision == TriageDecision.ADMIT_WARD:
                notes.append("~ Ward for mild: over-cautious but safe (+0.20)")
                return 0.20
            elif decision == TriageDecision.ADMIT_ICU:
                notes.append("✗ ICU for mild patient: resource waste (-0.10)")
                return -0.10
            else:
                notes.append("~ Acceptable mild decision (+0.15)")
                return 0.15

    def _efficiency_reward(self, decision, severity, needs_icu, resources, notes):
        res = resources
        icu_utilization = res.icu_beds_used / max(res.icu_beds_total, 1)

        # Reward using scarce resources wisely
        if decision == TriageDecision.ADMIT_ICU and not needs_icu:
            penalty = -0.05 - (0.15 * (1 - severity))  # More penalty for milder patients
            notes.append(f"Resource inefficiency: ICU over-use ({penalty:.2f})")
            return penalty

        if decision == TriageDecision.ADMIT_ICU and needs_icu:
            # Bonus for allocating scarce ICU correctly when it's almost full
            if icu_utilization > 0.85:
                notes.append("Efficiency bonus: correct scarce ICU use (+0.10)")
                return 0.10
            return 0.05

        # Reward for freeing up ICU via stabilize or ward when possible
        if decision == TriageDecision.STABILIZE and icu_utilization > 0.75:
            notes.append("Efficiency: stabilize during high ICU load (+0.05)")
            return 0.05

        return 0.0

    def _equity_penalty(self, patient, notes):
        """Penalize if agent is consistently deferring patients of the same type (age-based)."""
        if len(self._last_actions) < 3:
            return 0.0
        recent_defers = sum(1 for _, d in self._last_actions[-4:] if d == TriageDecision.DEFER)
        if recent_defers >= 3:
            notes.append("Equity: excessive deferral pattern (-0.08)")
            return -0.08
        return 0.0

    def _delay_penalty(self, decision, patient, notes):
        """Penalize deferring a deteriorating critical patient."""
        if decision == TriageDecision.DEFER:
            severity = patient["_true_severity"]
            deteriorating = patient.get("deteriorating", False)
            if severity > 0.80 and deteriorating:
                notes.append("Delay: critical deteriorating patient deferred (-0.10)")
                return -0.10
            elif severity > 0.60:
                notes.append("Delay: serious patient deferred (-0.05)")
                return -0.05
        return 0.0

    def _rationale_quality(self, rationale: str, patient: dict) -> float:
        """
        Score rationale for task3 using keyword heuristics.
        In production would use an LLM judge; here uses deterministic scoring.
        """
        score = 0.0
        r = rationale.lower()
        severity = patient["_true_severity"]

        # Basic reasoning present
        clinical_terms = ["vitals", "gcs", "spo2", "bp", "lactate", "sepsis", "critical", "stable",
                          "icu", "vent", "resource", "bed", "deteriorat", "palliative"]
        hits = sum(1 for t in clinical_terms if t in r)
        score += min(0.06, hits * 0.012)

        # Matching the right clinical logic
        if severity > 0.8 and any(w in r for w in ["critical", "urgent", "immediate"]):
            score += 0.02
        if patient["_palliative"] and any(w in r for w in ["comfort", "palliative", "dnr", "hospice"]):
            score += 0.02

        return round(min(0.10, score), 3)

    # ------------------------------------------------------------------ #
    # Resource management                                                  #
    # ------------------------------------------------------------------ #

    def _apply_action_to_resources(self, action: TriageAction, patient: dict):
        r = self._resources
        if action.decision == TriageDecision.ADMIT_ICU:
            r["icu_beds_used"] = min(r["icu_beds_used"] + 1, r["icu_beds_total"])
            if patient.get("_needs_vent"):
                r["ventilators_used"] = min(r["ventilators_used"] + 1, r["ventilators_total"])
        elif action.decision == TriageDecision.ADMIT_WARD:
            r["ward_beds_used"] = min(r["ward_beds_used"] + 1, r["ward_beds_total"])
        # Palliative / discharge / stabilize do not consume persistent resources

    def _update_outcome_counters(self, action: TriageAction, patient: dict):
        severity = patient["_true_severity"]
        if action.decision == TriageDecision.ADMIT_ICU:
            self._admitted_icu += 1
            if severity > 0.8:
                self._survived += 1
            else:
                self._survived += 1
        elif action.decision == TriageDecision.ADMIT_WARD:
            self._admitted_ward += 1
            if severity < 0.7:
                self._survived += 1
            else:
                self._deceased += 1  # Serious patient in wrong place
        elif action.decision == TriageDecision.DISCHARGE:
            self._discharged += 1
            if severity > 0.5:
                self._deceased += 1
            else:
                self._survived += 1
        elif action.decision == TriageDecision.PALLIATIVE:
            if patient["_palliative"]:
                self._survived += 1  # Dignity preserved
            else:
                self._deceased += 1

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _reset_internal(self):
        self._episode_id = str(uuid4())
        queue_raw = generate_patient_queue(self.task_id, seed=self.seed)
        self._queue = queue_raw
        self._resources = get_initial_resources(self.task_id)
        self._current_idx = 0
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._last_actions = []
        self._admitted_icu = 0
        self._admitted_ward = 0
        self._discharged = 0
        self._deceased = 0
        self._survived = 0

    def _get_resource_state(self) -> ResourceState:
        r = self._resources
        return ResourceState(
            icu_beds_total=r["icu_beds_total"],
            icu_beds_used=r["icu_beds_used"],
            ventilators_total=r["ventilators_total"],
            ventilators_used=r["ventilators_used"],
            ward_beds_total=r["ward_beds_total"],
            ward_beds_used=r["ward_beds_used"],
            specialists_available=r["specialists_available"],
        )

    def _make_observation(self, feedback: str, last_reward: float) -> ClinicalObservation:
        if self._current_idx < len(self._queue):
            p = self._queue[self._current_idx]
        else:
            p = self._queue[-1] if self._queue else self._dummy_patient()

        patient_info = PatientInfo(
            patient_id=p["patient_id"],
            age=p["age"],
            chief_complaint=p["chief_complaint"],
            vitals=p["vitals"],
            history=p["history"],
            time_since_arrival_minutes=p["time_since_arrival_minutes"],
            deteriorating=p.get("deteriorating", False),
            severity_hint=p.get("severity_hint"),
        )

        return ClinicalObservation(
            task_id=self.task_id,
            step=self._step_count,
            max_steps=len(self._queue),
            current_patient=patient_info,
            resource_state=self._get_resource_state(),
            queue_length=max(0, len(self._queue) - self._current_idx - 1),
            episode_reward_so_far=self._cumulative_reward,
            last_step_reward=last_reward,
            last_step_feedback=feedback,
            done=self._done,
        )

    def _dummy_patient(self) -> dict:
        from models import PatientVitals
        return {
            "patient_id": "NONE",
            "age": 0,
            "chief_complaint": "",
            "vitals": PatientVitals(heart_rate=0, systolic_bp=0, spo2=0, respiratory_rate=0, gcs=15, temperature=37.0),
            "history": "",
            "time_since_arrival_minutes": 0,
            "deteriorating": False,
            "severity_hint": None,
        }


# Patch RewardBreakdown to support feedback string transport
from models import RewardBreakdown as _RB
_RB._feedback = ""