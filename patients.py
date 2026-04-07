"""
Patient generator for ClinicalTriageEnv.
Produces realistic synthetic patients with ground-truth severity scores.
Severity is HIDDEN from the agent in task2/task3 (partial observability).
"""

import random
from typing import List, Optional
from models import PatientInfo, PatientVitals


# ------------------------------------------------------------------ #
# Patient templates — each has a ground-truth severity and a          #
# clinically plausible presentation                                   #
# ------------------------------------------------------------------ #

PATIENT_TEMPLATES = [
    # ---- CRITICAL (needs ICU + vent) ----
    {
        "id_prefix": "PT",
        "chief_complaint": "Respiratory failure, unresponsive",
        "history": "67M found unresponsive at home. Bystander CPR performed. History of COPD.",
        "age_range": (55, 80),
        "vitals": {"hr": (120, 140), "sbp": (70, 90), "spo2": (72, 82), "rr": (28, 36), "gcs": (3, 7), "temp": (36.0, 37.2), "lactate": (6.0, 10.0)},
        "true_severity": 0.95,
        "needs_icu": True,
        "needs_vent": True,
        "deteriorating_prob": 0.85,
        "hint": "CRITICAL",
    },
    {
        "id_prefix": "PT",
        "chief_complaint": "Massive GI bleed, hypotensive",
        "history": "52M with hematemesis x3 hours. BP crashing. On anticoagulants for AFib.",
        "age_range": (45, 65),
        "vitals": {"hr": (130, 160), "sbp": (60, 80), "spo2": (88, 93), "rr": (22, 30), "gcs": (9, 12), "temp": (36.2, 37.0), "lactate": (4.5, 8.0)},
        "true_severity": 0.92,
        "needs_icu": True,
        "needs_vent": False,
        "deteriorating_prob": 0.75,
        "hint": "CRITICAL",
    },
    {
        "id_prefix": "PT",
        "chief_complaint": "STEMI, cardiogenic shock",
        "history": "61F acute anterior STEMI. EMS started heparin. Now hypotensive.",
        "age_range": (50, 75),
        "vitals": {"hr": (110, 140), "sbp": (70, 85), "spo2": (85, 91), "rr": (24, 32), "gcs": (10, 14), "temp": (36.5, 37.5), "lactate": (3.5, 6.0)},
        "true_severity": 0.90,
        "needs_icu": True,
        "needs_vent": False,
        "deteriorating_prob": 0.70,
        "hint": "CRITICAL",
    },
    # ---- SERIOUS (ICU preferred, may survive ward) ----
    {
        "id_prefix": "PT",
        "chief_complaint": "Severe sepsis, confusion",
        "history": "74F nursing home resident. Fever, altered mental status x2 days. UTI suspected.",
        "age_range": (65, 85),
        "vitals": {"hr": (105, 125), "sbp": (85, 100), "spo2": (90, 95), "rr": (20, 26), "gcs": (10, 13), "temp": (38.8, 40.2), "lactate": (2.5, 4.5)},
        "true_severity": 0.72,
        "needs_icu": True,
        "needs_vent": False,
        "deteriorating_prob": 0.50,
        "hint": "SERIOUS",
    },
    {
        "id_prefix": "PT",
        "chief_complaint": "Acute pancreatitis, severe",
        "history": "45M alcoholic, third admission. Severe epigastric pain. Lipase >3000.",
        "age_range": (35, 60),
        "vitals": {"hr": (100, 120), "sbp": (90, 110), "spo2": (92, 96), "rr": (18, 24), "gcs": (13, 15), "temp": (37.8, 39.5), "lactate": (2.0, 3.5)},
        "true_severity": 0.65,
        "needs_icu": True,
        "needs_vent": False,
        "deteriorating_prob": 0.40,
        "hint": "SERIOUS",
    },
    # ---- MODERATE (ward appropriate) ----
    {
        "id_prefix": "PT",
        "chief_complaint": "COPD exacerbation, dyspnea",
        "history": "69M COPD Gold III, increasing SOB x3 days. Productive cough. On home O2.",
        "age_range": (55, 80),
        "vitals": {"hr": (90, 108), "sbp": (130, 155), "spo2": (88, 93), "rr": (22, 28), "gcs": (14, 15), "temp": (37.2, 38.5), "lactate": (1.5, 2.5)},
        "true_severity": 0.48,
        "needs_icu": False,
        "needs_vent": False,
        "deteriorating_prob": 0.20,
        "hint": "MODERATE",
    },
    {
        "id_prefix": "PT",
        "chief_complaint": "Pneumonia, moderate severity",
        "history": "58F community-acquired pneumonia. PSI class III. No ICU criteria met.",
        "age_range": (40, 70),
        "vitals": {"hr": (88, 105), "sbp": (120, 145), "spo2": (90, 95), "rr": (18, 24), "gcs": (14, 15), "temp": (38.2, 39.5), "lactate": (1.0, 2.2)},
        "true_severity": 0.42,
        "needs_icu": False,
        "needs_vent": False,
        "deteriorating_prob": 0.15,
        "hint": "MODERATE",
    },
    # ---- MILD (discharge or observation) ----
    {
        "id_prefix": "PT",
        "chief_complaint": "Chest pain, low risk",
        "history": "38M anxious, sharp chest pain reproducible with palpation. ECG normal. Troponin negative x2.",
        "age_range": (25, 50),
        "vitals": {"hr": (82, 98), "sbp": (125, 145), "spo2": (97, 99), "rr": (14, 18), "gcs": (15, 15), "temp": (36.5, 37.2), "lactate": (0.8, 1.4)},
        "true_severity": 0.12,
        "needs_icu": False,
        "needs_vent": False,
        "deteriorating_prob": 0.02,
        "hint": "MILD",
    },
    {
        "id_prefix": "PT",
        "chief_complaint": "Mild asthma exacerbation",
        "history": "24F known asthmatic. Mild wheeze after allergen exposure. Responded to salbutamol.",
        "age_range": (18, 35),
        "vitals": {"hr": (92, 105), "sbp": (115, 130), "spo2": (94, 97), "rr": (16, 20), "gcs": (15, 15), "temp": (36.8, 37.4), "lactate": (0.9, 1.6)},
        "true_severity": 0.18,
        "needs_icu": False,
        "needs_vent": False,
        "deteriorating_prob": 0.05,
        "hint": "MILD",
    },
    # ---- DECEPTIVE: looks bad, actually stable (tests reasoning) ----
    {
        "id_prefix": "PT",
        "chief_complaint": "Anxiety, hyperventilation",
        "history": "28F panic attack. SpO2 reads low due to nail polish artifact. Alert and oriented x4.",
        "age_range": (20, 40),
        "vitals": {"hr": (110, 130), "sbp": (140, 160), "spo2": (85, 91), "rr": (28, 36), "gcs": (15, 15), "temp": (36.8, 37.3), "lactate": (0.8, 1.3)},
        "true_severity": 0.08,
        "needs_icu": False,
        "needs_vent": False,
        "deteriorating_prob": 0.02,
        "hint": None,  # Hidden in all tasks — deceptive case
    },
    # ---- PALLIATIVE: very ill, but intervention unlikely to help ----
    {
        "id_prefix": "PT",
        "chief_complaint": "End-stage cancer, respiratory distress",
        "history": "79M metastatic lung Ca, hospice patient brought in by family against DNR. Comfort care appropriate.",
        "age_range": (70, 90),
        "vitals": {"hr": (115, 135), "sbp": (80, 95), "spo2": (74, 82), "rr": (28, 38), "gcs": (6, 10), "temp": (35.8, 37.0), "lactate": (7.0, 12.0)},
        "true_severity": 0.98,
        "needs_icu": False,   # Palliative appropriate — ICU would be harmful
        "needs_vent": False,
        "deteriorating_prob": 1.0,
        "hint": "PALLIATIVE",
    },
]


def _rand_vitals(template: dict, seed: Optional[int] = None) -> PatientVitals:
    rng = random.Random(seed)
    v = template["vitals"]
    return PatientVitals(
        heart_rate=rng.randint(*v["hr"]),
        systolic_bp=rng.randint(*v["sbp"]),
        spo2=round(rng.uniform(*v["spo2"]), 1),
        respiratory_rate=rng.randint(*v["rr"]),
        gcs=rng.randint(*v["gcs"]),
        temperature=round(rng.uniform(*v["temp"]), 1),
        lactate=round(rng.uniform(*v["lactate"]), 1) if v.get("lactate") else None,
    )


def generate_patient_queue(
    task_id: str,
    seed: int = 42,
    n_patients: Optional[int] = None,
) -> List[dict]:
    """
    Generate a queue of patients with their hidden ground-truth data.
    Returns list of dicts with patient info + hidden fields.
    task1: hints shown, balanced queue, small
    task2: no hints, more patients, resource pressure
    task3: no hints, surge scenario, some deceptive cases, resource scarcity
    """
    rng = random.Random(seed)

    if task_id == "task1":
        n = n_patients or 6
        templates = rng.choices(PATIENT_TEMPLATES[:7], k=n)
    elif task_id == "task2":
        n = n_patients or 9
        templates = rng.choices(PATIENT_TEMPLATES, k=n)
    else:  # task3 — mass casualty, forced hard choices
        n = n_patients or 12
        # Weight toward critical and palliative to force hard trade-offs
        weights = [3, 3, 2, 2, 2, 1, 1, 1, 1, 3, 3]
        templates = rng.choices(PATIENT_TEMPLATES, weights=weights, k=n)

    patients = []
    for i, tmpl in enumerate(templates):
        age = rng.randint(*tmpl["age_range"])
        vitals = _rand_vitals(tmpl, seed=seed + i)
        patient_seed = seed + i * 100

        # In task1 show hint. task2/task3 hide it (partial observability).
        show_hint = task_id == "task1"
        hint = tmpl["hint"] if show_hint else None

        p = {
            # Public (agent sees these)
            "patient_id": f"PT-{patient_seed % 9000 + 1000}",
            "age": age,
            "chief_complaint": tmpl["chief_complaint"],
            "history": tmpl["history"],
            "vitals": vitals,
            "time_since_arrival_minutes": rng.randint(5, 90),
            "deteriorating": rng.random() < tmpl["deteriorating_prob"],
            "severity_hint": hint,
            # Private (grader uses these, agent never sees)
            "_true_severity": tmpl["true_severity"],
            "_needs_icu": tmpl["needs_icu"],
            "_needs_vent": tmpl["needs_vent"],
            "_palliative": tmpl.get("hint") == "PALLIATIVE",
        }
        patients.append(p)

    return patients


def get_initial_resources(task_id: str) -> dict:
    """
    Resource availability per task.
    task1: comfortable (easy decisions)
    task2: moderate pressure
    task3: scarcity (forces ethical trade-offs)
    """
    if task_id == "task1":
        return {"icu_beds_total": 6, "icu_beds_used": 2, "ventilators_total": 4, "ventilators_used": 1, "ward_beds_total": 20, "ward_beds_used": 8, "specialists_available": 3}
    elif task_id == "task2":
        return {"icu_beds_total": 4, "icu_beds_used": 3, "ventilators_total": 3, "ventilators_used": 2, "ward_beds_total": 15, "ward_beds_used": 10, "specialists_available": 2}
    else:  # task3 — mass casualty
        return {"icu_beds_total": 4, "icu_beds_used": 4, "ventilators_total": 3, "ventilators_used": 3, "ward_beds_total": 12, "ward_beds_used": 11, "specialists_available": 1}