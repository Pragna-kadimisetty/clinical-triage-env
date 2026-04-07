"""
inference.py — Baseline inference for ClinicalTriageEnv.
Uses OpenAI-compatible client. Runs all 3 tasks end-to-end.

Required env vars:
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — Model identifier
  HF_TOKEN      — API key

Optional:
  ENV_URL       — Environment server URL (default: http://localhost:7860)
"""

import os
import json
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an experienced emergency physician and intensivist making rapid triage decisions in a busy hospital.

For each patient presented, analyze their vitals, history, and clinical presentation, then decide:

DECISIONS (pick exactly one):
- admit_icu      → Admit to ICU (use when: critical vitals, immediate life threat, needs intensive monitoring)
- admit_ward     → Admit to general ward (use when: serious but stable, needs monitoring not ICU)
- stabilize      → Stabilize and reassess (use when: improving or resources full, can wait safely)
- discharge      → Discharge home (use when: mild, not at risk)
- palliative     → Comfort care only (use when: terminal illness, intervention futile/against wishes)
- defer          → Defer decision (use RARELY — only if genuinely need more info)

RESOURCES to track: icu_beds_used/total, ventilators_used/total, ward_beds_used/total.
ICU full → cannot admit unless you defer someone else.

Always respond with valid JSON:
{
  "patient_id": "<id>",
  "decision": "<one of the decisions above>",
  "resource_allocation": {},
  "rationale": "<your clinical reasoning>",
  "priority_override": null
}

CRITICAL rules:
- NEVER discharge a patient with GCS < 12, SpO2 < 88%, or systolic BP < 90
- Always consider the severity_hint if provided
- If ICU beds are 0 remaining, use stabilize or ward instead
- Palliative patients should NOT get ICU (it prolongs suffering, wastes resources)
- Justify every decision with clinical data from the vitals
"""


def call_llm(messages: list) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


def parse_action(raw: str, patient_id: str) -> dict:
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
            clean = clean.replace("```", "")
        action = json.loads(clean.strip())
        action["patient_id"] = patient_id
        valid = ["admit_icu", "admit_ward", "stabilize", "discharge", "palliative", "defer"]
        if action.get("decision") not in valid:
            action["decision"] = "stabilize"
        if not isinstance(action.get("resource_allocation"), dict):
            action["resource_allocation"] = {}
        return action
    except Exception as e:
        print(f"    [WARN] JSON parse failed: {e}. Defaulting to stabilize.")
        return {
            "patient_id": patient_id,
            "decision": "stabilize",
            "resource_allocation": {},
            "rationale": "Parse error — defaulting to stabilize",
            "priority_override": None,
        }


def format_patient_prompt(obs: dict, task_id: str) -> str:
    p = obs["current_patient"]
    r = obs["resource_state"]
    v = p["vitals"]

    lines = [
        f"Task: {task_id} | Step {obs['step']+1}/{obs['max_steps']} | Patients waiting: {obs['queue_length']}",
        f"",
        f"CURRENT PATIENT: {p['patient_id']}",
        f"Age: {p['age']} | Chief complaint: {p['chief_complaint']}",
        f"History: {p['history']}",
        f"",
        f"VITALS:",
        f"  HR: {v['heart_rate']} bpm | BP: {v['systolic_bp']} mmHg | SpO2: {v['spo2']}%",
        f"  RR: {v['respiratory_rate']} /min | GCS: {v['gcs']}/15 | Temp: {v['temperature']}°C",
    ]
    if v.get("lactate"):
        lines.append(f"  Lactate: {v['lactate']} mmol/L")
    if p.get("severity_hint"):
        lines.append(f"  Severity hint: {p['severity_hint']}")
    if p.get("deteriorating"):
        lines.append(f"  *** DETERIORATING — vitals worsening ***")

    lines += [
        f"",
        f"RESOURCES:",
        f"  ICU beds: {r['icu_beds_used']}/{r['icu_beds_total']} used | Ventilators: {r['ventilators_used']}/{r['ventilators_total']} used",
        f"  Ward beds: {r['ward_beds_used']}/{r['ward_beds_total']} used | Specialists: {r['specialists_available']} available",
        f"",
        f"Cumulative reward so far: {obs['episode_reward_so_far']:.3f}",
        f"Last feedback: {obs['last_step_feedback'][:120]}",
        f"",
        f"Respond with JSON only.",
    ]
    return "\n".join(lines)


def run_task(task_id: str) -> float:
    print(f"\n{'='*55}")
    print(f"  TASK: {task_id.upper()}")
    print("=" * 55)

    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": 42})
    resp.raise_for_status()
    obs = resp.json()

    total_reward = 0.0
    step = 0
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while not obs.get("done", False):
        patient_id = obs["current_patient"]["patient_id"]
        user_msg = format_patient_prompt(obs, task_id)
        messages.append({"role": "user", "content": user_msg})

        raw = call_llm(messages)
        messages.append({"role": "assistant", "content": raw})

        action = parse_action(raw, patient_id)

        print(f"\n  Step {step+1}: [{patient_id}] {obs['current_patient']['chief_complaint'][:50]}")
        print(f"    Decision: {action['decision']} | Rationale: {action.get('rationale','')[:80]}")

        step_resp = requests.post(f"{ENV_URL}/step", json=action)
        step_resp.raise_for_status()
        data = step_resp.json()

        reward = data["reward"]
        total_reward += reward
        obs = data["observation"]

        print(f"    Reward: {reward:+.3f} | {obs['last_step_feedback'][:100]}")
        step += 1

    avg = total_reward / max(step, 1)
    print(f"\n  {task_id} → avg score: {avg:.3f} over {step} patients (total: {total_reward:.3f})")
    return avg


def main():
    print("\n🏥 ClinicalTriageEnv — Baseline Inference")
    print(f"   Model:  {MODEL_NAME}")
    print(f"   API:    {API_BASE_URL}")
    print(f"   Server: {ENV_URL}\n")

    scores = {}
    for task in ["task1", "task2", "task3"]:
        try:
            scores[task] = run_task(task)
        except Exception as e:
            print(f"\n[ERROR] {task}: {e}")
            scores[task] = 0.0

    diff = {"task1": "Easy", "task2": "Medium", "task3": "Hard"}
    print("\n" + "=" * 55)
    print("  FINAL BASELINE SCORES")
    print("=" * 55)
    for t, s in scores.items():
        print(f"  {t} ({diff[t]:6s}): {s:.3f}")
    print(f"  Average:        {sum(scores.values())/3:.3f}")
    print("=" * 55)


if __name__ == "__main__":
    main()