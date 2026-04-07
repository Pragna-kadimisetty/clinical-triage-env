"""
inference.py — Baseline inference for ClinicalTriageEnv.
Uses OpenAI-compatible client. Runs all 3 tasks end-to-end.

Required env vars:
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — Model identifier
  HF_TOKEN      — API key (Mandatory)

Optional:
  ENV_URL       — Environment server URL (default: http://localhost:7860)
"""

import os
import json
import requests
from openai import OpenAI

# 1. UPDATED: Environment Variables as per Hackathon Requirements
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "") # Use HF_TOKEN as per rules
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an experienced emergency physician making rapid triage decisions.

For each patient presented, analyze their vitals and decide:

DECISIONS (pick exactly one):
- admit_icu
- admit_ward
- discharge

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
- If ICU beds are 0 remaining, use ward instead
- Justify every decision with clinical data from the vitals
- Do not add any text before or after the JSON.
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
        if clean.startswith("json"):
            clean = clean[4:].strip()
            
        action = json.loads(clean.strip())
        action["patient_id"] = patient_id
        
        valid = ["admit_icu", "admit_ward", "discharge"]
        if action.get("decision") not in valid:
            action["decision"] = "admit_ward"
            
        if not isinstance(action.get("resource_allocation"), dict):
            action["resource_allocation"] = {}
        return action
    except Exception as e:
        return {
            "patient_id": patient_id,
            "decision": "admit_ward",
            "resource_allocation": {},
            "rationale": f"Parse error: {e}",
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
        f"   HR: {v['heart_rate']} bpm | BP: {v['systolic_bp']} mmHg | SpO2: {v['spo2']}%",
        f"   RR: {v['respiratory_rate']} /min | GCS: {v['gcs']}/15 | Temp: {v['temperature']}°C",
    ]
    if v.get("lactate"):
        lines.append(f"   Lactate: {v['lactate']} mmol/L")
    if p.get("severity_hint"):
        lines.append(f"   Severity hint: {p['severity_hint']}")
    if p.get("deteriorating"):
        lines.append(f"   *** DETERIORATING — vitals worsening ***")

    lines += [
        f"",
        f"RESOURCES:",
        f"   ICU beds: {r['icu_beds_used']}/{r['icu_beds_total']} used | Ward beds: {r['ward_beds_used']}/{r['ward_beds_total']} used",
        f"",
        f"Cumulative reward so far: {obs['episode_reward_so_far']:.3f}",
        f"Last feedback: {obs['last_step_feedback'][:120]}",
        f"",
        f"Respond with JSON only.",
    ]
    return "\n".join(lines)

def run_task(task_id: str):
    # 2. UPDATED: Start Log
    print(f"[START] task={task_id} env=clinical-triage-env model={MODEL_NAME}")

    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": 42})
        resp.raise_for_status()
        obs = resp.json()

        rewards_list = []
        step_count = 0
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        while not obs.get("done", False):
            step_count += 1
            patient_id = obs["current_patient"]["patient_id"]
            user_msg = format_patient_prompt(obs, task_id)
            messages.append({"role": "user", "content": user_msg})

            raw = call_llm(messages)
            messages.append({"role": "assistant", "content": raw})

            action = parse_action(raw, patient_id)

            step_resp = requests.post(f"{ENV_URL}/step", json=action)
            step_resp.raise_for_status()
            data = step_resp.json()

            reward = data["reward"]
            done = data["done"]
            obs = data["observation"]
            
            rewards_list.append(reward)

            # 3. UPDATED: Step Log (Lowercase booleans, 2 decimal rewards)
            reward_val = f"{reward:.2f}"
            done_val = str(done).lower() 
            print(f"[STEP] step={step_count} action={action['decision']} reward={reward_val} done={done_val} error=null")

        # 4. UPDATED: End Log
        success_val = "true" if (sum(rewards_list)/max(len(rewards_list), 1)) > 0.6 else "false"
        rewards_str = ",".join([f"{r:.2f}" for r in rewards_list])
        print(f"[END] success={success_val} steps={step_count} rewards={rewards_str}")

    except Exception as e:
        # Fallback end log if something crashes
        print(f"[END] success=false steps=0 rewards=0.00 error={str(e)}")

def main():
    for task in ["task1", "task2", "task3"]:
        run_task(task)

if __name__ == "__main__":
    main()