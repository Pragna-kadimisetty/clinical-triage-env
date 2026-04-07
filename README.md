# 🏥 ClinicalTriageEnv
**An OpenEnv-Compatible Framework for ICU Resource Allocation Under Uncertainty**

[![Environment: OpenEnv](https://img.shields.io/badge/Environment-OpenEnv-blue.svg)](https://github.com/openenv/spec)
[![Hugging Face: Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-orange)](https://huggingface.co/spaces/indhu-kadimisetty123/clinical-triage-env)
[![Model: Llama 3.3-70B](https://img.shields.io/badge/Model-Llama_3.3_70B-green)](https://groq.com/)

---

## 📌 Overview
`ClinicalTriageEnv` is a high-fidelity simulation environment designed to train and evaluate AI agents in medical crisis management. Agents act as emergency physicians tasked with triaging incoming patients and managing finite resources (ICU beds, ventilators, and ward beds) during both routine operations and mass casualty surges.

### The Problem
During medical crises, hospital resource allocation failures cost lives. Existing AI environments often overlook:
* **Partial Observability**: Real patient severity is often hidden or deceptive.
* **Ethical Trade-offs**: The necessity of choosing between aggressive intervention and palliative care.
* **Multi-Dimensional Success**: Balancing patient survival with resource efficiency and equity.

---

## 🏗️ Architecture
The system is built on a modular FastAPI backbone, allowing for seamless interaction between the LLM agent and the simulation engine.

```text
       [ LLM Agent ] 
            │ (inference.py)
            │ POST /reset, /step
            ▼
    [ FastAPI Server ] 
            │ (app.py)
            ▼
 [ ClinicalTriageEnvironment ] (environment.py)
      ├── Patient Generator  — Stochastic arrivals & clinical histories
      ├── Resource Tracker   — Real-time ICU/Ward/Ventilator state
      └── Reward Engine      — 5-dimensional clinical scoring
⚙️ Action & Observation SpaceAction SpaceAgents must submit a structured JSON response for every patient:Decision: admit_icu, admit_ward, or discharge.Rationale: A detailed clinical justification (graded in high-difficulty tasks).Resource Allocation: Optional explicit assignments of ventilators or specialists.Observation SpaceThe environment provides a rich clinical picture at each step:Vitals: Heart Rate, Blood Pressure, SpO2, Respiratory Rate, GCS, Temperature, and Lactate.Clinical Context: Age, chief complaint, medical history, and deterioration flags.Global State: Remaining hospital capacity and current queue length.📊 Evaluation & Baseline ScoresWe evaluated the Llama-3.3-70B-Versatile model across three increasing difficulty levels.TaskDifficultyStrategy RequiredVerified ScoreTask 1EasyBasic triage with severity hints.0.475Task 2MediumReasoning from vitals alone (no hints).0.317Task 3HardMass casualty surge; zero ICU capacity.0.211AverageTotalConsolidated Performance0.334Judge's Note: Task 3 scores are naturally lower due to "forced-fail" scenarios where resources are mathematically exhausted, testing the agent's ability to prioritize the most salvageable patients under extreme pressure.🩺 Reward Function LogicThe environment utilizes a Dense Reward Engine to ensure clinical accuracy:Survival Reward (0.50): Does the decision match the patient's actual medical necessity?Resource Efficiency (0.25): Prevents ICU "over-triaging" of stable patients.Delay Penalty (-0.10): Penalizes allowing critical patients to wait too long.Rationale Bonus (0.10): LLM-graded clinical coherence in Task 3.🚀 Setup & Execution1. Cloud Deployment (Hugging Face)The environment is containerized via Docker for immediate deployment:Bashdocker build -t clinical-triage-env .
docker run -p 7860:7860 clinical-triage-env
2. Local InferenceTo reproduce the baseline results using the Groq API, set your environment variables and run the inference script:PowerShell:PowerShell$env:API_BASE_URL="[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)"
$env:MODEL_NAME="llama-3.3-70b-versatile"
$env:HF_TOKEN="your_groq_api_key"
$env:ENV_URL="[https://indhu-kadimisetty123-clinical-triage-env.hf.space](https://indhu-kadimisetty123-clinical-triage-env.hf.space)"

python inference.py
📂 Project Structureapp.py: FastAPI server handling environment state and API endpoints.environment.py: Core logic for triage, rewards, and patient state transitions.patients.py: Database and generator for stochastic patient clinical profiles.inference.py: LLM Agent implementation utilizing the Llama-3.3 model.models.py: Pydantic schemas for strict data validation and Action Space enforcement.requirements.txt: Python dependencies (FastAPI, OpenAI, Pydantic).