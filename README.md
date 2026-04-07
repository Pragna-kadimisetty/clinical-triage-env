# 🏥 ClinicalTriageEnv

**An OpenEnv-Compatible Framework for ICU Resource Allocation Under Uncertainty**

[![Environment: OpenEnv](https://img.shields.io/badge/Environment-OpenEnv-blue.svg)](https://github.com/openenv/spec)
[![Hugging Face: Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-orange)](https://huggingface.co/spaces/indhu-kadimisetty123/clinical-triage-env)
[![Model: Llama 3.3-70B](https://img.shields.io/badge/Model-Llama_3.3_70B-green)](https://groq.com/)

---

## 📌 Overview

`ClinicalTriageEnv` is a high-fidelity simulation environment designed to train and evaluate AI agents in **medical crisis decision-making**.

Agents operate as emergency physicians responsible for:

* Triaging incoming patients
* Allocating scarce hospital resources (ICU beds, ventilators, ward beds)
* Making ethically constrained decisions under uncertainty

The environment simulates both **routine hospital flow** and **mass casualty scenarios**, emphasizing real-world complexity.

---

## ⚠️ The Problem

During healthcare crises, poor resource allocation directly impacts survival outcomes. Most existing AI environments fail to model:

* **Partial Observability**
  True patient severity is often hidden or misleading

* **Ethical Trade-offs**
  Balancing aggressive intervention vs. palliative care

* **Multi-Dimensional Objectives**
  Survival, efficiency, fairness, and timeliness must all be optimized

This environment addresses these gaps with a clinically grounded simulation.

---

## 🏗️ Architecture

The system is modular and API-driven, built on a FastAPI backbone:

```text
       [ LLM Agent ] 
            │ (inference.py)
            │ POST /reset, /step
            ▼
    [ FastAPI Server ] 
            │ (app.py)
            ▼
 [ ClinicalTriageEnvironment ] (environment.py)
      ├── Patient Generator  — stochastic arrivals & histories
      ├── Resource Tracker   — ICU / ward / ventilator state
      └── Reward Engine      — multi-dimensional scoring
```

---

## ⚙️ Action Space

Agents must return structured JSON decisions per patient:

| Field                   | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| **Decision**            | `admit_icu`, `admit_ward`, `discharge`, `palliative`, `defer` |
| **Rationale**           | Clinical reasoning (graded in high-difficulty tasks)          |
| **Resource Allocation** | Optional explicit assignment of resources                     |

---

## 🔍 Observation Space

Each timestep provides a comprehensive clinical snapshot:

### Patient Data

* Vitals: HR, BP, SpO2, RR, GCS, Temperature, Lactate
* Demographics: age, complaint, medical history
* Flags: deterioration indicators

### System State

* ICU / ward / ventilator capacity
* Queue length and episode progress

👉 **Partial observability**: severity signals are hidden in harder tasks.

---

## 📊 Evaluation & Baselines

Performance evaluated using the **Llama-3.3-70B-Versatile** model:

| Task        | Difficulty | Strategy Required           | Score     |
| ----------- | ---------- | --------------------------- | --------- |
| Task 1      | Easy       | Severity-guided triage      | 0.475     |
| Task 2      | Medium     | Reasoning from vitals       | 0.317     |
| Task 3      | Hard       | Resource-constrained triage | 0.211     |
| **Average** | —          | —                           | **0.334** |

**Note:** Task 3 includes *forced-failure scenarios* where resources are exhausted, testing prioritization under extreme scarcity.

---

## 🩺 Reward Function

The environment uses a **dense, multi-objective reward system**:

* **Survival Reward (0.50)** → clinical correctness
* **Resource Efficiency (0.25)** → optimal allocation
* **Delay Penalty (-0.10)** → penalizes late intervention
* **Rationale Bonus (0.10)** → evaluates reasoning quality

This prevents trivial or exploitative policies.

---

## 🚀 Setup & Execution

### 1. Docker Deployment (Recommended)

```bash
docker build -t clinical-triage-env .
docker run -p 7860:7860 clinical-triage-env
```

---

### 2. Local Inference

Set environment variables:

```powershell
$env:API_BASE_URL="https://api.groq.com/openai/v1"
$env:MODEL_NAME="llama-3.3-70b-versatile"
$env:HF_TOKEN="your_groq_api_key"
$env:ENV_URL="https://indhu-kadimisetty123-clinical-triage-env.hf.space"

python inference.py
```

---

## 📂 Project Structure

```text
clinical_triage_env/
├── app.py            # FastAPI server
├── environment.py    # Core simulation logic
├── patients.py       # Patient generator
├── inference.py      # LLM agent
├── models.py         # Pydantic schemas
├── requirements.txt  # Dependencies
```

---

## ✨ Key Contributions

* **Realistic clinical decision modeling** under uncertainty
* **Ethics-aware triage simulation**
* **Partial observability by design**
* **Multi-objective reward system**
* **Reproducible + stochastic environment**

---

## 📎 Links

* 🔗 Hugging Face Space:
  https://huggingface.co/spaces/indhu-kadimisetty123/clinical-triage-env

---

## 📌 Final Note

This environment is designed for **researchers and engineers building high-stakes decision-making agents**, particularly in domains where uncertainty, ethics, and resource constraints intersect.
