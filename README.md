# 🏥 ClinicalTriageEnv

**An OpenEnv-Compatible Framework for ICU Resource Allocation Under Uncertainty**

[![Environment: OpenEnv](https://img.shields.io/badge/Environment-OpenEnv-blue.svg)](https://github.com/openenv/spec)
[![Hugging Face: Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-orange)](https://huggingface.co/spaces/indhu-kadimisetty123/clinical-triage-env)
[![Model: Llama 3.3-70B](https://img.shields.io/badge/Model-Llama_3.3_70B-green)](https://groq.com/)

---

## 📌 Executive Summary

`ClinicalTriageEnv` is a high-fidelity simulation designed to evaluate AI agents in **high-stakes medical decision-making**.

Unlike standard benchmarks, this environment forces agents to manage:

* **Finite resources** (ICU beds, ventilators)
* **Partial observability**
* **Ethical scarcity**

---

## ⚠️ The Core Challenge

In a crisis, correct diagnosis is not enough.

> *If only one ICU bed is available and two critical patients arrive — who survives?*

This environment evaluates an agent’s ability to:

* Prioritize
* Justify
* Allocate under pressure

---

## 🏗️ System Architecture

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

## 🎮 Task Specifications

### Task 1 — Baseline Triage (Easy)

* 6 patients
* Severity hints available
* Resources abundant
* Goal: learn vitals → action mapping

---

### Task 2 — Multi-Resource Pressure (Medium)

* 9 patients
* No severity hints
* Moderate scarcity

Includes **deceptive cases** (e.g. anxiety mimicking critical distress)

Goal: reasoning from vitals alone

---

### Task 3 — Mass Casualty Surge (Hard)

* 12 patients
* No hints
* ICU + ventilators exhausted

Forces:

* Ethical triage
* Palliative decisions
* Justified prioritization

---

## 📊 Performance Benchmarks

| Task        | Difficulty | Strategy           | Score     |
| ----------- | ---------- | ------------------ | --------- |
| Task 1      | Easy       | Guided triage      | 0.475     |
| Task 2      | Medium     | Vitals reasoning   | 0.317     |
| Task 3      | Hard       | Ethical allocation | 0.211     |
| **Average** | —          | —                  | **0.334** |

**Note:** Task 3 includes forced-failure scenarios to test prioritization.

---

## ⚙️ Action Space

Agents output structured JSON:

* **Decision**:
  `admit_icu`, `admit_ward`, `discharge`, `palliative`, `defer`

* **Rationale**:
  Clinical reasoning (graded in Task 3)

* **Resource Allocation**:
  Optional explicit assignments

---

## 🔍 Observation Space

Each step provides:

### Patient

* Vitals: HR, BP, SpO2, RR, GCS, Temp, Lactate
* Age, complaint, history
* Deterioration flags

### System

* ICU / ward / ventilator capacity
* Queue length

👉 Severity is hidden in harder tasks (partial observability)

---

## 🩺 Reward Function

Dense multi-objective scoring:

* **Survival Reward (0.50)**
* **Resource Efficiency (0.25)**
* **Delay Penalty (-0.10)**
* **Rationale Bonus (0.10)**

Designed to prevent trivial or exploitable policies.

---

## 🚀 Setup & Execution

### Docker

```bash
docker build -t clinical-triage-env .
docker run -p 7860:7860 clinical-triage-env
```

---

### Local Inference (Groq)

```powershell
$env:API_BASE_URL="https://api.groq.com/openai/v1"
$env:MODEL_NAME="llama-3.3-70b-versatile"
$env:HF_TOKEN="your_groq_api_key"
$env:ENV_URL="https://your-huggingface-space-url"

python inference.py
```

---

## 📂 Project Structure

```text
clinical_triage_env/
├── app.py
├── environment.py
├── patients.py
├── inference.py
├── models.py
├── requirements.txt
```

---

## ✨ Key Contributions

* Partial observability by design
* Ethical decision modeling
* Resource-constrained reasoning
* Multi-objective reward system
* Reproducible + stochastic environment

---

## 📎 Links

* Hugging Face Space:
  https://huggingface.co/spaces/indhu-kadimisetty123/clinical-triage-env

---

## 📌 Final Note

Built for researchers and engineers working on **high-stakes AI decision systems**, where correctness, ethics, and resource constraints intersect.

---
