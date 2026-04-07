🏥 ClinicalTriageEnvAn OpenEnv environment for ICU resource allocation under uncertainty.Agents act as emergency physicians:Triage incoming patientsAllocate scarce hospital resources (ICU beds, ventilators, ward beds)Make ethically charged decisions during mass casualty eventsAll under partial observability.Why This Environment?Hospital resource allocation failures cost lives. During COVID-19, ICU shortages forced impossible decisions daily.No existing OpenEnv environment models:Partial observability: true patient severity is hidden in medium/hard tasksEthical trade-offs: palliative care vs. aggressive interventionMulti-dimensional reward: survival, resource efficiency, equity, and speedStochastic patient generation: reproducible yet randomizable episodesArchitectureinference.py (LLM Agent)
      │ POST /reset, /step, /state
      ▼
FastAPI Server (app.py)
      │
      ▼
ClinicalTriageEnvironment (environment.py)
      ├── Patient Generator (patients.py)   — stochastic arrivals
      ├── Resource Tracker                  — ICU/ward/vent state
      └── Reward Engine                     — 5-dimensional scoring
Action SpaceFieldTypeDescriptionpatient_idstringCurrent patient IDdecisionenumadmit_icu / admit_ward / dischargeresource_allocationdictOptional explicit resource assignmentsrationalestringFree-text clinical reasoning (graded in task3)priority_overrideintnullObservation SpaceEach step returns the current patient's full clinical picture plus resource state:Patient: age, chief complaint, full vitals (HR, BP, SpO2, RR, GCS, Temp, Lactate), history, deteriorating flag, severity hint (task1 only)Resources: ICU beds, ventilators, ward beds — used/totalEpisode meta: step, reward so far, last feedback, queue lengthTasks & ResultsTask 1 — Single-Resource Triage (Easy)6 patients, severity hints visible.Actual Score: 0.475Task 2 — Multi-Resource Pressure (Medium)9 patients, no hints, moderate scarcity.Includes deceptive cases (e.g. anxiety vs. true distress).Actual Score: 0.317Task 3 — Mass Casualty Surge (Hard)12 patients, no hints. All ICU beds full.rationale field is scored for clinical coherence.Actual Score: 0.211Reward FunctionDense, multi-dimensional. Cannot be gamed by naive policies.reward = survival_reward     (0.0 – 0.50)   # Does decision match true need?
       + resource_efficiency (0.0 – 0.25)   # Scarce resources to highest need
       + equity_penalty      (up to -0.08)  # Penalizes repeated deferral
       + delay_penalty       (up to -0.10)  # Penalizes deteriorating patients
       + rationale_bonus     (0.0 – 0.10)   # task3: reasoning quality
Setup & UsageDocker (Hugging Face Deployment)Bashdocker build -t clinical-triage-env .
docker run -p 7860:7860 clinical-triage-env
Run Baseline Inference (Local VS Code)To run the evaluation using the Groq Llama-3.3-70b model:PowerShell$env:API_BASE_URL="https://api.groq.com/openai/v1"
$env:MODEL_NAME="llama-3.3-70b-versatile"
$env:HF_TOKEN="your_groq_key_here"
$env:ENV_URL="https://indhu-kadimisetty123-clinical-triage-env.hf.space"

python inference.py
Final Baseline Scores (Verified)TaskDifficultyScore (Llama-3.3-70b)task1Easy0.475task2Medium0.317task3Hard0.211Average0.334Project Structureclinical_triage_env/
├── models.py         # Data structures (TriageAction, etc.)
├── patients.py       # Patient database and generator
├── environment.py    # Core triage logic and reward engine
├── app.py            # FastAPI Server
├── inference.py      # LLM Agent script
├── Dockerfile        # Container configuration
└── README.md         # Documentation