# SIGMA — Signal Intelligence & Guided Modulation Analysis

An Automated RF Signal Analysis & Hypothesis Validation Platform.

## Project Description
SIGMA is a modular platform designed to ingest `.IQ` and `.wav` RF signal recordings, preprocess them, perform DSP feature estimation, and leverage machine learning to generate, evaluate, and rank demodulation hypotheses.

## High-Level Architecture

```text
  IQ / WAV
     ↓
 Ingestion
     ↓
DSP Analysis
     ↓
Feature Extraction
     ↓
 ML Assistance
     ↓
Hypothesis Testing
     ↓
  Validation
     ↓
Confidence Ranking
```

---

## Project Status

> [!IMPORTANT]
> This project is currently in **Phase 0 — Project Setup**. No digital signal processing or machine learning features are implemented yet.

---

## Folder Structure Overview

```text
SIGMA/
├── README.md                 # Project overview and run instructions
├── .gitignore                # Global ignore configuration
│
├── frontend/                 # React + TypeScript (Vite) UI dashboard
│   ├── README.md
│   └── src/                  # Components, pages, services, types
│
├── backend/                  # FastAPI web application
│   ├── README.md
│   ├── requirements.txt      # Python dependencies
│   ├── main.py               # API entry point & configuration
│   ├── api/                  # API routers (upload, analysis, results)
│   ├── ingestion/            # File format parsing (IQ, WAV)
│   ├── preprocessing/        # Signal filtering & normalization
│   ├── dsp/                  # DSP algorithms (FFT, PSD, spectrogram)
│   ├── modulation/           # ML classifier & signal feature extraction
│   ├── synchronization/      # Carrier & timing recovery loops
│   ├── demodulation/         # FSK, PSK, QAM demodulators
│   ├── hypothesis/           # Hypothesis generation, evaluation & ranking
│   ├── reporting/            # Analysis reporting tools
│   └── core/                 # Shared configuration and logging
│
├── ml/                       # ML models, features, training & inference code
├── datasets/                 # Local directory for raw, synthetic, processed datasets
├── models/                   # Serialized model weights (.pt, .pkl)
├── notebooks/                # Jupyter Notebooks for exploration and DSP prototypes
├── tests/                    # Testing suite (backend, frontend, integration)
└── docs/                     # Platform documentation
```

---

## Run Instructions

### Prerequisites
- **Node.js**: v18+ (tested on v24.13)
- **Python**: v3.10+ (tested on v3.11)

### 1. Backend Setup & Run
From the root of the project:
```bash
# Navigate to backend
cd backend

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload --port 8000
```
Verify the backend is running by navigating to `http://localhost:8000/health`.

### 2. Frontend Setup & Run
From the root of the project in a new terminal window:
```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```
Verify the frontend is running by navigating to `http://localhost:5173`.

---

## Development Roadmap

* **Phase 0 — Project Setup** (Current)
  * Project structure, FastAPI setup, React setup, configuration.
* **Phase 1 — File Ingestion**
  * IQ and WAV file parsing and metadata extraction.
* **Phase 2 — Signal Triage**
  * FFT, PSD, spectrogram, signal detection, SNR and bandwidth estimation.
* **Phase 3 — Dataset and Feature Pipeline**
  * Synthetic signal generation, feature extraction and dataset preparation.
* **Phase 4 — Modulation Recognition**
  * Feature-based baseline classifier followed by optional CNN classifier.
* **Phase 5 — Signal Recovery**
  * Carrier offset estimation, matched filtering and timing recovery.
* **Phase 6 — Demodulation**
  * Support FSK, BPSK, QPSK and 16-QAM.
* **Phase 7 — Hypothesis Validation**
  * Generate multiple signal hypotheses, evaluate downstream quality and rank results.
* **Phase 8 — GUI Integration and Demo**
  * Connect backend analysis to the frontend and create the final demonstration workflow.
