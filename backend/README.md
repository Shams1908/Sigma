# SIGMA FastAPI Backend

This directory houses the FastAPI web backend for the SIGMA platform.

## Structure
* `api/`: API Routers mapping HTTP endpoints to underlying signal analysis tools.
* `core/`: Shared config settings (`config.py`).
* `ingestion/`: Input handlers for IQ and WAV recordings.
* `preprocessing/`: Time-domain noise reduction, normalization, and segment markers.
* `dsp/`: Frequency-domain calculations (FFT, PSD, spectrograms, SNR, carrier/symbol estimation).
* `modulation/`: Features and decision classifiers to identify modulations.
* `synchronization/`: Loops to recover timing and carrier offset.
* `demodulation/`: Symbol-to-bit maps.
* `hypothesis/`: Generators and testers to score signal configurations.
* `reporting/`: Serialization of metrics into final reports.

## Getting Started

### 1. Set up Environment
Ensure you have Python 3.10+ installed.
```bash
python -m venv .venv
# Activate
# Windows:
.venv\Scripts\activate
# Unix:
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run FastAPI Application
```bash
uvicorn main:app --reload --port 8000
```
API endpoints will load on `http://localhost:8000`. Access `http://localhost:8000/docs` for the interactive Swagger documentation.
