# Architecture

This document describes the actual runtime architecture of the SIGMA platform
as it exists on the `integration` branch.

## Processing Pipeline Flow

```text
Uploaded Signal File
        │
        ▼
ml.input.pipeline.process_file()          ← WAV / raw-IQ ingestion and validation
        │
        ▼
dsp.extract_parameters()                  ← SNR, carrier offset, bandwidth, symbol rate
        │
        ▼
ml.inference.pipeline.analyze_file()      ← CNN modulation classification (top-k)
        │
        ▼
hypothesis.candidates.generate_candidates()  ← build testable HypothesisSpec list
        │
        ▼
hypothesis.evaluator.evaluate_hypothesis()   ← sync → demod → FEC → bitstream stages
        │
        ▼
hypothesis.ranking.rank_hypotheses()         ← final_score + integer rank
        │
        ▼
reporting.results.*                          ← assemble API response schemas
```

---

## Backend Package Responsibilities

### `backend/api/`
FastAPI routers — thin HTTP layer only, no business logic.

| File | Responsibility |
|------|---------------|
| `upload.py` | `POST /api/v1/upload` — file ingestion, ml.input validation, Signal persistence |
| `analysis.py` | `POST /api/v1/analysis/{signal_id}` — launch pipeline as BackgroundTask; `GET …/status` — poll |
| `results.py` | `GET /api/v1/results/{id}` and `…/report` — retrieve completed analysis data |
| `visualizations.py` | `GET /api/v1/visualizations/{signal_id}/{type}` — waveform / FFT / PSD / spectrogram from real DSP |
| `schemas.py` | Pydantic request/response models; field names mirror `frontend/src/types/index.ts` |
| `__init__.py` | Router exports |

### `backend/core/`
Application configuration via `pydantic-settings`.  All settings are loaded
from environment variables / `.env`.  MongoDB URI, upload directory, model path.

### `backend/db/`
Beanie ODM document models (`Signal`, `Analysis`).  Connection is non-fatal —
the application starts and operates fully without MongoDB configured.

### `backend/ingestion/`
**Superseded.**  WAV and IQ parsing is handled by `ml.input.pipeline`
(see `ml/input/`).  The stub files in this directory are retained as
placeholders and markers for the original design boundary.

### `backend/preprocessing/`
IQ signal preprocessing utilities.

| File | Responsibility |
|------|---------------|
| `normalize.py` | DC removal, power normalisation; defines canonical `[2, N]` float32 IQ format |
| `filtering.py` | Butterworth low-pass filter; `extract_signal()` baseband extraction |
| `segmentation.py` | Re-export shim: exposes `dsp.detection` symbols under the `preprocessing` namespace |

### `backend/dsp/`
All DSP signal-processing algorithms.

| File | Responsibility |
|------|---------------|
| `fft.py` | Windowed FFT; `canonical_to_complex()` conversion |
| `psd.py` | Welch PSD; noise floor and signal power helpers |
| `spectrogram.py` | STFT spectrogram via scipy |
| `detection.py` | PSD-threshold signal region detection (`SignalRegion`) |
| `extraction.py` | Re-export shim: `ExtractedSignal`, `extract_signal`, `lowpass_filter` from preprocessing |
| `bandwidth.py` | 99 % occupied bandwidth estimation |
| `carrier.py` | 4th-power CFO estimation; `remove_carrier_offset()` |
| `snr.py` | PSD-based SNR; M2M4 fallback |
| `symbol_rate.py` | Cyclostationary + autocorrelation symbol-rate estimation |
| `visualization.py` | Structured JSON-serializable visualization data (waveform, FFT, PSD, spectrogram, regions) |
| `pipeline.py` | End-to-end DSP orchestrator (`run_dsp_pipeline`) |
| `__init__.py` | Public API: `extract_parameters()`, `estimate_region_parameters()`, all visualization exports |
| `benchmark_dsp.py` | Standalone performance benchmark utility (not imported by runtime code) |

### `backend/synchronization/`
Full sync chain: RRC matched filter → feedforward phase correction → Costas loop → Gardner TED.
Public entry point: `synchronize()` in `__init__.py`.

### `backend/demodulation/`
Hard-decision BPSK and QPSK demodulators.  16-QAM and FSK are documented stubs.
Public entry point: `demodulate()` in `__init__.py`.

### `backend/fec/`
Rate-1/2 constraint-length-7 convolutional encoder and Viterbi decoder.
Bitstream statistical validity checks (entropy, balance, run-length).

### `backend/hypothesis/`
| File | Responsibility |
|------|---------------|
| `candidates.py` | Build `HypothesisSpec` list from ML top-k + DSP estimates |
| `evaluator.py` | Run sync → demod → FEC → bitstream stages; return `EvaluatedHypothesis` |
| `ranking.py` | Score and rank evaluated hypotheses; return sorted list of dicts |
| `generator.py` | `run_hypothesis_pipeline()` — full orchestrator called by the API background task |

### `backend/modulation/`
**Future responsibility.**  All files are documented stubs for modulation-specific
feature extraction (constellation mapping, cumulants, etc.).  No runtime code
imports from this package yet.

### `backend/reporting/`
Converts `Analysis` document data (or in-memory equivalents) into
`ResultsResponse` and `ReportResponse` Pydantic schemas for the API.

### `backend/main.py`
Thin FastAPI application entry point.  Registers routers, configures CORS,
initialises the upload directory, and attempts (non-fatal) DB connection on startup.

---

## ML Layer (`ml/`)

Separate from the backend packages.  Accessed by the backend via lazy imports:

- `ml.input.pipeline.process_file()` — signal ingestion and validation
- `ml.inference.pipeline.analyze_file()` — CNN modulation classification

The ML layer is not imported at module load time; imports happen inside
`async` background tasks and are isolated from the API startup path.

---

## Database

MongoDB Atlas via Motor / Beanie ODM.  **Optional** — the application
operates fully without a configured `MONGODB_URI`.  When not connected,
all state is held in in-memory Python dicts (`_UPLOAD_STORE`, `_ANALYSIS_STORE`).

---

## API Surface (`/api/v1/`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload a signal file |
| `POST` | `/analysis/{signal_id}` | Create and launch an analysis job |
| `GET`  | `/analysis/{analysis_id}/status` | Poll analysis status |
| `GET`  | `/results/{analysis_id}` | Retrieve full results |
| `GET`  | `/results/{analysis_id}/report` | Retrieve exportable JSON report |
| `GET`  | `/visualizations/{signal_id}/waveform` | Time-domain IQ waveform |
| `GET`  | `/visualizations/{signal_id}/fft` | FFT spectrum (magnitude dB) |
| `GET`  | `/visualizations/{signal_id}/psd` | PSD (Welch, dBW/Hz) |
| `GET`  | `/visualizations/{signal_id}/spectrogram` | STFT spectrogram matrix |
| `GET`  | `/health` | Health check |
