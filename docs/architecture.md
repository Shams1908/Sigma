# Architecture Design

This document details the architectural flow and structure of the SIGMA platform.

## Processing Pipeline Flow

The core processing pipeline progresses sequentially:

```text
+----------+     +--------------------+     +---------------+     +--------------------+     +--------------------+     +--------------------+
| Ingestion| --> | Preprocessing & DSP| --> | ML Classifier | --> | Hypothesis Engine  | --> | Decoder Validation | --> | Confidence Ranking |
+----------+     +--------------------+     +---------------+     +--------------------+     +--------------------+     +--------------------+
```

### 1. Ingestion (`backend/ingestion/`)
* **WAV Reader**: Parses standard audio format recordings containing baseband or intermediate frequency (IF) signals.
* **IQ Reader**: Ingests raw complex (I/Q) float or integer sample recordings.
* **Metadata Extraction**: Extracts sample rate, center frequency, timestamp, and duration info from files or associated sidecar metadata files.

### 2. Preprocessing & DSP (`backend/preprocessing/` & `backend/dsp/`)
* **Filtering & Normalization**: Applies bandpass filters and normalizes power levels.
* **Spectral Analysis**: Computes Fast Fourier Transform (FFT), Power Spectral Density (PSD), and Spectrograms.
* **Parameter Estimation**: Automatically estimates SNR, bandwidth, and carrier frequency offsets.

### 3. Machine Learning Assistant (`backend/modulation/` & `ml/`)
* Extracts statistical, spectral, and cyclostationary features.
* Runs lightweight classification models (e.g., Random Forest or CNN) to predict modulation types.

### 4. Synchronization & Demodulation (`backend/synchronization/` & `backend/demodulation/`)
* Performs carrier recovery (frequency/phase correction) and symbol timing recovery.
* Demodulates signals (FSK, BPSK, QPSK, 16-QAM) to produce symbol/bit streams.

### 5. Hypothesis Engine (`backend/hypothesis/`)
* Generates alternative parameter/modulation/decoding hypotheses.
* Evaluates downstream quality (e.g., error rate, transition metrics, sync stability).
* Ranks candidates by confidence score.

### 6. Reporting & GUI (`backend/reporting/` & `frontend/`)
* Packages results into JSON payloads.
* Displays spectrograms, waterfalls, constellations, and parameters on the React frontend dashboard.
