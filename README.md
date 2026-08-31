# SIGMA — Signal Intelligence & Guided Modulation Analysis

An Automated RF Signal Analysis & Hypothesis Validation Platform.

---

## Overview

SIGMA is a modular signal processing and machine learning platform designed to parse, validate, segment, and classify the modulation format of radio-frequency (RF) signals. The platform targets real-world modulation recognition in non-cooperative RF environments, providing both an end-to-end production-oriented inference adapter and a scientific research track to investigate domain adaptation.

The project incorporates two major pipelines:
1. **Production/MVP Inference Pipeline**: Agnostic file ingestion (WAV/IQ), canonical float32 representation parsing, window segmentation, and frozen 1D CNN prediction.
2. **ML Research/Experimentation Pipeline**: Impairment sweeps simulator, dynamic amplitude calibration, physical channel refinement, and synthetic-assisted training.

---

## Problem & Solution

### The Problem
Classifying RF signals in the real world is challenging due to channel impairments (carrier offsets, phase drifts, complex DC offsets, multi-path fading, and timing sync jitters). Training models purely on clean synthetic data results in a severe **domain gap**, leading to prediction collapse when evaluated on real-world acquisition data.

### The Solution
SIGMA addresses this by:
- Enforcing strict **architectural boundaries** separating format parsing from model-specific preprocessing.
- Building a **reproducible synthetic generator** mimicking real-world acquisition offsets.
- Designing a **dynamic normalization scheme** using checkpoint-cached parameters to prevent dataset-to-dataset drift.
- Providing a **modular aggregation strategy** averaging window-level probabilities to achieve robust file-level predictions.

---

## System Architecture

### 1. Production/MVP Inference Pipeline (File to Prediction)
```text
User File (WAV, NPY, NPZ, BIN, DAT)
   │
   ▼
Format Detection (detector.py)
   │
   ▼
WAV / IQ Adapter (wav.py, iq.py)
   │
   ▼
Canonical IQ representation [2, N] (dtype=float32)
   │
   ▼
Segmentation [M, 2, 128] (segment.py)
   │
   ▼
Dynamic Checkpoint Normalization (/ rms_factor)
   │
   ▼
M5 Raw IQ 1D CNN Inference (architecture.py)
   │
   ▼
Per-Window Softmax Probabilities [M, 11]
   │
   ▼
Probability Aggregation (Mean Probability Vector) (aggregation.py)
   │
   ▼
Final Modulation Class & Confidence
```

### 2. ML Research/Experimentation Pipeline
```text
RadioML 2016.10A Split / Synthetic Generator Sweeps
              │
              ▼
Synthetic Evaluation Dataset (M6.1)
              │
              ▼
Domain Gap Statistical Analysis (M6.3)
              │
              ▼
Data-Driven Calibration & Refinement (M6.4)
              │
              ▼
Synthetic-Assisted Training (M6.5)
              │
              ▼
Research Findings & Mismatch Insights
```

---

## Project Status

| Area | Status | Description |
|---|---|---|
| **Dataset Foundation** | **COMPLETE** | RadioML 2016.10A ingestion, splitting, and verification. |
| **Synthetic Generation** | **COMPLETE** | Configurable impairment sweeps, Wiener phase noise, and FIR filters. |
| **DSP Features** | **COMPLETE** | 36 deterministic hand-crafted features. |
| **Classical ML Baseline** | **COMPLETE** | HistGradientBoosting and Random Forest classifiers. |
| **Raw IQ CNN Baseline** | **COMPLETE** | PyTorch 1D CNN architecture and checkpointing. |
| **Synthetic/Domain Research** | **COMPLETE** | Cross-domain evaluation, gap analysis, and training sweeps. |
| **WAV/IQ Ingestion** | **COMPLETE** | Format detection, validation checks, and canonical `[2, N]` conversion. |
| **File → CNN Inference** | **COMPLETE** | End-to-end processing, probability aggregation, and CLI tool. |
| **Backend API Integration** | **INTEGRATION PENDING** | FastAPI routers are skeleton structures; backend endpoints are pending. |
| **Frontend Visualization** | **INTEGRATION PENDING** | React UI dashboard exists but is not connected to active ML endpoints. |
| **Production Deployment** | **NOT IMPLEMENTED** | Production hosting configurations are not implemented. |

---

## Milestone Progress & Historic Benchmarks

### Milestone M1: Dataset Foundation
- **Target**: Integrate the RadioML 2016.10A dataset (`raw/RML2016.10a_dict.pkl`).
- **Implementation**: In-memory caching, split validation, and deterministic partitioning (`base_seed=42`) into:
  - **Train**: 154,000 samples (70%)
  - **Validation**: 33,000 samples (15%)
  - **Test**: 33,000 samples (15%)

### Milestone M2: Synthetic Signal Generation
- Implemented a parameterized simulator mapping digital modulations to IQ arrays:
  - **Supported formats**: BPSK, QPSK, 8PSK, QAM16, QAM64.
  - **Nominal configuration**: RRC pulse shaping with rolloff `0.35`, symbol rate `100,000 Hz`, sample rate `800,000 Hz` (8 samples/symbol).
  - **Composition Order of Impairments**:
    $$\text{symbols} \rightarrow \text{Pulse Shaping (RRC)} \rightarrow \text{FIR Filter} \rightarrow \text{Phase Noise} \rightarrow \text{AWGN} \rightarrow \text{Offset Impairments}$$
  - **RF Impairments**: Carrier frequency offsets, phase offsets, DC offsets, timing delays, and IQ imbalances.
  - **Channel Refinements**: Configurable Wiener process phase noise and multi-tap FIR channel responses.

### Milestone M3: Feature Engineering
- Extraction of **36 deterministic DSP features** using vectorized NumPy math (no learned parameters) to prevent train-validation-test leakage:
  - **Amplitude statistics**: mean, variance, kurtosis, peak-to-average ratio.
  - **Phase statistics**: variance, kurtosis, histogram bin counts.
  - **Frequency statistics**: variance, mean.
  - **Autocorrelation**: complex values at lags 1, 2, 4, 8, 16.
  - **Cumulants**: Normalized C40.

### Milestone M4: Classical ML Baseline
- Trained classical models on the 36 DSP features:
  - **Champion**: `HistGradientBoostingClassifier`
  - **Historical Benchmark on Real Test Set**:
    - **Accuracy**: `0.5494`
    - **Macro F1**: `0.5656`

### Milestone M5: Raw IQ CNN Baseline
- Trained a 1D CNN classifier directly on Raw IQ samples `[2, 128]` using dynamic training-only RMS scaling:
  - **Architecture**: Conv1D blocks $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool $\rightarrow$ Global Average Pooling $\rightarrow$ Dense classification head.
  - **Historical Benchmark on Real Test Set**:
    - **Accuracy**: `0.5668`
    - **Macro F1**: `0.5844`

### Milestone M6: Synthetic Data & Domain Gap Research (Research Track)
- **M6.1 (Synthetic Evaluation Dataset)**: Generated a dataset of 18,000 samples (covering 5 classes, 7 impairment parameters, and 100 seeds per condition).
- **M6.2 (Cross-Domain Validation)**: Evaluated the frozen M5 CNN on synthetic signals, revealing a severe **domain gap** (Macro F1 dropped to **0.1244**).
- **M6.3 (Domain Gap Analysis)**: Pinpointed root causes: a 140x power scaling mismatch, a phase difference distribution kurtosis mismatch (5.77 real vs -0.18 synthetic), and imaginary autocorrelation carrier drift offsets.
- **M6.4.1 (Amplitude Calibration)**: Implemented dynamic clean-reference RMS matching. Running frozen CNN predictions on calibrated signals improved Macro F1 to **0.1989** and resolved prediction collapse.
- **M6.4.2 (Realistic Refinement)**: Added Wiener phase noise and FIR channel modeling, yielding **0.1958 F1**.
- **M6.5 (Synthetic-Assisted Training)**: Ran controlled 5-class training sweeps:
  - *Real-only baseline*: **0.6403 F1**.
  - *Synthetic-only*: Generalizes to real data at **0.2942 F1**.
  - *Mixed ratios*: Performance degrades as synthetic ratio increases (10% ratio drops F1 to **0.5603**).
  - *Pretrain-Finetune*: Pretraining on synthetic and fine-tuning on real data achieves **0.6372 F1**.
  - *Verification*: Real + 10% Original Synthetic yields **0.4337 F1**, while Real + 10% Calibrated/Refined Synthetic achieves **0.5704 / 0.5603 F1**, proving that dynamic calibration significantly reduces mismatch degradation (+0.13 F1).

### Milestone M7.1: Universal Signal Ingest Adapter
- Implemented file format detection, container signature checking, validation, and canonical transformation to shape `[2, N]` and float32.
- **Supported Formats**: Stereo WAV (channel 0 = I, channel 1 = Q; mono is rejected), NPY, NPZ, and raw binary format (`.bin`/`.dat`) with explicit config.
- **Segmentation**: Configurable window slicing (defaults to 128). Short inputs produce validation errors unless zero-padding is configured (`pad_short=True`). Long inputs generate multiple windows and discard residuals.

### Milestone M7.2: End-to-End Inference Pipeline
- Connected the input adapter with the cached frozen CNN model.
- **Dynamic Normalization**: Automatically retrieves the scaling factor from the loaded checkpoint file (`models/m5_iq_cnn.pt`) to divide incoming segments. No scaling is computed from user file to avoid inference drift.
- **Aggregation**: Averages probability vectors across all segmented windows:
  $$\mathbf{p}_{\text{avg}} = \frac{1}{M}\sum_{m=1}^{M}\mathbf{p}_m$$
  and performs argmax to resolve file-level prediction. Returns complete metadata, top-5 predictions, and entropy measures.

---

## External Validation Dataset

An external, real-world validation dataset has been configured locally for independent generalization checks:
* **Filename**: `subset_test.h5`
* **Structure**: `X` of shape `(80000, 1024, 2)` (float16), `y_mod`, `y_snr`, and `y_chan`.
* **Label Mapping**:
  - `0`: BPSK, `1`: QPSK, `2`: QAM, `3`: GMSK, `4`: OFDM, `5`: NBFM, `6`: WBFM.
* **Channel mapping**: `0`: clean, `1`: multipath.
* **Generalization Role**: This dataset represents unseen out-of-domain signals and will be utilized to validate final model generalization. No validation results are claimed yet.

---

## Testing

SIGMA incorporates a comprehensive unit and integration test suite:
- **Total Test Count**: **`101 / 101 passed`**
- **Coverage**: Loader integrity, RRC pulse shaping, offsets impairments, 36 DSP features math, baseline ML training, 1D CNN training splits, calibration matching, Wiener processes, M7.1 input pipeline structure, and M7.2 end-to-end inference correctness.

---

## Installation & Setup

### Prerequisites
- **Node.js**: v18+
- **Python**: v3.11
- **PyTorch**: v2.13.0 (CPU version)

### Installation
From the root directory:
```powershell
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running SIGMA

### 1. Execute Complete Test Suite
Verify environment state and pass all tests:
```powershell
set PYTHONPATH=%CD%
.\backend\venv\Scripts\pytest
```

### 2. Run M7.2 End-to-End Prediction CLI
Test and analyze any input signal file:
```powershell
set PYTHONPATH=%CD%
.\backend\venv\Scripts\python.exe -m ml.inference.pipeline_test "path/to/signal.wav"
```

### 3. Python API Integration Example
```python
from ml.inference import analyze_file
from ml.input.types import PipelineConfig

# 1. Configure input pipeline
config = PipelineConfig(segment_length=128)

# 2. Execute analysis
result = analyze_file("path/to/signal.wav", config)

# 3. Access results
print(f"Modulation: {result.predicted_class_name} ({result.confidence * 100:.1f}%)")
print(f"Windows Analyzed: {result.num_windows}")
```

---

## Project Structure

```text
SIGMA/
├── README.md                     # Root project documentation
├── .gitignore                    # Global file exclusions
├── models/                       # Checkpoints directory
│   └── m5_iq_cnn.pt              # Frozen CNN model checkpoint
├── backend/                      # FastAPI application
├── frontend/                     # React dashboard
├── datasets/                     # Directory for signal splits
├── results/                      # Output csv summaries and plots
│   └── ml/m6/                    # M6 research plots
├── tests/                        # Full test suite
└── ml/                           # Signal Intelligence package
    ├── dataset/                  # M1 splits loading
    ├── features/                 # M3 feature extraction
    ├── generators/               # M2 synthetic generator
    ├── baselines/                # M4 classical baseline
    ├── cnn_model/                # M5 Raw IQ CNN baseline
    ├── synthetic/                # M6 evaluation and calibration
    ├── synthetic_training/       # M6.5 synthetic-assisted training
    ├── input/                    # M7.1 Universal Ingestion pipeline
    └── inference/                # M7.2 End-to-end predictions
```

---

## Current Limitations

- **Baseline CNN Classification Space**: The model output space is restricted to the 11 classes utilized by M5. Incoming signals outside this space are still forced onto these classes.
- **Raw Binary Configuration Required**: BIN/DAT files cannot be parsed without explicit `BinaryIQConfig` specifications. Silent guessing is disabled.
- **Mono WAV files Rejected**: Ingestion rejects single-channel WAVs to prevent ambiguity.
- **Frontend & Backend API Integration**: Integrations are pending; the ML layer is currently CLI-based.

---

## Future Work

- **Selective Channel Modeling**: Implementing multi-path selective fading and timing synchronization loop tracking in the generator.
- **Broader Domain Adaptation**: Investigating domain adversarial training to align synthetic-to-real distributions.
- **Production API Integration**: Connecting the `ml.inference` package to the FastAPI endpoints andReact frontend UI.
