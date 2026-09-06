# SIGMA — Signal Intelligence & Guided Modulation Analysis

An Automated RF Signal Intelligence, Modulation Recognition, and Channel Characterization Platform.

---

## 1. Core Vision

**SIGMA** is envisioned as an autonomous, end-to-end Radio Frequency (RF) Signal Intelligence (SIGINT) platform. Rather than functioning merely as an isolated modulation classifier, SIGMA is designed to ingest arbitrary, unknown RF waveforms from heterogeneous sources, automatically characterize what the signal is, evaluate how reliable that determination is, and quantify what physical and hardware channel impairments affected the transmission along the propagation path.

### The Big System Vision
```text
                          UNKNOWN RF SIGNAL
             (SDR Stream, WAV, Raw IQ, Capture File)
                                │
                                ▼
                       UNIVERSAL INGESTION
             (Format Detection, Validation, Canonicalization)
                                │
                                ▼
                      SIGNAL IDENTIFICATION
                    & QUALITY PRE-SCREENING
                                │
                                ▼
                       SIGNAL PREPROCESSING
            (Filtering, Framing, Dynamic Scaling)
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
             RAW IQ WAVEFORM          DSP FEATURES
             (Deep Learning)       (Domain Statistics)
                     │                     │
                     └──────────┬──────────┘
                                ▼
                    MULTI-MODEL SIGNAL ENGINE
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
    MODULATION            SIGNAL QUALITY          CHANNEL & RF
  CLASSIFICATION            ASSESSMENT             IMPAIRMENTS
 (Type, Top-K, P(c))    (SNR, Power, Entropy)  (CFO, Imbalance, Jitter)
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                ▼
                     EVIDENCE FUSION ENGINE
                                │
                                ▼
                      INTELLIGENCE DASHBOARD
                                │
                 ┌──────────────────────────────┐
                 │  Modulation: QPSK (94.2%)    │
                 │  Estimated SNR: +14.8 dB     │
                 │  Signal Quality: HIGH        │
                 │  Detected CFO: +120 Hz       │
                 │  IQ Imbalance: 0.05 amp      │
                 │  Channel: Multipath (Mild)   │
                 │  Entropy: 0.18 (Certain)     │
                 │  Spectrum / Constellation    │
                 │  Anomaly Indicators: None    │
                 └──────────────────────────────┘
```

> [!NOTE]
> The diagram above represents the **intended final platform architecture**. The current repository implementation provides the complete, working machine learning and ingestion core (Milestones M1–M7.2), establishing the foundation upon which multi-model fusion and real-time backend services are built.

---

## 2. The Problem

In modern electronic warfare, spectrum monitoring, and non-cooperative communications, identifying transmitted signals presents significant operational challenges:
* **Manual Inspection Bottleneck**: Traditional RF signal analysis relies heavily on expert operators manually examining instantaneous spectrograms, Fast Fourier Transforms (FFT), and IQ constellation diagrams.
* **Complex Environmental Degradation**: Real-world signals rarely resemble clean textbook models. They are distorted by carrier frequency offsets (CFO), carrier phase jitter, local oscillator drifts, receiver IQ amplitude/phase imbalances, multi-path fading, and thermal noise.
* **Toolchain Fragmentation**: Analysts frequently switch between separate utilities for file conversion, DSP parameter estimation, and classification, lacking an integrated pipeline.
* **Uncertainty Blind Spots**: Traditional modulation classifiers output a single categorical label without reporting whether the signal was severely degraded or completely out-of-distribution (OOD).

**SIGMA automates this entire pipeline**—transforming uncharacterized raw RF captures into structured, actionable intelligence.

---

## 3. The Big Solution

SIGMA replaces fragmented manual workflows with a unified, data-driven analysis engine. The platform is architected around five pillars:

1. **Universal Format-Agnostic Ingestion**: Ingests files across formats (stereo WAV, NumPy arrays, raw binary dumps, and future streaming Software-Defined Radios) without requiring manual reshaping or feature preparation.
2. **Multi-Representation Analysis**: Analyzes signals simultaneously in the raw time-domain (IQ), the handcrafted statistical domain (36 DSP features), and the spectral domain (PSDs and spectrograms).
3. **Joint Modulation & Impairment Characterization**: Simultaneously answers *"What modulation is this?"* and *"What physical distortions did this signal experience?"*
4. **Data-Centric Domain Alignment**: Integrates a parameterized synthetic signal generator with a closed-loop research feedback framework to isolate and bridge the synthetic-to-real domain gap.
5. **Calibrated Confidence & Explainability**: Delivers class probability distributions, top-k alternatives, Shannon entropy measurements, and visual evidence (constellations, PSDs) to provide trustworthy insights.

---

## 4. End-to-End System Architecture

SIGMA cleanly isolates the **Production Inference Pipeline** from the **ML Research Track**:

```text
========================================================================================
                        PRODUCTION / MVP INFERENCE PIPELINE
========================================================================================

  User File Upload (WAV, NPY, NPZ, BIN, DAT)
     │
     ▼
  Format & Signature Detection (ml/input/detector.py)
     │
     ▼
  WAV / IQ Parser & Validator (ml/input/wav.py, ml/input/iq.py)
     │
     ▼
  Canonical Complex Baseband IQ [2, N] (dtype=float32)
     │
     ▼
  Non-Overlapping Window Segmentation [M, 2, 128] (ml/input/segment.py)
     │
     ▼
  Dynamic Checkpoint-Driven Normalization (/ rms_factor)
     │
     ▼
  1D Raw IQ Deep CNN Batch Inference (ml/cnn_model/architecture.py)
     │
     ▼
  Per-Window Softmax Logits & Probabilities [M, 11]
     │
     ▼
  Element-Wise Mean Probability Vector Aggregation (ml/inference/aggregation.py)
     │
     ▼
  Final Modulation Prediction, Confidence, Entropy, & Top-K Ranking

========================================================================================
                         ML RESEARCH & DOMAIN-GAP TRACK
========================================================================================

  RadioML 2016.10A (M1) ───┐
                           ▼
  Parametric Synthetic ──► Real-vs-Synthetic Domain Gap Analysis (M6.3)
  Generator (M2, M6.1)     (140x Scale Mismatch, Phase Kurtosis Discrepancy)
                           │
                           ▼
                         Data-Driven Amplitude Calibration (M6.4.1)
                         & Physical Channel Refinement (M6.4.2)
                           │
                           ▼
                         Synthetic-Assisted Training Sweeps (M6.5)
                         (Real-Only vs Mixed Ratios vs Pretraining)
                           │
                           ▼
                         Iterative Generator Feedback Loop
```

---

## 5. Universal Signal Ingestion Layer

The ingestion layer ([`ml/input/`](ml/input/)) standardizes raw signals into a canonical mathematical representation before downstream processing:

* **Format Detection**: Inspects container headers and binary magic numbers (`RIFF/WAVE`, `\x93NUMPY`, `PK\x03\x04` ZIP) rather than relying solely on file extensions.
* **WAV Support**: Strictly processes **stereo WAV files** where Channel 0 represents In-phase ($I$) and Channel 1 represents Quadrature ($Q$). Dynamically extracts sampling rate and rejects mono files to eliminate ambiguity.
* **Array & Binary Support**: Reads `.npy`, `.npz`, and raw binary `.bin`/`.dat` files, mapping layouts (`[2, N]`, `[N, 2]`, complex `[N]`, and pre-framed `[N, 2, 128]`) into the canonical form.
* **Canonical Internal Representation**:
  - **Layout**: Array of shape `[2, N]` (Row 0 = $I$, Row 1 = $Q$).
  - **Dtype**: `float32`.
  - **Integrity**: Guaranteed finite numeric values (zero NaNs, zero Infs, non-empty).
* **Deterministic Segmentation**: Slices continuous `[2, N]` arrays into non-overlapping `[M, 2, 128]` windows. Long recordings generate $M = \lfloor N / 128 \rfloor$ segments, discarding incomplete trailing residuals. Short signals trigger clear validation errors unless zero-padding is explicitly requested (`pad_short=True`).

---

## 6. Multi-Representation Signal Analysis Engine

Rather than forcing the platform into a single representation, SIGMA combines complementary analytical views:

```text
                                CANONICAL IQ [2, N]
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
  RAW TIME-DOMAIN                 DSP STATISTICAL                 TIME-FREQUENCY /
   IQ WAVEFORMS                       FEATURES                        SPECTRAL
  (M5 RawIQCNN)                 (M3 36-Feature Set)              (FFT, PSD Curves)
        │                                │                                │
        ▼                                ▼                                ▼
Deep Spatial & Temporal         Physics-Based Cumulants,        Spectral Bandwidth,
Phase-Trajectory Shapes         Kurtosis, & Autocorrelations     Carrier Peaks, SNR
```

1. **Raw IQ Waveforms**: Directly feeds continuous in-phase and quadrature trajectories into deep 1D convolutional layers, preserving subtle phase transitions and instantaneous constellation paths.
2. **DSP Feature Statistics**: Evaluates 36 vectorized mathematical descriptors (cumulants, amplitude kurtosis, phase variance, autocorrelation lags) without learned parameters.
3. **Spectral Descriptors**: Computes Power Spectral Density (PSD) roll-offs, signal bandwidths, and spectral energy concentrations.

---

## 7. Modulation Recognition & Classification

### Supported Baseline Classes (RadioML 2016.10A)
The primary M5 baseline CNN classifies signals across **11 modulation formats**:
* **Digital Constellations**: BPSK, QPSK, 8PSK, 16-QAM, 64-QAM, 4-PAM, CPFSK, GFSK.
* **Analog Broadcasts**: WBFM, AM-DSB, AM-SSB.

### Model Architecture (`RawIQCNN`)
A dedicated 1D Deep Convolutional Neural Network optimized for raw complex baseband waveforms:
* **Input Layer**: `[Batch, 2, 128]`
* **Block 1**: `Conv1D(2 -> 64, kernel=7, pad=3)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool1d(2)` (128 $\rightarrow$ 64)
* **Block 2**: `Conv1D(64 -> 128, kernel=5, pad=2)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool1d(2)` (64 $\rightarrow$ 32)
* **Block 3**: `Conv1D(128 -> 256, kernel=3, pad=1)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU`
* **Pooling**: Global Average Pooling (GAP) across time dimension (`[Batch, 256, 32]` $\rightarrow$ `[Batch, 256]`)
* **Classification Head**: `Linear(256 -> 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(128 -> 11)`

---

## 8. RF Impairment & Channel Characterization

A core design objective of the final SIGMA platform is distinguishing **transmitter modulation schemes** from **channel/hardware distortions**:

```text
                           OBSERVED SIGNAL
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
    TRANSMITTER PROPERTIES              CHANNEL / HARDWARE EFFECTS
   • Symbol constellation              • Carrier Frequency Offset (CFO)
   • Pulse shaping filter              • Phase noise / oscillator drift
   • Raw bit encoding                  • Multi-path delay spread
                                       • IQ gain & phase imbalance
                                       • Complex DC offset bias
```

The platform's parameterized simulator directly models and isolates each impairment parameter:
* **Carrier Frequency Offset (CFO)**: Simulates Doppler shifts and receiver frequency mistuning:
  $$x_{\text{cfo}}[n] = x[n] \cdot e^{j 2 \pi \Delta f n / f_s}$$
* **Carrier Phase Drift & Wiener Phase Noise**: Simulates local oscillator instabilities via cumulative Gaussian random walks:
  $$\phi[n] = \phi[n-1] + w[n], \quad w[n] \sim \mathcal{N}(0, \sigma_{\phi}^2)$$
* **IQ Amplitude & Phase Imbalance**: Models front-end analog quadrature mixer non-idealities:
  $$x_{\text{imb}}[n] = (1 + \alpha) \cdot I[n] + j (1 - \alpha) \cdot [Q[n] \cos(\Delta \phi) - I[n] \sin(\Delta \phi)]$$
* **Complex DC Offset**: Simulates direct-conversion receiver LO leakage ($I_{\text{dc}} + j Q_{\text{dc}}$).
* **Fractional Timing Offsets**: Evaluates non-integer symbol sampling errors using polyphase filter banks.
* **FIR Channel Response**: Convolves waveforms with multi-tap channel impulse responses to model band-limiting and static multi-path reflections.

---

## 9. Signal Quality, Uncertainty & Explainability

SIGMA avoids "black-box" decision making by providing rich uncertainty and quality metrics alongside every classification:

* **Element-Wise Probability Aggregation**: Averages prediction vectors across all valid windows to prevent single-window anomalies from dominating:
  $$\mathbf{p}_{\text{avg}} = \frac{1}{M} \sum_{m=1}^{M} \mathbf{p}_m, \quad \hat{y} = \operatorname{argmax}(\mathbf{p}_{\text{avg}})$$
* **Window Prediction Distribution**: Tracks individual segment decisions across time, identifying temporal modulation shifts or transient interference.
* **Mean Prediction Entropy**: Computes average Shannon entropy across windows to quantify classification ambiguity:
  $$\bar{H} = -\frac{1}{M} \sum_{m=1}^{M} \sum_{c=1}^{C} p_{m,c} \log(p_{m,c} + 10^{-12})$$
* **Top-K Ranking**: Surfaces close alternative modulation hypotheses with respective confidence percentages.

---

## 10. Synthetic RF Intelligence Engine & Research Feedback Loop

To overcome data scarcity in electronic intelligence, SIGMA implements a **closed-loop synthetic adaptation framework**:

```text
                     ┌───────────────────────────────┐
                     │     REAL RF ACQUISITION       │
                     │      (RadioML 2016.10A)       │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │    STATISTICAL GAP ANALYSIS   │
                     │  (Power, Kurtosis, Autocorr)  │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │   GENERATOR CALIBRATION &     │
                     │     CHANNEL REFINEMENT        │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │    SYNTHETIC RF GENERATOR     │
                     │  (BPSK, QPSK, 8PSK, QAM, etc.)│
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │    FROZEN / MIXED TRAINING    │
                     │     EVALUATION EXPERIMENTS    │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │   MEASURABLE FEEDBACK LOOP    │
                     │   (Continuous Refinement)     │
                     └───────────────────────────────┘
```

### Research Findings (Milestones M6.1–M6.5)
1. **The Synthetic-to-Real Domain Gap is Measurable**:
   - Evaluating a frozen real-trained CNN on uncalibrated synthetic data caused Macro F1 to drop from **0.5844 down to 0.1244**.
   - Analysis revealed a **140x amplitude/power mismatch** and heavy-tailed phase kurtosis discrepancies ($5.77$ in real signals vs $-0.18$ in standard Gaussian models).
2. **Data-Driven Amplitude Calibration Restores Transferability**:
   - Dynamically matching synthetic RMS power to real training data boosted transfer F1 from **0.1244 to 0.1989** (+0.07 F1) and eliminated severe majority-class prediction collapse.
3. **Synthetic Augmentation vs Representation Pretraining**:
   - Directly mixing synthetic data with real training data degraded real test performance (10% synthetic reduced 5-class Macro F1 from **0.6403 to 0.5603**), showing that models readily memorize simulation-specific artifacts.
   - However, **pretraining on synthetic data followed by real-data fine-tuning achieved 0.6372 F1** (approaching the 0.6403 real-only baseline), proving that synthetic signals provide a viable initialization prior for feature representation.

---

## 11. Current Implementation Status

| Component / Capability | Implementation Location | Current Status |
|---|---|---|
| **M1: Dataset Foundation** | [`ml/dataset/`](ml/dataset/) | **COMPLETE** |
| **M2: Synthetic Generator** | [`ml/generators/`](ml/generators/) | **COMPLETE** |
| **M3: DSP Feature Extraction** | [`ml/features/`](ml/features/) | **COMPLETE** |
| **M4: Classical ML Baseline** | [`ml/baselines/`](ml/baselines/) | **COMPLETE** |
| **M5: Raw IQ 1D CNN Baseline** | [`ml/cnn_model/`](ml/cnn_model/) | **COMPLETE** |
| **M6: Synthetic & Domain Gap Research** | [`ml/synthetic/`](ml/synthetic/), [`ml/synthetic_training/`](ml/synthetic_training/) | **COMPLETE** |
| **M7.1: Universal Signal Ingestion** | [`ml/input/`](ml/input/) | **COMPLETE** |
| **M7.2: End-to-End File → CNN Inference** | [`ml/inference/`](ml/inference/) | **COMPLETE** |
| **FastAPI Backend Services** | `backend/` | **INTEGRATION PENDING** |
| **React Visualization Dashboard** | `frontend/` | **INTEGRATION PENDING** |
| **Multi-Model Decision Fusion** | — | **FUTURE ROADMAP** |
| **Parametric Impairment Estimator** | — | **FUTURE ROADMAP** |
| **Out-of-Distribution / Anomaly Detector** | — | **FUTURE ROADMAP** |
| **Live SDR Streaming Ingestion** | — | **FUTURE ROADMAP** |

---

## 12. Current ML Results & Historical Benchmarks

### 11-Class Real RadioML 2016.10A Benchmark
| Model | Input Representation | Test Accuracy | Test Macro F1 | Status |
|---|---|---|---|---|
| **Random Forest** | 36 Handcrafted DSP Features | 0.5218 | 0.5312 | Benchmark |
| **HistGradientBoosting** | 36 Handcrafted DSP Features | 0.5494 | 0.5656 | M4 Champion |
| **Raw IQ 1D CNN** | Raw Complex Waveforms `[2, 128]` | **0.5668** | **0.5844** | **M5 Primary Champion** |

### 5-Class Synthetic Transfer & Training Sweeps (M6.5)
| Configuration | Training Split | Test Split | Test Macro F1 | Test Accuracy |
|---|---|---|---|---|
| **Real Only** | 70,000 Real | 15,000 Real | **0.6403** | 0.6385 |
| **Synthetic Only (Refined)** | 18,000 Synthetic | 15,000 Real | **0.2942** | 0.2863 |
| **Mixed 10% Synthetic** | 63,000 Real + 7,000 Syn | 15,000 Real | **0.5603** | 0.5677 |
| **Mixed 25% Synthetic** | 52,500 Real + 17,500 Syn | 15,000 Real | **0.5602** | 0.5692 |
| **Mixed 50% Synthetic** | 17,500 Real + 17,500 Syn | 15,000 Real | **0.3218** | 0.3435 |
| **Pretrain $\rightarrow$ Finetune** | 18k Syn Pretrain $\rightarrow$ 70k Real | 15,000 Real | **0.6372** | 0.6274 |
| **Quality Check: 10% Original** | 63k Real + 7k Orig Syn | 15,000 Real | **0.4337** | 0.4755 |
| **Quality Check: 10% Calibrated**| 63k Real + 7k Cal Syn | 15,000 Real | **0.5704** | 0.5796 |

---

## 13. External Validation Dataset Context

To evaluate model generalization beyond the RadioML distribution, an external dataset structure is documented for independent testing:
* **Filename**: `subset_test.h5`
* **Structure**: `X` of shape `(80000, 1024, 2)` (float16), `y_mod`, `y_snr`, `y_chan`.
* **Modulation Mapping**: `0`: BPSK, `1`: QPSK, `2`: QAM, `3`: GMSK, `4`: OFDM, `5`: NBFM, `6`: WBFM.
* **Channel Profiles**: `0`: Clean AWGN, `1`: Multipath selective fading.
* **Role**: Serves as an independent out-of-domain benchmark. *(Formal evaluation pending integration).*

---

## 14. Testing & Verification

The repository enforces strict test-driven development, continuous integration, and data integrity verification:
* **Full Test Suite Count**: **`101 / 101 unit tests passing`**
* **Verification Coverage**:
  - `test_dataset.py`: Dataset caching, deterministic partitioning, label mappings.
  - `test_generator.py`, `test_modulators.py`, `test_pulse_shaping.py`: Constellation geometry, RRC filtering, symbol rates.
  - `test_impairments.py`, `test_channel.py`, `test_refined_generator.py`: Frequency offsets, phase jitter, IQ imbalances, FIR taps.
  - `test_features.py`: Vectorized math verification for all 36 DSP features.
  - `test_baselines.py`, `test_cnn.py`: Classical model training and PyTorch 1D CNN architectures.
  - `test_cross_domain.py`, `test_domain_analysis.py`, `test_calibration.py`, `test_synthetic_training.py`: Leakage audits, RMS calibration, transfer metrics, state frozen assertions.
  - `test_input_pipeline.py`: Format detection, stereo WAV parsing, mono rejection, NPY/NPZ layouts, binary configs, window segmentation.
  - `test_end_to_end_inference.py`: End-to-end file predictions, sum-to-one probabilities, checkpoint RMS scaling, model parameter immutability.

---

## 15. Installation & Setup

### Prerequisites
* **Operating System**: Windows / Linux / macOS
* **Python**: `v3.11`
* **PyTorch**: `v2.13.0` (CPU build)

### Step-by-Step Installation
From the root repository directory:
```powershell
# 1. Navigate to backend directory
cd backend

# 2. Create Python virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 16. Running SIGMA

### 1. Run Complete Test Suite
```powershell
set PYTHONPATH=%CD%
.\backend\venv\Scripts\pytest
```

### 2. Run End-to-End Prediction CLI
Analyze any supported signal file directly from the command line:
```powershell
set PYTHONPATH=%CD%
.\backend\venv\Scripts\python.exe -m ml.inference.pipeline_test "path\to\signal.wav"
```

*Example CLI Output:*
```text
SIGMA Signal Analysis
=====================

Input
-----
File: sample_recording.wav
Format: WAV
Sample rate: 16000.0 Hz
Original samples: 4096
Canonical IQ shape: (2, 4096)
Segments: 32

Inference
---------
Model: M5 Raw IQ CNN
Normalization: checkpoint training RMS
Windows analyzed: 32

Final Prediction
----------------
Modulation: QPSK
Confidence: 94.8%

Top Predictions
---------------
QPSK    94.8%
8PSK     3.1%
BPSK     1.2%
QAM16    0.6%
QAM64    0.3%

Window Distribution
-------------------
QPSK: 31
8PSK: 1
```

### 3. Python API Usage
```python
from ml.inference import analyze_file
from ml.input.types import PipelineConfig

# Configure input options (segment length defaults to 128)
config = PipelineConfig(segment_length=128)

# Execute end-to-end analysis
result = analyze_file("path/to/signal.wav", config=config)

# Access predictions and metadata
print(f"File: {result.filename} ({result.detected_format})")
print(f"Modulation: {result.predicted_class_name} ({result.confidence * 100:.1f}%)")
print(f"Mean Prediction Entropy: {result.mean_prediction_entropy:.4f}")
print(f"Total Segments Analyzed: {result.num_windows}")
```

---

## 17. Project Directory Structure

```text
SIGMA/
├── README.md                           # Master project documentation
├── .gitignore                          # Global file and artifact exclusions
│
├── backend/                            # FastAPI backend application
│   ├── requirements.txt                # Python package dependencies
│   ├── main.py                         # Application entry point
│   ├── api/                            # API routers (Upload, Analysis, Results)
│   ├── core/                           # Shared configuration and logging
│   └── dsp/                            # Core DSP utilities
│
├── frontend/                           # React + TypeScript UI Dashboard
│   ├── package.json                    # Frontend package dependencies
│   └── src/                            # Dashboard views, components, services
│
├── datasets/                           # Signal storage directory
│   ├── raw/                            # RadioML 2016.10A pickle archives
│   ├── processed/                      # Extracted DSP feature matrices (.npz)
│   └── synthetic/                      # Synthetic evaluation datasets
│
├── models/                             # Checkpoints directory
│   └── m5_iq_cnn.pt                    # Primary frozen M5 1D CNN checkpoint
│
├── results/                            # Empirical evaluation summaries & plots
│   └── ml/m6/                          # M6 research plots, CSVs, confusion matrices
│
├── tests/                              # Workspace test suite (101 tests)
│   └── ml/                             # Unit tests for M1–M7.2 components
│
└── ml/                                 # Machine Learning & DSP Core Package
    ├── README.md                       # Detailed ML milestone documentation
    ├── dataset/                        # M1: RadioML loader & deterministic splits
    ├── generators/                     # M2: Parameterized synthetic signal engine
    ├── features/                       # M3: 36 deterministic DSP feature extractors
    ├── baselines/                      # M4: Classical ML baselines (HGB, RF)
    ├── cnn_model/                      # M5: PyTorch 1D Raw IQ CNN baseline
    ├── synthetic/                      # M6.1-M6.4: Cross-domain gap analysis
    ├── synthetic_training/             # M6.5: Synthetic-assisted training sweeps
    ├── input/                          # M7.1: Universal format ingestion adapter
    └── inference/                      # M7.2: End-to-end prediction & aggregation
```

---

## 18. Architectural Principles

1. **Modular Layer Decoupling**: File parsing (`ml.input`), model preprocessing, model inference (`ml.cnn_model`), and aggregation (`ml.inference`) exist as separate modules with clean interfaces.
2. **Dynamic Checkpoint-Driven Preprocessing**: Normalization statistics are extracted from trained checkpoints at runtime. No empirical scaling constants are hardcoded into inference.
3. **Strict Data Isolation**: Normalization factors are computed strictly on training splits. Validation and test sets are never inspected during preprocessing setup.
4. **Frozen Inference Integrity**: Production inference executes under strict evaluation modes (`torch.no_grad()`), guaranteeing model parameter immutability.
5. **Agnostic Ingestion via Adapters**: New signal formats (e.g. streaming SDR buffers) are added via input adapters rather than modifying core ML architectures.

---

## 19. Current Limitations

* **Fixed Model Classification Space**: The baseline CNN operates over the 11 classes present in RadioML 2016.10A. Signals belonging to unmodeled classes are mapped onto this 11-class logit space.
* **Explicit Binary Configuration**: Raw `.bin`/`.dat` files require explicit `BinaryIQConfig` parameters (dtype, interleaving, endianness) because unformatted binary data cannot be inferred without ambiguity.
* **Strict Stereo WAV Requirement**: Mono audio files are rejected to prevent arbitrary single-channel signal interpretations.
* **Pending UI/API Integration**: The ML core is fully functional via Python API and CLI; FastAPI routing and React frontend dashboard connections are currently pending.
* **Synthetic Domain Gap Persistence**: While data-driven amplitude calibration and Wiener phase noise significantly improve synthetic signal realism, synthetic-only training still exhibits domain mismatch relative to real RF signals.

---

## 20. Future Roadmap

1. **Phase Recovery & Advanced Channel Modeling**:
   - Integrate multi-path Rayleigh/Rician selective fading simulators into the generator.
   - Implement timing synchronization jitter and non-Gaussian phase slip models.
2. **Multi-Model Decision Fusion Engine**:
   - Combine raw IQ deep neural network probabilities with classical DSP feature tree decisions.
3. **Automated Impairment & SNR Estimators**:
   - Train dedicated regression heads to estimate carrier frequency offset (Hz), IQ imbalance ($\alpha, \phi$), and SNR (dB).
4. **Out-of-Distribution (OOD) & Anomaly Detection**:
   - Implement energy-based uncertainty thresholds to flag unclassifiable or corrupted signals as *"Unknown / Low Confidence"*.
5. **Full API & UI Dashboard Deployment**:
   - Wire `ml.inference` into FastAPI asynchronous endpoints with WebSocket live spectrogram feeds.
   - Connect the React dashboard for interactive IQ constellation plotting, spectrogram visualization, and detailed signal intelligence reporting.
