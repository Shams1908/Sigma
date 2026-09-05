# SIGMA - Codebase Status Reference

> **Generated:** 2026-09-01
> **Purpose:** Single-file context document describing every implemented and placeholder component across the entire SIGMA repository.

---

## Table of Contents
1. Project Overview
2. High-Level Implementation Status
3. ML Layer - Detailed Module Breakdown
4. Backend Layer - Detailed Module Breakdown
5. Frontend Layer - Detailed Module Breakdown
6. Tests Layer
7. Models and Artifacts
8. Datasets and Results
9. Documentation
10. Key Numbers and Performance Benchmarks
11. Known Gaps and Future Roadmap

---

## 1. Project Overview

**SIGMA** (Signal Intelligence & Guided Modulation Analysis) is an end-to-end RF signal analysis platform. It ingests arbitrary unknown RF waveforms (WAV, IQ, NumPy arrays, raw binary), classifies modulation schemes using ML, quantifies channel impairments, and presents structured intelligence via a web dashboard.

**Tech stack:**
- **ML/Backend:** Python 3.11, PyTorch 2.x (CPU), scikit-learn, NumPy, SciPy, Pandas, FastAPI, Uvicorn
- **Frontend:** React 19 + TypeScript, Vite 8, Vanilla CSS

**Entry points:**
- ML CLI: `python -m ml.inference.pipeline_test <signal_file>`
- Backend API: `python backend/main.py` -> FastAPI on port 8000
- Frontend: `npm run dev` in `frontend/`

---

## 2. High-Level Implementation Status

| Milestone | Description | Location | Status |
|---|---|---|---|
| M1 | Dataset Foundation (RadioML 2016.10A) | `ml/dataset/` | COMPLETE |
| M2.1 | Synthetic Generator Foundation | `ml/generators/` | COMPLETE |
| M2.2 | Digital Modulation Mappers (BPSK/QPSK/8PSK/QAM16/QAM64) | `ml/generators/modulation/` | COMPLETE |
| M2.3 | RRC Pulse Shaping | `ml/generators/pulse_shaping/` | COMPLETE |
| M2.4 | AWGN Channel Model | `ml/generators/channel/` | COMPLETE |
| M2.5 | RF Impairments (CFO, Phase, IQ, DC, Timing) | `ml/generators/impairments/` | COMPLETE |
| M3 | DSP Feature Extraction (36 features) | `ml/features/` | COMPLETE |
| M4 | Classical ML Baseline (HGB, RF) | `ml/baselines/` | COMPLETE |
| M5 | Raw IQ 1D CNN Baseline (PyTorch) | `ml/cnn_model/` | COMPLETE |
| M6.1 | Synthetic Evaluation Dataset Generation | `ml/synthetic/` | COMPLETE |
| M6.2 | Cross-Domain Evaluation (Frozen CNN on Synthetic) | `ml/synthetic/evaluate.py` | COMPLETE |
| M6.3 | Real-vs-Synthetic Domain Gap Analysis | `ml/synthetic/domain_analysis.py` | COMPLETE |
| M6.4.1 | Data-Driven Amplitude Calibration | `ml/synthetic/calibrate.py` | COMPLETE |
| M6.4.2 | Wiener Phase Noise + FIR Channel Refinement | `ml/synthetic/refine_generator.py` | COMPLETE |
| M6.5 | Synthetic-Assisted Training Sweeps | `ml/synthetic_training/` | COMPLETE |
| M7.1 | Universal Signal Input / Format Adapter | `ml/input/` | COMPLETE |
| M7.2 | End-to-End File to CNN Prediction Pipeline | `ml/inference/` | COMPLETE |
| Backend API | FastAPI endpoints (Upload, Analysis, Results) | `backend/` | SKELETON - integration pending |
| Frontend Dashboard | React UI with visualizations | `frontend/` | SKELETON - health check only |
| Multi-Model Fusion | Combining CNN + DSP classifier outputs | - | FUTURE |
| Impairment Estimators | Regression heads for CFO, IQ, SNR | - | FUTURE |
| OOD Detector | Out-of-distribution anomaly flagging | - | FUTURE |
| Live SDR Streaming | Real-time software-defined radio ingestion | - | FUTURE |

---

## 3. ML Layer - Detailed Module Breakdown

### ml/dataset/ - M1: Dataset Foundation

**Purpose:** Load, cache, and deterministically split the RadioML 2016.10A benchmark dataset.

| File | Role |
|---|---|
| `loader.py` | Loads `datasets/raw/RML2016.10a_dict.pkl`; implements singleton `DatasetCache` class |
| `labels.py` | `MODULATION_LABELS` list (11 classes, alphabetically sorted); `MODULATION_TO_INDEX` / `INDEX_TO_MODULATION` dicts |
| `schema.py` | Typed dataclass definitions for dataset entries |
| `metadata.py` | Dataset-level descriptors (220k total, 11 classes, 20 SNR levels from -20 to +18 dB) |
| `splitting.py` | Stratified 70/15/15 split using flat index arrays; fixed seed=42; 154k train / 33k val / 33k test |
| `validation.py` | Shape, dtype, NaN/Inf, and integrity checks on raw samples |
| `inspect_dataset.py` | CLI diagnostic - prints split stats and verifies correctness |

**Key facts:**
- Dataset shape per example: `[2, 128]` (I=row 0, Q=row 1), float32
- 220,000 total examples; 1,000 per `(modulation, SNR)` key
- 11 modulation classes: 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM
- Split uses index arrays pointing to shared cache - no array copying

---

### ml/generators/ - M2: Synthetic Signal Generator

**Purpose:** Parameterized, physics-based RF signal generator with deterministic seeding.

#### ml/generators/ (root)
| File | Role |
|---|---|
| `config.py` | `GeneratorConfig` Pydantic model: modulation, num_symbols, sample_rate, symbol_rate, samples_per_symbol, random_seed, optional SNR/offset fields |
| `signal.py` | `GeneratedSignal` container (IQ array `[2,N]` + `SyntheticGroundTruth`); `complex_to_iq` / `iq_to_complex` lossless converters |
| `bits.py` | `generate_bits(num_bits, rng)` - deterministic, no global state mutation |
| `validation.py` | Parameter consistency checks (e.g., sample_rate / symbol_rate == samples_per_symbol) |

#### ml/generators/modulation/
| File | Role |
|---|---|
| `interface.py` | Abstract modulator interface + full pipeline entry point |
| `bpsk.py` | BPSK: 1 bit/symbol, mapping {0 -> -1, 1 -> +1} |
| `qpsk.py` | QPSK: 2 bits/symbol, Gray-coded, normalized by 1/sqrt(2) |
| `psk8.py` | 8PSK: 3 bits/symbol, Gray-coded on unit circle |
| `qam16.py` | 16-QAM: 4 bits/symbol, square Gray-coded, normalized by 1/sqrt(10) |
| `qam64.py` | 64-QAM: 6 bits/symbol, square Gray-coded, normalized by 1/sqrt(42) |

All modulators output canonical `[2, N]` float32 IQ and wrap in `GeneratedSignal`.

#### ml/generators/pulse_shaping/
| File | Role |
|---|---|
| `rrc.py` | Root Raised Cosine (RRC) filter: apply_rrc_pulse_shaping(). Zero-insert upsampling -> full convolution -> linear FIR delay included. Output length: (num_symbols + filter_span) * sps |

#### ml/generators/channel/
| File | Role |
|---|---|
| `awgn.py` | AWGN: dynamic signal power estimation -> noise power from SNR -> complex Gaussian noise. Zero-power guard. complex64 output |
| `interface.py` | `add_awgn(signal, config, snr_db)` - applies AWGN at specified SNR |

#### ml/generators/impairments/
| File | Role |
|---|---|
| `interface.py` | `apply_impairments(signal, config, ...)` - composable, applies 5 impairments in strict order |
| `frequency.py` | Carrier Frequency Offset: y[n] = x[n] * exp(j2*pi*df*n/Fs) |
| `phase.py` | Carrier Phase Offset: constant complex rotation |
| `iq_imbalance.py` | IQ Amplitude + Phase Imbalance: 2x2 real matrix on I/Q components |
| `dc_offset.py` | Complex DC bias: y[n] = x[n] + (Idc + jQdc) |
| `timing.py` | Fractional Timing Offset: linear interpolation with zero-padding |

**Impairment pipeline order:** Frequency -> Phase -> IQ Imbalance -> DC Offset -> Timing Offset

---

### ml/features/ - M3: DSP Feature Extraction

**Purpose:** Extract a deterministic, fixed-length 36-feature vector from a single `[2, 128]` IQ sample.

| File | Role |
|---|---|
| `extractor.py` | `extract_features(samples)`, `extract_batch_features(batch)`, `extract_feature_matrix(dataset, indices)` |
| `amplitude.py` | 4 features: mean, variance, kurtosis, PAR of signal envelope |
| `phase.py` | 12 features: phase variance, phase-diff mean/variance/kurtosis, 8-bin phase histogram |
| `frequency.py` | 2 features: instantaneous frequency mean and variance |
| `cumulants.py` | 3 features: normalized 4th-order cumulant C40 (real, imag, magnitude) - complex128 for stability |
| `correlation.py` | 15 features: normalized complex autocorrelation at lags {1,2,4,8,16} (real + imag + magnitude each) |
| `schema.py` | Feature name ordering - stable canonical 36-element list |
| `inspect_features.py` | Full pipeline CLI: extracts all 220k samples, validates for NaN/Inf, performs correlation analysis, saves `datasets/processed/RML2016.10a_features.npz` |

**Policy:** Zero-leakage - no learned parameters. Epsilon-guarded to produce 0 NaN and 0 Inf on any input.

**36-feature schema:**
- Amplitude (4): amplitude_mean, amplitude_variance, amplitude_kurtosis, amplitude_peak_to_average_ratio
- Phase (12): phase_variance, phase_difference_mean, phase_difference_variance, phase_difference_kurtosis, phase_hist_bin_0 to phase_hist_bin_7
- Instantaneous Frequency (2): instantaneous_frequency_mean, instantaneous_frequency_variance
- Cumulant (3): c40_real, c40_imag, c40_magnitude
- Autocorrelation (15): autocorr_lag_{1,2,4,8,16}_{real,imag,magnitude}

---

### ml/baselines/ - M4: Classical ML Baseline

**Purpose:** Train and evaluate classical ML classifiers on 36 DSP features.

| File | Role |
|---|---|
| `train.py` | Grid search over Random Forest and HistGradientBoosting hyperparameters; selects champion by validation macro F1; saves to `models/` |
| `inference.py` | `predict(features)` - loads `models/baseline_hgb.joblib`, returns class name + index + confidence + probabilities |

**Champion model:** HistGradientBoosting(max_iter=100, max_depth=10, lr=0.1) - saved at `models/baseline_hgb.joblib`.

**Performance (11-class test set):**
- Accuracy: 0.5494 | Macro F1: 0.5656 | Macro Precision: 0.6410

**Top 5 most important features (by Random Forest):**
1. amplitude_variance
2. autocorr_lag_1_real
3. amplitude_mean
4. autocorr_lag_1_magnitude
5. amplitude_peak_to_average_ratio

---

### ml/cnn_model/ - M5: Raw IQ 1D CNN

**Purpose:** Train a deep 1D CNN directly on raw IQ waveforms `[2, 128]`.

| File | Role |
|---|---|
| `architecture.py` | `RawIQCNN(nn.Module)` - 3 Conv1D blocks with BatchNorm + ReLU + MaxPool, Global Average Pooling, FC head with Dropout(0.3). Input `[B,2,128]` -> output `[B,11]` |
| `dataset.py` | `RadioMLDataset(torch.utils.data.Dataset)` - wraps index arrays into PyTorch-compatible form |
| `train.py` | Full training loop: Adam optimizer, CrossEntropy loss, early stopping (patience=5), training RMS normalization saved to checkpoint, model saved at `models/m5_iq_cnn.pt` |
| `inference.py` | `get_cnn_model(path)` - loads frozen model + extracts `rms_factor` from checkpoint dict; module-level LRU cache for repeated calls |

**Architecture:**
```
[B, 2, 128]
  -> Conv1d(2->64, k=7) + BatchNorm1d + ReLU + MaxPool(2) -> [B, 64, 64]
  -> Conv1d(64->128, k=5) + BatchNorm1d + ReLU + MaxPool(2) -> [B, 128, 32]
  -> Conv1d(128->256, k=3) + BatchNorm1d + ReLU -> [B, 256, 32]
  -> GlobalAvgPool (mean over time dim) -> [B, 256]
  -> Linear(256->128) + ReLU + Dropout(0.3) + Linear(128->11)
  -> [B, 11] logits
```

**Performance (11-class test set):**
- Accuracy: 0.5668 | Macro F1: 0.5844 (+0.0188 vs M4)
- Best val F1: 0.5900 at Epoch 8; early stop at Epoch 13
- Training RMS factor: `0.006048427` (stored in checkpoint, never hardcoded)

---

### ml/synthetic/ - M6: Synthetic Research Track

**Purpose:** Generate synthetic evaluation datasets, diagnose the domain gap, and calibrate the generator.

| File | Role |
|---|---|
| `config.py` | Synthetic dataset generation configuration: SNR ranges, impairment sweep parameters, supported modulations |
| `generator.py` | `SyntheticDatasetGenerator` - generates 18,000 samples: 5 modulations x 36 sweep conditions x 100 examples |
| `dataset.py` | Loads generated .npz datasets back for evaluation |
| `evaluate.py` | M6.2: Evaluates frozen M5 CNN on synthetic data; confusion matrices, per-class metrics, impairment robustness reports |
| `domain_analysis.py` | M6.3: Matched real vs. synthetic feature comparison; Cohen's d for all 36 features; PCA separability; Decision Tree domain classifier; spectral comparisons |
| `calibrate.py` | M6.4.1: Data-driven RMS calibration - dynamically matches synthetic power to real training split; no hardcoded constants |
| `refine_generator.py` | M6.4.2: Adds Wiener phase noise (LO drift random walk) and FIR channel response to generator for realism |
| `inspect_synthetic.py` | CLI diagnostic: generates sample waveforms, plots, verifies seed reproducibility |

**Supported modulations in synthetic:** BPSK, QPSK, 8PSK, QAM16, QAM64 (5 of 11 RadioML classes)
**Unsupported:** AM-DSB, AM-SSB, CPFSK, GFSK, PAM4, WBFM

**M6.1 Dataset parameters:**
- num_symbols=8, filter_span=8, sps=8 -> output length = (8+8)*8 = 128 samples
- 7 isolated impairment sweeps (AWGN, CFO, Phase, IQ Amplitude, IQ Phase, DC, Timing)
- Base seed = 42000 for reproducibility

**Key findings:**
- Frozen CNN macro F1 on synthetic (uncalibrated): 0.1244 vs 0.5844 on real data
- Root cause: 140x amplitude mismatch (synthetic RMS = 0.852, real RMS = 0.006)
- After amplitude calibration: F1 improved to 0.1989
- Phase kurtosis mismatch: real=5.77 vs synthetic=-0.18 (Cohen's d = -0.97)
- Domain classifier separates real vs synthetic with 100% accuracy using only amplitude_mean

---

### ml/synthetic_training/ - M6.5: Synthetic-Assisted Training

**Purpose:** Controlled experiments studying whether synthetic data can assist 5-class training.

| File | Role |
|---|---|
| `experiments.py` | Defines 7 training configurations: real-only, synthetic-only, mixed 10/25/50%, pretrain->finetune, quality checks |
| `train.py` | Training loop for 5-class experiments; evaluates on real test split |
| `evaluate.py` | Metrics computation for 5-class evaluation |
| `dataset.py` | Dataset class for mixed real+synthetic experiments |
| `sampler.py` | Balanced sampler for mixed-ratio experiments |

**Key results (5-class: BPSK/QPSK/8PSK/QAM16/QAM64):**

| Config | Test Macro F1 |
|---|---|
| Real Only | 0.6403 |
| Pretrain (Syn) -> Finetune (Real) | 0.6372 |
| Mixed 10% Calibrated Synthetic | 0.5704 |
| Mixed 10% Synthetic | 0.5603 |
| Mixed 25% Synthetic | 0.5602 |
| Mixed 50% Synthetic | 0.3218 |
| Synthetic Only | 0.2942 |
| Real + 10% Original Synthetic | 0.4337 |

**Conclusion:** Mixing synthetic data hurts performance (domain mismatch). Pretrain->finetune is viable (0.6372 vs 0.6403 real-only baseline).

---

### ml/input/ - M7.1: Universal Signal Ingestion

**Purpose:** Format-agnostic file adapter - parses any supported signal file to canonical `[2, N]` float32 IQ.

| File | Role |
|---|---|
| `detector.py` | `detect_format(path)` - inspects binary magic numbers (RIFF/WAVE, \x93NUMPY, ZIP/NPZ); returns `SignalFormat` enum |
| `wav.py` | `parse_wav(path, config)` - stereo WAV only (rejects mono); extracts sample rate; normalizes int16->float32 |
| `iq.py` | `parse_iq(path, config)` - handles .npy, .npz, and raw .bin/.dat with BinaryIQConfig; normalizes layouts [2,N], [N,2], complex [N], framed [N,2,128] |
| `segment.py` | `segment_iq(iq, segment_length, pad_short)` - non-overlapping windows [M, 2, segment_length]; discards trailing residuals |
| `pipeline.py` | `process_file(path, config)` -> `(segments[M,2,128], SignalMetadata)` - orchestrates detect -> parse -> validate -> segment |
| `types.py` | `PipelineConfig`, `BinaryIQConfig`, `SignalMetadata`, `SignalFormat` dataclasses |
| `pipeline_test.py` | CLI diagnostic: `python -m ml.input.pipeline_test <file>` |

**Supported formats:** Stereo WAV, .npy, .npz, .bin, .dat
**Rejected:** Mono WAV, corrupt/empty files, files with NaN/Inf, unsupported extensions
**Max file size:** 50 MB (configurable via `PipelineConfig.max_file_size_bytes`)
**Segment length:** 128 (configurable via `PipelineConfig.segment_length`)

---

### ml/inference/ - M7.2: End-to-End Prediction Pipeline

**Purpose:** Ties M7.1 input + M5 CNN into a single unified file-to-prediction API.

| File | Role |
|---|---|
| `pipeline.py` | `analyze_file(path, config, model_path, top_k)` -> `SignalAnalysisResult` - full orchestration |
| `aggregation.py` | Element-wise mean probability aggregation across windows; Shannon entropy computation; window distribution counts |
| `pipeline_test.py` | CLI entry: `python -m ml.inference.pipeline_test <file>` - prints formatted analysis report |

**`SignalAnalysisResult` dataclass fields:**
- Input metadata: `filename`, `detected_format`, `sample_rate`, `original_sample_count`, `canonical_iq_shape`
- Predictions: `predicted_class_index`, `predicted_class_name`, `confidence`, `probability_vector`, `top_predictions` (top-5 list of (name, prob) tuples)
- Window results: `num_windows`, `window_predictions`, `window_confidences`, `window_probabilities`, `window_distribution`, `mean_prediction_entropy`
- Processing info: `model_checkpoint_path`, `normalization_source`, `segment_length`, `processing_status`

**Inference flow:**
```
1. process_file() -> segments [M, 2, 128] + input metadata
2. get_cnn_model() -> frozen RawIQCNN + rms_factor from checkpoint
3. segments / rms_factor  (normalization - no hardcoding)
4. torch.no_grad() batch inference -> softmax probs [M, 11]
5. aggregate_window_probabilities() -> file-level prediction + entropy
6. Return SignalAnalysisResult
```

**Aggregation:**
- p_avg = element-wise mean across all M windows
- predicted class = argmax(p_avg)
- mean entropy = -(1/M) sum_m sum_c p_m,c * log(p_m,c + 1e-12)

---

## 4. Backend Layer - Detailed Module Breakdown

> **Status: Skeleton implementation.** All files exist with stubs only; no ML integration has been wired.

**Entry point:** `backend/main.py`
FastAPI app with CORS middleware (ports 3000 and 5173), 3 API routers, and a `/health` endpoint.

### backend/core/
| File | Role |
|---|---|
| `config.py` | `Settings` (pydantic-settings): PROJECT_NAME="SIGMA", API_V1_STR="/api/v1", CORS origins configured |

### backend/api/
| File | Role |
|---|---|
| `upload.py` | Router stub registered at `/api/v1/upload` - no logic |
| `analysis.py` | Router stub registered at `/api/v1/analysis` - no logic |
| `results.py` | Router stub registered at `/api/v1/results` - no logic |

### backend/dsp/ (stub files)
Placeholder files with no implementations:
- `fft.py`, `psd.py`, `spectrogram.py`, `snr.py`, `bandwidth.py`, `carrier.py`, `symbol_rate.py`

### backend/modulation/ (stub files)
- `classifier.py`, `features.py`, `fsk.py`, `psk.py`, `qam.py`

### backend/demodulation/ (stub files)
- `fsk_demod.py`, `psk_demod.py`, `qam_demod.py`

### backend/synchronization/ (stub files)
- `carrier_recovery.py`, `matched_filter.py`, `timing_recovery.py`

### backend/hypothesis/ (stub files)
- `generator.py`, `candidates.py`, `evaluator.py`, `ranking.py`

### backend/ingestion/ (stub files, superseded by ml/input/)
- `iq_reader.py`, `wav_reader.py`, `metadata.py`

### backend/preprocessing/ (stub files)
- `filtering.py`, `normalize.py`, `segmentation.py`

### backend/reporting/ (stub files)
- `results.py`

**Python dependencies (requirements.txt):**
fastapi, uvicorn, pydantic, pydantic-settings, numpy, scipy, pandas, scikit-learn, pytest, torch, matplotlib

---

## 5. Frontend Layer - Detailed Module Breakdown

> **Status: Phase 0 skeleton.** Only the backend health-check display is functional.

**Stack:** React 19 + TypeScript, Vite 8, Vanilla CSS, no external UI library.

### frontend/src/ (root files)
| File | Role |
|---|---|
| `main.tsx` | React 19 DOM root, mounts App component |
| `App.tsx` | Root component - renders AnalysisDashboard (9 lines) |
| `index.css` | 11,715-byte global stylesheet: dark theme, glassmorphism cards, glow effects, animations |
| `App.css` | Minimal 51-byte override file |

### frontend/src/pages/AnalysisDashboard/index.tsx (186 lines)
The only implemented page. Contains:
- **Backend health polling:** `checkBackendHealth()` called on mount and every 10 seconds
- **Connection badge:** Shows ONLINE/OFFLINE/CONNECTING with animated pulse dot
- **System info card:** Project name, "Phase 0 - Setup", 12% progress bar
- **Placeholder panels (not functional):**
  - File Ingestion (drag/drop UI shell, no upload logic)
  - Estimated Parameters (SNR, Symbol Rate, Carrier Offset, Bandwidth - all showing "--")
  - Spectrum (PSD) visualizer placeholder (static SVG mock wave)
  - Waterfall Spectrogram placeholder
  - Hypothesis Rankings table placeholder
  - IQ Constellation Diagram placeholder

### frontend/src/components/ (scaffold directories only)
Seven component directories exist but each contains only a README stub (no implementation):
- `ConstellationViewer/` - for IQ constellation scatter plot
- `FileUpload/` - for drag-and-drop file upload
- `HypothesisRanking/` - for modulation hypothesis table
- `ParameterPanel/` - for DSP parameter display
- `SignalOverview/` - for high-level signal summary
- `SpectrumViewer/` - for FFT/PSD chart
- `WaterfallViewer/` - for time-frequency waterfall

### frontend/src/services/api.ts
| Function | Role |
|---|---|
| `checkBackendHealth()` | Fetches `GET http://localhost:8000/health`, returns `{status, project}`. Only wired API call. |

### frontend/src/types/index.ts
Three TypeScript interfaces defined but not yet used in any component:
- `SignalMetadata` - fileName, sampleRate, centerFrequency, duration, fileSize, ingestionTime
- `EstimatedParameters` - snr, bandwidth, carrierOffset, symbolRate
- `HypothesisCandidate` - id, modulation, symbolRate, confidenceScore, details, status

---

## 6. Tests Layer

> **Status: 101/101 tests passing.** All tests are in `tests/ml/`.

| Test File | Milestone | Coverage |
|---|---|---|
| `test_dataset.py` | M1 | DatasetCache, split sizes, label mappings, determinism |
| `test_generator.py` | M2.1 | GeneratorConfig validation, bit generation, IQ conversion |
| `test_modulators.py` | M2.2 | BPSK/QPSK/8PSK/QAM16/QAM64 constellation geometry, Gray coding, power normalization |
| `test_pulse_shaping.py` | M2.3 | RRC coefficients, upsampling, output length, filter energy normalization |
| `test_channel.py` | M2.4 | AWGN SNR accuracy, complex noise power split, zero-power guard |
| `test_impairments.py` | M2.5 | All 5 impairments, identity conditions, pipeline order |
| `test_features.py` | M3 | All 36 features - numeric correctness, NaN/Inf-free on zero-power signals |
| `test_baselines.py` | M4 | HGB training, inference API, model loading |
| `test_cnn.py` | M5 | Architecture shape, forward pass, training loop, checkpoint saving |
| `test_synthetic.py` | M6.1 | Synthetic dataset generation, seed reproducibility |
| `test_cross_domain.py` | M6.2 | Cross-domain evaluation, leakage audits, CNN frozen during eval |
| `test_domain_analysis.py` | M6.3 | Feature statistics, Cohen's d calculation |
| `test_calibration.py` | M6.4.1 | RMS calibration, dynamic scaling without hardcoded constants |
| `test_refined_generator.py` | M6.4.2 | Wiener noise, FIR channel, refined generator outputs |
| `test_synthetic_training.py` | M6.5 | State frozen assertions, transfer metrics |
| `test_input_pipeline.py` | M7.1 | Format detection, stereo WAV parse, mono rejection, NPY/NPZ/BIN layouts, segmentation |
| `test_end_to_end_inference.py` | M7.2 | Full file->prediction flow, probabilities sum to 1, RMS scaling, model immutability |
| `test_final_integration.py` | Integration | Cross-milestone integration checks |
| `manual_test_m4.py` | M4 | Manual script for interactive M4 baseline testing |

**Empty test directories (no tests yet):**
- `tests/backend/` - only README stub
- `tests/frontend/` - only README stub
- `tests/integration/` - only README stub

---

## 7. Models and Artifacts

| File | Description |
|---|---|
| `models/baseline_hgb.joblib` | Trained HistGradientBoostingClassifier (M4 champion, ~3.6 MB serialized) |
| `models/baseline_hgb_metadata.json` | Full training metadata: hyperparameters, val/test metrics (Accuracy 0.5494, Macro F1 0.5656), all 36 feature names, label mapping, training timing |
| `models/m5_iq_cnn.pt` | Frozen PyTorch checkpoint for M5 RawIQCNN (includes rms_factor = 0.006048427 in state dict) |

`m5_iq_cnn.pt` is the primary production inference model used by `ml/inference/pipeline.py`.
The `rms_factor` is read dynamically from the checkpoint at inference time - never hardcoded.

---

## 8. Datasets and Results

### datasets/
| Path | Content |
|---|---|
| `datasets/raw/RML2016.10a_dict.pkl` | RadioML 2016.10A raw pickle (220k samples, keyed by (mod, snr) tuples) |
| `datasets/processed/RML2016.10a_features.npz` | Precomputed M3 36-feature matrix: X=[220k,36], y=[220k], snrs=[220k] |
| `datasets/synthetic/synthetic_evaluation_dataset.npz` | 18,000 synthetic evaluation samples: X=[18000,2,128], y=[18000] |
| `datasets/synthetic/synthetic_evaluation_metadata.json` | Per-sample ground truth configurations and sweep parameters |

### results/
Generated evaluation artifacts stored under `results/ml/m6/`:
- `impairment_robustness.csv` / `.json` - per-impairment F1 degradation curves (M6.2)
- `cross_domain_comparison.json` - M6.2 vs real baseline comparison
- `domain_analysis/domain_feature_comparison.csv` - M6.3 feature Cohen's d ranking
- `domain_analysis/domain_gap_by_class.csv` - per-class amplitude/kurtosis stats
- `domain_analysis/raw_iq_statistics.csv` - raw RMS values (real=0.006056, synthetic=0.852382)
- `domain_analysis/pca_domain_separability.png` - 2D PCA domain plot
- `domain_analysis/domain_gap_features.png` - feature distribution histograms
- `domain_analysis/spectral_comparison.png` - PSD overlay plots
- `domain_visuals/` - I/Q waveform, envelope, and constellation visual comparisons

---

## 9. Documentation

| File | Content |
|---|---|
| `README.md` (root) | Master documentation: full system vision, architecture diagrams, all milestone summaries, performance tables, setup/run instructions, directory structure, limitations, 5-point future roadmap |
| `ml/README.md` | Detailed ML milestone documentation with full mathematical derivations for M1-M7.2 (774 lines) |
| `docs/architecture.md` | High-level 6-stage processing pipeline overview |
| `docs/development-phases.md` | 8-phase development roadmap (Phase 0 Setup -> Phase 8 GUI Demo) |
| `backend/README.md` | Brief backend overview |
| `frontend/README.md` | Brief frontend overview |
| `notebooks/README.md` | Placeholder only - no notebooks exist |
| `datasets/README.md` | Brief dataset storage description |
| `models/README.md` | Brief models directory description |

---

## 10. Key Numbers and Performance Benchmarks

### Model Comparison - 11-Class RadioML 2016.10A Test Set

| Model | Test Accuracy | Test Macro F1 | Test Macro Precision |
|---|---|---|---|
| Random Forest (M4) | 0.5218 | 0.5312 | - |
| HistGradientBoosting (M4 Champion) | 0.5494 | 0.5656 | 0.6410 |
| 1D CNN (M5 Primary Model) | 0.5668 | 0.5844 | 0.6811 |

CNN outperforms classical baseline by +0.0188 Macro F1.

### 5-Class Synthetic Training Experiments - M6.5

| Config | Training Data | Test Macro F1 | Test Accuracy |
|---|---|---|---|
| Real Only | 70k Real | 0.6403 | 0.6385 |
| Pretrain Syn -> Finetune Real | 18k Syn + 70k Real | 0.6372 | 0.6274 |
| Mixed 10% Calibrated | 63k Real + 7k Cal Syn | 0.5704 | 0.5796 |
| Mixed 10% Syn | 63k Real + 7k Syn | 0.5603 | 0.5677 |
| Mixed 25% Syn | 52.5k Real + 17.5k Syn | 0.5602 | 0.5692 |
| Mixed 50% Syn | 17.5k Real + 17.5k Syn | 0.3218 | 0.3435 |
| Synthetic Only | 18k Syn | 0.2942 | 0.2863 |
| Real + 10% Original Syn | 63k Real + 7k Orig Syn | 0.4337 | 0.4755 |

### Domain Gap Metrics - M6.3

| Metric | Real (RadioML) | Synthetic (Generator) |
|---|---|---|
| Raw RMS | 0.006056 | 0.852382 |
| Normalized RMS (CNN input) | 1.001 | 140.9 |
| Phase Diff Kurtosis | 5.777 | -0.183 |
| Cohen's d (phase kurtosis) | - | -0.97 |

### Cross-Domain Evaluation - M6.2

| Condition | Macro F1 |
|---|---|
| Frozen M5 CNN on real test set | 0.5844 |
| Frozen M5 CNN on uncalibrated synthetic | 0.1244 |
| Frozen M5 CNN on amplitude-calibrated synthetic | 0.1989 |
| Frozen M5 CNN on refined synthetic (Wiener + FIR) | 0.1958 |

### CNN Training Details (M5)
- Optimizer: Adam, lr=0.001, batch_size=128
- Loss: CrossEntropyLoss
- Best epoch: 8 (val Macro F1 = 0.5900)
- Early stopping triggered: Epoch 13 (patience=5)
- Training RMS factor: 0.006048427

### Baseline Training Details (M4)
- HGB training time: 14.67 seconds
- HGB per-sample inference latency: 0.025 ms
- HGB validation Macro F1: 0.5643

---

## 11. Known Gaps and Future Roadmap

### Current Limitations

1. **Backend not connected to ML core** - FastAPI routes (upload, analysis, results) are empty stubs. `ml.inference.analyze_file` is not called by any endpoint. The ML core works perfectly via CLI and Python API but is not web-accessible.

2. **Frontend is a Phase 0 skeleton** - Only backend health polling is functional. All 6 visualization panels (Spectrum, Waterfall, Constellation, Parameters, Hypothesis Table, File Upload) are static HTML placeholders with no real data.

3. **No backend or frontend tests** - Only the 101 ML tests exist. `tests/backend/`, `tests/frontend/`, and `tests/integration/` directories contain only empty README stubs.

4. **No Jupyter notebooks** - The `notebooks/` directory exists but is completely empty.

5. **Synthetic generator covers only 5 of 11 modulation classes** - AM-DSB, AM-SSB, CPFSK, GFSK, PAM4, WBFM cannot be generated synthetically because analog/FM modulators are not implemented.

6. **Only stereo WAV supported** - Mono WAV files are explicitly rejected. Raw binary .bin/.dat files require explicit `BinaryIQConfig` (dtype, interleaved, endianness must be specified manually).

7. **Fixed 11-class output space** - The CNN cannot flag out-of-distribution (OOD) signals. Unknown modulations are mapped onto the 11-class logit space.

8. **Synthetic domain gap persists** - Even after amplitude calibration (+0.07 F1) and Wiener/FIR refinement, the synthetic generator still produces waveforms with measurable statistical differences from real RadioML data.

### Immediate Next Steps (Integration Priority)

- Wire `ml.inference.analyze_file` into FastAPI `/api/v1/analysis` endpoint
- Implement `/api/v1/upload` with multipart file handling, temp storage, inference call, result return
- Connect frontend `FileUpload` component to real upload API
- Implement `SpectrumViewer` component using FFT/PSD computed from IQ segments
- Implement `ConstellationViewer` component using windowed IQ sample scatter plot
- Implement `HypothesisRanking` table from `top_predictions` field of `SignalAnalysisResult`
- Add backend unit tests for API endpoints
- Add frontend component tests

### Future Roadmap

1. **Multi-Model Decision Fusion** - Combine CNN + DSP HGB classifier probabilities into a weighted ensemble for improved accuracy.
2. **Parametric Impairment Estimators** - Train dedicated regression heads to estimate CFO (Hz), IQ imbalance (amplitude + phase), and SNR (dB) from the signal.
3. **Out-of-Distribution (OOD) Detector** - Implement energy-based uncertainty thresholds to flag unclassifiable or corrupted signals as "Unknown / Low Confidence."
4. **Advanced Channel Modeling** - Integrate multi-path Rayleigh/Rician selective fading simulators and timing synchronization jitter into the synthetic generator.
5. **Live SDR Streaming** - Real-time ingestion from RTL-SDR or USRP devices via streaming buffer adapter.
6. **External Dataset Validation** - Formal evaluation on `subset_test.h5` (80k samples, shape [80000, 1024, 2], 7-class, 2 channel profiles: clean AWGN and multipath fading).
7. **Analog Modulation Generators** - Add CPFSK, GFSK, AM-DSB, AM-SSB, PAM4, WBFM to the synthetic generator to cover all 11 RadioML classes.
