# Development Phases Roadmap

This document outlines the phased development roadmap for SIGMA (Signal Intelligence & Guided Modulation Analysis).

## Phase 0 — Project Setup
Project structure, FastAPI setup, React setup, configuration. Establish directory rules, build pipelines, configuration handling, and basic health checking.

## Phase 1 — File Ingestion
IQ and WAV file parsing and metadata extraction. Focus on handling high-throughput binary files, mapping complex signal samples, and extracting embedded recording metadata.

## Phase 2 — Signal Triage
FFT, PSD, spectrogram, signal detection, SNR and bandwidth estimation. Transition signals from time domain to frequency domain and extract initial channel/signal boundaries.

## Phase 3 — Dataset and Feature Pipeline
Synthetic signal generation, feature extraction and dataset preparation. Build datasets containing diverse RF channel configurations (fading, noise, carrier offsets) for ML model validation.

## Phase 4 — Modulation Recognition
Feature-based baseline classifier followed by optional CNN classifier. Enable automatic classification of common digital and analog modulation styles.

## Phase 5 — Signal Recovery
Carrier offset estimation, matched filtering and timing recovery. Synchronize the signal to retrieve symbols from the analog waveforms.

## Phase 6 — Demodulation
Support FSK, BPSK, QPSK and 16-QAM. Extract raw bit streams from synchronized signal symbols.

## Phase 7 — Hypothesis Validation
Generate multiple signal hypotheses, evaluate downstream quality and rank results. Validate signal features and demodulation decodes against confidence scoring rules.

## Phase 8 — GUI Integration and Demo
Connect backend analysis to the frontend and create the final demonstration workflow. Enable visual control over ingestion, signal analysis, constellation viewing, and hypothesis tracking.
