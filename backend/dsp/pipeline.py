"""
DSP Pipeline Orchestrator — End-to-End DSP Signal Processing.

Orchestrates existing DSP components:
1. Signal validation & IQ format conversion
2. IQ Signal Normalization
3. Spectral estimations (FFT, PSD, STFT as configured)
4. PSD-based Signal Region Detection
5. Per-region Parameter Estimation and Signal Extraction
6. Lightweight, JSON-serializable Visualization Data Generation

Does not duplicate DSP algorithms — reuses existing modules cleanly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, is_dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from preprocessing.normalize import validate_canonical_iq, normalize_signal
from dsp.fft import canonical_to_complex, compute_fft
from dsp.psd import estimate_psd
from dsp.spectrogram import compute_spectrogram
from dsp.detection import SignalRegion, detect_signal_regions
from dsp.extraction import ExtractedSignal, extract_signal
from dsp.visualization import generate_visualization_data, to_json_serializable

logger = logging.getLogger(__name__)


@dataclass
class DSPPipelineConfig:
    """
    Configuration options for the end-to-end DSP pipeline.
    """
    enable_normalization: bool = True
    enable_fft: bool = True
    enable_psd: bool = True
    enable_stft: bool = False  # Disabled by default to avoid heavy matrix payloads
    generate_visualization: bool = True
    extract_detected_signals: bool = True
    max_visualization_points: int = 2048
    threshold_db: float = 6.0
    min_bins: int = 3
    merge_gap_bins: int = 2


@dataclass
class RegionResult:
    """
    Processing results associated with a single detected SignalRegion.
    """
    region: SignalRegion
    parameters: Dict[str, Any]
    extracted_signal: Optional[ExtractedSignal] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert RegionResult to a JSON-serializable dictionary."""
        out: Dict[str, Any] = {
            "region": to_json_serializable(self.region),
            "parameters": to_json_serializable(self.parameters),
        }
        if self.extracted_signal is not None:
            out["extracted_signal"] = {
                "center_frequency": float(self.extracted_signal.original_region.center_frequency),
                "bandwidth": float(self.extracted_signal.original_region.bandwidth),
                "sample_rate": float(self.extracted_signal.sample_rate),
                "num_samples": int(self.extracted_signal.iq.shape[1]) if self.extracted_signal.iq.ndim == 2 else int(self.extracted_signal.iq.shape[0]),
            }
        else:
            out["extracted_signal"] = None
        return out


@dataclass
class DSPPipelineResult:
    """
    Structured container holding the outputs of an end-to-end DSP pipeline run.
    """
    normalized_signal: np.ndarray  # Shape [2, N] canonical IQ (float32)
    sample_rate: float
    detected_regions: List[SignalRegion] = field(default_factory=list)
    region_results: List[RegionResult] = field(default_factory=list)
    fft_data: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (freqs, magnitude_db)
    psd_data: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (freqs, power_db)
    stft_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None  # (times, freqs, mag_db)
    visualization_data: Optional[Dict[str, Any]] = None

    def to_dict(self, include_visualization: bool = True) -> Dict[str, Any]:
        """
        Convert complete pipeline results into a clean, JSON-serializable dictionary.
        Does not serialize huge raw extracted IQ arrays.
        """
        out: Dict[str, Any] = {
            "sample_rate": float(self.sample_rate),
            "num_samples": int(self.normalized_signal.shape[1]) if self.normalized_signal.ndim == 2 else 0,
            "detected_regions": to_json_serializable(self.detected_regions),
            "region_results": [rr.to_dict() for rr in self.region_results],
        }

        if self.fft_data is not None:
            out["fft"] = {
                "frequency": to_json_serializable(self.fft_data[0]),
                "magnitude_db": to_json_serializable(self.fft_data[1]),
            }

        if self.psd_data is not None:
            out["psd"] = {
                "frequency": to_json_serializable(self.psd_data[0]),
                "power_db": to_json_serializable(self.psd_data[1]),
            }

        if self.stft_data is not None:
            out["stft"] = {
                "time": to_json_serializable(self.stft_data[0]),
                "frequency": to_json_serializable(self.stft_data[1]),
                "spectrogram_db": to_json_serializable(self.stft_data[2]),
            }

        if include_visualization and self.visualization_data is not None:
            out["visualization"] = to_json_serializable(self.visualization_data)

        return out


def run_dsp_pipeline(
    iq: np.ndarray,
    sample_rate: float,
    config: Optional[DSPPipelineConfig] = None,
) -> DSPPipelineResult:
    """
    Run the end-to-end DSP pipeline on wideband IQ data.

    Logical Workflow:
        Raw IQ Input -> Validation -> Normalization -> PSD/FFT/STFT ->
        Signal Detection -> Region Parameter Estimation & Signal Extraction ->
        Structured Pipeline Result & Visualization Data

    Args:
        iq:          Canonical IQ array [2, N] or complex 1D array.
        sample_rate: Wideband sample rate in Hz (> 0).
        config:      Optional DSPPipelineConfig settings.

    Returns:
        DSPPipelineResult containing all stage outputs and structured data.
    """
    if config is None:
        config = DSPPipelineConfig()

    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")

    # ── Step 1: Input Validation & Canonical IQ Conversion ────────────────────
    if isinstance(iq, np.ndarray) and iq.ndim == 2:
        validate_canonical_iq(iq)
        iq_canonical = np.asarray(iq, dtype=np.float32)
    elif isinstance(iq, np.ndarray) and iq.ndim == 1:
        if not np.all(np.isfinite(iq)):
            raise ValueError("Input complex IQ array contains non-finite values (NaN or Inf)")
        if len(iq) == 0:
            raise ValueError("Input IQ array must contain at least one sample")
        complex_iq_in = np.asarray(iq, dtype=np.complex64)
        iq_canonical = np.array([complex_iq_in.real, complex_iq_in.imag], dtype=np.float32)
    elif not isinstance(iq, np.ndarray):
        raise TypeError(f"Input must be a NumPy ndarray, got {type(iq).__name__}")
    else:
        raise ValueError(f"Input array must be 1D complex or 2D canonical IQ [2, N], got shape {iq.shape}")

    # ── Step 2: Normalization ─────────────────────────────────────────────────
    if config.enable_normalization:
        iq_normalized = normalize_signal(iq_canonical)
    else:
        iq_normalized = iq_canonical.copy()

    iq_complex = canonical_to_complex(iq_normalized)

    # ── Step 3: Spectral Computations (FFT, PSD, STFT as configured) ─────────
    psd_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if config.enable_psd:
        freqs_p, psd_db = estimate_psd(iq_complex, sample_rate)
        psd_data = (freqs_p, psd_db)

    fft_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if config.enable_fft:
        freqs_f, fft_db = compute_fft(iq_complex, sample_rate)
        fft_data = (freqs_f, fft_db)

    stft_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    if config.enable_stft:
        freqs_s, times_s, stft_db = compute_spectrogram(iq_complex, sample_rate)
        stft_data = (times_s, freqs_s, stft_db)

    # ── Step 4: PSD-based Signal Detection ───────────────────────────────────
    detected_regions = detect_signal_regions(
        iq_normalized,
        sample_rate,
        threshold_db=config.threshold_db,
        min_bins=config.min_bins,
        merge_gap_bins=config.merge_gap_bins,
        psd=psd_data,
    )

    # ── Step 5: Region Parameter Estimation & Signal Extraction ───────────────
    from dsp import estimate_region_parameters  # Lazy import to prevent circularity

    region_results: List[RegionResult] = []
    for region in detected_regions:
        params = estimate_region_parameters(iq_normalized, sample_rate, region)
        extracted: Optional[ExtractedSignal] = None
        if config.extract_detected_signals:
            extracted = extract_signal(
                iq_normalized,
                sample_rate,
                region=region,
            )
        region_results.append(
            RegionResult(
                region=region,
                parameters=params,
                extracted_signal=extracted,
            )
        )

    # ── Step 6: Visualization Data Generation ─────────────────────────────────
    visualization_data: Optional[Dict[str, Any]] = None
    if config.generate_visualization:
        visualization_data = generate_visualization_data(
            iq=iq_normalized,
            sample_rate=sample_rate,
            include_waveform=True,
            include_fft=config.enable_fft,
            include_psd=config.enable_psd,
            include_spectrogram=config.enable_stft,
            regions=detected_regions,
            params=region_results[0].parameters if len(region_results) == 1 else None,
            max_points=config.max_visualization_points,
            fft_data=fft_data,
            psd_data=psd_data,
            stft_data=stft_data,
        )

    return DSPPipelineResult(
        normalized_signal=iq_normalized,
        sample_rate=sample_rate,
        detected_regions=detected_regions,
        region_results=region_results,
        fft_data=fft_data,
        psd_data=psd_data,
        stft_data=stft_data,
        visualization_data=visualization_data,
    )
