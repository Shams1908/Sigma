"""
DSP package — signal parameter estimation and region detection utilities.

Public entry points:
- extract_parameters(): Runs estimators on a wideband IQ signal.
- estimate_region_parameters(): Estimates parameters for a specific detected SignalRegion.
"""
from __future__ import annotations

import logging

import numpy as np

from dsp.fft import canonical_to_complex
from dsp.snr import estimate_snr
from dsp.carrier import estimate_carrier_offset, remove_carrier_offset
from dsp.bandwidth import estimate_bandwidth
from dsp.symbol_rate import estimate_symbol_rate
from dsp.detection import SignalRegion, detect_signal_regions
from dsp.extraction import ExtractedSignal, extract_signal, lowpass_filter
from dsp.visualization import (
    generate_waveform_data,
    generate_fft_data,
    generate_psd_data,
    generate_spectrogram_data,
    generate_detection_data,
    generate_parameter_data,
    generate_visualization_data,
    to_json_serializable,
    downsample_for_visualization,
)
from dsp.pipeline import (
    DSPPipelineConfig,
    RegionResult,
    DSPPipelineResult,
    run_dsp_pipeline,
)



logger = logging.getLogger(__name__)


def extract_parameters(
    iq_2d: np.ndarray,
    sample_rate: float,
) -> dict:
    """
    Run all DSP estimators on a [2, N] canonical IQ array and return a
    dictionary with keys matching ParameterEstimate field names.

    Args:
        iq_2d:       Shape [2, N] float32 array (row 0 = I, row 1 = Q).
        sample_rate: Sample rate in Hz (must be > 0).

    Returns:
        dict with keys: snr, carrier_offset, bandwidth, symbol_rate_estimate
    """
    iq = canonical_to_complex(iq_2d)

    results: dict = {}

    try:
        results["snr"] = estimate_snr(iq, sample_rate)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SNR estimation failed: %s", exc)
        results["snr"] = -10.0

    try:
        results["carrier_offset"] = estimate_carrier_offset(iq, sample_rate)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Carrier offset estimation failed: %s", exc)
        results["carrier_offset"] = 0.0

    try:
        results["bandwidth"] = estimate_bandwidth(iq, sample_rate)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Bandwidth estimation failed: %s", exc)
        results["bandwidth"] = 0.0

    try:
        results["symbol_rate_estimate"] = estimate_symbol_rate(iq, sample_rate)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Symbol rate estimation failed: %s", exc)
        results["symbol_rate_estimate"] = None

    return results


def estimate_region_parameters(
    iq_2d: np.ndarray,
    sample_rate: float,
    region: SignalRegion,
) -> dict:
    """
    Estimate DSP parameters (SNR, CFO, BW, Symbol Rate) specific to a detected
    SignalRegion within a wideband IQ signal.

    Args:
        iq_2d:       Canonical IQ array [2, N] (or 1D complex array).
        sample_rate: Wideband sample rate in Hz.
        region:      SignalRegion instance from detect_signal_regions.

    Returns:
        dict containing region parameter estimates.
    """
    iq = canonical_to_complex(iq_2d) if isinstance(iq_2d, np.ndarray) and iq_2d.ndim == 2 else iq_2d

    # Shift signal to center the region at baseband (0 Hz)
    iq_shifted = remove_carrier_offset(iq, sample_rate, region.center_frequency)

    cfo_residual = estimate_carrier_offset(iq_shifted, sample_rate)
    estimated_cfo = float(region.center_frequency + cfo_residual)
    estimated_snr = float(region.snr_estimate_db if region.snr_estimate_db > -10.0 else estimate_snr(iq_shifted, sample_rate))
    estimated_bw = float(region.bandwidth if region.bandwidth > 0 else estimate_bandwidth(iq_shifted, sample_rate))
    estimated_sr = estimate_symbol_rate(iq_shifted, sample_rate)

    return {
        "start_frequency": float(region.start_frequency),
        "end_frequency": float(region.end_frequency),
        "center_frequency": float(region.center_frequency),
        "bandwidth": estimated_bw,
        "peak_power_db": float(region.peak_power_db),
        "average_power_db": float(region.average_power_db),
        "snr": estimated_snr,
        "carrier_offset": estimated_cfo,
        "symbol_rate_estimate": estimated_sr,
    }

