"""
DSP package — signal parameter estimation utilities.

Public entry point: extract_parameters() runs all estimators and returns
a ParameterEstimate-compatible dict suitable for constructing the DB model.
"""
from __future__ import annotations

import logging

import numpy as np

from dsp.fft import canonical_to_complex
from dsp.snr import estimate_snr
from dsp.carrier import estimate_carrier_offset
from dsp.bandwidth import estimate_bandwidth
from dsp.symbol_rate import estimate_symbol_rate

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
        results["snr"] = 0.0

    try:
        results["carrier_offset"] = estimate_carrier_offset(iq, sample_rate)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Carrier offset estimation failed: %s", exc)
        results["carrier_offset"] = 0.0

    try:
        results["bandwidth"] = estimate_bandwidth(iq, sample_rate)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Bandwidth estimation failed: %s", exc)
        results["bandwidth"] = sample_rate / 2.0

    try:
        results["symbol_rate_estimate"] = estimate_symbol_rate(iq, sample_rate)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Symbol rate estimation failed: %s", exc)
        results["symbol_rate_estimate"] = sample_rate / 4.0

    return results
