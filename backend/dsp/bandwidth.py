"""
Occupied bandwidth estimation.

Uses the PSD to find the band that contains a specified fraction of the
total signal power (default 99 %).  This is the standard "occupied bandwidth"
definition used by spectrum analysers.
"""
from __future__ import annotations

import numpy as np

from dsp.psd import estimate_psd, noise_floor_db


def estimate_bandwidth(
    iq: np.ndarray,
    sample_rate: float,
    power_fraction: float = 0.99,
    nperseg: int = 256,
) -> float:
    """
    Estimate the occupied bandwidth (Hz) of a complex IQ signal.

    Finds the narrowest contiguous frequency band that captures
    `power_fraction` (0–1) of the total signal power.

    Args:
        iq:              1-D complex IQ array.
        sample_rate:     Sample rate in Hz.
        power_fraction:  Fraction of total power to capture (default 0.99).
        nperseg:         Welch segment size.

    Returns:
        Occupied bandwidth in Hz.
    """
    iq = np.asarray(iq, dtype=np.complex64)

    freqs, psd_db = estimate_psd(iq, sample_rate, nperseg=nperseg)

    # Convert dB back to linear for power integration
    psd_lin = 10.0 ** (psd_db / 10.0)

    # Only consider bins above the noise floor
    floor = noise_floor_db(psd_db, percentile=10.0)
    floor_lin = 10.0 ** (floor / 10.0)
    signal_psd = np.where(psd_lin > floor_lin, psd_lin, 0.0)

    total_power = np.sum(signal_psd)
    if total_power == 0:
        # No signal detected; return a minimum resolution bandwidth
        df = freqs[1] - freqs[0] if len(freqs) > 1 else sample_rate
        return float(abs(df))

    # Cumulative power from low to high frequency (freqs already fftshifted)
    cumpower = np.cumsum(signal_psd)
    threshold = total_power * power_fraction
    low_margin = (1.0 - power_fraction) / 2.0
    high_margin = 1.0 - low_margin

    low_idx = int(np.searchsorted(cumpower, total_power * low_margin))
    high_idx = int(np.searchsorted(cumpower, total_power * high_margin))

    low_idx = max(0, min(low_idx, len(freqs) - 1))
    high_idx = max(0, min(high_idx, len(freqs) - 1))

    bw = abs(float(freqs[high_idx]) - float(freqs[low_idx]))
    # Enforce a minimum of one frequency bin
    df = abs(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    return max(bw, float(df))
