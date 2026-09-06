"""
SNR estimation from a complex IQ signal.

Two complementary approaches are used and the best available one is returned:

1. PSD-based (primary):  Uses the noise floor from Welch's PSD to compute
   SNR = signal_power_dB - noise_floor_dB.  Works for most signals.

2. Moment-based (fallback):  Uses the second-moment / fourth-moment ratio
   (M2M4 estimator, Wiesel et al.) when the PSD estimate is unreliable due
   to a very short signal.  Valid for BPSK/QPSK/QAM.
"""
from __future__ import annotations

import numpy as np

from dsp.psd import estimate_psd, noise_floor_db, signal_power_db


def estimate_snr(
    iq: np.ndarray,
    sample_rate: float,
    nperseg: int = 256,
) -> float:
    """
    Estimate SNR (dB) of a complex IQ signal.

    Returns the PSD-based SNR when the signal has enough samples for a
    reliable Welch estimate; falls back to the M2M4 moment estimator
    for very short signals (< 512 samples).

    Args:
        iq:          1-D complex IQ array.
        sample_rate: Sample rate in Hz.
        nperseg:     Welch segment length.

    Returns:
        Estimated SNR in dB (may be negative for very noisy signals).
    """
    iq = np.asarray(iq, dtype=np.complex64)

    if len(iq) >= 512:
        return _psd_snr(iq, sample_rate, nperseg)
    return _m2m4_snr(iq)


# ── Private helpers ───────────────────────────────────────────────────────────

def _psd_snr(iq: np.ndarray, sample_rate: float, nperseg: int) -> float:
    """SNR via noise floor from Welch's PSD."""
    try:
        _freqs, psd_db = estimate_psd(iq, sample_rate, nperseg=nperseg)
        floor = noise_floor_db(psd_db, percentile=10.0)
        sig_power = signal_power_db(psd_db, noise_floor=floor)
        snr = sig_power - floor
        # Clamp to a physically reasonable range
        return float(np.clip(snr, -10.0, 60.0))
    except Exception:  # noqa: BLE001
        return _m2m4_snr(iq)


def _m2m4_snr(iq: np.ndarray) -> float:
    """
    M2M4 moment-based SNR estimator for PSK/QAM.

    SNR_linear = sqrt(2*M2^2 - M4) / (M2 - sqrt(2*M2^2 - M4))
    where M2 = E[|x|^2], M4 = E[|x|^4].

    Reference: Wiesel, Goldberg, Messer (2002).
    """
    try:
        m2 = float(np.mean(np.abs(iq) ** 2))
        m4 = float(np.mean(np.abs(iq) ** 4))

        discriminant = 2.0 * m2 ** 2 - m4
        if discriminant <= 0 or m2 <= 0:
            return 10.0  # Default fallback

        signal_sq = np.sqrt(max(discriminant, 0.0))
        noise_sq = m2 - signal_sq

        if noise_sq <= 0:
            return 30.0  # Essentially noise-free

        snr_linear = signal_sq / noise_sq
        snr_db = 10.0 * np.log10(max(snr_linear, 1e-10))
        return float(np.clip(snr_db, -10.0, 60.0))
    except Exception:  # noqa: BLE001
        return 10.0  # Safe fallback
