"""
SNR estimation from a complex IQ signal.

Two complementary approaches are used and the best available one is returned:

1. PSD-based (primary): Uses the noise floor from Welch's PSD to compute
   SNR = signal_power_dB - noise_floor_dB. Works for most signals.

2. Moment-based (fallback): Uses the second-moment / fourth-moment ratio
   (M2M4 estimator, Wiesel et al.) when the PSD estimate is unreliable due
   to a very short signal. Valid for BPSK/QPSK/QAM.

Mathematical Behavior:
- For valid signals, numerical estimates are clamped to a physical dynamic range of [-10.0, 60.0] dB.
- For zero-power signals (all zeros), empty inputs, or non-finite inputs, SNR (0/0) is mathematically
  undefined and None is returned to avoid reporting an arbitrary/fabricated SNR number.
"""
from __future__ import annotations

import numpy as np

from dsp.fft import canonical_to_complex
from dsp.psd import estimate_psd, noise_floor_db, signal_power_db


def estimate_snr(
    iq: np.ndarray,
    sample_rate: float,
    nperseg: int = 256,
) -> float | None:
    """
    Estimate SNR (dB) of a complex IQ signal.

    Returns the PSD-based SNR when the signal has enough samples for a
    reliable Welch estimate; falls back to the M2M4 moment estimator
    for very short signals (< 512 samples). Returns None if SNR is undefined.

    Args:
        iq:          1-D complex IQ array or [2, N] canonical IQ array.
        sample_rate: Sample rate in Hz.
        nperseg:     Welch segment length.

    Returns:
        Estimated SNR in dB (clamped between -10.0 and 60.0), or None if undefined.
    """
    if isinstance(iq, np.ndarray) and iq.ndim == 2:
        iq = canonical_to_complex(iq)
    else:
        iq = np.asarray(iq, dtype=np.complex64)

    if iq.ndim != 1 or len(iq) == 0:
        return None

    if not np.all(np.isfinite(iq)):
        return None

    # Zero signal check (0/0 SNR is mathematically undefined)
    if np.all(iq == 0):
        return None

    if len(iq) >= 512:
        return _psd_snr(iq, sample_rate, nperseg)
    return _m2m4_snr(iq)


# ── Private helpers ───────────────────────────────────────────────────────────

def _psd_snr(iq: np.ndarray, sample_rate: float, nperseg: int) -> float | None:
    """SNR via noise floor from Welch's PSD."""
    try:
        _freqs, psd_db = estimate_psd(iq, sample_rate, nperseg=nperseg)
        floor = noise_floor_db(psd_db, percentile=10.0)
        sig_power = signal_power_db(psd_db, noise_floor=floor)
        snr = sig_power - floor
        # Clamp to a physically reasonable dynamic range
        return float(np.clip(snr, -10.0, 60.0))
    except Exception:  # noqa: BLE001
        return _m2m4_snr(iq)


def _m2m4_snr(iq: np.ndarray) -> float | None:
    """
    M2M4 moment-based SNR estimator for PSK/QAM.

    SNR_linear = sqrt(2*M2^2 - M4) / (M2 - sqrt(2*M2^2 - M4))
    where M2 = E[|x|^2], M4 = E[|x|^4].

    Reference: Wiesel, Goldberg, Messer (2002).
    """
    try:
        m2 = float(np.mean(np.abs(iq) ** 2))
        m4 = float(np.mean(np.abs(iq) ** 4))

        if m2 <= 0:
            return None

        discriminant = 2.0 * m2 ** 2 - m4
        if discriminant <= 0:
            return None

        signal_sq = np.sqrt(max(discriminant, 0.0))
        noise_sq = m2 - signal_sq

        if noise_sq <= 0:
            return 30.0  # Essentially noise-free

        snr_linear = signal_sq / noise_sq
        snr_db = 10.0 * np.log10(max(snr_linear, 1e-10))
        return float(np.clip(snr_db, -10.0, 60.0))
    except Exception:  # noqa: BLE001
        return None


