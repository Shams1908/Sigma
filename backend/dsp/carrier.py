"""
Carrier frequency offset (CFO) estimation.

Two techniques are provided and combined:

1. Spectral peak method (primary):
   For BPSK/QPSK the carrier shows up as a peak in the squared / fourth-power
   spectrum.  For an arbitrary signal, squaring the complex envelope removes
   BPSK modulation (4th power for QPSK), leaving a tone at 2× (or 4×) the
   carrier offset, which can be located precisely.

2. Phase-difference (Δφ) method (fallback / cross-check):
   Mean phase increment per sample ≈ 2π·f_offset / f_sample.

The CFO is returned in Hz relative to the receiver's nominal centre frequency.
"""
from __future__ import annotations

import numpy as np

from dsp.psd import estimate_psd


def estimate_carrier_offset(
    iq: np.ndarray,
    sample_rate: float,
    nperseg: int = 256,
) -> float:
    """
    Estimate carrier frequency offset (Hz).

    Uses the 4th-power spectral peak method, which cancels QPSK modulation
    and leaves a tone at 4 × f_offset.  Falls back to the phase-difference
    estimator for signals with fewer than 64 samples.

    Args:
        iq:          1-D complex IQ array.
        sample_rate: Sample rate in Hz.
        nperseg:     Welch segment length for PSD.

    Returns:
        Carrier frequency offset in Hz.  Positive means the received carrier
        is above the nominal centre frequency.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    n = len(iq)

    if n >= 64:
        try:
            return _fourth_power_cfo(iq, sample_rate)
        except Exception:  # noqa: BLE001
            pass

    return _phase_diff_cfo(iq, sample_rate)


# ── Private helpers ───────────────────────────────────────────────────────────

def _fourth_power_cfo(iq: np.ndarray, sample_rate: float) -> float:
    """
    Fourth-power method: raises IQ to the 4th power to suppress BPSK/QPSK
    modulation, then finds the spectral peak at 4·f_offset.

    Works for BPSK (2nd power) and QPSK (4th power); 4th power handles both.
    """
    z = iq.astype(np.complex128) ** 4

    n_fft = min(int(2 ** np.ceil(np.log2(len(z)))), 65536)
    win = np.hanning(len(z))
    spectrum = np.fft.fft(z * win, n=n_fft)
    power = np.abs(spectrum)

    peak_idx = int(np.argmax(power))
    freqs = np.fft.fftfreq(n_fft, d=1.0 / sample_rate)
    f_at_4 = float(freqs[peak_idx])

    return f_at_4 / 4.0


def _phase_diff_cfo(iq: np.ndarray, sample_rate: float) -> float:
    """
    Phase-difference estimator: mean phase increment per sample.
    CFO = mean(Δφ) * sample_rate / (2π).
    """
    if len(iq) < 2:
        return 0.0
    phase_diff = np.angle(iq[1:] * np.conj(iq[:-1]))
    mean_delta = float(np.mean(phase_diff))
    return mean_delta * sample_rate / (2.0 * np.pi)


def remove_carrier_offset(iq: np.ndarray, sample_rate: float, cfo_hz: float) -> np.ndarray:
    """
    Correct a carrier frequency offset by multiplying with a complex exponential.

    Args:
        iq:          1-D complex IQ array.
        sample_rate: Sample rate in Hz.
        cfo_hz:      Carrier frequency offset to correct (Hz).

    Returns:
        Frequency-corrected complex IQ array.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    t = np.arange(len(iq), dtype=np.float64) / sample_rate
    correction = np.exp(-1j * 2.0 * np.pi * cfo_hz * t).astype(np.complex64)
    return iq * correction
