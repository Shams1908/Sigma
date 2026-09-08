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

from dsp.fft import canonical_to_complex


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
        iq:          1-D complex IQ array or [2, N] canonical IQ array.
        sample_rate: Sample rate in Hz.
        nperseg:     Welch segment length for PSD.

    Returns:
        Carrier frequency offset in Hz.  Positive means the received carrier
        is above the nominal centre frequency.
    """
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")

    if isinstance(iq, np.ndarray) and iq.ndim == 2:
        iq = canonical_to_complex(iq)
    else:
        iq = np.asarray(iq, dtype=np.complex64)

    if iq.ndim != 1 or len(iq) == 0 or not np.all(np.isfinite(iq)):
        return 0.0

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
    Fourth-power spectral peak method.

    Concept:
        Raises complex envelope to 4th power (z = x^4) to eliminate BPSK/QPSK
        phase transitions and isolate a single carrier harmonic at 4·f_offset.

    Assumptions & Limitations:
        - Assumes BPSK or QPSK phase symmetry (pi or pi/2 rotational symmetry).
        - Unreliable for higher-order QAM (16-QAM, 64-QAM), FSK, or asymmetric modulations.
        - May suffer from spectral aliasing if 4 · |f_offset| > sample_rate / 2.
    """
    if len(iq) == 0:
        return 0.0

    z = iq.astype(np.complex128) ** 4

    n_fft = min(int(2 ** np.ceil(np.log2(len(z)))), 65536)
    if n_fft <= 0:
        return 0.0

    win = np.hanning(len(z))
    spectrum = np.fft.fft(z * win, n=n_fft)
    power = np.abs(spectrum)

    peak_idx = int(np.argmax(power))
    freqs = np.fft.fftfreq(n_fft, d=1.0 / sample_rate)
    f_at_4 = float(freqs[peak_idx])

    return f_at_4 / 4.0


def _phase_diff_cfo(iq: np.ndarray, sample_rate: float) -> float:
    """
    Phase-difference estimator.

    Concept:
        Computes mean phase increment per sample: CFO = mean(Δφ) * fs / (2π).

    Assumptions & Limitations:
        - Assumes unmodulated carrier or low-deviation tone.
        - High modulation index or wideband data modulation causes rapid phase swings
          that introduce variance and bias into the mean phase increment.
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
        iq:          1-D complex IQ array or [2, N] canonical IQ array.
        sample_rate: Sample rate in Hz.
        cfo_hz:      Carrier frequency offset to correct (Hz).

    Returns:
        Frequency-corrected IQ array (matching input shape).
    """
    is_2d = isinstance(iq, np.ndarray) and iq.ndim == 2 and iq.shape[0] == 2

    if is_2d:
        complex_iq = canonical_to_complex(iq)
    else:
        complex_iq = np.asarray(iq, dtype=np.complex64)

    t = np.arange(len(complex_iq), dtype=np.float64) / sample_rate
    correction = np.exp(-1j * 2.0 * np.pi * cfo_hz * t).astype(np.complex64)
    corrected = complex_iq * correction

    if is_2d:
        return np.stack([corrected.real, corrected.imag]).astype(np.float32)
    return corrected

