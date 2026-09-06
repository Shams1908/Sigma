"""
FFT helpers for the SIGMA DSP pipeline.

All functions accept a 1-D complex IQ array (I + jQ) and return
frequency-domain results as plain NumPy arrays.
"""
from __future__ import annotations

import numpy as np


def compute_fft(
    iq: np.ndarray,
    sample_rate: float,
    n_fft: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the single-sided power spectrum of a complex IQ signal.

    Args:
        iq:          1-D complex array  (I + jQ).
        sample_rate: Sample rate in Hz.
        n_fft:       FFT size (defaults to next power-of-2 >= len(iq), capped at 65536).
        window:      NumPy window name ('hann', 'blackman', …).

    Returns:
        freqs:       Frequency bins in Hz (centred at 0, double-sided, length n_fft).
        spectrum_db: Power spectrum in dB (length n_fft).
    """
    iq = np.asarray(iq, dtype=np.complex64)
    n = len(iq)
    if n == 0:
        raise ValueError("IQ array is empty.")

    if n_fft is None:
        n_fft = min(int(2 ** np.ceil(np.log2(n))), 65536)

    win = np.hanning(n) if window == "hann" else getattr(np, window)(n)
    win = win.astype(np.float32)
    windowed = iq[:n] * win

    spectrum = np.fft.fftshift(np.fft.fft(windowed, n=n_fft))
    power = np.abs(spectrum) ** 2
    # Avoid log(0)
    power = np.where(power == 0, 1e-20, power)
    spectrum_db = 10.0 * np.log10(power)

    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sample_rate))
    return freqs, spectrum_db


def canonical_to_complex(iq_2d: np.ndarray) -> np.ndarray:
    """
    Convert a [2, N] canonical IQ array (I row 0, Q row 1) to a 1-D complex array.
    Also handles [N, 2] layout (second axis is I/Q).
    """
    iq_2d = np.asarray(iq_2d)
    if iq_2d.ndim == 2 and iq_2d.shape[0] == 2:
        return (iq_2d[0] + 1j * iq_2d[1]).astype(np.complex64)
    if iq_2d.ndim == 2 and iq_2d.shape[1] == 2:
        return (iq_2d[:, 0] + 1j * iq_2d[:, 1]).astype(np.complex64)
    raise ValueError(f"Unexpected IQ array shape: {iq_2d.shape}")


def peak_frequency(freqs: np.ndarray, spectrum_db: np.ndarray) -> float:
    """Return the frequency (Hz) of the maximum spectral power."""
    return float(freqs[np.argmax(spectrum_db)])
