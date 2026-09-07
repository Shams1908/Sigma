"""
Short-Time Fourier Transform (STFT) spectrogram for visualisation / diagnostics.
Not used directly in the parameter estimation pipeline but available for
reporting and debug tooling.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram  # type: ignore[import]


def compute_spectrogram(
    iq: np.ndarray,
    sample_rate: float,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the STFT spectrogram of a complex IQ signal.

    Returns:
        freqs:  Frequency array (Hz), centred at 0.
        times:  Time array (seconds).
        Sxx_db: Spectrogram matrix in dB, shape (len(freqs), len(times)).
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, times, Sxx = scipy_spectrogram(
        iq,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
    )

    freqs = np.fft.fftshift(freqs)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    Sxx = np.where(Sxx <= 0, 1e-30, Sxx)
    Sxx_db = 10.0 * np.log10(Sxx)
    return freqs, times, Sxx_db
