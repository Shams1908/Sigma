"""
Power Spectral Density estimation via Welch's method.

Welch's method averages periodograms over overlapping windows, giving a
lower-variance PSD estimate than a single FFT — essential for reliable
noise-floor detection.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch  # type: ignore[import]


def estimate_psd(
    iq: np.ndarray,
    sample_rate: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the two-sided PSD of a complex IQ signal using Welch's method.

    Args:
        iq:          1-D complex array (I + jQ).
        sample_rate: Sample rate in Hz.
        nperseg:     Samples per Welch segment (controls frequency resolution).
        noverlap:    Overlap between segments (defaults to nperseg // 2).
        window:      Window function name recognised by scipy.signal.welch.

    Returns:
        freqs:    Frequency bins in Hz (centred around 0).
        psd_dbhz: PSD in dBW/Hz.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) < nperseg:
        # Fall back to a single segment if the signal is very short
        nperseg = max(16, int(2 ** np.floor(np.log2(len(iq)))))

    if noverlap is None:
        noverlap = nperseg // 2

    freqs, psd = welch(
        iq,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,  # two-sided for complex signals
        scaling="density",
    )

    # fftshift so DC is at centre
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)

    psd = np.where(psd <= 0, 1e-30, psd)
    psd_dbhz = 10.0 * np.log10(psd)
    return freqs, psd_dbhz


def noise_floor_db(psd_dbhz: np.ndarray, percentile: float = 10.0) -> float:
    """
    Estimate the noise floor as the given percentile of the PSD (dB).

    Using the lowest 10 % of PSD bins as the noise floor is a standard
    heuristic; it works well for narrowband signals occupying < 90 % of
    the bandwidth.
    """
    return float(np.percentile(psd_dbhz, percentile))


def signal_power_db(psd_dbhz: np.ndarray, noise_floor: float) -> float:
    """
    Estimate the in-band signal power (dB) as the mean of bins above
    the noise floor.  Returns noise_floor if no bins qualify.
    """
    above = psd_dbhz[psd_dbhz > noise_floor]
    if len(above) == 0:
        return noise_floor
    return float(np.mean(above))
