"""
Symbol rate estimation from a complex IQ signal.

Two independent methods are implemented and their results averaged:

1. Cyclostationary / spectral correlation (primary):
   A digitally modulated signal has cyclostationary features at multiples of
   the symbol rate.  Squaring the absolute value of the signal and computing
   its FFT reveals peaks at f = k·Rs (k=1,2,…).  The first significant
   peak above DC gives Rs.

2. Autocorrelation zero-crossing (secondary):
   The autocorrelation of |iq|² has zeros near multiples of the symbol
   period T = 1/Rs.  The lag of the first zero crossing ≈ T.

The two estimates are averaged (median when they diverge by more than 50 %).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks  # type: ignore[import]


def estimate_symbol_rate(
    iq: np.ndarray,
    sample_rate: float,
    min_rate: float = 100.0,
    max_rate: float | None = None,
) -> float:
    """
    Estimate symbol rate (symbols / second) from a complex IQ signal.

    Args:
        iq:          1-D complex IQ array.
        sample_rate: Sample rate in Hz.
        min_rate:    Minimum plausible symbol rate (Hz).
        max_rate:    Maximum plausible symbol rate (Hz).  Defaults to
                     sample_rate / 2.

    Returns:
        Estimated symbol rate in symbols/second.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if max_rate is None:
        max_rate = sample_rate / 2.0

    estimates: list[float] = []

    try:
        rs_spectral = _cyclostationary_estimate(iq, sample_rate, min_rate, max_rate)
        if rs_spectral is not None:
            estimates.append(rs_spectral)
    except Exception:  # noqa: BLE001
        pass

    try:
        rs_autocorr = _autocorr_estimate(iq, sample_rate, min_rate, max_rate)
        if rs_autocorr is not None:
            estimates.append(rs_autocorr)
    except Exception:  # noqa: BLE001
        pass

    if not estimates:
        # Last-resort fallback: assume SPS ≈ 4
        return float(sample_rate / 4.0)

    if len(estimates) == 2:
        ratio = max(estimates) / (min(estimates) + 1e-9)
        if ratio > 1.5:
            # Estimates diverge — trust the cyclostationary one
            return estimates[0]
        return float(np.mean(estimates))

    return estimates[0]


# ── Private helpers ───────────────────────────────────────────────────────────

def _cyclostationary_estimate(
    iq: np.ndarray,
    sample_rate: float,
    min_rate: float,
    max_rate: float,
) -> float | None:
    """
    Spectral correlation / cyclostationary method.
    Computes |iq|² then finds spectral peaks that correspond to Rs.
    """
    power_signal = np.abs(iq).astype(np.float64) ** 2
    # Remove DC component
    power_signal -= np.mean(power_signal)

    n_fft = min(int(2 ** np.ceil(np.log2(len(power_signal)))), 65536)
    win = np.hanning(len(power_signal))
    spectrum = np.abs(np.fft.rfft(power_signal * win, n=n_fft))

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    # Restrict to plausible symbol rate range (> 0 Hz offset from DC)
    min_idx = max(1, int(min_rate / df))
    max_idx = min(len(freqs) - 1, int(max_rate / df))

    if min_idx >= max_idx:
        return None

    sub_spectrum = spectrum[min_idx:max_idx]
    if len(sub_spectrum) == 0:
        return None

    # Find peaks with prominence filtering to avoid noise peaks
    prominence_threshold = float(np.std(sub_spectrum))
    peaks, props = find_peaks(sub_spectrum, prominence=prominence_threshold)

    if len(peaks) == 0:
        # No prominent peaks — take the argmax in the range
        peak_local = int(np.argmax(sub_spectrum))
        return float(freqs[min_idx + peak_local])

    # Pick the peak with the highest prominence
    best_peak = peaks[int(np.argmax(props["prominences"]))]
    return float(freqs[min_idx + best_peak])


def _autocorr_estimate(
    iq: np.ndarray,
    sample_rate: float,
    min_rate: float,
    max_rate: float,
) -> float | None:
    """
    Autocorrelation zero-crossing estimator.
    The first zero of the normalised autocorrelation of |iq|² ≈ T_symbol.
    """
    env = np.abs(iq).astype(np.float64) ** 2
    env -= np.mean(env)

    n = len(env)
    max_lag = min(n - 1, int(sample_rate / min_rate) + 1)
    min_lag = max(1, int(sample_rate / max_rate) - 1)

    # Full-length autocorrelation via FFT
    f = np.fft.rfft(env, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] == 0:
        return None
    acf /= acf[0]

    lags = np.arange(n)
    search_acf = acf[min_lag:max_lag]
    search_lags = lags[min_lag:max_lag]

    if len(search_acf) < 2:
        return None

    # Find first zero crossing (sign change)
    for i in range(len(search_acf) - 1):
        if search_acf[i] >= 0 and search_acf[i + 1] < 0:
            # Linear interpolation to find exact zero
            t = search_acf[i] / (search_acf[i] - search_acf[i + 1])
            lag_samples = search_lags[i] + t
            return float(sample_rate / lag_samples)

    return None
