"""
M8: Deterministic DSP Baseline Parameter Estimators.

Provides:
  1. Symbol Rate Estimation via instantaneous power spectral line / cyclostationary peak.
  2. SNR Estimation via the M2M4 moment-based estimator.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np


@dataclass(frozen=True)
class DSPSymbolRateResult:
    """Structured result of DSP symbol rate estimation."""
    estimate: float           # Estimated Baud rate (Hz)
    confidence: float         # 0.0 to 1.0
    uncertainty: float        # Estimated std deviation in Baud
    peak_snr_db: float        # Peak-to-average ratio of spectral line in dB
    method: str = "dsp_spectral_line"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimate": float(self.estimate),
            "confidence": float(self.confidence),
            "uncertainty": float(self.uncertainty),
            "peak_snr_db": float(self.peak_snr_db),
            "method": self.method,
        }


@dataclass(frozen=True)
class DSPSNRResult:
    """Structured result of DSP SNR estimation."""
    estimate: float           # Estimated SNR in dB
    confidence: float         # 0.0 to 1.0
    uncertainty: float        # Estimated std deviation in dB
    kurtosis_ratio: float     # M4 / M2^2
    method: str = "dsp_m2m4_moments"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimate": float(self.estimate),
            "confidence": float(self.confidence),
            "uncertainty": float(self.uncertainty),
            "kurtosis_ratio": float(self.kurtosis_ratio),
            "method": self.method,
        }


def _ensure_complex_1d(iq: np.ndarray) -> np.ndarray:
    """Standardizes input shape to a 1D complex64/128 array."""
    if not isinstance(iq, np.ndarray):
        raise TypeError(f"Input must be numpy ndarray, got {type(iq)}")
    if np.iscomplexobj(iq):
        return iq.flatten()
    if iq.ndim == 2:
        if iq.shape[0] == 2:
            return (iq[0] + 1j * iq[1]).astype(np.complex64)
        elif iq.shape[1] == 2:
            return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
        else:
            raise ValueError(f"Expected 2 rows or 2 columns for I/Q, got shape {iq.shape}")
    raise ValueError(f"Expected 1D complex or 2D [2, N] IQ array, got shape {iq.shape}")


def estimate_symbol_rate_dsp(
    iq: np.ndarray,
    sample_rate: float,
    min_baud: float = 2000.0,
    max_baud: Optional[float] = None,
) -> DSPSymbolRateResult:
    """
    Estimates the digital symbol rate (Baud) using transition envelope autocorrelation
    with fundamental lag extraction and parabolic peak refinement.
    
    Args:
        iq: Complex baseband samples or float array of shape [2, N].
        sample_rate: Sampling frequency in Hz.
        min_baud: Minimum plausible symbol rate in Hz.
        max_baud: Maximum plausible symbol rate in Hz (defaults to sample_rate / 2).
        
    Returns:
        DSPSymbolRateResult containing estimate, confidence, uncertainty, and peak metric.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    z = _ensure_complex_1d(iq)
    N = len(z)
    if N < 64:
        raise ValueError(f"Signal length ({N}) is too short for spectral estimation (minimum 64).")

    if max_baud is None:
        max_baud = sample_rate * 0.48
    max_baud = min(max_baud, sample_rate * 0.49)

    # 1. Compute transition energy envelope |diff(z)|^2
    trans = np.abs(np.diff(z, prepend=z[0])) ** 2
    trans_ac = trans - np.mean(trans)

    # 2. Autocorrelation of transition energy
    r = np.correlate(trans_ac, trans_ac, mode="full")[len(trans_ac) - 1 :]

    # 3. Determine lag search bounds (sps = sample_rate / baud)
    min_lag = max(2, int(np.floor(sample_rate / max_baud)))
    max_lag = min(len(r) // 2, int(np.ceil(sample_rate / min_baud)))

    if min_lag >= max_lag or max_lag >= len(r):
        return DSPSymbolRateResult(
            estimate=float((min_baud + max_baud) / 2.0),
            confidence=0.10,
            uncertainty=float((max_baud - min_baud) / 2.0),
            peak_snr_db=0.0,
        )

    r_band = r[: max_lag + 1]

    # 4. Find all local peaks in the valid lag band
    peaks = []
    for i in range(min_lag, len(r_band) - 1):
        if r_band[i] > r_band[i - 1] and r_band[i] > r_band[i + 1] and r_band[i] > 0:
            peaks.append((i, float(r_band[i])))

    if not peaks:
        # Fallback to spectral peak if transition correlation had no peaks
        fft_vals = np.abs(np.fft.rfft(trans_ac * np.hanning(N))) ** 2
        freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)
        valid_mask = (freqs >= min_baud) & (freqs <= max_baud)
        if np.any(valid_mask):
            peak_idx = int(np.argmax(fft_vals[valid_mask]))
            est_baud = float(freqs[valid_mask][peak_idx])
            return DSPSymbolRateResult(
                estimate=est_baud,
                confidence=0.30,
                uncertainty=float(sample_rate / N),
                peak_snr_db=3.0,
            )
        return DSPSymbolRateResult(
            estimate=float((min_baud + max_baud) / 2.0),
            confidence=0.10,
            uncertainty=float((max_baud - min_baud) / 2.0),
            peak_snr_db=0.0,
        )

    # 5. Extract fundamental symbol period (smallest lag among prominent peaks)
    max_peak_val = max(p[1] for p in peaks)
    # Retain peaks with at least 35% of the global max peak
    sig_peaks = [p for p in peaks if p[1] >= 0.35 * max_peak_val]
    fund_lag, fund_val = sig_peaks[0]

    # 6. Parabolic interpolation for sub-sample accuracy
    if 0 < fund_lag < len(r) - 1:
        y0, y1, y2 = r[fund_lag - 1], r[fund_lag], r[fund_lag + 1]
        denom = 2.0 * y1 - y0 - y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        refined_lag = float(fund_lag + np.clip(delta, -0.5, 0.5))
    else:
        refined_lag = float(fund_lag)

    refined_lag = max(1.0, refined_lag)
    est_baud = float(sample_rate / refined_lag)

    # 7. Confidence & Uncertainty estimation
    # Peak prominence relative to background noise floor
    noise_floor = float(np.median(np.abs(r_band[min_lag:]))) + 1e-12
    peak_ratio = fund_val / noise_floor
    peak_snr_db = float(10.0 * np.log10(max(1.0, peak_ratio)))

    # Confidence mapped from peak prominence and sample length
    conf = float(1.0 / (1.0 + np.exp(-(peak_snr_db - 6.0) / 2.5)))
    conf = float(np.clip(conf * min(1.0, np.sqrt(N / 512.0)), 0.10, 0.98))

    # Uncertainty in Baud
    uncertainty = float(max(est_baud * 0.005, (sample_rate / (refined_lag ** 2)) * (1.0 / max(1.0, peak_ratio))))

    return DSPSymbolRateResult(
        estimate=est_baud,
        confidence=conf,
        uncertainty=uncertainty,
        peak_snr_db=peak_snr_db,
    )


def estimate_snr_dsp(
    iq: np.ndarray,
    modulation_kurtosis: float = 1.0, # 1.0 for PSK, 1.32 for QAM16
) -> DSPSNRResult:
    """
    Estimates Signal-to-Noise Ratio (SNR in dB) using the classic M2M4 moment estimator.
    
    Formula:
      M2 = E[|r|^2] = S + 2*sigma^2
      M4 = E[|r|^4] = ka*S^2 + 4*S*sigma^2 + 8*sigma^4
      Ratio z = M4 / M2^2
      For ka=1: S = M2 * sqrt(2 - z), N = M2 - S
      SNR = 10 * log10(S / N)
    """
    z = _ensure_complex_1d(iq)
    N = len(z)
    if N < 32:
        raise ValueError(f"Signal length ({N}) is too short for moment estimation.")

    r2 = np.abs(z) ** 2
    m2 = float(np.mean(r2))
    m4 = float(np.mean(r2 ** 2))

    if m2 <= 1e-12:
        return DSPSNRResult(estimate=-10.0, confidence=0.05, uncertainty=15.0, kurtosis_ratio=2.0)

    ratio = m4 / (m2 ** 2)

    # Valid physical ratio for complex signals is between 1.0 (clean constant envelope)
    # and 2.0 (pure Gaussian noise)
    ka = float(modulation_kurtosis)
    # Discriminant under ka assumption
    disc = (2.0 - ratio) / max(0.1, 2.0 - ka)
    
    if disc > 0:
        s_frac = float(np.sqrt(np.clip(disc, 0.0, 1.0)))
        n_frac = max(1e-5, 1.0 - s_frac)
        snr_linear = s_frac / n_frac
        snr_db = float(10.0 * np.log10(snr_linear))
        # Confidence depends on sample count and physical plausibility of ratio
        conf = float(1.0 - abs(ratio - 1.5) * 0.5) if (1.0 <= ratio <= 2.0) else 0.20
        conf = float(np.clip(conf * min(1.0, np.sqrt(N / 512.0)), 0.10, 0.95))
        unc = float(max(0.5, 3.0 / (conf * np.sqrt(N / 128.0))))
    else:
        # Heavily noise-dominated or out of bounds
        snr_db = -5.0
        conf = 0.15
        unc = 8.0

    # Bound reasonable SNR range [-20 dB, +35 dB]
    snr_db = float(np.clip(snr_db, -20.0, 35.0))

    return DSPSNRResult(
        estimate=snr_db,
        confidence=conf,
        uncertainty=unc,
        kurtosis_ratio=ratio,
    )
