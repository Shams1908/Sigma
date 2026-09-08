"""
PSD-based signal region detection module.

Identifies occupied frequency bands in a wideband IQ signal using Welch PSD
estimation and thresholding against the noise floor.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from preprocessing.normalize import validate_canonical_iq
from dsp.fft import canonical_to_complex
from dsp.psd import estimate_psd, noise_floor_db


@dataclass
class SignalRegion:
    """
    Metadata for a detected occupied signal region in frequency domain.
    """
    start_frequency: float
    end_frequency: float
    center_frequency: float
    bandwidth: float
    peak_power_db: float
    average_power_db: float
    snr_estimate_db: float


def merge_spectral_gaps(mask: np.ndarray, merge_gap_bins: int = 2) -> np.ndarray:
    """
    Merge small spectral gaps (contiguous False runs) of length <= merge_gap_bins
    that occur between occupied (True) regions.
    """
    if merge_gap_bins <= 0 or len(mask) == 0:
        return np.asarray(mask, dtype=bool).copy()

    merged = np.asarray(mask, dtype=bool).copy()
    n_bins = len(merged)
    in_gap = False
    gap_start = -1

    for i in range(n_bins):
        if not merged[i]:
            if not in_gap:
                in_gap = True
                gap_start = i
        else:
            if in_gap:
                in_gap = False
                gap_len = i - gap_start
                if gap_start > 0 and gap_len <= merge_gap_bins:
                    merged[gap_start:i] = True

    return merged


def find_contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Find contiguous True segments in a boolean mask.

    Returns:
        List of (start_idx, end_idx) tuples with inclusive index bounds.
    """
    mask = np.asarray(mask, dtype=bool)
    regions: list[tuple[int, int]] = []
    n_bins = len(mask)
    idx = 0

    while idx < n_bins:
        if mask[idx]:
            start_idx = idx
            while idx < n_bins and mask[idx]:
                idx += 1
            end_idx = idx - 1
            regions.append((start_idx, end_idx))
        else:
            idx += 1

    return regions


def detect_signal_regions(
    iq: np.ndarray,
    sample_rate: float,
    threshold_db: float = 6.0,
    min_bins: int = 3,
    merge_gap_bins: int = 2,
    nperseg: int = 256,
    psd: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[SignalRegion]:
    """
    Detect occupied frequency regions in a canonical IQ signal using Welch PSD.

    Args:
        iq:             Canonical IQ array [2, N] (or 1D complex array).
        sample_rate:    Sampling rate in Hz (must be > 0).
        threshold_db:   Detection threshold in dB above estimated noise floor.
        min_bins:       Minimum contiguous occupied frequency bins required.
        merge_gap_bins: Maximum gap (in bins) between occupied regions to merge.
        nperseg:        Segment length for Welch PSD calculation.

    Returns:
        List of detected SignalRegion dataclass instances.
    """
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")

    # Validate and convert input to 1D complex IQ
    if isinstance(iq, np.ndarray) and iq.ndim == 2:
        validate_canonical_iq(iq)
        iq_complex = canonical_to_complex(iq)
    elif isinstance(iq, np.ndarray) and iq.ndim == 1:
        if not np.all(np.isfinite(iq)):
            raise ValueError("Input complex IQ array contains non-finite values (NaN or Inf)")
        if len(iq) == 0:
            raise ValueError("Input IQ array must contain at least one sample")
        iq_complex = np.asarray(iq, dtype=np.complex64)
    elif not isinstance(iq, np.ndarray):
        raise TypeError(f"Input must be a NumPy ndarray, got {type(iq).__name__}")
    else:
        raise ValueError(f"Input array must be 1D complex or 2D canonical IQ [2, N], got shape {iq.shape}")

    # Estimate PSD via Welch's method (or reuse precomputed PSD)
    if psd is not None and len(psd) == 2:
        freqs, psd_dbhz = psd
    else:
        freqs, psd_dbhz = estimate_psd(iq_complex, sample_rate, nperseg=nperseg)

    if len(freqs) == 0:
        return []

    # Estimate noise floor
    nf_db = noise_floor_db(psd_dbhz, percentile=10.0)
    threshold_level = nf_db + threshold_db

    # Create binary occupancy mask
    mask = psd_dbhz >= threshold_level

    # Merge small spectral gaps
    merged_mask = merge_spectral_gaps(mask, merge_gap_bins=merge_gap_bins)

    # Find contiguous occupied regions
    contiguous = find_contiguous_regions(merged_mask)

    regions: list[SignalRegion] = []
    df = abs(freqs[1] - freqs[0]) if len(freqs) > 1 else sample_rate / nperseg

    for start_idx, end_idx in contiguous:
        bin_count = end_idx - start_idx + 1
        if bin_count >= min_bins:
            f1 = float(freqs[start_idx])
            f2 = float(freqs[end_idx])

            start_freq = min(f1, f2)
            end_freq = max(f1, f2)

            center_freq = (start_freq + end_freq) / 2.0
            bw = float(bin_count * df)

            region_psd = psd_dbhz[start_idx : end_idx + 1]
            peak_pwr = float(np.max(region_psd))
            avg_pwr = float(np.mean(region_psd))
            snr_est = float(avg_pwr - nf_db)

            regions.append(
                SignalRegion(
                    start_frequency=start_freq,
                    end_frequency=end_freq,
                    center_frequency=center_freq,
                    bandwidth=bw,
                    peak_power_db=peak_pwr,
                    average_power_db=avg_pwr,
                    snr_estimate_db=snr_est,
                )
            )

    return regions

