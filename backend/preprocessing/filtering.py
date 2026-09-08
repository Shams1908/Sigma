"""
Signal filtering and extraction module for canonical IQ signals.

Provides time-domain and frequency-domain filtering utilities (low-pass, band-pass)
and signal isolation/extraction for detected SignalRegion objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, filtfilt, lfilter  # type: ignore[import]

from preprocessing.normalize import validate_canonical_iq
from dsp.fft import canonical_to_complex
from dsp.carrier import remove_carrier_offset
from dsp.detection import SignalRegion


@dataclass
class ExtractedSignal:
    """
    Extracted signal region shifted to baseband (0 Hz) and filtered.
    """
    iq: np.ndarray
    sample_rate: float
    original_region: SignalRegion


def lowpass_filter(
    iq: np.ndarray,
    sample_rate: float,
    cutoff_hz: float,
    order: int = 5,
) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter to a canonical [2, N] or 1D complex IQ signal.

    Args:
        iq:          Canonical IQ array [2, N] or 1D complex array.
        sample_rate: Sampling rate in Hz.
        cutoff_hz:   Low-pass cutoff frequency in Hz (must be 0 < cutoff < sample_rate/2).
        order:       Filter order (default 5).

    Returns:
        Filtered IQ array matching input format.
    """
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")

    is_2d = isinstance(iq, np.ndarray) and iq.ndim == 2 and iq.shape[0] == 2

    if is_2d:
        validate_canonical_iq(iq)
        iq_complex = canonical_to_complex(iq)
    elif isinstance(iq, np.ndarray) and iq.ndim == 1:
        if not np.all(np.isfinite(iq)):
            raise ValueError("Input IQ array contains non-finite values (NaN or Inf)")
        if len(iq) == 0:
            raise ValueError("Input IQ array must contain at least one sample")
        iq_complex = np.asarray(iq, dtype=np.complex64)
    elif not isinstance(iq, np.ndarray):
        raise TypeError(f"Input must be a NumPy ndarray, got {type(iq).__name__}")
    else:
        raise ValueError(f"Input array must be 1D complex or 2D canonical IQ [2, N], got shape {iq.shape}")

    nyquist = sample_rate / 2.0
    safe_cutoff = float(np.clip(cutoff_hz, 1e-3, nyquist * 0.99))
    norm_cutoff = safe_cutoff / nyquist

    b, a = butter(order, norm_cutoff, btype="lowpass")

    # Handle short signals where padlen >= len(iq_complex)
    n_samples = len(iq_complex)
    padlen = 3 * max(len(a), len(b))

    if n_samples > padlen:
        i_filt = filtfilt(b, a, iq_complex.real)
        q_filt = filtfilt(b, a, iq_complex.imag)
    else:
        i_filt = lfilter(b, a, iq_complex.real)
        q_filt = lfilter(b, a, iq_complex.imag)

    complex_filtered = (i_filt + 1j * q_filt).astype(np.complex64)

    if is_2d:
        return np.stack([complex_filtered.real, complex_filtered.imag]).astype(np.float32)
    return complex_filtered


def extract_signal(
    iq: np.ndarray,
    sample_rate: float,
    region: SignalRegion,
    lowpass_margin: float = 1.2,
    filter_order: int = 5,
) -> ExtractedSignal:
    """
    Extract and isolate a signal region from a wideband IQ recording.

    Pipeline:
        1. Validate input IQ and region parameters.
        2. Convert canonical IQ to complex representation.
        3. Frequency-shift signal center to 0 Hz (baseband).
        4. Calculate low-pass cutoff based on region bandwidth * lowpass_margin / 2.
        5. Apply zero-phase Butterworth low-pass filter to I and Q components.
        6. Convert back to canonical IQ format [2, N].
        7. Return ExtractedSignal container.

    Args:
        iq:             Canonical IQ array [2, N] (or 1D complex array).
        sample_rate:    Sample rate in Hz.
        region:         Target SignalRegion instance.
        lowpass_margin: Multiplier applied to half-bandwidth for low-pass cutoff.
        filter_order:   Butterworth filter order.

    Returns:
        ExtractedSignal container with baseband canonical IQ data [2, N].
    """
    if not isinstance(region, SignalRegion):
        raise TypeError(f"region must be a SignalRegion instance, got {type(region).__name__}")

    if region.bandwidth <= 0:
        raise ValueError(f"SignalRegion bandwidth must be positive, got {region.bandwidth}")

    is_2d = isinstance(iq, np.ndarray) and iq.ndim == 2 and iq.shape[0] == 2
    if is_2d:
        validate_canonical_iq(iq)
    elif not isinstance(iq, np.ndarray):
        raise TypeError(f"Input must be a NumPy ndarray, got {type(iq).__name__}")

    # Step 1: Baseband Shift (shift region center_frequency down to 0 Hz)
    shifted = remove_carrier_offset(iq, sample_rate, region.center_frequency)

    # Step 2: Determine cutoff frequency
    half_bw = region.bandwidth / 2.0
    cutoff_hz = half_bw * lowpass_margin

    # Step 3: Low-pass filter baseband signal
    filtered = lowpass_filter(shifted, sample_rate, cutoff_hz=cutoff_hz, order=filter_order)

    # Step 4: Ensure output is canonical float32 [2, N]
    if isinstance(filtered, np.ndarray) and filtered.ndim == 1:
        canonical_out = np.stack([filtered.real, filtered.imag]).astype(np.float32)
    else:
        canonical_out = filtered.astype(np.float32)

    return ExtractedSignal(
        iq=canonical_out,
        sample_rate=sample_rate,
        original_region=region,
    )
