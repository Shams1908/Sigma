"""
Matched filtering for RF signal recovery (Root-Raised Cosine, bypass).
"""
from typing import Optional, Union
import numpy as np

try:
    from ml.generators.pulse_shaping.rrc import design_rrc_filter
except ImportError:
    design_rrc_filter = None


def apply_matched_filter(
    signal: np.ndarray,
    filter_type: str = "rrc",
    rolloff: float = 0.35,
    span: int = 6,
    sps: int = 4,
) -> np.ndarray:
    """
    Applies matched filtering to input IQ waveform.

    Args:
        signal: 1D complex or 2D real [2, N] array.
        filter_type: 'rrc' (Root-Raised Cosine), 'rc', or 'none'.
        rolloff: Excess bandwidth factor (0.0 to 1.0).
        span: Filter span in symbols.
        sps: Samples per symbol.

    Returns:
        np.ndarray: Filtered signal preserving input shape and dtype.
    """
    if signal is None:
        raise ValueError("Input signal must not be None")

    ft = filter_type.strip().lower()
    if ft in ("none", "bypass", "identity"):
        return signal.copy()

    if ft not in ("rrc", "root_raised_cosine", "rc"):
        raise ValueError(f"Unsupported matched filter type: '{filter_type}'. Supported: 'rrc', 'none'.")

    if design_rrc_filter is not None:
        taps = design_rrc_filter(
            samples_per_symbol=int(sps),
            rolloff=float(rolloff),
            filter_span_symbols=int(span),
        )
    else:
        # Fallback simple symmetric moving average filter
        taps = np.ones(sps, dtype=np.float32) / float(sps)

    # Convolve based on shape
    is_2d_real = (signal.ndim == 2 and signal.shape[0] == 2 and not np.iscomplexobj(signal))
    if is_2d_real:
        filtered_i = np.convolve(signal[0], taps, mode="same")
        filtered_q = np.convolve(signal[1], taps, mode="same")
        return np.vstack([filtered_i, filtered_q]).astype(np.float32)
    elif np.iscomplexobj(signal):
        filtered = np.convolve(signal, taps, mode="same")
        return filtered.astype(signal.dtype)
    else:
        raise ValueError(f"Input signal must be 1D complex or 2D [2, N] real array, got shape {signal.shape}")
