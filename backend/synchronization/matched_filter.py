"""
Matched filtering using a Root Raised Cosine (RRC) filter.

Reuses ml.generators.pulse_shaping.rrc.design_rrc_filter — the same
filter coefficients used during synthetic signal generation — so the
backend receives with the exact matched pair of the transmit filter.
"""
from __future__ import annotations

import numpy as np

from ml.generators.pulse_shaping.rrc import design_rrc_filter  # type: ignore[import]


def apply_matched_filter(
    iq: np.ndarray,
    samples_per_symbol: int,
    rolloff: float = 0.35,
    filter_span_symbols: int = 8,
) -> np.ndarray:
    """
    Apply an RRC matched filter to a complex IQ signal.

    The filter is designed via the same design_rrc_filter() used during
    waveform generation, guaranteeing zero ISI at the sampling instants
    (under ideal channel conditions).

    Args:
        iq:                  1-D complex IQ array.
        samples_per_symbol:  Oversampling factor (samples per symbol).
        rolloff:             RRC roll-off factor β ∈ [0, 1].
        filter_span_symbols: Filter length in symbols (impulse response span).

    Returns:
        Filtered complex IQ array (same length as input — 'same' convolution).
    """
    iq = np.asarray(iq, dtype=np.complex64)

    taps = design_rrc_filter(
        samples_per_symbol=int(samples_per_symbol),
        rolloff=float(rolloff),
        filter_span_symbols=int(filter_span_symbols),
    )

    # Convolve real and imaginary parts separately (avoids complex convolution
    # issues with older scipy/numpy builds) then recombine.
    filtered_i = np.convolve(iq.real, taps, mode="same")
    filtered_q = np.convolve(iq.imag, taps, mode="same")

    return (filtered_i + 1j * filtered_q).astype(np.complex64)


def downsample(iq: np.ndarray, samples_per_symbol: int, timing_offset: int = 0) -> np.ndarray:
    """
    Downsample a matched-filtered signal to one sample per symbol.

    Args:
        iq:                  1-D complex IQ array (at sps samples/symbol).
        samples_per_symbol:  Number of samples per symbol.
        timing_offset:       Sample offset for the first symbol (0 ≤ offset < sps).

    Returns:
        Symbol-rate complex array.
    """
    offset = int(timing_offset) % int(samples_per_symbol)
    return iq[offset::samples_per_symbol]
