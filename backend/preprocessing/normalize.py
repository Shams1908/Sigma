"""
Normalization and preprocessing module for canonical IQ signals.

Canonical IQ representation:
    NumPy ndarray with shape [2, N]
    iq[0] = I samples
    iq[1] = Q samples
    dtype = float32
"""

from __future__ import annotations

import numpy as np


def validate_canonical_iq(iq: np.ndarray) -> np.ndarray:
    """
    Validate that input is a canonical float32/numeric IQ array with shape [2, N].

    Responsibilities:
    - Ensure input is a NumPy ndarray.
    - Ensure input has exactly 2 dimensions.
    - Ensure shape[0] == 2.
    - Ensure there is at least one sample (shape[1] > 0).
    - Ensure all values are finite (no NaN, no Inf).
    - Raise TypeError or ValueError with clear messages.
    - Return the validated IQ array.
    """
    if not isinstance(iq, np.ndarray):
        raise TypeError(f"Input must be a NumPy ndarray, got {type(iq).__name__}")

    if iq.ndim != 2:
        raise ValueError(f"Input must have exactly 2 dimensions [2, N], got {iq.ndim} dimensions with shape {iq.shape}")

    if iq.shape[0] != 2:
        raise ValueError(f"First dimension of canonical IQ array must be 2 [2, N], got shape {iq.shape}")

    if iq.shape[1] == 0:
        raise ValueError("Input IQ array must contain at least one sample (N > 0), got 0 samples")

    if not np.all(np.isfinite(iq)):
        raise ValueError("Input IQ array contains non-finite values (NaN or Inf)")

    return iq


def remove_dc(iq: np.ndarray) -> np.ndarray:
    """
    Remove DC offset independently from I and Q channels.

    For:
        I = iq[0]
        Q = iq[1]
    Perform:
        I_clean = I - mean(I)
        Q_clean = Q - mean(Q)

    Preserves shape [2, N], returns float32 array, does not modify input in-place.
    """
    validated = validate_canonical_iq(iq)
    i = validated[0] - np.mean(validated[0])
    q = validated[1] - np.mean(validated[1])
    return np.stack([i, q], axis=0).astype(np.float32)


def normalize_power(iq: np.ndarray) -> np.ndarray:
    """
    Normalize complex IQ signal to unit average power mean(|x|^2) ≈ 1.0.

    Concept:
        x = I + jQ
        Power = mean(|x|^2) = mean(I^2 + Q^2)
        x_norm = x / sqrt(Power)

    Preserves shape [2, N], returns float32 array, handles zero-power signal safely,
    does not modify input in-place.
    """
    validated = validate_canonical_iq(iq)
    float_iq = validated.astype(np.float32)

    power = np.mean(float_iq[0] ** 2 + float_iq[1] ** 2)

    if power == 0 or not np.isfinite(power):
        return float_iq.copy()

    scale = np.sqrt(power)
    return (float_iq / scale).astype(np.float32)


def normalize_peak(iq: np.ndarray) -> np.ndarray:
    """
    Perform peak magnitude normalization (iq / max(abs(iq))).

    Optional utility. Preserves shape [2, N], returns float32 array,
    handles all-zero input safely, does not modify input in-place.
    """
    validated = validate_canonical_iq(iq)
    float_iq = validated.astype(np.float32)

    peak = np.max(np.abs(float_iq))

    if peak == 0 or not np.isfinite(peak):
        return float_iq.copy()

    return (float_iq / peak).astype(np.float32)


def normalize_signal(
    iq: np.ndarray,
    remove_dc_offset: bool = True,
    unit_power: bool = True,
) -> np.ndarray:
    """
    Main signal normalization pipeline.

    Pipeline:
        INPUT
          ↓
        validate_canonical_iq()
          ↓
        convert to float32
          ↓
        optional DC removal
          ↓
        optional unit-power normalization
          ↓
        OUTPUT

    Returns float32 canonical [2, N] array without mutating original input.
    """
    validated = validate_canonical_iq(iq)
    out = validated.astype(np.float32, copy=True)

    if remove_dc_offset:
        out = remove_dc(out)

    if unit_power:
        out = normalize_power(out)

    return out
