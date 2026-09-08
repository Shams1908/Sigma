"""
Signal representation transform functions.

All transforms are pure functions that accept canonical IQ arrays and return
float32 arrays in the representation-specific channel layout.

Input conventions:
    Single sample: ndarray of shape [2, N]
    Batch:         ndarray of shape [B, 2, N]

Output:
    Single sample: ndarray of shape [C, N]   (C = channel count)
    Batch:         ndarray of shape [B, C, N]

Contract:
    - Input must be float32 or will be cast to float32.
    - Input must be finite (no NaN or Inf).
    - The I and Q channels (rows 0 and 1) are passed through exactly for
      representations that include them (RAW_IQ, IQ_AMPLITUDE, IQ_AMP_PHASE).
    - Phase-difference is zero-padded at sample index 0 (first sample is 0).
    - Output dtype is always float32.
    - Both single-sample [2, N] and batch [B, 2, N] are supported.
    - ndim 1 or ≥ 4 inputs are rejected.
"""
from __future__ import annotations

import numpy as np

from ml.representations.definitions import Representation


def _validate(iq: np.ndarray) -> None:
    """Shared validation: type, shape, finite."""
    if not isinstance(iq, np.ndarray):
        raise TypeError(
            f"Input must be a numpy ndarray, got {type(iq).__name__}"
        )
    if iq.ndim not in (2, 3):
        raise ValueError(
            f"Input must be 2-D [2, N] or 3-D [B, 2, N], got shape {iq.shape}"
        )
    # Check the channel dimension (axis=0 for single, axis=1 for batch)
    ch_axis = 0 if iq.ndim == 2 else 1
    if iq.shape[ch_axis] != 2:
        raise ValueError(
            f"Channel dimension must be 2 (I and Q), got shape {iq.shape}"
        )
    if not np.all(np.isfinite(iq)):
        raise ValueError("Input IQ array contains non-finite values (NaN or Inf)")


def _instantaneous_amplitude(iq: np.ndarray) -> np.ndarray:
    """
    Compute |z| = sqrt(I² + Q²).
    Handles both [2, N] and [B, 2, N] inputs.
    Returns [N] or [B, N] float32 array.
    """
    if iq.ndim == 2:
        i_ch, q_ch = iq[0], iq[1]
    else:
        i_ch, q_ch = iq[:, 0, :], iq[:, 1, :]
    return np.sqrt(i_ch.astype(np.float32) ** 2 + q_ch.astype(np.float32) ** 2)


def _phase_difference(iq: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous phase difference Δφ[n] = angle(z[n] · conj(z[n-1])).
    Δφ[0] = 0.0 (zero-padded).
    Handles both [2, N] and [B, 2, N] inputs.
    Returns [N] or [B, N] float32 array.
    """
    if iq.ndim == 2:
        z = iq[0].astype(np.float32) + 1j * iq[1].astype(np.float32)
        phase_diff = np.angle(z[1:] * np.conj(z[:-1])).astype(np.float32)
        return np.concatenate([[0.0], phase_diff]).astype(np.float32)
    else:
        # Batch [B, 2, N]
        z = iq[:, 0, :].astype(np.float32) + 1j * iq[:, 1, :].astype(np.float32)
        # z shape: [B, N]
        phase_diff = np.angle(z[:, 1:] * np.conj(z[:, :-1])).astype(np.float32)
        zeros = np.zeros((z.shape[0], 1), dtype=np.float32)
        return np.concatenate([zeros, phase_diff], axis=1)


# ── Public transform functions ─────────────────────────────────────────────────

def to_raw_iq(iq: np.ndarray) -> np.ndarray:
    """
    RAW_IQ representation: [I, Q]  — 2 channels.
    Returns the input unchanged except for dtype cast to float32.
    """
    _validate(iq)
    return iq.astype(np.float32)


def to_iq_amplitude(iq: np.ndarray) -> np.ndarray:
    """
    IQ_AMPLITUDE representation: [I, Q, |z|]  — 3 channels.
    I and Q are the original channels; |z| is appended as channel 2.
    """
    _validate(iq)
    iq_f = iq.astype(np.float32)
    amp = _instantaneous_amplitude(iq_f)
    if iq.ndim == 2:
        return np.stack([iq_f[0], iq_f[1], amp], axis=0)          # [3, N]
    else:
        return np.stack([iq_f[:, 0, :], iq_f[:, 1, :], amp], axis=1)  # [B, 3, N]


def to_amplitude_phase(iq: np.ndarray) -> np.ndarray:
    """
    AMPLITUDE_PHASE representation: [|z|, Δφ]  — 2 channels.
    """
    _validate(iq)
    iq_f = iq.astype(np.float32)
    amp = _instantaneous_amplitude(iq_f)
    pdiff = _phase_difference(iq_f)
    if iq.ndim == 2:
        return np.stack([amp, pdiff], axis=0)                      # [2, N]
    else:
        return np.stack([amp, pdiff], axis=1)                      # [B, 2, N]


def to_iq_amp_phase(iq: np.ndarray) -> np.ndarray:
    """
    IQ_AMP_PHASE representation: [I, Q, |z|, Δφ]  — 4 channels.
    """
    _validate(iq)
    iq_f = iq.astype(np.float32)
    amp = _instantaneous_amplitude(iq_f)
    pdiff = _phase_difference(iq_f)
    if iq.ndim == 2:
        return np.stack([iq_f[0], iq_f[1], amp, pdiff], axis=0)   # [4, N]
    else:
        return np.stack(
            [iq_f[:, 0, :], iq_f[:, 1, :], amp, pdiff], axis=1
        )                                                           # [B, 4, N]


# ── Generic dispatcher ─────────────────────────────────────────────────────────

_DISPATCH: dict = {
    Representation.RAW_IQ:          to_raw_iq,
    Representation.IQ_AMPLITUDE:    to_iq_amplitude,
    Representation.AMPLITUDE_PHASE: to_amplitude_phase,
    Representation.IQ_AMP_PHASE:    to_iq_amp_phase,
}


def apply_representation(iq: np.ndarray, rep: Representation | str) -> np.ndarray:
    """
    Apply the named representation transform to a canonical IQ array.

    Args:
        iq:  Canonical float32 IQ array of shape [2, N] or [B, 2, N].
        rep: Representation enum value or its string name.

    Returns:
        float32 array of shape [C, N] or [B, C, N] where C = num_channels(rep).
    """
    if isinstance(rep, str):
        rep = Representation(rep)
    fn = _DISPATCH.get(rep)
    if fn is None:
        raise ValueError(
            f"Unsupported representation '{rep}'. "
            f"Supported: {list(_DISPATCH.keys())}"
        )
    return fn(iq)
