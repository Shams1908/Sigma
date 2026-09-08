"""
Deterministic Signal Representation Transforms for RF IQ Signals.

This module provides transformations between the canonical [2, N] IQ representation
and candidate richer representations for machine learning models.

Note on Representations:
- RAW_IQ [I, Q]: Baseline representation (2 channels). Preserves raw complex baseband.
- IQ_AMPLITUDE [I, Q, |z|]: Experimentally motivated representation (3 channels).
  Appends the instantaneous envelope |z| = sqrt(I^2 + Q^2). Evaluates whether explicit
  magnitude dynamics help convolutional kernels separate constant-envelope from
  variable-envelope modulations without phase wrap discontinuities.
  Treated as an empirical candidate, not an a priori proven improvement.
- AMPLITUDE_PHASE [|z|, dphi/dt]: Polar representation with instantaneous frequency.
- IQ_AMP_PHASE [I, Q, |z|, dphi/dt]: 4-channel hybrid representation.
"""
from enum import Enum
from typing import Union
import numpy as np


class RepresentationType(str, Enum):
    RAW_IQ = "RAW_IQ"
    IQ_AMPLITUDE = "IQ_AMPLITUDE"
    AMPLITUDE_PHASE = "AMPLITUDE_PHASE"
    IQ_AMP_PHASE = "IQ_AMP_PHASE"


_CHANNEL_COUNTS = {
    RepresentationType.RAW_IQ: 2,
    RepresentationType.IQ_AMPLITUDE: 3,
    RepresentationType.AMPLITUDE_PHASE: 2,
    RepresentationType.IQ_AMP_PHASE: 4,
}


def get_representation_channels(rep_type: Union[RepresentationType, str]) -> int:
    """Returns the expected number of channels for a given representation."""
    if isinstance(rep_type, str):
        try:
            rep_type = RepresentationType(rep_type)
        except ValueError:
            raise ValueError(
                f"Unknown representation type: '{rep_type}'. "
                f"Must be one of: {[t.value for t in RepresentationType]}"
            )
    return _CHANNEL_COUNTS[rep_type]


def compute_representation(
    iq: np.ndarray,
    rep_type: Union[RepresentationType, str] = RepresentationType.IQ_AMPLITUDE,
    normalize_amplitude: bool = False,
) -> np.ndarray:
    """
    Transforms an IQ signal array into the specified representation.
    
    Supports:
        - Single signal of shape [2, N] -> returns [C, N]
        - Batch of signals of shape [B, 2, N] -> returns [B, C, N]
        
    Args:
        iq: NumPy array of shape [2, N] or [B, 2, N], dtype float32 or float64.
            iq[..., 0, :] is In-phase (I) and iq[..., 1, :] is Quadrature (Q).
        rep_type: Target RepresentationType enum or string.
        normalize_amplitude: If True, normalizes the amplitude channel by its RMS.
        
    Returns:
        NumPy array of shape [C, N] or [B, C, N] with float32 dtype.
        
    Raises:
        TypeError: If input is not a numpy array.
        ValueError: If input dimensions are invalid or contain NaN/Inf.
    """
    if not isinstance(iq, np.ndarray):
        raise TypeError(f"Input must be a numpy ndarray, got {type(iq)}")
        
    if np.isnan(iq).any() or np.isinf(iq).any():
        raise ValueError("Input IQ contains NaN or Infinite values.")

    if isinstance(rep_type, str):
        try:
            rep_type = RepresentationType(rep_type)
        except ValueError:
            raise ValueError(
                f"Unknown representation type: '{rep_type}'. "
                f"Must be one of: {[t.value for t in RepresentationType]}"
            )

    is_batch = (iq.ndim == 3)
    if iq.ndim == 2:
        if iq.shape[0] != 2:
            raise ValueError(f"Single signal must have shape [2, N], got {iq.shape}")
        arr = np.expand_dims(iq, axis=0)  # [1, 2, N]
    elif iq.ndim == 3:
        if iq.shape[1] != 2:
            raise ValueError(f"Batch signal must have shape [B, 2, N], got {iq.shape}")
        arr = iq
    else:
        raise ValueError(f"Input array must have 2 or 3 dimensions, got {iq.ndim}")

    I_chan = arr[:, 0, :]  # [B, N]
    Q_chan = arr[:, 1, :]  # [B, N]

    if rep_type == RepresentationType.RAW_IQ:
        out = arr.astype(np.float32)

    elif rep_type == RepresentationType.IQ_AMPLITUDE:
        # Instantaneous envelope: sqrt(I^2 + Q^2)
        # Avoid zero underflow with small epsilon
        amp = np.sqrt(I_chan**2 + Q_chan**2)
        if normalize_amplitude:
            rms = np.sqrt(np.mean(amp**2, axis=-1, keepdims=True)) + 1e-12
            amp = amp / rms
            
        out = np.stack([I_chan, Q_chan, amp], axis=1).astype(np.float32)

    elif rep_type == RepresentationType.AMPLITUDE_PHASE:
        # Polar representation: amplitude and instantaneous frequency (phase difference)
        amp = np.sqrt(I_chan**2 + Q_chan**2)
        if normalize_amplitude:
            rms = np.sqrt(np.mean(amp**2, axis=-1, keepdims=True)) + 1e-12
            amp = amp / rms
            
        # Instantaneous phase: arctan2(Q, I)
        phase = np.arctan2(Q_chan, I_chan)
        # Phase difference (derivative), normalized by pi
        dphi = np.diff(phase, axis=-1, prepend=phase[:, :1])
        # Wrap phase difference to [-pi, pi]
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        dphi_norm = dphi / np.pi
        
        out = np.stack([amp, dphi_norm], axis=1).astype(np.float32)

    elif rep_type == RepresentationType.IQ_AMP_PHASE:
        amp = np.sqrt(I_chan**2 + Q_chan**2)
        if normalize_amplitude:
            rms = np.sqrt(np.mean(amp**2, axis=-1, keepdims=True)) + 1e-12
            amp = amp / rms
            
        phase = np.arctan2(Q_chan, I_chan)
        dphi = np.diff(phase, axis=-1, prepend=phase[:, :1])
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        dphi_norm = dphi / np.pi
        
        out = np.stack([I_chan, Q_chan, amp, dphi_norm], axis=1).astype(np.float32)

    else:
        raise NotImplementedError(f"Representation {rep_type} is not implemented.")

    if not is_batch:
        return out[0]
    return out
