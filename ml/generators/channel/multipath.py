"""
Rayleigh multipath fading channel model for ML training data augmentation.

This is a training/data-augmentation component.
It is NOT part of the runtime DSP analysis pipeline.
It is NOT applied to uploaded signals during normal API inference.

Model:
    The channel has ``num_taps`` complex-valued taps drawn from independent
    complex Gaussian distributions (Rayleigh fading).  The power-delay
    profile decays exponentially with per-tap power:

        p[k] = exp(-k / decay_factor),   k = 0, 1, …, num_taps-1

    The tap coefficients are normalized so that the total channel energy
    (sum of |h[k]|²) equals 1 in expectation, preserving the average
    signal power after convolution.

Input / Output:
    - Accepts a 1-D complex64 waveform.
    - Returns a 1-D complex64 waveform of the same length.
      The output is trimmed to match the input length after convolution.

Determinism:
    Fully deterministic when given a seeded numpy RNG
    (np.random.default_rng or equivalent).
"""
from __future__ import annotations

import numpy as np


class RayleighMultipathChannel:
    """
    Rayleigh multipath fading channel.

    Args:
        num_taps:     Number of multipath delay taps (≥ 1).
        decay_factor: Exponential power-delay profile decay constant.
                      Larger values → slower power roll-off.
                      Must be > 0.
    """

    def __init__(self, num_taps: int = 6, decay_factor: float = 1.0) -> None:
        if num_taps < 1:
            raise ValueError(f"num_taps must be ≥ 1, got {num_taps}")
        if decay_factor <= 0:
            raise ValueError(f"decay_factor must be > 0, got {decay_factor}")
        self.num_taps = int(num_taps)
        self.decay_factor = float(decay_factor)

    def apply(
        self,
        waveform: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Apply the Rayleigh multipath channel to a complex baseband waveform.

        Args:
            waveform: 1-D complex64 array.
            rng:      Seeded numpy random generator for deterministic operation.

        Returns:
            1-D complex64 array of the same length as the input.

        Raises:
            TypeError:  If waveform is not a numpy array.
            ValueError: If waveform is not 1-D or is empty.
        """
        if not isinstance(waveform, np.ndarray):
            raise TypeError(
                f"waveform must be a numpy ndarray, got {type(waveform).__name__}"
            )
        if waveform.ndim != 1:
            raise ValueError(
                f"waveform must be 1-D, got shape {waveform.shape}"
            )
        if len(waveform) == 0:
            raise ValueError("waveform must not be empty")

        # ── Generate complex Rayleigh tap coefficients ──────────────────────
        # Exponential power-delay profile: p[k] = exp(-k / decay_factor)
        k = np.arange(self.num_taps, dtype=np.float64)
        tap_power = np.exp(-k / self.decay_factor)

        # Complex Gaussian taps: h[k] ~ CN(0, p[k])
        # Each component ~ N(0, sqrt(p[k]/2))
        tap_std = np.sqrt(tap_power / 2.0)
        h_real = rng.standard_normal(self.num_taps) * tap_std
        h_imag = rng.standard_normal(self.num_taps) * tap_std
        h = (h_real + 1j * h_imag).astype(np.complex64)

        # ── Normalize to unit channel energy ────────────────────────────────
        total_energy = float(np.sum(np.abs(h) ** 2))
        if total_energy > 0:
            h = h / np.sqrt(total_energy)

        # ── Convolve and trim to original length ────────────────────────────
        waveform_c = np.asarray(waveform, dtype=np.complex64)
        output_full = np.convolve(waveform_c, h, mode="full")
        output = output_full[: len(waveform_c)].astype(np.complex64)

        # Guard: ensure finite output
        if not np.all(np.isfinite(output)):
            raise ValueError(
                "RayleighMultipathChannel produced non-finite output. "
                "Check that the input waveform is finite."
            )

        return output
