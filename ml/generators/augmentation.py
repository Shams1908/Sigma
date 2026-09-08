"""
Domain augmentation for ML training data robustness.

PURPOSE:
    This module provides bounded, stochastic augmentation of canonical IQ
    signals for use during ML model training and robustness experimentation.

CRITICAL — INFERENCE BOUNDARY:
    This augmentation is NOT applied to uploaded signals during runtime
    signal analysis inference.  It operates exclusively on training data.
    Do NOT insert DomainAugmentor into the production API inference path.

USAGE:
    augmentor = DomainAugmentor(seed=42)
    iq_aug = augmentor.augment(iq_2d, sample_rate)

    # Or with explicit control over which augmentations are enabled:
    augmentor = DomainAugmentor(
        seed=0,
        amplitude_scale_range=(0.8, 1.2),
        enable_multipath=True,
        enable_freq_offset=True,
        freq_offset_max_hz=2000.0,
        ...
    )

Input / Output:
    Input:  canonical [2, N] float32 IQ array
    Output: canonical [2, N] float32 IQ array (augmented)

Augmentation catalogue:
    1. Amplitude scaling         — multiplicative, uniform in log-scale
    2. Rayleigh multipath        — convolutive fading channel
    3. IQ imbalance              — amplitude + phase cross-coupling
    4. Frequency offset          — carrier phase rotation
    5. Phase offset              — constant phase rotation
    6. Oscillator phase noise    — random phase walk
    7. DC offset                 — I/Q bias addition
    8. AWGN                      — additive white Gaussian noise

Each augmentation can be individually enabled / disabled.
Augmentation parameters are bounded to prevent unrealistic distortions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class DomainAugmentor:
    """
    Bounded stochastic augmentor for ML training data.

    M6 defaults have all augmentations ON.  Individual switches can be
    set to False (or ranges set to None) to disable specific augmentations
    for controlled experiments.

    Args:
        seed: Base random seed for reproducibility.  The seed is advanced
              deterministically per augmentation step so that enabling /
              disabling independent augmentations does not disturb the
              randomness of others.

        amplitude_scale_range: (low, high) multiplicative amplitude scale
              sampled uniformly in [low, high].  Values close to 1.0 are
              mild; (0.5, 2.0) is a moderate range.  None disables.

        enable_multipath: If True, apply Rayleigh multipath fading.
        multipath_num_taps: Number of Rayleigh taps (≥ 1).
        multipath_decay_factor: Exponential PDP decay constant (> 0).

        enable_iq_imbalance: If True, apply IQ amplitude / phase imbalance.
        iq_amplitude_imbalance_max: Maximum amplitude imbalance in dB.
        iq_phase_imbalance_max_deg: Maximum phase imbalance in degrees.

        enable_freq_offset: If True, apply a random carrier frequency offset.
        freq_offset_max_hz: Maximum absolute frequency offset in Hz.

        enable_phase_offset: If True, apply a random constant phase offset.

        enable_phase_noise: If True, apply oscillator phase noise (random walk).
        phase_noise_std_deg: Standard deviation of per-sample phase increment
              (degrees).  Typical oscillator noise is 0.1–2 degrees.

        enable_dc_offset: If True, apply a small random DC offset.
        dc_offset_max: Maximum DC offset magnitude (linear, applied to I and Q
              independently).

        snr_db_range: (low, high) SNR range for AWGN augmentation in dB.
              None disables AWGN.  Only applied when both bounds are provided.
    """

    seed: int = 0

    # Amplitude scaling — M6 default: enabled, (0.8, 1.2) range
    amplitude_scale_range: Optional[Tuple[float, float]] = (0.8, 1.2)

    # Rayleigh multipath — M6 default: enabled
    enable_multipath: bool = True
    multipath_num_taps: int = 6
    multipath_decay_factor: float = 1.0

    # IQ imbalance — M6 default: enabled
    enable_iq_imbalance: bool = True
    iq_amplitude_imbalance_max: float = 1.0   # dB
    iq_phase_imbalance_max_deg: float = 5.0   # degrees

    # Frequency offset — M6 default: enabled
    enable_freq_offset: bool = True
    freq_offset_max_hz: float = 1000.0

    # Phase offset — M6 default: enabled
    enable_phase_offset: bool = True

    # Oscillator phase noise — M6 default: enabled
    enable_phase_noise: bool = True
    phase_noise_std_deg: float = 0.5

    # DC offset — M6 default: enabled
    enable_dc_offset: bool = True
    dc_offset_max: float = 0.05

    # AWGN — M6 default: enabled, (10.0, 30.0) dB range
    snr_db_range: Optional[Tuple[float, float]] = (10.0, 30.0)

    def augment(
        self,
        iq: np.ndarray,
        sample_rate: float,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Apply enabled augmentations to a canonical [2, N] IQ array.

        Args:
            iq:          Canonical float32 IQ array of shape [2, N].
            sample_rate: Sample rate in Hz (required for frequency offset).
            rng:         Optional pre-seeded RNG.  If None, one is created
                         from ``self.seed``.

        Returns:
            Augmented canonical [2, N] float32 array.

        Raises:
            TypeError:  If iq is not a numpy ndarray.
            ValueError: If iq is not [2, N] or contains non-finite values.
        """
        if not isinstance(iq, np.ndarray):
            raise TypeError(
                f"iq must be a numpy ndarray, got {type(iq).__name__}"
            )
        if iq.ndim != 2 or iq.shape[0] != 2:
            raise ValueError(
                f"iq must be [2, N] canonical IQ, got shape {iq.shape}"
            )
        if not np.all(np.isfinite(iq)):
            raise ValueError("iq contains non-finite values (NaN or Inf)")

        if rng is None:
            rng = np.random.default_rng(self.seed)

        # Convert to complex for most operations
        z = iq[0].astype(np.float32) + 1j * iq[1].astype(np.float32)
        n = len(z)

        # 1. Amplitude scaling
        if self.amplitude_scale_range is not None:
            lo, hi = self.amplitude_scale_range
            scale = float(rng.uniform(lo, hi))
            z = z * scale

        # 2. Rayleigh multipath
        if self.enable_multipath:
            from ml.generators.channel.multipath import RayleighMultipathChannel  # noqa: PLC0415
            ch = RayleighMultipathChannel(
                num_taps=self.multipath_num_taps,
                decay_factor=self.multipath_decay_factor,
            )
            z = ch.apply(z, rng)

        # 3. IQ imbalance
        if self.enable_iq_imbalance:
            amp_db = float(rng.uniform(
                -self.iq_amplitude_imbalance_max,
                self.iq_amplitude_imbalance_max,
            ))
            phase_rad = float(rng.uniform(
                -np.deg2rad(self.iq_phase_imbalance_max_deg),
                np.deg2rad(self.iq_phase_imbalance_max_deg),
            ))
            alpha = 10.0 ** (amp_db / 20.0)
            # IQ imbalance: I' = I, Q' = alpha * (I * sin(phi) + Q * cos(phi))
            i_ch = z.real.copy()
            q_ch = z.imag.copy()
            q_new = alpha * (i_ch * np.sin(phase_rad) + q_ch * np.cos(phase_rad))
            z = (i_ch + 1j * q_new).astype(np.complex64)

        # 4. Frequency offset
        if self.enable_freq_offset and sample_rate > 0:
            fo = float(rng.uniform(-self.freq_offset_max_hz, self.freq_offset_max_hz))
            t = np.arange(len(z), dtype=np.float64) / sample_rate
            z = (z * np.exp(1j * 2.0 * np.pi * fo * t)).astype(np.complex64)

        # 5. Phase offset
        if self.enable_phase_offset:
            phi = float(rng.uniform(-np.pi, np.pi))
            z = (z * np.exp(1j * phi)).astype(np.complex64)

        # 6. Oscillator phase noise (random walk)
        if self.enable_phase_noise and self.phase_noise_std_deg > 0:
            std_rad = np.deg2rad(self.phase_noise_std_deg)
            increments = rng.normal(0.0, std_rad, size=len(z)).astype(np.float64)
            phase_noise = np.cumsum(increments)
            z = (z * np.exp(1j * phase_noise)).astype(np.complex64)

        # 7. DC offset
        if self.enable_dc_offset:
            dc_i = float(rng.uniform(-self.dc_offset_max, self.dc_offset_max))
            dc_q = float(rng.uniform(-self.dc_offset_max, self.dc_offset_max))
            z = (z + (dc_i + 1j * dc_q)).astype(np.complex64)

        # 8. AWGN
        if self.snr_db_range is not None:
            lo_snr, hi_snr = self.snr_db_range
            snr_db = float(rng.uniform(lo_snr, hi_snr))
            p_signal = float(np.mean(np.abs(z) ** 2))
            if p_signal > 0:
                p_noise = p_signal / (10.0 ** (snr_db / 10.0))
                noise_std = np.sqrt(p_noise / 2.0)
                noise = (
                    rng.normal(0.0, noise_std, len(z))
                    + 1j * rng.normal(0.0, noise_std, len(z))
                ).astype(np.complex64)
                z = (z + noise).astype(np.complex64)

        # Pack back to canonical [2, N] float32
        out = np.stack([z.real, z.imag], axis=0).astype(np.float32)

        # Final guard
        if not np.all(np.isfinite(out)):
            raise ValueError(
                "DomainAugmentor produced non-finite output. "
                "Check augmentation parameter bounds."
            )

        return out
