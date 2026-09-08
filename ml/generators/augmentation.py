"""
Channel and RF Impairment Augmentation for Robust Signal Training.

Applies physically bounded domain randomization across channel and transceiver
impairments to reduce the synthetic-to-real domain gap.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

from ml.generators.channel.multipath import RayleighMultipathChannel


@dataclass(frozen=True)
class AugmentationConfig:
    """
    Physically plausible configuration parameters for RF channel augmentation.
    Ranges are bounded based on measured transceiver characteristics.
    """
    # AWGN SNR range (dB) - matches external dataset (20-30 dB) and typical conditions
    enable_awgn: bool = True
    snr_min: float = 12.0
    snr_max: float = 30.0

    # Carrier frequency offset: delta_f / fs (up to +/- 2% of sample rate)
    enable_frequency_offset: bool = True
    max_freq_offset_ratio: float = 0.02

    # Carrier phase offset: uniform [0, 2*pi)
    enable_phase_offset: bool = True

    # Oscillator phase noise (Wiener process step std in radians)
    enable_phase_noise: bool = True
    max_phase_noise_std: float = 0.015

    # Transceiver IQ imbalance
    enable_iq_imbalance: bool = True
    max_amp_imbalance_db: float = 0.5   # +/- 0.5 dB amplitude mismatch
    max_phase_imbalance_deg: float = 3.0 # +/- 3 degrees quadrature skew

    # DC offset as ratio of signal RMS (LO leakage / ADC bias)
    enable_dc_offset: bool = True
    max_dc_offset_ratio: float = 0.02

    # Multipath Rayleigh fading
    enable_multipath: bool = True
    min_multipath_taps: int = 2
    max_multipath_taps: int = 4
    multipath_decay: float = 1.5

    # Amplitude scaling / fading (shadowing / AGC variation)
    enable_amplitude_scaling: bool = True
    scale_min: float = 0.85
    scale_max: float = 1.20

    # Random seed
    random_seed: Optional[int] = None


class DomainAugmentor:
    """
    Applies bounded, reproducible RF domain augmentations to signal samples of shape [2, N].
    """
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config if config is not None else AugmentationConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

    def augment_signal(
        self,
        samples: np.ndarray,
        sample_rate: float = 800000.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Augments a single signal of shape [2, N].
        
        Args:
            samples: Input IQ array of shape [2, N] (float32).
            sample_rate: Sampling rate in Hz.
            rng: Optional explicit NumPy random generator.
            
        Returns:
            Tuple of (augmented_samples [2, N], metadata_dict).
        """
        if not isinstance(samples, np.ndarray):
            raise TypeError(f"samples must be a numpy ndarray, got {type(samples)}")
        if samples.ndim != 2 or samples.shape[0] != 2:
            raise ValueError(f"samples must have shape [2, N], got {samples.shape}")

        active_rng = rng if rng is not None else self._rng
        meta: Dict[str, Any] = {}
        N = samples.shape[1]
        z = samples[0].astype(np.float64) + 1j * samples[1].astype(np.float64)

        # 1. Amplitude scaling (slow fading / AGC uncertainty)
        if self.config.enable_amplitude_scaling:
            scale = float(active_rng.uniform(self.config.scale_min, self.config.scale_max))
            z = z * scale
            meta["amplitude_scale"] = scale

        # 2. Multipath frequency-selective fading
        if self.config.enable_multipath:
            num_taps = int(active_rng.integers(
                self.config.min_multipath_taps,
                self.config.max_multipath_taps + 1
            ))
            mp_channel = RayleighMultipathChannel(
                num_taps=num_taps,
                decay_rate=self.config.multipath_decay
            )
            taps = mp_channel.generate_taps(rng=active_rng)
            z = np.convolve(z, taps, mode="same")
            meta["multipath_num_taps"] = num_taps
            meta["multipath_taps"] = [float(np.abs(t)) for t in taps]

        # 3. Transceiver IQ Imbalance (Amplitude & Phase)
        if self.config.enable_iq_imbalance:
            amp_imb_db = float(active_rng.uniform(
                -self.config.max_amp_imbalance_db,
                self.config.max_amp_imbalance_db
            ))
            # Linear amplitude imbalance factor A in [-0.05, +0.05]
            A = (10.0 ** (amp_imb_db / 20.0) - 1.0) / (10.0 ** (amp_imb_db / 20.0) + 1.0)
            phase_imb_deg = float(active_rng.uniform(
                -self.config.max_phase_imbalance_deg,
                self.config.max_phase_imbalance_deg
            ))
            theta = float(np.deg2rad(phase_imb_deg))
            
            # 2x2 imbalance matrix
            a11 = (1.0 + A) * np.cos(theta / 2.0)
            a12 = -(1.0 + A) * np.sin(theta / 2.0)
            a21 = -(1.0 - A) * np.sin(theta / 2.0)
            a22 = (1.0 - A) * np.cos(theta / 2.0)
            
            i_in = np.real(z)
            q_in = np.imag(z)
            i_out = a11 * i_in + a12 * q_in
            q_out = a21 * i_in + a22 * q_in
            z = i_out + 1j * q_out
            
            meta["iq_amp_imbalance_db"] = amp_imb_db
            meta["iq_phase_imbalance_deg"] = phase_imb_deg

        # 4. Carrier Frequency & Phase Offset
        n = np.arange(N, dtype=np.float64)
        total_phase = np.zeros(N, dtype=np.float64)
        
        if self.config.enable_frequency_offset:
            max_fo = self.config.max_freq_offset_ratio * sample_rate
            fo = float(active_rng.uniform(-max_fo, max_fo))
            total_phase += 2.0 * np.pi * fo * n / sample_rate
            meta["frequency_offset_hz"] = fo

        if self.config.enable_phase_offset:
            po = float(active_rng.uniform(-np.pi, np.pi))
            total_phase += po
            meta["phase_offset_rad"] = po

        # 5. Oscillator Phase Noise (Wiener process random walk)
        if self.config.enable_phase_noise and self.config.max_phase_noise_std > 0:
            pn_std = float(active_rng.uniform(0.001, self.config.max_phase_noise_std))
            dphi = active_rng.normal(0.0, pn_std, size=N)
            total_phase += np.cumsum(dphi)
            meta["phase_noise_std"] = pn_std

        if np.any(total_phase != 0.0):
            z = z * np.exp(1j * total_phase)

        # 6. DC Offset
        if self.config.enable_dc_offset:
            rms = float(np.sqrt(np.mean(np.abs(z)**2)))
            max_dc = self.config.max_dc_offset_ratio * (rms + 1e-9)
            dc_i = float(active_rng.uniform(-max_dc, max_dc))
            dc_q = float(active_rng.uniform(-max_dc, max_dc))
            z = (np.real(z) + dc_i) + 1j * (np.imag(z) + dc_q)
            meta["dc_offset_i"] = dc_i
            meta["dc_offset_q"] = dc_q

        # 7. AWGN Noise Injection
        if self.config.enable_awgn:
            snr_db = float(active_rng.uniform(self.config.snr_min, self.config.snr_max))
            sig_power = float(np.mean(np.abs(z)**2))
            snr_linear = 10.0 ** (snr_db / 10.0)
            noise_power = sig_power / (snr_linear + 1e-12)
            noise_sigma = np.sqrt(noise_power / 2.0)
            noise_r = active_rng.normal(0.0, noise_sigma, size=N)
            noise_i = active_rng.normal(0.0, noise_sigma, size=N)
            z = z + (noise_r + 1j * noise_i)
            meta["snr_db"] = snr_db

        out = np.empty((2, N), dtype=np.float32)
        out[0] = np.real(z).astype(np.float32)
        out[1] = np.imag(z).astype(np.float32)
        return out, meta

    def augment_batch(
        self,
        batch_samples: np.ndarray,
        sample_rate: float = 800000.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Augments a batch of samples of shape [B, 2, N].
        """
        if not isinstance(batch_samples, np.ndarray) or batch_samples.ndim != 3 or batch_samples.shape[1] != 2:
            raise ValueError(f"Expected batch of shape [B, 2, N], got {batch_samples.shape}")

        active_rng = rng if rng is not None else self._rng
        B, C, N = batch_samples.shape
        out = np.empty((B, C, N), dtype=np.float32)

        for i in range(B):
            out[i], _ = self.augment_signal(batch_samples[i], sample_rate, rng=active_rng)

        return out
