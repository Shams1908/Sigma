"""
M8: Synthetic Parameter Estimation Dataset Generator and Splitter.

Generates IQ waveforms with randomized ground-truth symbol rates, samples-per-symbol,
and SNR levels across multiple modulations and channel conditions.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import os
import numpy as np


@dataclass
class ParameterDatasetSplit:
    """Container for parameter estimation dataset splits."""
    X: np.ndarray             # Shape: [N, 2, 128] float32
    sps: np.ndarray           # Shape: [N] float32 (samples per symbol)
    symbol_rate: np.ndarray   # Shape: [N] float32 (Baud)
    snr_db: np.ndarray        # Shape: [N] float32 (dB)
    sample_rate: np.ndarray   # Shape: [N] float32 (Hz)
    modulations: List[str]    # List of modulation names
    provenance: str           # E.g. "M8-SYNTHETIC-TRAIN"
    is_evaluation_set: bool = False

    def __post_init__(self):
        N = len(self.X)
        if not (len(self.sps) == len(self.symbol_rate) == len(self.snr_db) == len(self.sample_rate) == N):
            raise ValueError("Dimension mismatch in ParameterDatasetSplit arrays.")


def synthesize_parameter_sample(
    modulation: str,
    sps: float,
    snr_db: float,
    sample_rate: float,
    length: int = 128,
    rng: Optional[np.random.Generator] = None,
    apply_multipath: bool = False,
    freq_offset_hz: float = 0.0,
) -> np.ndarray:
    """
    Synthesizes a single 128-sample IQ window with precise symbol rate and SNR.
    
    Args:
        modulation: 'BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', etc.
        sps: Samples per symbol (sample_rate / symbol_rate).
        snr_db: Target SNR in dB.
        sample_rate: Sampling frequency in Hz.
        length: Window length (default 128).
        rng: NumPy random generator.
        apply_multipath: Whether to apply frequency-selective channel taps.
        freq_offset_hz: Carrier frequency offset in Hz.
        
    Returns:
        np.ndarray of shape [2, length], float32.
    """
    if rng is None:
        rng = np.random.default_rng()

    num_symbols = int(np.ceil(length / sps)) + 16
    t = np.arange(length) / sample_rate

    # Generate symbols according to modulation
    if modulation == "BPSK":
        syms_i = rng.choice([-1.0, 1.0], size=num_symbols)
        syms_q = np.zeros(num_symbols)
    elif modulation == "QPSK":
        syms_i = rng.choice([-1.0, 1.0], size=num_symbols) / np.sqrt(2)
        syms_q = rng.choice([-1.0, 1.0], size=num_symbols) / np.sqrt(2)
    elif modulation == "8PSK":
        angles = rng.choice(np.arange(8) * (2 * np.pi / 8), size=num_symbols)
        syms_i = np.cos(angles)
        syms_q = np.sin(angles)
    elif modulation == "QAM16":
        levels = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10)
        syms_i = rng.choice(levels, size=num_symbols)
        syms_q = rng.choice(levels, size=num_symbols)
    else: # Default 4-QAM / QPSK
        syms_i = rng.choice([-1.0, 1.0], size=num_symbols) / np.sqrt(2)
        syms_q = rng.choice([-1.0, 1.0], size=num_symbols) / np.sqrt(2)

    # Pulse shaping / sample interpolation
    sym_times = np.arange(num_symbols) * sps
    sample_indices = np.arange(length)
    
    # Nearest / rectangular symbol hold followed by low-pass smoothing
    symbol_idx_per_sample = np.clip(np.floor(sample_indices / sps).astype(int), 0, num_symbols - 1)
    I_bb = syms_i[symbol_idx_per_sample]
    Q_bb = syms_q[symbol_idx_per_sample]

    # Carrier frequency offset
    if freq_offset_hz != 0.0:
        cfo_phase = 2.0 * np.pi * freq_offset_hz * t
        z_cfo = (I_bb + 1j * Q_bb) * np.exp(1j * cfo_phase)
        I_sig = np.real(z_cfo)
        Q_sig = np.imag(z_cfo)
    else:
        I_sig = I_bb
        Q_sig = Q_bb

    # Multipath fading
    if apply_multipath:
        taps = np.array([1.0, 0.3 * np.exp(1j * rng.uniform(0, 2 * np.pi)), 0.15 * np.exp(1j * rng.uniform(0, 2 * np.pi))])
        taps = taps / np.sqrt(np.sum(np.abs(taps)**2))
        z_faded = np.convolve(I_sig + 1j * Q_sig, taps, mode="same")
        I_sig = np.real(z_faded)
        Q_sig = np.imag(z_faded)

    # Add AWGN calibrated to target SNR
    sig_power = float(np.mean(I_sig**2 + Q_sig**2))
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / max(1e-12, snr_linear)
    noise_sigma = np.sqrt(noise_power / 2.0)

    I_noisy = I_sig + rng.normal(0, noise_sigma, size=length)
    Q_noisy = Q_sig + rng.normal(0, noise_sigma, size=length)

    sample = np.stack([I_noisy, Q_noisy], axis=0).astype(np.float32)
    return sample


def generate_parameter_dataset(
    num_samples: int = 2000,
    sample_rate: float = 800000.0,
    random_seed: int = 42,
    provenance: str = "M8-PARAMETER-DATASET",
    is_evaluation_set: bool = False,
    output_path: Optional[str] = None,
) -> ParameterDatasetSplit:
    """
    Generates a deterministic synthetic parameter estimation dataset.
    
    Ranges:
      symbol_rate in [10 kHz, 200 kHz]
      sps = sample_rate / symbol_rate in [4.0, 80.0]
      snr_db in [-10.0 dB, +30.0 dB]
    """
    rng = np.random.default_rng(random_seed)
    mods = ["BPSK", "QPSK", "8PSK", "QAM16"]

    samples_list = []
    sps_list = []
    sym_rate_list = []
    snr_list = []
    mod_list = []
    sr_list = []

    for i in range(num_samples):
        mod = rng.choice(mods)
        # Sample symbol rate uniformly in [10k, 200k]
        sym_rate = float(rng.uniform(12500.0, 200000.0))
        sps = float(sample_rate / sym_rate)
        snr = float(rng.uniform(-10.0, 30.0))
        apply_multi = (rng.uniform() > 0.5)
        cfo = float(rng.uniform(-5000.0, 5000.0))

        iq_sample = synthesize_parameter_sample(
            modulation=mod,
            sps=sps,
            snr_db=snr,
            sample_rate=sample_rate,
            length=128,
            rng=rng,
            apply_multipath=apply_multi,
            freq_offset_hz=cfo,
        )

        samples_list.append(iq_sample)
        sps_list.append(sps)
        sym_rate_list.append(sym_rate)
        snr_list.append(snr)
        mod_list.append(mod)
        sr_list.append(sample_rate)

    X_all = np.array(samples_list, dtype=np.float32)
    sps_all = np.array(sps_list, dtype=np.float32)
    sym_rate_all = np.array(sym_rate_list, dtype=np.float32)
    snr_all = np.array(snr_list, dtype=np.float32)
    sr_all = np.array(sr_list, dtype=np.float32)

    split = ParameterDatasetSplit(
        X=X_all,
        sps=sps_all,
        symbol_rate=sym_rate_all,
        snr_db=snr_all,
        sample_rate=sr_all,
        modulations=mod_list,
        provenance=provenance,
        is_evaluation_set=is_evaluation_set,
    )

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(
            output_path,
            X=X_all,
            sps=sps_all,
            symbol_rate=sym_rate_all,
            snr_db=snr_all,
            sample_rate=sr_all,
            modulations=np.array(mod_list, dtype=object),
            provenance=provenance,
            random_seed=random_seed,
        )

    return split
