"""
External Real-World Validation Dataset Loader and Provenance Handler.

Dataset: "Real-World IQ Dataset for Automatic Radio Modulation Recognition under Multipath Channels"
Contract:
  - Modulations: BPSK (0), QPSK (1), QAM (2), GMSK (3), OFDM (4), NBFM (5), WBFM (6)
  - Channel conditions: clean (0), multipath (1)
  - SNRs: [20, 22, 24, 26, 28, 30] dB
  - Frame length: 1024 IQ samples
  - Canonical benchmark shape: X = (80000, 1024, 2), y_mod, y_chan, y_snr

IMPORTANT:
This dataset is strictly an evaluation/generalization resource.
Test labels must NEVER be contaminated into training loops.
"""
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

EXTERNAL_MODULATION_CLASSES: List[str] = [
    "BPSK",
    "QPSK",
    "QAM",
    "GMSK",
    "OFDM",
    "NBFM",
    "WBFM",
]

EXTERNAL_MOD_TO_IDX: Dict[str, int] = {
    label: idx for idx, label in enumerate(EXTERNAL_MODULATION_CLASSES)
}

EXTERNAL_IDX_TO_MOD: Dict[int, str] = {
    idx: label for idx, label in enumerate(EXTERNAL_MODULATION_CLASSES)
}

EXTERNAL_CHANNEL_CONDITIONS: List[str] = ["clean", "multipath"]
EXTERNAL_CHAN_TO_IDX: Dict[str, int] = {"clean": 0, "multipath": 1}
EXTERNAL_IDX_TO_CHAN: Dict[int, str] = {0: "clean", 1: "multipath"}

EXTERNAL_SNR_VALUES: List[int] = [20, 22, 24, 26, 28, 30]


@dataclass(frozen=True)
class RealWorldDatasetMetadata:
    """Metadata specification for the external real-world dataset."""
    dataset_name: str = "Real-World IQ Dataset for Automatic Radio Modulation Recognition under Multipath Channels"
    canonical_filename: str = "subset_test.h5"
    num_benchmark_samples: int = 80000
    frame_length: int = 1024
    iq_channels: int = 2
    classes: List[str] = field(default_factory=lambda: list(EXTERNAL_MODULATION_CLASSES))
    channel_conditions: List[str] = field(default_factory=lambda: list(EXTERNAL_CHANNEL_CONDITIONS))
    snr_levels: List[int] = field(default_factory=lambda: list(EXTERNAL_SNR_VALUES))


@dataclass
class ExternalDatasetSplit:
    """
    Container for external dataset samples with explicit provenance tracking.
    """
    X: np.ndarray             # Shape: [N, 1024, 2] or [N, 2, 1024]
    y_mod: np.ndarray         # Shape: [N] (integers 0..6)
    y_chan: np.ndarray        # Shape: [N] (0=clean, 1=multipath)
    y_snr: np.ndarray         # Shape: [N] (dB integer values)
    provenance: str           # E.g. "EXTERNAL-BENCHMARK-TEST" or "M6-DEV-SUBSET"
    is_evaluation_set: bool = True

    def __post_init__(self):
        if not isinstance(self.X, np.ndarray):
            raise TypeError("X must be a numpy ndarray")
        if not isinstance(self.y_mod, np.ndarray):
            raise TypeError("y_mod must be a numpy ndarray")
        if len(self.X) != len(self.y_mod) or len(self.X) != len(self.y_chan) or len(self.X) != len(self.y_snr):
            raise ValueError("Mismatched dimensions across X, y_mod, y_chan, y_snr")

    def assert_evaluation_isolation(self) -> None:
        """Enforces that this dataset cannot be accidentally treated as training data."""
        if not self.is_evaluation_set:
            raise ValueError(f"Provenance violation: {self.provenance} must be strictly an evaluation set.")


def slice_frames_to_windows(
    X_frames: np.ndarray,
    window_length: int = 128,
) -> np.ndarray:
    """
    Converts 1024-sample frames into non-overlapping canonical windows of shape [M, 2, window_length].
    
    Args:
        X_frames: NumPy array of shape [N, 1024, 2] or [N, 2, 1024].
        window_length: Target segment length (default 128).
        
    Returns:
        NumPy array of shape [N * (1024 // window_length), 2, window_length], float32.
    """
    if X_frames.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {X_frames.shape}")

    # Standardize to [N, 2, L]
    if X_frames.shape[2] == 2 and X_frames.shape[1] != 2:
        # [N, 1024, 2] -> [N, 2, 1024]
        X_canonical = np.transpose(X_frames, (0, 2, 1))
    elif X_frames.shape[1] == 2:
        X_canonical = X_frames
    else:
        raise ValueError(f"Neither axis 1 nor axis 2 has dimension 2 (IQ channels): {X_frames.shape}")

    N, C, total_len = X_canonical.shape
    num_windows = total_len // window_length
    if num_windows < 1:
        raise ValueError(f"Signal length {total_len} is shorter than window length {window_length}")

    # Truncate any remainder
    usable_len = num_windows * window_length
    X_trimmed = X_canonical[:, :, :usable_len]

    # Reshape to [N, 2, num_windows, window_length] -> [N * num_windows, 2, window_length]
    windows = X_trimmed.reshape(N, 2, num_windows, window_length)
    windows = np.transpose(windows, (0, 2, 1, 3))  # [N, num_windows, 2, window_length]
    windows = windows.reshape(N * num_windows, 2, window_length)

    return windows.astype(np.float32)


def generate_external_dev_subset(
    output_path: Optional[str] = "datasets/processed/external_dev_subset.npz",
    num_per_condition: int = 5,
    random_seed: int = 42,
) -> ExternalDatasetSplit:
    """
    Generates a deterministic, strictly isolated development subset conforming
    to the external dataset schema for testing, auditing, and development.
    
    Guarantees:
      - Shape: [N, 1024, 2], float32
      - All 7 modulations, 2 channel conditions, and 6 SNRs represented
      - Explicit provenance metadata: "M6-DEV-SUBSET"
      - Documented seed for zero test-set leakage
    """
    rng = np.random.default_rng(random_seed)
    
    samples_list = []
    y_mod_list = []
    y_chan_list = []
    y_snr_list = []

    frame_len = 1024
    for mod_idx, mod_name in enumerate(EXTERNAL_MODULATION_CLASSES):
        for chan_idx in [0, 1]:  # clean, multipath
            for snr in EXTERNAL_SNR_VALUES:
                for _ in range(num_per_condition):
                    # Synthesize deterministic base waveform according to class type
                    t = np.linspace(0, 1, frame_len, endpoint=False)
                    carrier = 10.0  # Normalized cycles
                    
                    if mod_name == "BPSK":
                        symbols = rng.choice([-1.0, 1.0], size=128)
                        bb = np.repeat(symbols, 8)
                        I = bb * np.cos(2 * np.pi * carrier * t)
                        Q = bb * np.sin(2 * np.pi * carrier * t)
                    elif mod_name == "QPSK":
                        sym_i = rng.choice([-1.0, 1.0], size=128) / np.sqrt(2)
                        sym_q = rng.choice([-1.0, 1.0], size=128) / np.sqrt(2)
                        I = np.repeat(sym_i, 8)
                        Q = np.repeat(sym_q, 8)
                    elif mod_name == "QAM":
                        levels = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10)
                        sym_i = rng.choice(levels, size=128)
                        sym_q = rng.choice(levels, size=128)
                        I = np.repeat(sym_i, 8)
                        Q = np.repeat(sym_q, 8)
                    elif mod_name == "GMSK":
                        bits = rng.choice([-1.0, 1.0], size=128)
                        phase = np.cumsum(np.repeat(bits, 8)) * (np.pi / 16.0)
                        I = np.cos(phase)
                        Q = np.sin(phase)
                    elif mod_name == "OFDM":
                        # Multi-carrier summation
                        I = np.zeros(frame_len)
                        Q = np.zeros(frame_len)
                        num_carriers = 16
                        for k in range(1, num_carriers + 1):
                            phase_k = rng.uniform(0, 2 * np.pi)
                            I += (1.0 / np.sqrt(num_carriers)) * np.cos(2 * np.pi * k * 2 * t + phase_k)
                            Q += (1.0 / np.sqrt(num_carriers)) * np.sin(2 * np.pi * k * 2 * t + phase_k)
                    elif mod_name == "NBFM":
                        # Narrowband FM: small deviation
                        msg = np.sin(2 * np.pi * 2 * t)
                        phase = 2 * np.pi * carrier * t + 0.5 * np.cumsum(msg) / frame_len
                        I = np.cos(phase)
                        Q = np.sin(phase)
                    elif mod_name == "WBFM":
                        # Wideband FM: larger deviation
                        msg = np.sin(2 * np.pi * 4 * t)
                        phase = 2 * np.pi * carrier * t + 5.0 * np.cumsum(msg) / frame_len
                        I = np.cos(phase)
                        Q = np.sin(phase)
                    else:
                        I = rng.normal(0, 1, size=frame_len)
                        Q = rng.normal(0, 1, size=frame_len)

                    # Multipath channel if chan_idx == 1
                    if chan_idx == 1:
                        # 3-tap multipath
                        taps = np.array([1.0, 0.4 * np.exp(1j * rng.uniform(0, 2 * np.pi)), 0.2 * np.exp(1j * rng.uniform(0, 2 * np.pi))])
                        taps = taps / np.sqrt(np.sum(np.abs(taps)**2))
                        z = np.convolve(I + 1j * Q, taps, mode="same")
                        I = np.real(z)
                        Q = np.imag(z)

                    # Add AWGN
                    sig_power = np.mean(I**2 + Q**2)
                    snr_linear = 10.0 ** (snr / 10.0)
                    noise_power = sig_power / snr_linear
                    noise_sigma = np.sqrt(noise_power / 2.0)
                    I += rng.normal(0, noise_sigma, size=frame_len)
                    Q += rng.normal(0, noise_sigma, size=frame_len)

                    # Store as [1024, 2]
                    sample = np.stack([I, Q], axis=1).astype(np.float32)
                    samples_list.append(sample)
                    y_mod_list.append(mod_idx)
                    y_chan_list.append(chan_idx)
                    y_snr_list.append(snr)

    X_all = np.array(samples_list, dtype=np.float32)
    y_mod_all = np.array(y_mod_list, dtype=np.int64)
    y_chan_all = np.array(y_chan_list, dtype=np.int64)
    y_snr_all = np.array(y_snr_list, dtype=np.int64)

    split = ExternalDatasetSplit(
        X=X_all,
        y_mod=y_mod_all,
        y_chan=y_chan_all,
        y_snr=y_snr_all,
        provenance="M6-DEV-SUBSET",
        is_evaluation_set=True,
    )

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(
            output_path,
            X=X_all,
            y_mod=y_mod_all,
            y_chan=y_chan_all,
            y_snr=y_snr_all,
            provenance="M6-DEV-SUBSET",
            random_seed=random_seed,
        )

    return split


def generate_pipeline_verification_subset(
    output_path: Optional[str] = "datasets/processed/pipeline_verification_subset.npz",
    num_per_condition: int = 5,
    random_seed: int = 999,
) -> ExternalDatasetSplit:
    """
    Generates a deterministic, strictly isolated pipeline verification subset
    with seed=999 (completely independent from dev split seed=42) to verify the
    evaluation pipeline when the real-world benchmark subset_test.h5 is unavailable.
    """
    rng = np.random.default_rng(random_seed)
    frame_len = 1024
    samples_list = []
    y_mod_list = []
    y_chan_list = []
    y_snr_list = []

    for mod_idx, mod_name in enumerate(EXTERNAL_MODULATION_CLASSES):
        for chan_idx in [0, 1]:
            for snr in EXTERNAL_SNR_VALUES:
                for _ in range(num_per_condition):
                    t = np.linspace(0, 1, frame_len, endpoint=False)
                    carrier = 10.0
                    if mod_name == "BPSK":
                        symbols = rng.choice([-1.0, 1.0], size=128)
                        bb = np.repeat(symbols, 8)
                        I = bb * np.cos(2 * np.pi * carrier * t)
                        Q = bb * np.sin(2 * np.pi * carrier * t)
                    elif mod_name == "QPSK":
                        sym_i = rng.choice([-1.0, 1.0], size=128) / np.sqrt(2)
                        sym_q = rng.choice([-1.0, 1.0], size=128) / np.sqrt(2)
                        I = np.repeat(sym_i, 8)
                        Q = np.repeat(sym_q, 8)
                    elif mod_name == "QAM":
                        levels = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10)
                        sym_i = rng.choice(levels, size=128)
                        sym_q = rng.choice(levels, size=128)
                        I = np.repeat(sym_i, 8)
                        Q = np.repeat(sym_q, 8)
                    elif mod_name == "GMSK":
                        bits = rng.choice([-1.0, 1.0], size=128)
                        phase = np.cumsum(np.repeat(bits, 8)) * (np.pi / 16.0)
                        I = np.cos(phase)
                        Q = np.sin(phase)
                    elif mod_name == "OFDM":
                        I = np.zeros(frame_len)
                        Q = np.zeros(frame_len)
                        num_carriers = 16
                        for k in range(1, num_carriers + 1):
                            phase_k = rng.uniform(0, 2 * np.pi)
                            I += (1.0 / np.sqrt(num_carriers)) * np.cos(2 * np.pi * k * 2 * t + phase_k)
                            Q += (1.0 / np.sqrt(num_carriers)) * np.sin(2 * np.pi * k * 2 * t + phase_k)
                    elif mod_name == "NBFM":
                        msg = np.sin(2 * np.pi * 2 * t)
                        phase = 2 * np.pi * carrier * t + 0.5 * np.cumsum(msg) / frame_len
                        I = np.cos(phase)
                        Q = np.sin(phase)
                    elif mod_name == "WBFM":
                        msg = np.sin(2 * np.pi * 4 * t)
                        phase = 2 * np.pi * carrier * t + 5.0 * np.cumsum(msg) / frame_len
                        I = np.cos(phase)
                        Q = np.sin(phase)
                    else:
                        I = rng.normal(0, 1, size=frame_len)
                        Q = rng.normal(0, 1, size=frame_len)

                    if chan_idx == 1:
                        taps = np.array([1.0, 0.4 * np.exp(1j * rng.uniform(0, 2 * np.pi)), 0.2 * np.exp(1j * rng.uniform(0, 2 * np.pi))])
                        taps = taps / np.sqrt(np.sum(np.abs(taps)**2))
                        z = np.convolve(I + 1j * Q, taps, mode="same")
                        I = np.real(z)
                        Q = np.imag(z)

                    sig_power = np.mean(I**2 + Q**2)
                    snr_linear = 10.0 ** (snr / 10.0)
                    noise_power = sig_power / snr_linear
                    noise_sigma = np.sqrt(noise_power / 2.0)
                    I += rng.normal(0, noise_sigma, size=frame_len)
                    Q += rng.normal(0, noise_sigma, size=frame_len)

                    sample = np.stack([I, Q], axis=1).astype(np.float32)
                    samples_list.append(sample)
                    y_mod_list.append(mod_idx)
                    y_chan_list.append(chan_idx)
                    y_snr_list.append(snr)

    X_all = np.array(samples_list, dtype=np.float32)
    y_mod_all = np.array(y_mod_list, dtype=np.int64)
    y_chan_all = np.array(y_chan_list, dtype=np.int64)
    y_snr_all = np.array(y_snr_list, dtype=np.int64)

    split = ExternalDatasetSplit(
        X=X_all,
        y_mod=y_mod_all,
        y_chan=y_chan_all,
        y_snr=y_snr_all,
        provenance="PIPELINE-VERIFICATION-TEST",
        is_evaluation_set=True,
    )

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(
            output_path,
            X=X_all,
            y_mod=y_mod_all,
            y_chan=y_chan_all,
            y_snr=y_snr_all,
            provenance="PIPELINE-VERIFICATION-TEST",
            random_seed=random_seed,
        )

    return split



def load_external_dataset(
    path: Optional[str] = None,
    allow_dev_subset_fallback: bool = True,
) -> ExternalDatasetSplit:
    """
    Loads the external real-world dataset.
    
    Checks in order:
      1. Explicit path if given
      2. datasets/subset_test.h5 or datasets/raw/subset_test.h5
      3. datasets/processed/external_dev_subset.npz (if fallback enabled)
      4. Auto-generates development split if missing
      
    Returns:
        ExternalDatasetSplit object.
    """
    candidate_paths = []
    if path is not None:
        candidate_paths.append(path)
    else:
        candidate_paths.extend([
            "ml/dataset/external/realworld/subset_test.h5",
            "datasets/external/realworld/subset_test.h5",
            "datasets/raw/subset_test.h5",
            "datasets/subset_test.h5",
            "datasets/processed/external_dev_subset.npz",
        ])

    for p in candidate_paths:
        if os.path.exists(p):
            if p.endswith(".h5"):
                try:
                    import h5py
                    with h5py.File(p, "r") as f:
                        X = np.array(f["X"], dtype=np.float32)
                        y_mod = np.array(f["y_mod"], dtype=np.int64)
                        y_chan = np.array(f["y_chan"], dtype=np.int64)
                        y_snr = np.array(f["y_snr"], dtype=np.int64)
                    return ExternalDatasetSplit(
                        X=X,
                        y_mod=y_mod,
                        y_chan=y_chan,
                        y_snr=y_snr,
                        provenance="REAL-WORLD-EXTERNAL-BENCHMARK",
                        is_evaluation_set=True,
                    )
                except Exception as e:
                    print(f"Warning: Failed reading H5 file {p}: {e}")
            elif p.endswith(".npz"):
                data = np.load(p)
                return ExternalDatasetSplit(
                    X=data["X"],
                    y_mod=data["y_mod"],
                    y_chan=data["y_chan"],
                    y_snr=data["y_snr"],
                    provenance=str(data.get("provenance", "M6-DEV-SUBSET")),
                    is_evaluation_set=True,
                )

    if allow_dev_subset_fallback:
        return generate_external_dev_subset()

    raise FileNotFoundError(
        f"External dataset not found. Checked: {candidate_paths}"
    )
