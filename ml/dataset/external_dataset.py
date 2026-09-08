"""
External benchmark dataset infrastructure for SIGMA ML evaluation.

PURPOSE:
    Provides loading, conversion, and provenance-tracking utilities for
    external RF signal classification benchmark data.

EVALUATION ISOLATION:
    This module is strictly for offline evaluation and calibration.
    It MUST NOT be called during normal API signal inference.
    Do NOT import this module from the runtime inference pipeline.

ACTUAL EXTERNAL BENCHMARK:
    Dataset file: subset_test.h5
    Expected location: datasets/subset_test.h5  (or configured path)
    Format: HDF5 with keys for IQ samples and labels.
    Frame length: 1024 samples per frame, shape [N, 1024, 2] or [N, 2, 1024].

    7-class taxonomy:
        BPSK, QPSK, QAM, GMSK, OFDM, NBFM, WBFM

    This file is NOT included in the repository.
    Tests that require it will skip when it is absent.

DEV SUBSET WARNING:
    generate_external_dev_subset() produces M6-DEV-SUBSET synthetic data.
    It is SYNTHETIC DEVELOPMENT DATA, NOT real-world benchmark data.
    It MUST NOT be described or reported as real-world validation evidence.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── External dataset taxonomy (7 classes) ─────────────────────────────────────

EXTERNAL_CLASSES: List[str] = [
    "BPSK",
    "QPSK",
    "QAM",
    "GMSK",
    "OFDM",
    "NBFM",
    "WBFM",
]

EXTERNAL_CLASS_TO_INDEX: Dict[str, int] = {
    c: i for i, c in enumerate(EXTERNAL_CLASSES)
}

EXTERNAL_INDEX_TO_CLASS: Dict[int, str] = {
    i: c for i, c in enumerate(EXTERNAL_CLASSES)
}

# Frame length used by the external dataset
EXTERNAL_FRAME_LENGTH: int = 1024

# Window length expected by the SIGMA CNN
SIGMA_WINDOW_LENGTH: int = 128


@dataclass(frozen=True)
class ExternalDatasetMetadata:
    """Provenance metadata for an external benchmark dataset."""

    name: str
    frame_length: int
    num_classes: int
    class_taxonomy: Tuple[str, ...]
    source: str
    notes: str = ""


EXTERNAL_BENCHMARK_METADATA = ExternalDatasetMetadata(
    name="subset_test",
    frame_length=EXTERNAL_FRAME_LENGTH,
    num_classes=len(EXTERNAL_CLASSES),
    class_taxonomy=tuple(EXTERNAL_CLASSES),
    source="External benchmark — see datasets/subset_test.h5",
    notes=(
        "7-class taxonomy. Frames are 1024 samples each. "
        "Must NOT be used as training data. Evaluation only."
    ),
)


# ── Frame-to-window conversion ────────────────────────────────────────────────

def frames_to_windows(
    frames: np.ndarray,
    window_length: int = SIGMA_WINDOW_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert external 1024-sample frames to non-overlapping 128-sample windows.

    The external dataset provides IQ frames of 1024 samples in one of two
    layouts:
        [N, 1024, 2]  (samples-last)
        [N, 2, 1024]  (channels-first)

    Both are accepted and converted to the SIGMA canonical [M, 2, L] layout.

    Frames that are not exactly divisible by ``window_length`` are truncated.

    Args:
        frames:        ndarray of shape [N, 1024, 2] or [N, 2, 1024].
        window_length: Target window size (default 128).

    Returns:
        (windows, frame_indices)
        windows:       ndarray of shape [M, 2, window_length] float32
        frame_indices: ndarray of shape [M] with the source frame index for
                       each window (useful for label propagation).
    """
    if not isinstance(frames, np.ndarray):
        raise TypeError(
            f"frames must be a numpy ndarray, got {type(frames).__name__}"
        )
    if frames.ndim != 3:
        raise ValueError(
            f"frames must be 3-D, got shape {frames.shape}"
        )

    n_frames = frames.shape[0]

    # Normalise to [N, 2, 1024]
    if frames.shape[1] == 1024 and frames.shape[2] == 2:
        # [N, 1024, 2] → [N, 2, 1024]
        frames_ch_first = frames.transpose(0, 2, 1).astype(np.float32)
    elif frames.shape[1] == 2 and frames.shape[2] == EXTERNAL_FRAME_LENGTH:
        frames_ch_first = frames.astype(np.float32)
    else:
        raise ValueError(
            f"Unexpected frame shape {frames.shape}. "
            f"Expected [N, {EXTERNAL_FRAME_LENGTH}, 2] or [N, 2, {EXTERNAL_FRAME_LENGTH}]."
        )

    windows_per_frame = EXTERNAL_FRAME_LENGTH // window_length
    n_windows = n_frames * windows_per_frame

    windows = np.zeros((n_windows, 2, window_length), dtype=np.float32)
    frame_indices = np.zeros(n_windows, dtype=np.int64)

    w = 0
    for f in range(n_frames):
        for i in range(windows_per_frame):
            start = i * window_length
            end = start + window_length
            windows[w] = frames_ch_first[f, :, start:end]
            frame_indices[w] = f
            w += 1

    return windows, frame_indices


# ── Actual dataset loader ──────────────────────────────────────────────────────

def load_external_dataset(
    path: str = "datasets/subset_test.h5",
) -> Tuple[np.ndarray, np.ndarray, ExternalDatasetMetadata]:
    """
    Load the external benchmark HDF5 dataset.

    Returns:
        (frames, labels, metadata)
        frames: float32 ndarray [N, 2, 1024] canonical channels-first IQ.
        labels: int64 ndarray [N] external class indices (0–6).
        metadata: ExternalDatasetMetadata provenance record.

    Raises:
        FileNotFoundError: If the dataset file is not present.
        ImportError:       If h5py is not installed.
        ValueError:        If the file does not contain the expected keys.

    NOTE:
        This loader must only be called from evaluation / calibration code.
        Do NOT call from the production inference pipeline.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"External benchmark dataset not found at '{path}'. "
            "This file is not included in the repository and must be "
            "obtained separately. See datasets/README.md."
        )

    try:
        import h5py  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "h5py is required to load the external HDF5 dataset. "
            "Install it with: pip install h5py"
        ) from exc

    with h5py.File(str(p), "r") as f:
        available_keys = list(f.keys())

        # Try common key conventions
        data_key = None
        label_key = None
        for dk in ("X", "data", "samples", "iq"):
            if dk in f:
                data_key = dk
                break
        for lk in ("Y", "labels", "label", "y"):
            if lk in f:
                label_key = lk
                break

        if data_key is None or label_key is None:
            raise ValueError(
                f"Cannot find IQ data / label arrays in '{path}'. "
                f"Available keys: {available_keys}"
            )

        raw_data = f[data_key][:]
        raw_labels = f[label_key][:]

    frames = np.asarray(raw_data, dtype=np.float32)
    labels = np.asarray(raw_labels, dtype=np.int64)

    # Normalise layout to [N, 2, EXTERNAL_FRAME_LENGTH]
    if frames.ndim == 3 and frames.shape[-1] == 2:
        frames = frames.transpose(0, 2, 1)

    return frames, labels, EXTERNAL_BENCHMARK_METADATA


# ── Dev subset (SYNTHETIC — NOT real-world data) ─────────────────────────────

def generate_external_dev_subset(
    n_per_class: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Generate a tiny SYNTHETIC development subset for offline testing.

    WARNING:
        This is M6-DEV-SUBSET SYNTHETIC DATA.
        It is generated by an RNG — it does NOT represent real-world
        RF channel behaviour and MUST NOT be used as benchmark evidence.
        It exists solely to allow tests to run without the actual
        external dataset file.

    Returns:
        (frames, labels, provenance_tag)
        frames:          float32 ndarray [N, 2, 1024]
        labels:          int64 ndarray   [N]  (class indices 0–6)
        provenance_tag:  always "M6-DEV-SUBSET"
    """
    warnings.warn(
        "generate_external_dev_subset() produces SYNTHETIC development data "
        "(M6-DEV-SUBSET). This is NOT real-world benchmark data and MUST NOT "
        "be used as validation evidence.",
        UserWarning,
        stacklevel=2,
    )

    rng = np.random.default_rng(seed)
    n_total = n_per_class * len(EXTERNAL_CLASSES)

    frames = rng.standard_normal(
        (n_total, 2, EXTERNAL_FRAME_LENGTH)
    ).astype(np.float32)

    labels = np.repeat(
        np.arange(len(EXTERNAL_CLASSES), dtype=np.int64), n_per_class
    )

    return frames, labels, "M6-DEV-SUBSET"
