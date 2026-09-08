"""
Unit tests for ml/dataset/external_dataset.py  (M6 external dataset infrastructure).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from ml.dataset.external_dataset import (
    EXTERNAL_CLASSES,
    EXTERNAL_CLASS_TO_INDEX,
    EXTERNAL_INDEX_TO_CLASS,
    EXTERNAL_BENCHMARK_METADATA,
    EXTERNAL_FRAME_LENGTH,
    SIGMA_WINDOW_LENGTH,
    frames_to_windows,
    load_external_dataset,
    generate_external_dev_subset,
)


# ── 1. Taxonomy constants ─────────────────────────────────────────────────────

def test_external_classes_count():
    assert len(EXTERNAL_CLASSES) == 7


def test_external_classes_content():
    expected = {"BPSK", "QPSK", "QAM", "GMSK", "OFDM", "NBFM", "WBFM"}
    assert set(EXTERNAL_CLASSES) == expected


def test_external_index_maps_consistent():
    for i, c in enumerate(EXTERNAL_CLASSES):
        assert EXTERNAL_CLASS_TO_INDEX[c] == i
        assert EXTERNAL_INDEX_TO_CLASS[i] == c


def test_benchmark_metadata_fields():
    assert EXTERNAL_BENCHMARK_METADATA.frame_length == EXTERNAL_FRAME_LENGTH
    assert EXTERNAL_BENCHMARK_METADATA.num_classes == 7


# ── 2. frames_to_windows — channels-last input ───────────────────────────────

def test_frames_to_windows_channels_last():
    """[N, 1024, 2] → [M, 2, 128]"""
    rng = np.random.default_rng(0)
    n_frames = 5
    frames = rng.standard_normal((n_frames, EXTERNAL_FRAME_LENGTH, 2)).astype(np.float32)
    windows, frame_idx = frames_to_windows(frames)
    expected_w = n_frames * (EXTERNAL_FRAME_LENGTH // SIGMA_WINDOW_LENGTH)
    assert windows.shape == (expected_w, 2, SIGMA_WINDOW_LENGTH)
    assert frame_idx.shape == (expected_w,)


def test_frames_to_windows_channels_first():
    """[N, 2, 1024] → [M, 2, 128]"""
    rng = np.random.default_rng(1)
    n_frames = 3
    frames = rng.standard_normal((n_frames, 2, EXTERNAL_FRAME_LENGTH)).astype(np.float32)
    windows, frame_idx = frames_to_windows(frames)
    expected_w = n_frames * (EXTERNAL_FRAME_LENGTH // SIGMA_WINDOW_LENGTH)
    assert windows.shape == (expected_w, 2, SIGMA_WINDOW_LENGTH)


def test_frames_to_windows_dtype():
    rng = np.random.default_rng(2)
    frames = rng.standard_normal((2, EXTERNAL_FRAME_LENGTH, 2)).astype(np.float64)
    windows, _ = frames_to_windows(frames)
    assert windows.dtype == np.float32


def test_frames_to_windows_frame_indices():
    """frame_idx must map each window back to its source frame."""
    rng = np.random.default_rng(3)
    n_frames = 4
    frames = rng.standard_normal((n_frames, EXTERNAL_FRAME_LENGTH, 2)).astype(np.float32)
    windows, frame_idx = frames_to_windows(frames)
    wpf = EXTERNAL_FRAME_LENGTH // SIGMA_WINDOW_LENGTH
    for f in range(n_frames):
        for i in range(wpf):
            assert frame_idx[f * wpf + i] == f


def test_frames_to_windows_content_correct():
    """The content of the first window of the first frame must match the source."""
    rng = np.random.default_rng(4)
    n = 2
    frames = rng.standard_normal((n, EXTERNAL_FRAME_LENGTH, 2)).astype(np.float32)
    windows, _ = frames_to_windows(frames)
    # windows[0] = first 128 samples of frames[0]
    expected = frames[0, :SIGMA_WINDOW_LENGTH, :].T  # [2, 128]
    np.testing.assert_allclose(windows[0], expected, atol=1e-6)


def test_frames_to_windows_bad_shape():
    with pytest.raises(ValueError):
        frames_to_windows(np.zeros((3, 512, 2), dtype=np.float32))


def test_frames_to_windows_non_ndarray():
    with pytest.raises(TypeError):
        frames_to_windows([[1, 2], [3, 4]])  # type: ignore


def test_frames_to_windows_1d():
    with pytest.raises(ValueError):
        frames_to_windows(np.zeros(1024, dtype=np.float32))


# ── 3. load_external_dataset — absent file ────────────────────────────────────

def test_load_external_dataset_missing_file():
    """load_external_dataset must raise FileNotFoundError when absent."""
    with pytest.raises(FileNotFoundError):
        load_external_dataset(path="datasets/nonexistent_file_sigma_test.h5")


# ── 4. generate_external_dev_subset ──────────────────────────────────────────

def test_dev_subset_emits_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frames, labels, tag = generate_external_dev_subset(n_per_class=2)
    assert any("M6-DEV-SUBSET" in str(w.message) for w in caught)


def test_dev_subset_tag():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        _, _, tag = generate_external_dev_subset(n_per_class=2)
    assert tag == "M6-DEV-SUBSET"


def test_dev_subset_shape():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        frames, labels, _ = generate_external_dev_subset(n_per_class=5)
    assert frames.shape == (5 * 7, 2, EXTERNAL_FRAME_LENGTH)
    assert labels.shape == (5 * 7,)


def test_dev_subset_dtype():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        frames, labels, _ = generate_external_dev_subset(n_per_class=3)
    assert frames.dtype == np.float32
    assert labels.dtype == np.int64


def test_dev_subset_labels_cover_all_classes():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        _, labels, _ = generate_external_dev_subset(n_per_class=2)
    assert set(labels.tolist()) == set(range(7))


def test_dev_subset_deterministic():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        f1, l1, _ = generate_external_dev_subset(n_per_class=3, seed=7)
        f2, l2, _ = generate_external_dev_subset(n_per_class=3, seed=7)
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(l1, l2)
