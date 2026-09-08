"""
Unit tests for M6 External Real-World Dataset Handler (external_dataset.py).
Verifies:
  - Dataset metadata structure & constants
  - Frame-to-window slicing (1024 -> 8 non-overlapping 128-sample windows)
  - Provenance isolation and assertions
  - Development subset generator completeness (7 classes, 2 channels, 6 SNRs)
"""
import pytest
import numpy as np

from ml.dataset.external_dataset import (
    RealWorldDatasetMetadata,
    EXTERNAL_MODULATION_CLASSES,
    EXTERNAL_CHANNEL_CONDITIONS,
    EXTERNAL_SNR_VALUES,
    slice_frames_to_windows,
    generate_external_dev_subset,
    load_external_dataset,
    ExternalDatasetSplit,
)


def test_external_dataset_metadata():
    """Verifies metadata constants match the specification."""
    meta = RealWorldDatasetMetadata()
    assert meta.frame_length == 1024
    assert meta.num_benchmark_samples == 80000
    assert meta.classes == ["BPSK", "QPSK", "QAM", "GMSK", "OFDM", "NBFM", "WBFM"]
    assert meta.channel_conditions == ["clean", "multipath"]
    assert meta.snr_levels == [20, 22, 24, 26, 28, 30]


def test_slice_frames_to_windows():
    """Verifies that 1024-sample frames slice into 8 non-overlapping 128-sample windows."""
    # Create 4 mock frames of shape [4, 1024, 2]
    rng = np.random.default_rng(42)
    mock_frames = rng.standard_normal((4, 1024, 2)).astype(np.float32)

    windows = slice_frames_to_windows(mock_frames, window_length=128)
    
    # 4 frames * (1024 / 128) = 4 * 8 = 32 windows
    assert windows.shape == (32, 2, 128)
    assert windows.dtype == np.float32

    # Verify first window content matches exactly the start of the first frame
    # mock_frames[0, :128, 0] is I channel, mock_frames[0, :128, 1] is Q channel
    assert np.allclose(windows[0, 0], mock_frames[0, :128, 0])
    assert np.allclose(windows[0, 1], mock_frames[0, :128, 1])


def test_provenance_isolation_guard():
    """Verifies that external dataset enforces evaluation isolation."""
    mock_x = np.zeros((10, 1024, 2), dtype=np.float32)
    mock_y = np.zeros(10, dtype=np.int64)

    # Valid evaluation split
    eval_split = ExternalDatasetSplit(
        X=mock_x,
        y_mod=mock_y,
        y_chan=mock_y,
        y_snr=mock_y,
        provenance="BENCHMARK-TEST",
        is_evaluation_set=True,
    )
    eval_split.assert_evaluation_isolation()  # Must not raise

    # Contaminated split flagged as training
    corrupt_split = ExternalDatasetSplit(
        X=mock_x,
        y_mod=mock_y,
        y_chan=mock_y,
        y_snr=mock_y,
        provenance="LEAKED-TEST",
        is_evaluation_set=False,
    )
    with pytest.raises(ValueError, match="Provenance violation"):
        corrupt_split.assert_evaluation_isolation()


def test_generate_external_dev_subset():
    """Verifies generation of deterministic development split."""
    split = generate_external_dev_subset(output_path=None, num_per_condition=2, random_seed=42)
    
    # Total combinations: 7 classes * 2 channels * 6 SNRs * 2 per condition = 168
    expected_count = 7 * 2 * 6 * 2
    assert len(split.X) == expected_count
    assert split.X.shape == (expected_count, 1024, 2)
    assert split.X.dtype == np.float32
    assert np.isfinite(split.X).all()

    # Verify all classes, channels, SNRs are represented
    assert set(np.unique(split.y_mod)) == set(range(7))
    assert set(np.unique(split.y_chan)) == {0, 1}
    assert set(np.unique(split.y_snr)) == set(EXTERNAL_SNR_VALUES)
    assert split.provenance == "M6-DEV-SUBSET"
    assert split.is_evaluation_set is True


def test_load_external_dataset_fallback():
    """Verifies load_external_dataset fallback to dev split when h5 is missing."""
    split = load_external_dataset(allow_dev_subset_fallback=True)
    assert isinstance(split, ExternalDatasetSplit)
    assert split.X.ndim == 3
    assert split.X.shape[1] == 1024 or split.X.shape[2] == 1024
