"""
Unit tests for M7 Dataset Integrity and Split Isolation (ml/evaluation/integrity.py).
"""
import pytest
import numpy as np

from ml.dataset.external_dataset import ExternalDatasetSplit
from ml.evaluation.integrity import (
    verify_dataset_integrity,
    DatasetIntegrityError,
    DatasetIntegrityReport,
)


@pytest.fixture
def valid_mock_split():
    """Generates a small valid mock split of shape [10, 1024, 2]."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 1024, 2)).astype(np.float32)
    y_mod = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2], dtype=np.int64)
    y_chan = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    y_snr = np.array([20, 22, 24, 26, 28, 30, 20, 22, 24, 26], dtype=np.int64)
    return ExternalDatasetSplit(
        X=X,
        y_mod=y_mod,
        y_chan=y_chan,
        y_snr=y_snr,
        provenance="MOCK-EVAL-SPLIT",
        is_evaluation_set=True,
    )


def test_integrity_valid_split(valid_mock_split):
    """Verifies that a valid split passes all integrity checks."""
    report = verify_dataset_integrity(valid_mock_split, raise_on_error=True)
    assert isinstance(report, DatasetIntegrityReport)
    assert report.is_valid is True
    assert report.num_samples == 10
    assert report.frame_length == 1024
    assert report.iq_channels == 2
    assert report.has_nans is False
    assert report.has_infs is False
    assert len(report.errors) == 0
    assert "length_consistency" in report.checks_passed
    assert "no_nans_or_infs" in report.checks_passed


def test_integrity_nan_detection(valid_mock_split):
    """Verifies that NaN values in X fail loudly."""
    valid_mock_split.X[0, 5, 0] = np.nan
    with pytest.raises(DatasetIntegrityError, match="NaN"):
        verify_dataset_integrity(valid_mock_split, raise_on_error=True)


def test_integrity_inf_detection(valid_mock_split):
    """Verifies that Inf values in X fail loudly."""
    valid_mock_split.X[0, 5, 1] = np.inf
    with pytest.raises(DatasetIntegrityError, match="infinite"):
        verify_dataset_integrity(valid_mock_split, raise_on_error=True)


def test_integrity_invalid_shape(valid_mock_split):
    """Verifies that incorrect frame dimensions are rejected."""
    bad_split = ExternalDatasetSplit(
        X=np.zeros((10, 512, 2), dtype=np.float32),
        y_mod=valid_mock_split.y_mod,
        y_chan=valid_mock_split.y_chan,
        y_snr=valid_mock_split.y_snr,
        provenance="MOCK-BAD-SHAPE",
        is_evaluation_set=True,
    )
    with pytest.raises(DatasetIntegrityError, match="Invalid dimensions"):
        verify_dataset_integrity(bad_split, expected_frame_length=1024, raise_on_error=True)


def test_integrity_unexpected_labels(valid_mock_split):
    """Verifies that out-of-bounds modulation classes fail."""
    valid_mock_split.y_mod[0] = 99  # Class 99 is invalid
    with pytest.raises(DatasetIntegrityError, match="Unexpected modulation classes"):
        verify_dataset_integrity(valid_mock_split, raise_on_error=True)


def test_integrity_split_contamination():
    """Verifies that overlapping samples between dev and test are caught."""
    rng = np.random.default_rng(42)
    shared_sample = rng.standard_normal((1, 1024, 2)).astype(np.float32)

    dev_x = np.concatenate([shared_sample, rng.standard_normal((4, 1024, 2)).astype(np.float32)], axis=0)
    test_x = np.concatenate([shared_sample, rng.standard_normal((4, 1024, 2)).astype(np.float32)], axis=0)

    dev_split = ExternalDatasetSplit(
        X=dev_x, y_mod=np.zeros(5, dtype=np.int64),
        y_chan=np.zeros(5, dtype=np.int64), y_snr=np.full(5, 20, dtype=np.int64),
        provenance="DEV", is_evaluation_set=True,
    )
    test_split = ExternalDatasetSplit(
        X=test_x, y_mod=np.zeros(5, dtype=np.int64),
        y_chan=np.zeros(5, dtype=np.int64), y_snr=np.full(5, 20, dtype=np.int64),
        provenance="TEST", is_evaluation_set=True,
    )

    with pytest.raises(DatasetIntegrityError, match="Split contamination detected"):
        verify_dataset_integrity(test_split, reference_dev_split=dev_split, raise_on_error=True)
