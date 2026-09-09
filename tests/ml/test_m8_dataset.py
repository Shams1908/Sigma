"""
Unit tests for M8 Dataset Generation and Split Strategy (ml/parameters/dataset.py).
"""
import pytest
import numpy as np

from ml.parameters.dataset import (
    generate_parameter_dataset,
    synthesize_parameter_sample,
    ParameterDatasetSplit,
)


def test_synthesize_parameter_sample():
    """Verifies synthesis of single parameter sample."""
    sample = synthesize_parameter_sample(
        modulation="QPSK",
        sps=8.0,
        snr_db=20.0,
        sample_rate=800000.0,
        length=128,
    )
    assert isinstance(sample, np.ndarray)
    assert sample.shape == (2, 128)
    assert sample.dtype == np.float32
    assert np.isfinite(sample).all()


def test_generate_parameter_dataset():
    """Verifies parameter dataset generator shapes, bounds, and reproducibility."""
    split = generate_parameter_dataset(num_samples=50, random_seed=42)
    assert isinstance(split, ParameterDatasetSplit)
    assert split.X.shape == (50, 2, 128)
    assert len(split.sps) == 50
    assert len(split.symbol_rate) == 50
    assert len(split.snr_db) == 50

    # Bounds check
    assert (split.sps >= 3.0).all()
    assert (split.symbol_rate >= 10000.0).all()
    assert (split.snr_db >= -15.0).all()
    assert (split.snr_db <= 35.0).all()

    # Reproducibility check
    split2 = generate_parameter_dataset(num_samples=50, random_seed=42)
    assert np.allclose(split.X, split2.X)
    assert np.allclose(split.sps, split2.sps)
    assert np.allclose(split.snr_db, split2.snr_db)
