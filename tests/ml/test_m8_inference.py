"""
Unit tests for M8 estimate_parameters Inference API (ml/parameters/inference.py).
"""
import pytest
import numpy as np

from ml.parameters.inference import (
    estimate_parameters,
    ParameterAnalysisResult,
    SingleParameterResult,
)


def test_estimate_parameters_structure():
    """Verifies output schema of estimate_parameters."""
    rng = np.random.default_rng(42)
    # Synthetic signal: 256 samples of complex noise
    iq = (rng.standard_normal((2, 256)) * 0.1).astype(np.float32)

    res = estimate_parameters(iq, sample_rate=800000.0, model_path="non_existent_model.pt")
    assert isinstance(res, ParameterAnalysisResult)
    assert isinstance(res.symbol_rate, SingleParameterResult)
    assert isinstance(res.snr_db, SingleParameterResult)

    assert res.symbol_rate.estimate > 0.0
    assert 0.0 <= res.symbol_rate.confidence <= 1.0
    assert res.symbol_rate.uncertainty > 0.0

    d = res.to_dict()
    assert "symbol_rate" in d
    assert "snr_db" in d
    assert d["sample_rate"] == 800000.0
    assert d["signal_length_samples"] == 256


def test_estimate_parameters_invalid_input():
    """Verifies rejection of malformed or too short inputs."""
    with pytest.raises(ValueError):
        estimate_parameters(np.zeros((2, 10)), sample_rate=800000.0) # Length < 64

    with pytest.raises(ValueError):
        estimate_parameters(np.zeros((2, 128)), sample_rate=-1000.0) # Invalid sample_rate
