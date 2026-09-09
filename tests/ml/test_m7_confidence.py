"""
Unit tests for M7 Confidence Analysis and Temperature Scaling (ml/evaluation/confidence.py).
"""
import pytest
import numpy as np

from ml.evaluation.confidence import (
    analyze_confidence_reliability,
    TemperatureScaler,
    _compute_distribution,
)


def test_compute_distribution():
    """Verifies statistical quantile extraction."""
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dist = _compute_distribution(values)
    assert dist is not None
    assert dist.mean == pytest.approx(0.55)
    assert dist.median == pytest.approx(0.55)
    assert dist.count == 10
    assert dist.min == 0.1
    assert dist.max == 1.0


def test_analyze_confidence_reliability():
    """Verifies overconfidence rate and correct/incorrect separation."""
    # 2 correct with confidence 0.85 and 0.90
    # 2 incorrect with confidence 0.85 (overconfident) and 0.40
    confs = [0.85, 0.90, 0.85, 0.40]
    accs = [True, True, False, False]

    analysis = analyze_confidence_reliability(confs, accs, high_conf_threshold=0.80)
    assert analysis["is_calibrated"] is False
    assert analysis["high_confidence_error_count"] == 1
    # 1 out of 2 errors is overconfident -> 50%
    assert analysis["overconfidence_rate"] == 0.50
    assert analysis["high_confidence_correct_count"] == 2


def test_temperature_scaler_fitting_and_calibration():
    """Verifies TemperatureScaler fits strictly and scales logits."""
    scaler = TemperatureScaler()
    assert scaler.is_fitted is False

    # Synthetic dev logits: [20 samples, 4 classes]
    rng = np.random.default_rng(42)
    dev_logits = rng.standard_normal((20, 4)).astype(np.float32)
    dev_targets = rng.integers(0, 4, size=20)

    t_val = scaler.fit_on_dev(dev_logits, dev_targets, max_iter=20)
    assert scaler.is_fitted is True
    assert t_val > 0.0

    # Calibrate new test logits
    test_logits = rng.standard_normal((5, 4)).astype(np.float32)
    probs = scaler.calibrate_probs(test_logits)
    assert probs.shape == (5, 4)
    # Must sum to 1
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
