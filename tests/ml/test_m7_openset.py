"""
Unit tests for M7 Open-Set Rejection and Threshold Freezing (ml/evaluation/openset.py).
"""
import pytest
import numpy as np

from ml.evaluation.openset import (
    OpenSetThresholds,
    OpenSetEvaluator,
    select_thresholds_on_dev,
    compute_entropy,
)


def test_compute_entropy():
    """Verifies Shannon entropy calculation."""
    # Deterministic 1-hot has 0 entropy
    one_hot = np.array([1.0, 0.0, 0.0])
    assert compute_entropy(one_hot) == pytest.approx(0.0, abs=1e-5)

    # Uniform has ln(3) entropy
    uniform = np.array([1/3, 1/3, 1/3])
    assert compute_entropy(uniform) == pytest.approx(np.log(3), abs=1e-4)


def test_select_thresholds_on_dev():
    """Verifies threshold calibration on development samples."""
    probs = np.array([
        [0.9, 0.05, 0.05],  # BPSK (known)
        [0.8, 0.1, 0.1],   # QPSK (known)
        [0.4, 0.3, 0.3],   # OFDM (unknown)
    ])
    y_mod = [0, 1, 4] # 0=BPSK, 1=QPSK, 4=OFDM

    thresh = select_thresholds_on_dev(probs, y_mod, target_known_acceptance=0.90)
    assert isinstance(thresh, OpenSetThresholds)
    assert 0.0 < thresh.confidence_threshold <= 1.0
    assert thresh.entropy_threshold > 0.0
    assert thresh.source == "M6-DEV-SUBSET-CALIBRATED"


def test_openset_evaluator_decisions():
    """Verifies known acceptance and unknown rejection calculation."""
    # Thresholds: require MSP >= 0.70 and Entropy <= 1.50
    thresh = OpenSetThresholds(confidence_threshold=0.70, entropy_threshold=1.50)
    evaluator = OpenSetEvaluator(thresh)

    probs = np.array([
        [0.85, 0.10, 0.05], # High confidence -> Accepted
        [0.34, 0.33, 0.33], # Flat / Low confidence -> Rejected
    ])
    assert evaluator.is_accepted(probs[0]) is True
    assert evaluator.is_accepted(probs[1]) is False

    # Sample 0 is BPSK (known, index 0)
    # Sample 1 is OFDM (unknown, index 4)
    y_mod = [0, 4]
    metrics = evaluator.evaluate(probs, y_mod)

    assert metrics.known_samples == 1
    assert metrics.unknown_samples == 1
    assert metrics.known_accepted == 1
    assert metrics.known_acceptance_rate == 1.0
    assert metrics.unknown_rejected == 1
    assert metrics.unknown_rejection_rate == 1.0
    assert metrics.false_acceptance_rate == 0.0
    assert metrics.false_rejection_rate == 0.0
