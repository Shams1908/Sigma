"""
Unit tests for M6 Taxonomy Alignment and Open-Set Rejection (taxonomy.py).
Verifies:
  - Exact and family-level class compatibility mappings
  - Unsupported modulation detection (OFDM, GMSK, NBFM)
  - Configurable open-set rejection (entropy and MSP thresholds)
  - Preservation of class taxonomy without forced mislabeling
"""
import pytest
import numpy as np

from ml.dataset.taxonomy import (
    is_external_class_supported,
    is_prediction_compatible,
    OpenSetClassifier,
    EXACT_OVERLAP_CLASSES,
    QAM_FAMILY_CLASSES,
    UNSUPPORTED_EXTERNAL_CLASSES,
    SUPPORTED_M5_CLASSES,
)


def test_class_support_identification():
    """Verifies that supported vs unsupported modulations are properly partitioned."""
    # Supported exact matches
    assert is_external_class_supported("BPSK") is True
    assert is_external_class_supported("QPSK") is True
    assert is_external_class_supported("WBFM") is True

    # Supported family match
    assert is_external_class_supported("QAM") is True

    # Unsupported modulations
    assert is_external_class_supported("OFDM") is False
    assert is_external_class_supported("GMSK") is False
    assert is_external_class_supported("NBFM") is False


def test_prediction_compatibility_logic():
    """Verifies prediction compatibility checks between M5 predictions and external classes."""
    # Exact matches
    assert is_prediction_compatible("BPSK", "BPSK") is True
    assert is_prediction_compatible("QPSK", "BPSK") is False

    # QAM family matches
    assert is_prediction_compatible("QAM16", "QAM") is True
    assert is_prediction_compatible("QAM64", "QAM") is True
    assert is_prediction_compatible("8PSK", "QAM") is False

    # Unsupported classes are never compatible
    assert is_prediction_compatible("BPSK", "OFDM") is False
    assert is_prediction_compatible("CPFSK", "GMSK") is False


def test_openset_classifier_threshold_validation():
    """Verifies parameter validation for OpenSetClassifier."""
    clf = OpenSetClassifier(confidence_threshold=0.40, entropy_threshold=1.5)
    assert clf.confidence_threshold == 0.40
    assert clf.entropy_threshold == 1.5

    with pytest.raises(ValueError):
        OpenSetClassifier(confidence_threshold=-0.1)
    with pytest.raises(ValueError):
        OpenSetClassifier(confidence_threshold=1.5)
    with pytest.raises(ValueError):
        OpenSetClassifier(entropy_threshold=0.0)


def test_openset_classifier_decisions():
    """Verifies KNOWN, LOW_CONFIDENCE, and UNSUPPORTED decisions."""
    clf = OpenSetClassifier(confidence_threshold=0.35, entropy_threshold=1.8)

    # 1. High confidence supported class
    probs_confident = np.zeros(11, dtype=np.float32)
    probs_confident[3] = 0.90  # BPSK
    probs_confident[0] = 0.10
    
    pred_confident = clf.classify_probs(probs_confident, ground_truth="BPSK")
    assert pred_confident.decision == "KNOWN"
    assert pred_confident.is_supported is True
    assert pred_confident.is_correct is True
    assert pred_confident.predicted_class == "BPSK"

    # 2. Low confidence / ambiguous prediction
    probs_flat = np.ones(11, dtype=np.float32) / 11.0  # Max prob ~ 0.091 < 0.35
    pred_flat = clf.classify_probs(probs_flat, ground_truth="BPSK")
    assert pred_flat.decision == "LOW_CONFIDENCE"
    assert pred_flat.is_correct is False

    # 3. Unsupported modulation (OFDM)
    # Even if model happens to output a high probability on some closed class,
    # the classifier flags it as UNSUPPORTED because ground truth is outside M5 taxonomy
    pred_ofdm = clf.classify_probs(probs_confident, ground_truth="OFDM")
    assert pred_ofdm.decision == "UNSUPPORTED"
    assert pred_ofdm.is_supported is False
    assert pred_ofdm.is_correct is False
