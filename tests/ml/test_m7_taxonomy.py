"""
Unit tests for M7 Taxonomy Alignment and Class Mapping (ml/evaluation/taxonomy.py).
"""
import pytest
from ml.evaluation.taxonomy import (
    EVALUATION_TAXONOMY,
    CLOSED_SET_CLASSES,
    OPEN_SET_CLASSES,
    is_external_class_supported,
    map_prediction_to_eval_class,
    is_eval_prediction_correct,
    partition_closed_and_open_set_indices,
)


def test_taxonomy_constants():
    """Verifies defined evaluation taxonomy constants."""
    assert EVALUATION_TAXONOMY == ["BPSK", "QPSK", "QAM", "WBFM", "UNKNOWN/UNSUPPORTED"]
    assert CLOSED_SET_CLASSES == ["BPSK", "QPSK", "QAM", "WBFM"]
    assert OPEN_SET_CLASSES == ["GMSK", "OFDM", "NBFM"]


def test_is_external_class_supported():
    """Verifies supported vs unsupported class identification."""
    assert is_external_class_supported("BPSK") is True
    assert is_external_class_supported("QPSK") is True
    assert is_external_class_supported("QAM") is True
    assert is_external_class_supported("WBFM") is True

    # Unsupported classes
    assert is_external_class_supported("GMSK") is False
    assert is_external_class_supported("OFDM") is False
    assert is_external_class_supported("NBFM") is False
    assert is_external_class_supported("UNKNOWN") is False


def test_qam_family_mapping():
    """
    CRITICAL: Verifies that QAM is evaluated at the family level:
      QAM16 -> QAM (correct)
      QAM64 -> QAM (correct)
    """
    assert map_prediction_to_eval_class("QAM16") == "QAM"
    assert map_prediction_to_eval_class("QAM64") == "QAM"

    # Exact matches
    assert map_prediction_to_eval_class("BPSK") == "BPSK"
    assert map_prediction_to_eval_class("QPSK") == "QPSK"
    assert map_prediction_to_eval_class("WBFM") == "WBFM"

    # Other classes in 11-class model map to OTHER_CLOSED
    assert map_prediction_to_eval_class("8PSK") == "OTHER_CLOSED"
    assert map_prediction_to_eval_class("AM-DSB") == "OTHER_CLOSED"
    assert map_prediction_to_eval_class("PAM4") == "OTHER_CLOSED"


def test_is_eval_prediction_correct():
    """Verifies correctness logic under M7 evaluation taxonomy rules."""
    # BPSK
    assert is_eval_prediction_correct("BPSK", "BPSK") is True
    assert is_eval_prediction_correct("QPSK", "BPSK") is False

    # QAM family matches
    assert is_eval_prediction_correct("QAM16", "QAM") is True
    assert is_eval_prediction_correct("QAM64", "QAM") is True
    assert is_eval_prediction_correct("8PSK", "QAM") is False

    # Unsupported classes are never correct in closed-set matching
    assert is_eval_prediction_correct("BPSK", "OFDM") is False
    assert is_eval_prediction_correct("CPFSK", "GMSK") is False


def test_partition_closed_and_open_set():
    """Verifies proper separation of closed-set and open-set indices."""
    # 0=BPSK, 1=QPSK, 2=QAM, 3=GMSK, 4=OFDM, 5=NBFM, 6=WBFM
    y_mod = [0, 1, 2, 3, 4, 5, 6, 0, 4]
    closed_idx, open_idx = partition_closed_and_open_set_indices(y_mod)

    # Closed indices: 0(BPSK), 1(QPSK), 2(QAM), 6(WBFM), 7(BPSK)
    assert closed_idx == [0, 1, 2, 6, 7]
    # Open indices: 3(GMSK), 4(OFDM), 5(NBFM), 8(OFDM)
    assert open_idx == [3, 4, 5, 8]
