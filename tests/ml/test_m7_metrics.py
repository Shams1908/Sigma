"""
Unit tests for M7 Metrics, Confusion Matrix, SNR & Channel Breakdowns, and ECE (ml/evaluation/metrics.py).
"""
import pytest
import numpy as np

from ml.evaluation.metrics import (
    calculate_core_metrics,
    calculate_confusion_matrix,
    calculate_snr_breakdown,
    calculate_channel_breakdown,
    calculate_ece,
)


def test_calculate_core_metrics_perfect():
    """Verifies 100% metrics when all predictions match ground truth."""
    y_true = ["BPSK", "QPSK", "QAM", "WBFM"]
    y_pred = ["BPSK", "QPSK", "QAM16", "WBFM"] # QAM16 matches QAM at family level

    res = calculate_core_metrics(y_true, y_pred)
    assert res["accuracy"] == 1.0
    assert res["macro_f1"] == 1.0
    assert res["weighted_f1"] == 1.0
    assert res["per_class"]["QAM"]["f1"] == 1.0
    assert res["per_class"]["BPSK"]["f1"] == 1.0


def test_calculate_core_metrics_mixed():
    """Verifies exact calculation on mixed predictions."""
    y_true = ["BPSK", "BPSK", "QAM", "WBFM"]
    # Sample 1: BPSK correct
    # Sample 2: predicted QPSK (error)
    # Sample 3: predicted QAM64 (correct family match)
    # Sample 4: predicted 8PSK (error, non-supported class)
    y_pred = ["BPSK", "QPSK", "QAM64", "8PSK"]

    res = calculate_core_metrics(y_true, y_pred)
    # 2 out of 4 correct
    assert res["accuracy"] == 0.5
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert res["per_class"]["QAM"]["recall"] == 1.0
    assert res["per_class"]["WBFM"]["recall"] == 0.0


def test_calculate_confusion_matrix():
    """Verifies confusion matrix generation with OTHER_CLOSED column."""
    y_true = ["BPSK", "QAM", "WBFM"]
    y_pred = ["BPSK", "QAM16", "8PSK"] # 8PSK goes to OTHER_CLOSED

    cm = calculate_confusion_matrix(y_true, y_pred)
    assert cm["rows"] == ["BPSK", "QPSK", "QAM", "WBFM"]
    assert cm["columns"] == ["BPSK", "QPSK", "QAM", "WBFM", "OTHER_CLOSED"]

    matrix = np.array(cm["matrix"])
    assert matrix.shape == (4, 5)
    # Row 0 (BPSK) -> Col 0 (BPSK)
    assert matrix[0, 0] == 1
    # Row 2 (QAM) -> Col 2 (QAM)
    assert matrix[2, 2] == 1
    # Row 3 (WBFM) -> Col 4 (OTHER_CLOSED)
    assert matrix[3, 4] == 1


def test_calculate_snr_breakdown():
    """Verifies SNR-level grouped calculations."""
    y_true = ["BPSK", "BPSK", "QAM", "QAM"]
    y_pred = ["BPSK", "QPSK", "QAM16", "QAM64"]
    snrs = [20, 20, 30, 30]

    res = calculate_snr_breakdown(y_true, y_pred, snrs)
    assert "20" in res
    assert "30" in res
    assert res["20"]["accuracy"] == 0.5
    assert res["30"]["accuracy"] == 1.0


def test_calculate_channel_breakdown():
    """Verifies Clean vs Multipath breakdown and degradation delta."""
    y_true = ["BPSK", "BPSK", "QAM", "QAM"]
    y_pred = ["BPSK", "BPSK", "QPSK", "QPSK"] # Clean 100% correct, multipath 0% correct
    channels = [0, 0, 1, 1]

    res = calculate_channel_breakdown(y_true, y_pred, channels)
    assert res["clean"]["accuracy"] == 1.0
    assert res["multipath"]["accuracy"] == 0.0
    assert res["degradation"]["accuracy_drop_abs"] == 1.0
    assert res["degradation"]["accuracy_drop_pct"] == 100.0


def test_calculate_ece_perfect():
    """Verifies ECE = 0 for perfectly calibrated confidence."""
    # 5 samples with 1.0 confidence and 100% accuracy
    confs = [1.0, 1.0, 1.0, 1.0, 1.0]
    accs = [True, True, True, True, True]

    res = calculate_ece(confs, accs, num_bins=10)
    assert res["ece"] == 0.0
    assert len(res["bins"]) == 10


def test_calculate_ece_uncalibrated():
    """Verifies positive ECE when model is overconfident."""
    # 5 samples with 0.95 confidence but 0% accuracy
    confs = [0.95, 0.95, 0.95, 0.95, 0.95]
    accs = [False, False, False, False, False]

    res = calculate_ece(confs, accs, num_bins=10)
    assert res["ece"] == pytest.approx(0.95, abs=1e-3)
