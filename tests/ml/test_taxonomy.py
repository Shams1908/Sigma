"""
Unit tests for ml/dataset/taxonomy.py  (M6 taxonomy alignment).
"""
from __future__ import annotations

import pytest

from ml.dataset.taxonomy import (
    SIGMA_TO_EXTERNAL,
    EXTERNAL_TO_SIGMA,
    EXTERNAL_UNSUPPORTED_BY_SIGMA,
    SIGMA_UNSUPPORTED_BY_EXTERNAL,
    sigma_to_external,
    external_to_sigma_candidates,
    is_external_class_supported,
    OpenSetClassifier,
    OpenSetResult,
)
from ml.dataset.labels import MODULATION_CLASSES
from ml.dataset.external_dataset import EXTERNAL_CLASSES


# ── 1. SIGMA → External mapping ───────────────────────────────────────────────

def test_exact_overlaps():
    assert sigma_to_external("BPSK") == "BPSK"
    assert sigma_to_external("QPSK") == "QPSK"
    assert sigma_to_external("WBFM") == "WBFM"


def test_qam_family_mapping():
    assert sigma_to_external("QAM16") == "QAM"
    assert sigma_to_external("QAM64") == "QAM"


def test_sigma_unsupported_classes_return_none():
    for label in ("8PSK", "AM-DSB", "AM-SSB", "CPFSK", "GFSK", "PAM4"):
        assert sigma_to_external(label) is None


def test_sigma_to_external_all_classes_covered():
    for label in MODULATION_CLASSES:
        # Must not raise
        result = sigma_to_external(label)
        assert result is None or result in EXTERNAL_CLASSES


def test_sigma_to_external_invalid():
    with pytest.raises(ValueError):
        sigma_to_external("NONEXISTENT")


# ── 2. External → SIGMA mapping ───────────────────────────────────────────────

def test_external_to_sigma_bpsk():
    assert external_to_sigma_candidates("BPSK") == ["BPSK"]


def test_external_to_sigma_qpsk():
    assert external_to_sigma_candidates("QPSK") == ["QPSK"]


def test_external_to_sigma_qam_family():
    candidates = external_to_sigma_candidates("QAM")
    assert set(candidates) == {"QAM16", "QAM64"}


def test_external_to_sigma_wbfm():
    assert external_to_sigma_candidates("WBFM") == ["WBFM"]


def test_external_unsupported_classes_return_empty():
    for label in ("GMSK", "OFDM", "NBFM"):
        assert external_to_sigma_candidates(label) == []


def test_external_to_sigma_invalid():
    with pytest.raises(ValueError):
        external_to_sigma_candidates("NOPE")


# ── 3. Support checks ─────────────────────────────────────────────────────────

def test_is_external_class_supported_true():
    for label in ("BPSK", "QPSK", "QAM", "WBFM"):
        assert is_external_class_supported(label) is True


def test_is_external_class_supported_false():
    for label in ("GMSK", "OFDM", "NBFM"):
        assert is_external_class_supported(label) is False


# ── 4. Unsupported class lists ────────────────────────────────────────────────

def test_external_unsupported_by_sigma_list():
    assert set(EXTERNAL_UNSUPPORTED_BY_SIGMA) == {"GMSK", "OFDM", "NBFM"}


def test_sigma_unsupported_by_external_list():
    for label in SIGMA_UNSUPPORTED_BY_EXTERNAL:
        assert sigma_to_external(label) is None


# ── 5. OpenSetClassifier ──────────────────────────────────────────────────────

def test_open_set_supported_class():
    osc = OpenSetClassifier()
    result = osc.classify("BPSK", ground_truth_external="BPSK")
    assert isinstance(result, OpenSetResult)
    assert result.is_supported is True
    assert result.rejection_reason is None
    assert result.external_equivalent == "BPSK"


def test_open_set_unsupported_class():
    osc = OpenSetClassifier()
    result = osc.classify("BPSK", ground_truth_external="OFDM")
    assert result.is_supported is False
    assert result.rejection_reason is not None
    assert "OFDM" in result.rejection_reason


def test_open_set_gmsk_unsupported():
    osc = OpenSetClassifier()
    for ext_cls in ("GMSK", "OFDM", "NBFM"):
        result = osc.classify("QPSK", ground_truth_external=ext_cls)
        assert result.is_supported is False


def test_open_set_no_ground_truth_defaults_to_in_set():
    osc = OpenSetClassifier()
    result = osc.classify("QPSK", ground_truth_external=None)
    assert result.is_supported is True
    assert result.ground_truth_external is None


def test_open_set_invalid_ground_truth():
    osc = OpenSetClassifier()
    with pytest.raises(ValueError):
        osc.classify("BPSK", ground_truth_external="INVALID_CLASS")


def test_open_set_result_repr():
    osc = OpenSetClassifier()
    result = osc.classify("BPSK", ground_truth_external="BPSK")
    r = repr(result)
    assert "BPSK" in r
    assert "is_supported=True" in r


# ── 6. Coverage — all external classes go through classify ───────────────────

def test_all_external_classes_classifiable():
    osc = OpenSetClassifier()
    for ext_cls in EXTERNAL_CLASSES:
        result = osc.classify("BPSK", ground_truth_external=ext_cls)
        assert isinstance(result, OpenSetResult)


# ── 7. Mapping consistency ────────────────────────────────────────────────────

def test_sigma_to_external_dict_covers_all_sigma_classes():
    for label in MODULATION_CLASSES:
        assert label in SIGMA_TO_EXTERNAL


def test_external_to_sigma_dict_covers_all_external_classes():
    for label in EXTERNAL_CLASSES:
        assert label in EXTERNAL_TO_SIGMA
