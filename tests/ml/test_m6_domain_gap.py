"""
Unit tests for M6 Domain-Gap Statistical Audit (domain_gap_audit.py).
Verifies:
  - Statistical metric calculations (amplitude, phase, spectral, temporal)
  - Mathematical correctness of PAPR, IQ imbalance, and autocorrelation
  - Existence and structural integrity of exported audit reports
"""
import os
import json
import pytest
import numpy as np

from ml.synthetic.domain_gap_audit import compute_comprehensive_statistics


def test_compute_comprehensive_statistics_keys_and_values():
    """Verifies that all required metrics are present and numerically finite."""
    rng = np.random.default_rng(42)
    # Generate 10 test signals of shape [10, 2, 128]
    test_signals = rng.standard_normal((10, 2, 128)).astype(np.float32)

    stats = compute_comprehensive_statistics(test_signals)

    expected_keys = [
        "complex_rms",
        "signal_power",
        "power_std",
        "amplitude_mean",
        "amplitude_std",
        "amplitude_skewness",
        "amplitude_kurtosis",
        "papr_db",
        "phase_variance",
        "phase_diff_variance",
        "inst_freq_mean",
        "inst_freq_kurtosis",
        "iq_amplitude_imbalance_db",
        "iq_orthogonality_corr",
        "dc_offset_i",
        "dc_offset_q",
        "total_dc_offset",
        "spectral_flatness",
        "occupied_bw_ratio",
        "autocorr_lag1",
        "autocorr_lag2",
    ]

    for k in expected_keys:
        assert k in stats, f"Missing metric key: {k}"
        assert isinstance(stats[k], float)
        assert np.isfinite(stats[k]), f"Non-finite metric value for {k}: {stats[k]}"


def test_papr_and_autocorr_math():
    """Verifies PAPR and autocorrelation on known deterministic signals."""
    # Constant envelope pure tone: cos(wt) + j*sin(wt) -> PAPR should be close to 0 dB
    t = np.linspace(0, 2 * np.pi * 4, 128, endpoint=False)
    I = np.cos(t).reshape(1, 1, 128)
    Q = np.sin(t).reshape(1, 1, 128)
    tone = np.concatenate([I, Q], axis=1).astype(np.float32)

    stats = compute_comprehensive_statistics(tone)
    # Peak amplitude is 1.0, mean power is 1.0 -> PAPR in dB is ~ 0.0
    assert np.isclose(stats["papr_db"], 0.0, atol=0.2)
    # High lag-1 temporal correlation for smooth tone
    assert stats["autocorr_lag1"] > 0.8


def test_domain_gap_audit_report_artifacts_exist():
    """Verifies that audit JSON and Markdown files were created with valid schemas."""
    paths_to_check = [
        "results/ml/m6/domain_gap_audit.json",
        "results/ml/m6/domain_gap_audit.md",
        "reports/m6_domain_gap.json",
        "reports/m6_domain_gap.md",
    ]

    for p in paths_to_check:
        assert os.path.exists(p), f"Report file missing: {p}"

    # Verify JSON structure
    with open("results/ml/m6/domain_gap_audit.json", "r") as f:
        data = json.load(f)

    assert "provenance" in data
    assert "root_causes_identified" in data
    assert "metrics_summary" in data
    assert len(data["root_causes_identified"]) > 0
    assert len(data["metrics_summary"]) >= 20
