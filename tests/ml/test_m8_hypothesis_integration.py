"""
Integration unit tests for M8 Hypothesis Adapter (ml/parameters/hypothesis_adapter.py).
"""
import pytest
import numpy as np

from ml.parameters.inference import (
    ParameterAnalysisResult,
    SingleParameterResult,
)
from ml.parameters.hypothesis_adapter import generate_hypotheses_with_parameters


def test_generate_hypotheses_with_parameters():
    """Verifies that M8 parameters enrich candidate hypotheses without mutating scoring logic."""
    sr_res = SingleParameterResult(
        estimate=100000.0,
        confidence=0.85,
        uncertainty=250.0,
        method="hybrid_consensus",
    )
    snr_res = SingleParameterResult(
        estimate=22.5,
        confidence=0.90,
        uncertainty=1.2,
        method="hybrid_consensus",
    )
    param_res = ParameterAnalysisResult(
        symbol_rate=sr_res,
        snr_db=snr_res,
        sample_rate=800000.0,
        signal_length_samples=1024,
        duration_ms=1.28,
    )

    ml_probs = {"BPSK": 0.82, "QPSK": 0.12, "8PSK": 0.04}
    candidates = generate_hypotheses_with_parameters(
        ml_probabilities=ml_probs,
        parameter_result=param_res,
        supported_classes=["BPSK", "QPSK", "8PSK"],
        top_k=2,
    )

    assert len(candidates) == 2
    top_cand = candidates[0]
    assert top_cand.modulation == "BPSK"
    assert top_cand.symbolRate == 100000.0
    assert top_cand.sync_assumptions is not None
    assert top_cand.sync_assumptions["estimated_symbol_rate_baud"] == 100000.0
    assert top_cand.sync_assumptions["estimated_snr_db"] == 22.5
    assert top_cand.evidence.ml.status.value == "available"
