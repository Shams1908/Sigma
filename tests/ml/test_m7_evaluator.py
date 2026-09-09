"""
Integration unit tests for M7ModelEvaluator and End-to-End Evaluation Pipeline (ml/evaluation/evaluator.py).
"""
import pytest
import numpy as np
import torch

from ml.dataset.external_dataset import ExternalDatasetSplit
from ml.evaluation.evaluator import M7ModelEvaluator


@pytest.fixture
def mock_external_split():
    """Generates a small deterministic 14-sample split (2 samples per modulation)."""
    rng = np.random.default_rng(42)
    # 14 frames of [1024, 2]
    X = rng.standard_normal((14, 1024, 2)).astype(np.float32)
    y_mod = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype=np.int64)
    y_chan = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    y_snr = np.array([20, 22, 24, 26, 28, 30, 20, 22, 24, 26, 28, 30, 20, 22], dtype=np.int64)
    return ExternalDatasetSplit(
        X=X,
        y_mod=y_mod,
        y_chan=y_chan,
        y_snr=y_snr,
        provenance="MOCK-INTEGRATION-TEST",
        is_evaluation_set=True,
    )


def test_evaluator_initialization():
    """Verifies that M5 and M6 models load in eval mode."""
    evaluator = M7ModelEvaluator()
    assert isinstance(evaluator.model_m5, torch.nn.Module)
    assert isinstance(evaluator.model_m6, torch.nn.Module)
    assert evaluator.model_m5.training is False
    assert evaluator.model_m6.training is False
    assert evaluator.rms_m5 > 0.0
    assert evaluator.rms_m6 > 0.0


def test_evaluator_run_mock_split(mock_external_split):
    """Verifies end-to-end evaluation execution and output schema."""
    evaluator = M7ModelEvaluator()

    # Generate an independent dev split with a distinct seed to avoid split contamination
    rng_dev = np.random.default_rng(999)
    X_dev = rng_dev.standard_normal((14, 1024, 2)).astype(np.float32)
    dev_split = ExternalDatasetSplit(
        X=X_dev,
        y_mod=mock_external_split.y_mod.copy(),
        y_chan=mock_external_split.y_chan.copy(),
        y_snr=mock_external_split.y_snr.copy(),
        provenance="MOCK-DEV-SPLIT",
        is_evaluation_set=True,
    )

    results = evaluator.evaluate_split(
        split=mock_external_split,
        dev_split_for_tuning=dev_split,
        fit_temperature_scaling=True,
    )

    # Core schema assertions
    assert "status" in results
    assert "evaluation_data_type" in results
    assert "dataset_integrity" in results
    assert "models" in results
    assert "m5" in results["models"]
    assert "m6" in results["models"]
    assert "comparison" in results
    assert "hypothesis_engine_interface" in results

    # Checkpoint immutability
    assert evaluator.model_m5.training is False
    assert evaluator.model_m6.training is False

    # Metrics present for both models
    for m_key in ["m5", "m6"]:
        m = results["models"][m_key]
        assert "frame_level_metrics" in m
        assert "window_level_metrics" in m
        assert "confusion_matrix" in m
        assert "snr_breakdown" in m
        assert "channel_breakdown" in m
        assert "confidence_analysis" in m
        assert "open_set_rejection" in m
        assert "latency" in m

        # Check values are floats between 0 and 1
        assert 0.0 <= m["frame_level_metrics"]["accuracy"] <= 1.0
        assert 0.0 <= m["frame_level_metrics"]["macro_f1"] <= 1.0

    # Comparison deltas exist
    assert "delta_m6_minus_m5" in results["comparison"]["frame_level_accuracy"]
    assert "delta_m6_minus_m5" in results["comparison"]["frame_level_macro_f1"]
