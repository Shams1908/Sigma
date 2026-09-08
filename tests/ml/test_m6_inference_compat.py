"""
Unit tests for M6 Inference Compatibility and Model Checkpoints.
Verifies:
  - Frozen M5 checkpoint (models/m5_iq_cnn.pt) loads with in_channels=2 and RAW_IQ
  - M6 checkpoint (models/m6_robust_cnn.pt) loads with in_channels=3 and IQ_AMPLITUDE
  - predict_iq compatibility across single-sample and batch modes
  - analyze_file end-to-end execution with both M5 and M6 checkpoints
  - Numerical safety and sum-to-one probabilities
"""
import os
import pytest
import numpy as np
from scipy.io import wavfile

from ml.cnn_model.inference import get_cnn_model, predict_iq, clear_cnn_cache
from ml.inference.pipeline import analyze_file


@pytest.fixture(autouse=True)
def clean_cache():
    """Ensures model cache is cleared before each test."""
    clear_cnn_cache()
    yield
    clear_cnn_cache()


def test_m5_checkpoint_metadata_and_dimensions():
    """Verifies that M5 checkpoint remains frozen with in_channels=2."""
    model_path = "models/m5_iq_cnn.pt"
    if not os.path.exists(model_path):
        pytest.skip(f"{model_path} not found")

    model, rms = get_cnn_model(model_path)
    assert model.in_channels == 2
    assert model.num_classes == 11
    assert model.representation == "RAW_IQ"
    assert rms > 0.0
    assert model.block1[0].in_channels == 2


def test_m6_checkpoint_metadata_and_dimensions():
    """Verifies that M6 checkpoint loads with in_channels=3 and IQ_AMPLITUDE."""
    model_path = "models/m6_robust_cnn.pt"
    if not os.path.exists(model_path):
        pytest.skip(f"{model_path} not found")

    model, rms = get_cnn_model(model_path)
    assert model.in_channels == 3
    assert model.num_classes == 11
    assert model.representation == "IQ_AMPLITUDE"
    assert rms > 0.0
    assert model.block1[0].in_channels == 3


def test_predict_iq_with_m5_and_m6():
    """Verifies predict_iq works seamlessly with both 2-channel and 3-channel models."""
    rng = np.random.default_rng(42)
    single_iq = rng.standard_normal((2, 128)).astype(np.float32)
    batch_iq = rng.standard_normal((4, 2, 128)).astype(np.float32)

    # 1. M5 Baseline
    m5_path = "models/m5_iq_cnn.pt"
    if os.path.exists(m5_path):
        res_m5_single = predict_iq(single_iq, model_path=m5_path)
        assert isinstance(res_m5_single["class_index"], int)
        assert isinstance(res_m5_single["confidence"], float)
        assert np.isclose(np.sum(res_m5_single["probabilities"]), 1.0, atol=1e-5)

        res_m5_batch = predict_iq(batch_iq, model_path=m5_path)
        assert len(res_m5_batch["class_index"]) == 4
        assert res_m5_batch["probabilities"].shape == (4, 11)

    # 2. M6 Candidate
    m6_path = "models/m6_robust_cnn.pt"
    if os.path.exists(m6_path):
        # 2-channel input should be automatically expanded to 3 channels by predict_iq
        res_m6_single = predict_iq(single_iq, model_path=m6_path)
        assert isinstance(res_m6_single["class_index"], int)
        assert np.isclose(np.sum(res_m6_single["probabilities"]), 1.0, atol=1e-5)

        res_m6_batch = predict_iq(batch_iq, model_path=m6_path)
        assert len(res_m6_batch["class_index"]) == 4
        assert res_m6_batch["probabilities"].shape == (4, 11)


def test_analyze_file_with_m5_and_m6(tmp_path):
    """Verifies analyze_file end-to-end with WAV file for both M5 and M6 checkpoints."""
    wav_path = os.path.join(tmp_path, "test_signal.wav")
    fs = 16000
    n_frames = 256
    i_chan = np.sin(np.linspace(0, 20, n_frames)).astype(np.float32)
    q_chan = np.cos(np.linspace(0, 20, n_frames)).astype(np.float32)
    stereo = np.stack([i_chan, q_chan], axis=1)
    wavfile.write(wav_path, fs, stereo)

    # Test M5
    m5_path = "models/m5_iq_cnn.pt"
    if os.path.exists(m5_path):
        res_m5 = analyze_file(wav_path, model_path=m5_path)
        assert res_m5.processing_status == "SUCCESS"
        assert res_m5.num_windows == 2
        assert len(res_m5.probability_vector) == 11
        assert np.isclose(sum(res_m5.probability_vector), 1.0, atol=1e-5)

    # Test M6
    m6_path = "models/m6_robust_cnn.pt"
    if os.path.exists(m6_path):
        res_m6 = analyze_file(wav_path, model_path=m6_path)
        assert res_m6.processing_status == "SUCCESS"
        assert res_m6.num_windows == 2
        assert len(res_m6.probability_vector) == 11
        assert np.isclose(sum(res_m6.probability_vector), 1.0, atol=1e-5)
