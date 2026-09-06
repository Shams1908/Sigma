import os
import torch
import pytest
import numpy as np
from scipy.io import wavfile

from ml.input.types import PipelineConfig, BinaryIQConfig
from ml.cnn_model.inference import get_cnn_model
from ml.inference.pipeline import analyze_file

# 1. End-to-end prediction with real checkpoint, sum-to-one, and window counts
def test_end_to_end_real_cnn_prediction(tmp_path):
    wav_path = os.path.join(tmp_path, "signal.wav")
    
    # Generate 256-sample stereo WAV file
    fs = 16000
    n_frames = 256
    i_chan = np.sin(np.linspace(0, 10, n_frames)).astype(np.float32)
    q_chan = np.cos(np.linspace(0, 10, n_frames)).astype(np.float32)
    stereo_data = np.stack([i_chan, q_chan], axis=1)
    wavfile.write(wav_path, fs, stereo_data)
    
    # Process using default model
    result = analyze_file(wav_path)
    
    # Validate SignalAnalysisResult properties
    assert result.processing_status == "SUCCESS"
    assert result.detected_format == "WAV"
    assert result.sample_rate == 16000.0
    assert result.original_sample_count == n_frames
    assert result.predicted_class_index in range(11)
    assert isinstance(result.predicted_class_name, str)
    assert 0.0 <= result.confidence <= 1.0
    
    # Check probabilities sum to 1.0
    probs = np.array(result.probability_vector)
    assert len(probs) == 11
    assert np.all(probs >= 0.0)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
    
    # Check top-k count
    assert len(result.top_predictions) == 5
    assert result.top_predictions[0][0] == result.predicted_class_name
    
    # Check window metrics
    assert result.num_windows == 2
    assert len(result.window_predictions) == 2
    assert len(result.window_confidences) == 2
    assert len(result.window_probabilities) == 2
    
    # Check Shannon entropy is finite
    assert np.isfinite(result.mean_prediction_entropy)

# 2. Determinism check
def test_repeated_inference_determinism(tmp_path):
    npy_path = os.path.join(tmp_path, "signal.npy")
    data = np.random.randn(2, 256).astype(np.float32)
    np.save(npy_path, data)
    
    res1 = analyze_file(npy_path)
    res2 = analyze_file(npy_path)
    
    assert res1.predicted_class_index == res2.predicted_class_index
    assert res1.predicted_class_name == res2.predicted_class_name
    assert np.allclose(res1.confidence, res2.confidence)
    assert np.allclose(res1.probability_vector, res2.probability_vector)
    assert res1.window_predictions == res2.window_predictions
    assert np.allclose(res1.window_confidences, res2.window_confidences)

# 3. Model parameters frozen integrity check
def test_model_integrity_after_inference(tmp_path):
    npy_path = os.path.join(tmp_path, "signal.npy")
    data = np.random.randn(2, 256).astype(np.float32)
    np.save(npy_path, data)
    
    # Load model state dict prior to execution
    model, _ = get_cnn_model("models/m5_iq_cnn.pt")
    state_dict_before = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Run analysis
    _ = analyze_file(npy_path)
    
    # Assert parameters match exactly
    state_dict_after = model.state_dict()
    for k, v in state_dict_before.items():
        assert torch.equal(v, state_dict_after[k])

# 4. Checkpoint Normalization is used (mocked verification)
def test_checkpoint_normalization_used(tmp_path, monkeypatch):
    class MockCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.captured_inputs = []
        def eval(self):
            pass
        def state_dict(self):
            return {}
        def forward(self, x):
            self.captured_inputs.append(x.numpy())
            # Return uniform dummy logits [batch, 11]
            return torch.zeros(x.shape[0], 11)
            
    mock_model = MockCNN()
    
    # Mock get_cnn_model to return custom model and RMS scaling 2.5
    monkeypatch.setattr("ml.inference.pipeline.get_cnn_model", lambda path: (mock_model, 2.5))
    
    # Create NPY file containing value 5.0
    npy_path = os.path.join(tmp_path, "signal.npy")
    data = np.ones((2, 128), dtype=np.float32) * 5.0
    np.save(npy_path, data)
    
    # Run pipeline
    result = analyze_file(npy_path)
    
    # Ensure inputs are normalized by exactly 2.5
    assert len(mock_model.captured_inputs) == 1
    captured = mock_model.captured_inputs[0]
    # Expected: 5.0 / 2.5 = 2.0
    assert np.allclose(captured, 2.0)
    assert result.normalization_source == "checkpoint training RMS"

# 5. Invalid files and too-short inputs
def test_error_handling_and_bounds(tmp_path):
    from ml.cnn_model.inference import clear_cnn_cache
    clear_cnn_cache()
    
    # 1. Missing checkpoint path
    npy_path = os.path.join(tmp_path, "signal.npy")
    np.save(npy_path, np.random.randn(2, 128).astype(np.float32))
    
    with pytest.raises(FileNotFoundError):
        analyze_file(npy_path, model_path="models/missing_cnn.pt")
        
    # 2. Too short signal
    short_path = os.path.join(tmp_path, "short.npy")
    np.save(short_path, np.random.randn(2, 50).astype(np.float32))
    
    with pytest.raises(ValueError, match="shorter than the expected segment length"):
        analyze_file(short_path)
