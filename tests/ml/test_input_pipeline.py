import os
import pytest
import numpy as np
from scipy.io import wavfile

from ml.input.types import PipelineConfig, BinaryIQConfig
from ml.input.pipeline import process_file, detect_format, load_signal, segment_iq

# 1. Format Detection and Validation tests
def test_format_detection_and_validation(tmp_path):
    # Setup test paths
    wav_path = os.path.join(tmp_path, "signal.wav")
    npy_path = os.path.join(tmp_path, "signal.npy")
    npz_path = os.path.join(tmp_path, "signal.npz")
    bin_path = os.path.join(tmp_path, "signal.bin")
    txt_path = os.path.join(tmp_path, "signal.txt")
    
    # 1. Nonexistent path
    with pytest.raises(FileNotFoundError):
        detect_format(os.path.join(tmp_path, "missing.wav"))
        
    # 2. Unsupported extension
    with open(txt_path, "w") as f:
        f.write("hello")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        detect_format(txt_path)
        
    # 3. Oversized file
    with open(bin_path, "wb") as f:
        f.write(b"\x00" * 1024)
    # Configure max size as 500 bytes
    with pytest.raises(ValueError, match="exceeds the configured maximum upload limit"):
        detect_format(bin_path, max_size_bytes=500)
        
    # 4. Malformed WAV header signature checks
    bad_wav_path = os.path.join(tmp_path, "bad.wav")
    with open(bad_wav_path, "wb") as f:
        f.write(b"NOT_A_WAV_HEADER_BYTES")
    with pytest.raises(ValueError, match="Malformed WAV file"):
        detect_format(bad_wav_path)
        
    # 5. Malformed NPY header checks
    bad_npy_path = os.path.join(tmp_path, "bad.npy")
    with open(bad_npy_path, "wb") as f:
        f.write(b"NOT_NPY")
    with pytest.raises(ValueError, match="Malformed NPY file"):
        detect_format(bad_npy_path)

# 2. WAV Parser tests (Valid Stereo, I/Q Ordering, Mono Failures, Invalid bit-widths)
def test_wav_parsing(tmp_path):
    wav_path = os.path.join(tmp_path, "valid_stereo.wav")
    mono_path = os.path.join(tmp_path, "mono.wav")
    
    # Create valid stereo WAV
    # scipy expects shape [N, 2] for stereo
    fs = 16000
    n_frames = 256
    i_channel = np.linspace(-0.5, 0.5, n_frames, dtype=np.float32)
    q_channel = np.linspace(0.5, -0.5, n_frames, dtype=np.float32)
    stereo_data = np.stack([i_channel, q_channel], axis=1) # Shape [256, 2]
    
    wavfile.write(wav_path, fs, stereo_data)
    
    # Process stereo WAV
    segments, meta = process_file(wav_path)
    assert meta.validation_status == "OK"
    assert meta.detected_format == "WAV"
    assert meta.sample_rate == 16000.0
    assert meta.num_channels == 2
    assert meta.original_sample_count == n_frames
    assert meta.canonical_iq_shape == [2, n_frames]
    assert meta.num_generated_segments == 2
    assert segments.shape == (2, 2, 128)
    
    # Load raw signal to check I/Q ordering: channel 0 -> I, channel 1 -> Q
    raw_iq, sr, chans = load_signal(wav_path, "WAV")
    assert np.allclose(raw_iq[0], i_channel)
    assert np.allclose(raw_iq[1], q_channel)
    
    # Create mono WAV file
    mono_data = np.random.randn(n_frames).astype(np.float32)
    wavfile.write(mono_path, fs, mono_data)
    
    # Process mono WAV and assert error
    _, err_meta = process_file(mono_path)
    assert err_meta.validation_status == "ERROR"
    assert "mono" in err_meta.error_message.lower()

# 3. IQ Layout Parsing tests ([2, N], [N, 2], complex [N], NPZ parsing)
def test_npy_and_npz_layouts(tmp_path):
    # Shape [2, 300]
    npy_2xN = os.path.join(tmp_path, "signal_2xN.npy")
    data_2xN = np.random.randn(2, 300).astype(np.float32)
    np.save(npy_2xN, data_2xN)
    
    segments, meta = process_file(npy_2xN)
    assert meta.validation_status == "OK"
    assert meta.detected_format == "NPY"
    assert meta.canonical_iq_shape == [2, 300]
    assert segments.shape == (2, 2, 128) # Discard remaining 44 samples
    
    # Shape [300, 2]
    npy_Nx2 = os.path.join(tmp_path, "signal_Nx2.npy")
    data_Nx2 = np.random.randn(300, 2).astype(np.float32)
    np.save(npy_Nx2, data_Nx2)
    
    segments, meta = process_file(npy_Nx2)
    assert meta.validation_status == "OK"
    assert meta.canonical_iq_shape == [2, 300]
    
    # Complex [300]
    npy_complex = os.path.join(tmp_path, "signal_complex.npy")
    data_complex = (np.random.randn(300) + 1j * np.random.randn(300)).astype(np.complex64)
    np.save(npy_complex, data_complex)
    
    segments, meta = process_file(npy_complex)
    assert meta.validation_status == "OK"
    assert meta.canonical_iq_shape == [2, 300]
    
    # NPZ archiving parsing
    npz_path = os.path.join(tmp_path, "signal.npz")
    np.savez(npz_path, X=data_2xN)
    
    segments, meta = process_file(npz_path)
    assert meta.validation_status == "OK"
    assert meta.detected_format == "NPZ"
    assert meta.canonical_iq_shape == [2, 300]

# 4. Binary IQ parsing tests (interleaved, non-interleaved, missing configs)
def test_binary_iq_parsing(tmp_path):
    bin_path = os.path.join(tmp_path, "signal.bin")
    
    # Interleaved float32 samples: I0, Q0, I1, Q1... (total 400 float32s = 200 complex samples)
    data = np.linspace(-1.0, 1.0, 400, dtype=np.float32)
    data.tofile(bin_path)
    
    # 1. No configuration details provided
    _, err_meta = process_file(bin_path)
    assert err_meta.validation_status == "ERROR"
    assert "binary_config" in err_meta.error_message.lower()
    
    # 2. Configured correctly
    cfg = PipelineConfig(
        binary_config=BinaryIQConfig(dtype="float32", interleaved=True, endianness="little")
    )
    segments, meta = process_file(bin_path, cfg)
    assert meta.validation_status == "OK"
    assert meta.detected_format == "BIN"
    assert meta.canonical_iq_shape == [2, 200]
    assert segments.shape == (1, 2, 128) # segments count = 1
    
    # 3. Non-interleaved config
    cfg_non = PipelineConfig(
        binary_config=BinaryIQConfig(dtype="float32", interleaved=False, endianness="little")
    )
    segments, meta = process_file(bin_path, cfg_non)
    assert meta.validation_status == "OK"
    # Verify shape
    assert meta.canonical_iq_shape == [2, 200]

# 5. Finite / NaN / Inf checks
def test_nan_inf_validation_handling(tmp_path):
    npy_nan = os.path.join(tmp_path, "nan.npy")
    data = np.random.randn(2, 256).astype(np.float32)
    data[0, 10] = np.nan
    np.save(npy_nan, data)
    
    _, err_meta = process_file(npy_nan)
    assert err_meta.validation_status == "ERROR"
    assert "nan" in err_meta.error_message.lower()

# 6. Segmentation checks (short signal padding, custom segment length)
def test_segmentation_edge_cases():
    # Signal shorter than 128, pad_short = False
    iq_short = np.random.randn(2, 50).astype(np.float32)
    
    with pytest.raises(ValueError, match="shorter than the expected segment length"):
        segment_iq(iq_short, segment_length=128, pad_short=False)
        
    # pad_short = True
    padded = segment_iq(iq_short, segment_length=128, pad_short=True)
    assert padded.shape == (1, 2, 128)
    assert np.all(padded[0, :, 50:] == 0.0) # Check zero padding
    
    # Custom segment size
    iq_long = np.random.randn(2, 500).astype(np.float32)
    segments = segment_iq(iq_long, segment_length=150)
    assert segments.shape == (3, 2, 150) # 500 // 150 = 3 segments
