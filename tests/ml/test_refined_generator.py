import pytest
import numpy as np
from ml.generators.config import GeneratorConfig
from ml.generators.modulation.interface import generate_modulated_signal
from ml.generators.signal import SyntheticGroundTruth

# 1. Configuration Validation and backward compatibility
def test_validation_refined_generator_config():
    # Backward compatibility: old parameter sets load without error
    cfg_old = GeneratorConfig(
        modulation="BPSK",
        num_symbols=8,
        sample_rate=800000.0,
        symbol_rate=100000.0,
        samples_per_symbol=8,
        random_seed=42
    )
    assert cfg_old.phase_noise_std is None
    assert cfg_old.channel_taps is None
    
    # Valid new configuration
    cfg_new = GeneratorConfig(
        modulation="QPSK",
        num_symbols=8,
        sample_rate=800000.0,
        symbol_rate=100000.0,
        samples_per_symbol=8,
        random_seed=42,
        phase_noise_std=0.02,
        channel_taps=[1.0, 0.5, 0.25]
    )
    assert cfg_new.phase_noise_std == 0.02
    assert cfg_new.channel_taps == [1.0, 0.5, 0.25]
    
    # Invalid type checking
    with pytest.raises((TypeError, ValueError)):
        GeneratorConfig(
            modulation="QPSK",
            num_symbols=8,
            sample_rate=800000.0,
            symbol_rate=100000.0,
            samples_per_symbol=8,
            random_seed=42,
            phase_noise_std=-0.01 # negative std is invalid
        )
        
    with pytest.raises((TypeError, ValueError)):
        GeneratorConfig(
            modulation="QPSK",
            num_symbols=8,
            sample_rate=800000.0,
            symbol_rate=100000.0,
            samples_per_symbol=8,
            random_seed=42,
            channel_taps="invalid_string" # taps must be list/array
        )

# 2. No-op behavior when new effects are disabled
def test_no_op_behavior_when_disabled():
    cfg_disabled = GeneratorConfig(
        modulation="BPSK",
        num_symbols=8,
        sample_rate=800000.0,
        symbol_rate=100000.0,
        samples_per_symbol=8,
        rolloff=0.35,
        filter_span_symbols=8,
        random_seed=42
    )
    
    cfg_none = GeneratorConfig(
        modulation="BPSK",
        num_symbols=8,
        sample_rate=800000.0,
        symbol_rate=100000.0,
        samples_per_symbol=8,
        rolloff=0.35,
        filter_span_symbols=8,
        random_seed=42,
        phase_noise_std=None,
        channel_taps=None
    )
    
    bits = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    sig_disabled = generate_modulated_signal(bits, cfg_disabled)
    sig_none = generate_modulated_signal(bits, cfg_none)
    
    # Signals must be identical
    assert np.array_equal(sig_disabled.samples, sig_none.samples)

# 3. Deterministic Phase Noise, Shape and Dtype Preservation
def test_deterministic_phase_noise():
    cfg1 = GeneratorConfig(
        modulation="QPSK",
        num_symbols=8,
        sample_rate=800000.0,
        symbol_rate=100000.0,
        samples_per_symbol=8,
        rolloff=0.35,
        filter_span_symbols=8,
        random_seed=42,
        phase_noise_std=0.05
    )
    cfg2 = GeneratorConfig(
        modulation="QPSK",
        num_symbols=8,
        sample_rate=800000.0,
        symbol_rate=100000.0,
        samples_per_symbol=8,
        rolloff=0.35,
        filter_span_symbols=8,
        random_seed=42,
        phase_noise_std=0.05
    )
    cfg_different_seed = GeneratorConfig(
        modulation="QPSK",
        num_symbols=8,
        sample_rate=800000.0,
        symbol_rate=100000.0,
        samples_per_symbol=8,
        rolloff=0.35,
        filter_span_symbols=8,
        random_seed=43, # different seed
        phase_noise_std=0.05
    )
    
    bits = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]) # QPSK needs 16 bits for 8 symbols
    sig1 = generate_modulated_signal(bits, cfg1)
    sig2 = generate_modulated_signal(bits, cfg2)
    sig_diff = generate_modulated_signal(bits, cfg_different_seed)
    
    # Confirm shape/dtype/finite invariants
    assert sig1.samples.shape == (2, 128)
    assert sig1.samples.dtype == np.float32
    assert np.isfinite(sig1.samples).all()
    
    # Determinism: same seed yields identical signal
    assert np.array_equal(sig1.samples, sig2.samples)
    
    # Seed Variation: different seed yields different signal
    assert not np.array_equal(sig1.samples, sig_diff.samples)

# 4. Channel Response Behavior (FIR Convolution)
def test_channel_response_behavior():
    cfg_no_filter = GeneratorConfig(
        modulation="BPSK",
        num_symbols=8,
        sample_rate=800000.0,
        symbol_rate=100000.0,
        samples_per_symbol=8,
        rolloff=0.35,
        filter_span_symbols=8,
        random_seed=42
    )
    
    # Mild channel filter taps
    taps = [1.0, 0.5, 0.2]
    cfg_filter = GeneratorConfig(
        modulation="BPSK",
        num_symbols=8,
        sample_rate=800000.0,
        symbol_rate=100000.0,
        samples_per_symbol=8,
        rolloff=0.35,
        filter_span_symbols=8,
        random_seed=42,
        channel_taps=taps
    )
    
    bits = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    sig_no_filter = generate_modulated_signal(bits, cfg_no_filter)
    sig_filter = generate_modulated_signal(bits, cfg_filter)
    
    # Content must differ due to channel convolution
    assert not np.array_equal(sig_no_filter.samples, sig_filter.samples)
    assert sig_filter.samples.shape == sig_no_filter.samples.shape
    
    # Metadata preservation
    assert sig_filter.metadata.channel_taps == taps

# 5. Metadata Preservation
def test_metadata_preservation():
    taps = [1.0, 0.1]
    cfg = GeneratorConfig(
        modulation="BPSK",
        num_symbols=8,
        sample_rate=800000.0,
        symbol_rate=100000.0,
        samples_per_symbol=8,
        rolloff=0.35,
        filter_span_symbols=8,
        random_seed=42,
        phase_noise_std=0.03,
        channel_taps=taps
    )
    
    bits = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    sig = generate_modulated_signal(bits, cfg)
    
    assert sig.metadata.phase_noise_std == 0.03
    assert sig.metadata.channel_taps == taps
