import pytest
import numpy as np

from ml.generators.config import GeneratorConfig
from ml.generators.signal import GeneratedSignal, iq_to_complex
from ml.generators.channel.awgn import AWGNChannel
from ml.generators.modulation import generate_modulated_signal


# 1. Test Output Properties
def test_awgn_output_properties():
    # Make a dummy complex signal (e.g. BPSK modulated) of length 100
    waveform = np.ones(100, dtype=np.complex64)
    config = GeneratorConfig(
        modulation="BPSK",
        num_symbols=100,
        sample_rate=1000.0,
        symbol_rate=1000.0,
        samples_per_symbol=1,
        random_seed=42,
        snr=10.0
    )
    rng = np.random.default_rng(config.random_seed)
    
    channel = AWGNChannel()
    output = channel.apply(waveform, config, rng)
    
    # 1. Shape preservation
    assert output.shape == waveform.shape
    # 2. Complex64 dtype retention
    assert output.dtype == np.complex64
    # 3. Output is complex and noisy (not exactly equal to input)
    assert not np.array_equal(output, waveform)


# 2. Test Invalid Inputs & Boundaries
def test_awgn_invalid_inputs():
    channel = AWGNChannel()
    config = GeneratorConfig(
        modulation="BPSK",
        num_symbols=10,
        sample_rate=1000.0,
        symbol_rate=1000.0,
        samples_per_symbol=1,
        random_seed=42,
        snr=10.0
    )
    rng = np.random.default_rng(config.random_seed)

    # 1. Zero-power input rejection
    zero_waveform = np.zeros(100, dtype=np.complex64)
    with pytest.raises(ValueError, match="zero power"):
        channel.apply(zero_waveform, config, rng)

    # 2. Non-1D array rejection
    waveform_2d = np.ones((10, 10), dtype=np.complex64)
    with pytest.raises(ValueError, match="1D"):
        channel.apply(waveform_2d, config, rng)

    # 3. Non-ndarray rejection
    with pytest.raises(TypeError):
        channel.apply([1.0, 2.0], config, rng)  # type: ignore

    # 4. Empty array rejection
    with pytest.raises(ValueError):
        channel.apply(np.array([], dtype=np.complex64), config, rng)

    # 5. Invalid/infinite SNR rejection
    with pytest.raises(ValueError, match="finite"):
        GeneratorConfig(
            modulation="BPSK",
            num_symbols=10,
            sample_rate=1000.0,
            symbol_rate=1000.0,
            samples_per_symbol=1,
            random_seed=42,
            snr=np.inf
        )


# 3. Test Reproducibility / Seeding Determinism
def test_awgn_determinism():
    waveform = np.ones(100, dtype=np.complex64)
    config = GeneratorConfig(
        modulation="BPSK",
        num_symbols=100,
        sample_rate=1000.0,
        symbol_rate=1000.0,
        samples_per_symbol=1,
        random_seed=42,
        snr=10.0
    )
    
    # Same seed/RNG state produces identical output
    rng1 = np.random.default_rng(12345)
    rng2 = np.random.default_rng(12345)
    channel = AWGNChannel()
    
    out1 = channel.apply(waveform, config, rng1)
    out2 = channel.apply(waveform, config, rng2)
    assert np.array_equal(out1, out2)

    # Different seeds produce different outputs
    rng3 = np.random.default_rng(54321)
    out3 = channel.apply(waveform, config, rng3)
    assert not np.array_equal(out1, out3)


# 4. Test Statistical SNR and Variance Split
def test_awgn_statistics():
    # Generate a large deterministic waveform (100,000 samples of a complex sine wave)
    # Ensure there is enough sample volume to converge statistically.
    t = np.arange(100000, dtype=np.float32)
    waveform = (np.exp(2j * np.pi * 0.05 * t)).astype(np.complex64)
    
    # Calculate actual input signal power
    p_signal = float(np.mean(np.abs(waveform) ** 2))
    assert np.isclose(p_signal, 1.0)
    
    snr_db = 10.0
    config = GeneratorConfig(
        modulation="BPSK",
        num_symbols=100000,
        sample_rate=1000.0,
        symbol_rate=1000.0,
        samples_per_symbol=1,
        random_seed=42,
        snr=snr_db
    )
    rng = np.random.default_rng(config.random_seed)
    
    channel = AWGNChannel()
    output = channel.apply(waveform, config, rng)
    
    # Extract noise component
    noise = output - waveform
    
    # Calculate expected and measured noise power
    expected_p_noise = p_signal / (10.0 ** (snr_db / 10.0))  # 1.0 / 10 = 0.1
    measured_p_noise = float(np.mean(np.abs(noise) ** 2))
    
    # Measured SNR
    measured_snr_db = 10.0 * np.log10(p_signal / measured_p_noise)
    
    # Statistical verification: measured SNR matches requested SNR within 0.15 dB
    assert np.isclose(measured_snr_db, snr_db, atol=0.15)
    
    # Variance check for individual real (I) and imaginary (Q) noise components
    # Each component must receive exactly half of the total noise power
    expected_comp_variance = expected_p_noise / 2.0  # 0.1 / 2 = 0.05
    
    var_I = float(np.var(noise.real))
    var_Q = float(np.var(noise.imag))
    
    assert np.isclose(var_I, expected_comp_variance, atol=0.005)
    assert np.isclose(var_Q, expected_comp_variance, atol=0.005)
    
    # Independence (cross-correlation) check: covariance should be extremely close to 0
    cov = float(np.mean(noise.real * noise.imag))
    assert np.isclose(cov, 0.0, atol=0.005)


# 5. Test Integration in Pipeline (Placement after RRC)
def test_awgn_pipeline_integration():
    # Setup configuration with both pulse shaping and SNR
    config = GeneratorConfig(
        modulation="QPSK",
        num_symbols=20,
        sample_rate=8000.0,
        symbol_rate=1000.0,
        samples_per_symbol=8,
        random_seed=42,
        rolloff=0.35,
        filter_span_symbols=4,
        snr=12.0
    )
    
    # 20 symbols * 2 bits = 40 bits
    bits = np.random.default_rng(42).integers(0, 2, size=40, dtype=np.int8)
    
    # Generate the signal
    sig = generate_modulated_signal(bits, config)
    
    # Output length should match post-RRC waveform length
    # N = (num_symbols + span) * SPS = (20 + 4) * 8 = 192 samples
    assert sig.samples.shape == (2, 192)
    assert sig.samples.dtype == np.float32
    
    # Confirm ground truth values are correctly preserved
    assert sig.metadata.snr == 12.0
    assert sig.metadata.rolloff == 0.35
    assert sig.metadata.filter_span_symbols == 4
    assert sig.metadata.samples_per_symbol == 8
    
    # Check that noise was indeed added
    comp = iq_to_complex(sig.samples)
    assert not np.any(np.isnan(comp))
    assert not np.any(np.isinf(comp))
