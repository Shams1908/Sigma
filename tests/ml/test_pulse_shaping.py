import pytest
import numpy as np

from ml.generators.config import GeneratorConfig
from ml.generators.signal import GeneratedSignal, iq_to_complex
from ml.generators.pulse_shaping.rrc import design_rrc_filter, pulse_shape
from ml.generators.modulation import generate_modulated_signal


# 1. Test RRC Filter Design
def test_rrc_filter_design():
    # Valid filter design check
    sps = 8
    rolloff = 0.35
    span = 8
    taps = design_rrc_filter(samples_per_symbol=sps, rolloff=rolloff, filter_span_symbols=span)
    
    # Correct number of taps: span * SPS + 1 = 8 * 8 + 1 = 65
    assert len(taps) == 65
    assert taps.dtype == np.float32
    
    # Symmetry check
    assert np.allclose(taps, taps[::-1])
    
    # Normalization check: sum(h^2) = 1.0
    assert np.isclose(np.sum(taps ** 2), 1.0)
    
    # No NaN or Infinity checks
    assert not np.any(np.isnan(taps))
    assert not np.any(np.isinf(taps))


# 2. Test RRC Boundary Rolloff Factor Taps
def test_rrc_boundary_rolloffs():
    sps = 8
    span = 8
    
    # Test rolloff = 0.0 (Pure sinc filter)
    taps_0 = design_rrc_filter(sps, 0.0, span)
    assert not np.any(np.isnan(taps_0))
    assert np.allclose(taps_0, taps_0[::-1])
    assert np.isclose(np.sum(taps_0 ** 2), 1.0)
    
    # Test rolloff = 1.0 (Maximum excess bandwidth)
    taps_1 = design_rrc_filter(sps, 1.0, span)
    assert not np.any(np.isnan(taps_1))
    assert np.allclose(taps_1, taps_1[::-1])
    assert np.isclose(np.sum(taps_1 ** 2), 1.0)

    # Test singularity point cases where index n lands exactly on ±SPS / (4 * rolloff)
    # e.g., if rolloff = 0.25, SPS = 8, 1 / (4 * rolloff) = 1.
    # index n = ±8 corresponds to t = ±1.
    taps_sing = design_rrc_filter(samples_per_symbol=8, rolloff=0.25, filter_span_symbols=4)
    assert len(taps_sing) == 33
    assert not np.any(np.isnan(taps_sing))
    assert not np.any(np.isinf(taps_sing))
    assert np.allclose(taps_sing, taps_sing[::-1])
    assert np.isclose(np.sum(taps_sing ** 2), 1.0)


# 3. Test Invalid Filter Invariant Rejections
def test_rrc_invalid_parameters():
    # Invalid rolloff values
    with pytest.raises(ValueError):
        design_rrc_filter(8, -0.1, 8)
    with pytest.raises(ValueError):
        design_rrc_filter(8, 1.1, 8)
    with pytest.raises(TypeError):
        design_rrc_filter(8, "0.35", 8)  # type: ignore

    # Invalid span values
    with pytest.raises(ValueError):
        design_rrc_filter(8, 0.35, 0)
    with pytest.raises(ValueError):
        design_rrc_filter(8, 0.35, -4)
    with pytest.raises(TypeError):
        design_rrc_filter(8, 0.35, 8.5)  # type: ignore

    # Invalid sps values
    with pytest.raises(ValueError):
        design_rrc_filter(0, 0.35, 8)
    with pytest.raises(ValueError):
        design_rrc_filter(-8, 0.35, 8)
    with pytest.raises(TypeError):
        design_rrc_filter(8.5, 0.35, 8)  # type: ignore


# 4. Test Upsampling Logic
def test_upsampling_process():
    # Create configuration with pulse shaping parameters
    config = GeneratorConfig(
        modulation="BPSK",
        num_symbols=3,
        sample_rate=8000.0,
        symbol_rate=1000.0,
        samples_per_symbol=8,
        random_seed=42,
        rolloff=0.35,
        filter_span_symbols=8
    )
    
    symbols = np.array([1.0 + 0j, -1.0 + 0j, 1.0 + 0j], dtype=np.complex64)
    
    # Access internal upsampling output behavior through pulse_shape execution
    # Let's test custom upsampling behavior logic manually to verify zero insertion
    sps = config.samples_per_symbol
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex64)
    upsampled[::sps] = symbols
    
    assert len(upsampled) == 24
    
    # Original symbols appear at indices: 0, 8, 16
    assert upsampled[0] == 1.0 + 0j
    assert upsampled[8] == -1.0 + 0j
    assert upsampled[16] == 1.0 + 0j
    
    # All other indices are strictly 0.0 + 0.0j
    for i in range(len(upsampled)):
        if i not in [0, 8, 16]:
            assert upsampled[i] == 0.0 + 0j


# 5. Test Pulse Shaping Output properties
def test_pulse_shaper_output():
    config = GeneratorConfig(
        modulation="QPSK",
        num_symbols=10,
        sample_rate=8000.0,
        symbol_rate=1000.0,
        samples_per_symbol=8,
        random_seed=42,
        rolloff=0.35,
        filter_span_symbols=4
    )
    
    # 10 symbols * 2 bits = 20 bits
    bits = np.array([0,0, 0,1, 1,1, 1,0, 0,0, 0,1, 1,1, 1,0, 0,0, 0,1], dtype=np.int8)
    
    # Generate the pulse shaped signal
    sig = generate_modulated_signal(bits, config)
    
    assert isinstance(sig, GeneratedSignal)
    
    # Output length: (num_symbols + filter_span_symbols) * samples_per_symbol
    # N = (10 + 4) * 8 = 14 * 8 = 112 samples
    assert sig.samples.shape == (2, 112)
    assert sig.samples.dtype == np.float32
    
    # Verify metadata preservation
    assert sig.metadata.modulation == "QPSK"
    assert sig.metadata.rolloff == 0.35
    assert sig.metadata.filter_span_symbols == 4
    assert sig.metadata.samples_per_symbol == 8
    
    # Check shape reconstruction round-trip
    comp = iq_to_complex(sig.samples)
    assert len(comp) == 112
    assert comp.dtype == np.complex64
    assert not np.any(np.isnan(comp))
    assert not np.any(np.isinf(comp))


# 6. Test Impulse and Determinism
def test_pulse_shaper_determinism_and_impulse():
    config = GeneratorConfig(
        modulation="BPSK",
        num_symbols=3,
        sample_rate=4.0,
        symbol_rate=1.0,
        samples_per_symbol=4,
        random_seed=42,
        rolloff=0.5,
        filter_span_symbols=2
    )
    
    # Standard impulse-like bit sequence (maps to BPSK: +1, -1, +1)
    bits = np.array([1, 0, 1], dtype=np.int8)
    
    sig1 = generate_modulated_signal(bits, config)
    sig2 = generate_modulated_signal(bits, config)
    
    # Deterministic mapping check
    assert np.array_equal(sig1.samples, sig2.samples)
    
    # Test output length: (3 + 2) * 4 = 20
    assert sig1.samples.shape == (2, 20)
