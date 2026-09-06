import pytest
import numpy as np

from ml.generators.config import GeneratorConfig
from ml.generators.signal import GeneratedSignal, iq_to_complex
from ml.generators.modulation import modulate, generate_modulated_signal
from ml.generators.modulation.interface import validate_bits


# Helper to create config
def make_config(modulation: str, num_symbols: int) -> GeneratorConfig:
    return GeneratorConfig(
        modulation=modulation,
        num_symbols=num_symbols,
        sample_rate=1.0,
        symbol_rate=1.0,
        samples_per_symbol=1,
        random_seed=42
    )


# 1. Test Input Bit Validation
def test_bit_validation():
    # Correct bits array type
    with pytest.raises(TypeError):
        validate_bits([0, 1, 0], 1, "BPSK")  # type: ignore

    # Dimensions check
    with pytest.raises(ValueError):
        validate_bits(np.zeros((2, 2)), 1, "BPSK")

    # Value checks (must only contain 0 and 1)
    with pytest.raises(ValueError):
        validate_bits(np.array([0, 1, 2], dtype=np.int8), 1, "BPSK")
    with pytest.raises(ValueError):
        validate_bits(np.array([-1, 0, 1], dtype=np.int8), 1, "BPSK")

    # Check compatibility checks
    with pytest.raises(ValueError):
        validate_bits(np.array([0, 1, 0], dtype=np.int8), 2, "QPSK")


# 2. Test BPSK Modulator
def test_bpsk_modulator():
    config = make_config("BPSK", 4)
    bits = np.array([0, 1, 0, 1], dtype=np.int8)
    
    symbols = modulate(bits, config)
    assert symbols.dtype == np.complex64
    assert len(symbols) == 4
    
    # Check known mapping patterns: 0 -> -1 + 0j, 1 -> 1 + 0j
    expected = np.array([-1.0 + 0j, 1.0 + 0j, -1.0 + 0j, 1.0 + 0j], dtype=np.complex64)
    assert np.allclose(symbols, expected)

    # Check average power
    assert np.isclose(np.mean(np.abs(symbols) ** 2), 1.0)


# 3. Test QPSK Modulator
def test_qpsk_modulator():
    config = make_config("QPSK", 4)
    bits = np.array([0, 0,  0, 1,  1, 1,  1, 0], dtype=np.int8)
    
    symbols = modulate(bits, config)
    assert symbols.dtype == np.complex64
    assert len(symbols) == 4
    
    # Check Gray coding:
    # 00 -> (1 + 1j) / sqrt(2)
    # 01 -> (-1 + 1j) / sqrt(2)
    # 11 -> (-1 - 1j) / sqrt(2)
    # 10 -> (1 - 1j) / sqrt(2)
    s = np.sqrt(2.0)
    expected = np.array([
        (1.0 + 1j) / s,
        (-1.0 + 1j) / s,
        (-1.0 - 1j) / s,
        (1.0 - 1j) / s
    ], dtype=np.complex64)
    assert np.allclose(symbols, expected)

    # Check average power
    assert np.isclose(np.mean(np.abs(symbols) ** 2), 1.0)


# 4. Test 8PSK Modulator
def test_8psk_modulator():
    config = make_config("8PSK", 8)
    bits = np.array([
        0,0,0,  # index 0 -> phase 0
        0,0,1,  # index 1 -> phase 1
        0,1,1,  # index 3 -> phase 2
        0,1,0,  # index 2 -> phase 3
        1,1,0,  # index 6 -> phase 4
        1,1,1,  # index 7 -> phase 5
        1,0,1,  # index 5 -> phase 6
        1,0,0   # index 4 -> phase 7
    ], dtype=np.int8)
    
    symbols = modulate(bits, config)
    assert symbols.dtype == np.complex64
    assert len(symbols) == 8
    
    # Verify angles k * pi / 4
    for k in range(8):
        angle = np.angle(symbols[k])
        # Wrap angle to [0, 2*pi)
        if angle < 0:
            angle += 2 * np.pi
        expected_angle = k * np.pi / 4.0
        assert np.isclose(angle, expected_angle, atol=1e-5)

    # Check average power
    assert np.isclose(np.mean(np.abs(symbols) ** 2), 1.0)


# 5. Test 16QAM Modulator
def test_16qam_modulator():
    # Test using standard name "QAM16"
    config = make_config("QAM16", 4)
    bits = np.array([
        0,0, 0,0,  # I=00(+3), Q=00(+3)
        0,1, 1,1,  # I=01(+1), Q=11(-1)
        1,0, 0,1,  # I=10(-3), Q=01(+1)
        1,1, 1,0   # I=11(-1), Q=10(-3)
    ], dtype=np.int8)
    
    symbols = modulate(bits, config)
    assert symbols.dtype == np.complex64
    assert len(symbols) == 4
    
    s = np.sqrt(10.0)
    expected = np.array([
        (3.0 + 3.0j) / s,
        (1.0 - 1.0j) / s,
        (-3.0 + 1.0j) / s,
        (-1.0 - 3.0j) / s
    ], dtype=np.complex64)
    assert np.allclose(symbols, expected)

    # Check average power on a complete constellation sweep
    # Complete sweep: all 16 possible combinations of 4 bits
    sweep_bits = []
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            for b2 in [0, 1]:
                for b3 in [0, 1]:
                    sweep_bits.extend([b0, b1, b2, b3])
    
    sweep_symbols = modulate(np.array(sweep_bits, dtype=np.int8), make_config("QAM16", 16))
    assert np.isclose(np.mean(np.abs(sweep_symbols) ** 2), 1.0)


# 6. Test 64QAM Modulator
def test_64qam_modulator():
    config = make_config("QAM64", 3)
    bits = np.array([
        0,0,0, 0,0,0,  # I=000(+7), Q=000(+7)
        0,0,1, 0,1,0,  # I=001(+5), Q=010(+1)
        1,0,0, 1,1,0   # I=100(-7), Q=110(-1)
    ], dtype=np.int8)
    
    symbols = modulate(bits, config)
    assert symbols.dtype == np.complex64
    assert len(symbols) == 3
    
    s = np.sqrt(42.0)
    expected = np.array([
        (7.0 + 7.0j) / s,
        (5.0 + 1.0j) / s,
        (-7.0 - 1.0j) / s
    ], dtype=np.complex64)
    assert np.allclose(symbols, expected)

    # Check average power on a complete constellation sweep
    sweep_bits = []
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            for b2 in [0, 1]:
                for b3 in [0, 1]:
                    for b4 in [0, 1]:
                        for b5 in [0, 1]:
                            sweep_bits.extend([b0, b1, b2, b3, b4, b5])
    
    sweep_symbols = modulate(np.array(sweep_bits, dtype=np.int8), make_config("QAM64", 64))
    assert np.isclose(np.mean(np.abs(sweep_symbols) ** 2), 1.0)


# 7. Test Determinism & RNG independence
def test_modulation_determinism():
    config = make_config("QPSK", 100)
    # Generate random bits
    rng = np.random.default_rng(12345)
    bits = rng.integers(0, 2, size=200, dtype=np.int8)
    
    symbols1 = modulate(bits, config)
    symbols2 = modulate(bits, config)
    
    assert np.array_equal(symbols1, symbols2)


# 8. Test Generated Signal Integration & Name Normalization
def test_generated_signal_integration():
    # Test that setting "16QAM" maps to "QAM16" in the generated output metadata
    config = GeneratorConfig(
        modulation="16QAM",
        num_symbols=10,
        sample_rate=1.0,
        symbol_rate=1.0,
        samples_per_symbol=1,
        random_seed=42
    )
    
    # 10 symbols * 4 bits = 40 bits
    bits = np.zeros(40, dtype=np.int8)
    
    sig = generate_modulated_signal(bits, config)
    assert isinstance(sig, GeneratedSignal)
    
    # Verify IQ sample shapes
    assert sig.samples.shape == (2, 10)
    assert sig.samples.dtype == np.float32
    
    # Verify ground-truth metadata
    assert sig.metadata.modulation == "QAM16"  # Normalized to canonical M1 label!
    assert sig.metadata.samples_per_symbol == 1
    
    # Reconvert IQ to complex and verify correct values
    comp = iq_to_complex(sig.samples)
    assert len(comp) == 10
    assert comp.dtype == np.complex64
