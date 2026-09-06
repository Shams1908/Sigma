import pytest
import numpy as np

from ml.generators.config import GeneratorConfig
from ml.generators.signal import (
    SyntheticGroundTruth,
    GeneratedSignal,
    complex_to_iq,
    iq_to_complex,
)
from ml.generators.bits import generate_bits


# 1. Test Configuration Validation
def test_generator_config_validation():
    # Valid config construction
    config = GeneratorConfig(
        modulation="QPSK",
        num_symbols=100,
        sample_rate=1e6,
        symbol_rate=2.5e5,
        samples_per_symbol=4,
        random_seed=42,
        snr=10.0,
        frequency_offset=100.0,
        phase_offset=0.5,
        timing_offset=0.1,
    )
    assert config.modulation == "QPSK"
    assert config.num_symbols == 100
    assert config.sample_rate == 1e6
    assert config.symbol_rate == 2.5e5
    assert config.samples_per_symbol == 4
    assert config.random_seed == 42
    assert config.snr == 10.0
    assert config.frequency_offset == 100.0
    assert config.phase_offset == 0.5
    assert config.timing_offset == 0.1


# 2. Test Invalid Configuration Rejections
def test_generator_config_invalid_rejections():
    # Invalid modulation label
    with pytest.raises(ValueError):
        GeneratorConfig(
            modulation="INVALID_MODULATION",
            num_symbols=100,
            sample_rate=1e6,
            symbol_rate=2.5e5,
            samples_per_symbol=4,
            random_seed=42,
        )

    # num_symbols invalid values
    with pytest.raises(ValueError):
        GeneratorConfig(
            modulation="QPSK",
            num_symbols=0,
            sample_rate=1e6,
            symbol_rate=2.5e5,
            samples_per_symbol=4,
            random_seed=42,
        )
    with pytest.raises(TypeError):
        GeneratorConfig(
            modulation="QPSK",
            num_symbols=100.5,  # type: ignore
            sample_rate=1e6,
            symbol_rate=2.5e5,
            samples_per_symbol=4,
            random_seed=42,
        )

    # sample_rate invalid values
    with pytest.raises(ValueError):
        GeneratorConfig(
            modulation="QPSK",
            num_symbols=100,
            sample_rate=-1e6,
            symbol_rate=2.5e5,
            samples_per_symbol=4,
            random_seed=42,
        )

    # symbol_rate invalid values
    with pytest.raises(ValueError):
        GeneratorConfig(
            modulation="QPSK",
            num_symbols=100,
            sample_rate=1e6,
            symbol_rate=0,
            samples_per_symbol=4,
            random_seed=42,
        )

    # samples_per_symbol type and value checks
    with pytest.raises(TypeError):
        GeneratorConfig(
            modulation="QPSK",
            num_symbols=100,
            sample_rate=1e6,
            symbol_rate=2.5e5,
            samples_per_symbol=4.5,  # type: ignore
            random_seed=42,
        )
    with pytest.raises(ValueError):
        GeneratorConfig(
            modulation="QPSK",
            num_symbols=100,
            sample_rate=1e6,
            symbol_rate=2.5e5,
            samples_per_symbol=0,
            random_seed=42,
        )

    # Mathematical inconsistency: samples_per_symbol != sample_rate / symbol_rate
    with pytest.raises(ValueError):
        GeneratorConfig(
            modulation="QPSK",
            num_symbols=100,
            sample_rate=1e6,
            symbol_rate=2.5e5,
            samples_per_symbol=5,  # Expected 4
            random_seed=42,
        )

    # Invalid random seeds
    with pytest.raises(ValueError):
        GeneratorConfig(
            modulation="QPSK",
            num_symbols=100,
            sample_rate=1e6,
            symbol_rate=2.5e5,
            samples_per_symbol=4,
            random_seed=-1,
        )
    with pytest.raises(TypeError):
        GeneratorConfig(
            modulation="QPSK",
            num_symbols=100,
            sample_rate=1e6,
            symbol_rate=2.5e5,
            samples_per_symbol=4,
            random_seed=4.2,  # type: ignore
        )

    # Non-finite optional parameters
    with pytest.raises(ValueError):
        GeneratorConfig(
            modulation="QPSK",
            num_symbols=100,
            sample_rate=1e6,
            symbol_rate=2.5e5,
            samples_per_symbol=4,
            random_seed=42,
            snr=float("nan"),
        )


# 3. Test Deterministic Bit Generation
def test_deterministic_bit_generation():
    # Length validation
    with pytest.raises(ValueError):
        generate_bits(-10, seed_or_generator=42)
    with pytest.raises(TypeError):
        generate_bits(10.5, seed_or_generator=42)  # type: ignore

    # Test seed correctness
    bits1 = generate_bits(100, seed_or_generator=42)
    bits2 = generate_bits(100, seed_or_generator=42)
    
    assert len(bits1) == 100
    assert bits1.dtype == np.int8
    assert np.array_equal(bits1, bits2)

    # Verify that different seeds produce different outputs
    bits_diff = generate_bits(100, seed_or_generator=100)
    assert not np.array_equal(bits1, bits_diff)

    # Verify accepting an explicit np.random.Generator works
    rng = np.random.default_rng(42)
    bits_rng = generate_bits(100, seed_or_generator=rng)
    assert np.array_equal(bits1, bits_rng)


# 4. Test IQ Representation shape and conversions
def test_iq_representation_and_conversions():
    # Type checks
    with pytest.raises(TypeError):
        complex_to_iq([1 + 1j, 2 + 2j])  # type: ignore
    with pytest.raises(TypeError):
        complex_to_iq(np.array([1.0, 2.0]))  # not complex

    with pytest.raises(TypeError):
        iq_to_complex([[1, 2], [3, 4]])  # type: ignore
    with pytest.raises(ValueError):
        iq_to_complex(np.array([1, 2, 3]))  # 1D array
    with pytest.raises(ValueError):
        iq_to_complex(np.zeros((3, 10)))  # shape not [2, N]

    # Verification of round-trip identity and shape [2, N]
    n_samples = 50
    complex_arr = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64)
    
    iq_arr = complex_to_iq(complex_arr)
    assert iq_arr.shape == (2, n_samples)
    assert iq_arr.dtype == np.float32

    # Reconvert back and check close values
    reconstructed_complex = iq_to_complex(iq_arr)
    assert reconstructed_complex.dtype == np.complex64
    assert np.allclose(complex_arr, reconstructed_complex, rtol=1e-6)

    # Identity verification from IQ to Complex to IQ
    iq_input = np.random.randn(2, n_samples).astype(np.float32)
    reconstructed_iq = complex_to_iq(iq_to_complex(iq_input))
    assert np.allclose(iq_input, reconstructed_iq, rtol=1e-6)


# 5. Test GeneratedSignal construction and metadata preservation
def test_generated_signal_construction():
    # Test valid construction
    metadata = SyntheticGroundTruth(
        modulation="QPSK",
        snr=15.0,
        sample_rate=1e6,
        symbol_rate=2.5e5,
        samples_per_symbol=4,
        random_seed=42,
    )
    
    samples = np.random.randn(2, 400).astype(np.float32)
    signal = GeneratedSignal(samples=samples, metadata=metadata)
    
    assert signal.samples.shape == (2, 400)
    assert signal.samples.dtype == np.float32
    assert signal.metadata == metadata

    # Invalid constructions
    # samples not ndarray
    with pytest.raises(TypeError):
        GeneratedSignal(samples=[[1, 2], [3, 4]], metadata=metadata)  # type: ignore
    # samples wrong shape
    with pytest.raises(ValueError):
        GeneratedSignal(samples=np.zeros((1, 400), dtype=np.float32), metadata=metadata)
    # samples wrong dtype
    with pytest.raises(TypeError):
        GeneratedSignal(samples=np.zeros((2, 400), dtype=np.float64), metadata=metadata)
    # metadata wrong type
    with pytest.raises(TypeError):
        GeneratedSignal(samples=samples, metadata="Not_Metadata")  # type: ignore
