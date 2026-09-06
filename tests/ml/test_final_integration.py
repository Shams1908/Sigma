import pytest
import numpy as np

from ml.generators.config import GeneratorConfig
from ml.generators.signal import GeneratedSignal, iq_to_complex
from ml.generators.bits import generate_bits
from ml.generators.modulation import generate_modulated_signal


def test_m2_final_integration_smoke():
    # 1. Setup end-to-end configuration with all features active
    config1 = GeneratorConfig(
        modulation="QPSK",
        num_symbols=50,
        sample_rate=8000.0,
        symbol_rate=1000.0,
        samples_per_symbol=8,
        random_seed=42,
        rolloff=0.35,
        filter_span_symbols=8,
        snr=10.0,
        frequency_offset=50.0,
        phase_offset=0.5,
        dc_offset_i=0.25,
        dc_offset_q=-0.25,
        iq_amplitude_imbalance=0.1,
        iq_phase_imbalance=0.05,
        timing_offset=0.5
    )

    # 50 symbols * 2 bits/symbol = 100 bits
    rng1 = np.random.default_rng(config1.random_seed)
    bits = generate_bits(num_bits=100, seed_or_generator=rng1)
    
    # Generate first signal
    sig1 = generate_modulated_signal(bits, config1)

    # 1. Verify Output shape: [2, N]
    # N = (num_symbols + filter_span) * SPS = (50 + 8) * 8 = 464
    assert sig1.samples.shape == (2, 464)

    # 2. Verify Output contains finite values only
    assert np.all(np.isfinite(sig1.samples))

    # 3. Verify Output dtype is consistent (float32)
    assert sig1.samples.dtype == np.float32

    # 4. Verify SyntheticGroundTruth preserves all configured parameters
    gt = sig1.metadata
    assert gt.modulation == "QPSK"
    assert gt.sample_rate == 8000.0
    assert gt.symbol_rate == 1000.0
    assert gt.samples_per_symbol == 8
    assert gt.rolloff == 0.35
    assert gt.filter_span_symbols == 8
    assert gt.snr == 10.0
    assert gt.frequency_offset == 50.0
    assert gt.phase_offset == 0.5
    assert gt.timing_offset == 0.5
    assert gt.dc_offset_i == 0.25
    assert gt.dc_offset_q == -0.25
    assert gt.iq_amplitude_imbalance == 0.1
    assert gt.iq_phase_imbalance == 0.05
    assert gt.random_seed == 42

    # 5. Verify Identical seed/configuration produces identical output
    rng2 = np.random.default_rng(config1.random_seed)
    bits_dup = generate_bits(num_bits=100, seed_or_generator=rng2)
    sig1_dup = generate_modulated_signal(bits_dup, config1)
    assert np.array_equal(sig1.samples, sig1_dup.samples)

    # 6. Verify Different seed changes the stochastic AWGN component
    config_diff_seed = GeneratorConfig(
        modulation="QPSK",
        num_symbols=50,
        sample_rate=8000.0,
        symbol_rate=1000.0,
        samples_per_symbol=8,
        random_seed=999,  # Different seed
        rolloff=0.35,
        filter_span_symbols=8,
        snr=10.0,
        frequency_offset=50.0,
        phase_offset=0.5,
        dc_offset_i=0.25,
        dc_offset_q=-0.25,
        iq_amplitude_imbalance=0.1,
        iq_phase_imbalance=0.05,
        timing_offset=0.5
    )
    rng_diff = np.random.default_rng(config_diff_seed.random_seed)
    bits_diff = generate_bits(num_bits=100, seed_or_generator=rng_diff)
    sig_diff = generate_modulated_signal(bits_diff, config_diff_seed)
    
    assert not np.array_equal(sig1.samples, sig_diff.samples)
