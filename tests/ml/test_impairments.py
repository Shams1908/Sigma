import pytest
import numpy as np

from ml.generators.config import GeneratorConfig
from ml.generators.signal import GeneratedSignal, iq_to_complex
from ml.generators.impairments import (
    apply_frequency_offset,
    apply_phase_offset,
    apply_dc_offset,
    apply_iq_imbalance,
    apply_timing_offset,
    apply_impairments,
)
from ml.generators.modulation import generate_modulated_signal


# 1. Test Carrier Frequency Offset
def test_frequency_offset():
    # Length 10 constant waveform
    x = np.ones(10, dtype=np.complex64)
    
    # Zero offset returns equivalent signal
    config_zero = GeneratorConfig(
        modulation="BPSK", num_symbols=10, sample_rate=1000.0, symbol_rate=1000.0,
        samples_per_symbol=1, random_seed=42, frequency_offset=0.0
    )
    y_zero = apply_frequency_offset(x, config_zero)
    assert np.allclose(y_zero, x)
    assert y_zero.dtype == np.complex64
    
    # Magnitude preservation check
    config_offset = GeneratorConfig(
        modulation="BPSK", num_symbols=10, sample_rate=1000.0, symbol_rate=1000.0,
        samples_per_symbol=1, random_seed=42, frequency_offset=50.0
    )
    y_offset = apply_frequency_offset(x, config_offset)
    assert np.allclose(np.abs(y_offset), 1.0)
    assert len(y_offset) == len(x)
    
    # Phase rotation check: index n should have angle 2*pi*delta_f*n/Fs
    for n in range(len(x)):
        angle = np.angle(y_offset[n])
        expected_angle = 2.0 * np.pi * 50.0 * n / 1000.0
        # Wrap to (-pi, pi]
        expected_angle = (expected_angle + np.pi) % (2.0 * np.pi) - np.pi
        assert np.isclose(angle, expected_angle, atol=1e-5)


# 2. Test Carrier Phase Offset
def test_phase_offset():
    x = np.ones(10, dtype=np.complex64)
    
    # Zero phase returns equivalent signal
    config_zero = GeneratorConfig(
        modulation="BPSK", num_symbols=10, sample_rate=1000.0, symbol_rate=1000.0,
        samples_per_symbol=1, random_seed=42, phase_offset=0.0
    )
    y_zero = apply_phase_offset(x, config_zero)
    assert np.allclose(y_zero, x)
    
    # Known phase shift
    config_phase = GeneratorConfig(
        modulation="BPSK", num_symbols=10, sample_rate=1000.0, symbol_rate=1000.0,
        samples_per_symbol=1, random_seed=42, phase_offset=np.pi / 4.0
    )
    y_phase = apply_phase_offset(x, config_phase)
    assert np.allclose(np.abs(y_phase), 1.0)
    assert np.allclose(np.angle(y_phase), np.pi / 4.0)


# 3. Test DC Offset
def test_dc_offset():
    x = np.ones(10, dtype=np.complex64)
    
    # Zero DC leaves signal unchanged
    config_zero = GeneratorConfig(
        modulation="BPSK", num_symbols=10, sample_rate=1000.0, symbol_rate=1000.0,
        samples_per_symbol=1, random_seed=42, dc_offset_i=0.0, dc_offset_q=0.0
    )
    y_zero = apply_dc_offset(x, config_zero)
    assert np.allclose(y_zero, x)
    
    # I-only shift
    config_i = GeneratorConfig(
        modulation="BPSK", num_symbols=10, sample_rate=1000.0, symbol_rate=1000.0,
        samples_per_symbol=1, random_seed=42, dc_offset_i=0.5
    )
    y_i = apply_dc_offset(x, config_i)
    assert np.allclose(y_i.real, 1.5)
    assert np.allclose(y_i.imag, 0.0)
    
    # Q-only shift
    config_q = GeneratorConfig(
        modulation="BPSK", num_symbols=10, sample_rate=1000.0, symbol_rate=1000.0,
        samples_per_symbol=1, random_seed=42, dc_offset_q=-0.5
    )
    y_q = apply_dc_offset(x, config_q)
    assert np.allclose(y_q.real, 1.0)
    assert np.allclose(y_q.imag, -0.5)


# 4. Test IQ Imbalance
def test_iq_imbalance():
    x = np.ones(10, dtype=np.complex64)
    
    # Zero imbalance returns original signal (identity)
    config_zero = GeneratorConfig(
        modulation="BPSK", num_symbols=10, sample_rate=1000.0, symbol_rate=1000.0,
        samples_per_symbol=1, random_seed=42, iq_amplitude_imbalance=0.0, iq_phase_imbalance=0.0
    )
    y_zero = apply_iq_imbalance(x, config_zero)
    assert np.allclose(y_zero, x)
    
    # Known amplitude imbalance (A = 0.1)
    # Scales I (real part) by 1.1, Q (imaginary part) by 0.9.
    config_amp = GeneratorConfig(
        modulation="BPSK", num_symbols=10, sample_rate=1000.0, symbol_rate=1000.0,
        samples_per_symbol=1, random_seed=42, iq_amplitude_imbalance=0.1
    )
    # Input has I = 1.0, Q = 0.0
    y_amp = apply_iq_imbalance(x, config_amp)
    assert np.allclose(y_amp.real, 1.1)
    assert np.allclose(y_amp.imag, 0.0)
    
    # Test on imaginary signal (I = 0.0, Q = 1.0)
    x_imag = 1j * np.ones(10, dtype=np.complex64)
    y_amp_imag = apply_iq_imbalance(x_imag, config_amp)
    assert np.allclose(y_amp_imag.real, 0.0)
    assert np.allclose(y_amp_imag.imag, 0.9)
    
    # Known phase imbalance (theta = np.pi / 6)
    # Cross-coupling produces cross terms
    config_phase = GeneratorConfig(
        modulation="BPSK", num_symbols=10, sample_rate=1000.0, symbol_rate=1000.0,
        samples_per_symbol=1, random_seed=42, iq_phase_imbalance=np.pi / 6.0
    )
    y_phase = apply_iq_imbalance(x, config_phase)
    # I_out = (1) * cos(pi/12), Q_out = -(1) * sin(pi/12)
    assert np.allclose(y_phase.real, np.cos(np.pi / 12.0))
    assert np.allclose(y_phase.imag, -np.sin(np.pi / 12.0))


# 5. Test Timing Offset (Linear Interpolation and Out-of-Range Zero Padding)
def test_timing_offset_simple_signal():
    # Simple deterministic waveform x = [1.0, 2.0, 3.0, 4.0]
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex64)
    
    # 1. Zero shift returns equivalent signal
    config_zero = GeneratorConfig(
        modulation="BPSK", num_symbols=4, sample_rate=1000.0, symbol_rate=1000.0,
        samples_per_symbol=1, random_seed=42, timing_offset=0.0
    )
    y_zero = apply_timing_offset(x, config_zero)
    assert np.allclose(y_zero, x)
    
    # 2. Positive Integer timing shift tau = 1.0 (delay by 1 sample)
    # Expected output: [0.0, 1.0, 2.0, 3.0] due to zero-padding boundaries
    config_one = GeneratorConfig(
        modulation="BPSK", num_symbols=4, sample_rate=1000.0, symbol_rate=1000.0,
        samples_per_symbol=1, random_seed=42, timing_offset=1.0
    )
    y_one = apply_timing_offset(x, config_one)
    assert np.allclose(y_one, np.array([0.0, 1.0, 2.0, 3.0], dtype=np.complex64))
    
    # 3. Fractional timing shift tau = 0.5
    # y[n] = (1 - d)*x[n - k] + d*x[n - k - 1]
    # Here k = 0, d = 0.5 -> y[n] = 0.5 * x[n] + 0.5 * x[n-1]
    # Expected:
    # y[0] = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    # y[1] = 0.5 * 2.0 + 0.5 * 1.0 = 1.5
    # y[2] = 0.5 * 3.0 + 0.5 * 2.0 = 2.5
    # y[3] = 0.5 * 4.0 + 0.5 * 3.0 = 3.5
    # Final output: [0.5, 1.5, 2.5, 3.5]
    config_frac = GeneratorConfig(
        modulation="BPSK", num_symbols=4, sample_rate=1000.0, symbol_rate=1000.0,
        samples_per_symbol=1, random_seed=42, timing_offset=0.5
    )
    y_frac = apply_timing_offset(x, config_frac)
    assert np.allclose(y_frac, np.array([0.5, 1.5, 2.5, 3.5], dtype=np.complex64))

    # 4. Negative shift tau = -0.5 (advance by 0.5 sample)
    # k = -1, d = 0.5 -> y[n] = 0.5 * x[n+1] + 0.5 * x[n]
    # Expected:
    # y[0] = 0.5 * 2.0 + 0.5 * 1.0 = 1.5
    # y[1] = 0.5 * 3.0 + 0.5 * 2.0 = 2.5
    # y[2] = 0.5 * 4.0 + 0.5 * 3.0 = 3.5
    # y[3] = 0.5 * 0.0 + 0.5 * 4.0 = 2.0
    # Final output: [1.5, 2.5, 3.5, 2.0]
    config_neg = GeneratorConfig(
        modulation="BPSK", num_symbols=4, sample_rate=1000.0, symbol_rate=1000.0,
        samples_per_symbol=1, random_seed=42, timing_offset=-0.5
    )
    y_neg = apply_timing_offset(x, config_neg)
    assert np.allclose(y_neg, np.array([1.5, 2.5, 3.5, 2.0], dtype=np.complex64))


# 6. Test Composition Order
def test_impairment_composition_order():
    # Start with standard complex constant signal
    x = np.ones(10, dtype=np.complex64)
    
    # We configure Frequency = 10Hz, Phase = pi/2, IQ Amplitude = 0.1, DC_I = 0.5, Timing = 1.0
    config = GeneratorConfig(
        modulation="BPSK", num_symbols=10, sample_rate=1000.0, symbol_rate=1000.0,
        samples_per_symbol=1, random_seed=42,
        frequency_offset=50.0,
        phase_offset=np.pi / 2.0,
        iq_amplitude_imbalance=0.2,
        dc_offset_i=1.0,
        timing_offset=1.0
    )
    
    # 1. Execute manual impairments sequentially in the exact specified order
    manual = x
    manual = apply_frequency_offset(manual, config)
    manual = apply_phase_offset(manual, config)
    manual = apply_iq_imbalance(manual, config)
    manual = apply_dc_offset(manual, config)
    manual = apply_timing_offset(manual, config)
    
    # 2. Execute composition runner
    composed = apply_impairments(x, config)
    
    # The output of both should be mathematically identical
    assert np.array_equal(composed, manual)
    assert composed.shape == x.shape


# 7. Test Integration in Waveform Pipeline
def test_impairments_pipeline_integration():
    config = GeneratorConfig(
        modulation="QAM16",
        num_symbols=32,
        sample_rate=8000.0,
        symbol_rate=1000.0,
        samples_per_symbol=8,
        random_seed=42,
        rolloff=0.35,
        filter_span_symbols=8,
        snr=15.0,
        frequency_offset=100.0,
        phase_offset=0.2,
        dc_offset_i=0.1,
        dc_offset_q=-0.1,
        iq_amplitude_imbalance=0.05,
        iq_phase_imbalance=0.01,
        timing_offset=0.25
    )
    
    # 32 symbols * 4 bits = 128 bits
    bits = np.random.default_rng(42).integers(0, 2, size=128, dtype=np.int8)
    
    sig = generate_modulated_signal(bits, config)
    
    # Check canonical shape: [2, N] where N = (32 + 8) * 8 = 320
    assert sig.samples.shape == (2, 320)
    assert sig.samples.dtype == np.float32
    
    # Check that metadata is perfectly saved in ground truth
    assert sig.metadata.modulation == "QAM16"
    assert sig.metadata.snr == 15.0
    assert sig.metadata.frequency_offset == 100.0
    assert sig.metadata.phase_offset == 0.2
    assert sig.metadata.dc_offset_i == 0.1
    assert sig.metadata.dc_offset_q == -0.1
    assert sig.metadata.iq_amplitude_imbalance == 0.05
    assert sig.metadata.iq_phase_imbalance == 0.01
    assert sig.metadata.timing_offset == 0.25
    
    # Confirm output limits
    assert not np.any(np.isnan(sig.samples))
    assert not np.any(np.isinf(sig.samples))
