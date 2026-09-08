"""
Unit tests for M6 Signal Representations (transforms.py).
Verifies:
  - Correct output shapes ([C, N] and [B, C, N])
  - Finite values (no NaN/Inf)
  - Strict determinism
  - Channel count helpers
  - Normalization consistency
"""
import pytest
import numpy as np

from ml.representations.transforms import (
    RepresentationType,
    compute_representation,
    get_representation_channels,
)


def test_representation_channel_counts():
    """Verifies that get_representation_channels returns exact expected dimensions."""
    assert get_representation_channels(RepresentationType.RAW_IQ) == 2
    assert get_representation_channels(RepresentationType.IQ_AMPLITUDE) == 3
    assert get_representation_channels(RepresentationType.AMPLITUDE_PHASE) == 2
    assert get_representation_channels(RepresentationType.IQ_AMP_PHASE) == 4
    
    # String input support
    assert get_representation_channels("RAW_IQ") == 2
    assert get_representation_channels("IQ_AMPLITUDE") == 3
    
    with pytest.raises(ValueError):
        get_representation_channels("INVALID_REP")


def test_single_sample_shapes_and_types():
    """Tests shape and dtype preservation for single signal of shape [2, 128]."""
    rng = np.random.default_rng(42)
    sig = rng.standard_normal((2, 128)).astype(np.float32)

    # RAW_IQ -> [2, 128]
    raw = compute_representation(sig, RepresentationType.RAW_IQ)
    assert raw.shape == (2, 128)
    assert raw.dtype == np.float32
    assert np.allclose(raw, sig)

    # IQ_AMPLITUDE -> [3, 128]
    iq_amp = compute_representation(sig, RepresentationType.IQ_AMPLITUDE)
    assert iq_amp.shape == (3, 128)
    assert iq_amp.dtype == np.float32
    # Verify channels 0 and 1 are identical to input
    assert np.allclose(iq_amp[0], sig[0])
    assert np.allclose(iq_amp[1], sig[1])
    # Verify channel 2 is sqrt(I^2 + Q^2)
    expected_amp = np.sqrt(sig[0]**2 + sig[1]**2)
    assert np.allclose(iq_amp[2], expected_amp, atol=1e-6)

    # AMPLITUDE_PHASE -> [2, 128]
    amp_phase = compute_representation(sig, RepresentationType.AMPLITUDE_PHASE)
    assert amp_phase.shape == (2, 128)
    assert amp_phase.dtype == np.float32

    # IQ_AMP_PHASE -> [4, 128]
    four_ch = compute_representation(sig, RepresentationType.IQ_AMP_PHASE)
    assert four_ch.shape == (4, 128)
    assert four_ch.dtype == np.float32


def test_batch_sample_shapes():
    """Tests shape handling for batch of shape [16, 2, 128]."""
    rng = np.random.default_rng(100)
    batch = rng.standard_normal((16, 2, 128)).astype(np.float32)

    out_raw = compute_representation(batch, RepresentationType.RAW_IQ)
    assert out_raw.shape == (16, 2, 128)

    out_3ch = compute_representation(batch, RepresentationType.IQ_AMPLITUDE)
    assert out_3ch.shape == (16, 3, 128)

    out_4ch = compute_representation(batch, RepresentationType.IQ_AMP_PHASE)
    assert out_4ch.shape == (16, 4, 128)


def test_representation_numerical_safety():
    """Verifies that all transforms are finite and reject NaN / Inf inputs."""
    rng = np.random.default_rng(200)
    sig = rng.standard_normal((2, 128)).astype(np.float32)

    # Zero signal (boundary case for sqrt and arctan)
    zero_sig = np.zeros((2, 128), dtype=np.float32)
    for rep in RepresentationType:
        res = compute_representation(zero_sig, rep)
        assert np.isfinite(res).all()
        assert not np.isnan(res).any()

    # Reject NaN
    nan_sig = sig.copy()
    nan_sig[0, 10] = np.nan
    with pytest.raises(ValueError):
        compute_representation(nan_sig, RepresentationType.IQ_AMPLITUDE)

    # Reject Inf
    inf_sig = sig.copy()
    inf_sig[1, 5] = np.inf
    with pytest.raises(ValueError):
        compute_representation(inf_sig, RepresentationType.IQ_AMPLITUDE)


def test_representation_determinism():
    """Verifies that identical inputs produce bit-for-bit identical representation tensors."""
    rng = np.random.default_rng(300)
    sig = rng.standard_normal((2, 128)).astype(np.float32)

    t1 = compute_representation(sig, RepresentationType.IQ_AMPLITUDE)
    t2 = compute_representation(sig, RepresentationType.IQ_AMPLITUDE)
    assert np.array_equal(t1, t2)


def test_invalid_input_dimensions():
    """Verifies dimension assertion error handling."""
    with pytest.raises(ValueError):
        # 1D array
        compute_representation(np.ones(128, dtype=np.float32))

    with pytest.raises(ValueError):
        # 4D array
        compute_representation(np.ones((2, 2, 2, 128), dtype=np.float32))

    with pytest.raises(ValueError):
        # Channel axis is not 2
        compute_representation(np.ones((3, 128), dtype=np.float32))

    with pytest.raises(TypeError):
        # Not a numpy array
        compute_representation([[1, 2], [3, 4]])
