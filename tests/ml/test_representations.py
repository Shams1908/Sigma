"""
Unit tests for ml/representations/  (M6 signal representation module).
"""
from __future__ import annotations

import numpy as np
import pytest

from ml.representations import (
    Representation,
    REPRESENTATION_INFO,
    num_channels,
    apply_representation,
    to_raw_iq,
    to_iq_amplitude,
    to_amplitude_phase,
    to_iq_amp_phase,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_iq(n: int = 128, seed: int = 42) -> np.ndarray:
    """Return a [2, N] float32 IQ array."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((2, n)).astype(np.float32)


def _make_batch(b: int = 4, n: int = 128, seed: int = 0) -> np.ndarray:
    """Return a [B, 2, N] float32 IQ array."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((b, 2, n)).astype(np.float32)


# ── 1. Metadata ────────────────────────────────────────────────────────────────

def test_representation_info_channel_counts():
    assert num_channels(Representation.RAW_IQ) == 2
    assert num_channels(Representation.IQ_AMPLITUDE) == 3
    assert num_channels(Representation.AMPLITUDE_PHASE) == 2
    assert num_channels(Representation.IQ_AMP_PHASE) == 4


def test_num_channels_accepts_string():
    assert num_channels("RAW_IQ") == 2
    assert num_channels("IQ_AMPLITUDE") == 3


# ── 2. RAW_IQ ─────────────────────────────────────────────────────────────────

def test_raw_iq_single_shape():
    iq = _make_iq()
    out = to_raw_iq(iq)
    assert out.shape == (2, 128)
    assert out.dtype == np.float32


def test_raw_iq_passthrough():
    iq = _make_iq()
    out = to_raw_iq(iq)
    np.testing.assert_array_equal(out, iq.astype(np.float32))


def test_raw_iq_batch():
    iq = _make_batch()
    out = to_raw_iq(iq)
    assert out.shape == (4, 2, 128)
    assert out.dtype == np.float32


# ── 3. IQ_AMPLITUDE ──────────────────────────────────────────────────────────

def test_iq_amplitude_single_shape():
    iq = _make_iq()
    out = to_iq_amplitude(iq)
    assert out.shape == (3, 128)
    assert out.dtype == np.float32


def test_iq_amplitude_channels():
    iq = _make_iq()
    out = to_iq_amplitude(iq)
    # I and Q channels must be identical to input
    np.testing.assert_array_equal(out[0], iq[0])
    np.testing.assert_array_equal(out[1], iq[1])
    # Amplitude channel
    expected_amp = np.sqrt(iq[0] ** 2 + iq[1] ** 2)
    np.testing.assert_allclose(out[2], expected_amp, atol=1e-6)


def test_iq_amplitude_batch():
    iq = _make_batch()
    out = to_iq_amplitude(iq)
    assert out.shape == (4, 3, 128)
    # I and Q pass-through in batch
    np.testing.assert_array_equal(out[:, 0, :], iq[:, 0, :])
    np.testing.assert_array_equal(out[:, 1, :], iq[:, 1, :])


def test_iq_amplitude_non_negative():
    iq = _make_iq()
    out = to_iq_amplitude(iq)
    assert np.all(out[2] >= 0.0)


# ── 4. AMPLITUDE_PHASE ───────────────────────────────────────────────────────

def test_amplitude_phase_single_shape():
    iq = _make_iq()
    out = to_amplitude_phase(iq)
    assert out.shape == (2, 128)
    assert out.dtype == np.float32


def test_amplitude_phase_first_sample_zero():
    iq = _make_iq()
    out = to_amplitude_phase(iq)
    # Phase-difference at index 0 must be 0 (zero-padded)
    assert out[1, 0] == 0.0


def test_amplitude_phase_batch():
    iq = _make_batch()
    out = to_amplitude_phase(iq)
    assert out.shape == (4, 2, 128)
    # First sample phase-diff must be 0 for all items in batch
    np.testing.assert_array_equal(out[:, 1, 0], np.zeros(4))


def test_amplitude_phase_range():
    iq = _make_iq()
    out = to_amplitude_phase(iq)
    # Amplitude ≥ 0
    assert np.all(out[0] >= 0.0)
    # Phase difference ∈ [-π, π]
    assert np.all(out[1] >= -np.pi - 1e-6)
    assert np.all(out[1] <= np.pi + 1e-6)


# ── 5. IQ_AMP_PHASE ──────────────────────────────────────────────────────────

def test_iq_amp_phase_single_shape():
    iq = _make_iq()
    out = to_iq_amp_phase(iq)
    assert out.shape == (4, 128)
    assert out.dtype == np.float32


def test_iq_amp_phase_channels():
    iq = _make_iq()
    out = to_iq_amp_phase(iq)
    # I and Q pass-through
    np.testing.assert_array_equal(out[0], iq[0])
    np.testing.assert_array_equal(out[1], iq[1])
    # Amplitude channel ≥ 0
    assert np.all(out[2] >= 0.0)
    # Phase-diff at 0 is 0
    assert out[3, 0] == 0.0


def test_iq_amp_phase_batch():
    iq = _make_batch()
    out = to_iq_amp_phase(iq)
    assert out.shape == (4, 4, 128)


# ── 6. Generic dispatcher ─────────────────────────────────────────────────────

def test_apply_representation_dispatch():
    iq = _make_iq()
    for rep, expected_channels in [
        (Representation.RAW_IQ, 2),
        (Representation.IQ_AMPLITUDE, 3),
        (Representation.AMPLITUDE_PHASE, 2),
        (Representation.IQ_AMP_PHASE, 4),
    ]:
        out = apply_representation(iq, rep)
        assert out.shape[0] == expected_channels, f"{rep}: expected {expected_channels} channels"
        assert out.dtype == np.float32


def test_apply_representation_accepts_string():
    iq = _make_iq()
    out = apply_representation(iq, "IQ_AMPLITUDE")
    assert out.shape == (3, 128)


def test_apply_representation_invalid_string():
    iq = _make_iq()
    with pytest.raises(ValueError):
        apply_representation(iq, "NONEXISTENT_REPR")


# ── 7. Validation ─────────────────────────────────────────────────────────────

def test_reject_non_ndarray():
    with pytest.raises(TypeError):
        to_raw_iq([[1.0, 2.0], [3.0, 4.0]])  # type: ignore


def test_reject_wrong_channel_dim():
    with pytest.raises(ValueError):
        to_iq_amplitude(np.zeros((3, 128), dtype=np.float32))


def test_reject_1d_input():
    with pytest.raises(ValueError):
        to_raw_iq(np.zeros(128, dtype=np.float32))


def test_reject_nan_input():
    iq = _make_iq()
    iq[0, 5] = np.nan
    with pytest.raises(ValueError):
        to_iq_amplitude(iq)


def test_reject_inf_input():
    iq = _make_iq()
    iq[1, 10] = np.inf
    with pytest.raises(ValueError):
        to_amplitude_phase(iq)


# ── 8. Determinism ────────────────────────────────────────────────────────────

def test_transforms_deterministic():
    iq = _make_iq(seed=7)
    for fn in [to_raw_iq, to_iq_amplitude, to_amplitude_phase, to_iq_amp_phase]:
        out1 = fn(iq)
        out2 = fn(iq)
        np.testing.assert_array_equal(out1, out2)


# ── 9. float32 output ─────────────────────────────────────────────────────────

def test_float64_input_returns_float32():
    iq = _make_iq().astype(np.float64)
    for fn in [to_raw_iq, to_iq_amplitude, to_amplitude_phase, to_iq_amp_phase]:
        out = fn(iq)
        assert out.dtype == np.float32
