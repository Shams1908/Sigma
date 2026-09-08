"""
Unit tests for ml/generators/channel/multipath.py  (M6 Rayleigh multipath channel).
"""
from __future__ import annotations

import numpy as np
import pytest

from ml.generators.channel.multipath import RayleighMultipathChannel


def _sine(n: int = 512, freq: float = 0.05, seed: int = 0) -> np.ndarray:
    """Return a deterministic complex sine-wave of length n."""
    t = np.arange(n, dtype=np.float32)
    return (np.exp(2j * np.pi * freq * t)).astype(np.complex64)


# ── 1. Construction guards ────────────────────────────────────────────────────

def test_invalid_num_taps():
    with pytest.raises(ValueError):
        RayleighMultipathChannel(num_taps=0)


def test_invalid_decay_factor():
    with pytest.raises(ValueError):
        RayleighMultipathChannel(decay_factor=0.0)
    with pytest.raises(ValueError):
        RayleighMultipathChannel(decay_factor=-1.0)


# ── 2. Output shape and dtype ─────────────────────────────────────────────────

def test_output_shape_preserved():
    ch = RayleighMultipathChannel(num_taps=4, decay_factor=1.0)
    rng = np.random.default_rng(42)
    w = _sine(256)
    out = ch.apply(w, rng)
    assert out.shape == w.shape


def test_output_dtype():
    ch = RayleighMultipathChannel()
    rng = np.random.default_rng(0)
    w = _sine(128)
    out = ch.apply(w, rng)
    assert out.dtype == np.complex64


# ── 3. Finite output ─────────────────────────────────────────────────────────

def test_output_finite():
    ch = RayleighMultipathChannel(num_taps=6, decay_factor=1.0)
    rng = np.random.default_rng(99)
    w = _sine(512)
    out = ch.apply(w, rng)
    assert np.all(np.isfinite(out))


# ── 4. Determinism ────────────────────────────────────────────────────────────

def test_deterministic_same_seed():
    ch = RayleighMultipathChannel(num_taps=4, decay_factor=2.0)
    w = _sine(128)
    out1 = ch.apply(w, np.random.default_rng(7))
    out2 = ch.apply(w, np.random.default_rng(7))
    np.testing.assert_array_equal(out1, out2)


def test_different_seeds_different_output():
    ch = RayleighMultipathChannel(num_taps=4, decay_factor=1.0)
    w = _sine(256)
    out1 = ch.apply(w, np.random.default_rng(1))
    out2 = ch.apply(w, np.random.default_rng(2))
    assert not np.array_equal(out1, out2)


# ── 5. Input validation ───────────────────────────────────────────────────────

def test_reject_non_ndarray():
    ch = RayleighMultipathChannel()
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError):
        ch.apply([1.0 + 0j, 2.0 + 0j], rng)  # type: ignore


def test_reject_2d_input():
    ch = RayleighMultipathChannel()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        ch.apply(np.zeros((4, 32), dtype=np.complex64), rng)


def test_reject_empty_input():
    ch = RayleighMultipathChannel()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        ch.apply(np.array([], dtype=np.complex64), rng)


# ── 6. Single-tap is an identity-like operation ───────────────────────────────

def test_single_tap_energy_preserved():
    """Single tap should scale energy but not spread it."""
    ch = RayleighMultipathChannel(num_taps=1, decay_factor=1.0)
    w = _sine(256)
    rng = np.random.default_rng(42)
    out = ch.apply(w, rng)
    # With normalization, energy should be approximately preserved
    in_power = float(np.mean(np.abs(w) ** 2))
    out_power = float(np.mean(np.abs(out) ** 2))
    assert np.isclose(in_power, out_power, rtol=0.1)


# ── 7. Channel export through generators package ─────────────────────────────

def test_exported_from_generators():
    from ml.generators.channel import RayleighMultipathChannel as RMC  # noqa: PLC0415
    assert RMC is RayleighMultipathChannel

    from ml.generators import RayleighMultipathChannel as RMC2  # noqa: PLC0415
    assert RMC2 is RayleighMultipathChannel
