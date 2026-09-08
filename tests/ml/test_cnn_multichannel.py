"""
Tests for the M6 multi-channel extension to RawIQCNN and the
representation-aware inference path.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from ml.cnn_model.architecture import RawIQCNN
from ml.cnn_model.inference import (
    _apply_checkpoint_representation,
    clear_cnn_cache,
    get_cnn_model,
)


# ── 1. Architecture — configurable in_channels ───────────────────────────────

def test_2ch_model_shape():
    """Backward-compatible 2-channel model (M5 baseline)."""
    model = RawIQCNN(num_classes=11, in_channels=2)
    x = torch.randn(4, 2, 128)
    out = model(x)
    assert out.shape == (4, 11)


def test_3ch_model_shape():
    """3-channel model for IQ_AMPLITUDE."""
    model = RawIQCNN(num_classes=11, in_channels=3)
    x = torch.randn(4, 3, 128)
    out = model(x)
    assert out.shape == (4, 11)


def test_4ch_model_shape():
    """4-channel model for IQ_AMP_PHASE."""
    model = RawIQCNN(num_classes=11, in_channels=4)
    x = torch.randn(4, 4, 128)
    out = model(x)
    assert out.shape == (4, 11)


def test_in_channels_stored():
    model = RawIQCNN(num_classes=11, in_channels=3)
    assert model.in_channels == 3


def test_default_in_channels():
    model = RawIQCNN()
    assert model.in_channels == 2


# ── 2. Checkpoint save/load with in_channels metadata ────────────────────────

def test_checkpoint_roundtrip_3ch():
    """3-channel checkpoint saves and loads in_channels correctly."""
    model = RawIQCNN(num_classes=11, in_channels=3)
    model.eval()

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "m6_test.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "rms_factor": 0.01,
                "in_channels": 3,
                "representation": "IQ_AMPLITUDE",
            },
            ckpt_path,
        )

        clear_cnn_cache()
        loaded_model, rms, rep = get_cnn_model(ckpt_path)
        assert rms == pytest.approx(0.01)
        assert rep == "IQ_AMPLITUDE"
        assert loaded_model.in_channels == 3

        # Forward pass with 3-channel input
        x = torch.randn(2, 3, 128)
        with torch.no_grad():
            out = loaded_model(x)
        assert out.shape == (2, 11)
        clear_cnn_cache()


def test_checkpoint_roundtrip_2ch_legacy():
    """Legacy 2-channel checkpoint (no in_channels key) loads as RAW_IQ."""
    model = RawIQCNN(num_classes=11, in_channels=2)
    model.eval()

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "m5_legacy.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "rms_factor": 0.005,
                # Intentionally omit in_channels and representation
            },
            ckpt_path,
        )

        clear_cnn_cache()
        loaded_model, rms, rep = get_cnn_model(ckpt_path)
        assert rms == pytest.approx(0.005)
        assert rep is None
        assert loaded_model.in_channels == 2
        clear_cnn_cache()


# ── 3. _apply_checkpoint_representation ──────────────────────────────────────

def test_apply_none_representation_passthrough():
    """None → RAW_IQ passthrough."""
    X = np.random.randn(4, 2, 128).astype(np.float32)
    out = _apply_checkpoint_representation(X, None)
    np.testing.assert_array_equal(out, X)


def test_apply_raw_iq_representation_passthrough():
    X = np.random.randn(4, 2, 128).astype(np.float32)
    out = _apply_checkpoint_representation(X, "RAW_IQ")
    np.testing.assert_array_equal(out, X)


def test_apply_iq_amplitude_representation():
    X = np.random.randn(4, 2, 128).astype(np.float32)
    out = _apply_checkpoint_representation(X, "IQ_AMPLITUDE")
    assert out.shape == (4, 3, 128)
    assert out.dtype == np.float32
    # I and Q channels preserved
    np.testing.assert_array_equal(out[:, 0, :], X[:, 0, :])
    np.testing.assert_array_equal(out[:, 1, :], X[:, 1, :])


def test_apply_iq_amp_phase_representation():
    X = np.random.randn(3, 2, 128).astype(np.float32)
    out = _apply_checkpoint_representation(X, "IQ_AMP_PHASE")
    assert out.shape == (3, 4, 128)


def test_apply_unknown_representation_falls_back_to_raw():
    """Unknown representation should fall back to RAW_IQ with a warning."""
    import warnings  # noqa: PLC0415
    X = np.random.randn(2, 2, 128).astype(np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _apply_checkpoint_representation(X, "FUTURE_UNKNOWN_REPR")
    assert out.shape == (2, 2, 128)
    assert any("falling back" in str(w.message).lower() for w in caught)


# ── 4. Integration: 3-channel inference end-to-end ───────────────────────────

def test_3ch_inference_e2e():
    """Simulate the full inference flow for a 3-channel IQ_AMPLITUDE checkpoint."""
    model = RawIQCNN(num_classes=11, in_channels=3)
    model.eval()

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "m6_e2e.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "rms_factor": 1.0,
                "in_channels": 3,
                "representation": "IQ_AMPLITUDE",
            },
            ckpt_path,
        )

        clear_cnn_cache()

        from ml.cnn_model.inference import predict_iq  # noqa: PLC0415

        # predict_iq takes raw [2, 128] IQ input; transformation happens internally
        sample = np.random.randn(2, 128).astype(np.float32)
        result = predict_iq(sample, model_path=ckpt_path)

        assert isinstance(result["class_index"], int)
        assert isinstance(result["class_name"], str)
        assert result["probabilities"].shape == (11,)
        assert np.isclose(result["probabilities"].sum(), 1.0, atol=1e-4)

        clear_cnn_cache()


# ── 5. Cache keyed by model_path (regression for Correction 1) ───────────────

def test_cache_independence_two_paths():
    """
    Two different checkpoint paths must be cached and returned independently.
    Loading path A must not pollute the cache for path B.
    """
    from ml.cnn_model.inference import _MODEL_CACHE  # noqa: PLC0415

    model_2ch = RawIQCNN(num_classes=11, in_channels=2)
    model_2ch.eval()
    model_3ch = RawIQCNN(num_classes=11, in_channels=3)
    model_3ch.eval()

    with tempfile.TemporaryDirectory() as td:
        path_a = os.path.join(td, "ckpt_a.pt")
        path_b = os.path.join(td, "ckpt_b.pt")

        torch.save(
            {
                "model_state_dict": model_2ch.state_dict(),
                "rms_factor": 0.01,
                # no in_channels / representation — legacy 2-ch
            },
            path_a,
        )
        torch.save(
            {
                "model_state_dict": model_3ch.state_dict(),
                "rms_factor": 0.02,
                "in_channels": 3,
                "representation": "IQ_AMPLITUDE",
            },
            path_b,
        )

        clear_cnn_cache()

        # Load A
        m_a, rms_a, rep_a = get_cnn_model(path_a)
        assert rms_a == pytest.approx(0.01)
        assert rep_a is None
        assert m_a.in_channels == 2

        # Load B — must be independent, not reuse A's cache entry
        m_b, rms_b, rep_b = get_cnn_model(path_b)
        assert rms_b == pytest.approx(0.02)
        assert rep_b == "IQ_AMPLITUDE"
        assert m_b.in_channels == 3

        # Both paths are independently cached
        assert path_a in _MODEL_CACHE
        assert path_b in _MODEL_CACHE

        # Re-requesting A returns the same object (from cache, no reload)
        m_a2, rms_a2, rep_a2 = get_cnn_model(path_a)
        assert m_a2 is m_a
        assert rms_a2 == pytest.approx(0.01)

        # Re-requesting B returns the same object
        m_b2, _, _ = get_cnn_model(path_b)
        assert m_b2 is m_b

        clear_cnn_cache()


def test_clear_cnn_cache_removes_all_paths():
    """clear_cnn_cache() must clear all cached paths."""
    from ml.cnn_model.inference import _MODEL_CACHE  # noqa: PLC0415

    model = RawIQCNN(num_classes=11, in_channels=2)

    with tempfile.TemporaryDirectory() as td:
        for name in ("x.pt", "y.pt"):
            torch.save(
                {"model_state_dict": model.state_dict(), "rms_factor": 1.0},
                os.path.join(td, name),
            )
            clear_cnn_cache()
            get_cnn_model(os.path.join(td, name))

        assert len(_MODEL_CACHE) == 1  # only last loaded
        clear_cnn_cache()
        assert len(_MODEL_CACHE) == 0
