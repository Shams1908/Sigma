"""
Unit tests for ml/generators/augmentation.py  (M6 DomainAugmentor).
"""
from __future__ import annotations

import numpy as np
import pytest

from ml.generators.augmentation import DomainAugmentor


SR = 80_000.0  # sample rate used across tests


def _iq(n: int = 256, seed: int = 42) -> np.ndarray:
    """Return a [2, N] float32 canonical IQ array."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((2, n)).astype(np.float32)


# ── 1. No-op (all disabled explicitly) ───────────────────────────────────────

def test_noop_when_all_disabled():
    """With all augmentations explicitly disabled, output equals input."""
    aug = DomainAugmentor(
        seed=0,
        amplitude_scale_range=None,
        enable_multipath=False,
        enable_iq_imbalance=False,
        enable_freq_offset=False,
        enable_phase_offset=False,
        enable_phase_noise=False,
        enable_dc_offset=False,
        snr_db_range=None,
    )
    iq = _iq()
    out = aug.augment(iq, SR)
    np.testing.assert_array_equal(out, iq)


def test_noop_shape_and_dtype():
    """Shape and dtype are preserved regardless of augmentation config."""
    aug = DomainAugmentor(
        seed=0,
        amplitude_scale_range=None,
        enable_multipath=False,
        enable_iq_imbalance=False,
        enable_freq_offset=False,
        enable_phase_offset=False,
        enable_phase_noise=False,
        enable_dc_offset=False,
        snr_db_range=None,
    )
    iq = _iq()
    out = aug.augment(iq, SR)
    assert out.shape == (2, 256)
    assert out.dtype == np.float32


# ── 2. Amplitude scaling ──────────────────────────────────────────────────────

def test_amplitude_scaling_changes_signal():
    aug = DomainAugmentor(seed=1, amplitude_scale_range=(0.5, 2.0))
    iq = _iq()
    out = aug.augment(iq, SR)
    assert not np.array_equal(out, iq)


def test_amplitude_scaling_shape_preserved():
    aug = DomainAugmentor(seed=2, amplitude_scale_range=(0.9, 1.1))
    iq = _iq()
    out = aug.augment(iq, SR)
    assert out.shape == iq.shape


# ── 3. Multipath ──────────────────────────────────────────────────────────────

def test_multipath_changes_signal():
    aug = DomainAugmentor(seed=3, enable_multipath=True, multipath_num_taps=4)
    iq = _iq()
    out = aug.augment(iq, SR)
    assert not np.array_equal(out, iq)
    assert out.shape == iq.shape
    assert np.all(np.isfinite(out))


# ── 4. IQ imbalance ───────────────────────────────────────────────────────────

def test_iq_imbalance_changes_signal():
    aug = DomainAugmentor(
        seed=4,
        enable_iq_imbalance=True,
        iq_amplitude_imbalance_max=2.0,
        iq_phase_imbalance_max_deg=10.0,
    )
    iq = _iq()
    out = aug.augment(iq, SR)
    assert not np.array_equal(out, iq)
    assert out.shape == iq.shape


# ── 5. Frequency offset ───────────────────────────────────────────────────────

def test_freq_offset_changes_signal():
    aug = DomainAugmentor(seed=5, enable_freq_offset=True, freq_offset_max_hz=2000.0)
    iq = _iq()
    out = aug.augment(iq, SR)
    assert not np.array_equal(out, iq)
    assert out.shape == iq.shape


# ── 6. Phase offset ───────────────────────────────────────────────────────────

def test_phase_offset_changes_signal():
    aug = DomainAugmentor(seed=6, enable_phase_offset=True)
    iq = _iq()
    out = aug.augment(iq, SR)
    assert not np.array_equal(out, iq)


# ── 7. Phase noise ────────────────────────────────────────────────────────────

def test_phase_noise_changes_signal():
    aug = DomainAugmentor(seed=7, enable_phase_noise=True, phase_noise_std_deg=1.0)
    iq = _iq()
    out = aug.augment(iq, SR)
    assert not np.array_equal(out, iq)
    assert np.all(np.isfinite(out))


# ── 8. DC offset ──────────────────────────────────────────────────────────────

def test_dc_offset_changes_signal():
    aug = DomainAugmentor(seed=8, enable_dc_offset=True, dc_offset_max=0.1)
    iq = _iq()
    out = aug.augment(iq, SR)
    assert not np.array_equal(out, iq)


# ── 9. AWGN ───────────────────────────────────────────────────────────────────

def test_awgn_changes_signal():
    aug = DomainAugmentor(seed=9, snr_db_range=(10.0, 20.0))
    iq = _iq()
    out = aug.augment(iq, SR)
    assert not np.array_equal(out, iq)
    assert np.all(np.isfinite(out))


# ── 10. All augmentations combined ────────────────────────────────────────────

def test_all_augmentations_combined():
    aug = DomainAugmentor(
        seed=42,
        amplitude_scale_range=(0.8, 1.2),
        enable_multipath=True,
        multipath_num_taps=4,
        enable_iq_imbalance=True,
        enable_freq_offset=True,
        freq_offset_max_hz=1000.0,
        enable_phase_offset=True,
        enable_phase_noise=True,
        phase_noise_std_deg=0.5,
        enable_dc_offset=True,
        snr_db_range=(15.0, 30.0),
    )
    iq = _iq(256)
    out = aug.augment(iq, SR)
    assert out.shape == (2, 256)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


# ── 11. Determinism ───────────────────────────────────────────────────────────

def test_determinism_same_seed():
    aug1 = DomainAugmentor(seed=77, enable_freq_offset=True, snr_db_range=(10.0, 20.0))
    aug2 = DomainAugmentor(seed=77, enable_freq_offset=True, snr_db_range=(10.0, 20.0))
    iq = _iq()
    out1 = aug1.augment(iq, SR)
    out2 = aug2.augment(iq, SR)
    np.testing.assert_array_equal(out1, out2)


def test_different_seeds_different_output():
    aug1 = DomainAugmentor(seed=10, snr_db_range=(10.0, 20.0))
    aug2 = DomainAugmentor(seed=20, snr_db_range=(10.0, 20.0))
    iq = _iq()
    out1 = aug1.augment(iq, SR)
    out2 = aug2.augment(iq, SR)
    assert not np.array_equal(out1, out2)


# ── 12. Input validation ─────────────────────────────────────────────────────

def test_reject_non_ndarray():
    aug = DomainAugmentor()
    with pytest.raises(TypeError):
        aug.augment([[1.0, 0.0], [0.0, 1.0]], SR)  # type: ignore


def test_reject_wrong_shape():
    aug = DomainAugmentor()
    with pytest.raises(ValueError):
        aug.augment(np.zeros((3, 128), dtype=np.float32), SR)


def test_reject_nan_input():
    aug = DomainAugmentor()
    iq = _iq()
    iq[0, 0] = np.nan
    with pytest.raises(ValueError):
        aug.augment(iq, SR)


# ── 13. External RNG override ────────────────────────────────────────────────

def test_external_rng_override():
    aug = DomainAugmentor(seed=0, snr_db_range=(5.0, 20.0))
    iq = _iq()
    rng = np.random.default_rng(999)
    out1 = aug.augment(iq, SR, rng=np.random.default_rng(999))
    out2 = aug.augment(iq, SR, rng=np.random.default_rng(999))
    np.testing.assert_array_equal(out1, out2)


# ── 14. Export ───────────────────────────────────────────────────────────────

def test_exported_from_generators():
    from ml.generators import DomainAugmentor as DA  # noqa: PLC0415
    assert DA is DomainAugmentor

# ── 15. M6 default augmentation configuration ────────────────────────────────

def test_m6_default_all_augmentations_enabled():
    """
    Verify that the M6 default DomainAugmentor has all augmentation switches
    enabled.  This test documents the intended M6 default configuration.
    """
    aug = DomainAugmentor()

    # All boolean switches must be True
    assert aug.enable_multipath is True
    assert aug.enable_iq_imbalance is True
    assert aug.enable_freq_offset is True
    assert aug.enable_phase_offset is True
    assert aug.enable_phase_noise is True
    assert aug.enable_dc_offset is True

    # Amplitude scaling must be a non-None range (not disabled)
    assert aug.amplitude_scale_range is not None
    lo, hi = aug.amplitude_scale_range
    assert lo < hi

    # AWGN must be a non-None range (not disabled)
    assert aug.snr_db_range is not None
    lo_snr, hi_snr = aug.snr_db_range
    assert lo_snr < hi_snr


def test_m6_default_augmentor_changes_signal():
    """
    With M6 defaults, augment() must actually modify the signal
    (not pass through unchanged).  This is not an inference-pipeline test —
    DomainAugmentor is training/data-augmentation only.
    """
    aug = DomainAugmentor(seed=42)
    iq = _iq()
    out = aug.augment(iq, SR)
    assert out.shape == iq.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    # Output must differ from input (all augmentations are on)
    assert not np.array_equal(out, iq)
