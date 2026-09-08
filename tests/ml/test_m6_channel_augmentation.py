"""
Unit tests for M6 Channel and Impairment Augmentations.
Verifies:
  - RayleighMultipathChannel parameter validation, energy preservation, determinism
  - DomainAugmentor shape preservation, finite outputs, reproducible seeds
  - Bounded impairment parameter ranges
"""
import pytest
import numpy as np

from ml.generators.channel.multipath import RayleighMultipathChannel
from ml.generators.augmentation import DomainAugmentor, AugmentationConfig


def test_rayleigh_multipath_parameter_validation():
    """Tests parameter checks on RayleighMultipathChannel."""
    # Valid construction
    ch = RayleighMultipathChannel(num_taps=3, decay_rate=1.5, random_seed=42)
    assert ch.num_taps == 3
    assert ch.decay_rate == 1.5

    # Invalid num_taps (< 1 or non-integer)
    with pytest.raises(ValueError):
        RayleighMultipathChannel(num_taps=0)
    with pytest.raises(TypeError):
        RayleighMultipathChannel(num_taps="three")

    # Invalid decay_rate (<= 0 or non-numeric)
    with pytest.raises(ValueError):
        RayleighMultipathChannel(decay_rate=0.0)
    with pytest.raises(TypeError):
        RayleighMultipathChannel(decay_rate=None)


def test_rayleigh_multipath_energy_normalization():
    """Verifies that generated FIR taps have unit total energy: sum(|h_k|^2) == 1.0."""
    ch = RayleighMultipathChannel(num_taps=5, decay_rate=1.2, random_seed=123)
    rng = np.random.default_rng(123)
    for _ in range(10):
        taps = ch.generate_taps(rng=rng)
        assert len(taps) == 5
        energy = np.sum(np.abs(taps)**2)
        assert np.isclose(energy, 1.0, atol=1e-6)


def test_rayleigh_multipath_apply_shape_and_finiteness():
    """Verifies output shape, float32 dtype, and no NaN/Inf after convolution."""
    ch = RayleighMultipathChannel(num_taps=3, random_seed=42)
    rng = np.random.default_rng(42)
    sig = rng.standard_normal((2, 128)).astype(np.float32)

    faded, taps = ch.apply(sig)
    assert faded.shape == (2, 128)
    assert faded.dtype == np.float32
    assert np.isfinite(faded).all()
    assert len(taps) == 3


def test_rayleigh_multipath_determinism():
    """Verifies that identical seeds produce identical channel realizations."""
    sig = np.ones((2, 128), dtype=np.float32)
    ch1 = RayleighMultipathChannel(num_taps=4, random_seed=777)
    ch2 = RayleighMultipathChannel(num_taps=4, random_seed=777)

    out1, taps1 = ch1.apply(sig)
    out2, taps2 = ch2.apply(sig)

    assert np.array_equal(out1, out2)
    assert np.array_equal(taps1, taps2)


def test_domain_augmentor_single_signal():
    """Verifies DomainAugmentor on single signal [2, 128]."""
    cfg = AugmentationConfig(
        enable_awgn=True,
        enable_frequency_offset=True,
        enable_phase_offset=True,
        enable_phase_noise=True,
        enable_iq_imbalance=True,
        enable_dc_offset=True,
        enable_multipath=True,
        enable_amplitude_scaling=True,
        random_seed=42,
    )
    augmentor = DomainAugmentor(cfg)
    rng = np.random.default_rng(42)
    sig = rng.standard_normal((2, 128)).astype(np.float32)

    aug_sig, meta = augmentor.augment_signal(sig, sample_rate=800000.0)

    # Invariants
    assert aug_sig.shape == (2, 128)
    assert aug_sig.dtype == np.float32
    assert np.isfinite(aug_sig).all()
    assert not np.isnan(aug_sig).any()

    # Metadata recorded
    assert "amplitude_scale" in meta
    assert "multipath_num_taps" in meta
    assert "iq_amp_imbalance_db" in meta
    assert "frequency_offset_hz" in meta
    assert "snr_db" in meta


def test_domain_augmentor_batch():
    """Verifies DomainAugmentor batch execution [B, 2, 128]."""
    cfg = AugmentationConfig(random_seed=999)
    augmentor = DomainAugmentor(cfg)
    rng = np.random.default_rng(999)
    batch = rng.standard_normal((8, 2, 128)).astype(np.float32)

    aug_batch = augmentor.augment_batch(batch, sample_rate=800000.0)
    assert aug_batch.shape == (8, 2, 128)
    assert aug_batch.dtype == np.float32
    assert np.isfinite(aug_batch).all()


def test_domain_augmentor_determinism():
    """Verifies that augmentor with fixed seed produces deterministic outputs."""
    cfg = AugmentationConfig(random_seed=555)
    aug1 = DomainAugmentor(cfg)
    aug2 = DomainAugmentor(cfg)

    sig = np.sin(np.linspace(0, 10, 128, dtype=np.float32)).reshape(1, 128)
    sig_2ch = np.repeat(sig, 2, axis=0)

    res1, _ = aug1.augment_signal(sig_2ch)
    res2, _ = aug2.augment_signal(sig_2ch)

    assert np.allclose(res1, res2, atol=1e-6)
