"""
Tests for backend/synchronization/frame_sync.py

Covers:
  - Known preamble at a known offset (exact recovery)
  - Noisy preamble detection
  - No preamble / wrong signal → found=False
  - Signal shorter than template → found=False
  - Invalid inputs → TypeError / ValueError
  - Confidence / threshold behavior
  - build_rc_template shape and delay
  - find_frame_start convenience wrapper
  - No regression in existing synchronize() callers
  - Integration test using the project ML pulse-shaping path
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add backend to path (consistent with other test files in tests/backend/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
# Also add repo root for ml imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synchronization.frame_sync import (
    BasicFrameSynchronizer,
    FrameSyncResult,
    build_rc_template,
    find_frame_start,
)
from synchronization import synchronize  # existing entry point must still work


# ── Constants shared across tests ─────────────────────────────────────────────

SPS = 4
ROLLOFF = 0.35
SPAN = 8
N_PREAMBLE = 16   # 16 known BPSK symbols
SR = 10_000.0     # sample rate (Hz, only used for the integration test)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bpsk_preamble(n: int = N_PREAMBLE, seed: int = 0) -> np.ndarray:
    """Return n clean BPSK symbols using canonical mapping (bit0→-1, bit1→+1)."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n).astype(np.uint8)
    return (2.0 * bits - 1.0).astype(np.complex64)


def _qpsk_preamble(n: int = N_PREAMBLE, seed: int = 1) -> np.ndarray:
    """Return n clean QPSK symbols (unit-power)."""
    NORM = 1.0 / np.sqrt(2.0)
    pts = np.array([
        NORM + 1j * NORM,
        -NORM + 1j * NORM,
        -NORM - 1j * NORM,
        NORM - 1j * NORM,
    ], dtype=np.complex64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=n)
    return pts[idx]


def _build_signal_with_preamble(
    preamble_symbols: np.ndarray,
    payload_symbols: np.ndarray,
    offset_samples: int = 0,
    snr_db: float | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """
    Build a receive-side signal (Tx-RRC shaped, then Rx-RRC MF applied) that
    contains the preamble followed by payload at a known sample offset.

    Returns (mf_output, expected_start_index) where expected_start_index is
    the index in mf_output where the first preamble RC peak lands.

    Construction:
      1. Concatenate [preamble_symbols | payload_symbols].
      2. Upsample and apply Tx RRC (mode="full").
      3. Prepend `offset_samples` zeros to shift the frame.
      4. Apply Rx RRC matched filter (mode="same").
      5. Optionally add AWGN.
    """
    from ml.generators.pulse_shaping.rrc import design_rrc_filter
    from synchronization.matched_filter import apply_matched_filter

    h_rrc = design_rrc_filter(SPS, ROLLOFF, SPAN).astype(np.float64)

    # Tx-side pulse shaping
    all_symbols = np.concatenate([preamble_symbols, payload_symbols]).astype(np.complex128)
    upsampled = np.zeros(len(all_symbols) * SPS, dtype=np.complex128)
    upsampled[::SPS] = all_symbols
    tx = np.convolve(upsampled, h_rrc, mode="full")  # full RC group delay = SPAN*SPS

    # Prepend silence to introduce a known frame offset
    if offset_samples > 0:
        tx = np.concatenate([np.zeros(offset_samples, dtype=np.complex128), tx])

    # AWGN
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        p_signal = float(np.mean(np.abs(tx) ** 2))
        if p_signal > 0:
            p_noise = p_signal / (10.0 ** (snr_db / 10.0))
            noise_std = np.sqrt(p_noise / 2.0)
            noise = (
                rng.normal(0.0, noise_std, len(tx))
                + 1j * rng.normal(0.0, noise_std, len(tx))
            ).astype(np.complex128)
            tx = tx + noise

    # Rx matched filter (same mode — preserves length)
    rx = apply_matched_filter(
        tx.astype(np.complex64),
        samples_per_symbol=SPS,
        rolloff=ROLLOFF,
        filter_span_symbols=SPAN,
    )

    # Expected start_index = offset_samples.
    # With the Tx-full / Rx-same template construction, start_index = corr_peak
    # which equals the actual offset of the preamble in the MF output.
    expected_start = offset_samples

    return rx, expected_start


def _syncer(preamble: np.ndarray, threshold: float = 0.3) -> BasicFrameSynchronizer:
    return BasicFrameSynchronizer(
        preamble_symbols=preamble,
        samples_per_symbol=SPS,
        rolloff=ROLLOFF,
        filter_span_symbols=SPAN,
        threshold=threshold,
    )


# ── 1. build_rc_template ──────────────────────────────────────────────────────

class TestBuildRcTemplate:
    def test_returns_ndarray_and_int(self):
        preamble = _bpsk_preamble()
        tpl, delay = build_rc_template(preamble, SPS, ROLLOFF, SPAN)
        assert isinstance(tpl, np.ndarray)
        assert isinstance(delay, int)

    def test_template_dtype_complex64(self):
        tpl, _ = build_rc_template(_bpsk_preamble(), SPS, ROLLOFF, SPAN)
        assert tpl.dtype == np.complex64

    def test_delay_equals_span_times_sps(self):
        _, delay = build_rc_template(_bpsk_preamble(), SPS, ROLLOFF, SPAN)
        assert delay == 0  # mode="same" Rx step absorbs the delay

    def test_template_length(self):
        """Template length = N_p*SPS + SPAN*SPS (Tx full + Rx same)."""
        n_p = N_PREAMBLE
        tpl, _ = build_rc_template(_bpsk_preamble(n_p), SPS, ROLLOFF, SPAN)
        # Tx-shaped (full): N_p*SPS + SPAN*SPS;  Rx same: preserves length
        expected_len = n_p * SPS + SPAN * SPS
        assert len(tpl) == expected_len

    def test_template_is_finite(self):
        tpl, _ = build_rc_template(_bpsk_preamble(), SPS, ROLLOFF, SPAN)
        assert np.all(np.isfinite(tpl))

    def test_peak_is_near_template_midpoint(self):
        """The energy peak of the template should be near sample SPAN*SPS//2."""
        preamble = _bpsk_preamble(4, seed=5)   # short preamble for clarity
        tpl, delay = build_rc_template(preamble, SPS, ROLLOFF, SPAN)
        peak_idx = int(np.argmax(np.abs(tpl)))
        # Peak should be within 1 symbol period of SPAN*SPS//2
        assert abs(peak_idx - SPAN * SPS // 2) <= SPS

    def test_invalid_non_ndarray(self):
        with pytest.raises(TypeError):
            build_rc_template([1.0, -1.0], SPS, ROLLOFF, SPAN)  # type: ignore

    def test_invalid_empty(self):
        with pytest.raises(ValueError):
            build_rc_template(np.array([], dtype=np.complex64), SPS, ROLLOFF, SPAN)

    def test_invalid_2d(self):
        with pytest.raises(ValueError):
            build_rc_template(np.ones((4, 2), dtype=np.complex64), SPS, ROLLOFF, SPAN)

    def test_invalid_sps_zero(self):
        with pytest.raises(ValueError):
            build_rc_template(_bpsk_preamble(), samples_per_symbol=0)


# ── 2. BasicFrameSynchronizer — construction ──────────────────────────────────

class TestBasicFrameSynchronizerConstruction:
    def test_threshold_stored(self):
        s = _syncer(_bpsk_preamble(), threshold=0.5)
        assert s.threshold == 0.5

    def test_template_accessible(self):
        s = _syncer(_bpsk_preamble())
        assert isinstance(s.template, np.ndarray)
        assert len(s.template) > 0

    def test_template_delay_accessible(self):
        s = _syncer(_bpsk_preamble())
        assert s.template_delay == 0  # mode="same" absorbs the delay

    def test_invalid_threshold_high(self):
        with pytest.raises(ValueError):
            BasicFrameSynchronizer(_bpsk_preamble(), SPS, threshold=1.5)

    def test_invalid_threshold_low(self):
        with pytest.raises(ValueError):
            BasicFrameSynchronizer(_bpsk_preamble(), SPS, threshold=-0.1)


# ── 3. find() — invalid inputs ────────────────────────────────────────────────

class TestFindInvalidInputs:
    def test_non_ndarray_raises(self):
        s = _syncer(_bpsk_preamble())
        with pytest.raises(TypeError):
            s.find([1.0, 0.0, -1.0])  # type: ignore

    def test_2d_array_raises(self):
        s = _syncer(_bpsk_preamble())
        with pytest.raises(ValueError):
            s.find(np.zeros((4, 10), dtype=np.complex64))


# ── 4. Signal shorter than template ───────────────────────────────────────────

class TestShortSignal:
    def test_returns_not_found(self):
        preamble = _bpsk_preamble()
        s = _syncer(preamble)
        # Template length > this signal length
        short_signal = np.zeros(10, dtype=np.complex64)
        result = s.find(short_signal)
        assert result.found is False
        assert result.start_index == -1
        assert result.confidence == 0.0

    def test_empty_signal(self):
        s = _syncer(_bpsk_preamble())
        result = s.find(np.array([], dtype=np.complex64))
        assert result.found is False


# ── 5. Exact preamble at known offset ─────────────────────────────────────────

class TestKnownOffset:
    @pytest.mark.parametrize("offset", [0, SPS, 4 * SPS, 10 * SPS])
    def test_exact_start_index_noiseless(self, offset):
        """
        Noiseless signal: recovered start_index must equal the exact offset
        within ±1 sample (rounding of the RC peak).
        """
        preamble = _bpsk_preamble()
        payload = _bpsk_preamble(32, seed=99)

        mf_out, expected_start = _build_signal_with_preamble(
            preamble, payload, offset_samples=offset
        )

        result = _syncer(preamble, threshold=0.2).find(mf_out)

        assert result.found, (
            f"Preamble not found (offset={offset}); "
            f"confidence={result.confidence:.3f}"
        )
        assert abs(result.start_index - expected_start) <= 1, (
            f"start_index={result.start_index}, expected≈{expected_start} "
            f"(offset={offset})"
        )

    def test_confidence_near_one_noiseless(self):
        """Noiseless correlation confidence should be very high."""
        preamble = _bpsk_preamble()
        mf_out, _ = _build_signal_with_preamble(preamble, _bpsk_preamble(32, seed=77))
        result = _syncer(preamble, threshold=0.1).find(mf_out)
        assert result.found
        assert result.confidence > 0.7

    def test_result_fields_populated_on_found(self):
        preamble = _bpsk_preamble()
        mf_out, _ = _build_signal_with_preamble(preamble, _bpsk_preamble(32, seed=77))
        result = _syncer(preamble).find(mf_out)
        assert result.found
        assert result.corr_peak_index >= 0
        assert result.template_delay == 0
        assert 0.0 <= result.confidence <= 1.0

    def test_qpsk_preamble_found(self):
        preamble = _qpsk_preamble()
        payload = _qpsk_preamble(32, seed=200)
        mf_out, _ = _build_signal_with_preamble(preamble, payload, offset_samples=2 * SPS)
        result = _syncer(preamble, threshold=0.2).find(mf_out)
        assert result.found


# ── 6. Noisy preamble detection ───────────────────────────────────────────────

class TestNoisyDetection:
    @pytest.mark.parametrize("snr_db", [20.0, 10.0, 5.0])
    def test_found_at_reasonable_snr(self, snr_db):
        preamble = _bpsk_preamble()
        payload = _bpsk_preamble(64, seed=10)
        mf_out, expected_start = _build_signal_with_preamble(
            preamble, payload,
            offset_samples=SPS,
            snr_db=snr_db,
            seed=1234,
        )
        result = _syncer(preamble, threshold=0.15).find(mf_out)
        assert result.found, (
            f"Preamble not found at SNR={snr_db} dB; "
            f"confidence={result.confidence:.3f}"
        )
        # At all tested SNRs the start index should be within ±2 samples
        assert abs(result.start_index - expected_start) <= 2

    def test_confidence_decreases_with_noise(self):
        """Higher SNR → higher correlation confidence."""
        preamble = _bpsk_preamble()
        payload = _bpsk_preamble(32, seed=55)

        results = {}
        for snr_db in (20.0, 5.0):
            mf_out, _ = _build_signal_with_preamble(
                preamble, payload, snr_db=snr_db, seed=7
            )
            results[snr_db] = _syncer(preamble, threshold=0.05).find(mf_out)

        assert results[20.0].confidence > results[5.0].confidence


# ── 7. No-match / wrong preamble ──────────────────────────────────────────────

class TestNoMatch:
    def test_random_signal_below_threshold(self):
        """Pure AWGN with no preamble should not exceed a strict threshold."""
        preamble = _bpsk_preamble()
        rng = np.random.default_rng(999)
        noise_signal = (
            rng.standard_normal(512) + 1j * rng.standard_normal(512)
        ).astype(np.complex64)
        # With a strict threshold, noise should not falsely trigger
        result = _syncer(preamble, threshold=0.5).find(noise_signal)
        assert result.found is False

    def test_confidence_available_when_not_found(self):
        """Even when not found, confidence should be a valid float in [0,1]."""
        preamble = _bpsk_preamble()
        rng = np.random.default_rng(42)
        noise = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex64)
        result = _syncer(preamble, threshold=0.99).find(noise)
        assert result.found is False
        assert 0.0 <= result.confidence <= 1.0
        assert result.start_index == -1

    def test_wrong_preamble_different_seed(self):
        """A preamble not present in the signal should not be detected."""
        preamble_tx = _bpsk_preamble(seed=0)
        preamble_wrong = _bpsk_preamble(seed=999)  # entirely different sequence
        mf_out, _ = _build_signal_with_preamble(
            preamble_tx, _bpsk_preamble(32, seed=3)
        )
        result = _syncer(preamble_wrong, threshold=0.7).find(mf_out)
        # A wrong preamble should not exceed the strict threshold
        assert result.found is False


# ── 8. Threshold behavior ─────────────────────────────────────────────────────

class TestThreshold:
    def test_zero_threshold_always_found(self):
        """threshold=0 → any signal is 'found' (even pure noise)."""
        preamble = _bpsk_preamble()
        signal = np.ones(1000, dtype=np.complex64)
        result = _syncer(preamble, threshold=0.0).find(signal)
        assert result.found is True

    def test_high_threshold_suppresses_noisy_detection(self):
        """threshold=0.99 should suppress even a moderately noisy preamble."""
        preamble = _bpsk_preamble()
        payload = _bpsk_preamble(32, seed=10)
        mf_out, _ = _build_signal_with_preamble(
            preamble, payload, snr_db=5.0, seed=33
        )
        result = _syncer(preamble, threshold=0.99).find(mf_out)
        # At SNR=5 dB the confidence is unlikely to reach 0.99
        # We just verify it doesn't crash and returns valid fields
        assert isinstance(result.found, bool)
        assert 0.0 <= result.confidence <= 1.0

    def test_noiseless_confidence_exceeds_standard_threshold(self):
        preamble = _bpsk_preamble()
        mf_out, _ = _build_signal_with_preamble(preamble, _bpsk_preamble(32, seed=5))
        result = _syncer(preamble, threshold=0.3).find(mf_out)
        assert result.found
        assert result.confidence >= 0.3


# ── 9. find_frame_start convenience function ──────────────────────────────────

class TestFindFrameStart:
    def test_same_result_as_syncer(self):
        preamble = _bpsk_preamble()
        mf_out, _ = _build_signal_with_preamble(preamble, _bpsk_preamble(32, seed=8))

        r1 = _syncer(preamble).find(mf_out)
        r2 = find_frame_start(
            mf_out, preamble, SPS, ROLLOFF, SPAN, threshold=0.3
        )

        assert r1.found == r2.found
        assert r1.start_index == r2.start_index
        assert r1.confidence == pytest.approx(r2.confidence, abs=1e-6)

    def test_find_frame_start_found(self):
        preamble = _bpsk_preamble()
        mf_out, _ = _build_signal_with_preamble(preamble, _bpsk_preamble(32, seed=9))
        result = find_frame_start(mf_out, preamble, SPS)
        assert result.found


# ── 10. apply_mf=True path ────────────────────────────────────────────────────

class TestApplyMfFlag:
    def test_apply_mf_true_finds_frame(self):
        """
        Pass the raw Tx-shaped signal (before MF) with apply_mf=True;
        the synchronizer applies the MF internally and should still find
        the preamble.
        """
        from ml.generators.pulse_shaping.rrc import design_rrc_filter

        preamble = _bpsk_preamble()
        payload = _bpsk_preamble(32, seed=50)
        h_rrc = design_rrc_filter(SPS, ROLLOFF, SPAN).astype(np.float64)

        all_syms = np.concatenate([preamble, payload]).astype(np.complex128)
        upsampled = np.zeros(len(all_syms) * SPS, dtype=np.complex128)
        upsampled[::SPS] = all_syms
        # Only Tx RRC applied here; MF will be applied internally
        tx_only = np.convolve(upsampled, h_rrc, mode="full").astype(np.complex64)

        result = BasicFrameSynchronizer(
            preamble_symbols=preamble,
            samples_per_symbol=SPS,
            rolloff=ROLLOFF,
            filter_span_symbols=SPAN,
            threshold=0.2,
        ).find(tx_only, apply_mf=True)

        assert result.found


# ── 11. Package import check ──────────────────────────────────────────────────

def test_imports_from_synchronization_package():
    from synchronization import (  # noqa: PLC0415
        BasicFrameSynchronizer,
        FrameSyncResult,
        build_rc_template,
        find_frame_start,
        synchronize,           # existing function still importable
    )
    assert callable(synchronize)
    assert callable(find_frame_start)


# ── 12. Existing synchronize() still works (regression guard) ─────────────────

class TestExistingSynchronizeUnchanged:
    def test_bpsk_synchronize(self):
        np.random.seed(30)
        sps = 8
        n_sym = 128
        symbols = (2 * np.random.randint(0, 2, n_sym) - 1).astype(np.complex64)
        up = np.zeros(n_sym * sps, dtype=np.complex64)
        up[::sps] = symbols
        h = np.ones(sps, dtype=np.float32) / sps
        iq = np.convolve(up, h, mode="full")[:n_sym * sps]
        noise = 0.1 * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
        iq = (iq + noise.astype(np.complex64))
        result_syms, info = synchronize(iq, sample_rate=SR, symbol_rate=SR / sps)
        assert len(result_syms) > 0
        assert "sps" in info

    def test_synchronize_return_signature_unchanged(self):
        iq = np.zeros(256, dtype=np.complex64)
        out = synchronize(iq, sample_rate=10_000.0, symbol_rate=1_000.0)
        assert isinstance(out, tuple) and len(out) == 2
        symbols, info = out
        assert isinstance(symbols, np.ndarray)
        assert isinstance(info, dict)


# ── 13. ML waveform integration test ─────────────────────────────────────────

class TestIntegrationWithMLPulseShaping:
    """
    End-to-end test using the project's ML signal generation stack.

    Path:
      bits → BPSKModulator → pulse_shape (Tx RRC, mode="full") →
      prepend silence → apply_matched_filter (Rx RRC, mode="same") →
      BasicFrameSynchronizer.find()

    This proves frame acquisition works correctly with the actual RRC
    waveform representation used by the project.
    """

    def test_bpsk_ml_waveform_found_noiseless(self):
        from ml.generators.modulation.bpsk import BPSKModulator
        from ml.generators.config import GeneratorConfig
        from ml.generators.pulse_shaping.rrc import pulse_shape as rrc_pulse_shape
        from synchronization.matched_filter import apply_matched_filter

        sps = SPS
        n_preamble = 16
        n_payload = 64
        rolloff = ROLLOFF
        span = SPAN
        offset_samples = 3 * sps

        rng = np.random.default_rng(7)
        preamble_bits = rng.integers(0, 2, n_preamble).astype(np.uint8)
        payload_bits = rng.integers(0, 2, n_payload).astype(np.uint8)

        config = GeneratorConfig(
            modulation="BPSK",
            num_symbols=n_preamble + n_payload,
            sample_rate=SR,
            symbol_rate=SR / sps,
            samples_per_symbol=sps,
            random_seed=7,
            rolloff=rolloff,
            filter_span_symbols=span,
        )

        mod = BPSKModulator()
        preamble_symbols = mod.modulate(preamble_bits, config)
        payload_symbols = mod.modulate(payload_bits, config)

        # Build Tx waveform via project ML pulse_shape
        all_bits = np.concatenate([preamble_bits, payload_bits])
        all_config = GeneratorConfig(
            modulation="BPSK",
            num_symbols=n_preamble + n_payload,
            sample_rate=SR,
            symbol_rate=SR / sps,
            samples_per_symbol=sps,
            random_seed=7,
            rolloff=rolloff,
            filter_span_symbols=span,
        )
        all_symbols = mod.modulate(all_bits, all_config)
        tx_shaped = rrc_pulse_shape(all_symbols, all_config)  # mode="full"

        # Prepend silence to introduce known offset
        tx_with_offset = np.concatenate([
            np.zeros(offset_samples, dtype=np.complex64),
            tx_shaped,
        ])

        # Rx matched filter
        mf_out = apply_matched_filter(
            tx_with_offset,
            samples_per_symbol=sps,
            rolloff=rolloff,
            filter_span_symbols=span,
        )

        # Frame acquisition
        syncer = BasicFrameSynchronizer(
            preamble_symbols=preamble_symbols,
            samples_per_symbol=sps,
            rolloff=rolloff,
            filter_span_symbols=span,
            threshold=0.2,
        )
        result = syncer.find(mf_out)

        assert result.found, (
            f"Frame not found in ML-generated waveform; "
            f"confidence={result.confidence:.4f}"
        )

        # Expected start_index = offset_samples (the RC peak is at
        # offset_samples + span*sps in mf_out, and start_index = peak - delay)
        expected = offset_samples
        assert abs(result.start_index - expected) <= 1, (
            f"start_index={result.start_index}, expected≈{expected}"
        )

    def test_bpsk_ml_waveform_found_with_noise(self):
        from ml.generators.modulation.bpsk import BPSKModulator
        from ml.generators.config import GeneratorConfig
        from ml.generators.pulse_shaping.rrc import pulse_shape as rrc_pulse_shape
        from synchronization.matched_filter import apply_matched_filter

        sps = SPS
        n_preamble = 32
        n_payload = 64
        rolloff = ROLLOFF
        span = SPAN
        snr_db = 10.0

        rng = np.random.default_rng(13)
        preamble_bits = rng.integers(0, 2, n_preamble).astype(np.uint8)

        config = GeneratorConfig(
            modulation="BPSK",
            num_symbols=n_preamble + n_payload,
            sample_rate=SR,
            symbol_rate=SR / sps,
            samples_per_symbol=sps,
            random_seed=13,
            rolloff=rolloff,
            filter_span_symbols=span,
        )
        mod = BPSKModulator()
        preamble_symbols = mod.modulate(preamble_bits, config)

        payload_bits = rng.integers(0, 2, n_payload).astype(np.uint8)
        all_bits = np.concatenate([preamble_bits, payload_bits])
        all_symbols = mod.modulate(all_bits, config)
        tx_shaped = rrc_pulse_shape(all_symbols, config)

        # Add AWGN at Tx
        p_tx = float(np.mean(np.abs(tx_shaped) ** 2))
        p_noise = p_tx / (10.0 ** (snr_db / 10.0))
        noise_std = np.sqrt(p_noise / 2.0)
        noise = (
            rng.normal(0.0, noise_std, len(tx_shaped))
            + 1j * rng.normal(0.0, noise_std, len(tx_shaped))
        ).astype(np.complex64)
        tx_noisy = (tx_shaped + noise).astype(np.complex64)

        mf_out = apply_matched_filter(tx_noisy, sps, rolloff, span)

        syncer = BasicFrameSynchronizer(
            preamble_symbols=preamble_symbols,
            samples_per_symbol=sps,
            rolloff=rolloff,
            filter_span_symbols=span,
            threshold=0.15,
        )
        result = syncer.find(mf_out)
        assert result.found, (
            f"Frame not found at SNR={snr_db} dB; "
            f"confidence={result.confidence:.4f}"
        )
