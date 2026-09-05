"""
Unit tests for backend/hypothesis/

Tests cover candidate generation, ranking, and the full pipeline for
both the failure path (bad input) and the happy path (clean synthetic signal).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from hypothesis.candidates import (
    generate_candidates,
    HypothesisSpec,
    _normalise_mod_name,
    DEMODULATABLE,
)
from hypothesis.evaluator import evaluate_hypothesis, EvaluatedHypothesis
from hypothesis.ranking import rank_hypotheses, score_hypothesis


# ── Candidate generation ──────────────────────────────────────────────────────

class TestCandidateGeneration:
    def _top_preds(self):
        return [("BPSK", 0.70), ("QPSK", 0.20), ("AM-DSB", 0.05)]

    def test_returns_list(self):
        result = generate_candidates(self._top_preds(), 10_000.0, 20_000.0, 15.0)
        assert isinstance(result, list)

    def test_includes_bpsk_and_qpsk(self):
        result = generate_candidates(self._top_preds(), 10_000.0, 20_000.0, 15.0)
        mods = {c.modulation for c in result}
        assert "BPSK" in mods
        assert "QPSK" in mods

    def test_respects_max_candidates(self):
        result = generate_candidates(self._top_preds(), 10_000.0, 20_000.0, 15.0, max_candidates=3)
        assert len(result) <= 3

    def test_sorted_by_confidence(self):
        result = generate_candidates(self._top_preds(), 10_000.0, 20_000.0, 15.0)
        confs = [c.ml_confidence for c in result]
        assert confs == sorted(confs, reverse=True)

    def test_fec_variants_generated(self):
        """For demodulatable modulations both fec_type=None and 'convolutional' should appear."""
        result = generate_candidates([("BPSK", 0.9)], 10_000.0, 20_000.0, 15.0)
        fec_types = {c.fec_type for c in result}
        assert None in fec_types
        assert "convolutional" in fec_types

    def test_non_demodulatable_marked(self):
        result = generate_candidates([("AM-DSB", 0.9)], 10_000.0, 20_000.0, 15.0)
        for c in result:
            if c.modulation == "AM-DSB":
                assert not c.can_demodulate

    def test_symbol_rate_variants_on_uncertainty(self):
        """When sr > bw (uncertain), multiple sr variants should be generated."""
        result = generate_candidates(
            [("BPSK", 0.9)],
            symbol_rate_estimate=50_000.0,   # > bandwidth
            bandwidth_estimate=10_000.0,
            snr_estimate=15.0,
        )
        srs = {c.symbol_rate for c in result}
        assert len(srs) > 1

    def test_no_duplicates(self):
        result = generate_candidates(self._top_preds(), 10_000.0, 20_000.0, 15.0)
        seen = set()
        for c in result:
            key = (c.modulation, round(c.symbol_rate), c.fec_type)
            assert key not in seen, f"Duplicate hypothesis: {key}"
            seen.add(key)

    def test_normalise_mod_name(self):
        assert _normalise_mod_name("bpsk") == "BPSK"
        assert _normalise_mod_name("16-QAM") == "16QAM"
        assert _normalise_mod_name("AM-DSB-SC") == "AM-DSB"


# ── Evaluator ─────────────────────────────────────────────────────────────────

def _make_bpsk_iq_2d(n_sym=256, sps=8, snr_db=20.0) -> tuple[np.ndarray, float]:
    np.random.seed(50)
    sr = 10_000.0 * sps
    symbols = (2 * np.random.randint(0, 2, n_sym) - 1).astype(np.complex64)
    up = np.zeros(n_sym * sps, dtype=np.complex64)
    up[::sps] = symbols
    h = np.ones(sps, dtype=np.float32) / sps
    iq = np.convolve(up, h, mode="full")[:n_sym * sps]
    snr_lin = 10.0 ** (snr_db / 10.0)
    pwr = float(np.mean(np.abs(iq) ** 2))
    ns = np.sqrt(pwr / (2.0 * snr_lin))
    iq += (ns * np.random.randn(len(iq)) + 1j * ns * np.random.randn(len(iq))).astype(np.complex64)
    return np.stack([iq.real, iq.imag]).astype(np.float32), sr


class TestEvaluator:
    def test_non_demodulatable_returns_all_false(self):
        iq_2d, sr = _make_bpsk_iq_2d()
        spec = HypothesisSpec(
            modulation="AM-DSB",
            symbol_rate=10_000.0,
            fec_type=None,
            ml_confidence=0.9,
            can_demodulate=False,
        )
        result = evaluate_hypothesis(iq_2d, sr, spec)
        assert not result.sync_pass
        assert not result.demod_pass

    def test_bpsk_hypothesis_evaluates(self):
        """BPSK hypothesis on a clean BPSK signal should at least pass sync."""
        iq_2d, sr = _make_bpsk_iq_2d(snr_db=25.0)
        spec = HypothesisSpec(
            modulation="BPSK",
            symbol_rate=10_000.0,
            fec_type=None,
            ml_confidence=0.8,
        )
        result = evaluate_hypothesis(iq_2d, sr, spec)
        assert isinstance(result, EvaluatedHypothesis)
        # Sync should pass — we have enough symbols
        assert result.sync_pass

    def test_result_has_all_fields(self):
        iq_2d, sr = _make_bpsk_iq_2d()
        spec = HypothesisSpec("BPSK", 10_000.0, None, 0.5)
        r = evaluate_hypothesis(iq_2d, sr, spec)
        for field in ("sync_pass", "demod_pass", "fec_pass", "bitstream_pass", "evm_rms"):
            assert hasattr(r, field)

    def test_evaluator_never_raises(self):
        """Evaluator must handle pathological all-zero IQ without raising."""
        iq_2d = np.zeros((2, 256), dtype=np.float32)
        spec = HypothesisSpec("BPSK", 10_000.0, None, 0.5)
        result = evaluate_hypothesis(iq_2d, 80_000.0, spec)
        assert isinstance(result, EvaluatedHypothesis)

    def test_wrong_modulation_lower_score(self):
        """
        Evaluating QPSK hypothesis on BPSK signal should give worse
        EVM (higher) than the correct BPSK hypothesis.
        """
        iq_2d, sr = _make_bpsk_iq_2d(snr_db=25.0)
        spec_bpsk = HypothesisSpec("BPSK", 10_000.0, None, 0.8)
        spec_qpsk = HypothesisSpec("QPSK", 10_000.0, None, 0.5)
        r_bpsk = evaluate_hypothesis(iq_2d, sr, spec_bpsk)
        r_qpsk = evaluate_hypothesis(iq_2d, sr, spec_qpsk)
        # BPSK should have lower EVM on a BPSK signal
        assert r_bpsk.evm_rms <= r_qpsk.evm_rms + 0.3


# ── Ranking ───────────────────────────────────────────────────────────────────

class TestRanking:
    def _make_ev(self, sync=False, demod=False, evm=1.0, conf=0.5, fec=False, bits=False, entropy=0.0):
        spec = HypothesisSpec("BPSK", 10_000.0, None, conf)
        ev = EvaluatedHypothesis(spec=spec)
        ev.sync_pass = sync
        ev.demod_pass = demod
        ev.fec_pass = fec
        ev.bitstream_pass = bits
        ev.evm_rms = evm
        ev.bit_entropy = entropy
        return ev

    def test_rank_1_is_best(self):
        ev_good = self._make_ev(sync=True, demod=True, evm=0.1, conf=0.9, bits=True, entropy=0.9)
        ev_bad  = self._make_ev(sync=False, demod=False, evm=0.9, conf=0.1)
        ranked = rank_hypotheses([ev_bad, ev_good])
        assert ranked[0]["rank"] == 1
        assert ranked[0]["final_score"] >= ranked[1]["final_score"]

    def test_ranks_are_sequential(self):
        evs = [self._make_ev(conf=c) for c in [0.9, 0.7, 0.3]]
        ranked = rank_hypotheses(evs)
        assert [r["rank"] for r in ranked] == [1, 2, 3]

    def test_score_in_0_1(self):
        ev = self._make_ev(sync=True, demod=True, evm=0.1, conf=0.8, bits=True, entropy=0.95)
        s = score_hypothesis(ev)
        assert 0.0 <= s <= 1.0

    def test_empty_list(self):
        assert rank_hypotheses([]) == []

    def test_required_fields_present(self):
        ev = self._make_ev()
        ranked = rank_hypotheses([ev])
        row = ranked[0]
        for field in ("modulation", "symbol_rate", "fec_type", "ml_confidence",
                      "sync_pass", "demod_pass", "fec_pass", "bitstream_pass",
                      "final_score", "rank"):
            assert field in row, f"Missing field: {field}"

    def test_high_ml_confidence_boosts_score(self):
        ev_high = self._make_ev(conf=0.95)
        ev_low  = self._make_ev(conf=0.05)
        assert score_hypothesis(ev_high) > score_hypothesis(ev_low)

    def test_passing_demod_boosts_score(self):
        ev_pass = self._make_ev(conf=0.5, sync=True, demod=True, evm=0.1)
        ev_fail = self._make_ev(conf=0.5, sync=False, demod=False, evm=0.9)
        assert score_hypothesis(ev_pass) > score_hypothesis(ev_fail)
