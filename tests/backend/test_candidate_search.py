"""
Unit and integration tests for P5.2 Candidate Search Expansion & Bounded Hypothesis Generator.
Verifies all 15 requirements specified in the P5.2 objective:
  - Modulation search (ranking, top-k, threshold, unsupported filtering, fallback).
  - Symbol-rate expansion (estimate inclusion, uncertainty steps, fallback fraction, non-positive rejection, bounding).
  - Multi-dimensional Cartesian expansion (modulation x rate x sync x FEC x interleaver).
  - Strict bounding (max_candidates enforcement, pathological search space bounding).
  - Deduplication (identical collapsed, distinct sync/FEC/interleaver preserved).
  - Determinism (reproducible IDs, order, counts, no random UUIDs).
  - Provenance (actual ML probabilities preserved, un-evaluated dimensions not fabricated).
  - M8 parameter integration.
  - Backward compatibility with legacy generator and ranking.
"""
import math
from typing import Dict, List
import pytest

from backend.hypothesis.candidates import (
    HypothesisCandidate,
    EvidenceStatus,
    EvidenceTrace,
)
from backend.hypothesis.evaluator import calculate_raw_score
from backend.hypothesis.ranking import rank_hypotheses
from backend.hypothesis.generator import (
    create_candidate,
    generate_deterministic_candidate_id,
    expand_symbol_rate_candidates,
    SyncSearchConfig,
    DEFAULT_SYNC_CONFIG,
    HypothesisSearchConfig,
    HypothesisSearchSummary,
    generate_hypothesis_search,
    generate_hypothesis_search_with_summary,
    generate_hypotheses_from_ml,
)


# =====================================================================
# 1. Modulation Search Tests
# =====================================================================

def test_modulation_search_sorting_and_top_k():
    """Verifies that modulations are ranked descending by probability and bounded by top_k."""
    probs = {
        "8PSK": 0.05,
        "QPSK": 0.45,
        "BPSK": 0.35,
        "16QAM": 0.12,
        "64QAM": 0.03,
    }
    cfg = HypothesisSearchConfig(top_k_modulations=3, min_ml_probability=0.0)
    candidates, summary = generate_hypothesis_search_with_summary(
        ml_probabilities=probs,
        symbol_rate=10000.0,
        search_config=cfg,
    )

    # Top 3 should be QPSK (0.45), BPSK (0.35), 16QAM (0.12)
    retained = summary.retained_modulations
    assert retained == ["QPSK", "BPSK", "16QAM"]
    assert len(retained) == 3

    # Generated candidates should only contain these modulations
    candidate_mods = [c.modulation for c in candidates]
    assert set(candidate_mods) == {"QPSK", "BPSK", "16QAM"}
    assert "8PSK" not in candidate_mods
    assert "64QAM" not in candidate_mods


def test_modulation_search_min_probability_threshold():
    """Verifies that modulations below min_ml_probability are pruned."""
    probs = {
        "BPSK": 0.85,
        "QPSK": 0.14,
        "8PSK": 0.008,
        "16QAM": 0.002,
    }
    cfg = HypothesisSearchConfig(
        top_k_modulations=5,
        min_ml_probability=0.05,  # 8PSK and 16QAM should be excluded
    )
    candidates, summary = generate_hypothesis_search_with_summary(
        ml_probabilities=probs,
        symbol_rate=5000.0,
        search_config=cfg,
    )

    assert summary.retained_modulations == ["BPSK", "QPSK"]
    assert len(summary.retained_modulations) == 2


def test_modulation_search_fallback_when_all_below_threshold():
    """Verifies that fallback modulation is included if all candidates fall below threshold."""
    probs = {
        "8PSK": 0.004,
        "16QAM": 0.003,
    }
    cfg = HypothesisSearchConfig(
        min_ml_probability=0.01,
        include_fallback_modulation="BPSK",
    )
    candidates, summary = generate_hypothesis_search_with_summary(
        ml_probabilities=probs,
        symbol_rate=1000.0,
        search_config=cfg,
    )

    assert summary.retained_modulations == ["BPSK"]
    assert len(candidates) >= 1
    assert candidates[0].modulation == "BPSK"


def test_modulation_search_supported_classes_filtering():
    """Verifies that unsupported or out-of-taxonomy modulations are cleanly skipped."""
    probs = {
        "UNKNOWN_MOD": 0.70,
        "BPSK": 0.20,
        "QPSK": 0.10,
    }
    cfg = HypothesisSearchConfig(
        supported_modulations=["BPSK", "QPSK", "8PSK"],
        min_ml_probability=0.0,
    )
    candidates, summary = generate_hypothesis_search_with_summary(
        ml_probabilities=probs,
        symbol_rate=1000.0,
        search_config=cfg,
    )

    assert "UNKNOWN_MOD" not in summary.retained_modulations
    assert summary.retained_modulations == ["BPSK", "QPSK"]


# =====================================================================
# 2. Symbol-Rate Expansion Tests
# =====================================================================

def test_symbol_rate_expansion_includes_estimate():
    """Verifies that the estimated symbol rate is always included."""
    rates = expand_symbol_rate_candidates(estimated_baud=9600.0)
    assert 9600.0 in rates


def test_symbol_rate_expansion_with_uncertainty():
    """Verifies expansion around estimate using uncertainty steps."""
    cfg = HypothesisSearchConfig(
        symbol_rate_step_count=1,
        symbol_rate_uncertainty_multiplier=1.0,
        max_symbol_rate_candidates=5,
    )
    # Estimate 9600, uncertainty 300 -> steps -300, 0, +300 -> 9300, 9600, 9900
    rates = expand_symbol_rate_candidates(
        estimated_baud=9600.0,
        uncertainty_baud=300.0,
        config=cfg,
    )
    assert rates == [9300.0, 9600.0, 9900.0]


def test_symbol_rate_expansion_without_uncertainty_fallback():
    """Verifies deterministic fallback relative fraction when uncertainty is missing."""
    cfg = HypothesisSearchConfig(
        symbol_rate_step_count=1,
        fallback_uncertainty_fraction=0.10,  # 10% of 10000 = 1000
        max_symbol_rate_candidates=3,
    )
    rates = expand_symbol_rate_candidates(
        estimated_baud=10000.0,
        uncertainty_baud=None,
        config=cfg,
    )
    assert rates == [9000.0, 10000.0, 11000.0]


def test_symbol_rate_expansion_rejects_non_positive_rates():
    """Verifies that non-positive candidate rates are discarded, and invalid center raises error."""
    # A small center with wide negative step should not produce <= 0 rates
    cfg = HypothesisSearchConfig(
        symbol_rate_offsets=[-1500.0, -500.0, 0.0, 500.0],
        max_symbol_rate_candidates=5,
    )
    rates = expand_symbol_rate_candidates(
        estimated_baud=1000.0,
        config=cfg,
    )
    # 1000 - 1500 = -500 (rejected), 1000 - 500 = 500 (valid), 1000, 1500
    assert all(r > 0 for r in rates)
    assert rates == [500.0, 1000.0, 1500.0]

    # Non-positive center raises ValueError
    with pytest.raises(ValueError, match="strictly positive and finite"):
        expand_symbol_rate_candidates(estimated_baud=0.0)

    with pytest.raises(ValueError, match="strictly positive and finite"):
        expand_symbol_rate_candidates(estimated_baud=-9600.0)


def test_symbol_rate_expansion_deduplication():
    """Verifies duplicate rates (e.g. from identical offsets or tiny floating noise) are removed."""
    cfg = HypothesisSearchConfig(
        symbol_rate_offsets=[0.0, 0.0001, 100.0, 100.0],
        max_symbol_rate_candidates=5,
    )
    rates = expand_symbol_rate_candidates(
        estimated_baud=5000.0,
        config=cfg,
    )
    assert rates == [5000.0, 5100.0]


def test_symbol_rate_expansion_max_candidates_limit():
    """Verifies that max_symbol_rate_candidates truncates while keeping the center rate."""
    cfg = HypothesisSearchConfig(
        symbol_rate_step_count=3,  # 7 steps: -3, -2, -1, 0, 1, 2, 3
        max_symbol_rate_candidates=3,  # Should keep center and 2 closest
    )
    rates = expand_symbol_rate_candidates(
        estimated_baud=10000.0,
        uncertainty_baud=500.0,
        config=cfg,
    )
    assert len(rates) == 3
    assert 10000.0 in rates
    # Nearest steps are 9500, 10000, 10500
    assert rates == [9500.0, 10000.0, 10500.0]


# =====================================================================
# 3. Synchronization & Configuration Search Tests
# =====================================================================

def test_sync_search_config_presets_and_serialization():
    """Verifies SyncSearchConfig defaults and dictionary serialization."""
    s = DEFAULT_SYNC_CONFIG
    assert s.carrier_recovery == "costas"
    assert s.timing_recovery == "gardner"
    assert s.matched_filter == "rrc"

    d = s.to_dict()
    assert d["carrier_recovery"] == "costas"
    assert d["filter_rolloff"] == 0.35
    assert isinstance(d, dict)


def test_cartesian_expansion_combinations():
    """
    Verifies multi-dimensional Cartesian expansion:
    2 modulations x 2 symbol rates x 2 sync configs x 2 FEC x 1 interleaver = 16 combinations.
    """
    probs = {"QPSK": 0.60, "BPSK": 0.40}
    sync1 = SyncSearchConfig(carrier_recovery="costas", timing_recovery="gardner")
    sync2 = SyncSearchConfig(carrier_recovery="pll", timing_recovery="early_late")

    cfg = HypothesisSearchConfig(
        top_k_modulations=2,
        max_symbol_rate_candidates=2,
        symbol_rate_offsets=[0.0, 100.0],
        sync_configurations=[sync1, sync2],
        fec_candidates=[None, "conv_r1/2_k7"],
        interleaver_candidates=[None],
        max_candidates=100,  # Large enough to hold all 16
    )

    candidates, summary = generate_hypothesis_search_with_summary(
        ml_probabilities=probs,
        symbol_rate=10000.0,
        search_config=cfg,
    )

    assert summary.total_combinations_before_pruning == 16
    assert summary.final_candidate_count == 16
    assert len(candidates) == 16
    assert summary.pruned_by_limit is False


# =====================================================================
# 4. Strict Search Bounding Tests
# =====================================================================

def test_bounding_on_large_search_space():
    """
    Constructs an intentionally large search space and verifies that
    len(candidates) <= max_candidates.
    """
    probs = {f"MOD_{i}": 0.10 for i in range(10)}
    sync_list = [
        SyncSearchConfig(carrier_recovery="costas"),
        SyncSearchConfig(carrier_recovery="pll"),
        SyncSearchConfig(carrier_recovery="none"),
    ]
    fec_list = [None, "conv_r1/2_k7", "ldpc_r3/4", "turbo_r1/3"]
    intl_list = [None, "block_16x16", "block_32x32"]

    # 10 mods x 5 rates x 3 syncs x 4 FEC x 3 intl = 1,800 combinations
    cfg = HypothesisSearchConfig(
        top_k_modulations=10,
        max_symbol_rate_candidates=5,
        symbol_rate_offsets=[-200, -100, 0, 100, 200],
        sync_configurations=sync_list,
        fec_candidates=fec_list,
        interleaver_candidates=intl_list,
        max_candidates=20,  # Bound strictly to 20
        min_ml_probability=0.0,
    )

    candidates, summary = generate_hypothesis_search_with_summary(
        ml_probabilities=probs,
        symbol_rate=10000.0,
        search_config=cfg,
    )

    assert summary.total_combinations_before_pruning == 1800
    assert len(candidates) == 20
    assert summary.final_candidate_count == 20
    assert summary.pruned_by_limit is True


# =====================================================================
# 5. Deduplication & Identity Tests
# =====================================================================

def test_deduplication_identical_configurations():
    """Verifies that identical configurations are deduplicated."""
    probs = {"BPSK": 0.90}
    # Provide identical sync configs and identical rates
    cfg = HypothesisSearchConfig(
        top_k_modulations=1,
        symbol_rate_offsets=[0.0, 0.0],  # Duplicate rate
        sync_configurations=[DEFAULT_SYNC_CONFIG, DEFAULT_SYNC_CONFIG],  # Duplicate sync
        fec_candidates=[None, None],  # Duplicate FEC
        max_candidates=10,
    )

    candidates, summary = generate_hypothesis_search_with_summary(
        ml_probabilities=probs,
        symbol_rate=5000.0,
        search_config=cfg,
    )

    # 1 mod x 1 unique rate x 1 unique sync x 1 unique fec = 1 unique candidate
    assert len(candidates) == 1
    assert summary.candidates_after_deduplication == 1


def test_deduplication_preserves_distinct_sync_assumptions():
    """Verifies that candidates with different sync assumptions remain distinct."""
    probs = {"QPSK": 0.80}
    s1 = SyncSearchConfig(carrier_recovery="costas", timing_recovery="gardner")
    s2 = SyncSearchConfig(carrier_recovery="pll", timing_recovery="early_late")

    cfg = HypothesisSearchConfig(
        top_k_modulations=1,
        max_symbol_rate_candidates=1,
        symbol_rate_offsets=[0.0],
        sync_configurations=[s1, s2],
        max_candidates=10,
    )

    candidates, summary = generate_hypothesis_search_with_summary(
        ml_probabilities=probs,
        symbol_rate=5000.0,
        search_config=cfg,
    )

    # Must preserve both sync configurations
    assert len(candidates) == 2
    keys = {c.identity_key() for c in candidates}
    assert len(keys) == 2


def test_deduplication_preserves_distinct_fec_and_interleaver():
    """Verifies that different FEC and interleaver configurations remain distinct."""
    probs = {"BPSK": 0.80}
    cfg = HypothesisSearchConfig(
        top_k_modulations=1,
        max_symbol_rate_candidates=1,
        symbol_rate_offsets=[0.0],
        fec_candidates=[None, "conv_r1/2_k7"],
        interleaver_candidates=[None, "block_16x16"],
        max_candidates=10,
    )

    candidates, _ = generate_hypothesis_search_with_summary(
        ml_probabilities=probs,
        symbol_rate=5000.0,
        search_config=cfg,
    )

    # 1 mod x 1 rate x 1 sync x 2 FEC x 2 intl = 4 distinct candidates
    assert len(candidates) == 4
    keys = {c.identity_key() for c in candidates}
    assert len(keys) == 4


# =====================================================================
# 6. Determinism Tests
# =====================================================================

def test_determinism_across_multiple_runs():
    """Verifies that repeated generation with identical inputs produces identical IDs, order, and counts."""
    probs = {"8PSK": 0.15, "QPSK": 0.55, "BPSK": 0.30}
    cfg = HypothesisSearchConfig(
        top_k_modulations=3,
        max_symbol_rate_candidates=3,
        fec_candidates=[None, "conv_r1/2_k7"],
        max_candidates=12,
    )

    run1 = generate_hypothesis_search(probs, 10000.0, 200.0, search_config=cfg)
    run2 = generate_hypothesis_search(probs, 10000.0, 200.0, search_config=cfg)

    assert len(run1) == len(run2)

    for c1, c2 in zip(run1, run2):
        assert c1.id == c2.id
        assert c1.modulation == c2.modulation
        assert c1.symbolRate == c2.symbolRate
        assert c1.fec_config == c2.fec_config
        assert c1.details == c2.details
        assert c1.identity_key() == c2.identity_key()
        # Ensure no random UUID in candidate ID
        assert "uuid" not in c1.id.lower()


def test_deterministic_candidate_id_function():
    """Tests the deterministic candidate ID generator directly."""
    id1 = generate_deterministic_candidate_id(
        modulation="QPSK",
        symbol_rate=10000.0,
        fec_config="conv_r1/2_k7",
        interleaver_config=None,
        sync_assumptions={"carrier_recovery": "costas"},
    )
    id2 = generate_deterministic_candidate_id(
        modulation="QPSK",
        symbol_rate=10000.0,
        fec_config="conv_r1/2_k7",
        interleaver_config=None,
        sync_assumptions={"carrier_recovery": "costas"},
    )
    assert id1 == id2
    assert id1.startswith("hyp_qpsk_10000_conv_r1_2_k7_")


# =====================================================================
# 7. Provenance & Non-Fabrication Tests
# =====================================================================

def test_candidate_provenance_and_unfabricated_evidence():
    """
    Verifies that generated candidates retain genuine ML probabilities and
    explicit NOT_EVALUATED status for un-executed dimensions.
    """
    probs = {"QPSK": 0.72}
    cands = generate_hypothesis_search(
        ml_probabilities=probs,
        symbol_rate=4800.0,
        symbol_rate_uncertainty=120.0,
    )

    assert len(cands) >= 1
    c = cands[0]

    # ML evidence contains genuine probability
    assert c.evidence.ml.status == EvidenceStatus.AVAILABLE
    assert c.evidence.ml.score == 0.72

    # Un-evaluated dimensions must be strictly NOT_EVALUATED
    assert c.evidence.symbol_rate.status == EvidenceStatus.NOT_EVALUATED
    assert c.evidence.constellation.status == EvidenceStatus.NOT_EVALUATED
    assert c.evidence.timing.status == EvidenceStatus.NOT_EVALUATED
    assert c.evidence.fec.status == EvidenceStatus.NOT_EVALUATED
    assert c.evidence.bitstream.status == EvidenceStatus.NOT_EVALUATED

    # Details string explains generation provenance
    assert "QPSK" in c.details
    assert "4800" in c.details
    assert "0.7200" in c.details


# =====================================================================
# 8. M8 Parameter Result Integration Tests
# =====================================================================

def test_m8_parameter_result_search_integration():
    """Verifies that parameter_result can be passed directly to generate_hypothesis_search."""
    # Create mock-like M8 objects matching ParameterAnalysisResult structure
    class MockSingleParam:
        def __init__(self, est, unc, conf):
            self.estimate = est
            self.uncertainty = unc
            self.confidence = conf
        def to_dict(self):
            return {"estimate": self.estimate, "uncertainty": self.uncertainty}

    class MockParamAnalysis:
        def __init__(self):
            self.symbol_rate = MockSingleParam(25000.0, 250.0, 0.95)
            self.snr_db = MockSingleParam(18.5, 1.0, 0.90)
            self.sample_rate = 200000.0

    m8_result = MockParamAnalysis()
    probs = {"BPSK": 0.85, "QPSK": 0.15}

    candidates = generate_hypothesis_search(
        ml_probabilities=probs,
        parameter_result=m8_result,
    )

    assert len(candidates) >= 2
    top_cand = candidates[0]
    assert top_cand.sync_assumptions is not None
    assert top_cand.sync_assumptions["estimated_symbol_rate_baud"] == 25000.0
    assert top_cand.sync_assumptions["symbol_rate_uncertainty_baud"] == 250.0
    assert "symbol_rate_m8" in top_cand.sync_assumptions


# =====================================================================
# 9. Backward Compatibility Tests
# =====================================================================

def test_backward_compatibility_generate_hypotheses_from_ml():
    """
    Verifies that legacy generate_hypotheses_from_ml continues to function
    with expected outputs and works with downstream scoring and ranking.
    """
    probs = {"BPSK": 0.70, "QPSK": 0.25, "8PSK": 0.05}
    candidates = generate_hypotheses_from_ml(
        ml_probabilities=probs,
        symbol_rate=12000.0,
        fec_candidates=["none", "conv_r1/2_k7"],
        top_k=2,
    )

    # 2 top classes (BPSK, QPSK) x 2 FEC = 4 candidates
    assert len(candidates) == 4
    for c in candidates:
        assert c.status == "pending"
        assert c.symbolRate == 12000.0

    # Downstream scoring and ranking must execute without error
    for c in candidates:
        calculate_raw_score(c)

    ranked = rank_hypotheses(candidates)
    assert len(ranked) == 4
    assert ranked[0].confidenceScore > ranked[-1].confidenceScore
    assert math.isclose(sum(c.confidenceScore for c in ranked), 1.0, rel_tol=1e-5)
