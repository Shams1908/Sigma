"""
Unit and integration tests for hypothesis scoring, deduplication, and normalized confidence ranking.
Verifies all 14 requirements specified in the project intent.
"""
import math
import pytest
from typing import List, Dict

from backend.core.config import settings, HypothesisScoringSettings
from backend.hypothesis.candidates import (
    HypothesisCandidate,
    EvidenceStatus,
    EvidenceComponent,
    EvidenceTrace,
)
from backend.hypothesis.evaluator import (
    calculate_raw_score,
    create_ml_evidence,
    create_constellation_evidence,
    create_timing_evidence,
    create_fec_evidence,
    create_bitstream_evidence,
)
from backend.hypothesis.ranking import (
    deduplicate_candidates,
    normalize_confidences,
    rank_hypotheses,
)
from backend.hypothesis.generator import (
    create_candidate,
    generate_hypotheses_from_ml,
)


def _make_candidate(
    modulation: str,
    symbol_rate: float = 1000.0,
    raw_score: float = 0.0,
    fec: str = None,
    cand_id: str = None,
) -> HypothesisCandidate:
    cand = create_candidate(
        modulation=modulation,
        symbol_rate=symbol_rate,
        fec_config=fec,
        candidate_id=cand_id,
    )
    cand.rawScore = raw_score
    if raw_score > 0:
        cand.evidence.ml = create_ml_evidence(modulation, {modulation: raw_score})
    return cand


# 1. Different raw scores produce different normalized confidences
def test_different_raw_scores_produce_different_confidences():
    c1 = _make_candidate("BPSK", raw_score=0.85)
    c2 = _make_candidate("QPSK", raw_score=0.45)
    c3 = _make_candidate("8PSK", raw_score=0.20)

    normalized = normalize_confidences([c1, c2, c3], temperature=0.25)

    assert normalized[0].confidenceScore != normalized[1].confidenceScore
    assert normalized[1].confidenceScore != normalized[2].confidenceScore
    assert normalized[0].confidenceScore > normalized[1].confidenceScore > normalized[2].confidenceScore


# 2. Equal raw scores produce equal confidences
def test_equal_raw_scores_produce_equal_confidences():
    c1 = _make_candidate("BPSK", symbol_rate=1000.0, raw_score=0.75)
    c2 = _make_candidate("QPSK", symbol_rate=2000.0, raw_score=0.75)
    c3 = _make_candidate("8PSK", symbol_rate=3000.0, raw_score=0.75)

    normalized = normalize_confidences([c1, c2, c3], temperature=0.25)

    assert math.isclose(normalized[0].confidenceScore, normalized[1].confidenceScore, rel_tol=1e-6)
    assert math.isclose(normalized[1].confidenceScore, normalized[2].confidenceScore, rel_tol=1e-6)
    assert math.isclose(normalized[0].confidenceScore, 1.0 / 3.0, rel_tol=1e-6)


# 3. Confidence scores sum to 1
def test_confidence_scores_sum_to_one():
    scores = [0.92, 0.74, 0.58, 0.31, 0.15]
    candidates = [_make_candidate(f"MOD_{i}", symbol_rate=1000.0 * (i + 1), raw_score=s) for i, s in enumerate(scores)]

    normalized = normalize_confidences(candidates, temperature=0.30)
    conf_sum = sum(c.confidenceScore for c in normalized)

    assert math.isclose(conf_sum, 1.0, abs_tol=1e-6)


# 4. Confidence scores remain in [0, 1]
def test_confidence_scores_remain_in_range_zero_to_one():
    scores = [0.99, 0.85, 0.60, 0.40, 0.10, 0.0]
    candidates = [_make_candidate(f"MOD_{i}", symbol_rate=1000.0 * (i + 1), raw_score=s) for i, s in enumerate(scores)]

    normalized = normalize_confidences(candidates, temperature=0.20)

    for c in normalized:
        assert 0.0 <= c.confidenceScore <= 1.0
        assert math.isfinite(c.confidenceScore)


# 5. Ranking order is preserved
def test_ranking_order_preserved():
    c1 = _make_candidate("MOD_A", symbol_rate=1000.0, raw_score=0.91)
    c2 = _make_candidate("MOD_B", symbol_rate=2000.0, raw_score=0.73)
    c3 = _make_candidate("MOD_C", symbol_rate=3000.0, raw_score=0.52)
    c4 = _make_candidate("MOD_D", symbol_rate=4000.0, raw_score=0.22)

    ranked = rank_hypotheses([c3, c1, c4, c2])

    assert [c.modulation for c in ranked] == ["MOD_A", "MOD_B", "MOD_C", "MOD_D"]
    assert ranked[0].confidenceScore > ranked[1].confidenceScore > ranked[2].confidenceScore > ranked[3].confidenceScore


# 6. Stable behavior for very large/small raw scores
def test_numerical_stability_extreme_raw_scores():
    # Very large positive raw scores
    c_high1 = _make_candidate("HIGH_1", symbol_rate=1000.0, raw_score=10000.0)
    c_high2 = _make_candidate("HIGH_2", symbol_rate=2000.0, raw_score=9998.0)
    norm_high = normalize_confidences([c_high1, c_high2], temperature=1.0)
    
    for c in norm_high:
        assert math.isfinite(c.confidenceScore)
        assert 0.0 <= c.confidenceScore <= 1.0
    assert math.isclose(sum(c.confidenceScore for c in norm_high), 1.0, abs_tol=1e-5)

    # Very small negative raw scores
    c_low1 = _make_candidate("LOW_1", symbol_rate=1000.0, raw_score=-5000.0)
    c_low2 = _make_candidate("LOW_2", symbol_rate=2000.0, raw_score=-5002.0)
    norm_low = normalize_confidences([c_low1, c_low2], temperature=1.0)

    for c in norm_low:
        assert math.isfinite(c.confidenceScore)
        assert 0.0 <= c.confidenceScore <= 1.0
    assert math.isclose(sum(c.confidenceScore for c in norm_low), 1.0, abs_tol=1e-5)


# 7. Missing evidence does not introduce fake numeric confidence
def test_missing_evidence_does_not_introduce_fake_numeric_scores():
    cand = create_candidate("BPSK", 1000.0)
    # By default, all 5 evidence components are NOT_EVALUATED
    for dim_name in ["ml", "constellation", "timing", "fec", "bitstream"]:
        comp = getattr(cand.evidence, dim_name)
        assert comp.status == EvidenceStatus.NOT_EVALUATED
        assert comp.score is None
        # to_dict should NOT contain a numeric score for un-evaluated dimension
        serialized = comp.to_dict()
        assert "score" not in serialized
        assert serialized["status"] == "not_evaluated"

    raw = calculate_raw_score(cand)
    assert raw == 0.0
    assert cand.rawScore == 0.0


# 8. Weight renormalization works when evidence is unavailable
def test_weight_renormalization_when_evidence_unavailable():
    """
    User example:
      Available: M = 0.82 (w1=0.35), C = 0.91 (w2=0.25), T = 0.87 (w3=0.15)
      Unavailable: FEC (not_evaluated), Bitstream (not_evaluated)
      Total active weights = 0.35 + 0.25 + 0.15 = 0.75
      Expected score = (0.35*0.82 + 0.25*0.91 + 0.15*0.87) / 0.75 = 0.645 / 0.75 = 0.860
    """
    cand = create_candidate("BPSK", 1000.0)
    cand.evidence.ml = create_ml_evidence("BPSK", {"BPSK": 0.82})
    cand.evidence.constellation = create_constellation_evidence(0.91, EvidenceStatus.AVAILABLE)
    cand.evidence.timing = create_timing_evidence(0.87, EvidenceStatus.AVAILABLE)
    cand.evidence.fec = create_fec_evidence(status=EvidenceStatus.NOT_EVALUATED)
    cand.evidence.bitstream = create_bitstream_evidence(status=EvidenceStatus.NOT_EVALUATED)

    custom_config = HypothesisScoringSettings(
        weight_ml=0.35,
        weight_constellation=0.25,
        weight_timing=0.15,
        weight_fec=0.15,
        weight_bitstream=0.10,
    )

    raw = calculate_raw_score(cand, config=custom_config)
    expected_score = (0.35 * 0.82 + 0.25 * 0.91 + 0.15 * 0.87) / (0.35 + 0.25 + 0.15)

    assert math.isclose(raw, expected_score, abs_tol=1e-5)
    assert math.isclose(raw, 0.860, abs_tol=1e-3)


# 9. Duplicate hypotheses are removed before normalization
def test_duplicate_hypothesis_deduplication():
    # Two identical BPSK configurations generated by different search paths
    c1 = _make_candidate("BPSK", symbol_rate=1000.0, raw_score=0.80, cand_id="path_a_1")
    c2 = _make_candidate("BPSK", symbol_rate=1000.0, raw_score=0.82, cand_id="path_b_1")
    
    # Distinct QPSK
    c3 = _make_candidate("QPSK", symbol_rate=1000.0, raw_score=0.60, cand_id="path_a_2")

    deduped = deduplicate_candidates([c1, c2, c3])
    assert len(deduped) == 2
    
    # Should retain the superior candidate for the duplicate configuration
    bpsk_cand = next(c for c in deduped if c.modulation == "BPSK")
    assert bpsk_cand.rawScore == 0.82

    # Verify deduplication within rank_hypotheses
    ranked = rank_hypotheses([c1, c2, c3])
    assert len(ranked) == 2
    assert math.isclose(sum(c.confidenceScore for c in ranked), 1.0, abs_tol=1e-6)


# 10. Distinct hypotheses with differing parameters are preserved
def test_distinct_parameter_configurations_preserved():
    # QPSK vs QPSK + convolutional FEC vs QPSK at different symbol rate
    c1 = _make_candidate("QPSK", symbol_rate=1000.0, fec=None, cand_id="q1")
    c2 = _make_candidate("QPSK", symbol_rate=1000.0, fec="conv_r1/2_k7", cand_id="q2")
    c3 = _make_candidate("QPSK", symbol_rate=2000.0, fec=None, cand_id="q3")
    c4 = _make_candidate("BPSK", symbol_rate=1000.0, fec=None, cand_id="b1")
    c5 = _make_candidate("BPSK", symbol_rate=1000.0, fec="conv_r1/2_k7", cand_id="b2")

    deduped = deduplicate_candidates([c1, c2, c3, c4, c5])
    # All 5 have distinct parameter configurations and must remain distinct
    assert len(deduped) == 5


# 11. Deterministic repeated execution produces identical scores
def test_deterministic_scoring_and_ranking():
    def run_scoring():
        cand1 = create_candidate("BPSK", 1000.0)
        cand1.evidence.ml = create_ml_evidence("BPSK", {"BPSK": 0.88})
        cand1.evidence.constellation = create_constellation_evidence(0.92, EvidenceStatus.AVAILABLE)

        cand2 = create_candidate("QPSK", 1000.0)
        cand2.evidence.ml = create_ml_evidence("QPSK", {"QPSK": 0.55})
        cand2.evidence.constellation = create_constellation_evidence(0.70, EvidenceStatus.AVAILABLE)

        return rank_hypotheses([cand1, cand2])

    res1 = run_scoring()
    res2 = run_scoring()

    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2):
        assert r1.modulation == r2.modulation
        assert math.isclose(r1.rawScore, r2.rawScore, abs_tol=1e-9)
        assert math.isclose(r1.confidenceScore, r2.confidenceScore, abs_tol=1e-9)


# 12. Existing API / result schema remains compatible
def test_schema_compatibility_with_frontend():
    c = _make_candidate("BPSK", symbol_rate=1200.0, raw_score=0.85)
    ranked = rank_hypotheses([c])
    cand_dict = ranked[0].to_dict()

    # Required frontend fields from frontend/src/types/index.ts
    required_keys = ["id", "modulation", "symbolRate", "confidenceScore", "details", "status"]
    for key in required_keys:
        assert key in cand_dict, f"Missing frontend field '{key}'"

    # Additional preserved backend fields
    assert "rawScore" in cand_dict
    assert "evidence" in cand_dict
    assert isinstance(cand_dict["evidence"], dict)
    assert "ml" in cand_dict["evidence"]


# 13. A single returned hypothesis receives confidence 1.0
def test_single_returned_hypothesis_receives_confidence_one():
    c = _make_candidate("BPSK", symbol_rate=1000.0, raw_score=0.35)
    normalized = normalize_confidences([c])
    assert len(normalized) == 1
    assert normalized[0].confidenceScore == 1.0


# 14. Empty hypothesis lists are handled safely
def test_empty_hypothesis_list_handled_safely():
    assert normalize_confidences([]) == []
    assert deduplicate_candidates([]) == []
    assert rank_hypotheses([]) == []


# 15. Configurable weights and temperature
def test_configurable_weights_and_temperature():
    cand1 = _make_candidate("BPSK", symbol_rate=1000.0)
    cand1.evidence.ml = create_ml_evidence("BPSK", {"BPSK": 0.90})
    cand1.evidence.constellation = create_constellation_evidence(0.30, EvidenceStatus.AVAILABLE)

    # Config A: heavy ML weight
    config_a = HypothesisScoringSettings(weight_ml=0.9, weight_constellation=0.1, temperature=0.1)
    calculate_raw_score(cand1, config_a)
    score_a = cand1.rawScore

    # Config B: heavy Constellation weight
    config_b = HypothesisScoringSettings(weight_ml=0.1, weight_constellation=0.9, temperature=0.1)
    calculate_raw_score(cand1, config_b)
    score_b = cand1.rawScore

    assert score_a > score_b
    assert math.isclose(score_a, (0.9 * 0.90 + 0.1 * 0.30), abs_tol=1e-5)
    assert math.isclose(score_b, (0.1 * 0.90 + 0.9 * 0.30), abs_tol=1e-5)

    # Temperature effect: lower temp -> more peaked confidence, higher temp -> flatter
    c_high = _make_candidate("M1", symbol_rate=1000.0, raw_score=0.8)
    c_low = _make_candidate("M2", symbol_rate=1000.0, raw_score=0.6)

    norm_peaked = normalize_confidences([c_high, c_low], temperature=0.05)
    conf_peaked = norm_peaked[0].confidenceScore

    norm_flat = normalize_confidences([c_high, c_low], temperature=1.0)
    conf_flat = norm_flat[0].confidenceScore

    assert conf_peaked > conf_flat


# 16. Unsupported modulation does not fabricate score
def test_unsupported_modulation_not_fabricated():
    supported = ["BPSK", "QPSK", "8PSK"]
    probs = {"BPSK": 0.80, "QPSK": 0.20}
    comp = create_ml_evidence("16QAM", probs, supported_classes=supported)

    assert comp.status == EvidenceStatus.NOT_SUPPORTED
    assert comp.score is None


# 17. Reporting format integration
def test_reporting_format():
    from backend.reporting.results import format_hypotheses_report

    cand = _make_candidate("BPSK", raw_score=0.85)
    ranked = rank_hypotheses([cand])
    report = format_hypotheses_report(ranked, metadata={"file": "test.iq"})

    assert report["total_hypotheses"] == 1
    assert report["top_hypothesis"]["modulation"] == "BPSK"
    assert report["metadata"]["file"] == "test.iq"
    assert len(report["rankings"]) == 1

