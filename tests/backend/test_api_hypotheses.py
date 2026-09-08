"""
Tests for the hypothesis ranking API route handlers in backend.api.results.
"""
import pytest
import asyncio
from backend.api.results import (
    get_hypotheses_ranking,
    rank_candidates_endpoint,
    RankHypothesesRequest,
    HypothesisCandidateSchema,
)


@pytest.mark.anyio
async def test_get_hypotheses_endpoint():
    data = await get_hypotheses_ranking(symbol_rate=1200.0)

    assert data.total_candidates > 0
    candidates = data.candidates
    assert len(candidates) > 0

    # Verify confidence properties
    conf_sum = sum(c["confidenceScore"] for c in candidates)
    assert abs(conf_sum - 1.0) < 1e-5

    # Verify each candidate structure
    for c in candidates:
        assert "id" in c
        assert "modulation" in c
        assert "symbolRate" in c
        assert "confidenceScore" in c
        assert "rawScore" in c
        assert "evidence" in c
        assert 0.0 <= c["confidenceScore"] <= 1.0


@pytest.mark.anyio
async def test_post_rank_candidates_endpoint():
    req = RankHypothesesRequest(
        candidates=[
            HypothesisCandidateSchema(
                id="cand_1",
                modulation="BPSK",
                symbolRate=1000.0,
                evidence={
                    "ml": {"status": "available", "score": 0.85},
                    "constellation": {"status": "available", "score": 0.90},
                }
            ),
            HypothesisCandidateSchema(
                id="cand_2",
                modulation="QPSK",
                symbolRate=1000.0,
                evidence={
                    "ml": {"status": "available", "score": 0.40},
                    "constellation": {"status": "available", "score": 0.50},
                }
            ),
            # Duplicate BPSK from different search path
            HypothesisCandidateSchema(
                id="cand_3",
                modulation="BPSK",
                symbolRate=1000.0,
                evidence={
                    "ml": {"status": "available", "score": 0.82},
                    "constellation": {"status": "available", "score": 0.88},
                }
            )
        ],
        temperature=0.25,
    )

    data = await rank_candidates_endpoint(req)
    candidates = data.candidates

    # Candidate 3 should be deduplicated with Candidate 1
    assert len(candidates) == 2

    # Top candidate should be BPSK
    assert candidates[0]["modulation"] == "BPSK"
    assert candidates[0]["rawScore"] > candidates[1]["rawScore"]
    assert candidates[0]["confidenceScore"] > candidates[1]["confidenceScore"]

    # Sum of confidences = 1.0
    conf_sum = sum(c["confidenceScore"] for c in candidates)
    assert abs(conf_sum - 1.0) < 1e-5
