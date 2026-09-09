"""
HypothesisEngine: Multi-Candidate Decoding and Evidence Ranking Integration.
Orchestrates:
  HypothesisCandidate[] -> DecoderChain -> Evidence Extraction -> RankHypotheses.
Provides:
  - Error isolation across candidates (exception in one does not crash the search).
  - Canonical evidence mapping through decoder_result_to_evidence_trace.
  - Strict separation of best_hypothesis (highest confidence) from validated_decode (highest-ranked valid decode).
  - Non-fabrication of evidence scores for unrun/unsupported stages.
  - Deterministic evaluation and confidence normalization.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import numpy as np

try:
    from backend.core.config import settings, HypothesisScoringSettings
    from backend.hypothesis.candidates import HypothesisCandidate
    from backend.hypothesis.decoder_contracts import (
        DecoderConfig,
        DecoderRequest,
        DecoderResult,
        SynchronizationResult,
        DemodulationResult,
        InterleaverResult,
        FECResult,
        ValidationResult,
        update_hypothesis_from_decoder_result,
    )
    from backend.hypothesis.decoder_chain import DecoderChain
    from backend.hypothesis.ranking import rank_hypotheses
except ImportError:
    from core.config import settings, HypothesisScoringSettings
    from hypothesis.candidates import HypothesisCandidate
    from hypothesis.decoder_contracts import (
        DecoderConfig,
        DecoderRequest,
        DecoderResult,
        SynchronizationResult,
        DemodulationResult,
        InterleaverResult,
        FECResult,
        ValidationResult,
        update_hypothesis_from_decoder_result,
    )
    from hypothesis.decoder_chain import DecoderChain
    from hypothesis.ranking import rank_hypotheses


@dataclass
class HypothesisEngineResult:
    """
    Result container for multi-candidate hypothesis decoding, evidence aggregation, and ranking.

    Attributes:
        ranked_hypotheses: Candidate hypotheses ranked descending by confidenceScore.
        best_hypothesis: Top-ranked candidate (highest confidenceScore), regardless of decode status.
        validated_decode: Highest-ranked candidate among candidates where DecoderResult.success == True
                          (None if no candidate achieved a fully validated decode).
        decoder_results: Mapping from candidate ID to its respective canonical DecoderResult.
        total_evaluated: Total number of candidates evaluated.
        successful_decodes_count: Number of candidates with DecoderResult.success == True.
        failed_decodes_count: Number of candidates where decoding failed or was not validated.
        summary: Diagnostic execution telemetry and metadata.
    """
    ranked_hypotheses: List[HypothesisCandidate]
    best_hypothesis: Optional[HypothesisCandidate] = None
    validated_decode: Optional[HypothesisCandidate] = None
    decoder_results: Dict[str, DecoderResult] = field(default_factory=dict)
    total_evaluated: int = 0
    successful_decodes_count: int = 0
    failed_decodes_count: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result into an explicit, JSON-compatible dictionary."""
        return {
            "total_evaluated": self.total_evaluated,
            "successful_decodes_count": self.successful_decodes_count,
            "failed_decodes_count": self.failed_decodes_count,
            "best_hypothesis_id": self.best_hypothesis.id if self.best_hypothesis else None,
            "best_hypothesis_modulation": self.best_hypothesis.modulation if self.best_hypothesis else None,
            "best_hypothesis_confidence": float(self.best_hypothesis.confidenceScore) if self.best_hypothesis else 0.0,
            "has_validated_decode": self.validated_decode is not None,
            "validated_decode_id": self.validated_decode.id if self.validated_decode else None,
            "ranked_hypotheses": [h.to_dict() for h in self.ranked_hypotheses],
            "decoder_results": {cid: res.to_dict() for cid, res in self.decoder_results.items()},
            "summary": self.summary,
        }


class HypothesisEngine:
    """
    Orchestrates the evaluation of multiple candidate hypotheses through DecoderChain,
    maps outcomes into the canonical evidence trace, and applies the existing ranking
    and confidence normalization engine.
    """

    def __init__(
        self,
        decoder_chain: Optional[DecoderChain] = None,
        scoring_config: Optional[HypothesisScoringSettings] = None,
    ):
        self.decoder_chain = decoder_chain or DecoderChain()
        self.scoring_config = scoring_config or settings.HYPOTHESIS_SCORING

    def execute_and_rank(
        self,
        iq: np.ndarray,
        sample_rate: float,
        candidates: List[HypothesisCandidate],
        validation_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[DecoderConfig] = None,
        deduplicate: bool = True,
    ) -> HypothesisEngineResult:
        """
        Executes decoding across candidates, aggregates evidence, and produces ranked hypotheses.

        Args:
            iq: Canonical IQ signal array (2D float32 [2, N] or 1D complex [N]).
            sample_rate: Signal sampling rate in Hz.
            candidates: List of HypothesisCandidate instances to evaluate.
            validation_config: Optional validation options (e.g. {'crc_scheme': 'CRC-16-CCITT'}).
            decoder_config: Optional DecoderConfig controlling pipeline stages.
            deduplicate: Whether to deduplicate identical configurations during ranking.

        Returns:
            HypothesisEngineResult: Ranked candidates with distinct best_hypothesis and validated_decode.
        """
        if not candidates:
            return HypothesisEngineResult(
                ranked_hypotheses=[],
                best_hypothesis=None,
                validated_decode=None,
                decoder_results={},
                total_evaluated=0,
                successful_decodes_count=0,
                failed_decodes_count=0,
                summary={"reason": "Empty candidate list provided"},
            )

        decoder_results: Dict[str, DecoderResult] = {}
        successful_decodes = 0
        failed_decodes = 0

        cfg = decoder_config or DecoderConfig()

        # Execute DecoderChain for each candidate independently with strict error isolation
        for cand in candidates:
            # Build per-candidate context
            context_dict: Dict[str, Any] = {}
            if validation_config:
                context_dict["validation_config"] = dict(validation_config)

            # Check if candidate specifies validation / CRC in sync_assumptions
            if cand.sync_assumptions and "crc_scheme" in cand.sync_assumptions:
                if "validation_config" not in context_dict:
                    context_dict["validation_config"] = {}
                context_dict["validation_config"]["crc_scheme"] = cand.sync_assumptions["crc_scheme"]

            try:
                request = DecoderRequest(
                    iq=iq,
                    hypothesis=cand,
                    sample_rate=sample_rate,
                    config=cfg,
                    context=context_dict if context_dict else None,
                )
                result = self.decoder_chain.decode(request)
            except Exception as e:
                # Error isolation: an unexpected exception in one candidate does NOT abort the search.
                # Construct a canonical failed DecoderResult and pass it through the canonical evidence path.
                result = DecoderResult(
                    success=False,
                    hypothesis_id=cand.id,
                    failure_reason=f"Decoder execution exception: {str(e)}",
                    synchronization=SynchronizationResult.failed(failure_reason=f"Exception: {str(e)}"),
                    demodulation=DemodulationResult.not_evaluated(failure_reason="Skipped due to exception"),
                    interleaver=InterleaverResult.not_evaluated(failure_reason="Skipped due to exception"),
                    fec=FECResult.not_evaluated(failure_reason="Skipped due to exception"),
                    validation=ValidationResult.not_evaluated(failure_reason="Skipped due to exception"),
                    diagnostic_details={"exception": str(e), "exception_type": type(e).__name__},
                )

            # Canonical evidence update: pass through update_hypothesis_from_decoder_result
            update_hypothesis_from_decoder_result(cand, result)
            decoder_results[cand.id] = result

            if result.success:
                successful_decodes += 1
            else:
                failed_decodes += 1

        # Rank candidates using existing ranking system (raw scoring + softmax normalization)
        ranked = rank_hypotheses(
            candidates=candidates,
            config=self.scoring_config,
            deduplicate=deduplicate,
        )

        # 1. best_hypothesis: highest-confidence candidate, regardless of decode success
        best_hypothesis = ranked[0] if ranked else None

        # 2. validated_decode: highest-ranked candidate where DecoderResult.success == True
        validated_decode = None
        for h in ranked:
            h_res = decoder_results.get(h.id)
            if h_res is not None and h_res.success:
                validated_decode = h
                break

        summary = {
            "total_candidates": len(candidates),
            "ranked_candidates_count": len(ranked),
            "successful_decodes": successful_decodes,
            "failed_decodes": failed_decodes,
            "deduplicated": deduplicate,
            "temperature_used": self.scoring_config.temperature,
            "best_is_validated": (best_hypothesis is not None and validated_decode is not None and best_hypothesis.id == validated_decode.id),
        }

        return HypothesisEngineResult(
            ranked_hypotheses=ranked,
            best_hypothesis=best_hypothesis,
            validated_decode=validated_decode,
            decoder_results=decoder_results,
            total_evaluated=len(candidates),
            successful_decodes_count=successful_decodes,
            failed_decodes_count=failed_decodes,
            summary=summary,
        )
