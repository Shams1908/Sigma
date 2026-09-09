"""
DecoderLab: End-to-End Hypothesis Decoding and Validation Facade.

Orchestrates the entire P5 workflow:
  Raw IQ + Sample Rate
  → ML priors & M8 parameter estimates
  → Bounded hypothesis search (generate_hypothesis_search_with_summary)
  → Multi-candidate DecoderChain execution with error isolation
  → Stage outcomes (Sync, Demod, Deinterleaver, FEC, CRC Validation)
  → Evidence mapping & dynamic raw scoring
  → Temperature-scaled softmax confidence normalization
  → Best hypothesis vs. validated decode resolution
  → Diagnostic telemetry and provenance reporting
"""
from dataclasses import dataclass, field
import math
import time
from typing import Dict, Any, Optional, List, Union
import numpy as np

try:
    from backend.core.config import settings, HypothesisScoringSettings, HypothesisSearchSettings
    from backend.hypothesis.candidates import HypothesisCandidate, EvidenceStatus
    from backend.hypothesis.generator import (
        HypothesisSearchConfig,
        HypothesisSearchSummary,
        SyncSearchConfig,
        generate_hypothesis_search_with_summary,
    )
    from backend.hypothesis.decoder_contracts import (
        DecoderConfig,
        DecoderRequest,
        DecoderResult,
        DecoderStageStatus,
    )
    from backend.hypothesis.decoder_chain import DecoderChain
    from backend.hypothesis.engine import HypothesisEngine, HypothesisEngineResult
except ImportError:
    from core.config import settings, HypothesisScoringSettings, HypothesisSearchSettings
    from hypothesis.candidates import HypothesisCandidate, EvidenceStatus
    from hypothesis.generator import (
        HypothesisSearchConfig,
        HypothesisSearchSummary,
        SyncSearchConfig,
        generate_hypothesis_search_with_summary,
    )
    from hypothesis.decoder_contracts import (
        DecoderConfig,
        DecoderRequest,
        DecoderResult,
        DecoderStageStatus,
    )
    from hypothesis.decoder_chain import DecoderChain
    from hypothesis.engine import HypothesisEngine, HypothesisEngineResult


@dataclass
class DecoderLabResult:
    """
    Public result container for an end-to-end Decoder Lab execution.

    Attributes:
        search_summary: Diagnostic metadata from bounded candidate search expansion.
        ranked_hypotheses: Candidate hypotheses ranked descending by confidenceScore.
        best_hypothesis: Highest-confidence candidate after ranking (regardless of decode status).
        validated_decode: Highest-ranked candidate among candidates where DecoderResult.success == True
                          (None if no candidate achieved a fully validated decode).
        decoder_results: Mapping from candidate ID to its canonical DecoderResult.
        diagnostics: Comprehensive execution telemetry, stage summaries, and error logs.
        total_evaluated: Total candidates evaluated.
        successful_decodes_count: Number of candidates with DecoderResult.success == True.
        failed_decodes_count: Number of candidates where decoding failed.
    """
    search_summary: Optional[Dict[str, Any]] = None
    ranked_hypotheses: List[HypothesisCandidate] = field(default_factory=list)
    best_hypothesis: Optional[HypothesisCandidate] = None
    validated_decode: Optional[HypothesisCandidate] = None
    decoder_results: Dict[str, DecoderResult] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    total_evaluated: int = 0
    successful_decodes_count: int = 0
    failed_decodes_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes result into a clean, JSON-compatible dictionary without huge raw arrays.
        """
        validated_info: Optional[Dict[str, Any]] = None
        if self.validated_decode is not None:
            v_res = self.decoder_results.get(self.validated_decode.id)
            bits_preview: Optional[str] = None
            if v_res and v_res.bits is not None:
                # Preview up to first 64 bits as bitstring
                bits_sample = v_res.bits[:64]
                bits_preview = "".join(str(int(b)) for b in bits_sample)
                if len(v_res.bits) > 64:
                    bits_preview += f"... (+{len(v_res.bits) - 64} bits)"

            validated_info = {
                "id": self.validated_decode.id,
                "modulation": self.validated_decode.modulation,
                "symbol_rate": self.validated_decode.symbolRate,
                "confidence": float(self.validated_decode.confidenceScore),
                "fec_config": self.validated_decode.fec_config,
                "interleaver_config": self.validated_decode.interleaver_config,
                "crc_scheme": (self.validated_decode.sync_assumptions or {}).get("crc_scheme"),
                "bits_count": len(v_res.bits) if v_res and v_res.bits is not None else 0,
                "bits_preview": bits_preview,
            }

        return {
            "total_evaluated": self.total_evaluated,
            "successful_decodes_count": self.successful_decodes_count,
            "failed_decodes_count": self.failed_decodes_count,
            "has_validated_decode": self.validated_decode is not None,
            "best_hypothesis_id": self.best_hypothesis.id if self.best_hypothesis else None,
            "best_hypothesis_modulation": self.best_hypothesis.modulation if self.best_hypothesis else None,
            "best_hypothesis_confidence": float(self.best_hypothesis.confidenceScore) if self.best_hypothesis else 0.0,
            "validated_decode": validated_info,
            "best_is_validated": (
                self.best_hypothesis is not None
                and self.validated_decode is not None
                and self.best_hypothesis.id == self.validated_decode.id
            ),
            "search_summary": self.search_summary,
            "ranked_hypotheses": [h.to_dict() for h in self.ranked_hypotheses],
            "decoder_results": {cid: res.to_dict() for cid, res in self.decoder_results.items()},
            "diagnostics": self.diagnostics,
        }


class DecoderLab:
    """
    Facade coordinating end-to-end signal decoding, hypothesis search,
    evidence aggregation, and candidate ranking.
    """

    def __init__(
        self,
        decoder_chain: Optional[DecoderChain] = None,
        scoring_config: Optional[HypothesisScoringSettings] = None,
        default_search_config: Optional[HypothesisSearchConfig] = None,
    ):
        self.decoder_chain = decoder_chain or DecoderChain()
        self.scoring_config = scoring_config or settings.HYPOTHESIS_SCORING
        self.default_search_config = default_search_config or HypothesisSearchConfig(
            sync_configurations=[
                SyncSearchConfig(
                    carrier_recovery="none",
                    timing_recovery="none",
                    matched_filter="none",
                )
            ]
        )
        self.engine = HypothesisEngine(
            decoder_chain=self.decoder_chain,
            scoring_config=self.scoring_config,
        )

    def _validate_iq_input(self, iq: np.ndarray, sample_rate: float) -> None:
        """
        Validates the IQ array and sample rate using canonical DecoderRequest validation rules.
        """
        if iq is None:
            raise ValueError("IQ data must not be None")
        if not isinstance(iq, np.ndarray):
            raise TypeError(f"IQ data must be a numpy.ndarray, got {type(iq)}")
        if iq.size == 0:
            raise ValueError("IQ array contains 0 samples")
        if not np.all(np.isfinite(iq)):
            raise ValueError("IQ array contains non-finite values (NaN or Inf)")

        # Validate layout
        if np.iscomplexobj(iq):
            if iq.ndim != 1:
                raise ValueError(f"Complex IQ array must be 1D with shape (N,), got ndim={iq.ndim}")
            if iq.shape[0] == 0:
                raise ValueError("Complex IQ array contains 0 samples")
        else:
            if iq.ndim == 1:
                raise ValueError(
                    "1D real IQ array layout is ambiguous (cannot distinguish In-Phase and Quadrature). "
                    "Expected shape (2, N) real or 1D complex array."
                )
            elif iq.ndim == 2:
                if iq.shape[0] != 2:
                    raise ValueError(
                        f"2D real IQ array must have shape (2, N) where row 0 is I and row 1 is Q. Got shape {iq.shape}"
                    )
                if iq.shape[1] == 0:
                    raise ValueError("IQ array contains 0 samples")
            else:
                raise ValueError(
                    f"Unsupported multidimensional IQ array shape {iq.shape}. Expected (2, N) real or 1D complex."
                )

        if not isinstance(sample_rate, (int, float)) or sample_rate <= 0 or not math.isfinite(sample_rate):
            raise ValueError(f"Sample rate must be a finite positive number, got {sample_rate}")

    def run(
        self,
        iq: np.ndarray,
        sample_rate: float,
        ml_probabilities: Optional[Dict[str, float]] = None,
        parameter_result: Optional[Any] = None,
        symbol_rate: Optional[float] = None,
        symbol_rate_uncertainty: Optional[float] = None,
        search_config: Optional[HypothesisSearchConfig] = None,
        decoder_config: Optional[DecoderConfig] = None,
        context: Optional[Dict[str, Any]] = None,
        deduplicate: bool = True,
    ) -> DecoderLabResult:
        """
        Executes the full Decoder Lab workflow.

        Args:
            iq: Canonical IQ signal array ([2, N] real or [N] complex).
            sample_rate: Signal sampling frequency in Hz.
            ml_probabilities: Optional dictionary mapping modulation scheme to probability.
            parameter_result: Optional M8 ParameterAnalysisResult containing symbol rate and SNR.
            symbol_rate: Optional explicit center symbol rate estimate in Baud.
            symbol_rate_uncertainty: Optional symbol rate uncertainty in Baud.
            search_config: Optional HypothesisSearchConfig controlling candidate expansion bounds.
            decoder_config: Optional DecoderConfig controlling pipeline stages.
            context: Optional contextual parameters (e.g. {'validation_config': ...}).
            deduplicate: Whether to deduplicate identical configurations.

        Returns:
            DecoderLabResult: Comprehensive outcome with ranked hypotheses and validated decode.
        """
        start_time = time.perf_counter()

        # Step 1: Validate IQ input array and sample rate
        self._validate_iq_input(iq, sample_rate)

        # Step 2: Resolve search configuration (strictly bounded max_candidates <= 24)
        active_search_cfg = search_config or self.default_search_config
        max_limit = min(24, active_search_cfg.max_candidates)
        sync_cfgs = active_search_cfg.sync_configurations
        if sync_cfgs is None:
            # Default to executable pass-through sync assumption so downstream demodulation/FEC/validation can execute
            sync_cfgs = [
                SyncSearchConfig(
                    carrier_recovery="none",
                    timing_recovery="none",
                    matched_filter="none",
                )
            ]

        active_search_cfg = HypothesisSearchConfig(
            top_k_modulations=active_search_cfg.top_k_modulations,
            min_ml_probability=active_search_cfg.min_ml_probability,
            include_fallback_modulation=active_search_cfg.include_fallback_modulation,
            supported_modulations=active_search_cfg.supported_modulations,
            symbol_rate_offsets=active_search_cfg.symbol_rate_offsets,
            symbol_rate_step_count=active_search_cfg.symbol_rate_step_count,
            symbol_rate_uncertainty_multiplier=active_search_cfg.symbol_rate_uncertainty_multiplier,
            fallback_uncertainty_fraction=active_search_cfg.fallback_uncertainty_fraction,
            max_symbol_rate_candidates=active_search_cfg.max_symbol_rate_candidates,
            sync_configurations=sync_cfgs,
            fec_candidates=active_search_cfg.fec_candidates,
            interleaver_candidates=active_search_cfg.interleaver_candidates,
            crc_candidates=active_search_cfg.crc_candidates,
            max_candidates=max_limit,
        )

        # Step 3: Resolve symbol rate and M8 parameters
        active_baud = symbol_rate
        active_unc = symbol_rate_uncertainty
        resolved_param_res = parameter_result
        estimation_source = "user_supplied"

        if resolved_param_res is not None:
            estimation_source = "m8_parameter_result"
        elif active_baud is None:
            # Fallback: Attempt clean DSP baseline symbol-rate and SNR estimation
            try:
                from ml.parameters.dsp_estimators import estimate_symbol_rate_dsp, estimate_snr_dsp
                dsp_sr = estimate_symbol_rate_dsp(iq, sample_rate=sample_rate)
                dsp_snr = estimate_snr_dsp(iq)
                if dsp_sr.estimate > 0:
                    active_baud = float(dsp_sr.estimate)
                    active_unc = float(dsp_sr.uncertainty)
                    estimation_source = "dsp_baseline_auto"

                # Construct lightweight M8-compatible mock structure for downstream evidence
                class _AutoSingleParam:
                    def __init__(self, est, unc, conf, meth):
                        self.estimate = est
                        self.uncertainty = unc
                        self.confidence = conf
                        self.method = meth
                    def to_dict(self):
                        return {"estimate": self.estimate, "uncertainty": self.uncertainty, "method": self.method}

                class _AutoParamAnalysis:
                    def __init__(self, sr_p, snr_p, s_rate):
                        self.symbol_rate = sr_p
                        self.snr_db = snr_p
                        self.sample_rate = s_rate

                resolved_param_res = _AutoParamAnalysis(
                    sr_p=_AutoSingleParam(active_baud, active_unc, dsp_sr.confidence, "dsp_spectral_line"),
                    snr_p=_AutoSingleParam(dsp_snr.estimate, dsp_snr.uncertainty, dsp_snr.confidence, "dsp_m2m4"),
                    s_rate=sample_rate,
                )
            except Exception:
                # If auto-estimation fails or is unavailable, leave as None
                pass

        # If symbol rate is still unresolved, use default nominal rate (1000.0 Baud) with NOT_EVALUATED evidence
        if active_baud is None or active_baud <= 0:
            active_baud = 1000.0
            estimation_source = "default_nominal_fallback"

        # Step 4: Resolve modulation probabilities
        priors_source = "user_supplied"
        active_probs = ml_probabilities
        if not active_probs:
            # Neutral baseline prior across common supported classes
            active_probs = {"BPSK": 0.50, "QPSK": 0.50}
            priors_source = "neutral_baseline_priors"

        # Step 5: Bounded candidate hypothesis generation via P5.2 generator
        candidates, search_summary = generate_hypothesis_search_with_summary(
            ml_probabilities=active_probs,
            symbol_rate=active_baud,
            symbol_rate_uncertainty=active_unc,
            parameter_result=resolved_param_res,
            search_config=active_search_cfg,
        )

        # Step 6: Multi-candidate execution, evidence aggregation, and ranking
        engine_res: HypothesisEngineResult = self.engine.execute_and_rank(
            iq=iq,
            sample_rate=sample_rate,
            candidates=candidates,
            validation_config=context.get("validation_config") if context else None,
            decoder_config=decoder_config,
            deduplicate=deduplicate,
        )

        # Step 7: Tally diagnostic outcomes across candidates
        unsupported_candidates = 0
        for cid, res in engine_res.decoder_results.items():
            if (
                res.synchronization.status == DecoderStageStatus.NOT_SUPPORTED
                or res.demodulation.status == DecoderStageStatus.NOT_SUPPORTED
                or res.interleaver.status == DecoderStageStatus.NOT_SUPPORTED
                or res.fec.status == DecoderStageStatus.NOT_SUPPORTED
                or res.validation.status == DecoderStageStatus.NOT_SUPPORTED
            ):
                unsupported_candidates += 1

        elapsed_time = time.perf_counter() - start_time

        diagnostics = {
            "execution_time_seconds": float(round(elapsed_time, 4)),
            "sample_rate_hz": float(sample_rate),
            "estimation_source": estimation_source,
            "priors_source": priors_source,
            "center_symbol_rate_baud": float(active_baud),
            "symbol_rate_uncertainty_baud": float(active_unc) if active_unc is not None else None,
            "candidate_count": len(engine_res.ranked_hypotheses),
            "successful_candidate_count": engine_res.successful_decodes_count,
            "failed_candidate_count": engine_res.failed_decodes_count,
            "unsupported_candidate_count": unsupported_candidates,
            "has_validated_decode": engine_res.validated_decode is not None,
            "validated_decode_id": engine_res.validated_decode.id if engine_res.validated_decode else None,
            "best_hypothesis_id": engine_res.best_hypothesis.id if engine_res.best_hypothesis else None,
            "best_is_validated": (
                engine_res.best_hypothesis is not None
                and engine_res.validated_decode is not None
                and engine_res.best_hypothesis.id == engine_res.validated_decode.id
            ),
        }

        return DecoderLabResult(
            search_summary=search_summary.to_dict(),
            ranked_hypotheses=engine_res.ranked_hypotheses,
            best_hypothesis=engine_res.best_hypothesis,
            validated_decode=engine_res.validated_decode,
            decoder_results=engine_res.decoder_results,
            diagnostics=diagnostics,
            total_evaluated=engine_res.total_evaluated,
            successful_decodes_count=engine_res.successful_decodes_count,
            failed_decodes_count=engine_res.failed_decodes_count,
        )


def run_decoder_lab(
    iq: np.ndarray,
    sample_rate: float,
    ml_probabilities: Optional[Dict[str, float]] = None,
    parameter_result: Optional[Any] = None,
    symbol_rate: Optional[float] = None,
    symbol_rate_uncertainty: Optional[float] = None,
    search_config: Optional[HypothesisSearchConfig] = None,
    decoder_config: Optional[DecoderConfig] = None,
    context: Optional[Dict[str, Any]] = None,
    deduplicate: bool = True,
) -> DecoderLabResult:
    """
    Public convenience entrypoint executing the complete end-to-end Decoder Lab workflow.
    """
    lab = DecoderLab()
    return lab.run(
        iq=iq,
        sample_rate=sample_rate,
        ml_probabilities=ml_probabilities,
        parameter_result=parameter_result,
        symbol_rate=symbol_rate,
        symbol_rate_uncertainty=symbol_rate_uncertainty,
        search_config=search_config,
        decoder_config=decoder_config,
        context=context,
        deduplicate=deduplicate,
    )
