"""
Decoder-Chain Orchestration for SIGMA.
Orchestrates the execution of decoding stages in strict dependency order:
  Synchronization -> Demodulation -> Deinterleaving -> FEC -> Validation.
Propagates stage failures, maintains DecoderContext progression, and constructs
complete DecoderResult without fabricating evidence or decode success.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np

try:
    from backend.hypothesis.candidates import HypothesisCandidate
    from backend.hypothesis.decoder_contracts import (
        DecoderStageStatus,
        SynchronizationResult,
        DemodulationResult,
        InterleaverResult,
        FECResult,
        ValidationResult,
        DecoderConfig,
        DecoderRequest,
        DecoderContext,
        DecoderResult,
        DecoderEngine,
    )
except ImportError:
    from hypothesis.candidates import HypothesisCandidate
    from hypothesis.decoder_contracts import (
        DecoderStageStatus,
        SynchronizationResult,
        DemodulationResult,
        InterleaverResult,
        FECResult,
        ValidationResult,
        DecoderConfig,
        DecoderRequest,
        DecoderContext,
        DecoderResult,
        DecoderEngine,
    )


class DecoderStage(ABC):
    """
    Abstract interface for a discrete processing stage in the decoder chain.
    """
    @abstractmethod
    def execute(self, context: DecoderContext) -> Any:
        """
        Executes stage processing using and updating the shared DecoderContext.
        Returns the stage-specific result contract.
        """
        raise NotImplementedError


class SynchronizationAdapter(DecoderStage):
    """
    Adapter connecting synchronization algorithms to the decoder chain.
    Inspects candidate sync_assumptions (carrier recovery, timing recovery, matched filter)
    and dispatches to available implementations under backend/synchronization/.
    """
    def __init__(self, handler: Optional[Callable[[DecoderContext], SynchronizationResult]] = None):
        self._handler = handler

    def execute(self, context: DecoderContext) -> SynchronizationResult:
        if self._handler is not None:
            res = self._handler(context)
            context.sync_result = res
            return res

        sync_cfg = context.hypothesis.sync_assumptions if context.hypothesis and context.hypothesis.sync_assumptions else {}

        carrier_recovery = sync_cfg.get("carrier_recovery")
        timing_recovery = sync_cfg.get("timing_recovery")
        matched_filter_type = sync_cfg.get("matched_filter", "none")

        # 1. Check if carrier recovery is requested
        if carrier_recovery and str(carrier_recovery).lower() not in ("none", "bypass"):
            norm_cr = str(carrier_recovery).lower()
            res = SynchronizationResult.not_supported(
                failure_reason=f"Carrier recovery method '{norm_cr}' is not implemented in backend/synchronization/carrier_recovery.py"
            )
            context.sync_result = res
            return res

        # 2. Check if timing recovery is requested
        if timing_recovery and str(timing_recovery).lower() not in ("none", "bypass"):
            norm_tr = str(timing_recovery).lower()
            res = SynchronizationResult.not_supported(
                failure_reason=f"Timing recovery method '{norm_tr}' is not implemented in backend/synchronization/timing_recovery.py"
            )
            context.sync_result = res
            return res

        # 3. Matched filter execution
        norm_mf = str(matched_filter_type).lower()
        if norm_mf in ("none", "bypass"):
            # Pass-through identity sync
            context.synchronized_iq = context.original_iq.copy()
            res = SynchronizationResult.create_success(
                synchronized_signal=context.synchronized_iq,
                timing_score=1.0,
                details={"bypass": True, "method": "identity"},
            )
            context.sync_result = res
            return res
        elif norm_mf in ("rrc", "root_raised_cosine", "rc"):
            try:
                from backend.synchronization.matched_filter import apply_matched_filter
                rolloff = float(sync_cfg.get("filter_rolloff", 0.35))
                filtered = apply_matched_filter(
                    signal=context.original_iq,
                    filter_type=norm_mf,
                    rolloff=rolloff,
                )
                context.synchronized_iq = filtered
                res = SynchronizationResult.create_success(
                    synchronized_signal=filtered,
                    timing_score=0.95,
                    metrics={"matched_filter": norm_mf, "rolloff": rolloff},
                )
                context.sync_result = res
                return res
            except Exception as e:
                res = SynchronizationResult.failed(
                    failure_reason=f"Matched filter execution error: {str(e)}"
                )
                context.sync_result = res
                return res
        else:
            res = SynchronizationResult.not_supported(
                failure_reason=f"Matched filter type '{matched_filter_type}' is not supported"
            )
            context.sync_result = res
            return res


class DemodulationAdapter(DecoderStage):
    """
    Adapter connecting demodulation algorithms to the decoder chain.
    Maps candidate modulation to the appropriate demodulator under backend/demodulation/.
    """
    def __init__(self, handler: Optional[Callable[[DecoderContext], DemodulationResult]] = None):
        self._handler = handler

    def execute(self, context: DecoderContext) -> DemodulationResult:
        if self._handler is not None:
            res = self._handler(context)
            context.demod_result = res
            return res

        # Select input signal (synchronized IQ if available, else original IQ)
        signal = context.synchronized_iq if context.synchronized_iq is not None else context.original_iq
        if signal is None:
            res = DemodulationResult.failed(failure_reason="No input signal available for demodulation")
            context.demod_result = res
            return res

        mod_name = context.modulation.strip().upper()

        # 1. PSK Family (BPSK, QPSK, 8PSK)
        if mod_name in ("BPSK", "QPSK"):
            try:
                from backend.demodulation.psk_demod import demodulate_psk
                syms, bits, score, metrics = demodulate_psk(signal, modulation=mod_name)
                context.symbols = syms
                context.raw_bits = bits
                res = DemodulationResult.create_success(
                    symbols=syms,
                    bits=bits,
                    constellation_score=score,
                    quality_metrics=metrics,
                )
                context.demod_result = res
                return res
            except Exception as e:
                res = DemodulationResult.failed(failure_reason=f"PSK demodulation failed: {str(e)}")
                context.demod_result = res
                return res
        elif mod_name == "8PSK":
            res = DemodulationResult.not_supported(
                failure_reason="8PSK demodulation not yet implemented in backend/demodulation/psk_demod.py"
            )
            context.demod_result = res
            return res

        # 2. QAM Family (16QAM, 64QAM)
        elif mod_name in ("16QAM", "64QAM", "QAM16", "QAM64"):
            res = DemodulationResult.not_supported(
                failure_reason=f"QAM demodulator for '{mod_name}' not yet implemented in backend/demodulation/qam_demod.py"
            )
            context.demod_result = res
            return res

        # 3. FSK Family (CPFSK, GFSK, FSK)
        elif mod_name in ("CPFSK", "GFSK", "FSK"):
            res = DemodulationResult.not_supported(
                failure_reason=f"FSK demodulator for '{mod_name}' not yet implemented in backend/demodulation/fsk_demod.py"
            )
            context.demod_result = res
            return res

        # 4. Other / Unsupported
        else:
            res = DemodulationResult.not_supported(
                failure_reason=f"Modulation '{mod_name}' is not supported by available demodulators"
            )
            context.demod_result = res
            return res


class InterleaverAdapter(DecoderStage):
    """
    Adapter for deinterleaving stage.
    If no interleaver is configured, returns NOT_EVALUATED and passes raw bits through.
    Supports block_RxC configurations via BlockDeinterleaver.
    Returns NOT_SUPPORTED for unrecognized interleaver configurations.
    """
    def __init__(self, handler: Optional[Callable[[DecoderContext], InterleaverResult]] = None):
        self._handler = handler

    def execute(self, context: DecoderContext) -> InterleaverResult:
        if self._handler is not None:
            res = self._handler(context)
            context.interleaver_result = res
            return res

        intl_cfg = context.hypothesis.interleaver_config if context.hypothesis else None

        if not intl_cfg or str(intl_cfg).lower() in ("none", "bypass"):
            # Deinterleaver bypassed
            if context.raw_bits is not None:
                context.deinterleaved_bits = context.raw_bits
            res = InterleaverResult.not_evaluated(details={"reason": "No interleaver configured"})
            context.interleaver_result = res
            return res

        # Check for block interleaver configuration (e.g. block_16x16, block_4x8)
        block_dims = None
        try:
            from backend.interleaver import parse_block_interleaver_config, BlockDeinterleaver
            block_dims = parse_block_interleaver_config(intl_cfg)
        except ImportError:
            try:
                from interleaver import parse_block_interleaver_config, BlockDeinterleaver
                block_dims = parse_block_interleaver_config(intl_cfg)
            except ImportError:
                pass

        if block_dims is not None:
            rows, cols = block_dims
            raw_bits = context.raw_bits
            if raw_bits is None:
                res = InterleaverResult.failed(failure_reason="No raw bits available for deinterleaving")
                context.interleaver_result = res
                return res

            deinterleaver = BlockDeinterleaver(rows=rows, cols=cols)
            res = deinterleaver.deinterleave(raw_bits)
            if res.success and res.deinterleaved_bits is not None:
                context.deinterleaved_bits = res.deinterleaved_bits
            context.interleaver_result = res
            return res

        # Interleaver requested but unsupported
        res = InterleaverResult.not_supported(
            failure_reason=f"Interleaver '{intl_cfg}' is not supported: no deinterleaver algorithm available"
        )
        context.interleaver_result = res
        return res


class FECAdapter(DecoderStage):
    """
    Adapter for Forward Error Correction stage.
    If no FEC is configured, returns NOT_EVALUATED and passes bits through.
    Supports 'conv_r1/2_k7' via ConvolutionalFECDecoder.
    Returns NOT_SUPPORTED for unrecognized FEC configurations.
    """
    def __init__(self, handler: Optional[Callable[[DecoderContext], FECResult]] = None):
        self._handler = handler

    def execute(self, context: DecoderContext) -> FECResult:
        if self._handler is not None:
            res = self._handler(context)
            context.fec_result = res
            return res

        fec_cfg = context.hypothesis.fec_config if context.hypothesis else None

        if not fec_cfg or str(fec_cfg).lower() in ("none", "bypass"):
            # FEC bypassed
            input_bits = context.deinterleaved_bits if context.deinterleaved_bits is not None else context.raw_bits
            if input_bits is not None:
                context.decoded_bits = input_bits
            res = FECResult.not_evaluated(details={"reason": "No FEC configured"})
            context.fec_result = res
            return res

        norm_fec = str(fec_cfg).strip().lower()
        if norm_fec in ("conv_r1/2_k7", "conv_r12_k7", "convolutional_r1/2_k7"):
            input_bits = context.deinterleaved_bits if context.deinterleaved_bits is not None else context.raw_bits
            if input_bits is None:
                res = FECResult.failed(failure_reason="No input bits available for FEC decoding")
                context.fec_result = res
                return res

            try:
                from backend.fec import ConvolutionalFECDecoder
            except ImportError:
                from fec import ConvolutionalFECDecoder

            try:
                decoder = ConvolutionalFECDecoder()
                res = decoder.decode(input_bits)
                if res.success and res.decoded_bits is not None:
                    context.decoded_bits = res.decoded_bits
                context.fec_result = res
                return res
            except Exception as e:
                res = FECResult.failed(failure_reason=f"Convolutional FEC decoding failed: {str(e)}")
                context.fec_result = res
                return res

        # FEC requested but unsupported
        res = FECResult.not_supported(
            failure_reason=f"FEC scheme '{fec_cfg}' is not supported: no FEC decoder available"
        )
        context.fec_result = res
        return res


class ValidationAdapter(DecoderStage):
    """
    Adapter for bitstream, CRC, and frame validation stage.
    Validates decoded/demodulated bits when explicit validation criteria (CRC scheme or framing)
    are requested.
    Does NOT fabricate validation success or enforce arbitrary protocols unless configured.
    If no validation criteria are configured, returns NOT_SUPPORTED with is_valid=False.
    """
    def __init__(self, handler: Optional[Callable[[DecoderContext], ValidationResult]] = None):
        self._handler = handler

    def execute(self, context: DecoderContext) -> ValidationResult:
        if self._handler is not None:
            res = self._handler(context)
            context.validation_result = res
            return res

        # Extract bits to validate (priority: decoded_bits -> deinterleaved_bits -> raw_bits)
        input_bits = None
        if context.decoded_bits is not None:
            input_bits = context.decoded_bits
        elif context.deinterleaved_bits is not None:
            input_bits = context.deinterleaved_bits
        elif context.raw_bits is not None:
            input_bits = context.raw_bits

        # Check for explicit validation configuration
        val_cfg = context.metadata.get("validation_config") or {}
        crc_scheme = (
            val_cfg.get("crc_scheme")
            or context.metadata.get("crc_scheme")
            or (context.hypothesis.sync_assumptions.get("crc_scheme") if context.hypothesis and context.hypothesis.sync_assumptions else None)
            or (context.hypothesis.sync_assumptions.get("crc") if context.hypothesis and context.hypothesis.sync_assumptions else None)
        )
        sync_word = val_cfg.get("sync_word") or context.metadata.get("sync_word")
        frame_len = val_cfg.get("frame_length") or context.metadata.get("frame_length")

        if not crc_scheme and not sync_word:
            # No protocol or CRC validation criteria configured for this candidate
            res = ValidationResult.not_supported(
                failure_reason="No protocol or CRC validation scheme configured for candidate"
            )
            context.validation_result = res
            return res

        if input_bits is None:
            res = ValidationResult.failed(
                failure_reason="No bits available for validation stage",
                crc_status="not_evaluated",
            )
            context.validation_result = res
            return res

        try:
            from backend.validation import FrameValidator
        except ImportError:
            from validation import FrameValidator

        try:
            validator = FrameValidator()
            res = validator.validate(
                bits=input_bits,
                crc_scheme=crc_scheme,
                sync_word=sync_word,
                frame_length=frame_len,
            )
            context.validation_result = res
            return res
        except Exception as e:
            res = ValidationResult.failed(
                failure_reason=f"Validation execution failed: {str(e)}",
                crc_status="invalid",
            )
            context.validation_result = res
            return res


class DecoderChain(DecoderEngine):
    """
    Decoder-chain orchestration engine executing decoding stages in dependency order:
      1. Synchronization
      2. Demodulation
      3. Deinterleaving
      4. Forward Error Correction (FEC)
      5. Bitstream / CRC Validation

    Handles stage failures and unsupported configurations explicitly by propagating
    NOT_EVALUATED status to downstream stages.
    Guarantees DecoderResult.success is False unless the entire required pipeline succeeds
    and produces a validated result.
    """
    def __init__(
        self,
        sync_stage: Optional[SynchronizationAdapter] = None,
        demod_stage: Optional[DemodulationAdapter] = None,
        interleaver_stage: Optional[InterleaverAdapter] = None,
        fec_stage: Optional[FECAdapter] = None,
        validation_stage: Optional[ValidationAdapter] = None,
    ):
        self.sync_stage = sync_stage or SynchronizationAdapter()
        self.demod_stage = demod_stage or DemodulationAdapter()
        self.interleaver_stage = interleaver_stage or InterleaverAdapter()
        self.fec_stage = fec_stage or FECAdapter()
        self.validation_stage = validation_stage or ValidationAdapter()

    def decode(self, request: DecoderRequest) -> DecoderResult:
        """
        Executes the decoder chain on the provided request.

        Args:
            request: Canonical DecoderRequest with valid IQ, hypothesis, and sample rate.

        Returns:
            DecoderResult: Structured outcome across all decoding stages.
        """
        request.validate()
        context = DecoderContext.from_request(request)
        config = request.config

        # -------------------------------------------------------------
        # Stage 1: Synchronization
        # -------------------------------------------------------------
        if config.enable_sync:
            sync_res = self.sync_stage.execute(context)
            if not sync_res.success:
                # Synchronization failed or unsupported -> abort downstream stages
                return self._abort_after_stage(
                    context,
                    aborted_at="synchronization",
                    failure_reason=sync_res.failure_reason or "Synchronization stage failed",
                )
        else:
            context.sync_result = SynchronizationResult.not_evaluated(
                details={"bypassed": True, "reason": "Synchronization disabled in config"}
            )

        # -------------------------------------------------------------
        # Stage 2: Demodulation
        # -------------------------------------------------------------
        if config.enable_demod:
            demod_res = self.demod_stage.execute(context)
            if not demod_res.success:
                # Demodulation failed or unsupported -> abort downstream stages
                return self._abort_after_stage(
                    context,
                    aborted_at="demodulation",
                    failure_reason=demod_res.failure_reason or "Demodulation stage failed",
                )
        else:
            context.demod_result = DemodulationResult.not_evaluated(
                details={"bypassed": True, "reason": "Demodulation disabled in config"}
            )

        # -------------------------------------------------------------
        # Stage 3: Deinterleaving
        # -------------------------------------------------------------
        if config.enable_deinterleaver:
            intl_res = self.interleaver_stage.execute(context)
            if intl_res.status in (DecoderStageStatus.FAILED, DecoderStageStatus.NOT_SUPPORTED):
                return self._abort_after_stage(
                    context,
                    aborted_at="interleaver",
                    failure_reason=intl_res.failure_reason or "Interleaver stage failed",
                )
        else:
            context.interleaver_result = InterleaverResult.not_evaluated(
                details={"bypassed": True, "reason": "Deinterleaver disabled in config"}
            )

        # -------------------------------------------------------------
        # Stage 4: FEC Decoding
        # -------------------------------------------------------------
        if config.enable_fec:
            fec_res = self.fec_stage.execute(context)
            if fec_res.status in (DecoderStageStatus.FAILED, DecoderStageStatus.NOT_SUPPORTED):
                return self._abort_after_stage(
                    context,
                    aborted_at="fec",
                    failure_reason=fec_res.failure_reason or "FEC stage failed",
                )
        else:
            context.fec_result = FECResult.not_evaluated(
                details={"bypassed": True, "reason": "FEC disabled in config"}
            )

        # -------------------------------------------------------------
        # Stage 5: Validation
        # -------------------------------------------------------------
        if config.enable_validation:
            self.validation_stage.execute(context)
        else:
            context.validation_result = ValidationResult.not_evaluated(
                details={"bypassed": True, "reason": "Validation disabled in config"}
            )

        # -------------------------------------------------------------
        # Assemble Final Result
        # -------------------------------------------------------------
        val_res = context.validation_result or ValidationResult.not_evaluated()

        # DecoderResult.success is True ONLY when a complete valid decode was produced
        overall_success = (val_res.status == DecoderStageStatus.SUCCESS and val_res.is_valid)

        failure_reason = None
        if not overall_success:
            if val_res.status == DecoderStageStatus.NOT_SUPPORTED:
                failure_reason = val_res.failure_reason or "Validation stage not supported"
            elif val_res.failure_reason:
                failure_reason = val_res.failure_reason
            else:
                failure_reason = "Decoder chain did not produce a validated bitstream"

        return self._build_result(context, overall_success=overall_success, failure_reason=failure_reason)

    def _abort_after_stage(
        self,
        context: DecoderContext,
        aborted_at: str,
        failure_reason: str,
    ) -> DecoderResult:
        """
        Fills un-executed downstream stages with explicit NOT_EVALUATED results and constructs
        the final aborted DecoderResult.
        """
        msg = f"Skipped: stage '{aborted_at}' did not succeed"

        if aborted_at == "synchronization":
            context.demod_result = DemodulationResult.not_evaluated(failure_reason=msg)
            context.interleaver_result = InterleaverResult.not_evaluated(failure_reason=msg)
            context.fec_result = FECResult.not_evaluated(failure_reason=msg)
            context.validation_result = ValidationResult.not_evaluated(failure_reason=msg)
        elif aborted_at == "demodulation":
            context.interleaver_result = InterleaverResult.not_evaluated(failure_reason=msg)
            context.fec_result = FECResult.not_evaluated(failure_reason=msg)
            context.validation_result = ValidationResult.not_evaluated(failure_reason=msg)
        elif aborted_at == "interleaver":
            context.fec_result = FECResult.not_evaluated(failure_reason=msg)
            context.validation_result = ValidationResult.not_evaluated(failure_reason=msg)
        elif aborted_at == "fec":
            context.validation_result = ValidationResult.not_evaluated(failure_reason=msg)

        return self._build_result(context, overall_success=False, failure_reason=failure_reason)

    def _build_result(
        self,
        context: DecoderContext,
        overall_success: bool,
        failure_reason: Optional[str] = None,
    ) -> DecoderResult:
        """Constructs the canonical DecoderResult from progressive context state."""
        # Determine payload bits (decoded_bits -> deinterleaved_bits -> raw_bits)
        final_bits = None
        if context.decoded_bits is not None:
            final_bits = context.decoded_bits
        elif context.deinterleaved_bits is not None:
            final_bits = context.deinterleaved_bits
        elif context.raw_bits is not None:
            final_bits = context.raw_bits

        diag = dict(context.metadata)
        diag["orchestrator"] = "DecoderChain"
        diag["modulation"] = context.modulation
        diag["sample_rate"] = context.sample_rate
        if context.symbol_rate:
            diag["symbol_rate"] = context.symbol_rate

        return DecoderResult(
            success=overall_success,
            hypothesis_id=context.hypothesis.id if context.hypothesis else "unknown",
            bits=final_bits,
            symbols=context.symbols,
            synchronized_signal=context.synchronized_iq,
            synchronization=context.sync_result or SynchronizationResult.not_evaluated(),
            demodulation=context.demod_result or DemodulationResult.not_evaluated(),
            interleaver=context.interleaver_result or InterleaverResult.not_evaluated(),
            fec=context.fec_result or FECResult.not_evaluated(),
            validation=context.validation_result or ValidationResult.not_evaluated(),
            failure_reason=failure_reason,
            diagnostic_details=diag,
        )
