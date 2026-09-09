"""
Decoder contracts, stage interfaces, request/result data structures, and pipeline context for SIGMA.
Defines canonical typed interfaces for P5.1+ hypothesis decoding.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, Any, Optional, List, Union
import numpy as np

try:
    from backend.hypothesis.candidates import (
        HypothesisCandidate,
        EvidenceStatus,
        EvidenceComponent,
        EvidenceTrace,
    )
except ImportError:
    from hypothesis.candidates import (
        HypothesisCandidate,
        EvidenceStatus,
        EvidenceComponent,
        EvidenceTrace,
    )


class DecoderStageStatus(str, Enum):
    """
    Execution and evaluation status for an individual decoder stage.
    Explicitly distinguishes:
      - SUCCESS: Stage ran and succeeded.
      - FAILED: Stage ran and encountered an unrecoverable decoding/tracking failure.
      - NOT_SUPPORTED: Stage configuration is not supported (e.g. unknown code rate, unsupported modulation).
      - NOT_EVALUATED: Stage was not executed (bypassed or downstream of a prior failure).
    """
    SUCCESS = "success"
    FAILED = "failed"
    NOT_SUPPORTED = "not_supported"
    NOT_EVALUATED = "not_evaluated"

    def to_evidence_status(self) -> EvidenceStatus:
        """Maps decoder stage status to the corresponding hypothesis EvidenceStatus."""
        if self == DecoderStageStatus.SUCCESS:
            return EvidenceStatus.AVAILABLE
        elif self == DecoderStageStatus.FAILED:
            return EvidenceStatus.FAILED
        elif self == DecoderStageStatus.NOT_SUPPORTED:
            return EvidenceStatus.NOT_SUPPORTED
        else:
            return EvidenceStatus.NOT_EVALUATED


@dataclass
class SynchronizationResult:
    """
    Result of carrier and timing synchronization stage.

    Attributes:
        status: DecoderStageStatus indicating stage outcome.
        synchronized_signal: Synchronized IQ array if available.
        carrier_frequency_offset: Estimated carrier frequency offset in Hz.
        carrier_phase_offset: Estimated residual carrier phase offset in radians.
        timing_offset: Estimated fractional or sample timing offset.
        timing_score: Normalized timing/synchronization quality score in [0.0, 1.0].
        metrics: Diagnostic metrics (e.g. residual CFO, timing jitter variance).
        details: Additional human-readable or debug telemetry.
        failure_reason: Explanation of failure if status != SUCCESS.
    """
    status: DecoderStageStatus = DecoderStageStatus.NOT_EVALUATED
    synchronized_signal: Optional[np.ndarray] = None
    carrier_frequency_offset: Optional[float] = None
    carrier_phase_offset: Optional[float] = None
    timing_offset: Optional[float] = None
    timing_score: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Optional[Dict[str, Any]] = None
    failure_reason: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status == DecoderStageStatus.SUCCESS

    @classmethod
    def not_evaluated(cls, details: Optional[Dict[str, Any]] = None, failure_reason: Optional[str] = None) -> "SynchronizationResult":
        return cls(status=DecoderStageStatus.NOT_EVALUATED, details=details, failure_reason=failure_reason)

    @classmethod
    def not_supported(cls, failure_reason: str, details: Optional[Dict[str, Any]] = None) -> "SynchronizationResult":
        return cls(status=DecoderStageStatus.NOT_SUPPORTED, failure_reason=failure_reason, details=details)

    @classmethod
    def failed(cls, failure_reason: str, metrics: Optional[Dict[str, Any]] = None, details: Optional[Dict[str, Any]] = None) -> "SynchronizationResult":
        return cls(status=DecoderStageStatus.FAILED, failure_reason=failure_reason, metrics=metrics or {}, details=details)

    @classmethod
    def create_success(
        cls,
        synchronized_signal: Optional[np.ndarray] = None,
        carrier_frequency_offset: Optional[float] = None,
        carrier_phase_offset: Optional[float] = None,
        timing_offset: Optional[float] = None,
        timing_score: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> "SynchronizationResult":
        return cls(
            status=DecoderStageStatus.SUCCESS,
            synchronized_signal=synchronized_signal,
            carrier_frequency_offset=carrier_frequency_offset,
            carrier_phase_offset=carrier_phase_offset,
            timing_offset=timing_offset,
            timing_score=timing_score,
            metrics=metrics or {},
            details=details,
        )

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "status": self.status.value,
            "success": self.success,
            "has_synchronized_signal": self.synchronized_signal is not None,
        }
        if self.synchronized_signal is not None:
            res["signal_samples"] = int(self.synchronized_signal.shape[-1])
        if self.carrier_frequency_offset is not None:
            res["carrier_frequency_offset"] = float(self.carrier_frequency_offset)
        if self.carrier_phase_offset is not None:
            res["carrier_phase_offset"] = float(self.carrier_phase_offset)
        if self.timing_offset is not None:
            res["timing_offset"] = float(self.timing_offset)
        if self.timing_score is not None:
            res["timing_score"] = float(self.timing_score)
        if self.metrics:
            res["metrics"] = self.metrics
        if self.details:
            res["details"] = self.details
        if self.failure_reason:
            res["failure_reason"] = self.failure_reason
        return res


@dataclass
class DemodulationResult:
    """
    Result of symbol and bit demodulation stage.

    Attributes:
        status: DecoderStageStatus indicating stage outcome.
        symbols: Demodulated constellation symbols (complex or real float array).
        bits: Demodulated hard decision bits (uint8/int array containing 0s and 1s).
        soft_bits: Log-likelihood ratios (LLRs) or soft decision floats if supported.
        constellation_score: Agreement / EVM-based constellation score in [0.0, 1.0].
        quality_metrics: Constellation metrics (e.g. EVM dB, estimated SNR).
        details: Additional diagnostic telemetry.
        failure_reason: Reason for failure if status != SUCCESS.
    """
    status: DecoderStageStatus = DecoderStageStatus.NOT_EVALUATED
    symbols: Optional[np.ndarray] = None
    bits: Optional[np.ndarray] = None
    soft_bits: Optional[np.ndarray] = None
    constellation_score: Optional[float] = None
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    details: Optional[Dict[str, Any]] = None
    failure_reason: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status == DecoderStageStatus.SUCCESS

    @classmethod
    def not_evaluated(cls, details: Optional[Dict[str, Any]] = None, failure_reason: Optional[str] = None) -> "DemodulationResult":
        return cls(status=DecoderStageStatus.NOT_EVALUATED, details=details, failure_reason=failure_reason)

    @classmethod
    def not_supported(cls, failure_reason: str, details: Optional[Dict[str, Any]] = None) -> "DemodulationResult":
        return cls(status=DecoderStageStatus.NOT_SUPPORTED, failure_reason=failure_reason, details=details)

    @classmethod
    def failed(cls, failure_reason: str, quality_metrics: Optional[Dict[str, Any]] = None, details: Optional[Dict[str, Any]] = None) -> "DemodulationResult":
        return cls(status=DecoderStageStatus.FAILED, failure_reason=failure_reason, quality_metrics=quality_metrics or {}, details=details)

    @classmethod
    def create_success(
        cls,
        symbols: Optional[np.ndarray] = None,
        bits: Optional[np.ndarray] = None,
        soft_bits: Optional[np.ndarray] = None,
        constellation_score: Optional[float] = None,
        quality_metrics: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> "DemodulationResult":
        return cls(
            status=DecoderStageStatus.SUCCESS,
            symbols=symbols,
            bits=bits,
            soft_bits=soft_bits,
            constellation_score=constellation_score,
            quality_metrics=quality_metrics or {},
            details=details,
        )

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "status": self.status.value,
            "success": self.success,
            "has_symbols": self.symbols is not None,
            "has_bits": self.bits is not None,
        }
        if self.symbols is not None:
            res["num_symbols"] = int(len(self.symbols))
        if self.bits is not None:
            res["num_bits"] = int(len(self.bits))
        if self.constellation_score is not None:
            res["constellation_score"] = float(self.constellation_score)
        if self.quality_metrics:
            res["quality_metrics"] = self.quality_metrics
        if self.details:
            res["details"] = self.details
        if self.failure_reason:
            res["failure_reason"] = self.failure_reason
        return res


@dataclass
class InterleaverResult:
    """
    Result of deinterleaving stage.

    Attributes:
        status: DecoderStageStatus indicating stage outcome.
        attempted: Whether deinterleaving was attempted.
        supported: Whether the interleaver configuration is supported.
        deinterleaved_bits: Transformed bit array after deinterleaving.
        metrics: Diagnostic information (e.g. block size, depth).
        details: Additional diagnostic telemetry.
        failure_reason: Failure reason if unsuccessful.
    """
    status: DecoderStageStatus = DecoderStageStatus.NOT_EVALUATED
    attempted: bool = False
    supported: bool = True
    deinterleaved_bits: Optional[np.ndarray] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Optional[Dict[str, Any]] = None
    failure_reason: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status == DecoderStageStatus.SUCCESS

    @classmethod
    def not_evaluated(cls, details: Optional[Dict[str, Any]] = None, failure_reason: Optional[str] = None) -> "InterleaverResult":
        return cls(status=DecoderStageStatus.NOT_EVALUATED, attempted=False, supported=True, details=details, failure_reason=failure_reason)

    @classmethod
    def not_supported(cls, failure_reason: str, details: Optional[Dict[str, Any]] = None) -> "InterleaverResult":
        return cls(status=DecoderStageStatus.NOT_SUPPORTED, attempted=True, supported=False, failure_reason=failure_reason, details=details)

    @classmethod
    def failed(cls, failure_reason: str, metrics: Optional[Dict[str, Any]] = None, details: Optional[Dict[str, Any]] = None) -> "InterleaverResult":
        return cls(status=DecoderStageStatus.FAILED, attempted=True, supported=True, failure_reason=failure_reason, metrics=metrics or {}, details=details)

    @classmethod
    def create_success(
        cls,
        deinterleaved_bits: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> "InterleaverResult":
        return cls(
            status=DecoderStageStatus.SUCCESS,
            attempted=True,
            supported=True,
            deinterleaved_bits=deinterleaved_bits,
            metrics=metrics or {},
            details=details,
        )

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "status": self.status.value,
            "success": self.success,
            "attempted": self.attempted,
            "supported": self.supported,
            "has_deinterleaved_bits": self.deinterleaved_bits is not None,
        }
        if self.deinterleaved_bits is not None:
            res["num_bits"] = int(len(self.deinterleaved_bits))
        if self.metrics:
            res["metrics"] = self.metrics
        if self.details:
            res["details"] = self.details
        if self.failure_reason:
            res["failure_reason"] = self.failure_reason
        return res


@dataclass
class FECResult:
    """
    Result of Forward Error Correction decoding stage.

    Attributes:
        status: DecoderStageStatus indicating stage outcome.
        attempted: Whether FEC decoding was attempted.
        supported: Whether the FEC scheme/configuration is supported.
        decoded_bits: Information bits decoded from coded bitstream.
        corrected_bits_count: Number of bit errors corrected by decoder.
        uncorrectable_errors: Flag indicating presence of uncorrectable errors.
        syndrome: Error syndrome vector if applicable.
        error_rate: Estimated post-decode or pre-decode bit error rate.
        fec_score: FEC confidence score in [0.0, 1.0].
        metrics: Detailed metrics (e.g. iterations count, code rate, trellis length).
        details: Additional diagnostic telemetry.
        failure_reason: Failure reason if unsuccessful.
    """
    status: DecoderStageStatus = DecoderStageStatus.NOT_EVALUATED
    attempted: bool = False
    supported: bool = True
    decoded_bits: Optional[np.ndarray] = None
    corrected_bits_count: Optional[int] = None
    uncorrectable_errors: Optional[bool] = None
    syndrome: Optional[np.ndarray] = None
    error_rate: Optional[float] = None
    fec_score: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Optional[Dict[str, Any]] = None
    failure_reason: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status == DecoderStageStatus.SUCCESS

    @classmethod
    def not_evaluated(cls, details: Optional[Dict[str, Any]] = None, failure_reason: Optional[str] = None) -> "FECResult":
        return cls(status=DecoderStageStatus.NOT_EVALUATED, attempted=False, supported=True, details=details, failure_reason=failure_reason)

    @classmethod
    def not_supported(cls, failure_reason: str, details: Optional[Dict[str, Any]] = None) -> "FECResult":
        return cls(status=DecoderStageStatus.NOT_SUPPORTED, attempted=True, supported=False, failure_reason=failure_reason, details=details)

    @classmethod
    def failed(cls, failure_reason: str, metrics: Optional[Dict[str, Any]] = None, details: Optional[Dict[str, Any]] = None) -> "FECResult":
        return cls(status=DecoderStageStatus.FAILED, attempted=True, supported=True, uncorrectable_errors=True, failure_reason=failure_reason, metrics=metrics or {}, details=details)

    @classmethod
    def create_success(
        cls,
        decoded_bits: Optional[np.ndarray] = None,
        corrected_bits_count: Optional[int] = None,
        uncorrectable_errors: bool = False,
        syndrome: Optional[np.ndarray] = None,
        error_rate: Optional[float] = None,
        fec_score: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> "FECResult":
        return cls(
            status=DecoderStageStatus.SUCCESS,
            attempted=True,
            supported=True,
            decoded_bits=decoded_bits,
            corrected_bits_count=corrected_bits_count,
            uncorrectable_errors=uncorrectable_errors,
            syndrome=syndrome,
            error_rate=error_rate,
            fec_score=fec_score,
            metrics=metrics or {},
            details=details,
        )

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "status": self.status.value,
            "success": self.success,
            "attempted": self.attempted,
            "supported": self.supported,
            "has_decoded_bits": self.decoded_bits is not None,
        }
        if self.decoded_bits is not None:
            res["num_decoded_bits"] = int(len(self.decoded_bits))
        if self.corrected_bits_count is not None:
            res["corrected_bits_count"] = int(self.corrected_bits_count)
        if self.uncorrectable_errors is not None:
            res["uncorrectable_errors"] = bool(self.uncorrectable_errors)
        if self.error_rate is not None:
            res["error_rate"] = float(self.error_rate)
        if self.fec_score is not None:
            res["fec_score"] = float(self.fec_score)
        if self.metrics:
            res["metrics"] = self.metrics
        if self.details:
            res["details"] = self.details
        if self.failure_reason:
            res["failure_reason"] = self.failure_reason
        return res


@dataclass
class ValidationResult:
    """
    Result of bitstream and frame validation (CRC, frame framing, sync words).

    Attributes:
        status: DecoderStageStatus indicating stage outcome.
        is_valid: Boolean indicating whether validation checks passed.
        crc_status: CRC check result ("valid", "invalid", "not_present", "not_evaluated").
        syndrome_status: Syndrome check status if applicable.
        frame_status: Framing / header alignment status.
        validation_score: Bitstream/frame validation score in [0.0, 1.0].
        metrics: Detailed metrics (e.g. CRC polynomial, bit error count, frame length).
        details: Additional diagnostic telemetry.
        failure_reason: Reason for validation failure if not valid.
    """
    status: DecoderStageStatus = DecoderStageStatus.NOT_EVALUATED
    is_valid: bool = False
    crc_status: Optional[str] = None
    syndrome_status: Optional[str] = None
    frame_status: Optional[str] = None
    validation_score: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Optional[Dict[str, Any]] = None
    failure_reason: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status == DecoderStageStatus.SUCCESS and self.is_valid

    @classmethod
    def not_evaluated(cls, details: Optional[Dict[str, Any]] = None, failure_reason: Optional[str] = None) -> "ValidationResult":
        return cls(status=DecoderStageStatus.NOT_EVALUATED, is_valid=False, crc_status="not_evaluated", details=details, failure_reason=failure_reason)

    @classmethod
    def not_supported(cls, failure_reason: str, details: Optional[Dict[str, Any]] = None) -> "ValidationResult":
        return cls(status=DecoderStageStatus.NOT_SUPPORTED, is_valid=False, crc_status="not_supported", failure_reason=failure_reason, details=details)

    @classmethod
    def failed(cls, failure_reason: str, crc_status: str = "invalid", metrics: Optional[Dict[str, Any]] = None, details: Optional[Dict[str, Any]] = None) -> "ValidationResult":
        return cls(status=DecoderStageStatus.FAILED, is_valid=False, crc_status=crc_status, failure_reason=failure_reason, metrics=metrics or {}, details=details)

    @classmethod
    def create_success(
        cls,
        is_valid: bool = True,
        crc_status: str = "valid",
        syndrome_status: Optional[str] = None,
        frame_status: Optional[str] = None,
        validation_score: Optional[float] = 1.0,
        metrics: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> "ValidationResult":
        return cls(
            status=DecoderStageStatus.SUCCESS,
            is_valid=is_valid,
            crc_status=crc_status,
            syndrome_status=syndrome_status,
            frame_status=frame_status,
            validation_score=validation_score,
            metrics=metrics or {},
            details=details,
        )

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "status": self.status.value,
            "success": self.success,
            "is_valid": self.is_valid,
        }
        if self.crc_status is not None:
            res["crc_status"] = self.crc_status
        if self.syndrome_status is not None:
            res["syndrome_status"] = self.syndrome_status
        if self.frame_status is not None:
            res["frame_status"] = self.frame_status
        if self.validation_score is not None:
            res["validation_score"] = float(self.validation_score)
        if self.metrics:
            res["metrics"] = self.metrics
        if self.details:
            res["details"] = self.details
        if self.failure_reason:
            res["failure_reason"] = self.failure_reason
        return res


@dataclass
class DecoderConfig:
    """
    Configuration parameters controlling decoder execution stages.
    """
    enable_sync: bool = True
    enable_demod: bool = True
    enable_deinterleaver: bool = True
    enable_fec: bool = True
    enable_validation: bool = True
    max_symbols: Optional[int] = None
    timeout_seconds: Optional[float] = None
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_sync": self.enable_sync,
            "enable_demod": self.enable_demod,
            "enable_deinterleaver": self.enable_deinterleaver,
            "enable_fec": self.enable_fec,
            "enable_validation": self.enable_validation,
            "max_symbols": self.max_symbols,
            "timeout_seconds": self.timeout_seconds,
            "extra_options": dict(self.extra_options),
        }


@dataclass
class DecoderRequest:
    """
    Canonical request for attempting signal decoding under a HypothesisCandidate.

    Attributes:
        iq: Canonical IQ data array. Accepted formats:
            - 2D real array of shape (2, N), where row 0 is I (In-Phase) and row 1 is Q (Quadrature).
            - 1D complex array of shape (N,) complex64/complex128 (I + 1j * Q).
        hypothesis: HypothesisCandidate under test.
        sample_rate: Signal sampling rate in Hz (strictly positive float).
        config: Decoder configuration options.
        context: Optional diagnostic or pipeline context metadata.
    """
    iq: np.ndarray
    hypothesis: HypothesisCandidate
    sample_rate: float
    config: DecoderConfig = field(default_factory=DecoderConfig)
    context: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """
        Validates input fields according to canonical decoder rules:
        - IQ exists and is a numpy.ndarray.
        - IQ contains at least 1 sample.
        - IQ adheres to canonical layouts: 2D real (2, N) or 1D complex (N,).
        - 1D real arrays are rejected as ambiguous (cannot distinguish I vs Q).
        - IQ contains strictly finite numeric values (no NaNs or Infs).
        - sample_rate is strictly positive and finite.
        - hypothesis is a valid HypothesisCandidate.
        - hypothesis.modulation is a non-empty string.
        - hypothesis.symbolRate is strictly positive and finite.
        """
        if self.iq is None:
            raise ValueError("IQ data must not be None")

        if not isinstance(self.iq, np.ndarray):
            raise TypeError(f"IQ data must be a numpy.ndarray, got {type(self.iq)}")

        if self.iq.size == 0:
            raise ValueError("IQ array contains 0 samples")

        # Validate dimensional layout
        if np.iscomplexobj(self.iq):
            if self.iq.ndim != 1:
                raise ValueError(
                    f"Complex IQ array must be 1D with shape (N,), got ndim={self.iq.ndim} and shape={self.iq.shape}"
                )
            if self.iq.shape[0] == 0:
                raise ValueError("Complex IQ array contains 0 samples")
        else:
            if self.iq.ndim == 1:
                raise ValueError(
                    "1D real IQ array layout is ambiguous (cannot distinguish In-Phase and Quadrature components). "
                    "Expected shape (2, N) real or 1D complex array."
                )
            elif self.iq.ndim == 2:
                if self.iq.shape[0] != 2:
                    raise ValueError(
                        f"2D real IQ array must have shape (2, N) where row 0 is I and row 1 is Q. Got shape {self.iq.shape}"
                    )
                if self.iq.shape[1] == 0:
                    raise ValueError("IQ array contains 0 samples")
            else:
                raise ValueError(
                    f"Unsupported multidimensional IQ array shape {self.iq.shape}. Expected (2, N) real or 1D complex."
                )

        # Check for non-finite values (NaN / Inf)
        if not np.all(np.isfinite(self.iq)):
            raise ValueError("IQ array contains non-finite values (NaN or Inf)")

        # Validate sample rate
        if not isinstance(self.sample_rate, (int, float)) or self.sample_rate <= 0 or not math.isfinite(self.sample_rate):
            raise ValueError(f"Sample rate must be a finite positive number, got {self.sample_rate}")

        # Validate hypothesis candidate
        if self.hypothesis is None or not isinstance(self.hypothesis, HypothesisCandidate):
            raise TypeError(f"hypothesis must be a HypothesisCandidate instance, got {type(self.hypothesis)}")

        if not self.hypothesis.modulation or not str(self.hypothesis.modulation).strip():
            raise ValueError("Hypothesis candidate modulation must be a non-empty string")

        if not isinstance(self.hypothesis.symbolRate, (int, float)) or self.hypothesis.symbolRate <= 0 or not math.isfinite(self.hypothesis.symbolRate):
            raise ValueError(f"Hypothesis candidate symbolRate must be strictly positive and finite, got {self.hypothesis.symbolRate}")

    @property
    def num_samples(self) -> int:
        """Returns the number of time samples N."""
        if np.iscomplexobj(self.iq):
            return int(self.iq.shape[0])
        return int(self.iq.shape[1])

    def canonical_iq_2d(self) -> np.ndarray:
        """
        Returns a canonical 2D real float32 array of shape (2, N).
        Row 0 is In-Phase (I), Row 1 is Quadrature (Q).
        Does not mutate the underlying data.
        """
        if np.iscomplexobj(self.iq):
            return np.vstack([self.iq.real, self.iq.imag]).astype(np.float32)
        return self.iq.astype(np.float32, copy=False)

    def canonical_iq_complex(self) -> np.ndarray:
        """
        Returns a canonical 1D complex64 array of shape (N,).
        Values are I + 1j * Q.
        Does not mutate the underlying data.
        """
        if np.iscomplexobj(self.iq):
            return self.iq.astype(np.complex64, copy=False)
        return (self.iq[0] + 1j * self.iq[1]).astype(np.complex64)


@dataclass
class DecoderContext:
    """
    Lightweight pipeline context container shared across decoding stages.
    Enables future stages (synchronization, demodulation, deinterleaving, FEC, validation)
    to pass intermediate representations without sprawling argument lists.
    """
    original_iq: np.ndarray
    sample_rate: float
    modulation: str
    symbol_rate: Optional[float] = None
    hypothesis: Optional[HypothesisCandidate] = None
    synchronized_iq: Optional[np.ndarray] = None
    symbols: Optional[np.ndarray] = None
    raw_bits: Optional[np.ndarray] = None
    deinterleaved_bits: Optional[np.ndarray] = None
    decoded_bits: Optional[np.ndarray] = None
    sync_result: Optional[SynchronizationResult] = None
    demod_result: Optional[DemodulationResult] = None
    interleaver_result: Optional[InterleaverResult] = None
    fec_result: Optional[FECResult] = None
    validation_result: Optional[ValidationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_request(cls, request: DecoderRequest) -> "DecoderContext":
        """Instantiates a pipeline context directly from a canonical DecoderRequest."""
        return cls(
            original_iq=request.iq,
            sample_rate=float(request.sample_rate),
            modulation=str(request.hypothesis.modulation),
            symbol_rate=float(request.hypothesis.symbolRate) if request.hypothesis.symbolRate else None,
            hypothesis=request.hypothesis,
            metadata=dict(request.context) if request.context else {},
        )


@dataclass
class DecoderResult:
    """
    Canonical result of a decoder execution attempt.
    Tracks overall outcome, intermediate stage results, diagnostic metadata,
    and failure semantics without fabricating arbitrary numeric values.
    """
    success: bool
    hypothesis_id: str
    bits: Optional[np.ndarray] = None
    symbols: Optional[np.ndarray] = None
    synchronized_signal: Optional[np.ndarray] = None
    synchronization: SynchronizationResult = field(default_factory=SynchronizationResult.not_evaluated)
    demodulation: DemodulationResult = field(default_factory=DemodulationResult.not_evaluated)
    interleaver: InterleaverResult = field(default_factory=InterleaverResult.not_evaluated)
    fec: FECResult = field(default_factory=FECResult.not_evaluated)
    validation: ValidationResult = field(default_factory=ValidationResult.not_evaluated)
    bitstream_metrics: Optional[Dict[str, Any]] = None
    failure_reason: Optional[str] = None
    diagnostic_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the decoder result into an explicit, JSON-compatible dictionary.
        Does not fabricate values for missing or unattempted stages.
        """
        res: Dict[str, Any] = {
            "success": self.success,
            "hypothesis_id": self.hypothesis_id,
            "has_bits": self.bits is not None,
            "bits_count": int(len(self.bits)) if self.bits is not None else 0,
            "has_symbols": self.symbols is not None,
            "symbols_count": int(len(self.symbols)) if self.symbols is not None else 0,
            "has_synchronized_signal": self.synchronized_signal is not None,
            "synchronization": self.synchronization.to_dict(),
            "demodulation": self.demodulation.to_dict(),
            "interleaver": self.interleaver.to_dict(),
            "fec": self.fec.to_dict(),
            "validation": self.validation.to_dict(),
        }
        if self.bitstream_metrics is not None:
            res["bitstream_metrics"] = self.bitstream_metrics
        if self.failure_reason is not None:
            res["failure_reason"] = self.failure_reason
        if self.diagnostic_details:
            res["diagnostic_details"] = self.diagnostic_details
        return res


class DecoderEngine(ABC):
    """
    Top-level abstract interface for RF signal decoding engines.
    """
    @abstractmethod
    def decode(self, request: DecoderRequest) -> DecoderResult:
        """
        Executes signal decoding on the given canonical request.

        Args:
            request: Validated DecoderRequest instance.

        Returns:
            DecoderResult: Structured outcome across all decoding stages.
        """
        raise NotImplementedError


class ScaffoldDecoderEngine(DecoderEngine):
    """
    Scaffold decoder engine for P5.1 architecture verification.
    Validates the input request and returns an explicit NOT_EVALUATED result
    without fabricating decoding success.
    """
    def decode(self, request: DecoderRequest) -> DecoderResult:
        request.validate()
        return DecoderResult(
            success=False,
            hypothesis_id=request.hypothesis.id,
            synchronization=SynchronizationResult.not_evaluated(
                failure_reason="Scaffold phase P5.1: synchronization stage not yet executed"
            ),
            demodulation=DemodulationResult.not_evaluated(
                failure_reason="Scaffold phase P5.1: demodulation stage not yet executed"
            ),
            interleaver=InterleaverResult.not_evaluated(
                failure_reason="Scaffold phase P5.1: interleaver stage not yet executed"
            ),
            fec=FECResult.not_evaluated(
                failure_reason="Scaffold phase P5.1: FEC stage not yet executed"
            ),
            validation=ValidationResult.not_evaluated(
                failure_reason="Scaffold phase P5.1: validation stage not yet executed"
            ),
            failure_reason="Scaffold decoder engine: full decoding pipeline is implemented in P5.2-P5.7",
            diagnostic_details={
                "engine": "ScaffoldDecoderEngine",
                "phase": "P5.1",
                "modulation": request.hypothesis.modulation,
                "symbol_rate": request.hypothesis.symbolRate,
            },
        )


def decoder_result_to_evidence_trace(
    result: DecoderResult,
    base_trace: Optional[EvidenceTrace] = None,
) -> EvidenceTrace:
    """
    Maps a DecoderResult into an EvidenceTrace, updating diagnostic dimensions:
      - timing: derived from synchronization result
      - constellation: derived from demodulation result
      - fec: derived from FEC result
      - bitstream: derived from validation result

    Preserves pre-existing evidence (e.g. ML classification, symbol rate, SNR)
    from base_trace if supplied.

    Args:
        result: DecoderResult from a decoding execution.
        base_trace: Optional existing EvidenceTrace to preserve.

    Returns:
        EvidenceTrace: Updated evidence trace ready for calculate_raw_score.
    """
    trace = EvidenceTrace()
    if base_trace is not None:
        trace.ml = base_trace.ml
        trace.symbol_rate = base_trace.symbol_rate
        trace.snr = base_trace.snr
        trace.constellation = base_trace.constellation
        trace.timing = base_trace.timing
        trace.fec = base_trace.fec
        trace.bitstream = base_trace.bitstream

    # 1. Timing evidence from synchronization
    sync = result.synchronization
    if sync.status == DecoderStageStatus.SUCCESS:
        score = sync.timing_score if sync.timing_score is not None else 1.0
        trace.timing = EvidenceComponent(
            status=EvidenceStatus.AVAILABLE,
            score=max(0.0, min(1.0, float(score))),
            details=sync.details or sync.metrics or None,
        )
    elif sync.status == DecoderStageStatus.FAILED:
        trace.timing = EvidenceComponent(
            status=EvidenceStatus.FAILED,
            score=0.0,
            details={"failure_reason": sync.failure_reason} if sync.failure_reason else None,
        )
    elif sync.status == DecoderStageStatus.NOT_SUPPORTED:
        trace.timing = EvidenceComponent(
            status=EvidenceStatus.NOT_SUPPORTED,
            details={"failure_reason": sync.failure_reason} if sync.failure_reason else None,
        )
    elif sync.status == DecoderStageStatus.NOT_EVALUATED:
        if base_trace is None or base_trace.timing.status == EvidenceStatus.NOT_EVALUATED:
            trace.timing = EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED)

    # 2. Constellation evidence from demodulation
    demod = result.demodulation
    if demod.status == DecoderStageStatus.SUCCESS:
        score = demod.constellation_score if demod.constellation_score is not None else 1.0
        trace.constellation = EvidenceComponent(
            status=EvidenceStatus.AVAILABLE,
            score=max(0.0, min(1.0, float(score))),
            details=demod.details or demod.quality_metrics or None,
        )
    elif demod.status == DecoderStageStatus.FAILED:
        trace.constellation = EvidenceComponent(
            status=EvidenceStatus.FAILED,
            score=0.0,
            details={"failure_reason": demod.failure_reason} if demod.failure_reason else None,
        )
    elif demod.status == DecoderStageStatus.NOT_SUPPORTED:
        trace.constellation = EvidenceComponent(
            status=EvidenceStatus.NOT_SUPPORTED,
            details={"failure_reason": demod.failure_reason} if demod.failure_reason else None,
        )
    elif demod.status == DecoderStageStatus.NOT_EVALUATED:
        if base_trace is None or base_trace.constellation.status == EvidenceStatus.NOT_EVALUATED:
            trace.constellation = EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED)

    # 3. FEC evidence from fec result
    fec = result.fec
    if fec.status == DecoderStageStatus.SUCCESS:
        if fec.fec_score is not None:
            score = float(fec.fec_score)
        elif fec.error_rate is not None:
            score = max(0.0, 1.0 - float(fec.error_rate))
        else:
            score = 1.0
        trace.fec = EvidenceComponent(
            status=EvidenceStatus.AVAILABLE,
            score=max(0.0, min(1.0, score)),
            details=fec.details or fec.metrics or None,
        )
    elif fec.status == DecoderStageStatus.FAILED:
        trace.fec = EvidenceComponent(
            status=EvidenceStatus.FAILED,
            score=0.0,
            details={"failure_reason": fec.failure_reason} if fec.failure_reason else None,
        )
    elif fec.status == DecoderStageStatus.NOT_SUPPORTED:
        trace.fec = EvidenceComponent(
            status=EvidenceStatus.NOT_SUPPORTED,
            details={"failure_reason": fec.failure_reason} if fec.failure_reason else None,
        )
    elif fec.status == DecoderStageStatus.NOT_EVALUATED:
        if base_trace is None or base_trace.fec.status == EvidenceStatus.NOT_EVALUATED:
            trace.fec = EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED)

    # 4. Bitstream evidence from validation result
    val = result.validation
    if val.status == DecoderStageStatus.SUCCESS:
        score = val.validation_score if val.validation_score is not None else (1.0 if val.is_valid else 0.0)
        trace.bitstream = EvidenceComponent(
            status=EvidenceStatus.AVAILABLE,
            score=max(0.0, min(1.0, float(score))),
            details=val.details or val.metrics or None,
        )
    elif val.status == DecoderStageStatus.FAILED:
        trace.bitstream = EvidenceComponent(
            status=EvidenceStatus.FAILED,
            score=0.0,
            details={"failure_reason": val.failure_reason} if val.failure_reason else None,
        )
    elif val.status == DecoderStageStatus.NOT_SUPPORTED:
        trace.bitstream = EvidenceComponent(
            status=EvidenceStatus.NOT_SUPPORTED,
            details={"failure_reason": val.failure_reason} if val.failure_reason else None,
        )
    elif val.status == DecoderStageStatus.NOT_EVALUATED:
        if base_trace is None or base_trace.bitstream.status == EvidenceStatus.NOT_EVALUATED:
            trace.bitstream = EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED)

    # 5. Interleaver evidence from interleaver result (provenance-only, weight 0.0 in P5.5)
    intl = result.interleaver
    if intl.status == DecoderStageStatus.SUCCESS:
        trace.interleaver = EvidenceComponent(
            status=EvidenceStatus.AVAILABLE,
            score=None,
            details=intl.details or intl.metrics or {"success": True},
        )
    elif intl.status == DecoderStageStatus.FAILED:
        trace.interleaver = EvidenceComponent(
            status=EvidenceStatus.FAILED,
            score=0.0,
            details=intl.details or intl.metrics or ({"failure_reason": intl.failure_reason} if intl.failure_reason else None),
        )
    elif intl.status == DecoderStageStatus.NOT_SUPPORTED:
        trace.interleaver = EvidenceComponent(
            status=EvidenceStatus.NOT_SUPPORTED,
            details={"failure_reason": intl.failure_reason} if intl.failure_reason else None,
        )
    elif intl.status == DecoderStageStatus.NOT_EVALUATED:
        if base_trace is None or getattr(base_trace, "interleaver", None) is None or base_trace.interleaver.status == EvidenceStatus.NOT_EVALUATED:
            trace.interleaver = EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED)

    return trace


def update_hypothesis_from_decoder_result(
    candidate: HypothesisCandidate,
    result: DecoderResult,
) -> HypothesisCandidate:
    """
    Updates a HypothesisCandidate with the evidence and outcome from a DecoderResult.

    Updates:
      - candidate.evidence: Enriched with timing, constellation, FEC, and bitstream components.
      - candidate.status: Set to 'success' if result.success is True; 'failed' if not success
        and a failure reason was recorded.
      - candidate.details: Appends failure reason or success confirmation if appropriate.

    Args:
        candidate: The candidate hypothesis to update.
        result: The DecoderResult containing stage metrics.

    Returns:
        HypothesisCandidate: The updated candidate ready for scoring and ranking.
    """
    candidate.evidence = decoder_result_to_evidence_trace(result, candidate.evidence)
    if result.success:
        candidate.status = "success"
    elif result.failure_reason and candidate.status == "pending":
        candidate.status = "failed"
    return candidate
