"""
Data structures, evidence models, and candidate definitions for hypothesis tracking and ranking.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Tuple


class EvidenceStatus(str, Enum):
    """
    Provenance and evaluation status for an individual evidence dimension.
    """
    AVAILABLE = "available"
    NOT_EVALUATED = "not_evaluated"
    FAILED = "failed"
    NOT_SUPPORTED = "not_supported"


@dataclass
class EvidenceComponent:
    """
    Represents evidence from a single diagnostic dimension (ML, Constellation, Timing, FEC, Bitstream).
    """
    status: EvidenceStatus = EvidenceStatus.NOT_EVALUATED
    score: Optional[float] = None
    weight: float = 0.0
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize component to an explicit, unambiguous dictionary.
        Does not fabricate arbitrary numeric values for unavailable dimensions.
        """
        result: Dict[str, Any] = {"status": self.status.value}
        if self.status == EvidenceStatus.AVAILABLE and self.score is not None:
            result["score"] = float(self.score)
        elif self.status == EvidenceStatus.FAILED:
            result["score"] = 0.0
        
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class EvidenceTrace:
    """
    Traceability container tracking all defined evidence dimensions:
      - ml: ML modulation classification confidence
      - symbol_rate: Symbol rate parameter agreement
      - snr: SNR evidence / feasibility
      - constellation: Constellation agreement
      - timing: Timing/synchronization quality
      - fec: FEC validation
      - bitstream: Bitstream correlation
    """
    ml: EvidenceComponent = field(default_factory=lambda: EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED))
    symbol_rate: EvidenceComponent = field(default_factory=lambda: EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED))
    snr: EvidenceComponent = field(default_factory=lambda: EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED))
    constellation: EvidenceComponent = field(default_factory=lambda: EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED))
    timing: EvidenceComponent = field(default_factory=lambda: EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED))
    fec: EvidenceComponent = field(default_factory=lambda: EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED))
    bitstream: EvidenceComponent = field(default_factory=lambda: EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED))

    @property
    def mlModulation(self) -> EvidenceComponent:
        return self.ml

    @mlModulation.setter
    def mlModulation(self, val: EvidenceComponent):
        self.ml = val

    @property
    def symbolRate(self) -> EvidenceComponent:
        return self.symbol_rate

    @symbolRate.setter
    def symbolRate(self, val: EvidenceComponent):
        self.symbol_rate = val

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ml": self.ml.to_dict(),
            "mlModulation": self.ml.to_dict(),
            "symbolRate": self.symbol_rate.to_dict(),
            "symbol_rate": self.symbol_rate.to_dict(),
            "snr": self.snr.to_dict(),
            "constellation": self.constellation.to_dict(),
            "timing": self.timing.to_dict(),
            "fec": self.fec.to_dict(),
            "bitstream": self.bitstream.to_dict(),
        }


@dataclass
class HypothesisCandidate:
    """
    A signal decoding hypothesis candidate.
    
    Attributes:
        id: Unique identifier for this hypothesis instance.
        modulation: Modulation scheme (e.g. 'BPSK', 'QPSK', '16QAM').
        symbolRate: Estimated or tested symbol rate in Baud.
        fec_config: Optional forward error correction configuration (e.g. 'conv_r1/2_k7', 'none').
        interleaver_config: Optional interleaver configuration (e.g. 'block_32x32', 'none').
        sync_assumptions: Optional synchronization/carrier recovery parameters.
        rawScore: Unnormalized weighted evidence score calculated from available evidence.
        confidenceScore: Normalized relative confidence across the current candidate set.
                         (Not a calibrated statistical posterior probability).
        evidence: EvidenceTrace tracking the individual diagnostic components.
        details: Human-readable explanation of why the hypothesis scored well.
        status: Status flag ('pending' | 'success' | 'failed').
    """
    id: str
    modulation: str
    symbolRate: float
    fec_config: Optional[str] = None
    interleaver_config: Optional[str] = None
    sync_assumptions: Optional[Dict[str, Any]] = None
    rawScore: float = 0.0
    confidenceScore: float = 0.0
    evidence: EvidenceTrace = field(default_factory=EvidenceTrace)
    details: str = ""
    status: str = "pending"

    def identity_key(self) -> Tuple[Any, ...]:
        """
        Canonical hashable identity key for deduplicating genuinely identical candidate configurations.
        Distinguishes parameter configurations (modulation, symbol rate, FEC, interleaver, sync_assumptions).
        Different symbol rates (e.g. 9600 vs 4800) or distinct synchronization assumptions produce distinct keys.
        """
        def _freeze(val: Any) -> Any:
            if isinstance(val, dict):
                return tuple(sorted((k, _freeze(v)) for k, v in val.items()))
            elif isinstance(val, (list, tuple)):
                return tuple(_freeze(v) for v in val)
            return str(val)

        sync_key = (
            tuple(sorted((k, _freeze(v)) for k, v in self.sync_assumptions.items()))
            if self.sync_assumptions
            else ()
        )
        return (
            self.modulation.strip().upper(),
            round(float(self.symbolRate), 2),
            (self.fec_config or "").strip().lower(),
            (self.interleaver_config or "").strip().lower(),
            sync_key,
        )


    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize candidate to dictionary matching frontend and API specifications.
        """
        return {
            "id": self.id,
            "modulation": self.modulation,
            "symbolRate": float(self.symbolRate),
            "rawScore": float(self.rawScore),
            "confidenceScore": float(self.confidenceScore),
            "details": self.details,
            "status": self.status,
            "fec_config": self.fec_config,
            "interleaver_config": self.interleaver_config,
            "sync_assumptions": self.sync_assumptions,
            "evidence": self.evidence.to_dict(),
        }
