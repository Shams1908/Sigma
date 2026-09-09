"""
Abstract interface for Forward Error Correction (FEC) decoders in SIGMA.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np

try:
    from backend.hypothesis.decoder_contracts import FECResult
except ImportError:
    from hypothesis.decoder_contracts import FECResult


class FECDecoder(ABC):
    """
    Abstract base class for modular FEC decoders.
    """

    @abstractmethod
    def decode(
        self,
        bits: np.ndarray,
        soft_bits: Optional[np.ndarray] = None,
    ) -> FECResult:
        """
        Decodes the received (channel) bits into information bits.

        Args:
            bits: 1D uint8 array of received channel bits (0 or 1).
            soft_bits: Optional 1D float array of soft metrics or LLRs.

        Returns:
            FECResult: Canonical stage result containing decoded bits,
                       corrected error counts, and diagnostic telemetry.
        """
        raise NotImplementedError
