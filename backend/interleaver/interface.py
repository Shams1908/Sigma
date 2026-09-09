"""
Abstract interface for Deinterleavers in SIGMA.
"""
from abc import ABC, abstractmethod
import numpy as np

try:
    from backend.hypothesis.decoder_contracts import InterleaverResult
except ImportError:
    from hypothesis.decoder_contracts import InterleaverResult


class Deinterleaver(ABC):
    """
    Abstract base class for modular deinterleaving algorithms.
    """

    @abstractmethod
    def deinterleave(self, bits: np.ndarray) -> InterleaverResult:
        """
        Reverses interleaving transformation on the input bitstream.

        Args:
            bits: 1D uint8 array of interleaved channel bits.

        Returns:
            InterleaverResult: Deinterleaved bits and telemetry.
        """
        raise NotImplementedError
