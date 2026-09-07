"""Stable synchronization contracts consumed by future P3 pipeline stages."""

from typing import Protocol

import numpy as np


class CarrierRecovery(Protocol):
    """Correct carrier frequency and phase from complex baseband samples."""

    def recover(self, samples: np.ndarray, *, sample_rate: float) -> np.ndarray:
        ...


class TimingRecovery(Protocol):
    """Convert oversampled samples into one complex sample per symbol."""

    def recover(self, samples: np.ndarray, *, samples_per_symbol: float) -> np.ndarray:
        ...


class FrameSynchronizer(Protocol):
    """Locate a frame or symbol-aligned region in a sample stream."""

    def synchronize(self, samples: np.ndarray) -> tuple[np.ndarray, int]:
        """Return synchronized samples and their offset in the input stream."""
        ...