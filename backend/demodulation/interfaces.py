"""Protocols shared by P3 demodulation implementations."""

from typing import Protocol

import numpy as np

from .models import DemodulationResult


class Demodulator(Protocol):
    """Interface implemented by symbol-to-bit demodulators."""

    modulation: str

    def demodulate(
        self,
        symbols: np.ndarray,
        *,
        noise_variance: float | None = None,
    ) -> DemodulationResult:
        ...