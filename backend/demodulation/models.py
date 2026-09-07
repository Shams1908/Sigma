"""Shared result models for digital demodulators."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DemodulationResult:
    """Hard and optional soft decisions produced by a demodulator.

    Soft decisions are LLR-like values: positive values favor bit 0 and
    negative values favor bit 1. Values are ordered in the same flattened
    bit order as ``bits``.
    """

    modulation: str
    bits: np.ndarray
    symbols: np.ndarray
    soft_bits: np.ndarray | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bits = np.asarray(self.bits)
        symbols = np.asarray(self.symbols)
        if bits.ndim != 1 or not np.all((bits == 0) | (bits == 1)):
            raise ValueError("bits must be a one-dimensional binary array")
        if symbols.ndim != 1:
            raise ValueError("symbols must be a one-dimensional array")
        if self.soft_bits is not None:
            soft_bits = np.asarray(self.soft_bits, dtype=float)
            if soft_bits.shape != bits.shape:
                raise ValueError("soft_bits must have the same shape as bits")
            if not np.all(np.isfinite(soft_bits)):
                raise ValueError("soft_bits must contain only finite values")
            object.__setattr__(self, "soft_bits", soft_bits)
        object.__setattr__(self, "bits", bits.astype(np.uint8, copy=False))
        object.__setattr__(self, "symbols", symbols)

    @property
    def bit_count(self) -> int:
        return int(self.bits.size)

    def ber(self, expected_bits: np.ndarray) -> float:
        """Return BER against a same-length known bit sequence."""

        expected = np.asarray(expected_bits)
        if expected.ndim != 1 or expected.shape != self.bits.shape:
            raise ValueError("expected_bits must have the same one-dimensional shape as bits")
        if not np.all((expected == 0) | (expected == 1)):
            raise ValueError("expected_bits must be binary")
        return float(np.mean(self.bits != expected))