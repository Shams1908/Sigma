"""Understandable, lightweight synchronization for synthetic P3 signals."""

import numpy as np

from .models import CarrierSyncResult, FrameSyncResult, TimingSyncResult


def _complex_1d(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.complex128)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(result.real)) or not np.all(np.isfinite(result.imag)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _samples_per_symbol(value: int) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError("samples_per_symbol must be a positive integer")
    return int(value)


class BasicFrameSynchronizer:
    """Find a known oversampled preamble by normalized correlation."""

    def __init__(self, *, minimum_correlation: float = 0.50) -> None:
        if not 0.0 < minimum_correlation <= 1.0:
            raise ValueError("minimum_correlation must be between 0 and 1")
        self.minimum_correlation = float(minimum_correlation)

    def synchronize(
        self,
        samples: np.ndarray,
        *,
        preamble_symbols: np.ndarray,
        samples_per_symbol: int,
    ) -> FrameSyncResult:
        values = _complex_1d(samples, "samples")
        preamble = _complex_1d(preamble_symbols, "preamble_symbols")
        sps = _samples_per_symbol(samples_per_symbol)
        template = np.repeat(preamble, sps)
        if values.size < template.size:
            return FrameSyncResult(False, None, 0.0, "signal is shorter than the preamble")

        template_energy = float(np.vdot(template, template).real)
        sample_energy = np.convolve(np.abs(values) ** 2, np.ones(template.size), mode="valid")
        correlations = np.correlate(values, template, mode="valid")
        denominator = np.sqrt(template_energy * np.maximum(sample_energy, np.finfo(float).eps))
        scores = np.abs(correlations) / denominator
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score < self.minimum_correlation:
            return FrameSyncResult(False, None, score, "no reliable preamble found")
        return FrameSyncResult(True, best, score, "preamble detected")


class BasicTimingSynchronizer:
    """Sample one point near the center of each fixed-length symbol."""

    def recover(
        self,
        samples: np.ndarray,
        *,
        start: int,
        samples_per_symbol: int,
        symbol_count: int | None = None,
        sample_offset: int | None = None,
    ) -> TimingSyncResult:
        values = _complex_1d(samples, "samples")
        sps = _samples_per_symbol(samples_per_symbol)
        if isinstance(start, bool) or int(start) != start or start < 0:
            raise ValueError("start must be a non-negative integer")
        start = int(start)
        offset = sps // 2 if sample_offset is None else sample_offset
        if isinstance(offset, bool) or int(offset) != offset or not 0 <= offset < sps:
            raise ValueError("sample_offset must be an integer within one symbol")
        offset = int(offset)
        if start >= values.size:
            raise ValueError("start is outside the input samples")
        available = (values.size - start - offset - 1) // sps + 1
        count = available if symbol_count is None else symbol_count
        if isinstance(count, bool) or int(count) != count or count < 1:
            raise ValueError("symbol_count must be a positive integer")
        count = int(count)
        if count > available:
            raise ValueError("symbol_count exceeds available samples")
        indices = start + offset + np.arange(count) * sps
        return TimingSyncResult(values[indices], start, sps, offset)


class BasicCarrierSynchronizer:
    """Estimate a constant phase from known pilot symbols and correct it."""

    def recover(
        self,
        symbols: np.ndarray,
        *,
        reference_symbols: np.ndarray,
    ) -> CarrierSyncResult:
        received = _complex_1d(symbols, "symbols")
        reference = _complex_1d(reference_symbols, "reference_symbols")
        if received.size != reference.size:
            raise ValueError("symbols and reference_symbols must have the same length")
        cross_product = np.vdot(reference, received)
        if abs(cross_product) <= np.finfo(float).eps:
            raise ValueError("cannot estimate phase from zero-energy reference")
        phase = float(np.angle(cross_product))
        corrected = received * np.exp(-1j * phase)
        return CarrierSyncResult(corrected, phase, "constant phase corrected")