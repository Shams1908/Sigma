"""Small result models for the basic P3 synchronization stages."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameSyncResult:
    detected: bool
    start: int | None
    correlation: float
    status: str


@dataclass(frozen=True)
class TimingSyncResult:
    symbols: np.ndarray
    start: int
    samples_per_symbol: int
    sample_offset: int


@dataclass(frozen=True)
class CarrierSyncResult:
    symbols: np.ndarray
    estimated_phase: float
    status: str


@dataclass(frozen=True)
class Phase2SyncResult:
    frame: FrameSyncResult
    timing: TimingSyncResult
    carrier: CarrierSyncResult