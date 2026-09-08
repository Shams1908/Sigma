"""Composition of the basic frame, timing, and carrier synchronizers."""

import numpy as np

from backend.demodulation.models import DemodulationResult
from backend.demodulation.psk_demod import BPSKDemodulator, QPSKDemodulator

from .basic import BasicCarrierSynchronizer, BasicFrameSynchronizer, BasicTimingSynchronizer
from .models import Phase2SyncResult


class Phase2Decoder:
    """Run basic synchronization followed by an existing PSK demodulator."""

    def __init__(self) -> None:
        self.frame_sync = BasicFrameSynchronizer()
        self.timing_sync = BasicTimingSynchronizer()
        self.carrier_sync = BasicCarrierSynchronizer()

    def decode(
        self,
        samples: np.ndarray,
        *,
        preamble_symbols: np.ndarray,
        samples_per_symbol: int,
        payload_symbol_count: int,
        modulation: str,
    ) -> tuple[Phase2SyncResult, DemodulationResult]:
        frame = self.frame_sync.synchronize(
            samples,
            preamble_symbols=preamble_symbols,
            samples_per_symbol=samples_per_symbol,
        )
        if not frame.detected or frame.start is None:
            raise ValueError(frame.status)
        total_symbols = len(preamble_symbols) + payload_symbol_count
        timing = self.timing_sync.recover(
            samples,
            start=frame.start,
            samples_per_symbol=samples_per_symbol,
            symbol_count=total_symbols,
        )
        carrier = self.carrier_sync.recover(
            timing.symbols[: len(preamble_symbols)],
            reference_symbols=preamble_symbols,
        )
        corrected = np.empty_like(timing.symbols)
        corrected[:] = timing.symbols * np.exp(-1j * carrier.estimated_phase)
        payload = corrected[len(preamble_symbols) :]
        if modulation == "BPSK":
            result = BPSKDemodulator().demodulate(payload)
        elif modulation == "QPSK":
            result = QPSKDemodulator().demodulate(payload)
        else:
            raise ValueError("modulation must be BPSK or QPSK")
        return Phase2SyncResult(frame, timing, carrier), result