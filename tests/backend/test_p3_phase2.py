"""Focused tests for the lightweight Phase 2 synchronization path."""

import numpy as np
import pytest

from backend.demodulation import BPSKDemodulator, QPSKDemodulator
from backend.synchronization import (
    BasicCarrierSynchronizer,
    BasicFrameSynchronizer,
    BasicTimingSynchronizer,
    Phase2Decoder,
)


def bpsk_symbols(bits: np.ndarray) -> np.ndarray:
    return (1.0 - 2.0 * bits).astype(np.complex128)


def qpsk_symbols(bits: np.ndarray) -> np.ndarray:
    mapping = {
        (0, 0): 1.0 + 1.0j,
        (0, 1): -1.0 + 1.0j,
        (1, 1): -1.0 - 1.0j,
        (1, 0): 1.0 - 1.0j,
    }
    pairs = bits.reshape(-1, 2)
    return np.array([mapping[tuple(pair)] for pair in pairs], dtype=np.complex128) / np.sqrt(2.0)


def oversample(symbols: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    return np.repeat(symbols, samples_per_symbol)


def test_constant_phase_correction_reports_phase_and_restores_symbols() -> None:
    reference = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.complex128)
    phase = 0.37
    received = reference * np.exp(1j * phase)

    result = BasicCarrierSynchronizer().recover(received, reference_symbols=reference)

    assert result.estimated_phase == pytest.approx(phase)
    assert np.allclose(result.symbols, reference)


def test_timing_synchronizer_selects_symbol_centers() -> None:
    symbols = np.array([1.0, -1.0, 1.0], dtype=np.complex128)
    samples = oversample(symbols, 4)

    result = BasicTimingSynchronizer().recover(samples, start=0, samples_per_symbol=4)

    assert np.array_equal(result.symbols, symbols)
    assert result.sample_offset == 2


def test_frame_synchronizer_detects_preamble_start() -> None:
    preamble = np.array([1.0, -1.0, 1.0, 1.0], dtype=np.complex128)
    payload = np.array([-1.0, 1.0], dtype=np.complex128)
    samples = np.concatenate((np.zeros(5), oversample(np.concatenate((preamble, payload)), 3)))

    result = BasicFrameSynchronizer().synchronize(
        samples,
        preamble_symbols=preamble,
        samples_per_symbol=3,
    )

    assert result.detected
    assert result.start == 5
    assert result.correlation == pytest.approx(1.0)


def test_frame_synchronizer_reports_missing_preamble() -> None:
    result = BasicFrameSynchronizer(minimum_correlation=0.9).synchronize(
        np.ones(30, dtype=np.complex128),
        preamble_symbols=np.array([1.0, -1.0, 1.0]),
        samples_per_symbol=2,
    )

    assert not result.detected
    assert result.start is None
    assert "no reliable" in result.status


@pytest.mark.parametrize(
    ("modulation", "bits", "modulator"),
    [
        ("BPSK", np.array([0, 1, 1, 0, 1, 0], dtype=np.uint8), bpsk_symbols),
        ("QPSK", np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8), qpsk_symbols),
    ],
)
def test_phase2_pipeline_recovers_payload_bits(
    modulation: str,
    bits: np.ndarray,
    modulator: object,
) -> None:
    preamble_bits = np.array([0, 1, 0, 1], dtype=np.uint8)
    preamble = bpsk_symbols(preamble_bits)
    payload = modulator(bits)  # type: ignore[operator]
    phase = 0.43
    symbols = np.concatenate((preamble, payload)) * np.exp(1j * phase)
    samples = np.concatenate((np.zeros(7), oversample(symbols, 5)))

    sync_result, result = Phase2Decoder().decode(
        samples,
        preamble_symbols=preamble,
        samples_per_symbol=5,
        payload_symbol_count=payload.size,
        modulation=modulation,
    )

    assert sync_result.frame.start == 7
    assert sync_result.carrier.estimated_phase == pytest.approx(phase)
    assert np.array_equal(result.bits, bits)
    assert result.ber(bits) == 0.0


def test_phase2_rejects_unreliable_frame() -> None:
    with pytest.raises(ValueError, match="no reliable preamble"):
        Phase2Decoder().decode(
            np.zeros(100, dtype=np.complex128),
            preamble_symbols=np.ones(4, dtype=np.complex128),
            samples_per_symbol=4,
            payload_symbol_count=2,
            modulation="BPSK",
        )


def test_synchronizers_validate_inputs() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        BasicTimingSynchronizer().recover(
            np.ones(4, dtype=np.complex128), start=0, samples_per_symbol=0
        )
    with pytest.raises(ValueError, match="same length"):
        BasicCarrierSynchronizer().recover(
            np.ones(2, dtype=np.complex128),
            reference_symbols=np.ones(3, dtype=np.complex128),
        )