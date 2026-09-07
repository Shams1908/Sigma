"""Phase 1 P3 demodulation tests using synthetic symbols and AWGN."""

import numpy as np
import pytest

from backend.demodulation import BPSKDemodulator, QPSKDemodulator


def add_awgn(
    symbols: np.ndarray,
    noise_variance: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise_scale = np.sqrt(noise_variance / 2.0)
    noise = noise_scale * (
        rng.standard_normal(symbols.size) + 1j * rng.standard_normal(symbols.size)
    )
    return symbols + noise


def bpsk_symbols(bits: np.ndarray) -> np.ndarray:
    return (1.0 - 2.0 * bits).astype(np.complex128)


def qpsk_symbols(bits: np.ndarray) -> np.ndarray:
    pairs = bits.reshape(-1, 2)
    mapping = {
        (0, 0): (1.0 + 1.0j),
        (0, 1): (-1.0 + 1.0j),
        (1, 1): (-1.0 - 1.0j),
        (1, 0): (1.0 - 1.0j),
    }
    return np.array([mapping[tuple(pair)] for pair in pairs], dtype=np.complex128) / np.sqrt(2.0)


def test_bpsk_demodulates_known_bits_and_exposes_soft_decisions() -> None:
    transmitted = np.array([0, 1, 1, 0, 0, 1], dtype=np.uint8)
    result = BPSKDemodulator().demodulate(bpsk_symbols(transmitted), noise_variance=0.25)

    assert np.array_equal(result.bits, transmitted)
    assert result.soft_bits is not None
    assert np.all(result.soft_bits[transmitted == 0] > 0)
    assert np.all(result.soft_bits[transmitted == 1] < 0)
    assert result.ber(transmitted) == 0.0


def test_qpsk_demodulates_gray_mapped_bits() -> None:
    transmitted = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
    result = QPSKDemodulator().demodulate(qpsk_symbols(transmitted))

    assert np.array_equal(result.bits, transmitted)
    assert result.ber(transmitted) == 0.0


def test_bpsk_awgn_has_low_ber() -> None:
    transmitted = np.random.default_rng(7).integers(0, 2, size=20_000, dtype=np.uint8)
    received = add_awgn(bpsk_symbols(transmitted), noise_variance=0.20, seed=8)
    result = BPSKDemodulator().demodulate(received, noise_variance=0.20)

    assert result.ber(transmitted) < 0.02


def test_qpsk_awgn_has_low_ber() -> None:
    transmitted = np.random.default_rng(9).integers(0, 2, size=20_000, dtype=np.uint8)
    received = add_awgn(qpsk_symbols(transmitted), noise_variance=0.20, seed=10)
    result = QPSKDemodulator().demodulate(received, noise_variance=0.20)

    assert result.ber(transmitted) < 0.02


@pytest.mark.parametrize("demodulator", [BPSKDemodulator(), QPSKDemodulator()])
def test_demodulators_reject_invalid_inputs(demodulator: object) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        demodulator.demodulate(np.array([]))  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="positive"):
        demodulator.demodulate(np.array([1.0]), noise_variance=0.0)  # type: ignore[attr-defined]