"""Run the lightweight P3 Phase 2 synchronization demonstration."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.synchronization import Phase2Decoder


def bpsk_symbols(bits: np.ndarray) -> np.ndarray:
    return (1.0 - 2.0 * bits).astype(np.complex128)


def main() -> None:
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    preamble = bpsk_symbols(np.array([0, 1, 0, 1], dtype=np.uint8))
    payload = bpsk_symbols(bits)
    phase = 0.40
    samples_per_symbol = 4
    frame = np.concatenate((preamble, payload)) * np.exp(1j * phase)
    samples = np.concatenate((np.zeros(6), np.repeat(frame, samples_per_symbol)))

    sync_result, decoded = Phase2Decoder().decode(
        samples,
        preamble_symbols=preamble,
        samples_per_symbol=samples_per_symbol,
        payload_symbol_count=payload.size,
        modulation="BPSK",
    )
    ber = decoded.ber(bits)
    passed = ber == 0.0

    print(f"detected frame start: {sync_result.frame.start}")
    print(f"estimated phase: {sync_result.carrier.estimated_phase:.4f} rad")
    print(f"recovered bits: {decoded.bit_count}")
    print(f"BER: {ber:.4f}")
    print(f"status: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()