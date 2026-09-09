"""
8PSK demodulator.

Canonical mapping (exactly inverts the ML EightPSKModulator):

  bits (b0,b1,b2)  idx = b0*4+b1*2+b2   phase_index   angle        symbol
  000              0                     0             0            +1 + 0j
  001              1                     1             π/4          +√2/2 + j√2/2
  010              2                     3             3π/4         -√2/2 + j√2/2
  011              3                     2             2π/4 = π/2   0 + j
  100              4                     7             7π/4         +√2/2 - j√2/2
  101              5                     6             6π/4 = 3π/2  0 - j
  110              6                     4             π            -1 + 0j
  111              7                     5             5π/4         -√2/2 - j√2/2

Decision rule: nearest constellation point (minimum Euclidean distance).
The 8 points are equally spaced on the unit circle; each decision region is
a sector of π/4 radians centred on the corresponding symbol.  The boundaries
between sectors are at angles k·π/4 ± π/8.

EVM is computed as for the existing BPSK/QPSK demodulators (shared helper).
"""
from __future__ import annotations

import numpy as np

from demodulation.psk_demod import DemodResult, _rms_evm


# ── Constellation table ───────────────────────────────────────────────────────
# Indexed by (b0*4 + b1*2 + b2) exactly as in the ML generator.

_8PSK_BITS_PER_SYMBOL = 3

# phase_indices[idx] is the phase step k so that symbol = exp(j * k * π/4)
_PHASE_INDICES = np.array([0, 1, 3, 2, 7, 6, 4, 5], dtype=np.int32)

# Pre-compute the 8 constellation points indexed by bit-group index 0..7
_8PSK_ANGLES = _PHASE_INDICES * (np.pi / 4.0)
_8PSK_CONST = (np.cos(_8PSK_ANGLES) + 1j * np.sin(_8PSK_ANGLES)).astype(np.complex64)

# Inverse: angle → bit-group index (for hard-decision lookup)
# _IDX_FROM_PHASE[k] = bit-group index whose phase_index == k
_IDX_FROM_PHASE = np.zeros(8, dtype=np.int32)
for _bi, _pi in enumerate(_PHASE_INDICES):
    _IDX_FROM_PHASE[_pi] = _bi

# Pre-compute the 3-bit words for each bit-group index
_8PSK_BITS_TABLE = np.array(
    [[int(b) for b in f"{i:03b}"] for i in range(8)],
    dtype=np.uint8,
)  # shape [8, 3]; row i → [b0, b1, b2] for bit-group index i


# ── Demodulator ───────────────────────────────────────────────────────────────

def demod_8psk(symbols: np.ndarray) -> DemodResult:
    """
    Hard-decision 8PSK demodulator.

    Canonical mapping exactly inverts the ML EightPSKModulator:
      nearest constellation point → 3 bits (b0, b1, b2)
      where b0,b1,b2 ∈ {0,1} and the bit-group index is b0*4 + b1*2 + b2.

    The nearest point is found by computing the angle of each received
    symbol, rounding to the nearest π/4 sector, and looking up the
    corresponding bits.

    Args:
        symbols: 1-D complex array (one sample per symbol, synchronised).

    Returns:
        DemodResult with decoded bits (3n bits for n symbols), decisions, EVM.
    """
    symbols = np.asarray(symbols, dtype=np.complex64)
    n = len(symbols)

    if n == 0:
        return DemodResult(
            bits=np.array([], dtype=np.uint8),
            symbols=symbols,
            decisions=np.array([], dtype=np.complex64),
            evm_rms=0.0,
            modulation="8PSK",
        )

    # For each received symbol, find the nearest of the 8 constellation points.
    # The 8 points are on the unit circle; nearest-point is the one whose
    # angle is closest.  Use vectorized distance to all 8 points.
    # Shape: [n, 8]
    dists = np.abs(
        symbols[:, np.newaxis] - _8PSK_CONST[np.newaxis, :]
    )
    nearest_idx = np.argmin(dists, axis=1)  # bit-group index 0..7

    # Decisions (constellation points)
    decisions = _8PSK_CONST[nearest_idx]

    # Bits: look up [b0, b1, b2] for each symbol and flatten
    bits = _8PSK_BITS_TABLE[nearest_idx].ravel()

    evm = _rms_evm(symbols, decisions)

    return DemodResult(
        bits=bits,
        symbols=symbols,
        decisions=decisions,
        evm_rms=evm,
        modulation="8PSK",
    )
