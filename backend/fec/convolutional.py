"""
Rate-1/2, Constraint-length-7 convolutional encoder and Viterbi decoder.

Polynomials (octal): G1 = 0o171 = 0b1111001
                     G2 = 0o133 = 0b1011011

These are the standard CCSDS/NASA-131.0-B telemetry FEC polynomials and
match the generators used in the synthetic signal generator to the extent
those signals include convolutional coding.

Encoder:
  - Rate 1/2: each input bit produces 2 output bits (G1 output, G2 output).
  - Constraint length K=7 → 64 states (register length = K-1 = 6).
  - Tail-biting is NOT applied; the shift register flushes 6 zero bits.

Viterbi decoder:
  - Hard-decision Viterbi (Hamming distances).
  - Accepts a 1-D array of 0/1 bits (demodulated hard decisions).
  - Returns the maximum-likelihood decoded bit sequence.
  - Path memory length: 5*K = 35 steps (standard heuristic).
"""
from __future__ import annotations

import numpy as np

# ── Encoder parameters ────────────────────────────────────────────────────────
K = 7                        # Constraint length
MEMORY = K - 1               # Shift register depth
NUM_STATES = 2 ** MEMORY     # 64 states
G1 = 0b1111001               # Generator 1 (octal 171)
G2 = 0b1011011               # Generator 2 (octal 133)


def _parity(x: int) -> int:
    """Return the XOR parity of all set bits in x (popcount mod 2)."""
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1


# ── Pre-compute transition table ──────────────────────────────────────────────
# next_state[state][input_bit]  → next state
# output[state][input_bit]      → (out1, out2) as integer 0–3
_NEXT_STATE = np.zeros((NUM_STATES, 2), dtype=np.int32)
_OUTPUT     = np.zeros((NUM_STATES, 2), dtype=np.int32)

for _s in range(NUM_STATES):
    for _b in range(2):
        # Shift register: new MSB = input bit, shift right
        _reg = (_b << MEMORY) | (_s >> 1)       # full K-bit register
        _ns  = (_b << (MEMORY - 1)) | (_s >> 1) # next state (MEMORY bits)
        _o1  = _parity(_reg & G1)
        _o2  = _parity(_reg & G2)
        _NEXT_STATE[_s][_b] = _ns
        _OUTPUT[_s][_b]     = (_o1 << 1) | _o2


# ── Encoder ───────────────────────────────────────────────────────────────────

def encode(bits: np.ndarray) -> np.ndarray:
    """
    Convolutional encode a bit sequence using the rate-1/2 K=7 code.

    Args:
        bits: 1-D array of 0/1 integers (information bits).

    Returns:
        Encoded bit array of length 2*(len(bits) + MEMORY).
        The extra 2*MEMORY bits are the tail flush (zero padding).
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    # Append MEMORY zero bits to flush the encoder back to state 0
    padded = np.concatenate([bits, np.zeros(MEMORY, dtype=np.uint8)])

    state = 0
    encoded: list[int] = []
    for b in padded:
        out_pair = int(_OUTPUT[state][int(b)])
        encoded.append((out_pair >> 1) & 1)
        encoded.append(out_pair & 1)
        state = int(_NEXT_STATE[state][int(b)])

    return np.array(encoded, dtype=np.uint8)


# ── Viterbi decoder ───────────────────────────────────────────────────────────

# Path memory: 5 * K is a reliable traceback depth for K=7
_TRACEBACK_DEPTH = 5 * K

# Pre-compute inverse transition table for traceback:
# prev_state[ns][b] = s  such that  NEXT_STATE[s][b] == ns
# (unique because the trellis is regular: each (ns, b) pair has exactly one parent s)
_PREV_STATE = np.full((NUM_STATES, 2), -1, dtype=np.int32)
for _s in range(NUM_STATES):
    for _b in range(2):
        _ns = int(_NEXT_STATE[_s][_b])
        _PREV_STATE[_ns][_b] = _s


def decode(received: np.ndarray) -> np.ndarray:
    """
    Hard-decision Viterbi decoder for the rate-1/2 K=7 code.

    Args:
        received: 1-D array of hard-decision bits (0 or 1).
                  Length should be a multiple of 2 (rate-1/2 output).
                  If odd-length, the last bit is discarded.

    Returns:
        Decoded bit array.  Length = len(received)//2 - MEMORY
        (removes the tail flush bits added by the encoder).
    """
    received = np.asarray(received, dtype=np.uint8).ravel()
    if len(received) % 2 != 0:
        received = received[:-1]

    n_steps = len(received) // 2  # number of trellis steps

    INF = 10 ** 9
    metric = np.full(NUM_STATES, INF, dtype=np.int64)
    metric[0] = 0  # known initial state

    # Traceback tables — store both the winning input bit and the previous state
    survivor_input = np.zeros((n_steps, NUM_STATES), dtype=np.uint8)
    survivor_prev  = np.zeros((n_steps, NUM_STATES), dtype=np.int32)

    for t in range(n_steps):
        r1 = int(received[2 * t])
        r2 = int(received[2 * t + 1])
        new_metric = np.full(NUM_STATES, INF, dtype=np.int64)

        for s in range(NUM_STATES):
            if metric[s] == INF:
                continue
            for b in range(2):
                ns = int(_NEXT_STATE[s][b])
                out = int(_OUTPUT[s][b])
                o1 = (out >> 1) & 1
                o2 = out & 1
                branch = int(o1 != r1) + int(o2 != r2)
                candidate = metric[s] + branch
                if candidate < new_metric[ns]:
                    new_metric[ns] = candidate
                    survivor_input[t][ns] = b
                    survivor_prev[t][ns]  = s

        metric = new_metric

    # Traceback: start from state with best final metric
    best_state = int(np.argmin(metric))
    decoded = np.zeros(n_steps, dtype=np.uint8)

    state = best_state
    for t in range(n_steps - 1, -1, -1):
        decoded[t] = survivor_input[t][state]
        state      = int(survivor_prev[t][state])

    # Remove the tail-flush (last MEMORY bits are the zero-padding added by encode())
    return decoded[: max(0, n_steps - MEMORY)]


# ── Bit error rate helper ─────────────────────────────────────────────────────

def bit_error_rate(original: np.ndarray, decoded: np.ndarray) -> float:
    """
    Compute the bit error rate between two equal-length bit arrays.
    If they differ in length, the shorter one sets the comparison window.
    """
    original = np.asarray(original, dtype=np.uint8).ravel()
    decoded  = np.asarray(decoded,  dtype=np.uint8).ravel()
    n = min(len(original), len(decoded))
    if n == 0:
        return 1.0
    return float(np.sum(original[:n] != decoded[:n])) / n
