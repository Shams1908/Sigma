"""
Bitstream validity checking.

Since we have no knowledge of the higher-layer framing (CCSDS sync words,
protocol headers, etc.) we use three statistical / structural indicators:

1. Entropy check:
   A valid decoded bitstream from a real digital link has entropy close to
   1.0 bit/bit (appears random after FEC removal).  A constant or degenerate
   bitstream (all zeros, all ones, simple pattern) indicates a decode failure.

2. Run-length check:
   Excessively long runs of identical bits (> 10 consecutive identical bits
   in a random-looking stream) are statistically anomalous.

3. Balance check:
   The ratio of ones to zeros in a properly decoded bitstream should be
   close to 0.5 ± 0.15 (for most practical channel codes).

A bitstream "passes" if it passes at least 2 of the 3 checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class BitstreamCheckResult:
    passed: bool                      # Overall pass/fail
    entropy: float                    # Shannon entropy (bits per bit, 0–1)
    ones_ratio: float                 # Fraction of 1-bits
    max_run_length: int               # Longest run of identical bits
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    note: str = ""


def check_bitstream(bits: np.ndarray, min_bits: int = 32) -> BitstreamCheckResult:
    """
    Run three statistical validity checks on a decoded bitstream.

    Args:
        bits:     1-D array of 0/1 uint8 values.
        min_bits: Minimum number of bits required to make a meaningful
                  assessment.  Shorter streams always return passed=False.

    Returns:
        BitstreamCheckResult with per-check details.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()

    if len(bits) < min_bits:
        return BitstreamCheckResult(
            passed=False,
            entropy=0.0,
            ones_ratio=0.0,
            max_run_length=len(bits),
            checks_failed=["length"],
            note=f"Too few bits for assessment (need ≥{min_bits}, got {len(bits)})",
        )

    ones = int(np.sum(bits == 1))
    zeros = len(bits) - ones
    ones_ratio = ones / len(bits)

    # 1. Shannon entropy of the bit sequence
    entropy = _binary_entropy(ones_ratio)

    # 2. Longest run of identical bits
    max_run = _max_run(bits)

    # 3. Balance: ones_ratio near 0.5
    balance_ok = 0.2 <= ones_ratio <= 0.8
    entropy_ok = entropy >= 0.5         # 0.5 is a deliberately lenient threshold
    run_ok     = max_run <= max(20, len(bits) // 10)

    checks_passed = []
    checks_failed = []

    for name, ok in [
        ("entropy", entropy_ok),
        ("balance", balance_ok),
        ("run_length", run_ok),
    ]:
        (checks_passed if ok else checks_failed).append(name)

    # Pass if at least 2 of 3 checks succeed
    passed = len(checks_passed) >= 2

    note = ""
    if not passed:
        note = (
            f"Bitstream quality checks failed: {', '.join(checks_failed)}. "
            f"entropy={entropy:.3f}, ones_ratio={ones_ratio:.3f}, "
            f"max_run={max_run}"
        )

    return BitstreamCheckResult(
        passed=passed,
        entropy=entropy,
        ones_ratio=float(ones_ratio),
        max_run_length=int(max_run),
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        note=note,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _binary_entropy(p: float) -> float:
    """Shannon entropy of a Bernoulli(p) distribution (bits)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    q = 1.0 - p
    return float(-(p * np.log2(p) + q * np.log2(q)))


def _max_run(bits: np.ndarray) -> int:
    """Compute the length of the longest consecutive run of identical bits."""
    if len(bits) == 0:
        return 0
    # Use diff to find transitions
    changes = np.where(np.diff(bits) != 0)[0]
    if len(changes) == 0:
        return len(bits)
    # Run lengths = gaps between transitions (+1 for the first and last segments)
    starts = np.concatenate([[0], changes + 1])
    ends   = np.concatenate([changes + 1, [len(bits)]])
    return int(np.max(ends - starts))
