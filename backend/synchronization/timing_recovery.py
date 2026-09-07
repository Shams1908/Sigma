"""
Symbol timing recovery — Gardner Timing Error Detector (TED).

The Gardner TED is a non-data-aided, decision-directed algorithm that works
at 2 samples per symbol.  It estimates the timing error ε as:

    e[k] = Re{ (x[k] - x[k-2]) * conj(x[k-1]) }

where x[k-1] is the halfway sample and x[k-2], x[k] are the strobe samples.

A second-order loop filter (PI controller) drives a Farrow-style linear
interpolator to produce one output sample per symbol.

Reference:
  Gardner, F.M. (1986). "A BPSK/QPSK timing-error detector for sampled
  receivers." IEEE Trans. Commun., 34(5), 423–429.
"""
from __future__ import annotations

import numpy as np


def gardner_timing_recovery(
    iq: np.ndarray,
    samples_per_symbol: int = 2,
    loop_bw: float = 0.01,
    damping: float = 0.707,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gardner timing error detector with a second-order loop filter.

    For best results the input should already be matched-filtered and
    have exactly `samples_per_symbol` samples per symbol (commonly 2).

    Args:
        iq:                  1-D complex IQ array at `sps` samples/symbol.
        samples_per_symbol:  Oversampling factor (Gardner requires ≥ 2).
        loop_bw:             Normalised loop bandwidth (0.001–0.05).
        damping:             Damping factor (0.707 = critical).

    Returns:
        symbols:    Complex symbol array (one sample per symbol).
        timing_err: Per-symbol timing error estimate.
    """
    iq = np.asarray(iq, dtype=np.complex128)
    sps = max(2, int(samples_per_symbol))

    # Second-order loop gains
    omega_n = loop_bw * 2.0 * np.pi
    K1 = 2.0 * damping * omega_n
    K2 = omega_n ** 2

    symbols: list[complex] = []
    timing_errors: list[float] = []

    mu = 0.0          # fractional timing offset (0 ≤ μ < 1)
    integrator = 0.0  # integral accumulator
    k = 0             # current sample index

    while k + sps < len(iq):
        # Strobe sample (current symbol) via linear interpolation
        k_frac = k + mu * sps
        k0 = int(np.floor(k_frac))
        k1 = min(k0 + 1, len(iq) - 1)
        alpha = k_frac - k0
        strobe = (1.0 - alpha) * iq[k0] + alpha * iq[k1]

        # Halfway sample (midpoint between current and previous strobe)
        k_mid = k_frac - sps / 2.0
        km0 = int(np.floor(k_mid))
        km1 = min(km0 + 1, len(iq) - 1)
        beta = k_mid - km0
        mid = (1.0 - beta) * iq[max(km0, 0)] + beta * iq[max(km1, 0)]

        # Previous strobe (for Gardner TED)
        k_prev = k_frac - sps
        kp0 = int(np.floor(k_prev))
        kp1 = min(kp0 + 1, len(iq) - 1)
        gamma = k_prev - kp0
        prev = (1.0 - gamma) * iq[max(kp0, 0)] + gamma * iq[max(kp1, 0)]

        # Gardner timing error estimate
        ted = ((strobe - prev) * np.conj(mid)).real
        timing_errors.append(float(ted))
        symbols.append(strobe)

        # PI loop filter
        integrator += K2 * ted
        delta_mu = K1 * ted + integrator

        # Update fractional offset and advance integer sample pointer
        mu += delta_mu
        advance = sps + int(np.floor(mu))
        mu -= np.floor(mu)
        k += max(1, advance)

    return (
        np.array(symbols, dtype=np.complex64),
        np.array(timing_errors, dtype=np.float64),
    )


def coarse_timing_offset(iq: np.ndarray, samples_per_symbol: int) -> int:
    """
    Estimate the best starting sample offset (0 … sps-1) for downsampling
    by maximising the total energy at that phase — a simple non-data-aided
    heuristic that works when the timing loop is not used.

    Returns:
        Integer offset in [0, samples_per_symbol).
    """
    sps = int(samples_per_symbol)
    energies = np.array(
        [np.sum(np.abs(iq[offset::sps]) ** 2) for offset in range(sps)]
    )
    return int(np.argmax(energies))
