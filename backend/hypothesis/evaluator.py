"""
Hypothesis evaluator.

Runs each HypothesisSpec through the full signal processing chain:
  Stage 1 — Synchronisation (matched filter + carrier recovery + timing)
  Stage 2 — Demodulation   (BPSK / QPSK hard-decision)
  Stage 3 — FEC decoding   (Viterbi, optional)
  Stage 4 — Bitstream check (entropy / balance / run-length)

Each stage is attempted independently; a stage failure sets its pass flag
to False but does not abort the remaining stages (partial evidence is still
useful for ranking).

Returns an EvaluatedHypothesis with pass/fail per stage and numeric evidence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dsp.fft import canonical_to_complex
from hypothesis.candidates import HypothesisSpec

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class EvaluatedHypothesis:
    spec: HypothesisSpec

    # Stage pass/fail
    sync_pass: bool = False
    demod_pass: bool = False
    fec_pass: bool = False
    bitstream_pass: bool = False

    # Numeric evidence
    evm_rms: float = 1.0            # Lower is better; 0 = perfect
    timing_error_rms: float = 1.0   # RMS Gardner timing error
    bit_entropy: float = 0.0        # Decoded bitstream entropy (0–1)
    ones_ratio: float = 0.5
    fec_applied: bool = False       # Whether FEC decoding was attempted

    error_notes: list[str] = field(default_factory=list)


# ── Main evaluator ────────────────────────────────────────────────────────────

def evaluate_hypothesis(
    iq_2d: np.ndarray,
    sample_rate: float,
    spec: HypothesisSpec,
) -> EvaluatedHypothesis:
    """
    Evaluate a single hypothesis against the raw IQ data.

    Args:
        iq_2d:       Shape [2, N] canonical IQ array (I row 0, Q row 1).
        sample_rate: Sample rate in Hz.
        spec:        HypothesisSpec to evaluate.

    Returns:
        EvaluatedHypothesis with per-stage results.
    """
    result = EvaluatedHypothesis(spec=spec)

    # Modulations we cannot demodulate → mark all stages False immediately
    if not spec.can_demodulate:
        result.error_notes.append(
            f"{spec.modulation} demodulation not implemented; skipping stages."
        )
        return result

    iq = canonical_to_complex(iq_2d)
    mod_order = 2 if spec.modulation == "BPSK" else 4

    # ── Stage 1: Synchronisation ─────────────────────────────────────────────
    try:
        from synchronization import synchronize  # noqa: PLC0415

        symbols, sync_info = synchronize(
            iq,
            sample_rate=sample_rate,
            symbol_rate=spec.symbol_rate,
            modulation_order=mod_order,
        )
        result.sync_pass = len(symbols) >= 8
        result.timing_error_rms = sync_info.get("timing_error_rms", 1.0)

        if not result.sync_pass:
            result.error_notes.append(
                f"Sync produced only {len(symbols)} symbols (need ≥ 8)."
            )
    except Exception as exc:  # noqa: BLE001
        result.sync_pass = False
        result.error_notes.append(f"Sync failed: {exc}")
        return result  # Cannot continue without symbols

    # ── Stage 2: Demodulation ─────────────────────────────────────────────────
    try:
        from demodulation import demodulate  # noqa: PLC0415

        demod_result = demodulate(symbols, spec.modulation)
        result.evm_rms = demod_result.evm_rms
        # Pass demod if EVM is below a reasonable threshold (< 0.5 = 50 %)
        result.demod_pass = demod_result.evm_rms < 0.5
        bits_raw = demod_result.bits

        if not result.demod_pass:
            result.error_notes.append(
                f"Demod EVM too high: {demod_result.evm_rms:.3f} ≥ 0.5."
            )
    except Exception as exc:  # noqa: BLE001
        result.demod_pass = False
        result.error_notes.append(f"Demod failed: {exc}")
        bits_raw = np.array([], dtype=np.uint8)

    # ── Stage 3: FEC decoding (optional) ─────────────────────────────────────
    if spec.fec_type == "convolutional" and len(bits_raw) >= 14:
        try:
            from fec.convolutional import decode  # noqa: PLC0415

            decoded_bits = decode(bits_raw)
            result.fec_applied = True
            result.fec_pass = len(decoded_bits) > 0
            bits_for_check = decoded_bits

            if not result.fec_pass:
                result.error_notes.append("Viterbi decoder produced empty output.")
        except Exception as exc:  # noqa: BLE001
            result.fec_pass = False
            result.fec_applied = True
            result.error_notes.append(f"FEC decode failed: {exc}")
            bits_for_check = bits_raw
    else:
        result.fec_pass = True   # No FEC expected → vacuously true
        bits_for_check = bits_raw

    # ── Stage 4: Bitstream check ──────────────────────────────────────────────
    if len(bits_for_check) >= 32:
        try:
            from fec.bitstream_check import check_bitstream  # noqa: PLC0415

            bs_result = check_bitstream(bits_for_check)
            result.bitstream_pass = bs_result.passed
            result.bit_entropy    = bs_result.entropy
            result.ones_ratio     = bs_result.ones_ratio

            if not bs_result.passed:
                result.error_notes.append(
                    f"Bitstream check failed: {bs_result.note}"
                )
        except Exception as exc:  # noqa: BLE001
            result.bitstream_pass = False
            result.error_notes.append(f"Bitstream check error: {exc}")
    else:
        # Not enough bits to make a meaningful check — give benefit of doubt
        result.bitstream_pass = result.demod_pass
        result.error_notes.append(
            f"Only {len(bits_for_check)} bits available; "
            "bitstream check skipped (inherits demod_pass)."
        )

    return result
