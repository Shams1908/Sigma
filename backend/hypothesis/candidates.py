"""
Hypothesis candidate generation.

Given:
  - ML top-k predictions from ml.inference.pipeline.analyze_file()
  - DSP parameter estimates (symbol_rate, bandwidth, SNR)

Produces a short list of concrete, testable HypothesisSpec objects that the
evaluator will attempt to validate through sync → demod → FEC stages.

Generation rules
────────────────
1. Take the ML top-k predictions (up to MAX_ML_CANDIDATES = 5).
2. Only include modulations we can actually demodulate (BPSK, QPSK).
   Other ML predictions are carried forward as "ML-only" candidates with
   demod_pass pre-set to False.
3. For each qualifying modulation, generate symbol rate variants:
   - Primary: DSP-estimated symbol rate (as-is)
   - Secondary: ×0.5 and ×2 variants if the primary is uncertain
     (uncertainty is flagged when the DSP estimate is outside the plausible
     bandwidth range inferred from the PSD — see notes below)
4. FEC: always try both "none" and "convolutional" for BPSK/QPSK.
5. Deduplicate (same modulation + symbol_rate within 5 % tolerance + fec).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Supported modulations (those with working demodulators) ───────────────────
DEMODULATABLE = {"BPSK", "QPSK"}

# Maximum ML candidates to expand
MAX_ML_CANDIDATES = 5

# Symbol-rate variants to try when estimate uncertainty is high
_SR_VARIANTS = [1.0, 0.5, 2.0]  # multipliers on estimated symbol rate

# Deduplication tolerance (5 %)
_SR_TOL = 0.05


@dataclass
class HypothesisSpec:
    """One testable hypothesis — inputs to the evaluator."""

    modulation: str                    # e.g. "BPSK", "QPSK"
    symbol_rate: float                 # symbols/second
    fec_type: Optional[str]            # "convolutional" | None
    ml_confidence: float               # raw ML probability for this modulation
    source: str = "ml+dsp"            # "ml+dsp" | "ml_only"
    can_demodulate: bool = True        # False → skip sync/demod/fec stages


def generate_candidates(
    top_predictions: List[Tuple[str, float]],
    symbol_rate_estimate: float,
    bandwidth_estimate: float,
    snr_estimate: float,
    max_candidates: int = 10,
) -> List[HypothesisSpec]:
    """
    Generate a ranked list of testable HypothesisSpec objects.

    Args:
        top_predictions:      ML top-k list of (modulation_name, probability).
        symbol_rate_estimate: DSP symbol-rate estimate (symbols/s).
        bandwidth_estimate:   DSP bandwidth estimate (Hz) — used for plausibility.
        snr_estimate:         DSP SNR estimate (dB).
        max_candidates:       Cap on total candidates returned.

    Returns:
        List[HypothesisSpec] sorted by ml_confidence descending.
    """
    candidates: List[HypothesisSpec] = []

    # Determine if symbol rate estimate is plausible relative to bandwidth.
    # A symbol rate > bandwidth is physically impossible (Nyquist), so we
    # flag high uncertainty and try rate variants.
    sr_uncertain = (
        symbol_rate_estimate <= 0
        or (bandwidth_estimate > 0 and symbol_rate_estimate > bandwidth_estimate * 1.5)
    )

    seen: List[Tuple[str, float, Optional[str]]] = []  # (mod, sr, fec)

    def _is_dup(mod: str, sr: float, fec: Optional[str]) -> bool:
        for sm, ss, sf in seen:
            if sm == mod and sf == fec and abs(sr - ss) / max(ss, 1.0) < _SR_TOL:
                return True
        return False

    for ml_name, ml_prob in top_predictions[:MAX_ML_CANDIDATES]:
        # Normalise modulation name — ML labels may differ slightly
        mod = _normalise_mod_name(ml_name)
        can_demod = mod in DEMODULATABLE

        sr_multipliers = _SR_VARIANTS if sr_uncertain else [1.0]
        fec_options: List[Optional[str]] = [None, "convolutional"] if can_demod else [None]

        for mult in sr_multipliers:
            sr = max(1.0, symbol_rate_estimate * mult)
            for fec in fec_options:
                if _is_dup(mod, sr, fec):
                    continue
                seen.append((mod, sr, fec))
                candidates.append(
                    HypothesisSpec(
                        modulation=mod,
                        symbol_rate=sr,
                        fec_type=fec,
                        ml_confidence=float(ml_prob),
                        source="ml+dsp" if can_demod else "ml_only",
                        can_demodulate=can_demod,
                    )
                )

    # Sort by ML confidence descending; cap total
    candidates.sort(key=lambda c: c.ml_confidence, reverse=True)
    return candidates[:max_candidates]


# ── Helpers ───────────────────────────────────────────────────────────────────

_MOD_ALIASES: dict[str, str] = {
    # ML label variants → canonical name
    "BPSK":    "BPSK",
    "QPSK":    "QPSK",
    "8PSK":    "8PSK",
    "16QAM":   "16QAM",
    "16-QAM":  "16QAM",
    "64QAM":   "64QAM",
    "64-QAM":  "64QAM",
    "AM-DSB":  "AM-DSB",
    "AM-SSB":  "AM-SSB",
    "FM":      "FM",
    "WBFM":    "WBFM",
    "AM-DSB-SC": "AM-DSB",
}


def _normalise_mod_name(name: str) -> str:
    """Map ML label strings to canonical modulation names."""
    upper = name.upper().replace(" ", "").replace("_", "-")
    return _MOD_ALIASES.get(upper, name.upper())
