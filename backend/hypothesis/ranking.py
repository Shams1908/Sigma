"""
Hypothesis ranking.

Combines ML confidence with per-stage validation evidence into a single
final_score ∈ [0, 1] and assigns integer ranks (1 = best).

Scoring formula (heuristic starting point — weights are NOT tuned against
labelled data and should be revised once ground-truth results are available):

  final_score = (
      W_ML   * ml_confidence            # raw CNN probability
    + W_SYNC * sync_bonus               # 0 or BONUS_SYNC
    + W_DEMOD * demod_bonus             # 0 or BONUS_DEMOD  (scaled by 1-EVM)
    + W_FEC  * fec_bonus                # 0 or BONUS_FEC
    + W_BITS * bitstream_bonus          # 0 or BONUS_BITS   (scaled by entropy)
  )

Weights (must sum to 1.0 across all terms at their maximum):
  W_ML    = 0.40   — CNN is the primary discriminator
  W_SYNC  = 0.15   — successful synchronisation is strong evidence
  W_DEMOD = 0.20   — low EVM confirms the modulation choice
  W_FEC   = 0.10   — FEC decoding passing adds confidence
  W_BITS  = 0.15   — valid bitstream is the final validation gate

Rationale:
  - ML confidence carries the most weight because the CNN was trained on a
    large labelled corpus and generalises well.
  - Demodulation EVM is the most direct objective measure of modulation match.
  - FEC and bitstream checks are noisier at low SNR so they carry less weight.

These weights are intentionally conservative starting points.  As labelled
test results accumulate, they should be optimised (e.g. grid-search or
Bayesian optimisation on a held-out validation set).
"""
from __future__ import annotations

from typing import List

from hypothesis.evaluator import EvaluatedHypothesis

# ── Scoring weights ───────────────────────────────────────────────────────────
W_ML    = 0.40
W_SYNC  = 0.15
W_DEMOD = 0.20
W_FEC   = 0.10
W_BITS  = 0.15


def score_hypothesis(ev: EvaluatedHypothesis) -> float:
    """
    Compute final_score ∈ [0, 1] for one evaluated hypothesis.

    Each stage contributes its weight only when it passes;
    demod and bitstream contributions are additionally scaled by
    a continuous quality metric (1-EVM and entropy respectively)
    so small differences are preserved in the ranking.
    """
    score = W_ML * ev.spec.ml_confidence

    if ev.sync_pass:
        score += W_SYNC * 1.0

    if ev.demod_pass:
        # Scale by (1 - EVM) so lower EVM → higher contribution
        evm_factor = max(0.0, 1.0 - ev.evm_rms)
        score += W_DEMOD * evm_factor

    if ev.fec_pass:
        score += W_FEC * 1.0

    if ev.bitstream_pass:
        # Scale by Shannon entropy of the decoded bits (0–1)
        entropy_factor = min(1.0, ev.bit_entropy)
        score += W_BITS * entropy_factor

    return float(min(1.0, max(0.0, score)))


def rank_hypotheses(
    evaluated: List[EvaluatedHypothesis],
) -> List[dict]:
    """
    Score, sort, and assign integer ranks to evaluated hypotheses.

    Args:
        evaluated: List of EvaluatedHypothesis objects.

    Returns:
        List of dicts with all Hypothesis document fields, sorted by
        final_score descending (rank 1 = best).
    """
    scored: List[tuple[float, EvaluatedHypothesis]] = []

    for ev in evaluated:
        fs = score_hypothesis(ev)
        scored.append((fs, ev))

    # Sort descending by final score
    scored.sort(key=lambda x: x[0], reverse=True)

    results: List[dict] = []
    for rank, (final_score, ev) in enumerate(scored, start=1):
        results.append(
            {
                "modulation":            ev.spec.modulation,
                "symbol_rate":           ev.spec.symbol_rate,
                "fec_type":              ev.spec.fec_type,
                "ml_confidence":         ev.spec.ml_confidence,
                "calibrated_confidence": None,   # placeholder for post-hoc calibration
                "sync_pass":             ev.sync_pass,
                "demod_pass":            ev.demod_pass,
                "fec_pass":              ev.fec_pass,
                "bitstream_pass":        ev.bitstream_pass,
                "final_score":           round(final_score, 6),
                "rank":                  rank,
                # Extra evidence fields (stored but not in the DB Hypothesis model)
                "_evm_rms":              ev.evm_rms,
                "_bit_entropy":          ev.bit_entropy,
                "_timing_error_rms":     ev.timing_error_rms,
                "_error_notes":          ev.error_notes,
            }
        )

    return results
