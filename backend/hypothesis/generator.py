"""
Hypothesis generator — orchestrates the full per-file analysis pipeline.

This is the single function the background analysis task calls.
It wires together:
  1. ml.input.pipeline.process_file()     — load IQ
  2. ml.inference.pipeline.analyze_file() — CNN classification + top-k
  3. dsp.extract_parameters()             — SNR, CFO, BW, symbol rate
  4. hypothesis.candidates.generate_candidates() — candidate list
  5. hypothesis.evaluator.evaluate_hypothesis()  — per-stage validation
  6. hypothesis.ranking.rank_hypotheses()         — final score + rank

Returns a GeneratorResult dataclass with all artefacts needed to populate
the Analysis document.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeneratorResult:
    """All artefacts produced by the hypothesis generator for one file."""

    # DSP estimates (keyed to ParameterEstimate field names)
    snr: float
    carrier_offset: float
    bandwidth: float
    symbol_rate_estimate: float

    # Ranked hypothesis dicts (keyed to Hypothesis document field names)
    ranked_hypotheses: List[dict] = field(default_factory=list)

    # Whether any hypothesis fully validated
    any_validated: bool = False

    # Diagnostic
    ml_top_prediction: str = ""
    ml_top_confidence: float = 0.0
    num_candidates_evaluated: int = 0
    error_notes: List[str] = field(default_factory=list)


def run_hypothesis_pipeline(
    storage_path: str,
    model_path: str = "models/m5_iq_cnn.pt",
    max_candidates: int = 10,
) -> GeneratorResult:
    """
    Full hypothesis pipeline for one signal file.

    Args:
        storage_path:   Absolute path to the signal file on disk.
        model_path:     Path to the CNN model checkpoint.
        max_candidates: Maximum number of hypotheses to evaluate.

    Returns:
        GeneratorResult with DSP parameters + ranked hypotheses.
        Never raises — all errors are caught and stored in error_notes.
    """
    # ── Step 1: Load IQ via ml.input.pipeline ────────────────────────────────
    try:
        from ml.input.pipeline import process_file  # type: ignore[import]
        from ml.input.types import PipelineConfig    # type: ignore[import]

        config = PipelineConfig()
        segments, input_meta = process_file(storage_path, config)

        if input_meta.validation_status == "ERROR" or len(segments) == 0:
            return _fail_result(
                f"Input pipeline error: {input_meta.error_message or 'no segments'}"
            )

        # Reconstruct full [2, N] IQ from segments for DSP (use all samples)
        # segments shape: [M, 2, L] — stack along the last axis
        iq_full = segments.reshape(2, -1)  # [2, M*L]
        sample_rate = input_meta.sample_rate or 1.0

    except Exception as exc:  # noqa: BLE001
        return _fail_result(f"IQ loading failed: {exc}")

    # ── Step 2: DSP parameter extraction ─────────────────────────────────────
    try:
        from dsp import extract_parameters  # noqa: PLC0415

        params = extract_parameters(iq_full, sample_rate)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DSP extraction partial failure: %s", exc)
        params = {
            "snr": 0.0,
            "carrier_offset": 0.0,
            "bandwidth": sample_rate / 2.0,
            "symbol_rate_estimate": sample_rate / 4.0,
        }

    snr             = float(params.get("snr", 0.0))
    carrier_offset  = float(params.get("carrier_offset", 0.0))
    bandwidth       = float(params.get("bandwidth", sample_rate / 2.0))
    symbol_rate_est = float(params.get("symbol_rate_estimate", sample_rate / 4.0))

    # ── Step 3: ML classification ─────────────────────────────────────────────
    try:
        from ml.inference.pipeline import analyze_file  # type: ignore[import]

        ml_result = analyze_file(storage_path, model_path=model_path)
        top_predictions = ml_result.top_predictions   # List[(name, prob)]
        ml_top_name     = ml_result.predicted_class_name
        ml_top_conf     = ml_result.confidence

    except Exception as exc:  # noqa: BLE001
        logger.warning("ML inference failed (%s); using DSP-only candidates.", exc)
        # Fall back to two reasonable guesses with equal probability
        top_predictions = [("BPSK", 0.5), ("QPSK", 0.5)]
        ml_top_name     = "UNKNOWN"
        ml_top_conf     = 0.0

    # ── Step 4: Generate candidates ───────────────────────────────────────────
    from hypothesis.candidates import generate_candidates  # noqa: PLC0415

    candidates = generate_candidates(
        top_predictions=top_predictions,
        symbol_rate_estimate=symbol_rate_est,
        bandwidth_estimate=bandwidth,
        snr_estimate=snr,
        max_candidates=max_candidates,
    )
    logger.info("Generated %d hypothesis candidates for %s", len(candidates), storage_path)

    # ── Step 5: Evaluate each candidate ──────────────────────────────────────
    from hypothesis.evaluator import evaluate_hypothesis  # noqa: PLC0415

    evaluated = []
    for spec in candidates:
        try:
            ev = evaluate_hypothesis(iq_full, sample_rate, spec)
            evaluated.append(ev)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Evaluator crashed for %s: %s", spec.modulation, exc)

    # ── Step 6: Rank ──────────────────────────────────────────────────────────
    from hypothesis.ranking import rank_hypotheses  # noqa: PLC0415

    ranked = rank_hypotheses(evaluated)

    any_validated = any(
        h.get("sync_pass") and h.get("demod_pass") for h in ranked
    )

    return GeneratorResult(
        snr=snr,
        carrier_offset=carrier_offset,
        bandwidth=bandwidth,
        symbol_rate_estimate=symbol_rate_est,
        ranked_hypotheses=ranked,
        any_validated=any_validated,
        ml_top_prediction=ml_top_name,
        ml_top_confidence=ml_top_conf,
        num_candidates_evaluated=len(evaluated),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fail_result(msg: str) -> GeneratorResult:
    """Return a minimal GeneratorResult that represents a total failure."""
    return GeneratorResult(
        snr=0.0,
        carrier_offset=0.0,
        bandwidth=0.0,
        symbol_rate_estimate=0.0,
        ranked_hypotheses=[],
        any_validated=False,
        error_notes=[msg],
    )
