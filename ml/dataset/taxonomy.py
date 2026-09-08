"""
Taxonomy alignment between the SIGMA 11-class RadioML taxonomy and the
7-class external benchmark taxonomy.

PURPOSE:
    Evaluation-time helpers for mapping predictions between taxonomies and
    identifying classes that have no equivalent in the other taxonomy.

CRITICAL — OPEN-SET BOUNDARY:
    OpenSetClassifier provides EVALUATION-TIME taxonomy handling only.
    It requires ground-truth labels to determine whether a sample belongs
    to a class unsupported by the SIGMA 11-class model.

    This is NOT a runtime open-set signal detector.
    It DOES NOT infer "unknown" from IQ signal content alone.
    It MUST NOT be inserted into the production inference pipeline.
    It MUST NOT be used to alter hypothesis/fusion behaviour during inference.
    Do NOT claim this provides genuine open-set recognition from IQ alone.

Exact overlaps (SIGMA 11-class ↔ external 7-class):
    BPSK  ↔  BPSK
    QPSK  ↔  QPSK
    WBFM  ↔  WBFM

External QAM family (external "QAM" covers both):
    QAM16 → external QAM
    QAM64 → external QAM

Externally unsupported SIGMA classes (no counterpart in the 7-class taxonomy):
    8PSK, AM-DSB, AM-SSB, CPFSK, GFSK, PAM4

SIGMA-unsupported external classes (no counterpart in the 11-class taxonomy):
    GMSK, OFDM, NBFM
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ml.dataset.labels import MODULATION_CLASSES
from ml.dataset.external_dataset import EXTERNAL_CLASSES


# ── Overlap mappings ──────────────────────────────────────────────────────────

# SIGMA label → external label (exact or family)
SIGMA_TO_EXTERNAL: Dict[str, Optional[str]] = {
    "BPSK":   "BPSK",
    "QPSK":   "QPSK",
    "QAM16":  "QAM",
    "QAM64":  "QAM",
    "WBFM":   "WBFM",
    # No external counterpart:
    "8PSK":   None,
    "AM-DSB": None,
    "AM-SSB": None,
    "CPFSK":  None,
    "GFSK":   None,
    "PAM4":   None,
}

# External label → list of SIGMA labels that map to it
EXTERNAL_TO_SIGMA: Dict[str, List[str]] = {
    "BPSK": ["BPSK"],
    "QPSK": ["QPSK"],
    "QAM":  ["QAM16", "QAM64"],
    "WBFM": ["WBFM"],
    # No SIGMA counterpart:
    "GMSK": [],
    "OFDM": [],
    "NBFM": [],
}

# External classes that have no SIGMA equivalent → "unsupported" in SIGMA taxonomy
EXTERNAL_UNSUPPORTED_BY_SIGMA: List[str] = ["GMSK", "OFDM", "NBFM"]

# SIGMA classes that have no external equivalent
SIGMA_UNSUPPORTED_BY_EXTERNAL: List[str] = [
    s for s, e in SIGMA_TO_EXTERNAL.items() if e is None
]


def sigma_to_external(sigma_label: str) -> Optional[str]:
    """
    Map a SIGMA modulation label to its external taxonomy equivalent.

    Returns None if the SIGMA label has no external counterpart.
    """
    if sigma_label not in SIGMA_TO_EXTERNAL:
        raise ValueError(
            f"Unknown SIGMA label '{sigma_label}'. "
            f"Must be one of: {MODULATION_CLASSES}"
        )
    return SIGMA_TO_EXTERNAL[sigma_label]


def external_to_sigma_candidates(external_label: str) -> List[str]:
    """
    Return the list of SIGMA labels that correspond to an external label.

    Returns an empty list for external labels unsupported by SIGMA (GMSK, OFDM, NBFM).
    """
    if external_label not in EXTERNAL_TO_SIGMA:
        raise ValueError(
            f"Unknown external label '{external_label}'. "
            f"Must be one of: {EXTERNAL_CLASSES}"
        )
    return list(EXTERNAL_TO_SIGMA[external_label])


def is_external_class_supported(external_label: str) -> bool:
    """
    Return True if the external class has at least one SIGMA equivalent.
    """
    return bool(external_to_sigma_candidates(external_label))


# ── OpenSetClassifier ─────────────────────────────────────────────────────────

class OpenSetClassifier:
    """
    Evaluation-time open-set taxonomy classifier.

    EVALUATION USE ONLY.  See module docstring for the open-set boundary.

    Wraps a SIGMA class prediction and a ground-truth external label to
    determine whether the sample belongs to an "unsupported" external class.

    Usage:
        osc = OpenSetClassifier()
        result = osc.classify(
            sigma_prediction="BPSK",
            ground_truth_external="OFDM",
        )
        # result.is_supported  → False   (OFDM has no SIGMA counterpart)
        # result.rejection_reason → "OFDM is not in the SIGMA 11-class taxonomy"
    """

    def classify(
        self,
        sigma_prediction: str,
        ground_truth_external: Optional[str] = None,
    ) -> "OpenSetResult":
        """
        Classify a prediction relative to the external taxonomy.

        Args:
            sigma_prediction:       The SIGMA model's predicted class label.
            ground_truth_external:  Ground-truth external label (for evaluation).
                                    When None, assumes the ground truth is
                                    within the supported classes.

        Returns:
            OpenSetResult.
        """
        if ground_truth_external is None:
            # No ground truth → treat as in-set
            return OpenSetResult(
                sigma_prediction=sigma_prediction,
                ground_truth_external=None,
                is_supported=True,
                external_equivalent=sigma_to_external(sigma_prediction),
                rejection_reason=None,
            )

        if ground_truth_external not in EXTERNAL_CLASSES:
            raise ValueError(
                f"Unknown external label '{ground_truth_external}'. "
                f"Must be one of: {EXTERNAL_CLASSES}"
            )

        supported = is_external_class_supported(ground_truth_external)
        reason = (
            None if supported
            else f"{ground_truth_external} is not in the SIGMA 11-class taxonomy"
        )

        return OpenSetResult(
            sigma_prediction=sigma_prediction,
            ground_truth_external=ground_truth_external,
            is_supported=supported,
            external_equivalent=sigma_to_external(sigma_prediction),
            rejection_reason=reason,
        )


class OpenSetResult:
    """Result from OpenSetClassifier.classify()."""

    def __init__(
        self,
        sigma_prediction: str,
        ground_truth_external: Optional[str],
        is_supported: bool,
        external_equivalent: Optional[str],
        rejection_reason: Optional[str],
    ) -> None:
        self.sigma_prediction = sigma_prediction
        self.ground_truth_external = ground_truth_external
        self.is_supported = is_supported
        self.external_equivalent = external_equivalent
        self.rejection_reason = rejection_reason

    def __repr__(self) -> str:
        return (
            f"OpenSetResult(prediction={self.sigma_prediction!r}, "
            f"ground_truth={self.ground_truth_external!r}, "
            f"is_supported={self.is_supported}, "
            f"external_equivalent={self.external_equivalent!r})"
        )
