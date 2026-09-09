"""
M7-J: Open-Set and Unknown Modulation Rejection Evaluation.

Evaluates Maximum Softmax Probability (MSP) and Shannon Entropy rejection.
Thresholds are selected/calibrated strictly on the development subset, frozen,
and then evaluated on test data.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ml.evaluation.taxonomy import (
    CLOSED_SET_CLASSES,
    OPEN_SET_CLASSES,
    EXTERNAL_IDX_TO_NAME,
)


@dataclass(frozen=True)
class OpenSetThresholds:
    """Frozen rejection thresholds."""
    confidence_threshold: float  # MSP must be >= threshold to accept
    entropy_threshold: float     # Shannon entropy must be <= threshold to accept
    source: str = "FROZEN_DEV_SUBSET"


@dataclass
class OpenSetMetrics:
    """Summary of open-set acceptance and rejection rates."""
    total_samples: int
    known_samples: int
    unknown_samples: int
    known_accepted: int
    known_rejected: int
    unknown_rejected: int
    unknown_false_accepted: int
    known_acceptance_rate: float
    unknown_rejection_rate: float
    false_acceptance_rate: float
    false_rejection_rate: float
    thresholds: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "known_samples": self.known_samples,
            "unknown_samples": self.unknown_samples,
            "known_accepted": self.known_accepted,
            "known_rejected": self.known_rejected,
            "unknown_rejected": self.unknown_rejected,
            "unknown_false_accepted": self.unknown_false_accepted,
            "known_acceptance_rate": self.known_acceptance_rate,
            "unknown_rejection_rate": self.unknown_rejection_rate,
            "false_acceptance_rate": self.false_acceptance_rate,
            "false_rejection_rate": self.false_rejection_rate,
            "thresholds": self.thresholds,
        }


def compute_entropy(probs: np.ndarray) -> float:
    """Computes Shannon entropy: -sum p ln(p)."""
    safe_p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(safe_p * np.log(safe_p)))


def select_thresholds_on_dev(
    dev_probs: np.ndarray,
    dev_y_mod: List[int],
    target_known_acceptance: float = 0.90,
) -> OpenSetThresholds:
    """
    Selects rejection thresholds strictly from the development subset.
    
    Finds the MSP threshold that maintains target_known_acceptance (e.g. 90%)
    on known classes, and the corresponding entropy threshold at the 90th percentile.
    """
    known_msps = []
    known_entropies = []

    for p, mod_idx in zip(dev_probs, dev_y_mod):
        mod_name = EXTERNAL_IDX_TO_NAME.get(int(mod_idx), "")
        if mod_name in CLOSED_SET_CLASSES:
            known_msps.append(float(np.max(p)))
            known_entropies.append(compute_entropy(p))

    if not known_msps:
        # Fallback to default heuristic if dev subset lacks knowns
        return OpenSetThresholds(confidence_threshold=0.35, entropy_threshold=1.80)

    # Threshold for MSP at (1 - target_known_acceptance) quantile
    pct = (1.0 - target_known_acceptance) * 100.0
    conf_thresh = float(np.percentile(known_msps, pct))
    conf_thresh = max(0.15, min(0.90, conf_thresh))

    # Threshold for entropy at target_known_acceptance quantile
    ent_pct = target_known_acceptance * 100.0
    ent_thresh = float(np.percentile(known_entropies, ent_pct))
    ent_thresh = max(0.5, min(2.5, ent_thresh))

    return OpenSetThresholds(
        confidence_threshold=float(conf_thresh),
        entropy_threshold=float(ent_thresh),
        source="M6-DEV-SUBSET-CALIBRATED",
    )


class OpenSetEvaluator:
    """
    Evaluates open-set rejection on evaluation samples using frozen thresholds.
    """
    def __init__(self, thresholds: OpenSetThresholds):
        self.thresholds = thresholds

    def is_accepted(self, probs: np.ndarray) -> bool:
        """
        Decision rule:
          Accept as KNOWN if max(probs) >= conf_threshold AND entropy <= ent_threshold.
          Otherwise reject as UNKNOWN / LOW_CONFIDENCE.
        """
        msp = float(np.max(probs))
        ent = compute_entropy(probs)
        return (msp >= self.thresholds.confidence_threshold) and (ent <= self.thresholds.entropy_threshold)

    def evaluate(
        self,
        probs: np.ndarray,
        y_mod: List[int],
    ) -> OpenSetMetrics:
        """
        Evaluates acceptance/rejection across all samples.
        """
        known_samples = 0
        unknown_samples = 0
        known_accepted = 0
        known_rejected = 0
        unknown_rejected = 0
        unknown_false_accepted = 0

        for p, mod_idx in zip(probs, y_mod):
            mod_name = EXTERNAL_IDX_TO_NAME.get(int(mod_idx), "")
            is_known = (mod_name in CLOSED_SET_CLASSES)
            accepted = self.is_accepted(p)

            if is_known:
                known_samples += 1
                if accepted:
                    known_accepted += 1
                else:
                    known_rejected += 1
            else:
                unknown_samples += 1
                if accepted:
                    unknown_false_accepted += 1
                else:
                    unknown_rejected += 1

        known_acc_rate = float(known_accepted / known_samples) if known_samples > 0 else 0.0
        false_rej_rate = float(known_rejected / known_samples) if known_samples > 0 else 0.0
        unknown_rej_rate = float(unknown_rejected / unknown_samples) if unknown_samples > 0 else 0.0
        false_acc_rate = float(unknown_false_accepted / unknown_samples) if unknown_samples > 0 else 0.0

        return OpenSetMetrics(
            total_samples=len(probs),
            known_samples=known_samples,
            unknown_samples=unknown_samples,
            known_accepted=known_accepted,
            known_rejected=known_rejected,
            unknown_rejected=unknown_rejected,
            unknown_false_accepted=unknown_false_accepted,
            known_acceptance_rate=known_acc_rate,
            unknown_rejection_rate=unknown_rej_rate,
            false_acceptance_rate=false_acc_rate,
            false_rejection_rate=false_rej_rate,
            thresholds={
                "confidence_threshold": self.thresholds.confidence_threshold,
                "entropy_threshold": self.thresholds.entropy_threshold,
                "source": self.thresholds.source,
            },
        )
