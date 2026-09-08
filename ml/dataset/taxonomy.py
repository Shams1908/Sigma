"""
Taxonomy Alignment and Configurable Open-Set Rejection for SIGMA.

Handles mapping between the 11-class RadioML baseline taxonomy and the
7-class external Real-World benchmark, providing principled open-set rejection
for unknown or unsupported modulation schemes.

Threshold Selection Note for M7:
In accordance with M6 design principles, rejection thresholds (confidence and entropy)
are NOT tuned or optimized against the external test set. Default thresholds are provided
as configurable heuristics. During M7, formal calibration (e.g. Temperature Scaling,
Precision-Recall at 95% TPR, or AUROC optimization) should be performed on a dedicated
validation set before final benchmark reporting.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

from ml.dataset.labels import MODULATION_CLASSES
from ml.dataset.external_dataset import EXTERNAL_MODULATION_CLASSES


# Classes supported by the M5 11-class RadioML model
SUPPORTED_M5_CLASSES: List[str] = list(MODULATION_CLASSES)

# Direct exact matches between external and RadioML classes
EXACT_OVERLAP_CLASSES: Set[str] = {"BPSK", "QPSK", "WBFM"}

# Family-level match: External "QAM" covers both QAM16 and QAM64 in RadioML
QAM_FAMILY_CLASSES: Set[str] = {"QAM16", "QAM64"}

# Classes in external dataset completely unsupported by M5
UNSUPPORTED_EXTERNAL_CLASSES: Set[str] = {"GMSK", "OFDM", "NBFM"}


def is_external_class_supported(mod_name: str) -> bool:
    """Returns True if the external class is supported or maps to a supported family."""
    if mod_name in EXACT_OVERLAP_CLASSES:
        return True
    if mod_name == "QAM":
        return True
    return False


def is_prediction_compatible(predicted_m5_class: str, external_class: str) -> bool:
    """
    Checks if an M5 predicted class is compatible with the external ground-truth class.
    
    Examples:
      - predicted 'BPSK', external 'BPSK' -> True
      - predicted 'QAM16', external 'QAM' -> True
      - predicted 'QAM64', external 'QAM' -> True
      - predicted '8PSK', external 'QAM' -> False
      - predicted 'BPSK', external 'OFDM' -> False (OFDM is unsupported)
    """
    if external_class in EXACT_OVERLAP_CLASSES:
        return predicted_m5_class == external_class
    if external_class == "QAM":
        return predicted_m5_class in QAM_FAMILY_CLASSES
    return False


@dataclass(frozen=True)
class OpenSetPrediction:
    """
    Structured result of open-set / unsupported classification.
    """
    predicted_class: str
    confidence: float
    entropy: float
    decision: str            # "KNOWN", "LOW_CONFIDENCE", "UNSUPPORTED"
    is_supported: bool       # Whether ground truth (if provided) is in supported set
    is_correct: Optional[bool]  # Compatibility match (if ground truth provided)


class OpenSetClassifier:
    """
    Rejection filter for unsupported modulations and low-confidence predictions.
    
    Attributes:
        confidence_threshold: Minimum maximum softmax probability (MSP) to accept a prediction.
        entropy_threshold: Maximum Shannon entropy (-sum p ln p) to accept a prediction.
    """
    def __init__(
        self,
        confidence_threshold: float = 0.35,
        entropy_threshold: float = 1.80,
    ):
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0, 1], got {confidence_threshold}")
        if entropy_threshold <= 0.0:
            raise ValueError(f"entropy_threshold must be > 0, got {entropy_threshold}")
            
        self.confidence_threshold = float(confidence_threshold)
        self.entropy_threshold = float(entropy_threshold)

    def classify_probs(
        self,
        probs: np.ndarray,
        class_labels: List[str] = SUPPORTED_M5_CLASSES,
        ground_truth: Optional[str] = None,
    ) -> OpenSetPrediction:
        """
        Classifies a probability vector and applies open-set rejection heuristics.
        
        Args:
            probs: 1D NumPy array of probabilities summing to 1.
            class_labels: Ordered list of class names corresponding to probability indices.
            ground_truth: Optional ground truth class name from external dataset.
            
        Returns:
            OpenSetPrediction dataclass.
        """
        if not isinstance(probs, np.ndarray) or probs.ndim != 1:
            raise ValueError("probs must be a 1D numpy array")

        pred_idx = int(np.argmax(probs))
        pred_class = class_labels[pred_idx]
        confidence = float(probs[pred_idx])
        
        # Shannon entropy: -sum p_i ln(p_i)
        safe_probs = np.clip(probs, 1e-12, 1.0)
        entropy = float(-np.sum(safe_probs * np.log(safe_probs)))

        is_supported = True
        if ground_truth is not None:
            is_supported = is_external_class_supported(ground_truth)

        # Decision rule:
        # 1. If ground truth is provided and explicitly known to be unsupported -> UNSUPPORTED
        # 2. If confidence is below threshold or entropy is too high -> LOW_CONFIDENCE
        # 3. Otherwise -> KNOWN
        if ground_truth is not None and not is_supported:
            decision = "UNSUPPORTED"
        elif confidence < self.confidence_threshold or entropy > self.entropy_threshold:
            decision = "LOW_CONFIDENCE"
        else:
            decision = "KNOWN"

        is_correct = None
        if ground_truth is not None:
            is_correct = is_prediction_compatible(pred_class, ground_truth) and (decision == "KNOWN")

        return OpenSetPrediction(
            predicted_class=pred_class,
            confidence=confidence,
            entropy=entropy,
            decision=decision,
            is_supported=is_supported,
            is_correct=is_correct,
        )
