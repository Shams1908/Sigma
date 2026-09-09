"""
M7-B: Taxonomy Alignment and Classification Resolution.

Explicit Evaluation Taxonomy:
  1. BPSK
  2. QPSK
  3. QAM (evaluated strictly at the family level: QAM16 -> QAM, QAM64 -> QAM)
  4. WBFM
  5. UNKNOWN/UNSUPPORTED

Unsupported classes in external benchmark:
  - GMSK
  - OFDM
  - NBFM
These are strictly evaluated in open-set rejection and never coerced into known classes.
"""
from typing import List, Dict, Set, Optional, Tuple

# External dataset integer-to-name mapping
EXTERNAL_IDX_TO_NAME: Dict[int, str] = {
    0: "BPSK",
    1: "QPSK",
    2: "QAM",
    3: "GMSK",
    4: "OFDM",
    5: "NBFM",
    6: "WBFM",
}

EXTERNAL_NAME_TO_IDX: Dict[str, int] = {
    v: k for k, v in EXTERNAL_IDX_TO_NAME.items()
}

# The 4 supported classes evaluated in closed-set benchmarks
CLOSED_SET_CLASSES: List[str] = ["BPSK", "QPSK", "QAM", "WBFM"]

# The 3 unsupported classes evaluated in open-set benchmarks
OPEN_SET_CLASSES: List[str] = ["GMSK", "OFDM", "NBFM"]

# The explicit M7 evaluation taxonomy
EVALUATION_TAXONOMY: List[str] = [
    "BPSK",
    "QPSK",
    "QAM",
    "WBFM",
    "UNKNOWN/UNSUPPORTED",
]

# RadioML 11 classes mapped to evaluation classes
# QAM16 and QAM64 map to QAM at family level.
# BPSK, QPSK, WBFM map directly.
# Other RadioML classes (8PSK, AM-DSB, AM-SSB, CPFSK, GFSK, PAM4) map to OTHER_CLOSED.
RADIOML_TO_EVAL_MAP: Dict[str, str] = {
    "BPSK": "BPSK",
    "QPSK": "QPSK",
    "QAM16": "QAM",
    "QAM64": "QAM",
    "WBFM": "WBFM",
    "8PSK": "OTHER_CLOSED",
    "AM-DSB": "OTHER_CLOSED",
    "AM-SSB": "OTHER_CLOSED",
    "CPFSK": "OTHER_CLOSED",
    "GFSK": "OTHER_CLOSED",
    "PAM4": "OTHER_CLOSED",
}


def is_external_class_supported(external_class_name: str) -> bool:
    """Returns True if the external class is in the closed-set evaluation taxonomy."""
    return external_class_name in CLOSED_SET_CLASSES


def map_prediction_to_eval_class(model_predicted_class: str) -> str:
    """
    Maps a raw 11-class RadioML prediction to the evaluation taxonomy.
    
    QAM16 and QAM64 map to QAM at the family level.
    """
    return RADIOML_TO_EVAL_MAP.get(model_predicted_class, "OTHER_CLOSED")


def is_eval_prediction_correct(model_predicted_class: str, external_ground_truth: str) -> bool:
    """
    Checks correctness under the M7 evaluation contract.
    
    Rules:
      - external 'BPSK' matches predicted 'BPSK'
      - external 'QPSK' matches predicted 'QPSK'
      - external 'WBFM' matches predicted 'WBFM'
      - external 'QAM' matches predicted 'QAM16' or 'QAM64' (family level)
      - external 'GMSK', 'OFDM', 'NBFM' are unsupported -> always False in closed-set matching
    """
    if external_ground_truth in ("BPSK", "QPSK", "WBFM"):
        return model_predicted_class == external_ground_truth
    elif external_ground_truth == "QAM":
        return model_predicted_class in ("QAM16", "QAM64")
    return False


def partition_closed_and_open_set_indices(
    y_mod: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Partitions dataset sample indices into closed-set (supported) and open-set (unsupported).
    
    Returns:
        (closed_set_indices, open_set_indices)
    """
    closed_indices = []
    open_indices = []
    for idx, mod_val in enumerate(y_mod):
        mod_name = EXTERNAL_IDX_TO_NAME.get(int(mod_val), "")
        if mod_name in CLOSED_SET_CLASSES:
            closed_indices.append(idx)
        elif mod_name in OPEN_SET_CLASSES:
            open_indices.append(idx)
        else:
            raise ValueError(f"Unknown external modulation class index: {mod_val}")
    return closed_indices, open_indices
