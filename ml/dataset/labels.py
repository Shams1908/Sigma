from typing import List, Dict

# Central source of truth for the 11 RadioML modulation labels.
# Ordered alphabetically to ensure a deterministic and stable mapping.
MODULATION_CLASSES: List[str] = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
]

# Stable mapping between label string and integer class index.
MODULATION_TO_INDEX: Dict[str, int] = {
    label: idx for idx, label in enumerate(MODULATION_CLASSES)
}

INDEX_TO_MODULATION: Dict[int, str] = {
    idx: label for idx, label in enumerate(MODULATION_CLASSES)
}

def get_class_index(modulation: str) -> int:
    """
    Get the stable integer class index for a given modulation label.
    
    Args:
        modulation (str): The modulation class name (e.g. '8PSK').
        
    Returns:
        int: The stable class index.
        
    Raises:
        ValueError: If the modulation is not supported.
    """
    if modulation not in MODULATION_TO_INDEX:
        raise ValueError(
            f"Unsupported modulation class: '{modulation}'. "
            f"Must be one of: {MODULATION_CLASSES}"
        )
    return MODULATION_TO_INDEX[modulation]

def get_modulation_name(class_index: int) -> str:
    """
    Get the modulation class name for a given class index.
    
    Args:
        class_index (int): The integer class index.
        
    Returns:
        str: The modulation class name.
        
    Raises:
        ValueError: If the class index is out of bounds.
    """
    if class_index not in INDEX_TO_MODULATION:
        raise ValueError(
            f"Invalid class index: {class_index}. "
            f"Must be between 0 and {len(MODULATION_CLASSES) - 1}."
        )
    return INDEX_TO_MODULATION[class_index]
