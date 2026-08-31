import numpy as np
from typing import Tuple

def construct_mixed_dataset(
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    ratio_pct: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs a mixed training dataset with a controlled synthetic-to-real ratio,
    ensuring class balance across BPSK, QPSK, 8PSK, QAM16, QAM64.
    
    Total combined size per class is:
      - 14,000 for 10% and 25% ratios (to maintain overall size same as real-only)
      - 7,000 for 50% ratio (constrained by the available 3,600 synthetic samples per class)
      
    Deterministic subsampling is used.
    """
    # Define targets per class
    classes = [0, 1, 2, 3, 4]
    
    if ratio_pct == 10:
        syn_per_class = 1400
        real_per_class = 12600
    elif ratio_pct == 25:
        syn_per_class = 3500
        real_per_class = 10500
    elif ratio_pct == 50:
        syn_per_class = 3500
        real_per_class = 3500
    else:
        raise ValueError(f"Unsupported ratio percentage: {ratio_pct}")
        
    X_mixed_list = []
    y_mixed_list = []
    
    # Report class balance
    print(f"  Constructing {ratio_pct}% mixed dataset:")
    print(f"    Real examples per class:      {real_per_class}")
    print(f"    Synthetic examples per class: {syn_per_class}")
    print(f"    Total combined per class:     {real_per_class + syn_per_class}")
    
    for c in classes:
        # Subsample Real deterministically (first real_per_class instances)
        real_mask = np.where(y_real == c)[0]
        real_selected = real_mask[:real_per_class]
        
        # Subsample Synthetic deterministically (first syn_per_class instances)
        syn_mask = np.where(y_syn == c)[0]
        syn_selected = syn_mask[:syn_per_class]
        
        # Append samples
        X_mixed_list.append(X_real[real_selected])
        X_mixed_list.append(X_syn[syn_selected])
        
        y_mixed_list.extend([c] * (real_per_class + syn_per_class))
        
    X_mixed = np.concatenate(X_mixed_list, axis=0)
    y_mixed = np.array(y_mixed_list, dtype=np.int32)
    
    # Shuffle mixed dataset deterministically using seed 42 to mix real and synthetic samples in batches
    shuffler = np.random.default_rng(42)
    indices = shuffler.permutation(len(X_mixed))
    
    return X_mixed[indices], y_mixed[indices]
