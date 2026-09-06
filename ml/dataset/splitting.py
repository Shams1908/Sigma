import numpy as np
from typing import Tuple, List

def generate_splits(
    sorted_keys: List[Tuple[str, int]],
    examples_per_key: int = 1000,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministically generates stratified train, validation, and test indices.
    
    For each (modulation, SNR) group containing `examples_per_key` elements, 
    the indices are shuffled using a configuration seed and split based on the ratios.
    
    Args:
        sorted_keys (List[Tuple[str, int]]): Deterministically sorted dataset keys.
        examples_per_key (int): Number of examples per key (default 1000).
        train_ratio (float): Fraction of examples for training (default 0.70).
        val_ratio (float): Fraction of examples for validation (default 0.15).
        test_ratio (float): Fraction of examples for testing (default 0.15).
        seed (int): The seed for the random number generator (default 42).
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Flat integer indices for (train, validation, test) splits.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got: "
            f"train={train_ratio}, val={val_ratio}, test={test_ratio} "
            f"(sum = {train_ratio + val_ratio + test_ratio})"
        )

    # Use default_rng for modern, thread-safe, reproducible numpy random generation
    rng = np.random.default_rng(seed)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    n_train = int(examples_per_key * train_ratio)
    n_val = int(examples_per_key * val_ratio)
    n_test = examples_per_key - n_train - n_val  # Remainder goes to test to guarantee exactly examples_per_key
    
    for i, key in enumerate(sorted_keys):
        group_offset = i * examples_per_key
        
        # Generate stable permutation for the group
        perm = rng.permutation(examples_per_key)
        
        group_train = perm[:n_train] + group_offset
        group_val = perm[n_train:n_train + n_val] + group_offset
        group_test = perm[n_train + n_val:] + group_offset
        
        train_indices.extend(group_train)
        val_indices.extend(group_val)
        test_indices.extend(group_test)
        
    # Convert lists to NumPy arrays of integer type
    return (
        np.array(train_indices, dtype=np.int32),
        np.array(val_indices, dtype=np.int32),
        np.array(test_indices, dtype=np.int32)
    )
