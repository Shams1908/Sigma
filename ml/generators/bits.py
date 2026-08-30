import numpy as np
from typing import Union

def generate_bits(
    num_bits: int, 
    seed_or_generator: Union[int, np.random.Generator]
) -> np.ndarray:
    """
    Generates a deterministic sequence of random binary bits (0 or 1).
    
    This function avoids global NumPy random state mutation by requiring either
    a NumPy Generator or an explicit random seed value.
    
    Args:
        num_bits (int): The number of bits to generate.
        seed_or_generator (Union[int, np.random.Generator]): An integer random seed 
                                                            or a NumPy Generator instance.
                                                            
    Returns:
        np.ndarray: A 1D array of shape (num_bits,) and dtype np.int8 containing bits.
        
    Raises:
        TypeError: If num_bits is not an integer.
        ValueError: If num_bits is non-positive or if generated output length is incorrect.
    """
    if not isinstance(num_bits, (int, np.integer)):
        raise TypeError(f"num_bits must be an integer, got {type(num_bits)}")
    if num_bits <= 0:
        raise ValueError(f"num_bits must be greater than 0, got {num_bits}")

    # Resolve Generator instance without touching global np.random state
    if isinstance(seed_or_generator, np.random.Generator):
        rng = seed_or_generator
    else:
        if not isinstance(seed_or_generator, (int, np.integer)):
            raise TypeError(f"seed must be an integer, got {type(seed_or_generator)}")
        if seed_or_generator < 0:
            raise ValueError(f"seed must be non-negative, got {seed_or_generator}")
        rng = np.random.default_rng(seed_or_generator)

    # Generate bits
    bits = rng.integers(0, 2, size=num_bits, dtype=np.int8)

    # Enforce strict output length validation
    if len(bits) != num_bits:
        raise ValueError(
            f"Generated bit length ({len(bits)}) does not match requested num_bits ({num_bits})"
        )

    return bits
