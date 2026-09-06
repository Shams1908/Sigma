import numpy as np
from typing import Dict, Tuple, Any
from ml.dataset.labels import MODULATION_CLASSES

def validate_raw_data(data: Any, expected_examples_per_key: int = 1000) -> None:
    """
    Validates that the raw dataset object matches the expected structure of RadioML 2016.10A.
    
    Args:
        data: The loaded pickle object, expected to be a dictionary.
        expected_examples_per_key (int): Expected number of examples per key (default 1000).
        
    Raises:
        TypeError: If data is not a dictionary or contains invalid types.
        ValueError: If elements, shapes, SNRs, or labels are wrong/missing.
        KeyError: If an expected modulation/SNR combination key is missing.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Dataset must be a Python dictionary, got {type(data)}.")
    
    keys = list(data.keys())
    if len(keys) == 0:
        raise ValueError("Dataset dictionary is empty.")

    # Validate key structure: each key must be a tuple of (modulation_str, snr_int)
    for k in keys:
        if not isinstance(k, tuple):
            raise TypeError(f"Dataset key must be a tuple, got {type(k)} for key: {k}")
        if len(k) != 2:
            raise ValueError(f"Dataset key must be a 2-tuple (modulation, SNR), got length {len(k)} for key: {k}")
        
        modulation, snr = k
        if not isinstance(modulation, str):
            raise TypeError(f"Modulation label in key {k} must be a string, got type: {type(modulation)}")
        if not isinstance(snr, (int, float, np.integer)):
            raise TypeError(f"SNR value in key {k} must be numeric, got type: {type(snr)}")

    # Extract unique modulation classes and SNRs present in the data keys
    found_modulations = set(k[0] for k in keys)
    found_snrs = set(k[1] for k in keys)

    # 1. Verify exactly 11 modulation labels are present
    expected_modulations = set(MODULATION_CLASSES)
    if found_modulations != expected_modulations:
        missing_mods = expected_modulations - found_modulations
        extra_mods = found_modulations - expected_modulations
        raise ValueError(
            f"Modulation classes mismatch. Expected exactly 11 classes: {expected_modulations}.\n"
            f"Missing classes: {missing_mods}\n"
            f"Extra classes: {extra_mods}"
        )

    # 2. Verify exactly 20 SNR values are present
    expected_snrs = {-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18}
    # Convert found_snrs to integers if they are float representations (e.g. 2.0 -> 2)
    found_snrs_int = set(int(s) for s in found_snrs)
    if found_snrs_int != expected_snrs:
        missing_snrs = expected_snrs - found_snrs_int
        extra_snrs = found_snrs_int - expected_snrs
        raise ValueError(
            f"SNR values mismatch. Expected exactly 20 values: {sorted(list(expected_snrs))}.\n"
            f"Missing SNRs: {sorted(list(missing_snrs))}\n"
            f"Extra SNRs: {sorted(list(extra_snrs))}"
        )

    # 3. Verify each modulation/SNR combination exists and contains expected examples with shape (2, 128)
    total_examples = 0
    for mod in MODULATION_CLASSES:
        for snr in sorted(list(expected_snrs)):
            # Try to match the key as either (mod, snr) or (mod, float(snr)) or (mod, np.int64(snr))
            # Pickle loading could preserve exact key types
            matching_key = None
            for k in keys:
                if k[0] == mod and int(k[1]) == snr:
                    matching_key = k
                    break

            if matching_key is None:
                raise KeyError(f"Missing expected modulation/SNR combination key: ('{mod}', {snr})")

            val = data[matching_key]
            if not isinstance(val, np.ndarray):
                raise TypeError(f"Value for key {matching_key} must be a numpy ndarray, got {type(val)}")

            # Check dimensions and shape
            if val.ndim != 3:
                raise ValueError(
                    f"Array for key {matching_key} must be 3-dimensional (examples, channels, samples), "
                    f"got {val.ndim}-D array of shape {val.shape}"
                )
            
            n_examples, n_channels, n_samples = val.shape
            if n_examples != expected_examples_per_key:
                raise ValueError(
                    f"Array for key {matching_key} must have exactly {expected_examples_per_key} examples, got {n_examples}"
                )
            if n_channels != 2 or n_samples != 128:
                raise ValueError(
                    f"Array for key {matching_key} must have sample shape (2, 128), got ({n_channels}, {n_samples})"
                )

            # Check data type
            if not np.issubdtype(val.dtype, np.number):
                raise TypeError(f"Array for key {matching_key} must be numeric, got dtype {val.dtype}")

            # Check that it is safely convertible to float32
            if not np.can_cast(val.dtype, np.float32, casting="safe") and val.dtype != np.float32:
                raise TypeError(
                    f"Array for key {matching_key} has dtype {val.dtype} which is not safely convertible to float32"
                )

            total_examples += n_examples

    # 4. Verify total examples is exactly expected_total
    expected_total = len(MODULATION_CLASSES) * len(expected_snrs) * expected_examples_per_key
    if total_examples != expected_total:
        raise ValueError(f"Total examples mismatch. Expected exactly {expected_total}, got {total_examples}")

