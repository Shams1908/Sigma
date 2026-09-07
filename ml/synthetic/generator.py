import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple

from ml.generators.config import GeneratorConfig
from ml.generators import generate_bits, generate_modulated_signal
from ml.dataset.labels import get_class_index
from ml.synthetic.config import (
    SUPPORTED_MODULATION_CLASSES,
    NOMINAL_CONFIG,
    SWEEP_EXPERIMENTS
)

# Group mapping of bits per symbol for modulations
BITS_PER_SYMBOL: Dict[str, int] = {
    "BPSK": 1,
    "QPSK": 2,
    "8PSK": 3,
    "QAM16": 4,
    "QAM64": 6
}

def generate_synthetic_evaluation_dataset(
    num_examples_per_cond: int = 100,
    output_dir: str = "datasets/synthetic"
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Generates a controlled, deterministic synthetic evaluation dataset.
    Varies exactly one impairment parameter at a time.
    
    Args:
        num_examples_per_cond (int): Number of examples per modulation x SNR x impairment condition.
        output_dir (str): Directory where dataset will be saved.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, List[Dict]]:
            - X: Shape [N, 2, 128] float32 array
            - y: Shape [N] int32 array (class index)
            - metadata_list: List of dictionaries containing ground truth configurations
    """
    print(f"Generating synthetic evaluation dataset ({num_examples_per_cond} examples per combination)...")
    os.makedirs(output_dir, exist_ok=True)
    
    X_list = []
    y_list = []
    metadata_list = []
    
    global_index = 0
    # Base seed for generating bits and noise
    base_seed = 42000
    
    for exp_name, exp_info in SWEEP_EXPERIMENTS.items():
        sweep_param = exp_info["sweep_parameter"]
        sweep_values = exp_info["sweep_values"]
        defaults = exp_info["defaults"]
        
        print(f"  Processing experiment: {exp_name} (sweeping {sweep_param})...")
        
        for val in sweep_values:
            for mod in SUPPORTED_MODULATION_CLASSES:
                # Group-wise generation loop
                for ex_idx in range(num_examples_per_cond):
                    # Deterministic seed unique to this sample
                    seed = base_seed + global_index
                    
                    # Compute bit stream length
                    bits_per_sym = BITS_PER_SYMBOL[mod]
                    num_bits = NOMINAL_CONFIG["num_symbols"] * bits_per_sym
                    
                    # Generate deterministic bits
                    bits = generate_bits(num_bits, seed_or_generator=seed)
                    
                    # Formulate parameters for GeneratorConfig
                    config_args = {
                        "modulation": mod,
                        "num_symbols": NOMINAL_CONFIG["num_symbols"],
                        "sample_rate": NOMINAL_CONFIG["sample_rate"],
                        "symbol_rate": NOMINAL_CONFIG["symbol_rate"],
                        "samples_per_symbol": NOMINAL_CONFIG["samples_per_symbol"],
                        "filter_span_symbols": NOMINAL_CONFIG["filter_span_symbols"],
                        "rolloff": NOMINAL_CONFIG["rolloff"],
                        "random_seed": seed,
                        # Set default impairments
                        "snr": defaults.get("snr", None),
                        "frequency_offset": defaults.get("frequency_offset", None),
                        "phase_offset": defaults.get("phase_offset", None),
                        "timing_offset": defaults.get("timing_offset", None),
                        "dc_offset_i": defaults.get("dc_offset_i", None),
                        "dc_offset_q": defaults.get("dc_offset_q", None),
                        "iq_amplitude_imbalance": defaults.get("iq_amplitude_imbalance", None),
                        "iq_phase_imbalance": defaults.get("iq_phase_imbalance", None)
                    }
                    
                    # Overlay the swept value
                    if sweep_param == "snr":
                        config_args["snr"] = float(val)
                    elif sweep_param == "dc_offset":
                        # Set equal DC offset on both In-phase and Quadrature channels
                        config_args["dc_offset_i"] = float(val)
                        config_args["dc_offset_q"] = float(val)
                    else:
                        config_args[sweep_param] = float(val)
                        
                    # Construct GeneratorConfig (which runs validation automatically)
                    config = GeneratorConfig(**config_args)
                    
                    # Generate waveform with explicit deterministic RNG
                    rng = np.random.default_rng(seed)
                    signal = generate_modulated_signal(bits, config, rng=rng)
                    
                    # Verify shape is strictly [2, 128]
                    assert signal.samples.shape == (2, 128), f"Expected shape (2, 128), got {signal.samples.shape}"
                    assert signal.samples.dtype == np.float32, f"Expected float32, got {signal.samples.dtype}"
                    
                    # Save sample and label
                    X_list.append(signal.samples)
                    y_list.append(get_class_index(mod))
                    
                    # Save detailed metadata
                    metadata_list.append({
                        "index": global_index,
                        "experiment": exp_name,
                        "sweep_parameter": sweep_param,
                        "sweep_value": float(val),
                        "modulation": mod,
                        "class_index": get_class_index(mod),
                        "snr": config.snr,
                        "random_seed": seed,
                        "generation_config": {
                            "modulation": config.modulation,
                            "num_symbols": config.num_symbols,
                            "sample_rate": config.sample_rate,
                            "symbol_rate": config.symbol_rate,
                            "samples_per_symbol": config.samples_per_symbol,
                            "filter_span_symbols": config.filter_span_symbols,
                            "rolloff": config.rolloff,
                            "random_seed": config.random_seed,
                            "snr": config.snr,
                            "frequency_offset": config.frequency_offset,
                            "phase_offset": config.phase_offset,
                            "timing_offset": config.timing_offset,
                            "dc_offset_i": config.dc_offset_i,
                            "dc_offset_q": config.dc_offset_q,
                            "iq_amplitude_imbalance": config.iq_amplitude_imbalance,
                            "iq_phase_imbalance": config.iq_phase_imbalance
                        }
                    })
                    
                    global_index += 1

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Save to NPZ file
    npz_path = os.path.join(output_dir, "synthetic_evaluation_dataset.npz")
    print(f"Saving feature matrix to {npz_path}...")
    np.savez_compressed(npz_path, X=X, y=y)
    
    # Save to JSON metadata file
    json_path = os.path.join(output_dir, "synthetic_evaluation_metadata.json")
    print(f"Saving metadata list to {json_path}...")
    with open(json_path, "w") as f:
        json.dump(metadata_list, f, indent=4)
        
    print(f"Generation complete! Total generated examples: {X.shape[0]}")
    return X, y, metadata_list

if __name__ == "__main__":
    generate_synthetic_evaluation_dataset()
