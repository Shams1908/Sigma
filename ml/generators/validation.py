import numpy as np
from typing import Any
from ml.dataset.labels import MODULATION_CLASSES

def validate_config(config: Any) -> None:
    """
    Validates synthetic signal generator configuration parameters.
    
    Args:
        config (GeneratorConfig): The configuration object to validate.
        
    Raises:
        TypeError: If parameter types are incorrect.
        ValueError: If parameters violate mathematical or physical constraints.
    """
    # 1. Validate Modulation against central label definitions (one-way dependency)
    if config.modulation not in MODULATION_CLASSES:
        raise ValueError(
            f"Unsupported modulation class: '{config.modulation}'. "
            f"Must be one of: {MODULATION_CLASSES}"
        )

    # 2. Validate Type and Ranges of Core Generation Fields
    if not isinstance(config.num_symbols, (int, np.integer)):
        raise TypeError(f"num_symbols must be an integer, got {type(config.num_symbols)}")
    if config.num_symbols <= 0:
        raise ValueError(f"num_symbols must be greater than 0, got {config.num_symbols}")

    if not isinstance(config.sample_rate, (int, float, np.number)):
        raise TypeError(f"sample_rate must be numeric, got {type(config.sample_rate)}")
    if config.sample_rate <= 0:
        raise ValueError(f"sample_rate must be greater than 0, got {config.sample_rate}")

    if not isinstance(config.symbol_rate, (int, float, np.number)):
        raise TypeError(f"symbol_rate must be numeric, got {type(config.symbol_rate)}")
    if config.symbol_rate <= 0:
        raise ValueError(f"symbol_rate must be greater than 0, got {config.symbol_rate}")

    if not isinstance(config.samples_per_symbol, (int, np.integer)):
        raise TypeError(f"samples_per_symbol must be an integer, got {type(config.samples_per_symbol)}")
    if config.samples_per_symbol <= 0:
        raise ValueError(f"samples_per_symbol must be greater than 0, got {config.samples_per_symbol}")

    # 3. Validate Mathematical Consistency between Rates
    # samples_per_symbol should be mathematically consistent: samples_per_symbol == sample_rate / symbol_rate
    expected_sps = config.sample_rate / config.symbol_rate
    if not np.isclose(config.samples_per_symbol, expected_sps):
        raise ValueError(
            f"Mathematical inconsistency: samples_per_symbol ({config.samples_per_symbol}) "
            f"must equal sample_rate ({config.sample_rate}) / symbol_rate ({config.symbol_rate}) = {expected_sps}."
        )

    # 4. Validate Random Seed
    if not isinstance(config.random_seed, (int, np.integer)):
        raise TypeError(f"random_seed must be an integer, got {type(config.random_seed)}")
    if config.random_seed < 0:
        raise ValueError(f"random_seed must be non-negative, got {config.random_seed}")

    # 5. Validate Optional RF Impairment Parameters (if supplied, must be finite and numeric)
    if config.snr is not None:
        if not isinstance(config.snr, (int, float, np.number)):
            raise TypeError(f"snr must be numeric, got {type(config.snr)}")
        if not np.isfinite(config.snr):
            raise ValueError(f"snr must be finite, got {config.snr}")

    if config.frequency_offset is not None:
        if not isinstance(config.frequency_offset, (int, float, np.number)):
            raise TypeError(f"frequency_offset must be numeric, got {type(config.frequency_offset)}")
        if not np.isfinite(config.frequency_offset):
            raise ValueError(f"frequency_offset must be finite, got {config.frequency_offset}")

    if config.phase_offset is not None:
        if not isinstance(config.phase_offset, (int, float, np.number)):
            raise TypeError(f"phase_offset must be numeric, got {type(config.phase_offset)}")
        if not np.isfinite(config.phase_offset):
            raise ValueError(f"phase_offset must be finite, got {config.phase_offset}")

    if config.timing_offset is not None:
        if not isinstance(config.timing_offset, (int, float, np.number)):
            raise TypeError(f"timing_offset must be numeric, got {type(config.timing_offset)}")
        if not np.isfinite(config.timing_offset):
            raise ValueError(f"timing_offset must be finite, got {config.timing_offset}")

    # 6. Validate Pulse Shaping Parameters
    if config.rolloff is not None:
        if not isinstance(config.rolloff, (int, float, np.number)):
            raise TypeError(f"rolloff must be numeric, got {type(config.rolloff)}")
        if not (0.0 <= config.rolloff <= 1.0):
            raise ValueError(f"rolloff must satisfy 0 <= rolloff <= 1, got {config.rolloff}")

    if config.filter_span_symbols is not None:
        if not isinstance(config.filter_span_symbols, (int, np.integer)):
            raise TypeError(f"filter_span_symbols must be an integer, got {type(config.filter_span_symbols)}")
        if config.filter_span_symbols <= 0:
            raise ValueError(f"filter_span_symbols must be a positive integer, got {config.filter_span_symbols}")

    if config.dc_offset_i is not None:
        if not isinstance(config.dc_offset_i, (int, float, np.number)):
            raise TypeError(f"dc_offset_i must be numeric, got {type(config.dc_offset_i)}")
        if not np.isfinite(config.dc_offset_i):
            raise ValueError(f"dc_offset_i must be finite, got {config.dc_offset_i}")

    if config.dc_offset_q is not None:
        if not isinstance(config.dc_offset_q, (int, float, np.number)):
            raise TypeError(f"dc_offset_q must be numeric, got {type(config.dc_offset_q)}")
        if not np.isfinite(config.dc_offset_q):
            raise ValueError(f"dc_offset_q must be finite, got {config.dc_offset_q}")

    if config.iq_amplitude_imbalance is not None:
        if not isinstance(config.iq_amplitude_imbalance, (int, float, np.number)):
            raise TypeError(f"iq_amplitude_imbalance must be numeric, got {type(config.iq_amplitude_imbalance)}")
        if not np.isfinite(config.iq_amplitude_imbalance):
            raise ValueError(f"iq_amplitude_imbalance must be finite, got {config.iq_amplitude_imbalance}")

    if config.iq_phase_imbalance is not None:
        if not isinstance(config.iq_phase_imbalance, (int, float, np.number)):
            raise TypeError(f"iq_phase_imbalance must be numeric, got {type(config.iq_phase_imbalance)}")
        if not np.isfinite(config.iq_phase_imbalance):
            raise ValueError(f"iq_phase_imbalance must be finite, got {config.iq_phase_imbalance}")
