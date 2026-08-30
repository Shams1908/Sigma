from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

@dataclass(frozen=True)
class GeneratorConfig:
    """
    Typed configuration object for synthetic signal generation.
    
    Validates parameters immediately upon construction to enforce physical 
    and mathematical consistency.
    """
    modulation: str
    num_symbols: int
    sample_rate: float
    symbol_rate: float
    samples_per_symbol: int
    random_seed: int
    snr: Optional[float] = None
    frequency_offset: Optional[float] = None
    phase_offset: Optional[float] = None
    timing_offset: Optional[float] = None
    rolloff: Optional[float] = None
    filter_span_symbols: Optional[int] = None
    dc_offset_i: Optional[float] = None
    dc_offset_q: Optional[float] = None
    iq_amplitude_imbalance: Optional[float] = None
    iq_phase_imbalance: Optional[float] = None

    def __post_init__(self):
        # Normalize modulation input strings to standard M1 labels (one-way compatibility mapping)
        normalized = self.modulation
        if normalized == "16QAM":
            normalized = "QAM16"
        elif normalized == "64QAM":
            normalized = "QAM64"
            
        object.__setattr__(self, "modulation", normalized)
        
        from ml.generators.validation import validate_config
        validate_config(self)
