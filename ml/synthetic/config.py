import numpy as np
from typing import Dict, List, Any

# 1. Modulation Support Registration
SUPPORTED_MODULATION_CLASSES: List[str] = [
    "BPSK",
    "QPSK",
    "8PSK",
    "QAM16",
    "QAM64"
]

UNSUPPORTED_MODULATION_CLASSES: List[str] = [
    "AM-DSB",
    "AM-SSB",
    "CPFSK",
    "GFSK",
    "PAM4",
    "WBFM"
]

# 2. Base Waveform Parameters
# Designed to yield exactly 128 samples: output_length = (num_symbols + filter_span_symbols) * samples_per_symbol
# N = (8 + 8) * 8 = 16 * 8 = 128
NOMINAL_CONFIG: Dict[str, Any] = {
    "num_symbols": 8,
    "sample_rate": 800000.0,
    "symbol_rate": 100000.0,
    "samples_per_symbol": 8,
    "filter_span_symbols": 8,
    "rolloff": 0.35
}

# 3. Impairment Sweep Configurations
# Each experiment sweeps ONE parameter at a time. All other parameters are held at 0/None.
SWEEP_EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "awgn": {
        "sweep_parameter": "snr",
        "sweep_values": [-20.0, -10.0, 0.0, 10.0, 18.0],
        "defaults": {} # all impairments are 0 (None)
    },
    "frequency_offset": {
        "sweep_parameter": "frequency_offset",
        "sweep_values": [0.0, 100.0, 500.0, 2000.0, 10000.0], # in Hz (for Fs = 8e5)
        "defaults": {"snr": 18.0}
    },
    "phase_offset": {
        "sweep_parameter": "phase_offset",
        "sweep_values": [0.0, np.pi / 8.0, np.pi / 4.0, np.pi / 2.0, np.pi], # in radians
        "defaults": {"snr": 18.0}
    },
    "iq_amplitude_imbalance": {
        "sweep_parameter": "iq_amplitude_imbalance",
        "sweep_values": [0.0, 0.05, 0.1, 0.2, 0.3], # amplitude gain imbalance factor A
        "defaults": {"snr": 18.0}
    },
    "iq_phase_imbalance": {
        "sweep_parameter": "iq_phase_imbalance",
        "sweep_values": [0.0, 0.05, 0.1, 0.2, 0.3], # phase skew in radians
        "defaults": {"snr": 18.0}
    },
    "dc_offset": {
        "sweep_parameter": "dc_offset", # special handler will apply dc_offset_i and dc_offset_q equally
        "sweep_values": [0.0, 0.05, 0.1, 0.2, 0.3], # DC offset level added to I & Q
        "defaults": {"snr": 18.0}
    },
    "timing_offset": {
        "sweep_parameter": "timing_offset",
        "sweep_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], # fractional timing offset in samples
        "defaults": {"snr": 18.0}
    }
}
