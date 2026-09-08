"""
DSP visualization data generation module.

Prepares structured, JSON-serializable Python data structures from DSP outputs
for consumption by API response handlers and frontend charts.
Does not depend on React or frontend-specific frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, asdict
from typing import Any
import numpy as np

from preprocessing.normalize import validate_canonical_iq
from dsp.fft import canonical_to_complex, compute_fft
from dsp.psd import estimate_psd
from dsp.spectrogram import compute_spectrogram
from dsp.detection import SignalRegion


def to_json_serializable(obj: Any) -> Any:
    """
    Recursively converts NumPy numeric types, arrays, dataclasses, and dictionaries
    into standard JSON-serializable Python types (int, float, list, dict, None).
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, str, int, float)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [to_json_serializable(item) for item in obj.tolist()]
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_json_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    return str(obj)


def downsample_for_visualization(data: np.ndarray, max_points: int = 1000) -> np.ndarray:
    """
    Downsample a 1D NumPy array for visualization using deterministic decimation.
    Preserves first and last points where practical. Does not mutate original array.
    """
    data = np.asarray(data)
    n = len(data)
    if n <= max_points or max_points <= 0:
        return data.copy()

    step = int(np.ceil(n / max_points))
    indices = np.arange(0, n, step)
    if indices[-1] != n - 1 and len(indices) < max_points:
        indices = np.append(indices, n - 1)

    return data[indices]


def generate_waveform_data(
    iq: np.ndarray,
    sample_rate: float,
    max_points: int = 1000,
) -> dict:
    """
    Generate time-domain waveform visualization data (I, Q, Time).

    Args:
        iq:          Canonical IQ array [2, N] or 1D complex array.
        sample_rate: Sample rate in Hz.
        max_points:  Maximum sample points returned for visualization.

    Returns:
        JSON-serializable dict: {"time": [...], "i": [...], "q": [...], "sample_rate": float, "decimation": int}
    """
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")

    if isinstance(iq, np.ndarray) and iq.ndim == 2:
        validate_canonical_iq(iq)
        i_full = iq[0]
        q_full = iq[1]
    elif isinstance(iq, np.ndarray) and iq.ndim == 1:
        if not np.all(np.isfinite(iq)):
            raise ValueError("Input IQ array contains non-finite values (NaN or Inf)")
        complex_iq = np.asarray(iq, dtype=np.complex64)
        i_full = complex_iq.real
        q_full = complex_iq.imag
    elif not isinstance(iq, np.ndarray):
        raise TypeError(f"Input must be a NumPy ndarray, got {type(iq).__name__}")
    else:
        raise ValueError(f"Input array must be 1D complex or 2D canonical IQ [2, N], got shape {iq.shape}")

    n = len(i_full)
    if n == 0:
        return {
            "time": [],
            "i": [],
            "q": [],
            "sample_rate": float(sample_rate),
            "decimation": 1,
        }

    decimation = max(1, n // max_points) if max_points > 0 else 1
    i_sub = i_full[::decimation]
    q_sub = q_full[::decimation]
    t_sub = np.arange(len(i_sub)) * (decimation / sample_rate)

    return to_json_serializable({
        "time": t_sub,
        "i": i_sub,
        "q": q_sub,
        "sample_rate": float(sample_rate),
        "decimation": int(decimation),
    })


def generate_fft_data(
    iq: np.ndarray,
    sample_rate: float,
    n_fft: int | None = None,
    max_points: int = 1000,
) -> dict:
    """
    Generate FFT spectrum visualization data.

    Returns:
        JSON-serializable dict: {"frequency": [...], "magnitude_db": [...], "unit": "dB", "sample_rate": float}
    """
    complex_iq = canonical_to_complex(iq) if isinstance(iq, np.ndarray) and iq.ndim == 2 else np.asarray(iq, dtype=np.complex64)
    freqs, spec_db = compute_fft(complex_iq, sample_rate, n_fft=n_fft)

    if max_points > 0 and len(freqs) > max_points:
        step = int(np.ceil(len(freqs) / max_points))
        freqs = freqs[::step]
        spec_db = spec_db[::step]

    return to_json_serializable({
        "frequency": freqs,
        "magnitude_db": spec_db,
        "unit": "dB",
        "sample_rate": float(sample_rate),
    })


def generate_psd_data(
    iq: np.ndarray,
    sample_rate: float,
    nperseg: int = 256,
    max_points: int = 1000,
) -> dict:
    """
    Generate PSD spectrum visualization data.

    Returns:
        JSON-serializable dict: {"frequency": [...], "power_db": [...], "unit": "dBW/Hz", "sample_rate": float}
    """
    complex_iq = canonical_to_complex(iq) if isinstance(iq, np.ndarray) and iq.ndim == 2 else np.asarray(iq, dtype=np.complex64)
    freqs, psd_db = estimate_psd(complex_iq, sample_rate, nperseg=nperseg)

    if max_points > 0 and len(freqs) > max_points:
        step = int(np.ceil(len(freqs) / max_points))
        freqs = freqs[::step]
        psd_db = psd_db[::step]

    return to_json_serializable({
        "frequency": freqs,
        "power_db": psd_db,
        "unit": "dBW/Hz",
        "sample_rate": float(sample_rate),
    })


def generate_spectrogram_data(
    iq: np.ndarray,
    sample_rate: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    max_time_points: int = 200,
    max_freq_points: int = 200,
) -> dict:
    """
    Generate STFT / Spectrogram matrix visualization data.

    Returns:
        JSON-serializable dict: {"frequency": [...], "time": [...], "spectrogram_db": [[...], ...]}
    """
    complex_iq = canonical_to_complex(iq) if isinstance(iq, np.ndarray) and iq.ndim == 2 else np.asarray(iq, dtype=np.complex64)
    freqs, times, Sxx_db = compute_spectrogram(complex_iq, sample_rate, nperseg=nperseg, noverlap=noverlap)

    if max_freq_points > 0 and len(freqs) > max_freq_points:
        f_step = int(np.ceil(len(freqs) / max_freq_points))
        freqs = freqs[::f_step]
        Sxx_db = Sxx_db[::f_step, :]

    if max_time_points > 0 and len(times) > max_time_points:
        t_step = int(np.ceil(len(times) / max_time_points))
        times = times[::t_step]
        Sxx_db = Sxx_db[:, ::t_step]

    return to_json_serializable({
        "frequency": freqs,
        "time": times,
        "spectrogram_db": Sxx_db,
        "unit": "dBW/Hz",
        "sample_rate": float(sample_rate),
    })



def generate_detection_data(regions: list[SignalRegion]) -> list[dict]:
    """
    Convert a list of SignalRegion objects into JSON-serializable visualization dicts.
    Preserves exact region attributes without inventing values.
    """
    return to_json_serializable([
        {
            "start_frequency": float(r.start_frequency),
            "end_frequency": float(r.end_frequency),
            "center_frequency": float(r.center_frequency),
            "bandwidth": float(r.bandwidth),
            "peak_power_db": float(r.peak_power_db),
            "average_power_db": float(r.average_power_db),
            "snr_estimate_db": float(r.snr_estimate_db),
        }
        for r in regions
    ])


def generate_parameter_data(params: dict) -> dict:
    """
    Format DSP parameter estimates into a JSON-serializable visualization dict.
    Preserves None values honestly without replacing with fake numbers or 0.
    """
    snr_val = params.get("snr")
    sr_val = params.get("symbol_rate_estimate")

    return to_json_serializable({
        "snr": float(snr_val) if snr_val is not None else None,
        "bandwidth": float(params.get("bandwidth", 0.0)),
        "carrier_offset": float(params.get("carrier_offset", 0.0)),
        "symbol_rate": float(sr_val) if sr_val is not None else None,
    })


def generate_visualization_data(
    iq: np.ndarray,
    sample_rate: float,
    include_waveform: bool = True,
    include_fft: bool = True,
    include_psd: bool = True,
    include_spectrogram: bool = False,
    regions: list[SignalRegion] | None = None,
    params: dict | None = None,
    max_points: int = 1000,
    fft_data: tuple[np.ndarray, np.ndarray] | None = None,
    psd_data: tuple[np.ndarray, np.ndarray] | None = None,
    stft_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict:
    """
    Selective composite generator producing structured, JSON-serializable visualization data.
    Runs expensive computations (such as STFT spectrograms) only when explicitly requested.
    Reuses pre-computed FFT, PSD, or STFT arrays if provided to prevent duplicate computation.
    """
    out: dict = {}

    if include_waveform:
        out["waveform"] = generate_waveform_data(iq, sample_rate, max_points=max_points)

    if include_fft:
        if fft_data is not None and len(fft_data) == 2:
            freqs, spec_db = fft_data
            if max_points > 0 and len(freqs) > max_points:
                step = int(np.ceil(len(freqs) / max_points))
                freqs = freqs[::step]
                spec_db = spec_db[::step]
            out["fft"] = to_json_serializable({
                "frequency": freqs,
                "magnitude_db": spec_db,
                "unit": "dB",
                "sample_rate": float(sample_rate),
            })
        else:
            out["fft"] = generate_fft_data(iq, sample_rate, max_points=max_points)

    if include_psd:
        if psd_data is not None and len(psd_data) == 2:
            freqs, psd_db = psd_data
            if max_points > 0 and len(freqs) > max_points:
                step = int(np.ceil(len(freqs) / max_points))
                freqs = freqs[::step]
                psd_db = psd_db[::step]
            out["psd"] = to_json_serializable({
                "frequency": freqs,
                "power_db": psd_db,
                "unit": "dBW/Hz",
                "sample_rate": float(sample_rate),
            })
        else:
            out["psd"] = generate_psd_data(iq, sample_rate, max_points=max_points)

    if include_spectrogram:
        if stft_data is not None and len(stft_data) == 3:
            times, freqs, Sxx_db = stft_data
            max_time_points = 200
            max_freq_points = 200
            if max_freq_points > 0 and len(freqs) > max_freq_points:
                f_step = int(np.ceil(len(freqs) / max_freq_points))
                freqs = freqs[::f_step]
                Sxx_db = Sxx_db[::f_step, :]
            if max_time_points > 0 and len(times) > max_time_points:
                t_step = int(np.ceil(len(times) / max_time_points))
                times = times[::t_step]
                Sxx_db = Sxx_db[:, ::t_step]
            out["spectrogram"] = to_json_serializable({
                "frequency": freqs,
                "time": times,
                "spectrogram_db": Sxx_db,
                "unit": "dBW/Hz",
                "sample_rate": float(sample_rate),
            })
        else:
            out["spectrogram"] = generate_spectrogram_data(iq, sample_rate)

    if regions is not None:
        out["detected_regions"] = generate_detection_data(regions)

    if params is not None:
        out["parameters"] = generate_parameter_data(params)

    return out
