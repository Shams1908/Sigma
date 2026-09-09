"""
GET /api/v1/visualizations/{signal_id}/spectrum
GET /api/v1/visualizations/{signal_id}/waterfall  
GET /api/v1/visualizations/{signal_id}/constellation

Real-time visualization endpoints that compute DSP data directly from uploaded IQ files.
These endpoints work IMMEDIATELY after upload without waiting for full analysis completion.
"""
from __future__ import annotations

import logging
import asyncio
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from db.init import is_db_connected

logger = logging.getLogger(__name__)
router = APIRouter()


async def _get_signal_path(signal_id: str) -> str:
    """Resolve signal_id to storage path from DB or upload store."""
    if is_db_connected():
        try:
            from beanie import PydanticObjectId
            from db.models import Signal
            
            doc = await Signal.get(PydanticObjectId(signal_id))
            if doc and doc.storage_path:
                return doc.storage_path
        except Exception:
            pass
    
    from api.upload import _UPLOAD_STORE
    path = _UPLOAD_STORE.get(signal_id)
    if not path:
        raise HTTPException(404, f"Signal '{signal_id}' not found")
    return path


def _load_iq_data(storage_path: str) -> tuple[np.ndarray, float]:
    """Load IQ data from file and return [2, N] array + sample rate."""
    from ml.input.pipeline import process_file
    from ml.input.types import PipelineConfig
    
    config = PipelineConfig()
    segments, meta = process_file(storage_path, config)
    
    if meta.validation_status == "ERROR" or len(segments) == 0:
        raise HTTPException(422, f"Invalid signal file: {meta.error_message}")
    
    iq_full = segments.reshape(2, -1)
    sample_rate = meta.sample_rate or 1.0
    
    return iq_full, sample_rate


@router.get("/{signal_id}/spectrum")
async def get_spectrum(signal_id: str):
    """Compute FFT spectrum from uploaded IQ file."""
    storage_path = await _get_signal_path(signal_id)
    
    loop = asyncio.get_event_loop()
    iq_data, sample_rate = await loop.run_in_executor(None, _load_iq_data, storage_path)
    
    from dsp.psd import estimate_psd
    
    # Convert [2, N] to complex array
    iq_complex = iq_data[0, :] + 1j * iq_data[1, :]
    
    freqs, psd_db = await loop.run_in_executor(
        None, 
        lambda: estimate_psd(iq_complex, sample_rate)
    )
    
    spectrum_data = [
        {"frequency": float(f), "magnitudeDb": float(p)}
        for f, p in zip(freqs, psd_db)
    ]
    
    return {
        "data": spectrum_data,
        "sampleRate": float(sample_rate),
        "centerFrequency": 0.0
    }


@router.get("/{signal_id}/waterfall")
async def get_waterfall(signal_id: str):
    """Compute spectrogram/waterfall from uploaded IQ file."""
    storage_path = await _get_signal_path(signal_id)
    
    loop = asyncio.get_event_loop()
    iq_data, sample_rate = await loop.run_in_executor(None, _load_iq_data, storage_path)
    
    from dsp.spectrogram import compute_spectrogram
    
    # Convert [2, N] to complex array
    iq_complex = iq_data[0, :] + 1j * iq_data[1, :]
    
    # Limit data size for faster transfer - use only first 100k samples max
    if len(iq_complex) > 100000:
        iq_complex = iq_complex[:100000]
    
    freqs, times, Sxx = await loop.run_in_executor(
        None,
        lambda: compute_spectrogram(iq_complex, sample_rate, nperseg=128)
    )
    
    # Convert to list with limited precision
    waterfall_matrix = [[round(float(val), 1) for val in row] for row in Sxx.tolist()]
    
    return {
        "data": waterfall_matrix,
        "timeRange": [float(times[0]), float(times[-1])],
        "frequencyRange": [float(freqs[0]), float(freqs[-1])]
    }


@router.get("/{signal_id}/constellation")
async def get_constellation(signal_id: str):
    """Extract IQ constellation points from uploaded file."""
    storage_path = await _get_signal_path(signal_id)
    
    loop = asyncio.get_event_loop()
    iq_data, sample_rate = await loop.run_in_executor(None, _load_iq_data, storage_path)
    
    i_vals = iq_data[0, ::100][:1000]
    q_vals = iq_data[1, ::100][:1000]
    
    constellation_points = [
        {"i": float(i), "q": float(q)}
        for i, q in zip(i_vals, q_vals)
    ]
    
    return {
        "data": constellation_points,
        "modulation": None
    }


@router.get("/{signal_id}/parameters")
async def get_parameters(signal_id: str):
    """Extract real DSP parameters from uploaded IQ file (SNR, bandwidth, etc.)."""
    storage_path = await _get_signal_path(signal_id)
    
    loop = asyncio.get_event_loop()
    iq_data, sample_rate = await loop.run_in_executor(None, _load_iq_data, storage_path)
    
    from dsp import extract_parameters
    
    params = await loop.run_in_executor(
        None,
        lambda: extract_parameters(iq_data, sample_rate)
    )
    
    return {
        "carrierFrequency": 0.0,  # Unknown without metadata
        "sampleRate": float(sample_rate),
        "bandwidth": float(params.get("bandwidth", 0.0)),
        "snr": float(params.get("snr", 0.0)),
        "symbolRate": float(params.get("symbol_rate_estimate", 0.0))
    }


@router.get("/{signal_id}/waveform")
async def get_waveform(signal_id: str):
    """Extract real I/Q time-domain samples from uploaded file."""
    storage_path = await _get_signal_path(signal_id)
    
    loop = asyncio.get_event_loop()
    iq_data, sample_rate = await loop.run_in_executor(None, _load_iq_data, storage_path)
    
    decimation = max(1, iq_data.shape[1] // 10000)
    i_samples = iq_data[0, ::decimation][:10000]
    q_samples = iq_data[1, ::decimation][:10000]
    
    time_axis = np.arange(len(i_samples)) * (decimation / sample_rate)
    
    return {
        "i": i_samples.tolist(),
        "q": q_samples.tolist(),
        "time": time_axis.tolist(),
        "sampleRate": float(sample_rate),
        "decimation": int(decimation)
    }
