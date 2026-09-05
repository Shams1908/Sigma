from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import math
import random

router = APIRouter()

class SpectrumPoint(BaseModel):
    frequency: float
    magnitudeDb: float

class ConstellationPoint(BaseModel):
    i: float
    q: float

class AnalysisResponse(BaseModel):
    spectrum: List[SpectrumPoint]
    waterfall: List[List[float]]
    constellation: List[ConstellationPoint]

def generate_mock_spectrum(center_freq: float, sample_rate: float, bins: int = 512) -> List[SpectrumPoint]:
    spectrum = []
    freq_start = center_freq - sample_rate / 2
    freq_step = sample_rate / bins
    
    noise_floor = -80.0
    signal_center = center_freq + 18750
    signal_bw = 205000
    signal_peak = -30.0
    
    for i in range(bins):
        freq = freq_start + i * freq_step
        dist_from_center = abs(freq - signal_center)
        
        if dist_from_center < signal_bw / 2:
            gaussian_factor = math.exp(-0.5 * (dist_from_center / (signal_bw / 4)) ** 2)
            magnitude = noise_floor + (signal_peak - noise_floor) * gaussian_factor
        else:
            magnitude = noise_floor
        
        magnitude += random.gauss(0, 1.5)
        spectrum.append(SpectrumPoint(frequency=freq, magnitudeDb=magnitude))
    
    return spectrum

def generate_mock_waterfall(bins: int = 256, rows: int = 64) -> List[List[float]]:
    waterfall = []
    noise_floor = -80.0
    signal_peak = -30.0
    
    for row in range(rows):
        time_drift = math.sin(row * 0.15) * 0.1
        fade = 0.85 + 0.15 * math.sin(row * 0.08)
        
        row_data = []
        for col in range(bins):
            center_bin = bins / 2 + time_drift * bins / 4
            dist = abs(col - center_bin)
            
            if dist < bins / 8:
                gaussian = math.exp(-0.5 * (dist / (bins / 16)) ** 2)
                value = noise_floor + (signal_peak - noise_floor) * gaussian * fade
            else:
                value = noise_floor
            
            value += random.gauss(0, 2.0)
            row_data.append(value)
        
        waterfall.append(row_data)
    
    return waterfall

def generate_mock_constellation(modulation: str = "QPSK", points: int = 200) -> List[ConstellationPoint]:
    constellation = []
    
    if modulation == "QPSK":
        centers = [
            (1/math.sqrt(2), 1/math.sqrt(2)),
            (-1/math.sqrt(2), 1/math.sqrt(2)),
            (-1/math.sqrt(2), -1/math.sqrt(2)),
            (1/math.sqrt(2), -1/math.sqrt(2))
        ]
    elif modulation == "BPSK":
        centers = [(1.0, 0.0), (-1.0, 0.0)]
    else:
        centers = [(1/math.sqrt(2), 1/math.sqrt(2))]
    
    noise_std = 0.08
    
    for _ in range(points):
        center = random.choice(centers)
        i_val = center[0] + random.gauss(0, noise_std)
        q_val = center[1] + random.gauss(0, noise_std)
        constellation.append(ConstellationPoint(i=i_val, q=q_val))
    
    return constellation

@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_data(analysis_id: str):
    center_freq = 915e6
    sample_rate = 2.4e6
    
    spectrum = generate_mock_spectrum(center_freq, sample_rate)
    waterfall = generate_mock_waterfall()
    constellation = generate_mock_constellation("QPSK")
    
    return AnalysisResponse(
        spectrum=spectrum,
        waterfall=waterfall,
        constellation=constellation
    )
