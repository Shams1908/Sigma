"""
DSP Performance Benchmark Suite — SIGMA P2 Stage

Measures execution time, scaling behavior, and stage breakdown across
deterministic synthetic IQ signals for the end-to-end DSP pipeline.
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
from preprocessing.normalize import normalize_signal
from dsp.fft import canonical_to_complex, compute_fft
from dsp.psd import estimate_psd
from dsp.spectrogram import compute_spectrogram
from dsp.detection import detect_signal_regions
from dsp.extraction import extract_signal
from dsp.visualization import generate_visualization_data, to_json_serializable
from dsp.pipeline import run_dsp_pipeline, DSPPipelineConfig, DSPPipelineResult
from dsp import estimate_region_parameters


def generate_benchmark_signal(
    signal_type: str,
    n_samples: int,
    sample_rate: float = 100000.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate deterministic synthetic IQ signal [2, N].
    Types: 'single_tone', 'multi_tone', 'noise_only', 'zero_signal'
    """
    np.random.seed(seed)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate

    if signal_type == "single_tone":
        tone = 1.0 * np.exp(1j * 2.0 * np.pi * 10000.0 * t)
        noise = np.random.normal(0, 0.05, n_samples) + 1j * np.random.normal(0, 0.05, n_samples)
        complex_iq = tone + noise
    elif signal_type == "multi_tone":
        tone_a = 0.8 * np.exp(1j * 2.0 * np.pi * (-15000.0) * t)
        tone_b = 1.0 * np.exp(1j * 2.0 * np.pi * 20000.0 * t)
        noise = np.random.normal(0, 0.05, n_samples) + 1j * np.random.normal(0, 0.05, n_samples)
        complex_iq = tone_a + tone_b + noise
    elif signal_type == "noise_only":
        complex_iq = np.random.normal(0, 0.05, n_samples) + 1j * np.random.normal(0, 0.05, n_samples)
    elif signal_type == "zero_signal":
        complex_iq = np.zeros(n_samples, dtype=np.complex64)
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

    return np.array([complex_iq.real, complex_iq.imag], dtype=np.float32)


def profile_pipeline_stages(
    iq: np.ndarray,
    sample_rate: float,
    config: DSPPipelineConfig,
) -> Tuple[DSPPipelineResult, Dict[str, float]]:
    """
    Detailed timing breakdown of each pipeline stage using high-resolution perf_counter.
    Returns (DSPPipelineResult, stage_times_sec_dict).
    """
    timings: Dict[str, float] = {}

    # Stage 0: Input validation & prep
    t0 = time.perf_counter()
    iq_canonical = np.asarray(iq, dtype=np.float32)
    timings["0_validation_prep"] = time.perf_counter() - t0

    # Stage 1: Normalization
    t0 = time.perf_counter()
    if config.enable_normalization:
        iq_norm = normalize_signal(iq_canonical)
    else:
        iq_norm = iq_canonical.copy()
    timings["1_normalization"] = time.perf_counter() - t0

    iq_complex = canonical_to_complex(iq_norm)

    # Stage 2: FFT
    t0 = time.perf_counter()
    fft_data = None
    if config.enable_fft:
        freqs_f, fft_db = compute_fft(iq_complex, sample_rate)
        fft_data = (freqs_f, fft_db)
    timings["2_fft"] = time.perf_counter() - t0

    # Stage 3: PSD
    t0 = time.perf_counter()
    psd_data = None
    if config.enable_psd:
        freqs_p, psd_db = estimate_psd(iq_complex, sample_rate)
        psd_data = (freqs_p, psd_db)
    timings["3_psd"] = time.perf_counter() - t0

    # Stage 4: STFT
    t0 = time.perf_counter()
    stft_data = None
    if config.enable_stft:
        freqs_s, times_s, stft_db = compute_spectrogram(iq_complex, sample_rate)
        stft_data = (times_s, freqs_s, stft_db)
    timings["4_stft"] = time.perf_counter() - t0

    # Stage 5: Signal Detection
    t0 = time.perf_counter()
    regions = detect_signal_regions(
        iq_norm,
        sample_rate,
        threshold_db=config.threshold_db,
        min_bins=config.min_bins,
        merge_gap_bins=config.merge_gap_bins,
    )
    timings["5_signal_detection"] = time.perf_counter() - t0

    # Stage 6: Parameter Estimation & Extraction
    t0 = time.perf_counter()
    region_results = []
    for region in regions:
        params = estimate_region_parameters(iq_norm, sample_rate, region)
        extracted = None
        if config.extract_detected_signals:
            extracted = extract_signal(iq_norm, sample_rate, region=region)
        region_results.append(type("RegionResultObj", (), {
            "region": region, "parameters": params, "extracted_signal": extracted
        }))
    timings["6_param_and_extraction"] = time.perf_counter() - t0

    # Stage 7: Visualization Generation
    t0 = time.perf_counter()
    vis_data = None
    if config.generate_visualization:
        vis_data = generate_visualization_data(
            iq=iq_norm,
            sample_rate=sample_rate,
            include_waveform=True,
            include_fft=config.enable_fft,
            include_psd=config.enable_psd,
            include_spectrogram=config.enable_stft,
            regions=regions,
            params=region_results[0].parameters if len(region_results) == 1 else None,
            max_points=config.max_visualization_points,
        )
    timings["7_visualization"] = time.perf_counter() - t0

    res = run_dsp_pipeline(iq, sample_rate, config=config)
    return res, timings


def run_all_benchmarks() -> Dict[str, Any]:
    """Execute complete performance benchmark matrix and report measurements."""
    results: Dict[str, Any] = {}
    sample_rate = 100000.0
    sizes = [10000, 100000, 500000]

    print("=" * 75)
    print("        SIGMA P2 DSP PERFORMANCE BENCHMARK & BOTTLENECK ANALYSIS")
    print("=" * 75)

    # ──────────────────────────────────────────────────────────────────────────
    # SCENARIO 1: SINGLE SIGNAL SCALING (10k, 100k, 500k)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- SCENARIO 1: Single Signal Scaling ---")
    s1_res = {}
    for n in sizes:
        iq = generate_benchmark_signal("single_tone", n, sample_rate=sample_rate)
        config = DSPPipelineConfig(enable_stft=False)

        # Warmup
        run_dsp_pipeline(iq, sample_rate, config)

        # Measure 5 runs
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            res = run_dsp_pipeline(iq, sample_rate, config)
            times.append(time.perf_counter() - t0)

        avg_t_ms = (sum(times) / len(times)) * 1000.0
        min_t_ms = min(times) * 1000.0
        _, breakdown = profile_pipeline_stages(iq, sample_rate, config)

        s1_res[str(n)] = {
            "avg_ms": avg_t_ms,
            "min_ms": min_t_ms,
            "num_regions": len(res.detected_regions),
            "breakdown_ms": {k: v * 1000.0 for k, v in breakdown.items()},
        }

        print(f"  N = {n:6d} samples | Min Time: {min_t_ms:6.2f} ms | Avg Time: {avg_t_ms:6.2f} ms | Regions: {len(res.detected_regions)}")

    results["scenario_1_single_signal"] = s1_res

    # ──────────────────────────────────────────────────────────────────────────
    # SCENARIO 2: MULTIPLE SIGNALS SCALING (10k, 100k, 500k)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- SCENARIO 2: Multiple Signals Scaling ---")
    s2_res = {}
    for n in sizes:
        iq = generate_benchmark_signal("multi_tone", n, sample_rate=sample_rate)
        config = DSPPipelineConfig(enable_stft=False)

        # Warmup
        run_dsp_pipeline(iq, sample_rate, config)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            res = run_dsp_pipeline(iq, sample_rate, config)
            times.append(time.perf_counter() - t0)

        avg_t_ms = (sum(times) / len(times)) * 1000.0
        min_t_ms = min(times) * 1000.0
        _, breakdown = profile_pipeline_stages(iq, sample_rate, config)

        s2_res[str(n)] = {
            "avg_ms": avg_t_ms,
            "min_ms": min_t_ms,
            "num_regions": len(res.detected_regions),
            "breakdown_ms": {k: v * 1000.0 for k, v in breakdown.items()},
        }

        print(f"  N = {n:6d} samples | Min Time: {min_t_ms:6.2f} ms | Avg Time: {avg_t_ms:6.2f} ms | Regions: {len(res.detected_regions)}")

    results["scenario_2_multi_signal"] = s2_res

    # ──────────────────────────────────────────────────────────────────────────
    # SCENARIO 3 & 4: NOISE ONLY & ZERO SIGNAL (100k)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- SCENARIOS 3 & 4: Edge Cases (N = 100,000) ---")
    for stype in ["noise_only", "zero_signal"]:
        iq = generate_benchmark_signal(stype, 100000, sample_rate=sample_rate)
        config = DSPPipelineConfig(enable_stft=False)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            res = run_dsp_pipeline(iq, sample_rate, config)
            times.append(time.perf_counter() - t0)

        avg_t_ms = (sum(times) / len(times)) * 1000.0
        print(f"  Type: {stype:12s} | Avg Time: {avg_t_ms:6.2f} ms | Regions Detected: {len(res.detected_regions)}")
        results[f"scenario_{stype}"] = {"avg_ms": avg_t_ms, "num_regions": len(res.detected_regions)}

    # ──────────────────────────────────────────────────────────────────────────
    # SCENARIO 5: STFT ENABLED vs DISABLED COMPARISON (N = 100,000)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- SCENARIO 5: STFT Performance Impact (N = 100,000) ---")
    iq = generate_benchmark_signal("single_tone", 100000, sample_rate=sample_rate)

    cfg_no_stft = DSPPipelineConfig(enable_stft=False)
    t0 = time.perf_counter()
    r_no = run_dsp_pipeline(iq, sample_rate, cfg_no_stft)
    t_no_ms = (time.perf_counter() - t0) * 1000.0

    cfg_stft = DSPPipelineConfig(enable_stft=True)
    t0 = time.perf_counter()
    r_stft = run_dsp_pipeline(iq, sample_rate, cfg_stft)
    t_stft_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  STFT Disabled: {t_no_ms:6.2f} ms")
    print(f"  STFT Enabled:  {t_stft_ms:6.2f} ms | Over-head: {t_stft_ms - t_no_ms:6.2f} ms (+{((t_stft_ms/t_no_ms)-1)*100:.1f}%)")

    results["scenario_stft_comparison"] = {
        "disabled_ms": t_no_ms,
        "enabled_ms": t_stft_ms,
        "overhead_ms": t_stft_ms - t_no_ms,
    }

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE BREAKDOWN TABLE (N = 100,000 Single Tone)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- DETAILED STAGE TIMING BREAKDOWN (N = 100,000 Single Tone) ---")
    iq_100k = generate_benchmark_signal("single_tone", 100000, sample_rate=sample_rate)
    cfg_full = DSPPipelineConfig(enable_stft=True)
    _, stage_b = profile_pipeline_stages(iq_100k, sample_rate, cfg_full)

    tot_s = sum(stage_b.values())
    for stage_name, s_sec in stage_b.items():
        s_ms = s_sec * 1000.0
        pct = (s_sec / tot_s) * 100.0 if tot_s > 0 else 0.0
        print(f"  {stage_name:25s}: {s_ms:6.2f} ms ({pct:5.1f}%)")

    print(f"  {'TOTAL PIPELINE':25s}: {tot_s * 1000.0:6.2f} ms (100.0%)")
    print("=" * 75)

    return results


if __name__ == "__main__":
    run_all_benchmarks()
