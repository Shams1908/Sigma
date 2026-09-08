"""
Reproducible Real-vs-Synthetic Domain-Gap Statistical Audit for SIGMA M6.

Quantifies discrepancies between synthetic training signals and real-world IQ recordings
across 11 structural and statistical dimensions.
"""
import os
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.stats import skew, kurtosis

from ml.dataset.loader import RadioMLDataset, DatasetCache
from ml.dataset.external_dataset import load_external_dataset, slice_frames_to_windows
from ml.synthetic.dataset import load_synthetic_dataset


def compute_comprehensive_statistics(X: np.ndarray) -> Dict[str, float]:
    """
    Computes statistical and spectral metrics for a batch of IQ signals [N, 2, L].
    
    Metrics:
      1. Amplitude distribution: mean, std, skewness, kurtosis, PAPR
      2. Power & RMS distribution: complex RMS, mean power, power variance
      3. Phase behavior: phase variance, phase difference variance (dphi/dt)
      4. Instantaneous frequency: mean, std, kurtosis
      5. IQ imbalance: amplitude imbalance ratio (I_rms / Q_rms), phase orthogonality (corr(I, Q))
      6. DC offset: mean(I), mean(Q), total DC magnitude
      7. Spectral occupancy: 99% occupied bandwidth ratio, spectral flatness
      8. Temporal correlation: lag-1 autocorrelation, lag-2 autocorrelation
    """
    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(f"Expected array of shape [N, 2, L], got {X.shape}")

    N, _, L = X.shape
    I = X[:, 0, :].astype(np.float64)
    Q = X[:, 1, :].astype(np.float64)
    z = I + 1j * Q
    amp = np.abs(z)

    # 1. Amplitude Distribution & PAPR
    amp_mean = float(np.mean(amp))
    amp_std = float(np.std(amp))
    amp_skew = float(np.mean(skew(amp, axis=1, bias=False)))
    amp_kurt = float(np.mean(kurtosis(amp, axis=1, fisher=True)))
    
    peak_power = np.max(amp**2, axis=1)
    avg_power = np.mean(amp**2, axis=1) + 1e-12
    papr_db = float(np.mean(10.0 * np.log10(peak_power / avg_power)))

    # 2. Power & RMS
    complex_rms = float(np.mean(np.sqrt(avg_power)))
    signal_power = float(np.mean(avg_power))
    power_std = float(np.std(avg_power))

    # 3. Phase Dynamics & Instantaneous Frequency
    phase = np.angle(z)
    phase_var = float(np.mean(np.var(phase, axis=1)))
    
    # Phase difference (derivative), wrapped to [-pi, pi]
    dphi = np.diff(phase, axis=1)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    dphi_mean = float(np.mean(dphi))
    dphi_var = float(np.mean(np.var(dphi, axis=1)))
    dphi_kurt = float(np.mean(kurtosis(dphi, axis=1, fisher=True)))

    # 4. Transceiver IQ Imbalance
    i_rms = np.sqrt(np.mean(I**2, axis=1)) + 1e-12
    q_rms = np.sqrt(np.mean(Q**2, axis=1)) + 1e-12
    amp_imbalance_db = float(np.mean(20.0 * np.log10(i_rms / q_rms)))
    
    # Orthogonality correlation: E[I * Q] / (sigma_I * sigma_Q)
    i_std = np.std(I, axis=1) + 1e-12
    q_std = np.std(Q, axis=1) + 1e-12
    orthogonality = float(np.mean(np.mean((I - np.mean(I, axis=1, keepdims=True)) * 
                                         (Q - np.mean(Q, axis=1, keepdims=True)), axis=1) / (i_std * q_std)))

    # 5. DC Offset
    dc_i = float(np.mean(np.mean(I, axis=1)))
    dc_q = float(np.mean(np.mean(Q, axis=1)))
    total_dc = float(np.sqrt(dc_i**2 + dc_q**2))

    # 6. Spectral Occupancy & Flatness
    # FFT magnitude squared
    fft_spec = np.abs(np.fft.fft(z, axis=1))**2
    fft_spec = np.fft.fftshift(fft_spec, axes=1)
    
    # Spectral Flatness: geometric mean / arithmetic mean
    geo_mean = np.exp(np.mean(np.log(fft_spec + 1e-12), axis=1))
    ari_mean = np.mean(fft_spec, axis=1) + 1e-12
    spectral_flatness = float(np.mean(geo_mean / ari_mean))
    
    # 99% Occupied Bandwidth fraction
    total_energy = np.sum(fft_spec, axis=1, keepdims=True) + 1e-12
    cum_energy = np.cumsum(fft_spec, axis=1) / total_energy
    bw_fractions = []
    for row in cum_energy:
        low_idx = np.searchsorted(row, 0.005)
        high_idx = np.searchsorted(row, 0.995)
        bw_fractions.append((high_idx - low_idx) / float(L))
    occupied_bw_ratio = float(np.mean(bw_fractions))

    # 7. Temporal Correlation (Lag-1 and Lag-2 Autocorrelation)
    # Autocovariance normalized by variance
    lag1_corrs = []
    lag2_corrs = []
    for i in range(N):
        z_centered = z[i] - np.mean(z[i])
        var_z = np.sum(np.abs(z_centered)**2) + 1e-12
        if L > 2:
            r1 = np.real(np.sum(z_centered[1:] * np.conj(z_centered[:-1]))) / var_z
            r2 = np.real(np.sum(z_centered[2:] * np.conj(z_centered[:-2]))) / var_z
            lag1_corrs.append(r1)
            lag2_corrs.append(r2)
            
    autocorr_lag1 = float(np.mean(lag1_corrs)) if lag1_corrs else 0.0
    autocorr_lag2 = float(np.mean(lag2_corrs)) if lag2_corrs else 0.0

    return {
        "complex_rms": complex_rms,
        "signal_power": signal_power,
        "power_std": power_std,
        "amplitude_mean": amp_mean,
        "amplitude_std": amp_std,
        "amplitude_skewness": amp_skew,
        "amplitude_kurtosis": amp_kurt,
        "papr_db": papr_db,
        "phase_variance": phase_var,
        "phase_diff_variance": dphi_var,
        "inst_freq_mean": dphi_mean,
        "inst_freq_kurtosis": dphi_kurt,
        "iq_amplitude_imbalance_db": amp_imbalance_db,
        "iq_orthogonality_corr": orthogonality,
        "dc_offset_i": dc_i,
        "dc_offset_q": dc_q,
        "total_dc_offset": total_dc,
        "spectral_flatness": spectral_flatness,
        "occupied_bw_ratio": occupied_bw_ratio,
        "autocorr_lag1": autocorr_lag1,
        "autocorr_lag2": autocorr_lag2,
    }


def run_domain_gap_audit(
    sample_limit: int = 2000,
    random_seed: int = 42,
    output_json_path: str = "results/ml/m6/domain_gap_audit.json",
    output_md_path: str = "results/ml/m6/domain_gap_audit.md",
) -> Dict[str, Any]:
    """
    Executes a comprehensive domain gap comparison across synthetic and real domains.
    """
    print("=" * 60)
    print("M6 REPRODUCIBLE DOMAIN-GAP STATISTICAL AUDIT")
    print("=" * 60)
    
    rng = np.random.default_rng(random_seed)

    # 1. Ingest Real Data (RadioML Real + External Real-World Dev Split)
    print("\nIngesting real-world signal distributions...")
    _ = RadioMLDataset()
    real_radioml = DatasetCache.samples
    
    # Subsample real RadioML
    if len(real_radioml) > sample_limit:
        indices = rng.choice(len(real_radioml), size=sample_limit, replace=False)
        real_rml_sub = real_radioml[indices]
    else:
        real_rml_sub = real_radioml
    print(f"  RadioML real subset: {real_rml_sub.shape}")

    # External Real-World dataset
    ext_split = load_external_dataset(allow_dev_subset_fallback=True)
    # Slice external 1024 frames to 128 windows for fair statistical comparison
    ext_windows = slice_frames_to_windows(ext_split.X, window_length=128)
    if len(ext_windows) > sample_limit:
        ext_indices = rng.choice(len(ext_windows), size=sample_limit, replace=False)
        real_ext_sub = ext_windows[ext_indices]
    else:
        real_ext_sub = ext_windows
    print(f"  External real-world sliced subset: {real_ext_sub.shape} (provenance: {ext_split.provenance})")

    # 2. Ingest Synthetic Data
    print("\nIngesting synthetic signal distributions...")
    syn_samples, _, _ = load_synthetic_dataset("datasets/synthetic")
    if len(syn_samples) > sample_limit:
        syn_indices = rng.choice(len(syn_samples), size=sample_limit, replace=False)
        syn_sub = syn_samples[syn_indices]
    else:
        syn_sub = syn_samples
    print(f"  Synthetic evaluation subset: {syn_sub.shape}")

    # 3. Compute Metrics
    print("\nCalculating metrics across all domains...")
    metrics_syn = compute_comprehensive_statistics(syn_sub)
    metrics_rml = compute_comprehensive_statistics(real_rml_sub)
    metrics_ext = compute_comprehensive_statistics(real_ext_sub)

    # 4. Identify Root Causes & Compute Discrepancies
    gap_analysis = {}
    root_causes = []
    
    for metric_name in metrics_syn:
        val_syn = metrics_syn[metric_name]
        val_rml = metrics_rml[metric_name]
        val_ext = metrics_ext[metric_name]
        
        # Absolute and Relative differences
        diff_rml = val_syn - val_rml
        rel_diff_rml = diff_rml / (abs(val_rml) + 1e-9)
        diff_ext = val_syn - val_ext
        rel_diff_ext = diff_ext / (abs(val_ext) + 1e-9)
        
        gap_analysis[metric_name] = {
            "synthetic": val_syn,
            "real_radioml": val_rml,
            "real_external": val_ext,
            "delta_vs_radioml": diff_rml,
            "relative_delta_vs_radioml": rel_diff_rml,
            "delta_vs_external": diff_ext,
            "relative_delta_vs_external": rel_diff_ext,
        }

    # Root Cause Diagnostic Checkpoints
    # 1. Amplitude Scale mismatch
    rms_ratio_rml = metrics_syn["complex_rms"] / (metrics_rml["complex_rms"] + 1e-9)
    if abs(rms_ratio_rml - 1.0) > 0.5:
        root_causes.append(
            f"Global Amplitude Scale Mismatch: Synthetic RMS ({metrics_syn['complex_rms']:.4f}) is "
            f"{rms_ratio_rml:.1f}x of RadioML real RMS ({metrics_rml['complex_rms']:.4f}). "
            "Without checkpoint RMS scaling, 1D CNN activations collapse or saturate."
        )

    # 2. Phase Jitter & Frequency Drift
    if abs(metrics_syn["phase_diff_variance"] - metrics_rml["phase_diff_variance"]) > 0.1:
        root_causes.append(
            f"Phase Noise / Frequency Drift Discrepancy: Synthetic phase diff variance "
            f"({metrics_syn['phase_diff_variance']:.4f}) differs from real "
            f"({metrics_rml['phase_diff_variance']:.4f}), causing symbol timing misalignment."
        )

    # 3. Multipath Dispersion & Temporal Correlation
    if abs(metrics_syn["autocorr_lag1"] - metrics_ext["autocorr_lag1"]) > 0.15:
        root_causes.append(
            f"Channel Multipath Dispersion Gap: External real-world signals under multipath show "
            f"temporal autocorrelation lag-1 of {metrics_ext['autocorr_lag1']:.4f} vs synthetic "
            f"{metrics_syn['autocorr_lag1']:.4f}, demonstrating that frequency-selective fading is required."
        )

    # 4. PAPR and Envelope Dynamics
    if abs(metrics_syn["papr_db"] - metrics_rml["papr_db"]) > 1.0:
        root_causes.append(
            f"Envelope Dynamics & PAPR Gap: Synthetic PAPR ({metrics_syn['papr_db']:.2f} dB) vs "
            f"RadioML ({metrics_rml['papr_db']:.2f} dB), indicating pulse-shaping or amplifier non-linearity gap."
        )

    audit_result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "synthetic_dataset": "synthetic_evaluation_dataset.npz",
            "real_benchmark": "RadioML 2016.10A",
            "external_validation_source": ext_split.provenance,
            "random_seed": random_seed,
            "samples_analyzed_per_domain": sample_limit,
        },
        "root_causes_identified": root_causes,
        "metrics_summary": gap_analysis,
    }

    # 5. Export JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(audit_result, f, indent=4)
    print(f"\nSaved audit JSON to: {output_json_path}")
    
    # Also save to reports/m6_domain_gap.json if directory exists
    os.makedirs("reports", exist_ok=True)
    with open("reports/m6_domain_gap.json", "w") as f:
        json.dump(audit_result, f, indent=4)
    print("Exported copy to: reports/m6_domain_gap.json")

    # 6. Export Markdown
    md_content = f"""# ML Phase M6: Domain-Gap Statistical Audit Report

**Generated:** {audit_result['timestamp']}  
**Provenance:** Synthetic (`synthetic_evaluation_dataset.npz`), RadioML 2016.10A, and External Real-World Dev Subset (`{ext_split.provenance}`).

---

## 1. Executive Summary & Root Causes

The M6 Domain-Gap Audit reveals key statistical discrepancies between synthetic training signals and real-world IQ captures:

"""
    for i, rc in enumerate(root_causes, 1):
        md_content += f"{i}. **{rc}**\n"

    md_content += """
---

## 2. Statistical Metric Comparison Table

| Metric | Synthetic Baseline | Real (RadioML) | Real (External Dev) | $\\Delta$ vs RadioML | $\\Delta$ vs External |
|---|---|---|---|---|---|
"""
    for metric, vals in gap_analysis.items():
        md_content += (
            f"| `{metric}` | {vals['synthetic']:.4f} | {vals['real_radioml']:.4f} | "
            f"{vals['real_external']:.4f} | {vals['delta_vs_radioml']:+.4f} | "
            f"{vals['delta_vs_external']:+.4f} |\n"
        )

    md_content += """
---

## 3. Engineering Recommendations for M6 & M7

1. **Envelope Representation**: Providing explicit envelope $|z|$ in `[I, Q, |z|]` allows CNN kernels to inspect instantaneous amplitude without requiring deep non-linear convolutions.
2. **Multipath Augmentation**: Tapped delay line (TDL) Rayleigh fading with 2-4 taps and exponential power-delay profile models realistic channel coherence.
3. **Open-Set Rejection**: Configurable thresholding on maximum softmax probability (MSP < 0.35) and Shannon entropy (> 1.80) prevents forcing unsupported modulations (OFDM, GMSK, NBFM) into closed 11-class sets.
"""

    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w") as f:
        f.write(md_content)
    print(f"Saved audit Markdown to: {output_md_path}")
    
    with open("reports/m6_domain_gap.md", "w") as f:
        f.write(md_content)
    print("Exported copy to: reports/m6_domain_gap.md")
    print("=" * 60)
    
    return audit_result


if __name__ == "__main__":
    run_domain_gap_audit()
