# ML Phase M6: Domain-Gap Statistical Audit Report

**Generated:** 2026-09-08T02:50:29Z  
**Provenance:** Synthetic (`synthetic_evaluation_dataset.npz`), RadioML 2016.10A, and External Real-World Dev Subset (`M6-DEV-SUBSET`).

---

## 1. Executive Summary & Root Causes

The M6 Domain-Gap Audit reveals key statistical discrepancies between synthetic training signals and real-world IQ captures:

1. **Global Amplitude Scale Mismatch: Synthetic RMS (0.3428) is 40.2x of RadioML real RMS (0.0085). Without checkpoint RMS scaling, 1D CNN activations collapse or saturate.**
2. **Phase Noise / Frequency Drift Discrepancy: Synthetic phase diff variance (1.2559) differs from real (1.5455), causing symbol timing misalignment.**

---

## 2. Statistical Metric Comparison Table

| Metric | Synthetic Baseline | Real (RadioML) | Real (External Dev) | $\Delta$ vs RadioML | $\Delta$ vs External |
|---|---|---|---|---|---|
| `complex_rms` | 0.3428 | 0.0085 | 0.9708 | +0.3343 | -0.6280 |
| `signal_power` | 0.2580 | 0.0001 | 0.9831 | +0.2579 | -0.7251 |
| `power_std` | 1.0192 | 0.0000 | 0.3803 | +1.0192 | +0.6389 |
| `amplitude_mean` | 0.2752 | 0.0078 | 0.9467 | +0.2674 | -0.6716 |
| `amplitude_std` | 0.4269 | 0.0035 | 0.2946 | +0.4235 | +0.1323 |
| `amplitude_skewness` | 0.5082 | 0.2675 | -0.0811 | +0.2408 | +0.5894 |
| `amplitude_kurtosis` | -1.0479 | -0.0388 | 1.8588 | -1.0091 | -2.9067 |
| `papr_db` | 6.4174 | 5.4700 | 2.2804 | +0.9474 | +4.1370 |
| `phase_variance` | 3.0600 | 2.7875 | 3.0458 | +0.2725 | +0.0142 |
| `phase_diff_variance` | 1.2559 | 1.5455 | 0.1979 | -0.2896 | +1.0580 |
| `inst_freq_mean` | 0.0025 | 0.0242 | 0.0377 | -0.0217 | -0.0352 |
| `inst_freq_kurtosis` | 2.2768 | 2.2768 | 7.8049 | -0.0001 | -5.5282 |
| `iq_amplitude_imbalance_db` | 3.2015 | -0.8137 | -0.0035 | +4.0151 | +3.2050 |
| `iq_orthogonality_corr` | -0.0029 | -0.0262 | -0.0003 | +0.0233 | -0.0026 |
| `dc_offset_i` | 0.0179 | 0.0002 | 0.0069 | +0.0177 | +0.0110 |
| `dc_offset_q` | 0.0185 | -0.0006 | -0.0034 | +0.0191 | +0.0219 |
| `total_dc_offset` | 0.0257 | 0.0006 | 0.0077 | +0.0251 | +0.0180 |
| `spectral_flatness` | 0.0597 | 0.2945 | 0.0568 | -0.2348 | +0.0029 |
| `occupied_bw_ratio` | 0.3688 | 0.7085 | 0.4918 | -0.3397 | -0.1230 |
| `autocorr_lag1` | 0.8928 | 0.4269 | 0.9196 | +0.4658 | -0.0268 |
| `autocorr_lag2` | 0.8152 | 0.3848 | 0.8471 | +0.4304 | -0.0320 |

---

## 3. Engineering Recommendations for M6 & M7

1. **Envelope Representation**: Providing explicit envelope $|z|$ in `[I, Q, |z|]` allows CNN kernels to inspect instantaneous amplitude without requiring deep non-linear convolutions.
2. **Multipath Augmentation**: Tapped delay line (TDL) Rayleigh fading with 2-4 taps and exponential power-delay profile models realistic channel coherence.
3. **Open-Set Rejection**: Configurable thresholding on maximum softmax probability (MSP < 0.35) and Shannon entropy (> 1.80) prevents forcing unsupported modulations (OFDM, GMSK, NBFM) into closed 11-class sets.
