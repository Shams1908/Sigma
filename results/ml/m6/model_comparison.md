# ML Phase M6: Controlled Model Evaluation & Comparison

**Generated:** 2026-09-08T02:52:12Z  
**Random Seed:** 42 | **Epochs:** 5 | **Batch Size:** 64

---

## 1. Experimental Overview

In accordance with M6 design principles:
- **Baseline:** M5 Raw IQ 1D CNN with `[I, Q]` inputs (2 channels).
- **Candidate 1:** Robust 1D CNN with `[I, Q, |z|]` inputs (3 channels).
- **Candidate 2:** Domain-Augmented 1D CNN with `[I, Q, |z|]` inputs (3 channels) trained with Rayleigh multipath and RF front-end impairments.

> [!NOTE]
> `[I, Q, |z|]` is treated strictly as an experimentally motivated candidate representation. Findings below represent empirical observations under controlled conditions and are subject to full out-of-domain evaluation during M7.

---

## 2. Model Architecture & Parameter Count

| Model | Input Channels | Representation | Parameter Count | $\Delta$ Parameters |
|---|---|---|---|---|
| **Baseline M5** | 2 | `RAW_IQ` [I, Q] | 175,819 | Reference (0) |
| **M6 Representation** | 3 | `IQ_AMPLITUDE` [I, Q, \|z\|] | 176,267 | +448 (+0.25%) |
| **M6 Augmented** | 3 | `IQ_AMPLITUDE` [I, Q, \|z\|] | 176,267 | +448 (+0.25%) |

---

## 3. Performance Metrics (Clean vs Channel-Impaired)

| Configuration | Clean Acc | Clean Macro F1 | Impaired Acc | Impaired Macro F1 | Latency / sample |
|---|---|---|---|---|---|
| **Baseline M5 (`RAW_IQ`)** | 0.3342 | 0.2614 | 0.1925 | 0.1457 | 0.132 ms |
| **M6 (`IQ_AMPLITUDE`)** | 0.3041 | 0.2535 | 0.1451 | 0.0902 | 0.130 ms |
| **M6 (Augmented + `IQ_AMPLITUDE`)** | 0.3757 | 0.3538 | 0.2901 | 0.2633 | 0.135 ms |

---

## 4. Key Empirical Observations

1. **Computational Overhead**: Adding the 3rd input channel $|z|$ adds only 448 parameters to the first convolutional layer (a 0.25% parameter increase) and incurs negligible per-sample inference latency difference (~-0.002 ms).
2. **Channel Robustness**: Impairment augmentation exposes the convolutional network to multi-tap Rayleigh dispersion and carrier phase jitter during training, providing measurable robustness under impaired channel conditions without degrading clean signal classification.
3. **M7 Next Steps**: Formal validation against the external benchmark `subset_test.h5` will evaluate these representations under real multipath fading.
