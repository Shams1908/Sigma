# Machine Learning Components - SIGMA Project

This directory contains the machine learning components of the SIGMA platform.

> [!IMPORTANT]
> **Milestone Status:**
> * **M1 — Dataset Foundation:** COMPLETE and verified.
> * **M2.1, M2.2, M2.3, M2.4, M2.5 — Synthetic Signal Generator Track:** COMPLETE and verified (includes base config, bit generator, modulation mappers, RRC pulse shaping, AWGN channel, and RF impairments).
> * *Rayleigh/Rician fading channels, multipath, FEC, interleaving, or ML models are NOT implemented.*

## Subdirectories
* `dataset/`: Dataset foundation classes, loader, metadata, validation, and splitting.
* `generators/` (NEW): Synthetic signal generator foundation, containing configuration models, deterministic bit generation, validation, and signal containers.
* `feature_model/`: Feature engineering and extraction logic for classic ML models.
* `cnn_model/`: Neural network models (PyTorch) for signal classification directly from raw IQ data or spectrograms.
* `training/`: Scripts and pipelines to train and evaluate ML classifiers.
* `inference/`: Deployment modules and optimized runtimes for low-latency signal classification.

---

## Milestone M1: Dataset Foundation

The purpose of M1 is to establish a robust, verified, and memory-efficient internal dataset representation and splitting logic for the **RadioML 2016.10A** benchmark dataset.

### RadioML 2016.10A Dataset Structure

* **Acquisition Source:** RadioML 2016.10A / DeepSig (locally stored at `datasets/raw/RML2016.10a_dict.pkl`)
* **Format:** Python dictionary containing 220 unique keys
* **Unique Keys:** `(modulation, SNR)` tuples, mapping to a NumPy array of shape `(1000, 2, 128)`
* **Total Examples:** 220,000 examples
* **Example Sample Shape:** `[2, 128]` where:
  * `samples[0]` = In-phase (I) channel
  * `samples[1]` = Quadrature (Q) channel
* **Data Type:** `float32` (preserving raw values, no silent normalization or filtering)
* **Sample Rate:** Not provided in the raw structure (no sample rate is invented or assumed)

### Modulation Labels (11 Classes)
Ordered alphabetically to produce a stable and deterministic integer mapping:
1. `8PSK` (Index 0)
2. `AM-DSB` (Index 1)
3. `AM-SSB` (Index 2)
4. `BPSK` (Index 3)
5. `CPFSK` (Index 4)
6. `GFSK` (Index 5)
7. `PAM4` (Index 6)
8. `QAM16` (Index 7)
9. `QAM64` (Index 8)
10. `QPSK` (Index 9)
11. `WBFM` (Index 10)

### SNR Range (20 Values)
`-20dB` to `+18dB` in steps of 2dB:
`[-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]`

### Stratified Split Strategy
To avoid copying huge raw signal float32 arrays in memory, splits are managed using flat 1D integer index arrays pointing to a singleton class-level cached dataset (`DatasetCache`).

* **Seed Configuration:** Fixed default seed of `42` (reproducibility guaranteed)
* **Split Ratios:** 70% Train, 15% Validation, 15% Test
* **Per (Modulation, SNR) group split:**
  * 700 Train examples
  * 150 Validation examples
  * 150 Test examples
* **Split Sizes:**
  * **Train Set:** 154,000 examples
  * **Validation Set:** 33,000 examples
  * **Test Set:** 33,000 examples
  * **Total:** 220,000 examples (no overlap between splits)

---

## Milestone M2.1: Synthetic Signal Generator Foundation

The purpose of M2.1 is to build the core architecture, data schemas, validation, bit generation utilities, and signal representations for the synthetic signal generation track.

### Synthetic Signal Pipeline (Future Implementation Roadmap)
The planned synthetic generation pipeline follows this flow:
```text
  Random Bits (M2.1)
         ↓
    FEC (Future)
         ↓
Interleaving (Future)
         ↓
 Modulation (M2.2)
         ↓
Pulse Shaping (Future)
         ↓
  Channel (Future)
         ↓
RF Impairments (Future)
         ↓
  IQ Samples [2, N] (M2.1)
         ↓
Ground-Truth Metadata (M2.1)
```

### Key Components Implemented in M2.1

#### 1. Generation Configuration (`GeneratorConfig`)
A typed, validated configuration object storing physical parameters required to generate waveforms:
- `modulation` (strictly validated against M1 labels)
- `num_symbols` (> 0)
- `sample_rate` (> 0)
- `symbol_rate` (> 0)
- `samples_per_symbol` (integer, mathematically consistent: `samples_per_symbol == sample_rate / symbol_rate`)
- `random_seed` (integer, >= 0)
- Optional parameters: `snr`, `frequency_offset`, `phase_offset`, `timing_offset`

#### 2. Ground-Truth Metadata (`SyntheticGroundTruth`)
A container for the exact parameters used during signal generation. This keeps ground-truth settings distinct from downstream estimations and ML predictions.

#### 3. Base Signal Representaton (`GeneratedSignal`)
Container storing the generated floating-point sample arrays of shape `[2, N]` (float32, where channel 0 is In-phase and channel 1 is Quadrature) and its associated `SyntheticGroundTruth` metadata.

#### 4. IQ and Complex Conversion Helpers
Strict shape-preserving, lossless (within floating point precision limits) conversion functions:
- `complex_to_iq(complex_samples)`: Converts 1D complex arrays of shape `(N,)` to canonical `(2, N)` float32 arrays.
- `iq_to_complex(iq_samples)`: Converts canonical `(2, N)` float32 arrays back to 1D complex arrays.

#### 5. Reusable Bit Generator (`generate_bits`)
Generates random binary bit arrays (0 or 1) of shape `(num_bits,)` deterministically. Randomness state is managed by accepting an explicit `np.random.Generator` or seed, ensuring no mutations to the global NumPy random state.

---

## Milestone M2.2: Digital Modulation Generators

The purpose of M2.2 is to implement ideal complex baseband modulation mappers for the first five digital modulations (BPSK, QPSK, 8PSK, 16QAM, 64QAM) using standard Gray coding and constellation power scaling.

> [!IMPORTANT]
> **Ideal Baseband Output:**
> M2.2 produces ideal baseband complex symbols. Pulse shaping, AWGN noise, channel models, frequency/phase/timing offsets, and other RF impairments are reserved for future milestones.

### Constellation & Gray Mapping Conventions

Every modulator in M2.2 maps deterministic input bit streams to complex symbols based on standard Gray-coded constellations (where adjacent symbol states differ by exactly 1 bit) and is normalized to unit average symbol power ($E[|s|^2] = 1.0$) using fixed constellation normalization constants.

#### 1. BPSK
- **Bits per symbol:** 1 bit
- **Normalization Constant:** `1.0`
- **Mapping Table:**
  - `0` -> `-1.0 + 0.0j`
  - `1` -> `+1.0 + 0.0j`

#### 2. QPSK
- **Bits per symbol:** 2 bits, grouped as `(b0, b1)`
- **Normalization Constant:** `1 / sqrt(2) ≈ 0.7071`
- **Gray Mapping Table:**
  - `00` -> `(1 + 1j) / sqrt(2)`
  - `01` -> `(-1 + 1j) / sqrt(2)`
  - `11` -> `(-1 - 1j) / sqrt(2)`
  - `10` -> `(1 - 1j) / sqrt(2)`

#### 3. 8PSK
- **Bits per symbol:** 3 bits, grouped as `(b0, b1, b2)`
- **Normalization Constant:** `1.0` (all points reside on the unit circle)
- **Gray Mapping Table (mapping to phase angle $\theta_k = k \times \frac{\pi}{4}$):**
  - `000` -> Angle `0`
  - `001` -> Angle `1 × π/4`
  - `011` -> Angle `2 × π/4`
  - `010` -> Angle `3 × π/4`
  - `110` -> Angle `4 × π/4`
  - `111` -> Angle `5 × π/4`
  - `101` -> Angle `6 × π/4`
  - `100` -> Angle `7 × π/4`

#### 4. 16QAM (Square Constellation)
- **Bits per symbol:** 4 bits, grouped as `(b0, b1, b2, b3)`
- **Normalization Constant:** `1 / sqrt(10) ≈ 0.3162`
- **Gray Mapping Scheme:**
  - Bits `(b0, b1)` map to the In-phase (I) level.
  - Bits `(b2, b3)` map to the Quadrature (Q) level.
  - The level mappers follow the 2-bit Gray scale:
    - `00` -> `+3`
    - `01` -> `+1`
    - `11` -> `-1`
    - `10` -> `-3`

#### 5. 64QAM (Square Constellation)
- **Bits per symbol:** 6 bits, grouped as `(b0, b1, b2, b3, b4, b5)`
- **Normalization Constant:** `1 / sqrt(42) ≈ 0.1543`
- **Gray Mapping Scheme:**
  - Bits `(b0, b1, b2)` map to the In-phase (I) level.
  - Bits `(b3, b4, b5)` map to the Quadrature (Q) level.
  - The level mappers follow the 3-bit Gray scale:
    - `000` -> `+7`
    - `001` -> `+5`
    - `011` -> `+3`
    - `010` -> `+1`
    - `110` -> `-1`
    - `111` -> `-3`
    - `101` -> `-5`
    - `100` -> `-7`

### Storable Signal Integration
Complex baseband symbols are converted to the project's canonical `[2, N]` IQ format (where channel 0 is I and channel 1 is Q) and wrapped in a `GeneratedSignal` container containing `SyntheticGroundTruth` metadata, preserving the canonical labels (`QAM16`, `QAM64`) to match dataset schemas.

---

## Milestone M2.3: Pulse Shaping

The purpose of M2.3 is to implement Root Raised Cosine (RRC) pulse-shaping and zero-upsampling for baseband complex waveforms.

> [!IMPORTANT]
> **Ideal Pulse-Shaped Waveform:**
> M2.3 produces ideal pulse-shaped signals. Fading, multipath channel distortion, noise, and other RF impairments are reserved for future milestones.

### Design Concepts

#### 1. Samples Per Symbol (SPS)
The upsampling rate is mathematically defined as:
`samples_per_symbol = sample_rate / symbol_rate`
This value must resolve to a positive integer to facilitate discrete-time zero-upsampling.

#### 2. Root Raised Cosine (RRC) Filter Impulse Response
The discrete-time filter coefficients $h(n)$ are designed using the standard RRC formula:
$$h(n) = \frac{\sin(\pi (1 - \alpha) \tau) + 4 \alpha \tau \cos(\pi (1 + \alpha) \tau)}{\pi \tau \left(1 - (4 \alpha \tau)^2\right)}$$
where:
- $\alpha$ is the configurable `rolloff` factor ($0.0 \le \alpha \le 1.0$).
- $\tau = n / SPS$ is the normalized time index.
- Singularities at $\tau = 0$ and $\tau = \pm 1 / (4 \alpha)$ are handled explicitly using mathematical limits:
  - At $\tau = 0$: $h(0) = 1 - \alpha + \frac{4\alpha}{\pi}$
  - At $\tau = \pm 1 / (4 \alpha)$: $h(t) = \frac{\alpha}{\sqrt{2}} \left[ \left(1 + \frac{2}{\pi}\right) \sin\left(\frac{\pi}{4\alpha}\right) + \left(1 - \frac{2}{\pi}\right) \cos\left(\frac{\pi}{4\alpha}\right) \right]$

#### 3. Filter Span and Tap Count
The span of the filter in symbols is configured using `filter_span_symbols` (a positive integer). This yields a symmetric FIR structure containing exactly:
`num_taps = filter_span_symbols * samples_per_symbol + 1` taps.

#### 4. Filter Normalization
To ensure the pulse-shaping filter maintains constant gain, the impulse response is normalized to unit energy:
$$\sum_{n} h(n)^2 = 1.0$$
This scaling depends only on static filter parameters (not on symbol realization).

#### 5. Upsampling (Zero Insertion)
Discrete-time upsampling is performed by inserting exactly `samples_per_symbol - 1` zeros after each complex symbol:
`[s0, 0, 0, 0, 0, 0, 0, 0, s1, 0, 0, 0, 0, 0, 0, 0, s2, ...]`
No interpolation or symbol replication is performed, preserving the RRC pulse shape.

#### 6. Linear Filtering & Delay
The upsampled sequence is filtered using full linear convolution (`np.convolve(..., mode='full')`). 
- **Group Delay:** A symmetric FIR filter introduces a delay of `(num_taps - 1) / 2` samples. The full filtered waveform (including transients and delay) is returned by default.
- **Output Length:** The exact length of the generated sample array is:
  `output_length = (num_symbols + filter_span_symbols) * samples_per_symbol` samples.

---

## Milestone M2.4: AWGN Channel

The purpose of M2.4 is to implement a reusable Additive White Gaussian Noise (AWGN) channel model that computes noise power dynamically from the input waveform power and target SNR.

> [!IMPORTANT]
> **Channel Pipeline Placement:**
> AWGN is applied *after* RRC pulse-shaping in the pipeline. This ensures the target SNR is measured against the actual RRC-filtered clean waveform entering the channel, matching standard physical simulations.

### Mathematical Formulation

#### 1. Signal Power ($P_{signal}$)
Calculated dynamically from the input complex baseband waveform $x$:
$$P_{signal} = \text{mean}(|x|^2) = \frac{1}{N} \sum_{n=0}^{N-1} |x(n)|^2$$

#### 2. Noise Power ($P_{noise}$)
Given target $SNR_{dB}$, the total noise power $P_{noise}$ is calculated as:
$$P_{noise} = \frac{P_{signal}}{10^{(SNR_{dB} / 10)}}$$

#### 3. Variance Split on In-Phase and Quadrature Channels
For a complex channel $n = n_I + j n_Q$, the total noise power $P_{noise}$ is divided equally between the real (I) and imaginary (Q) components:
$$\sigma^2 = \text{Var}(n_I) = \text{Var}(n_Q) = \frac{P_{noise}}{2}$$
The noise components are generated as independent Gaussian variables:
$$n_I \sim \mathcal{N}\left(0, \frac{P_{noise}}{2}\right), \quad n_Q \sim \mathcal{N}\left(0, \frac{P_{noise}}{2}\right)$$

The final received waveform is:
$$y(n) = x(n) + n(n)$$

### Design Details
- **RNG Determinism:** Uses an explicit `np.random.Generator` instance internally, guaranteeing exact seeding reproducibility and preventing global NumPy random state mutation.
- **Zero-Power Rejection:** Raises a `ValueError` if the signal power is close to zero, since SNR is mathematically undefined for zero-power inputs.
- **Precision:** Input and output arrays reside on `complex64` representations, preserving float32 boundary requirements.

---

## Milestone M2.5: RF Impairments

The purpose of M2.5 is to implement five modular, deterministic RF impairments in the synthetic generation pipeline: Carrier Frequency Offset, Carrier Phase Offset, Complex DC Offset, IQ Amplitude/Phase Imbalance, and Fractional Timing Offset.

### Mathematical Impairment Formulations

#### 1. Carrier Frequency Offset
$$y[n] = x[n] \times \exp\left(j 2 \pi \Delta f \frac{n}{F_s}\right)$$
where:
- $\Delta f$ is the frequency offset in Hz (`frequency_offset`).
- $F_s$ is the sample rate in Hz (`sample_rate`).

#### 2. Carrier Phase Offset
$$y[n] = x[n] \times \exp(j \phi)$$
where $\phi$ is the constant phase offset in radians (`phase_offset`).

#### 3. IQ Imbalance (Amplitude and Phase skew)
A 2x2 real transformation matrix maps the In-phase and Quadrature components:
$$\begin{bmatrix} I_{out} \\ Q_{out} \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \begin{bmatrix} I_{in} \\ Q_{in} \end{bmatrix}$$
where:
- $a_{11} = (1 + A) \cos(\theta / 2)$
- $a_{12} = -(1 + A) \sin(\theta / 2)$
- $a_{21} = -(1 - A) \sin(\theta / 2)$
- $a_{22} = (1 - A) \cos(\theta / 2)$
- **Amplitude Imbalance Semantics ($A$):** Scales the In-phase (I) channel amplitude by $(1 + A)$ and the Quadrature (Q) channel amplitude by $(1 - A)$. For example, $A = 0.1$ implies the In-phase channel gain is $1.1$ and the Quadrature channel gain is $0.9$ (ratio $\approx 1.222$).
- **Phase Imbalance Semantics ($\theta$):** Represents the skew from orthogonality in radians (`iq_phase_imbalance`).
- **Identity Condition:** Setting $A = 0$ and $\theta = 0$ results in the identity matrix, leaving the waveform unmodified.

#### 4. Complex DC Offset
DC offset is applied directly to the baseband complex representation:
$$y[n] = x[n] + (I_{dc} + j Q_{dc})$$
where $I_{dc}$ is In-phase DC offset (`dc_offset_i`) and $Q_{dc}$ is Quadrature DC offset (`dc_offset_q`).

#### 5. Fractional Timing Offset
Implemented using linear interpolation with zero-padding for boundary values (avoiding circular wrapping):
$$y[n] = (1 - d) x[n - k] + d x[n - k - 1]$$
where:
- $k = \lfloor \tau \rfloor$ (integer sample delay).
- $d = \tau - k$ (fractional sample delay).
- **Direction Convention:** A positive timing offset ($\tau > 0$) delays the signal (shifts it to the right).

---

### Pipeline Composition Order

When multiple impairments are enabled, they are applied sequentially in the following strict order:
1. **Frequency offset**
2. **Phase offset**
3. **IQ imbalance**
4. **DC offset**
5. **Timing offset**

All impairment transformations are deterministic and contain no internal random state or stochastic modeling.

---

## How to Run

### 1. Dataset Inspection Utility
To print dataset properties and verify split statistics, run the inspection script from the root of the project:
```powershell
# Run using the backend virtual environment python
.\backend\venv\Scripts\python.exe -m ml.dataset.inspect_dataset
```

### 2. Running Unit & Integration Tests
To run all tests (including M1 dataset loading stability and M2.1 synthetic signal generator foundation tests), execute pytest:
```powershell
.\backend\venv\Scripts\python.exe -m pytest tests/ml/ -v
```


