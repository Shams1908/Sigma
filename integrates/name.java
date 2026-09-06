Machine Learning Components - SIGMA Project
This directory contains the machine learning components of the SIGMA platform.

[!IMPORTANT] Milestone Status:

M1 — Dataset Foundation: COMPLETE and verified.
M2.1, M2.2, M2.3, M2.4, M2.5 — Synthetic Signal Generator Track: COMPLETE and verified (includes base config, bit generator, modulation mappers, RRC pulse shaping, AWGN channel, and RF impairments).
M3 — Feature Engineering: COMPLETE and verified.
M4 — Classical ML Baseline: COMPLETE and verified.
Rayleigh/Rician fading channels, multipath, FEC, interleaving, or ML models are NOT implemented.
Subdirectories
dataset/: Dataset foundation classes, loader, metadata, validation, and splitting.
generators/ (NEW): Synthetic signal generator foundation, containing configuration models, deterministic bit generation, validation, and signal containers.
feature_model/: Feature engineering and extraction logic for classic ML models.
cnn_model/: Neural network models (PyTorch) for signal classification directly from raw IQ data or spectrograms.
training/: Scripts and pipelines to train and evaluate ML classifiers.
inference/: Deployment modules and optimized runtimes for low-latency signal classification.
Milestone M1: Dataset Foundation
The purpose of M1 is to establish a robust, verified, and memory-efficient internal dataset representation and splitting logic for the RadioML 2016.10A benchmark dataset.

RadioML 2016.10A Dataset Structure
Acquisition Source: RadioML 2016.10A / DeepSig (locally stored at datasets/raw/RML2016.10a_dict.pkl)
Format: Python dictionary containing 220 unique keys
Unique Keys: (modulation, SNR) tuples, mapping to a NumPy array of shape (1000, 2, 128)
Total Examples: 220,000 examples
Example Sample Shape: [2, 128] where:
samples[0] = In-phase (I) channel
samples[1] = Quadrature (Q) channel
Data Type: float32 (preserving raw values, no silent normalization or filtering)
Sample Rate: Not provided in the raw structure (no sample rate is invented or assumed)
Modulation Labels (11 Classes)
Ordered alphabetically to produce a stable and deterministic integer mapping:

8PSK (Index 0)
AM-DSB (Index 1)
AM-SSB (Index 2)
BPSK (Index 3)
CPFSK (Index 4)
GFSK (Index 5)
PAM4 (Index 6)
QAM16 (Index 7)
QAM64 (Index 8)
QPSK (Index 9)
WBFM (Index 10)
SNR Range (20 Values)
-20dB to +18dB in steps of 2dB: [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

Stratified Split Strategy
To avoid copying huge raw signal float32 arrays in memory, splits are managed using flat 1D integer index arrays pointing to a singleton class-level cached dataset (DatasetCache).

Seed Configuration: Fixed default seed of 42 (reproducibility guaranteed)
Split Ratios: 70% Train, 15% Validation, 15% Test
Per (Modulation, SNR) group split:
700 Train examples
150 Validation examples
150 Test examples
Split Sizes:
Train Set: 154,000 examples
Validation Set: 33,000 examples
Test Set: 33,000 examples
Total: 220,000 examples (no overlap between splits)
Milestone M2.1: Synthetic Signal Generator Foundation
The purpose of M2.1 is to build the core architecture, data schemas, validation, bit generation utilities, and signal representations for the synthetic signal generation track.

Synthetic Signal Pipeline (Future Implementation Roadmap)
The planned synthetic generation pipeline follows this flow:

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
Key Components Implemented in M2.1
1. Generation Configuration (GeneratorConfig)
A typed, validated configuration object storing physical parameters required to generate waveforms:

modulation (strictly validated against M1 labels)
num_symbols (> 0)
sample_rate (> 0)
symbol_rate (> 0)
samples_per_symbol (integer, mathematically consistent: samples_per_symbol == sample_rate / symbol_rate)
random_seed (integer, >= 0)
Optional parameters: snr, frequency_offset, phase_offset, timing_offset
2. Ground-Truth Metadata (SyntheticGroundTruth)
A container for the exact parameters used during signal generation. This keeps ground-truth settings distinct from downstream estimations and ML predictions.

3. Base Signal Representaton (GeneratedSignal)
Container storing the generated floating-point sample arrays of shape [2, N] (float32, where channel 0 is In-phase and channel 1 is Quadrature) and its associated SyntheticGroundTruth metadata.

4. IQ and Complex Conversion Helpers
Strict shape-preserving, lossless (within floating point precision limits) conversion functions:

complex_to_iq(complex_samples): Converts 1D complex arrays of shape (N,) to canonical (2, N) float32 arrays.
iq_to_complex(iq_samples): Converts canonical (2, N) float32 arrays back to 1D complex arrays.
5. Reusable Bit Generator (generate_bits)
Generates random binary bit arrays (0 or 1) of shape (num_bits,) deterministically. Randomness state is managed by accepting an explicit np.random.Generator or seed, ensuring no mutations to the global NumPy random state.

Milestone M2.2: Digital Modulation Generators
The purpose of M2.2 is to implement ideal complex baseband modulation mappers for the first five digital modulations (BPSK, QPSK, 8PSK, 16QAM, 64QAM) using standard Gray coding and constellation power scaling.

[!IMPORTANT] Ideal Baseband Output: M2.2 produces ideal baseband complex symbols. Pulse shaping, AWGN noise, channel models, frequency/phase/timing offsets, and other RF impairments are reserved for future milestones.

Constellation & Gray Mapping Conventions
Every modulator in M2.2 maps deterministic input bit streams to complex symbols based on standard Gray-coded constellations (where adjacent symbol states differ by exactly 1 bit) and is normalized to unit average symbol power (
E
[
∣
s
∣
2
]
=
1.0
E[∣s∣ 
2
 ]=1.0) using fixed constellation normalization constants.

1. BPSK
Bits per symbol: 1 bit
Normalization Constant: 1.0
Mapping Table:
0 -> -1.0 + 0.0j
1 -> +1.0 + 0.0j
2. QPSK
Bits per symbol: 2 bits, grouped as (b0, b1)
Normalization Constant: 1 / sqrt(2) ≈ 0.7071
Gray Mapping Table:
00 -> (1 + 1j) / sqrt(2)
01 -> (-1 + 1j) / sqrt(2)
11 -> (-1 - 1j) / sqrt(2)
10 -> (1 - 1j) / sqrt(2)
3. 8PSK
Bits per symbol: 3 bits, grouped as (b0, b1, b2)
Normalization Constant: 1.0 (all points reside on the unit circle)
Gray Mapping Table (mapping to phase angle 
θ
k
=
k
×
π
4
θ 
k
​
 =k× 
4
π
​
 ):
000 -> Angle 0
001 -> Angle 1 × π/4
011 -> Angle 2 × π/4
010 -> Angle 3 × π/4
110 -> Angle 4 × π/4
111 -> Angle 5 × π/4
101 -> Angle 6 × π/4
100 -> Angle 7 × π/4
4. 16QAM (Square Constellation)
Bits per symbol: 4 bits, grouped as (b0, b1, b2, b3)
Normalization Constant: 1 / sqrt(10) ≈ 0.3162
Gray Mapping Scheme:
Bits (b0, b1) map to the In-phase (I) level.
Bits (b2, b3) map to the Quadrature (Q) level.
The level mappers follow the 2-bit Gray scale:
00 -> +3
01 -> +1
11 -> -1
10 -> -3
5. 64QAM (Square Constellation)
Bits per symbol: 6 bits, grouped as (b0, b1, b2, b3, b4, b5)
Normalization Constant: 1 / sqrt(42) ≈ 0.1543
Gray Mapping Scheme:
Bits (b0, b1, b2) map to the In-phase (I) level.
Bits (b3, b4, b5) map to the Quadrature (Q) level.
The level mappers follow the 3-bit Gray scale:
000 -> +7
001 -> +5
011 -> +3
010 -> +1
110 -> -1
111 -> -3
101 -> -5
100 -> -7
Storable Signal Integration
Complex baseband symbols are converted to the project's canonical [2, N] IQ format (where channel 0 is I and channel 1 is Q) and wrapped in a GeneratedSignal container containing SyntheticGroundTruth metadata, preserving the canonical labels (QAM16, QAM64) to match dataset schemas.

Milestone M2.3: Pulse Shaping
The purpose of M2.3 is to implement Root Raised Cosine (RRC) pulse-shaping and zero-upsampling for baseband complex waveforms.

[!IMPORTANT] Ideal Pulse-Shaped Waveform: M2.3 produces ideal pulse-shaped signals. Fading, multipath channel distortion, noise, and other RF impairments are reserved for future milestones.

Design Concepts
1. Samples Per Symbol (SPS)
The upsampling rate is mathematically defined as: samples_per_symbol = sample_rate / symbol_rate This value must resolve to a positive integer to facilitate discrete-time zero-upsampling.

2. Root Raised Cosine (RRC) Filter Impulse Response
The discrete-time filter coefficients 
h
(
n
)
h(n) are designed using the standard RRC formula:

h
(
n
)
=
sin
⁡
(
π
(
1
−
α
)
τ
)
+
4
α
τ
cos
⁡
(
π
(
1
+
α
)
τ
)
π
τ
(
1
−
(
4
α
τ
)
2
)
h(n)= 
πτ(1−(4ατ) 
2
 )
sin(π(1−α)τ)+4ατcos(π(1+α)τ)
​
 

where:

α
α is the configurable rolloff factor (
0.0
≤
α
≤
1.0
0.0≤α≤1.0).
τ
=
n
/
S
P
S
τ=n/SPS is the normalized time index.
Singularities at 
τ
=
0
τ=0 and 
τ
=
±
1
/
(
4
α
)
τ=±1/(4α) are handled explicitly using mathematical limits:
At 
τ
=
0
τ=0: 
h
(
0
)
=
1
−
α
+
4
α
π
h(0)=1−α+ 
π
4α
​
 
At 
τ
=
±
1
/
(
4
α
)
τ=±1/(4α): 
h
(
t
)
=
α
2
[
(
1
+
2
π
)
sin
⁡
(
π
4
α
)
+
(
1
−
2
π
)
cos
⁡
(
π
4
α
)
]
h(t)= 
2
​
 
α
​
 [(1+ 
π
2
​
 )sin( 
4α
π
​
 )+(1− 
π
2
​
 )cos( 
4α
π
​
 )]
3. Filter Span and Tap Count
The span of the filter in symbols is configured using filter_span_symbols (a positive integer). This yields a symmetric FIR structure containing exactly: num_taps = filter_span_symbols * samples_per_symbol + 1 taps.

4. Filter Normalization
To ensure the pulse-shaping filter maintains constant gain, the impulse response is normalized to unit energy:

∑
n
h
(
n
)
2
=
1.0
n
∑
​
 h(n) 
2
 =1.0

This scaling depends only on static filter parameters (not on symbol realization).

5. Upsampling (Zero Insertion)
Discrete-time upsampling is performed by inserting exactly samples_per_symbol - 1 zeros after each complex symbol: [s0, 0, 0, 0, 0, 0, 0, 0, s1, 0, 0, 0, 0, 0, 0, 0, s2, ...] No interpolation or symbol replication is performed, preserving the RRC pulse shape.

6. Linear Filtering & Delay
The upsampled sequence is filtered using full linear convolution (np.convolve(..., mode='full')).

Group Delay: A symmetric FIR filter introduces a delay of (num_taps - 1) / 2 samples. The full filtered waveform (including transients and delay) is returned by default.
Output Length: The exact length of the generated sample array is: output_length = (num_symbols + filter_span_symbols) * samples_per_symbol samples.
Milestone M2.4: AWGN Channel
The purpose of M2.4 is to implement a reusable Additive White Gaussian Noise (AWGN) channel model that computes noise power dynamically from the input waveform power and target SNR.

[!IMPORTANT] Channel Pipeline Placement: AWGN is applied after RRC pulse-shaping in the pipeline. This ensures the target SNR is measured against the actual RRC-filtered clean waveform entering the channel, matching standard physical simulations.

Mathematical Formulation
1. Signal Power (
P
s
i
g
n
a
l
P 
signal
​
 )
Calculated dynamically from the input complex baseband waveform 
x
x:

P
s
i
g
n
a
l
=
mean
(
∣
x
∣
2
)
=
1
N
∑
n
=
0
N
−
1
∣
x
(
n
)
∣
2
P 
signal
​
 =mean(∣x∣ 
2
 )= 
N
1
​
  
n=0
∑
N−1
​
 ∣x(n)∣ 
2
 

2. Noise Power (
P
n
o
i
s
e
P 
noise
​
 )
Given target 
S
N
R
d
B
SNR 
dB
​
 , the total noise power 
P
n
o
i
s
e
P 
noise
​
  is calculated as:

P
n
o
i
s
e
=
P
s
i
g
n
a
l
10
(
S
N
R
d
B
/
10
)
P 
noise
​
 = 
10 
(SNR 
dB
​
 /10)
 
P 
signal
​
 
​
 

3. Variance Split on In-Phase and Quadrature Channels
For a complex channel 
n
=
n
I
+
j
n
Q
n=n 
I
​
 +jn 
Q
​
 , the total noise power 
P
n
o
i
s
e
P 
noise
​
  is divided equally between the real (I) and imaginary (Q) components:

σ
2
=
Var
(
n
I
)
=
Var
(
n
Q
)
=
P
n
o
i
s
e
2
σ 
2
 =Var(n 
I
​
 )=Var(n 
Q
​
 )= 
2
P 
noise
​
 
​
 

The noise components are generated as independent Gaussian variables:

n
I
∼
N
(
0
,
P
n
o
i
s
e
2
)
,
n
Q
∼
N
(
0
,
P
n
o
i
s
e
2
)
n 
I
​
 ∼N(0, 
2
P 
noise
​
 
​
 ),n 
Q
​
 ∼N(0, 
2
P 
noise
​
 
​
 )

The final received waveform is:

y
(
n
)
=
x
(
n
)
+
n
(
n
)
y(n)=x(n)+n(n)

Design Details
RNG Determinism: Uses an explicit np.random.Generator instance internally, guaranteeing exact seeding reproducibility and preventing global NumPy random state mutation.
Zero-Power Rejection: Raises a ValueError if the signal power is close to zero, since SNR is mathematically undefined for zero-power inputs.
Precision: Input and output arrays reside on complex64 representations, preserving float32 boundary requirements.
Milestone M2.5: RF Impairments
The purpose of M2.5 is to implement five modular, deterministic RF impairments in the synthetic generation pipeline: Carrier Frequency Offset, Carrier Phase Offset, Complex DC Offset, IQ Amplitude/Phase Imbalance, and Fractional Timing Offset.

Mathematical Impairment Formulations
1. Carrier Frequency Offset
y
[
n
]
=
x
[
n
]
×
exp
⁡
(
j
2
π
Δ
f
n
F
s
)
y[n]=x[n]×exp(j2πΔf 
F 
s
​
 
n
​
 )

where:

Δ
f
Δf is the frequency offset in Hz (frequency_offset).
F
s
F 
s
​
  is the sample rate in Hz (sample_rate).
2. Carrier Phase Offset
y
[
n
]
=
x
[
n
]
×
exp
⁡
(
j
ϕ
)
y[n]=x[n]×exp(jϕ)

where 
ϕ
ϕ is the constant phase offset in radians (phase_offset).

3. IQ Imbalance (Amplitude and Phase skew)
A 2x2 real transformation matrix maps the In-phase and Quadrature components:

[
I
o
u
t
Q
o
u
t
]
=
[
a
11
a
12
a
21
a
22
]
[
I
i
n
Q
i
n
]
[ 
I 
out
​
 
Q 
out
​
 
​
 ]=[ 
a 
11
​
 
a 
21
​
 
​
  
a 
12
​
 
a 
22
​
 
​
 ][ 
I 
in
​
 
Q 
in
​
 
​
 ]

where:

a
11
=
(
1
+
A
)
cos
⁡
(
θ
/
2
)
a 
11
​
 =(1+A)cos(θ/2)
a
12
=
−
(
1
+
A
)
sin
⁡
(
θ
/
2
)
a 
12
​
 =−(1+A)sin(θ/2)
a
21
=
−
(
1
−
A
)
sin
⁡
(
θ
/
2
)
a 
21
​
 =−(1−A)sin(θ/2)
a
22
=
(
1
−
A
)
cos
⁡
(
θ
/
2
)
a 
22
​
 =(1−A)cos(θ/2)
Amplitude Imbalance Semantics (
A
A): Scales the In-phase (I) channel amplitude by 
(
1
+
A
)
(1+A) and the Quadrature (Q) channel amplitude by 
(
1
−
A
)
(1−A). For example, 
A
=
0.1
A=0.1 implies the In-phase channel gain is 
1.1
1.1 and the Quadrature channel gain is 
0.9
0.9 (ratio 
≈
1.222
≈1.222).
Phase Imbalance Semantics (
θ
θ): Represents the skew from orthogonality in radians (iq_phase_imbalance).
Identity Condition: Setting 
A
=
0
A=0 and 
θ
=
0
θ=0 results in the identity matrix, leaving the waveform unmodified.
4. Complex DC Offset
DC offset is applied directly to the baseband complex representation:

y
[
n
]
=
x
[
n
]
+
(
I
d
c
+
j
Q
d
c
)
y[n]=x[n]+(I 
dc
​
 +jQ 
dc
​
 )

where 
I
d
c
I 
dc
​
  is In-phase DC offset (dc_offset_i) and 
Q
d
c
Q 
dc
​
  is Quadrature DC offset (dc_offset_q).

5. Fractional Timing Offset
Implemented using linear interpolation with zero-padding for boundary values (avoiding circular wrapping):

y
[
n
]
=
(
1
−
d
)
x
[
n
−
k
]
+
d
x
[
n
−
k
−
1
]
y[n]=(1−d)x[n−k]+dx[n−k−1]

where:

k
=
⌊
τ
⌋
k=⌊τ⌋ (integer sample delay).
d
=
τ
−
k
d=τ−k (fractional sample delay).
Direction Convention: A positive timing offset (
τ
>
0
τ>0) delays the signal (shifts it to the right).
Pipeline Composition Order
When multiple impairments are enabled, they are applied sequentially in the following strict order:

Frequency offset
Phase offset
IQ imbalance
DC offset
Timing offset
All impairment transformations are deterministic and contain no internal random state or stochastic modeling.

Milestone M3: Feature Engineering
The purpose of M3 is to build a reusable, deterministic, and highly optimized DSP feature-extraction pipeline that converts raw IQ samples (shape [2, 128]) into fixed-length numerical feature vectors (shape [36]).

Feature Schema & Ordering
The extractor outputs a stable feature vector containing exactly 36 features in a strict deterministic order:

Amplitude Features (4):
amplitude_mean: Sample mean of 
a
[
n
]
=
∣
x
[
n
]
∣
a[n]=∣x[n]∣.
amplitude_variance: Sample variance of 
a
[
n
]
a[n].
amplitude_kurtosis: Fisher's excess kurtosis of 
a
[
n
]
a[n]. If 
Var
(
a
)
<
1
e
−
9
Var(a)<1e−9, returns 
0.0
0.0.
amplitude_peak_to_average_ratio: PAR 
=
max
⁡
(
a
)
2
/
E
[
a
2
]
=max(a) 
2
 /E[a 
2
 ]. If 
E
[
a
2
]
<
1
e
−
9
E[a 
2
 ]<1e−9, returns 
0.0
0.0.
Phase Features (12):
phase_variance: Sample variance of wrapped phase 
ϕ
[
n
]
=
angle
(
x
[
n
]
)
∈
[
−
π
,
π
]
ϕ[n]=angle(x[n])∈[−π,π] (without unwrapping).
phase_difference_mean: Sample mean of phase difference 
d
ϕ
[
n
]
=
angle
(
x
[
n
]
x
∗
[
n
−
1
]
)
dϕ[n]=angle(x[n]x 
∗
 [n−1]).
phase_difference_variance: Sample variance of 
d
ϕ
[
n
]
dϕ[n].
phase_difference_kurtosis: Excess kurtosis of 
d
ϕ
[
n
]
dϕ[n]. If 
Var
(
d
ϕ
)
<
1
e
−
9
Var(dϕ)<1e−9, returns 
0.0
0.0.
phase_hist_bin_0 to phase_hist_bin_7: Normalized probability mass over 8 uniform phase bins covering 
[
−
π
,
π
)
[−π,π). Near-zero magnitude samples (
∣
x
[
n
]
∣
<
1
e
−
9
∣x[n]∣<1e−9) are safely forced to 
0.0
0.0 phase.
Instantaneous Frequency Features (2):
instantaneous_frequency_mean: Sample mean of normalized phase increments 
d
ϕ
[
n
]
dϕ[n] (angular change per sample, not converted to Hz).
instantaneous_frequency_variance: Sample variance of 
d
ϕ
[
n
]
dϕ[n].
Normalized 
C
40
C 
40
​
  Cumulant Features (3):
c40_real, c40_imag, c40_magnitude representing components of the normalized fourth-order cumulant:
C
40
=
E
[
x
4
]
−
3
(
E
[
x
2
]
)
2
E
[
∣
x
∣
2
]
2
C 
40
​
 = 
E[∣x∣ 
2
 ] 
2
 
E[x 
4
 ]−3(E[x 
2
 ]) 
2
 
​
 

Calculated in double-precision complex numbers (complex128) for stability. Returns 
0.0
+
0.0
j
0.0+0.0j if signal power is 
<
1
e
−
9
<1e−9.
Autocorrelation Features (15):
Complex normalized autocorrelation 
R
norm
[
k
]
=
R
[
k
]
/
R
[
0
]
R 
norm
​
 [k]=R[k]/R[0] components (real, imaginary, and magnitude) for lags 
k
∈
{
1
,
2
,
4
,
8
,
16
}
k∈{1,2,4,8,16}, where 
R
[
k
]
=
E
[
x
[
n
]
x
∗
[
n
−
k
]
]
R[k]=E[x[n]x 
∗
 [n−k]]. Returns 
0.0
+
0.0
j
0.0+0.0j if average power 
R
[
0
]
<
1
e
−
9
R[0]<1e−9.
Feature Extraction API
Exposes three high-level vectorized methods in ml/features/extractor.py:

extract_features(samples): Process single sample shape [2, 128] to return names, feature_vector.
extract_batch_features(batch_samples): Process batch of shape [N, 2, 128] to return feature matrix [N, 36].
extract_feature_matrix(dataset, indices): Batch process full or subset dataset to return feature matrix X of shape [N, 36], targets y of shape [N], and SNR metadata snrs of shape [N].
Numeric Robustness & Data Leakage Policy
Numeric Safety: All divisions are protected using epsilon masks, ensuring exactly 0 NaNs and 0 Infs are produced for any input signal (including zero-power signals).
Zero Leakage: No parameter of the feature extractor is learned from the dataset (no fit-step). Downstream scaling or dimensionality reduction (to be done in M4) must be fitted on training splits only.
Diagnostic & Inspection Utility
To run the extraction pipeline on all 220,000 samples, perform validation, analyze training split feature correlation, and generate the final summary report, run:

# Run the inspection utility
.\backend\venv\Scripts\python.exe ml/features/inspect_features.py
This utility automatically saves the processed feature matrix, labels, and SNR metadata as a compressed NumPy file at datasets/processed/RML2016.10a_features.npz.

Milestone M4: Classical ML Baseline
The purpose of M4 is to train, optimize, select, and evaluate classical machine-learning classifiers using the 36 DSP features extracted in M3, establishing a strong, interpretable baseline before deep learning (CNNs).

Models Evaluated
Random Forest: Scalable tree ensemble, evaluated with n_estimators={50, 100}, max_depth={15, 20}.
HistGradientBoosting: Histogram-based Gradient Boosting, optimized for large datasets, evaluated with max_iter={100, 150}, max_depth={10, 15}.
All models are trained with a fixed random_state = 42.

Preprocessing & Data Splits
Zero Preprocessing: Preprocessing and feature scaling are explicitly omitted since decision-tree ensembles are invariant to monotonic scaling, maintaining absolute simplicity.
Split Strategy: Reuses the exact M1 stratified index split (70% Train, 15% Val, 15% Test) with no split leakage. All hyperparameters are tuned using the validation split.
Model Selection
The champion model is selected using the validation split's macro F1 score:

Selected Champion: HistGradientBoostingClassifier (max_iter=100, max_depth=10, learning_rate=0.1)
Validation Macro F1: 0.5643 (Accuracy: 0.5482)
Feature Ablation Experiment
Highly correlated feature sets (
∣
r
∣
>
0.95
∣r∣>0.95 on the training split) were evaluated for redundancy. Removing 5 features (instantaneous_frequency_variance, instantaneous_frequency_mean, autocorr_lag_1_magnitude, autocorr_lag_2_magnitude, autocorr_lag_4_magnitude) resulted in:

Validation Macro F1 (31 features): 0.5644 (Accuracy: 0.5481)
Conclusion: Removing these redundant features maintains model validation performance perfectly, validating M3 correlation analysis.
Final Untouched Test Set Evaluation
The selected full-feature champion model was evaluated on the untouched test split (33,000 samples) and achieved:

Accuracy: 0.5494
Macro Precision: 0.6410
Macro Recall: 0.5494
Macro F1: 0.5656
Weighted F1: 0.5656
Performance degrades severely at lower SNRs but becomes highly robust at high SNRs (reaching over 86% accuracy above 12 dB).

Feature Importance & Explainability
For Random Forest, feature importances map directly back to the M3 schema. The top 5 ranked features are:

amplitude_variance
autocorr_lag_1_real
amplitude_mean
autocorr_lag_1_magnitude
amplitude_peak_to_average_ratio
This demonstrates that amplitude variations and short-lag correlation statistics carry the most discriminative power for classical modulation classification.

Model Artifacts & Inference API
Saved Model: Serialized at models/baseline_hgb.joblib along with comprehensive metadata at models/baseline_hgb_metadata.json.
Inference Usage: Use predict(features) in ml/baselines/inference.py to reload models and execute predictions:
from ml.baselines.inference import predict
result = predict(sample_features) # shape (36,)
# Returns: predicted class name, class index, confidence, probabilities
Milestone M5: Raw IQ CNN Baseline
The purpose of M5 is to build a baseline 1D Convolutional Neural Network (CNN) in PyTorch to perform modulation classification directly from raw IQ waveform samples of shape [2, 128], and compare its performance against the classical M4 baseline.

1D CNN Architecture
Input Layer: [batch, 2, 128]
Block 1: Conv1d(2 -> 64, k=7, p=3), BatchNorm1d, ReLU, MaxPool1d(2) (128 -> 64)
Block 2: Conv1d(64 -> 128, k=5, p=2), BatchNorm1d, ReLU, MaxPool1d(2) (64 -> 32)
Block 3: Conv1d(128 -> 256, k=3, p=1), BatchNorm1d, ReLU
Global Pooling: Global Average Pooling (GAP) mean(dim=-1) resulting in [batch, 256]
Fully Connected Head: Linear(256 -> 128), ReLU, Dropout(0.3), Linear(128 -> 11)
Preprocessing & Normalization
The training RMS scaling factor is dynamically computed from training split samples only:

rms
train
=
E
[
x
train
2
]
≈
0.006048427
rms 
train
​
 = 
E[x 
train
2
​
 ]
​
 ≈0.006048427

Input signals are normalized as 
x
normalized
=
x
/
rms
train
x 
normalized
​
 =x/rms 
train
​
  ensuring the training set RMS is exactly 1.0. This factor is saved in the model checkpoint metadata and loaded for inference.

Model Training & Performance
Optimizer: Adam (learning rate = 1e-3, batch size = 128)
Loss Function: Multiclass Cross Entropy Loss
Early Stopping: Epoch 13 (patience of 5 epochs based on Validation Macro F1). Best epoch: Epoch 8.
Best Validation Macro F1: 0.5900
Saved Model: Serialized at models/m5_iq_cnn.pt with scaling metadata.
Final Untouched Test Set Evaluation
Accuracy: 0.5668
Macro F1: 0.5844
Weighted F1: 0.5844
Macro Precision: 0.6811
Macro Recall: 0.5668
M4 vs M5 Baseline Comparison
Both models were evaluated on the same test dataset split:

Metric	M4 HGB (36 DSP Features)	M5 1D CNN (Raw IQ)	Difference
Accuracy	0.5494	0.5668	+0.0174
Macro F1	0.5656	0.5844	+0.0188
Conclusion: The learned representation from raw IQ waveforms outperforms the handcrafted classical baseline by 1.88% Macro F1 points.

Milestone M6.1: Synthetic Evaluation Dataset
The purpose of M6.1 is to build a controlled, deterministic synthetic evaluation dataset to study the robustness of the trained M5 CNN model under isolated, parameterized channel impairments.

[!NOTE] This milestone generates an evaluation dataset and does NOT alter the M5 checkpoint or use synthetic data for training.

Supported Modulation Classes
Reusing the M2 generator, the evaluation dataset supports the 5 digital modulation classes:

BPSK, QPSK, 8PSK, QAM16, QAM64 The other 6 modulation classes (AM-DSB, AM-SSB, CPFSK, GFSK, PAM4, WBFM) are explicitly logged as unsupported and omitted from generation.
Generation Parameters & Length
Each generated waveform has shape [2, 128], obtained by configuring:

num_symbols = 8
filter_span_symbols = 8
samples_per_symbol = 8 Output length 
=
(
8
+
8
)
×
8
=
128
=(8+8)×8=128 samples.
Isolated Impairment Sweeps
We perform 7 isolated experiments where exactly one impairment parameter is swept while the others are held constant (at 0 or nominal value) at a high SNR of 18 dB (except for the AWGN sweep):

AWGN: SNR 
∈
{
−
20
,
−
10
,
0
,
10
,
18
}
∈{−20,−10,0,10,18} dB
Frequency Offset: 
Δ
f
∈
{
0.0
,
100.0
,
500.0
,
2000.0
,
10000.0
}
Δf∈{0.0,100.0,500.0,2000.0,10000.0} Hz
Phase Offset: 
θ
∈
{
0.0
,
π
/
8
,
π
/
4
,
π
/
2
,
π
}
θ∈{0.0,π/8,π/4,π/2,π} rad
IQ Amplitude Imbalance: 
A
∈
{
0.0
,
0.05
,
0.1
,
0.2
,
0.3
}
A∈{0.0,0.05,0.1,0.2,0.3}
IQ Phase Imbalance: 
ϕ
∈
{
0.0
,
0.05
,
0.1
,
0.2
,
0.3
}
ϕ∈{0.0,0.05,0.1,0.2,0.3} rad
DC Offset: 
I
d
c
=
Q
d
c
∈
{
0.0
,
0.05
,
0.1
,
0.2
,
0.3
}
I 
dc
​
 =Q 
dc
​
 ∈{0.0,0.05,0.1,0.2,0.3}
Timing Offset: 
τ
∈
{
0.0
,
0.1
,
0.2
,
0.3
,
0.4
,
0.5
}
τ∈{0.0,0.1,0.2,0.3,0.4,0.5} samples
Dataset Format & Output
Total Generated Examples: 18,000 samples (36 conditions 
×
× 5 modulations 
×
× 100 examples per combination).
Dataset File: [datasets/synthetic/synthetic_evaluation_dataset.npz](file:///c:/Shambhavi/VS%20Code/Sigma/datasets/synthetic/synthetic_evaluation_dataset.npz) containing X ([18000, 2, 128]) and y ([18000]).
Metadata File: [datasets/synthetic/synthetic_evaluation_metadata.json](file:///c:/Shambhavi/VS%20Code/Sigma/datasets/synthetic/synthetic_evaluation_metadata.json) containing Ground Truth configurations, seeds, and sweep parameters for each sample.
RNG Determinism: Seeded generation (base_seed = 42000) guarantees exact replication of waveforms.
Milestone M6.2: Synthetic-to-Real Cross-Domain Evaluation
The purpose of M6.2 is to evaluate our frozen M5 Raw IQ CNN model (trained on real RadioML 2016.10A data) against the synthetic evaluation dataset generated in M6.1. This is a cross-domain evaluation experiment to determine how well the model transfers from real-world data to parameterized, simulated data.

[!IMPORTANT] M6.2 is evaluation ONLY. The CNN baseline model remains fully frozen, with weights and preprocessing scaling parameters unmodified.

5-Class Evaluation Limitation
Because the synthetic dataset only contains BPSK, QPSK, 8PSK, QAM16, and QAM64 modulations, accuracy and macro averages (Precision, Recall, F1) are calculated specifically over these 5 supported classes using Scikit-Learn labels parameters. The other 6 modulation classes are treated as unsupported.

Evaluation Findings & Domain Gap Analysis
Overall Accuracy on Synthetic Data: 0.2218
Overall Macro F1 on Synthetic Data: 0.1244 (compared to 0.5844 on Real RadioML test set).
This severe degradation reveals a major synthetic-to-real domain gap (Outcome B). The CNN is highly sensitive to waveform structures (such as pulse shaping filter spans, exact power scaling, impairment profiles, or sample rates) that differ between the real RadioML dataset and our synthetic simulator.

Per-Class Recall Analysis:
QPSK achieves high recall of 93.56% (acting as a majority predicted class).
BPSK and 8PSK recall values drop to near-zero, showing they are not correctly recognized under simulated conditions.
SNR and Impairment Robustness Reports
Under AWGN noise conditions, performance degrades uniformly. Non-AWGN impairments (such as DC offsets, timing delays, and frequency offsets) cause additional relative degradation compared to the clean baseline.

Consolidated robustness details are saved in [impairment_robustness.csv](file:///c:/Shambhavi/VS%20Code/Sigma/results/ml/m6/impairment_robustness.csv) and [impairment_robustness.json](file:///c:/Shambhavi/VS%20Code/Sigma/results/ml/m6/impairment_robustness.json).
Cross-domain summary comparisons are saved in [cross_domain_comparison.json](file:///c:/Shambhavi/VS%20Code/Sigma/results/ml/m6/cross_domain_comparison.json).
Performance and confusion matrix plots are stored under [results/ml/m6/](file:///c:/Shambhavi/VS%20Code/Sigma/results/ml/m6/).
Milestone M6.3: Real-vs-Synthetic Domain Gap Analysis
The purpose of M6.3 is to identify the root causes of the synthetic-to-real domain gap. We matched exactly 100 samples per class × SNR level (across 5 supported classes and 5 SNR levels, total 2,500 real and 2,500 synthetic samples) and performed statistical comparisons.

[!IMPORTANT] M6.3 is diagnostic ONLY. No modifications were made to the M2 generator, M3 features, or M5 CNN weights.

Key Measured Domain Differences
Amplitude Scaling / Power Normalization Mismatch (Primary Cause):

Real Raw RMS: 0.006056
Synthetic Raw RMS: 0.852382
Normalized Real RMS (CNN Input): 1.001312
Normalized Synthetic RMS (CNN Input): 140.926178
Observed Fact: The synthetic samples are scaled to have a normalized RMS that is 140 times larger than the real training samples. When presented to the CNN, this scaling mismatch saturates the activation functions and Batch Normalization layers, causing classification collapse.

Statistical Verification: The Decision Tree domain classifier separates the real and synthetic domains with 100% accuracy using ONLY the amplitude_mean feature.
Phase Difference kurtosis (Noise/Channel Mismatch):

Real Mean Kurtosis: 5.7769
Synthetic Mean Kurtosis: -0.1826 (Cohen's 
d
=
−
0.9711
d=−0.9711)
Observed Fact: Real signals have heavy-tailed phase differences caused by timing jitter and phase noise spikes, whereas synthetic signals have a flat, standard Gaussian phase difference distribution.

Carrier Offset & Autocorrelation Biases:

The imaginary components of autocorrelation features (autocorr_lag_4_imag, autocorr_lag_2_imag, autocorr_lag_1_imag) show a distinct non-zero bias in real signals compared to near-zero imaginary values in clean synthetic signals. This suggests a persistent frequency/phase bias or timing drift in the real acquisition setup that the synthetic simulator does not model.
Outputs & Visualizations
Consolidated feature ranking: [domain_feature_comparison.csv](file:///c:/Shambhavi/VS%20Code/Sigma/results/ml/m6/domain_analysis/domain_feature_comparison.csv)
Class-conditional statistics: [domain_gap_by_class.csv](file:///c:/Shambhavi/VS%20Code/Sigma/results/ml/m6/domain_analysis/domain_gap_by_class.csv)
Raw IQ statistics summary: [raw_iq_statistics.csv](file:///c:/Shambhavi/VS%20Code/Sigma/results/ml/m6/domain_analysis/raw_iq_statistics.csv)
Global separability plot: [pca_domain_separability.png](file:///c:/Shambhavi/VS%20Code/Sigma/results/ml/m6/domain_analysis/pca_domain_separability.png)
Feature distribution histograms: [domain_gap_features.png](file:///c:/Shambhavi/VS%20Code/Sigma/results/ml/m6/domain_analysis/domain_gap_features.png)
Average PSD curves: [spectral_comparison.png](file:///c:/Shambhavi/VS%20Code/Sigma/results/ml/m6/domain_analysis/spectral_comparison.png)
Visual comparisons (I/Q waveforms, envelopes, constellations): stored under [results/ml/m6/domain_visuals/](file:///c:/Shambhavi/VS%20Code/Sigma/results/ml/m6/domain_visuals/)
Milestone M6.4.1: Data-Driven Synthetic Amplitude Calibration
The purpose of M6.4.1 was to solve the severe amplitude scale gap (140x) identified in M6.3 without hardcoding any numeric constants.

Dynamic Calculation: We compute the training-split real RMS dynamically and the synthetic clean reference RMS dynamically.
Global Scaling: A scale factor is applied to synthetic signals so that their RMS matches the real-world reference split.
Results: Correcting the amplitude mismatch improved frozen CNN Macro F1 from 0.1244 to 0.1989 (+0.07 F1) and reduced the QPSK prediction collapse.
Milestone M6.4.2: Realistic Synthetic Channel Refinement
We refined the synthetic generator by implementing realistic receiver distortions matching M6.3 measurements:

Wiener Phase Noise: Simulates local oscillator drift step cumulative random walks.
FIR Channel Response: Convolves waveforms with power-normalized channel filter taps.
Results: Evaluating on refined signals yielded 0.1958 F1. This slight increase in difficulty reflects real-world propagation channels.
Milestone M6.5: Synthetic-Assisted Training
We conducted a fully controlled 5-class evaluation (BPSK, QPSK, 8PSK, QAM16, QAM64) to determine if synthetic data can assist training:

Real Only Baseline: Test Macro F1 of 0.6403.
Synthetic Only: Generalizes to real test set at 0.2942 F1.
Mixed Training: Adding synthetic data (10%, 25%, 50%) degrades real test performance (e.g., 10% mix drops F1 to 0.5603). This shows that domain mismatch causes models to learn simulation-specific features.
Pretrain-Finetune: Synthetic pretraining followed by real fine-tuning achieves 0.6372 F1, indicating its utility as a representation pretraining step.
Quality Verification: Real + 10% Original Synthetic yields 0.4337 F1, whereas Real + 10% Calibrated/Refined Synthetic achieves 0.5704 / 0.5603 F1. This programmatically validates that our calibration work reduced the mismatch degradation significantly (+0.13 F1).
Milestone M7.1: Universal Signal Input & WAV/IQ Adapter
This package provides a clean, reusable input adapter layer that automatically ingests user-uploaded signal files, validates them, and converts them to the canonical form expected by the ML layer.

Architectural Pipeline Separation
To prevent responsibilities from leaking, the architecture enforces a strict boundaries separation:

INPUT PARSING (Adapter Layer): Responsible for file format detection, container parsing, validation checks (NaNs/Infs, size limits), layout standardization to canonical [2, N], and segmentation into [M, 2, 128] windows.
PREPROCESSING (Model Layer): Responsible for dynamic amplitude scaling (e.g. dividing by model-specific training RMS factor) immediately before running inference.
MODEL INFERENCE (PyTorch Layer): Running the batch inputs through the RawIQCNN neural network to obtain predictions.
Supported Input Formats
Stereo WAV: Audio container containing In-phase and Quadrature channels.
NPY/NPZ: NumPy arrays containing signal data.
BIN/DAT: Raw interleaved or non-interleaved binary IQ files (requires explicit configuration).
WAV IQ Layout Convention
The input pipeline strictly expects stereo WAV files where:
Channel 0 represents the In-phase (I) component.
Channel 1 represents the Quadrature (Q) component.
Mono WAV files are explicitly rejected with a validation error to prevent silent, arbitrary audio interpretations.
Integer formats (e.g., int16) are dynamically normalized to float32 values in range [-1.0, 1.0].
Canonical IQ Representation
Downstream components receive a standardized Complex IQ layout:

Shape: [2, N] (2 rows: row 0 = I, row 1 = Q; N columns of samples).
Dtype: float32.
Validation constraints: Zero NaNs, zero Infs, finite numeric values, and non-empty signal samples.
Segmentation Behavior
Converts canonical [2, N] arrays into windows of shape [M, 2, segment_length].
Segment Length: Defaults to 128 (configurable).
Signals shorter than segment length: Rejected with a validation error by default. Trailing zero-padding can be explicitly enabled via configuration (pad_short=True).
Signals longer than segment length: Segmented into multiple complete windows. Residual trailing samples are discarded.
Ordering: Strict, deterministic sequence matching the original time-series.
Configuration Options
Centralized in PipelineConfig and BinaryIQConfig:

max_file_size_bytes: Configurable maximum allowed size (defaults to 50 MB).
segment_length: Expected window length (defaults to 128).
binary_config: Configuration required for raw binary files (defines dtype, interleaved boolean, and endianness). No silent guessing.
Unsupported Input Behavior
Mismatched container structures (e.g., non-ZIP .npz files), unsupported extensions, empty files, or files containing NaNs/Infs will produce a clear, descriptive validation error and set validation status to ERROR in the result metadata.
Milestone M7.2: End-to-End File → CNN Prediction Pipeline
This package integrates the M7.1 input layer with the pre-trained M5 1D CNN baseline to provide a unified end-to-end file-to-prediction analysis pipeline.

Architectural Responsibility Separation
To keep components decoupled, the interface separates responsibilities:

M7.1 (Input Package): Ingests, validates, normalizes container data formats to canonical float32 [2, N], and segments the signal into non-overlapping windows.
M7.2 (Inference Package): Loads the frozen model, extracts the checkpoint-specific normalization scaling factor, runs batch inference, and aggregates segment-level predictions.
File-to-Prediction Flow
               USER UPLOAD FILE (WAV, NPY, NPZ, BIN, DAT)
                                  ↓
                   M7.1 FORMAT & HEADER DETECTION
                                  ↓
                PARSING TO CANONICAL COMPLEX IQ [2, N]
                                  ↓
                    SEGMENTATION INTO [2, 128]
                                  ↓
                 LOAD CACHED FROZEN M5 MODEL & RMS
                                  ↓
                 RMS SCALING & PyTorch CPU INFERENCE
                                  ↓
                 SEGMENT Softmax LOGITS & ENTROPIES
                                  ↓
                FILE-LEVEL PROBABILITY AGGREGATION
Dynamic Normalization Handling
No Hardcoding: We never hardcode dataset RMS values.
The integration layer reads the training-only RMS scaling factor (rms_factor) dynamically from the checkpoint dictionary models/m5_iq_cnn.pt during model loading.
The input segment values are divided by this factor before PyTorch execution, guaranteeing normalization consistency across all signals.
Segment-Level Predictions
For each processed window, we compute:

Predicted class index and predicted modulation name.
Squeezed confidence score (maximum probability).
Raw probability vector across all 11 classes.
Top-k predictions (defaults to 5) sorted in descending order.
File-Level Aggregation Strategy
To avoid bias toward the first window in a long recording, we compute a unified file-level prediction:

Average Probability: Compute the element-wise mean probability vector across all valid segmented windows.
Argmax Class Selection: Run argmax on the average probability vector to determine the final predicted modulation.
Aggregated Confidence: Retrieve the value at the predicted class index in the average probability vector.
Shannon Entropy: Calculate the average Shannon entropy across windows to quantify prediction uncertainty:
H
ˉ
=
−
1
M
∑
m
=
1
M
∑
c
=
1
C
p
m
,
c
log
⁡
(
p
m
,
c
+
10
−
12
)
H
ˉ
 =− 
M
1
​
  
m=1
∑
M
​
  
c=1
∑
C
​
 p 
m,c
​
 log(p 
m,c
​
 +10 
−12
 )

Output Result Schema
The output is captured as a structured SignalAnalysisResult object, containing:

Input metadata: filename, detected format, sample rate, original sample count, and canonical IQ shape.
Model predictions: predicted class index, name, confidence, probability vector, and top-5 predictions.
Window results: window count, predictions, confidences, probabilities, and class distribution counts.
Execution info: checkpoint path, normalization source, window length, and success status.
How to Run
1. Dataset Inspection Utility
To print dataset properties and verify split statistics, run the inspection script from the root of the project:

.\backend\venv\Scripts\python.exe -m ml.dataset.inspect_dataset
2. M5 CNN Training Pipeline
To run CNN baseline training, validation model checkpointing, history plots, and test set evaluations:

$env:PYTHONPATH="."
.\backend\venv\Scripts\python.exe ml/cnn_model/train.py
3. M6.1 Synthetic Generator Diagnostics
To run the synthetic evaluation dataset generation, seed verification, and representative waveforms plotting:

$env:PYTHONPATH="."
.\backend\venv\Scripts\python.exe ml/synthetic/inspect_synthetic.py
4. M6.2 Synthetic Cross-Domain Evaluation
To execute model loading parameter hash checks, run inference on synthetic waveforms, and generate impairment robustness metrics/confusion matrices:

$env:PYTHONPATH="."
.\backend\venv\Scripts\python.exe ml/synthetic/evaluate.py
5. M6.3 Real-vs-Synthetic Domain Gap Analysis
To perform matched real vs synthetic comparisons, compute Cohen's d values, PCA projections, PSDs, and train a diagnostic domain classifier:

$env:PYTHONPATH="."
.\backend\venv\Scripts\python.exe ml/synthetic/domain_analysis.py
6. M7.1 Manual Test CLI Diagnostic Tool
To run the diagnostic CLI to parse, validate, and segment any input signal file:

$env:PYTHONPATH="."
.\backend\venv\Scripts\python.exe -m ml.input.pipeline_test "path/to/signal.wav"
7. M7.2 End-to-End Prediction CLI Tool
To execute the complete end-to-end signal prediction pipeline and print the predictions breakdown:

$env:PYTHONPATH="."
.\backend\venv\Scripts\python.exe -m ml.inference.pipeline_test "path/to/signal.wav"
8. Running Unit & Integration Tests
To run all 101 unit tests in the project (including synthetic generators, calibration tests, dataset loaders, baselines, CNNs, M7.1 input pipeline, and M7.2 end-to-end inference test suites):

$env:PYTHONPATH="."
.\backend\venv\Scripts\pytest