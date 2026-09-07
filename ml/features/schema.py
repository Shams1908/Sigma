from typing import List, Dict

# Central source of truth for the 36 extracted features in M3.
# Ordered deterministically: Amplitude -> Phase -> Frequency -> Cumulants -> Autocorrelation.
FEATURE_NAMES: List[str] = [
    # 1. Amplitude Features (4)
    "amplitude_mean",
    "amplitude_variance",
    "amplitude_kurtosis",
    "amplitude_peak_to_average_ratio",
    
    # 2. Phase Features (12)
    "phase_variance",
    "phase_difference_mean",
    "phase_difference_variance",
    "phase_difference_kurtosis",
    "phase_hist_bin_0",
    "phase_hist_bin_1",
    "phase_hist_bin_2",
    "phase_hist_bin_3",
    "phase_hist_bin_4",
    "phase_hist_bin_5",
    "phase_hist_bin_6",
    "phase_hist_bin_7",
    
    # 3. Instantaneous Frequency Features (2)
    "instantaneous_frequency_mean",
    "instantaneous_frequency_variance",
    
    # 4. Normalized C40 Cumulant Features (3)
    "c40_real",
    "c40_imag",
    "c40_magnitude",
    
    # 5. Autocorrelation Features (15: 5 lags * 3 components)
    "autocorr_lag_1_real",
    "autocorr_lag_1_imag",
    "autocorr_lag_1_magnitude",
    "autocorr_lag_2_real",
    "autocorr_lag_2_imag",
    "autocorr_lag_2_magnitude",
    "autocorr_lag_4_real",
    "autocorr_lag_4_imag",
    "autocorr_lag_4_magnitude",
    "autocorr_lag_8_real",
    "autocorr_lag_8_imag",
    "autocorr_lag_8_magnitude",
    "autocorr_lag_16_real",
    "autocorr_lag_16_imag",
    "autocorr_lag_16_magnitude",
]

# Stable mapping of feature name to index
FEATURE_TO_INDEX: Dict[str, int] = {
    name: idx for idx, name in enumerate(FEATURE_NAMES)
}

# Number of features
NUM_FEATURES: int = len(FEATURE_NAMES)
