"""
Shared feature extraction — imported by both ../../model_training/preprocess_data.py (model training)
and main.py (API), so the exact same transformation is applied at training time and prediction time.
"""
import numpy as np
import pandas as pd
from scipy import stats

def extract_features(flux_array):
    """
    Takes a 2D numpy array of shape (n_rows, n_flux_columns) and returns
    a DataFrame of engineered features: one row per input row.
    """
    n_rows = flux_array.shape[0]

    # --- Time-domain stats (per row) ---
    row_mean = flux_array.mean(axis=1)
    row_std = flux_array.std(axis=1)
    row_var = flux_array.var(axis=1)
    row_skew = stats.skew(flux_array, axis=1)
    row_kurtosis = stats.kurtosis(flux_array, axis=1)
    row_min = flux_array.min(axis=1)
    row_max = flux_array.max(axis=1)
    row_range = row_max - row_min

    # --- FFT (frequency-domain) features ---
    # Remove each row's own mean first (DC component) — do this per-row,
    # never using a global/train-wide mean, so it works identically on
    # a single new row at inference time.
    centered = flux_array - row_mean[:, None]
    fft_vals = np.fft.rfft(centered, axis=1)
    fft_magnitude = np.abs(fft_vals)  # shape: (n_rows, n_flux_cols//2 + 1)

    fft_max_mag = fft_magnitude.max(axis=1)
    fft_dominant_freq = fft_magnitude.argmax(axis=1)
    fft_energy = np.sum(fft_magnitude ** 2, axis=1)
    fft_mean_mag = fft_magnitude.mean(axis=1)
    fft_std_mag = fft_magnitude.std(axis=1)

    # Top-5 FFT peaks per row (captures multi-planet / multi-periodic signals)
    top5_idx = np.argsort(fft_magnitude, axis=1)[:, -5:]
    top5_vals = np.take_along_axis(fft_magnitude, top5_idx, axis=1)
    top5_vals_sorted = np.sort(top5_vals, axis=1)[:, ::-1]  # largest first

    features = pd.DataFrame({
        "row_mean": row_mean,
        "row_std": row_std,
        "row_var": row_var,
        "row_skew": row_skew,
        "row_kurtosis": row_kurtosis,
        "row_range": row_range,
        "fft_max_mag": fft_max_mag,
        "fft_dominant_freq": fft_dominant_freq,
        "fft_energy": fft_energy,
        "fft_mean_mag": fft_mean_mag,
        "fft_std_mag": fft_std_mag,
    })

    for i in range(5):
        features[f"fft_top{i+1}_mag"] = top5_vals_sorted[:, i]

    # Log-transform the features that span huge ranges (variance, energy,
    # magnitude). log1p handles zero/near-zero values safely.
    for col in ["row_var", "row_std", "fft_max_mag", "fft_energy",
                "fft_mean_mag", "fft_std_mag"] + [f"fft_top{i+1}_mag" for i in range(5)]:
        features[col] = np.log1p(features[col].clip(lower=0))

    return features