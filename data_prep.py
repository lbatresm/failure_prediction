"""We'll analyze discriminator_calibration with ML to predict malfunctions"""

import pandas as pd
import numpy as np
from pathlib import Path
import helpers as hp

TRAINING_DATA = Path(__file__).parent / "training_data/disc_calibration.tsv"
training_cal = pd.read_csv(TRAINING_DATA, sep="\t")

def safe_division (a, b):
    """
    Doing a normal division is risky since:
    - Dividing by zero returns either inf of NaN
    - Very small values in denominator make ratios huge and creates noise in the models.

    This safe division returns a/b if denominator is big enough, and NaN if it is too small.
    """
    return np.where(np.abs(b) > 1e-9, a/b, np.nan)

def _include_chip_uid (df: pd.DataFrame) -> pd.DataFrame:
    """
    Includes a new "chip_uid" column to df combining portID, slaveID and chipID in a single chain.
    Eg: "1-0-12"
    """
    df = df.copy()
    df["chip_uid"] = (
            df["portID"].astype(str) + "-" +
            df["slaveID"].astype(str) + "-" +
            df["chipID"].astype(str)
    )
    # Move to first position
    col = df.pop("chip_uid")
    df.insert(0, "chip_uid", col)

    df = df.drop(["portID", "slaveID", "chipID"], axis=1)

    return df

def _compute_relative_patterns (df: pd.DataFrame) -> pd.DataFrame:
    # Deltas
    relative_patterns = df.copy()
    relative_patterns["zero"] = relative_patterns["zero_T2"] - relative_patterns["zero_T1"]
    relative_patterns["d_noise_T"] = relative_patterns["noise_T2"] - relative_patterns["noise_T1"]

    # Differences
    relative_patterns["baseline_T_minus_E"] = relative_patterns["baseline_T"] - relative_patterns["baseline_E"]
    relative_patterns["zero_T1_minus_E"] = relative_patterns["zero_T1"] - relative_patterns["zero_E"]
    relative_patterns["zero_T2_minus_E"] = relative_patterns["zero_T2"] - relative_patterns["zero_E"]
    relative_patterns["noise_T1_minus_E"] = relative_patterns["noise_T1"] - relative_patterns["noise_E"]
    relative_patterns["noise_T2_minus_E"] = relative_patterns["noise_T2"] - relative_patterns["noise_E"]

    # Adimensional ratios
    relative_patterns["r_zero_T2_T1"] = safe_division (relative_patterns["zero_T2"], relative_patterns["zero_T1"])
    relative_patterns["r_noise_T2_T1"] = safe_division (relative_patterns["noise_T2"], relative_patterns["noise_T1"])
    relative_patterns["r_baseline_T_E"] = safe_division (relative_patterns["baseline_T"], relative_patterns["baseline_E"])

    return relative_patterns

def _compute_local_context (df: pd.DataFrame, focus_cols = None, rolling_window:int = 5) -> pd.DataFrame:

    df = df.copy()

    if focus_cols is None:
        focus_cols = ["baseline_T", "baseline_E", "zero_T1","zero_T2","zero_E", "noise_T1", "noise_T2","noise_E"]

    df = df.sort_values(["chip_uid", "channelID"])

    """
    Rolling mean / Rolling standard deviation: Compares channels with its 4 nearest neighbours.
    Why? Healthy chips should have subtle differences between channels
    """

    for col in focus_cols:
        df[f"{col}_rollmean{rolling_window}"] = (
            df.groupby("chip_uid")[col]
            .transform(lambda s: s.rolling(rolling_window, center=True, min_periods=1).mean())
        )

        df[f"{col}_rollstd{rolling_window}"] = (
            df.groupby("chip_uid")[col]
            .transform(lambda s: s.rolling(rolling_window, center=True, min_periods=2).std())
        )


    """
    Z-score: shows the relative position of each channel with respect to the rest of the group in 'dispersion units'
    Values close two zero are closer to the mean, while big z-scores are further away from the mean (outliers)

    Robust z-score: we compute Z-score with the median, instead of the mean, since median is less sensitive
    to extreme values. Therefore we use MAD (Median Absolute Deviation) instead of std deviation

    rz = (x-median)/(1.4826*MAD).
    """
    def rz(series: pd.Series) -> pd.Series:
        med = series.median()
        mad = (series - med).abs().median()
        scale = 1.4826 * mad if mad > 1e-9 else np.nan
        if np.isfinite(scale):
            return (series - med) / scale
        else:
            return pd.Series(np.nan, index=series.index)

    for col in focus_cols:
        df[f"rz_{col}"] = df.groupby("chip_uid")[col].transform(rz)

    return df

def _compute_chip_features (training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Features are individual measurable properties of a data set. In this function we analyze the chip by computing
    aggregates of all its channels.

    - mean, std, min, max, median, q10, q90

    Returns a df with a single row per chip.
    """
    df = training_df.copy()

    # We won't take into account channels
    df.drop(["channelID"], axis=1, inplace=True)

    # Define columns where we want to apply aggregate functions
    agg_cols = df.columns.tolist()
    agg_cols.remove("chip_uid")

    # Define aggregate functions
    agg_funcs = {
        c: ["mean", "std", "min", "max", "median",
            ("q10", lambda x: x.quantile(0.1)),
            ("q90", lambda x: x.quantile(0.9))]
        for c in agg_cols
    }

    # Apply functions
    chip_agg = (
        df.groupby("chip_uid")[agg_cols].agg(agg_funcs)
    )

    # Flatten column names
    chip_agg.columns = [
        "_".join([str(c) for c in col if c != ""])
        if isinstance(col, tuple) else col
        for col in chip_agg.columns
    ]
    chip_agg = chip_agg.rename(
        columns=lambda c: c.replace("<lambda_0>", "q10").replace("<lambda_1>", "q90")
    )

    return chip_agg


def main():
    # Include chip_uid and remove unneeded columns
    identified_df = _include_chip_uid(training_cal)

    # Add relative patterns (deltas, differences, ratios)
    relative_patterns = _compute_relative_patterns(identified_df)

    # Add local context (mean, std, z-score). Export.
    local_context = _compute_local_context(relative_patterns)
    hp.df_to_csv_no_index(local_context, "a_channel_training_data")

    # Create chip features table with aggregate functions for all channels (mean, std, min, max...). Export.
    chip_features = _compute_chip_features(identified_df)
    hp.df_to_csv_index(chip_features, "b_chip_training_data")

    print("Workflow finished")



if __name__ == "__main__":
    main()
