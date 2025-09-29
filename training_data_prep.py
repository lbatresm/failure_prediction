"""We'll analyze discriminator_calibration with ML to predict malfunctions"""

import pandas as pd
import numpy as np
from pathlib import Path

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
    Includes a new 'chip_uid' column to df combining portID, slaveID and chipID in a single chain.
    Eg: '1-0-12'
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

    df = df.drop(['portID', 'slaveID', 'chipID'], axis=1)

    return df

def _compute_relative_patterns (df: pd.DataFrame) -> pd.DataFrame:
    # Deltas
    relative_patterns = df.copy()
    relative_patterns['zero'] = relative_patterns['zero_T2'] - relative_patterns['zero_T1']
    relative_patterns['d_noise_T'] = relative_patterns['noise_T2'] - relative_patterns['noise_T1']

    # Differences
    relative_patterns['baseline_T_minus_E'] = relative_patterns['baseline_T'] - relative_patterns['baseline_E']
    relative_patterns['zero_T1_minus_E'] = relative_patterns['zero_T1'] - relative_patterns['zero_E']
    relative_patterns['zero_T2_minus_E'] = relative_patterns['zero_T2'] - relative_patterns['zero_E']
    relative_patterns['noise_T1_minus_E'] = relative_patterns['noise_T1'] - relative_patterns['noise_E']
    relative_patterns['noise_T2_minus_E'] = relative_patterns['noise_T2'] - relative_patterns['noise_E']

    # Adimensional ratios
    relative_patterns['r_zero_T2_T1'] = safe_division (relative_patterns['zero_T2'], relative_patterns['zero_T1'])
    relative_patterns['r_noise_T2_T1'] = safe_division (relative_patterns['noise_T2'], relative_patterns['noise_T1'])
    relative_patterns['r_baseline_T_E'] = safe_division (relative_patterns['baseline_T'], relative_patterns['baseline_E'])

    return relative_patterns

def _compute_local_context (df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    # TODO: Empiezo aplicando local contexts a baseline_T. Probar con otros y comparar fiabilidad modelo

    def _rolling_mean_std (df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling mean / Rolling standard deviation: Compares channels with its 4 nearest neighbours.
        Why? Healthy chips should have subtle differences between channels
        """
        df = df.sort_values(['chip_uid', 'channelID'])

        df['baseline_T_rollmean5'] = (
            df.groupby('chip_uid')['baseline_T'].transform(lambda s: s.rolling(5, center=True, min_periods=1).mean())
        )

        df["baseline_T_rollstd5"] = (
            df.groupby("chip_uid")["baseline_T"].transform(lambda s: s.rolling(5, center=True, min_periods=2).std())
        )

        return df

    def _robust_z_score (df: pd.DataFrame) -> pd.DataFrame:
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

        # Example on baseline_T; you can replicate for other signals
        df["rz_baseline_T"] = df.groupby("chip_uid")["baseline_T"].transform(rz)

        return df

    df1 = _rolling_mean_std(df)
    df2 = _robust_z_score(df1)
    return df2

def main():
    identified_df = _include_chip_uid(training_cal)
    relative_patterns = _compute_relative_patterns(identified_df)
    local_context = _compute_local_context(relative_patterns)
    print (relative_patterns)

if __name__ == "__main__":
    main()
