import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import linregress

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "processed" / "features" / "rolling_features_wide_labeled.parquet"
OUTPUT_FILE = PROJECT_ROOT / "processed" / "features" / "rolling_features_pattern_aware.parquet"

# Key sensors to compute rolling features for
KEY_SENSORS = [
    "Injection_pressure", "Cycle_time", "Peak_pressure_time", "Switch_pressure",
    "Cushion", "Holding_pressure", "Cyl_Tmp_Z1", "Cyl_Tmp_Z2", "Cyl_Tmp_Z3",
    "Cyl_Tmp_Z4", "Cyl_Tmp_Z5", "Dosing_time", "Injection_time"
]

WINDOW_SIZE = 10


def compute_slope(series):
    if len(series) < 3 or series.isna().all():
        return 0.0
    clean = series.dropna()
    if len(clean) < 3:
        return 0.0
    x = np.arange(len(clean))
    try:
        slope, _, _, _, _ = linregress(x, clean.values)
        return slope if np.isfinite(slope) else 0.0
    except Exception:
        return 0.0


def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Adding pattern-aware features to {len(df)} rows...")
    
    if "machine_id_normalized" not in df.columns:
        if "machine_id" in df.columns:
            df["machine_id_normalized"] = df["machine_id"]
        else:
            df["machine_id_normalized"] = "ALL"
    
    for sensor in KEY_SENSORS:
        if sensor not in df.columns:
            print(f"Sensor {sensor} not in dataframe, skipping...")
            continue
        
        print(f"  Processing {sensor}...")
        
        df[f"{sensor}_std_10"] = df.groupby("machine_id_normalized")[sensor].transform(
            lambda x: x.rolling(WINDOW_SIZE, min_periods=3).std()
        )
        
        df[f"{sensor}_trend_10"] = df.groupby("machine_id_normalized")[sensor].transform(
            lambda x: x.rolling(WINDOW_SIZE, min_periods=3).apply(compute_slope, raw=False)
        )
        
        df[f"{sensor}_mean_10"] = df.groupby("machine_id_normalized")[sensor].transform(
            lambda x: x.rolling(WINDOW_SIZE, min_periods=3).mean()
        )
        
        rolling_mean = df[f"{sensor}_mean_10"]
        df[f"{sensor}_dev_from_mean"] = (df[sensor] - rolling_mean) / (rolling_mean.abs() + 1e-6)
        
        std_col = df[f"{sensor}_std_10"]
        threshold = std_col.quantile(0.75)
        df[f"{sensor}_is_stable"] = (std_col < threshold).astype(int)
    
    pattern_cols = [c for c in df.columns if any(s in c for s in ["_std_10", "_trend_10", "_mean_10", "_dev_from_mean", "_is_stable"])]
    df[pattern_cols] = df[pattern_cols].fillna(0.0)
    
    print(f"Added {len(pattern_cols)} pattern-aware features.")
    return df


def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE, engine="pyarrow")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    
    df = add_pattern_features(df)
    
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, engine="pyarrow", index=False)
    print(f"Pattern-aware features saved to {OUTPUT_FILE}")
    print(f"[INFO] New column count: {len(df.columns)}")


if __name__ == "__main__":
    main()
