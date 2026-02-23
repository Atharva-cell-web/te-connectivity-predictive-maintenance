"""
Train Recursive Sensor Forecaster with Auto-Regressive Lag Features
====================================================================
Trains a MultiOutputRegressor(LGBMRegressor) that predicts the NEXT
time-step values for every raw sensor defined in SAFE_LIMITS.

Each raw sensor gets 5 lag features (t-1 through t-5) so the model
understands recent momentum and avoids sudden unphysical drops.

Input features (X)  = [sensor_current, sensor_lag_1, ..., sensor_lag_5]
                       for all 14 raw sensors  →  14 * 6 = 84 features
Target (y)          = shift(-1) of raw sensors  →  14 targets

Usage:
    cd "d:\\te connectivity 3"
    python scripts/train_sensor_forecaster.py

Output:
    models/sensor_forecaster_lagged.pkl
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "new_processed_data" / "FEB_TEST_RESULTS.parquet"
DEMO_FILE = PROJECT_ROOT / "processed" / "features" / "rolling_features_demo.parquet"
MODEL_OUT = PROJECT_ROOT / "models" / "sensor_forecaster_lagged.pkl"

NUM_LAGS = 5  # look back 5 time steps

# Allow importing from backend/
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from config_limits import SAFE_LIMITS  # noqa: E402

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import joblib
    import numpy as np
    import pandas as pd
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor

    # 1. Determine raw sensor columns from SAFE_LIMITS keys
    raw_sensors = sorted(SAFE_LIMITS.keys())
    print(f"[INFO] Raw sensor columns ({len(raw_sensors)}): {raw_sensors}")

    # 2. Load data
    data_path = DATA_FILE if DATA_FILE.exists() else DEMO_FILE
    if not data_path.exists():
        raise FileNotFoundError(
            f"No data file found. Checked:\n  {DATA_FILE}\n  {DEMO_FILE}"
        )
    print(f"[INFO] Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"[INFO] Loaded {len(df):,} rows x {df.shape[1]} columns")

    # 3. Keep only sensor columns that exist in the dataframe
    available = [s for s in raw_sensors if s in df.columns]
    missing = set(raw_sensors) - set(available)
    if missing:
        print(f"[WARN] Sensors not in dataframe (skipped): {sorted(missing)}")
    if not available:
        raise ValueError("None of the SAFE_LIMITS sensors found in the data!")
    raw_sensors = available
    print(f"[INFO] Using {len(raw_sensors)} sensors for training")

    # 4. Sort by timestamp if present, then work with sensor columns only
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    sensor_df = df[raw_sensors].copy()
    sensor_df = sensor_df.apply(pd.to_numeric, errors="coerce")

    # 5. Create lag features for each sensor (t-1 through t-NUM_LAGS)
    print(f"[INFO] Creating {NUM_LAGS} lag features per sensor ...")
    lag_columns = []
    for sensor in raw_sensors:
        for lag in range(1, NUM_LAGS + 1):
            col_name = f"{sensor}_lag_{lag}"
            sensor_df[col_name] = sensor_df[sensor].shift(lag)
            lag_columns.append(col_name)

    # Input features = current raw sensors + all lag columns
    input_features = raw_sensors + lag_columns
    print(f"[INFO] Total input features: {len(input_features)}  "
          f"({len(raw_sensors)} current + {len(lag_columns)} lags)")

    # 6. Create target: shift each raw sensor by -1 (predict next time step)
    target_df = sensor_df[raw_sensors].shift(-1)

    # 7. Drop rows with NaN (from lags at the top + shift at the bottom)
    valid_mask = (
        sensor_df[input_features].notna().all(axis=1)
        & target_df.notna().all(axis=1)
    )
    X = sensor_df.loc[valid_mask, input_features].values
    y = target_df.loc[valid_mask].values
    print(f"[INFO] Training samples after NaN removal: {X.shape[0]:,}")
    print(f"[INFO] Feature shape: {X.shape}  |  Target shape: {y.shape}")

    # 8. Build model: MultiOutputRegressor wrapping LGBMRegressor
    base_estimator = LGBMRegressor(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1,
    )
    model = MultiOutputRegressor(base_estimator)

    print("[INFO] Training MultiOutputRegressor(LGBMRegressor) ...")
    model.fit(X, y)
    print("[INFO] Training complete.")

    # 9. Quick sanity check — predict on last row
    sample_pred = model.predict(X[-1:])
    print(f"[INFO] Sanity check — last input (first 6):  {np.round(X[-1, :6], 4)}")
    print(f"[INFO] Sanity check — prediction:            {np.round(sample_pred[0], 4)}")

    # 10. Save model + metadata
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "sensor_columns": raw_sensors,
        "input_features": input_features,
        "num_lags": NUM_LAGS,
    }
    joblib.dump(artifact, MODEL_OUT)
    print(f"[INFO] Model saved to: {MODEL_OUT}")
    print(f"[INFO] File size: {MODEL_OUT.stat().st_size / 1024:.1f} KB")
    print(f"[INFO] Artifact keys: {list(artifact.keys())}")


if __name__ == "__main__":
    main()
