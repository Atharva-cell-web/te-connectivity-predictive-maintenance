"""
Train a multi-output sensor forecaster for the 21-feature pipeline.

Uses a LightGBM Regressor wrapped in MultiOutputRegressor to predict the
next time step of all 21 sensors from `num_lags` previous steps.

Output: models/sensor_forecaster_lagged.pkl  (dict artifact compatible
        with data_access._load_sensor_forecaster)
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "new_processed_data"
MODEL_DIR = PROJECT_ROOT / "models"
FEATURES_PATH = MODEL_DIR / "model_features.pkl"
OUTPUT_PATH = MODEL_DIR / "sensor_forecaster_lagged.pkl"

NUM_LAGS = 3  # how many past steps the model sees

# ── 1. Load sensor feature names ────────────────────────────────────
sensor_columns = list(joblib.load(FEATURES_PATH))
print(f"[1/5] Loaded {len(sensor_columns)} sensor columns: {sensor_columns[:5]}...")

# ── 2. Load & pivot raw machine data ────────────────────────────────
# Gather all *_TEST.parquet files so the forecaster sees multiple machines.
parquet_files = sorted(DATA_DIR.glob("*_TEST.parquet"))
if not parquet_files:
    print("ERROR: No *_TEST.parquet files found in", DATA_DIR)
    sys.exit(1)

print(f"[2/5] Found {len(parquet_files)} parquet file(s): {[f.name for f in parquet_files]}")

wide_frames = []
for pf in parquet_files:
    raw = pd.read_parquet(pf, engine="pyarrow")
    if "variable_name" not in raw.columns or "value" not in raw.columns:
        print(f"  Skipping {pf.name} (not in long format)")
        continue

    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["value"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp"])

    pivot = (
        raw.pivot_table(index="timestamp", columns="variable_name",
                        values="value", aggfunc="mean")
        .sort_index()
        .reset_index()
    )
    # Keep only the 21 sensors we care about
    available = [c for c in sensor_columns if c in pivot.columns]
    if len(available) < 5:
        print(f"  Skipping {pf.name} (only {len(available)} usable sensors)")
        continue

    sub = pivot[["timestamp"] + available].copy()
    # Fill remaining sensors with 0
    for c in sensor_columns:
        if c not in sub.columns:
            sub[c] = 0.0
    wide_frames.append(sub)
    print(f"  {pf.name}: {len(sub)} rows, {len(available)} sensors")

if not wide_frames:
    print("ERROR: No usable data found.")
    sys.exit(1)

wide = pd.concat(wide_frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
# Forward-fill small gaps within each sensor
wide[sensor_columns] = wide[sensor_columns].ffill().bfill().fillna(0.0)
print(f"[3/5] Combined dataset: {len(wide)} rows × {len(sensor_columns)} sensors")

# ── 3. Build lagged feature matrix ──────────────────────────────────
# Input = [sensor_t, sensor_lag_1, sensor_lag_2, ...]  → Output = sensor_t+1
input_feature_names = []
for s in sensor_columns:
    input_feature_names.append(s)  # current value
for lag in range(1, NUM_LAGS + 1):
    for s in sensor_columns:
        input_feature_names.append(f"{s}_lag_{lag}")

X_rows = []
y_rows = []

sensor_arr = wide[sensor_columns].values  # shape (N, 21)

for i in range(NUM_LAGS, len(sensor_arr) - 1):
    row_input = []
    # Current (t)
    row_input.extend(sensor_arr[i].tolist())
    # Lags
    for lag in range(1, NUM_LAGS + 1):
        row_input.extend(sensor_arr[i - lag].tolist())
    X_rows.append(row_input)
    # Target = next step (t+1)
    y_rows.append(sensor_arr[i + 1].tolist())

X = np.array(X_rows, dtype=np.float32)
y = np.array(y_rows, dtype=np.float32)
print(f"[4/5] Training matrix: X={X.shape}, y={y.shape}")

# ── 4. Train ────────────────────────────────────────────────────────
print("      Training MultiOutputRegressor(LGBMRegressor) ...")
base_estimator = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    verbose=-1,
)
estimator = MultiOutputRegressor(base_estimator, n_jobs=-1)
estimator.fit(X, y)

# Quick in-sample sanity check
preds = estimator.predict(X[:5])
print(f"      Sample predictions (first row): {np.round(preds[0][:5], 2)}")
print(f"      Sample actuals    (first row): {np.round(y[0][:5], 2)}")

# ── 5. Save artifact ────────────────────────────────────────────────
artifact = {
    "model": estimator,
    "sensor_columns": sensor_columns,
    "input_features": input_feature_names,
    "num_lags": NUM_LAGS,
    "hydra_features": [],  # none needed for the simple model
}
joblib.dump(artifact, OUTPUT_PATH)
print(f"[5/5] Saved forecaster to {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")
print("Done!")
