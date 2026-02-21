import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

DATA_FILE = Path("processed/features/rolling_features_wide_labeled.parquet")
MODEL_OUT = Path("models/lightgbm_scrap_risk_wide.pkl")

TARGET = "early_scrap_risk"
TIME_COL = "event_timestamp"
DROP_COLS = ["machine_id_normalized", TIME_COL, TARGET]

TRAIN_SAMPLE_FRAC = 0.25        # 🔑 critical
VALID_MAX_ROWS = 300_000        # 🔑 cap validation size
RANDOM_STATE = 42

print("Loading labeled wide data (metadata only)...")
df = pd.read_parquet(DATA_FILE)

# ------------------------------
# Time split
# ------------------------------
df = df.sort_values(TIME_COL)

split_ts = df[TIME_COL].quantile(0.20)

train_df = df[df[TIME_COL] < split_ts]
val_df   = df[df[TIME_COL] >= split_ts]

# ------------------------------
# Downsample TRAIN only
# ------------------------------
train_df = train_df.sample(
    frac=TRAIN_SAMPLE_FRAC,
    random_state=RANDOM_STATE
)

# ------------------------------
# Cap validation size
# ------------------------------
if len(val_df) > VALID_MAX_ROWS:
    val_df = val_df.sample(
        n=VALID_MAX_ROWS,
        random_state=RANDOM_STATE
    )

print("Train rows:", len(train_df))
print("Validation rows:", len(val_df))

# ------------------------------
# Prepare features safely
# ------------------------------
FEATURE_COLS = [
    c for c in df.columns
    if c not in DROP_COLS
]

X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
y_train = train_df[TARGET].to_numpy(dtype=np.int8)

X_val = val_df[FEATURE_COLS].to_numpy(dtype=np.float32)
y_val = val_df[TARGET].to_numpy(dtype=np.int8)

del df, train_df, val_df  # 🔑 free memory

# ------------------------------
# LightGBM datasets
# ------------------------------
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=True)
lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=True)

# ------------------------------
# Model parameters (industrial)
# ------------------------------
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 200,
    "verbose": -1,
    "num_threads": -1
}

print("Training LightGBM (wide model)...")

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_val],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(25)
    ]
)

MODEL_OUT.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_OUT)

print("Model saved to:", MODEL_OUT)
