import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score
import joblib
from lightgbm import early_stopping, log_evaluation

# ================= CONFIG =================
DATA_FILE = Path("processed/features/rolling_training_sampled.parquet")
MODEL_OUT = Path("models/lightgbm_scrap_risk.pkl")
VALID_SPLIT_TIME = "2025-10-25"
# ==========================================

print("Loading data...")
df = pd.read_parquet(DATA_FILE)

print("Total rows:", len(df))

# ------------------------------
# Sort by time (CRITICAL)
# ------------------------------
df = df.sort_values("timestamp")

# ------------------------------
# Train / validation split (time-based)
# ------------------------------
train_df = df[df["timestamp"] < VALID_SPLIT_TIME]
val_df   = df[df["timestamp"] >= VALID_SPLIT_TIME]

print("Train rows:", len(train_df))
print("Validation rows:", len(val_df))

# ------------------------------
# Features / target
# ------------------------------
DROP_COLS = [
    "machine_id",
    "variable_name",
    "window",
    "timestamp",
    "early_scrap_risk"
]

X_train = train_df.drop(columns=DROP_COLS)
y_train = train_df["early_scrap_risk"]

X_val = val_df.drop(columns=DROP_COLS)
y_val = val_df["early_scrap_risk"]

# ------------------------------
# LightGBM datasets
# ------------------------------
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train)

# ------------------------------
# Model parameters
# ------------------------------
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1
}

print("Training LightGBM...")

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_val],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=50)
    ]
)

# ------------------------------
# Evaluate
# ------------------------------
val_preds = model.predict(X_val)
auc = roc_auc_score(y_val, val_preds)
print("Validation ROC-AUC:", round(auc, 4))

# ------------------------------
# Save model
# ------------------------------
MODEL_OUT.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_OUT)
print("Model saved to:", MODEL_OUT)
