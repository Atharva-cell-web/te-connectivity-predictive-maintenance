import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/lightgbm_scrap_risk_wide.pkl"
DATA_PATH = "processed/features/rolling_features_wide.parquet"
FEATURE_NAMES_PATH = "processed/features/rolling_feature_columns.txt"

NON_FEATURE_COLS = {
    "machine_id_normalized",
    "event_timestamp",
    "early_scrap_risk"
}

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading feature data...")
df = pd.read_parquet(DATA_PATH)

print("Loading feature names...")
with open(FEATURE_NAMES_PATH) as f:
    all_cols = [line.strip() for line in f.readlines()]

# 🔑 keep ONLY real model features
FEATURE_NAMES = [c for c in all_cols if c not in NON_FEATURE_COLS]

importances = model.feature_importance(importance_type="gain")

# 🔒 hard safety check (industry habit)
assert len(FEATURE_NAMES) == len(importances), (
    f"Feature mismatch: names={len(FEATURE_NAMES)}, "
    f"importances={len(importances)}"
)

# pick one example timestamp (latest)
row = df.sort_values("event_timestamp").iloc[-1]

X = (
    row[FEATURE_NAMES]
    .to_numpy(dtype=np.float32)
    .reshape(1, -1)
)

risk = model.predict(X)[0]
print(f"\nPredicted early scrap risk: {risk:.3f}")

imp_df = (
    pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": importances
    })
    .sort_values("importance", ascending=False)
)

print("\nTop contributing features (interpretable):")
print(imp_df.head(10))
