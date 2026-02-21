import joblib
import pandas as pd
import lightgbm as lgb
import numpy as np

MODEL_PATH = "models/lightgbm_scrap_risk_wide.pkl"
DATA_PATH = "processed/features/rolling_features_wide.parquet"

TARGET_COL = "early_scrap_risk"
DROP_COLS = ["machine_id_normalized", "event_timestamp"]

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading feature data...")
df = pd.read_parquet(DATA_PATH)

# pick one example (latest row)
row = df.sort_values("event_timestamp").iloc[-1]

X = row.drop(DROP_COLS, errors="ignore").to_numpy(dtype=np.float32).reshape(1, -1)

risk = model.predict(X)[0]

print(f"\nPredicted early scrap risk: {risk:.3f}")

# ------------------------------
# Feature importance (gain)
# ------------------------------
importances = model.feature_importance(importance_type="gain")
features = model.feature_name()

imp_df = (
    pd.DataFrame({
        "feature": features,
        "importance": importances
    })
    .sort_values("importance", ascending=False)
)

print("\nTop contributing features:")
print(imp_df.head(10))
