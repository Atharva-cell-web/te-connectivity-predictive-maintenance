import joblib
import pandas as pd
from pathlib import Path

# ================= CONFIG =================
MODEL_FILE = Path("models/lightgbm_scrap_risk.pkl")
DATA_FILE = Path("processed/features/rolling_training_sampled.parquet")
SAFE_FILE = Path("processed/safe/AI_cup_parameter_info_cleaned.csv")
# ==========================================

print("Loading model and data...")
model = joblib.load(MODEL_FILE)

df = pd.read_parquet(DATA_FILE)

# Take a recent high-risk example
df = df.sort_values("timestamp", ascending=False)
sample = df[df["early_scrap_risk"] == 1].iloc[0:1]

DROP_COLS = [
    "machine_id",
    "variable_name",
    "window",
    "timestamp",
    "early_scrap_risk"
]

X = sample.drop(columns=DROP_COLS)

# Predict risk
risk = model.predict(X)[0]

# Feature contribution approximation using gain
feature_importance = pd.DataFrame({
    "feature": model.feature_name(),
    "importance_gain": model.feature_importance(importance_type="gain")
}).sort_values("importance_gain", ascending=False)

# Map to parameters
feature_importance["parameter"] = feature_importance["feature"].str.split("__").str[0]

top_params = (
    feature_importance.groupby("parameter", as_index=False)["importance_gain"]
    .sum()
    .sort_values("importance_gain", ascending=False)
    .head(5)
)

print("\nPredicted scrap risk:", round(risk, 3))
print("\nTop contributing parameters:")
print(top_params)
