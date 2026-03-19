import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, classification_report

model = joblib.load("models/lightgbm_scrap_risk_wide.pkl")
n_model = model.num_feature()
print(f"Model expects: {n_model} features")

df = pd.read_parquet("processed/features/rolling_features_with_context.parquet")

# All rolling feature columns except Time_on_machine (added later, not in model)
# Model was trained on 25 sensors x 15 stats = 375 cols
EXCLUDE = {
    "event_timestamp", "machine_id_normalized", "early_scrap_risk",
    "tool_id_encoded", "part_number_encoded", "machine_id_encoded"
}
feature_cols = [c for c in df.columns
                if c not in EXCLUDE
                and pd.api.types.is_numeric_dtype(df[c])
                and not c.startswith("Time_on_machine")]

print(f"Feature cols after excluding Time_on_machine: {len(feature_cols)}")

if len(feature_cols) != n_model:
    print(f"STILL MISMATCH: got {len(feature_cols)}, need {n_model}")
    print("Extra/missing — check manually")
else:
    y = df["early_scrap_risk"].fillna(0).astype(int)
    X = df[feature_cols].fillna(0).values

    print(f"Rows: {len(y)} | Scrap: {int(y.sum())} | Rate: {round(y.mean()*100,3)}%")
    print("Running model.predict()...")

    probs = model.predict(X)

    print("\n=== DEPLOYED MODEL: lightgbm_scrap_risk_wide.pkl ===")
    print("--- Threshold 0.5 ---")
    print(classification_report(y, (probs >= 0.5).astype(int),
                                 target_names=["Normal","Scrap"], digits=4))
    print("AUC:", round(roc_auc_score(y, probs), 4))

    print("\n--- Threshold 0.7 (current dashboard setting) ---")
    print(classification_report(y, (probs >= 0.7).astype(int),
                                 target_names=["Normal","Scrap"], digits=4))

    print("\n=== THRESHOLD SEARCH (precision 10% → 80%) ===")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}  <-- pick row where F1 is highest")
    pa, ra, ta = precision_recall_curve(y, probs)
    last_bucket = 0
    for p, r, t in zip(pa, ra, ta):
        f1 = (2*p*r/(p+r)) if (p+r) > 0 else 0
        bucket = int(p * 20) * 5
        if bucket >= 10 and bucket > last_bucket:
            print(f"{round(t,3):>10} {round(p*100,1):>9}% {round(r*100,1):>7}% {round(f1,3):>8}")
            last_bucket = bucket
        if p >= 0.80:
            break

    print("\nSave these results — this is your ground truth.")