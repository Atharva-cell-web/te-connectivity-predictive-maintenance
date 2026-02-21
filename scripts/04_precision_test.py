import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION ---
PROCESSED_DIR = r"D:\te connectivity 3\new_processed_data"
MODEL_DIR = r"D:\te connectivity 3\models"

print("🚀 Starting ULTIMATE PRECISION TEST (Feb 2026 Unseen Data)...")

# 1. Load Model and Features
model = joblib.load(os.path.join(MODEL_DIR, 'lightgbm_scrap_model.pkl'))
model_features = joblib.load(os.path.join(MODEL_DIR, 'model_features.pkl'))

# 2. Load Hydra Test Data (The "Ground Truth" for Feb)
hydra_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "HYDRA_TEST.parquet"))
scrap_events = hydra_test[hydra_test['scrap_quantity'] > 0].copy()
scrap_events['machine_id'] = scrap_events['machine_id'].astype(str).str.replace('-', '').str.upper().str.strip()
scrap_events['timestamp'] = pd.to_datetime(scrap_events['machine_event_create_date'], utc=True)
scrap_events['is_scrap_actual'] = 1

test_results = []

# 3. Process Test Data machine by machine
for file in os.listdir(PROCESSED_DIR):
    if file.endswith("_TEST.parquet") and "HYDRA" not in file:
        machine_id = file.split('_')[0]
        print(f"\n🔍 Testing Machine {machine_id}...")

        df = pd.read_parquet(os.path.join(PROCESSED_DIR, file))
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value'])

        # Pivot
        pivot_df = df.pivot_table(index='timestamp', columns='variable_name', values='value', aggfunc='mean').reset_index()
        pivot_df['timestamp'] = pd.to_datetime(pivot_df['timestamp'], utc=True)

        # Match features (fill missing sensors with 0)
        for col in model_features:
            if col not in pivot_df.columns:
                pivot_df[col] = 0
        
        X_test = pivot_df[model_features]

        # PREDICT
        pivot_df['predicted_scrap'] = model.predict(X_test)
        pivot_df['scrap_probability'] = model.predict_proba(X_test)[:, 1]

        # Merge with Actual Hydra Logs
        m_scrap = scrap_events[scrap_events['machine_id'] == machine_id]
        merged = pd.merge_asof(
            pivot_df.sort_values('timestamp'),
            m_scrap[['timestamp', 'is_scrap_actual']].sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta('5 minutes')
        )
        merged['is_scrap_actual'] = merged['is_scrap_actual'].fillna(0)
        test_results.append(merged)

# 4. Final Comparison
final_test_df = pd.concat(test_results)
print("\n🎯 FINAL PRECISION REPORT ON UNSEEN FEBRUARY DATA:")
print(classification_report(final_test_df['is_scrap_actual'], final_test_df['predicted_scrap']))

# Save results for dashboard
final_test_df.to_parquet(os.path.join(PROCESSED_DIR, "FEB_TEST_RESULTS.parquet"), index=False)
print("✅ Test results saved for Dashboard display.")