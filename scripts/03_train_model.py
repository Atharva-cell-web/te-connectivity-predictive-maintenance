import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# --- CONFIGURATION ---
PROCESSED_DIR = r"D:\te connectivity 3\new_processed_data"
MODEL_DIR = r"D:\te connectivity 3\models"

# Ensure the models folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

print("🚀 Step 1: Loading Master Training Data...")
df = pd.read_parquet(os.path.join(PROCESSED_DIR, "FINAL_TRAINING_MASTER.parquet"))

# --- FEATURE ENGINEERING ---
print("🧠 Step 2: Preparing features...")
# Isolate the sensors (X) from the answer (y)
drop_cols = ['timestamp', 'machine_id', 'is_scrap']
features = [col for col in df.columns if col not in drop_cols]

X = df[features]
y = df['is_scrap']

# Split the data 80/20 so we can test the AI on data it didn't see during the loop
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- MODEL TRAINING ---
print(f"\n⚙️ Step 3: Training LightGBM Model on {len(X_train)} rows...")
print("   -> Using 'is_unbalance=True' to force AI to focus on rare Scrap events...")

model = lgb.LGBMClassifier(
    n_estimators=300,        # Number of decision trees
    learning_rate=0.05,      # How fast it learns
    is_unbalance=True,       # CRITICAL for rare scrap detection
    random_state=42,
    n_jobs=-1                # Use all CPU cores for speed
)

# Train the model and watch it learn
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=30), 
        lgb.log_evaluation(50)
    ]
)

# --- EVALUATION ---
print("\n📊 Step 4: Internal Validation Results (on the 20% holdout):")
y_pred = model.predict(X_val)

print("\n--- Classification Report ---")
print(classification_report(y_val, y_pred, target_names=['Normal (0)', 'Scrap (1)']))

# --- SAVE MODEL ---
model_path = os.path.join(MODEL_DIR, 'lightgbm_scrap_model.pkl')
joblib.dump(model, model_path)

# Save the feature names so the backend knows exactly what sensors to look for
joblib.dump(features, os.path.join(MODEL_DIR, 'model_features.pkl'))

print(f"\n✅ SUCCESS! New AI Brain saved to: {model_path}")