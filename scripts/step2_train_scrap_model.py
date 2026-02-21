import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

print("Loading training data...")
df = pd.read_csv("processed/features/event_level_training.csv")

# 1. Separate features and target
# Separate target
y = df["scrap_flag"]

# Keep only numeric features
X = df.drop(columns=["scrap_flag"])
X = X.select_dtypes(include=["number"])
# Remove counter-based features (non-causal)
bad_keywords = [
    "Scrap_counter",
    "Shot_counter"
]

keep_cols = [
    c for c in X.columns
    if not any(bad in c for bad in bad_keywords)
]

X = X[keep_cols]

print("Filtered feature count:", X.shape[1])


print("Final feature count:", X.shape[1])

print("Feature columns used for training:")
print(X.columns.tolist())


# 2. Time-based split (IMPORTANT)
split_index = int(len(df) * 0.7)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_test))

# 3. Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

# 4. Predict probabilities
y_pred_prob = model.predict_proba(X_test)[:, 1]

# 5. Evaluate
roc = roc_auc_score(y_test, y_pred_prob)
print("ROC-AUC:", round(roc, 4))

print("Confusion Matrix (threshold=0.5):")
print(confusion_matrix(y_test, y_pred_prob > 0.5))

# 6. Save model
joblib.dump(model, "models/scrap_probability_model.pkl")
print("Model saved to models/scrap_probability_model.pkl")
