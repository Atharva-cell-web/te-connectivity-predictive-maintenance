import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_scrap_risk_wide.pkl"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def predict_risk(model, X):
    # Returns the probability of class 1 (Scrap Risk)
    return float(model.predict(X)[0])