import joblib
import pandas as pd
from pathlib import Path

# Connect to the new, fixed model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "lightgbm_scrap_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "model_features.pkl"

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

# In-memory buffer to hold live sensor readings as they stream in one-by-one
live_buffer = {}

def predict_from_raw(payload):
    # Handle the incoming data
    machine_id = getattr(payload, 'machine_id', None)
    var_name = getattr(payload, 'variable_name', None)
    val = getattr(payload, 'value', None)
    
    if machine_id not in live_buffer:
        live_buffer[machine_id] = {}
        
    # Update the machine's buffer with the newest sensor reading
    live_buffer[machine_id][var_name] = val
    
    # Prepare the data for the AI model
    current_state = live_buffer[machine_id]
    df = pd.DataFrame([current_state])
    
    # Fill any sensors the machine hasn't sent yet with 0
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
            
    X = df[features]
    
    # Generate the live risk score!
    if hasattr(model, 'predict_proba'):
        risk_score = float(model.predict_proba(X)[:, 1][0])
    else:
        risk_score = float(model.predict(X)[0])
        
    return {
        "machine_id": machine_id,
        "risk_score": risk_score,
        "sensors_buffered": len(current_state),
        "status": "success"
    }