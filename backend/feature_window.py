import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_MAP_FILE = PROJECT_ROOT / "processed" / "features" / "rolling_feature_columns.txt"

def load_feature_columns():
    """Loads the master list of features used during training."""
    if not FEATURE_MAP_FILE.exists():
        raise FileNotFoundError(f"Feature map not found: {FEATURE_MAP_FILE}")
        
    with open(FEATURE_MAP_FILE, "r") as f:
        # Read lines and strip newlines
        cols = [line.strip() for line in f.readlines()]
    
    # Filter out non-features just like we did in training
    # (machine_id, timestamp, and target are NOT features)
    non_features = {"machine_id_normalized", "event_timestamp", "early_scrap_risk"}
    return [c for c in cols if c not in non_features]

def extract_ml_features(snapshot_row):
    """
    Converts a single row (pandas Series) into a 2D numpy array 
    sorted exactly as the model expects.
    """
    feature_cols = load_feature_columns()
    
    # Create the vector. Use 0.0 for missing features to be safe.
    # This aligns the live data columns to the training columns order.
    values = [float(snapshot_row.get(col, 0.0)) for col in feature_cols]
    
    return np.array(values, dtype="float32").reshape(1, -1)