import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
import joblib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_scrap_risk_wide_v2.pkl"
ENCODINGS_PATH = PROJECT_ROOT / "models" / "part_tool_encodings.json"

model = joblib.load(MODEL_PATH)
FEATURE_NAMES = model.feature_name()

ENCODINGS = {}
if ENCODINGS_PATH.exists():
    with open(ENCODINGS_PATH, "r") as f:
        ENCODINGS = json.load(f)

ROLLING_BUFFERS = {}
BUFFER_MAX_SIZE = 1000


def get_buffer(machine_id):
    if machine_id not in ROLLING_BUFFERS:
        ROLLING_BUFFERS[machine_id] = deque(maxlen=BUFFER_MAX_SIZE)
    return ROLLING_BUFFERS[machine_id]


def compute_rolling_features(buffer, sensor_cols):
    if len(buffer) < 3:
        return {}
    
    df = pd.DataFrame(list(buffer))
    if "_ts" not in df.columns:
        return {}
    
    df = df.set_index("_ts").sort_index()
    now = df.index[-1]
    features = {}
    
    windows = {"5m": timedelta(minutes=5), "15m": timedelta(minutes=15), "30m": timedelta(minutes=30)}
    
    for col in sensor_cols:
        if col not in df.columns:
            continue
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        for wname, wdelta in windows.items():
            wdata = col_data[col_data.index >= (now - wdelta)]
            if len(wdata) == 0:
                val = float(col_data.iloc[-1])
                features[f"{col}__mean_{wname}"] = val
                features[f"{col}__std_{wname}"] = 0.0
                features[f"{col}__min_{wname}"] = val
                features[f"{col}__max_{wname}"] = val
                features[f"{col}__last_{wname}"] = val
            else:
                features[f"{col}__mean_{wname}"] = float(wdata.mean())
                features[f"{col}__std_{wname}"] = float(wdata.std()) if len(wdata) > 1 else 0.0
                features[f"{col}__min_{wname}"] = float(wdata.min())
                features[f"{col}__max_{wname}"] = float(wdata.max())
                features[f"{col}__last_{wname}"] = float(wdata.iloc[-1])
    
    return features


def predict_from_raw(machine_id, raw_data):
    try:
        ts = raw_data.get("timestamp", datetime.now())
        ts_dt = pd.to_datetime(ts) if isinstance(ts, str) else ts
        
        record = raw_data.copy()
        record["_ts"] = ts_dt
        
        buffer = get_buffer(machine_id)
        buffer.append(record)
        
        exclude = {"timestamp", "machine_id", "machine_definition", "tool_id", "part_number", "_ts"}
        sensor_cols = [k for k in raw_data.keys() if k not in exclude]
        
        features = compute_rolling_features(buffer, sensor_cols)
        features["tool_id_encoded"] = ENCODINGS.get("tool_id", {}).get(raw_data.get("tool_id", "UNKNOWN"), 0)
        features["machine_id_encoded"] = ENCODINGS.get("machine_id", {}).get(machine_id.upper().replace("-", ""), 0)
        features["part_number_encoded"] = 0
        
        X = np.array([[features.get(name, 0.0) for name in FEATURE_NAMES]], dtype=np.float32)
        risk_score = float(model.predict(X)[0])
        
        if risk_score < 0.3:
            level = "LOW"
        elif risk_score < 0.6:
            level = "MEDIUM"
        else:
            level = "HIGH"
        
        return {
            "status": "success",
            "risk_score": round(risk_score, 4),
            "risk_level": level,
            "timestamp": str(ts),
            "machine_id": machine_id,
            "buffer_size": len(buffer)
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "machine_id": machine_id}


def get_buffer_status(machine_id=None):
    if machine_id:
        return {"machine_id": machine_id, "buffer_size": len(ROLLING_BUFFERS.get(machine_id, []))}
    return {"machines": list(ROLLING_BUFFERS.keys()), "sizes": {k: len(v) for k, v in ROLLING_BUFFERS.items()}}


def clear_buffer(machine_id=None):
    if machine_id and machine_id in ROLLING_BUFFERS:
        ROLLING_BUFFERS[machine_id].clear()
    elif not machine_id:
        ROLLING_BUFFERS.clear()


if __name__ == "__main__":
    from datetime import datetime
    base = datetime.now()
    
    for i in range(10):
        data = {
            "timestamp": (base + timedelta(seconds=i*2)).isoformat(),
            "Cycle_time": 12.5 + np.random.randn() * 0.5,
            "Injection_pressure": 125.0 + np.random.randn() * 5,
            "Cushion": 8.2 + np.random.randn() * 0.3,
        }
        result = predict_from_raw("M-471", data)
        print(f"{i+1}: risk={result['risk_score']:.3f} ({result['risk_level']})")
