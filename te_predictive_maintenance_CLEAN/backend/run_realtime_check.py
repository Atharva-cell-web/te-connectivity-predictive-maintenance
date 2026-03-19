from data_access import get_recent_window
from config_limits import SAFE_LIMITS, ML_THRESHOLDS
from feature_window import extract_ml_features
from ml_inference import load_model, predict_risk

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL

def run(machine_id: str):
    """
    The 'Judge': Decides if machine is Safe (Green) or Critical (Red).
    Now supports intelligent column matching.
    """
    try:
        # 1. Get the latest data (1 row)
        df = get_recent_window(machine_id, minutes=5)
        
        if df.empty:
            return {
                "machine_id": machine_id,
                "timestamp": None,
                "ml_risk_probability": 0.0,
                "alert_level": "LOW",
                "decision_reason": "NO DATA",
                "violations": []
            }

        # Get the absolute last row (The "Now" point)
        latest_row = df.iloc[-1]
        timestamp = str(latest_row['event_timestamp']) if 'event_timestamp' in latest_row else str(latest_row.name)

        # 2. Check for Safety Violations
        violations = []
        
        # Iterate through every rule in config_limits.py
        for param, limits in SAFE_LIMITS.items():
            
            # --- INTELLIGENT MATCHING START ---
            # Try to find the correct column in the dataframe
            col_name = None
            
            # Case A: Exact Match (e.g. "Injection_pressure")
            if param in latest_row:
                col_name = param
            
            # Case B: Suffix Match (e.g. "Injection_pressure__last_5m")
            else:
                # Look for columns that start with the param name
                candidates = [c for c in latest_row.index if c.startswith(param + "__")]
                if candidates:
                    col_name = candidates[0] # Pick the first match
            
            # If we still can't find it, skip this rule
            if not col_name:
                continue
            # --- INTELLIGENT MATCHING END ---

            # Get the value
            current_val = float(latest_row[col_name])
            
            # Check Max Limit
            if "max" in limits and current_val > limits["max"]:
                violations.append({
                    "parameter": param, # Send the clean name to Frontend
                    "current": round(current_val, 2),
                    "limit": limits["max"],
                    "unit": limits.get("unit", ""),
                    "deviation": round(current_val - limits["max"], 2),
                    "direction": "above"
                })

            # Check Min Limit
            if "min" in limits and current_val < limits["min"]:
                violations.append({
                    "parameter": param,
                    "current": round(current_val, 2),
                    "limit": limits["min"],
                    "unit": limits.get("unit", ""),
                    "deviation": round(limits["min"] - current_val, 2),
                    "direction": "below"
                })

        # 3. Run ML inference using the pre-trained model
       # Read the risk directly from our pre-processed February test data
        ml_risk = float(latest_row.get('scrap_probability', 0.0))
        
        if len(violations) > 0:
            status = "HIGH"
            reason = "SAFETY VIOLATION"
        elif ml_risk >= ML_THRESHOLDS["MEDIUM"]:
            status = "HIGH"
            reason = "AI RISK PREDICTION"
        elif ml_risk >= ML_THRESHOLDS["LOW"]:
            status = "MEDIUM"
            reason = "AI WARNING"
        else:
            status = "LOW"
            reason = "OPTIMAL"

        return {
            "machine_id": machine_id,
            "timestamp": timestamp,
            "ml_risk_probability": ml_risk,
            "alert_level": status,
            "decision_reason": reason,
            "violations": violations
        }

    except Exception as e:
        print(f"Checker Error: {e}")
        return {
            "machine_id": machine_id,
            "alert_level": "CRITICAL",
            "decision_reason": f"SYSTEM ERROR: {str(e)}",
            "violations": []
        }
