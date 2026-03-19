"""
Live predictor module for real-time scrap-risk scoring.

Uses the 21-feature LightGBM model (lightgbm_scrap_model.pkl) and buffers
incoming raw sensor readings per machine_id.

Includes an "offline / frozen" detector: if the machine's sensor readings
are identical to the previous reading OR Cycle_time <= 0.1, the model is
bypassed and risk_score is returned as 0.0.
"""

import joblib
import pandas as pd
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

# ── Model & feature list ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_scrap_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "model_features.pkl"

print(f"[live_predictor] Loading model  : {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
print(f"[live_predictor] Model expects {len(features)} features: {features[:5]}...")

# ── Per-machine sensor buffer ───────────────────────────────────────
live_buffer: Dict[str, Dict[str, float]] = {}

# ── Per-machine state tracker (last 3 snapshots for frozen detection) ──
_state_history: Dict[str, deque] = {}
_STATE_HISTORY_SIZE = 3


def _is_frozen_or_offline(machine_id: str, current_state: Dict[str, float]) -> bool:
    """
    Return True if the machine appears offline or frozen:
      1. Cycle_time <= 0.1  (machine not cycling)
      2. Current sensor snapshot is identical to the previous one (frozen PLC)
    """
    # Check 1: low / zero Cycle_time
    cycle_time = current_state.get("Cycle_time", None)
    if cycle_time is not None and cycle_time <= 0.1:
        return True

    # Check 2: compare with previous snapshot
    if machine_id not in _state_history:
        _state_history[machine_id] = deque(maxlen=_STATE_HISTORY_SIZE)

    history = _state_history[machine_id]
    if len(history) > 0:
        prev = history[-1]
        # If every sensor that exists in both snapshots has the same value → frozen
        shared_keys = set(current_state.keys()) & set(prev.keys())
        if shared_keys and all(current_state[k] == prev[k] for k in shared_keys):
            return True

    # Record current snapshot into history
    history.append(dict(current_state))
    return False


def predict_from_raw(machine_id: str, raw_data: Dict[str, Any]) -> dict:
    """
    Accept one incoming sensor reading, buffer it, and return a live risk score.

    Parameters
    ----------
    machine_id : str
        Identifier for the machine (e.g. "M231").
    raw_data : dict
        Must contain at least ``variable_name`` and ``value``.
        May also contain ``timestamp`` (ignored for prediction).

    Returns
    -------
    dict with keys: status, machine_id, risk_score, sensors_buffered
    """
    var_name = raw_data.get("variable_name")
    val = raw_data.get("value")

    # Initialise buffer for this machine if first time
    if machine_id not in live_buffer:
        live_buffer[machine_id] = {}

    # Update the machine's buffer with the newest sensor reading
    if var_name is not None and val is not None:
        live_buffer[machine_id][var_name] = float(val)

    current_state = live_buffer[machine_id]

    # ── Frozen / offline check ──────────────────────────────────
    if _is_frozen_or_offline(machine_id, current_state):
        return {
            "machine_id": machine_id,
            "risk_score": 0.0,
            "sensors_buffered": len(current_state),
            "status": "offline",
        }

    # ── Build feature row and score ─────────────────────────────
    df = pd.DataFrame([current_state])

    # Fill any sensors the machine hasn't sent yet with 0
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    X = df[features]

    if hasattr(model, "predict_proba"):
        risk_score = float(model.predict_proba(X)[:, 1][0])
    else:
        risk_score = float(model.predict(X)[0])

    return {
        "machine_id": machine_id,
        "risk_score": round(risk_score, 4),
        "sensors_buffered": len(current_state),
        "status": "success",
    }


# ── Helper endpoints ────────────────────────────────────────────────

def get_buffer_status(machine_id: Optional[str] = None) -> dict:
    """Return buffer sizes for one or all machines."""
    if machine_id:
        return {
            "machine_id": machine_id,
            "buffer_size": len(live_buffer.get(machine_id, {})),
        }
    return {
        "machines": list(live_buffer.keys()),
        "sizes": {k: len(v) for k, v in live_buffer.items()},
    }


def clear_buffer(machine_id: Optional[str] = None) -> None:
    """Clear the buffer for one or all machines."""
    if machine_id and machine_id in live_buffer:
        live_buffer[machine_id].clear()
    elif not machine_id:
        live_buffer.clear()
