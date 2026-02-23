import re
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config_limits import ML_THRESHOLDS, SAFE_LIMITS

# Robust project root detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WIDE_FILE = PROJECT_ROOT / "processed" / "features" / "rolling_features_demo.parquet"
WIDE_FILE_FALLBACK = PROJECT_ROOT / "processed" / "features" / "rolling_features_wide.parquet"
FEB_RESULTS_FILE = PROJECT_ROOT / "new_processed_data" / "FEB_TEST_RESULTS.parquet"
MACHINE_TESTS_DIR = PROJECT_ROOT / "new_processed_data"
CONTROL_MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_scrap_risk_wide.pkl"
MODEL_FEATURES_PATH = PROJECT_ROOT / "processed" / "features" / "rolling_feature_columns.txt"
FORECASTER_MODEL_PATH = PROJECT_ROOT / "models" / "sensor_forecaster_lagged.pkl"
FUTURE_RISK_THRESHOLD = float(ML_THRESHOLDS.get("MEDIUM", 0.60))


def _normalize_machine_id(machine_id: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9]", "", str(machine_id or "")).upper()
    if compact.startswith("M"):
        return compact
    return f"M{compact}"


def _display_machine_id(machine_norm: str) -> str:
    match = re.match(r"^M(\d+)$", machine_norm)
    if match:
        return f"M-{match.group(1)}"
    return machine_norm


def _safe_float(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _downsample(df: pd.DataFrame, max_points: int = 360) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    sampled = df.iloc[::step].copy()
    if sampled.iloc[-1]["timestamp"] != df.iloc[-1]["timestamp"]:
        sampled = pd.concat([sampled, df.tail(1)], ignore_index=True)
    return sampled.drop_duplicates(subset=["timestamp"], keep="last")


def _clean_limit_payload():
    cleaned = {}
    for sensor, limits in SAFE_LIMITS.items():
        cleaned[sensor] = {
            "min": _safe_float(limits.get("min")) if "min" in limits else None,
            "max": _safe_float(limits.get("max")) if "max" in limits else None,
        }
    return cleaned



@lru_cache(maxsize=1)
def _load_control_model_and_features():
    if not CONTROL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {CONTROL_MODEL_PATH}")

    # Load the LightGBM model
    model = joblib.load(CONTROL_MODEL_PATH)
    
    # BULLETPROOF FIX: Ask the model exactly what 375 features it was trained on.
    # This completely ignores the .txt file and guarantees a perfect shape match.
    if hasattr(model, "feature_name_"):
        features = model.feature_name_
    elif hasattr(model, "booster_"):
        features = model.booster_.feature_name()
    else:
        # Emergency fallback (stripping the known meta-columns)
        with open(MODEL_FEATURES_PATH, "r") as f:
            features = [
                line.strip() for line in f.readlines() 
                if line.strip() and line.strip() not in ["machine_id_normalized", "event_timestamp", "timestamp", "machine_id", "is_scrap"]
            ]
            
    return model, tuple(features)


@lru_cache(maxsize=1)
def _load_sensor_forecaster():
    """Load the lagged recursive sensor forecaster model + metadata."""
    if not FORECASTER_MODEL_PATH.exists():
        raise FileNotFoundError(f"Sensor forecaster not found: {FORECASTER_MODEL_PATH}")
    artifact = joblib.load(FORECASTER_MODEL_PATH)
    return (
        artifact["model"],
        list(artifact["sensor_columns"]),
        list(artifact["input_features"]),
        int(artifact["num_lags"]),
    )


@lru_cache(maxsize=1)
def _load_feb_results():
    if not FEB_RESULTS_FILE.exists():
        raise FileNotFoundError(f"FEB results file not found: {FEB_RESULTS_FILE}")

    feb = pd.read_parquet(FEB_RESULTS_FILE)
    if "timestamp" not in feb.columns:
        raise ValueError("FEB_TEST_RESULTS.parquet must include a 'timestamp' column.")

    feb["timestamp"] = pd.to_datetime(feb["timestamp"], utc=True, errors="coerce")
    feb = feb.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for key_col in ("Injection_pressure", "Cycle_time"):
        if key_col in feb.columns:
            feb[key_col] = pd.to_numeric(feb[key_col], errors="coerce").round(4)

    return feb


@lru_cache(maxsize=16)
def _load_machine_pivot(machine_norm: str):
    machine_path = MACHINE_TESTS_DIR / f"{machine_norm}_TEST.parquet"
    if not machine_path.exists():
        raise FileNotFoundError(f"Machine test parquet not found: {machine_path}")

    raw = pd.read_parquet(machine_path, columns=["timestamp", "variable_name", "value", "machine_definition"])
    machine_definition = "UNKNOWN"
    defs = raw["machine_definition"].dropna().astype(str).unique()
    if len(defs) > 0:
        machine_definition = defs[0]

    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["value"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp"])

    pivot = raw.pivot_table(
        index="timestamp",
        columns="variable_name",
        values="value",
        aggfunc="mean",
    ).reset_index()
    pivot = pivot.sort_values("timestamp").reset_index(drop=True)

    _, model_features = _load_control_model_and_features()
    for feature in model_features:
        if feature not in pivot.columns:
            pivot[feature] = 0.0

    for key_col in ("Injection_pressure", "Cycle_time"):
        if key_col in pivot.columns:
            pivot[key_col] = pd.to_numeric(pivot[key_col], errors="coerce").round(4)

    return pivot, machine_definition


@lru_cache(maxsize=16)
def _build_machine_feb_history(machine_norm: str):
    feb = _load_feb_results()
    pivot, machine_definition = _load_machine_pivot(machine_norm)

    join_cols = ["timestamp", "Injection_pressure", "Cycle_time"]
    missing_join = [c for c in join_cols if c not in pivot.columns or c not in feb.columns]
    if missing_join:
        raise ValueError(f"Cannot map machine rows to FEB results. Missing join columns: {missing_join}")

    feb_unique = feb.drop_duplicates(subset=join_cols, keep="first")
    machine_key = pivot[join_cols].copy()
    history = machine_key.merge(feb_unique, on=join_cols, how="left")

    if history.empty:
        raise ValueError(f"No FEB history matched machine {machine_norm}.")

    if "scrap_probability" not in history.columns:
        history["scrap_probability"] = 0.0
    history["scrap_probability"] = pd.to_numeric(history["scrap_probability"], errors="coerce")

    if "is_scrap_actual" not in history.columns:
        history["is_scrap_actual"] = 0
    history["is_scrap_actual"] = pd.to_numeric(history["is_scrap_actual"], errors="coerce").fillna(0)

    # Backfill missing risk from model if needed.
    missing_prob_mask = history["scrap_probability"].isna()
    if missing_prob_mask.any():
        model, model_features = _load_control_model_and_features()
        feature_frame = history.loc[missing_prob_mask].copy()
        for feature in model_features:
            if feature not in feature_frame.columns:
                feature_frame[feature] = 0.0
        X_missing = feature_frame[list(model_features)].fillna(0.0)
        if hasattr(model, "predict_proba"):
            missing_probs = model.predict_proba(X_missing)[:, 1]
        else:
            missing_probs = model.predict(X_missing)
        history.loc[missing_prob_mask, "scrap_probability"] = missing_probs

    history["scrap_probability"] = history["scrap_probability"].fillna(0.0).clip(0, 1)
    history["machine_id_normalized"] = machine_norm
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
    history = history.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    tool_match = re.search(r"-([A-Za-z0-9]+)$", str(machine_definition))
    tool_id = tool_match.group(1) if tool_match else "UNKNOWN"

    machine_info = {
        "id": _display_machine_id(machine_norm),
        "tool_id": tool_id,
        "part_number": "UNKNOWN",
    }
    return history, machine_info


def _compute_root_causes(current_sensors: dict):
    exceeded = []
    nearby = []
    for sensor, limits in SAFE_LIMITS.items():
        sensor_value = _safe_float(current_sensors.get(sensor))
        if sensor_value is None:
            continue

        lower = _safe_float(limits.get("min")) if "min" in limits else None
        upper = _safe_float(limits.get("max")) if "max" in limits else None
        span_candidates = []
        if lower is not None and upper is not None:
            span_candidates.append(abs(upper - lower))
        if upper is not None:
            span_candidates.append(abs(upper))
        if lower is not None:
            span_candidates.append(abs(lower))
        span = max(max(span_candidates) if span_candidates else 1.0, 1.0)

        if upper is not None and sensor_value > upper:
            exceeded.append((sensor, (sensor_value - upper) / span))
            continue
        if lower is not None and sensor_value < lower:
            exceeded.append((sensor, (lower - sensor_value) / span))
            continue

        distances = []
        if lower is not None:
            distances.append(abs(sensor_value - lower))
        if upper is not None:
            distances.append(abs(upper - sensor_value))
        if distances:
            normalized_margin = min(distances) / span
            nearby.append((sensor, 1.0 - min(normalized_margin, 1.0)))

    if exceeded:
        exceeded_sorted = sorted(exceeded, key=lambda item: item[1], reverse=True)
        return [sensor for sensor, _ in exceeded_sorted[:3]], [sensor for sensor, _ in exceeded_sorted]

    nearby_sorted = sorted(nearby, key=lambda item: item[1], reverse=True)
    return [sensor for sensor, _ in nearby_sorted[:3]], []


def _infer_step_seconds(history: pd.DataFrame) -> int:
    if len(history) < 2:
        return 60
    diffs = history["timestamp"].diff().dropna().dt.total_seconds()
    if diffs.empty:
        return 60
    median_step = float(diffs.median())
    if not np.isfinite(median_step) or median_step <= 0:
        return 60
    return int(np.clip(round(median_step), 10, 120))


def _generate_future_horizon(past_window: pd.DataFrame, future_minutes: int):
    """Generate a future forecast using recursive AI sensor prediction with lag features."""
    scrap_model, model_features = _load_control_model_and_features()
    model_features = list(model_features)
    if past_window.empty:
        return pd.DataFrame(columns=["timestamp", "scrap_probability", "is_scrap_actual"])

    recent = past_window.sort_values("timestamp").tail(min(240, len(past_window))).copy()

    # Ensure all needed columns exist and are numeric
    all_columns = set(model_features + list(SAFE_LIMITS.keys()))
    for feature in all_columns:
        if feature not in recent.columns:
            recent[feature] = 0.0
        recent[feature] = pd.to_numeric(recent[feature], errors="coerce")
        recent[feature] = recent[feature].ffill().bfill().fillna(0.0)

    step_seconds = _infer_step_seconds(recent)
    total_seconds = max(int(future_minutes) * 60, step_seconds * 6)
    steps = max(6, int(np.ceil(total_seconds / step_seconds)))
    last_ts = recent["timestamp"].iloc[-1]

    # ------------------------------------------------------------------
    # Recursive AI Forecasting with Auto-Regressive Lag Features
    # ------------------------------------------------------------------
    sensor_forecaster, sensor_columns, input_features, num_lags = _load_sensor_forecaster()
    n_sensors = len(sensor_columns)

    # Build initial state buffer from the last (num_lags + 1) rows of history.
    # Layout: lag_buffer[0] = current values, lag_buffer[1] = t-1, ..., lag_buffer[num_lags] = t-num_lags
    seed_rows = recent.tail(num_lags + 1)
    lag_buffer = np.zeros((num_lags + 1, n_sensors), dtype=np.float64)

    for row_idx, (_, row) in enumerate(seed_rows.iloc[::-1].iterrows()):
        if row_idx > num_lags:
            break
        for col_idx, sensor in enumerate(sensor_columns):
            lag_buffer[row_idx, col_idx] = float(row.get(sensor, 0.0) or 0.0)

    # Pre-compute clamp bounds as float arrays (NaN = no limit)
    clamp_lo = np.full(n_sensors, -np.inf, dtype=np.float64)
    clamp_hi = np.full(n_sensors, np.inf, dtype=np.float64)
    for j, s in enumerate(sensor_columns):
        lim = SAFE_LIMITS.get(s, {})
        lo = _safe_float(lim.get("min"))
        hi = _safe_float(lim.get("max"))
        if lo is not None:
            clamp_lo[j] = lo
        if hi is not None:
            clamp_hi[j] = hi

    future_rows = []
    for _ in range(steps):
        # Assemble the input vector: [current_sensors, lag_1, lag_2, ..., lag_N]
        input_vec = lag_buffer.ravel()  # shape (n_sensors * (num_lags + 1),)

        # Predict the next time-step
        predicted = sensor_forecaster.predict(input_vec.reshape(1, -1))[0]

        # Clamp to safe physical limits
        predicted = np.clip(predicted, clamp_lo, clamp_hi)

        future_rows.append(predicted.copy())

        # Shift the lag buffer: move everything down by one slot, insert prediction at [0]
        lag_buffer[1:] = lag_buffer[:-1]
        lag_buffer[0] = predicted

    # Build future DataFrame with AI-predicted raw sensors
    future = pd.DataFrame(future_rows, columns=sensor_columns)
    future.insert(
        0,
        "timestamp",
        [last_ts + pd.Timedelta(seconds=step_seconds * i) for i in range(1, steps + 1)],
    )

    # ------------------------------------------------------------------
    # Forward-fill complex rolling features (300+ columns) from last state
    # Computing rolling windows recursively would be too expensive for API.
    # ------------------------------------------------------------------
    last_known = recent.iloc[-1]
    for feature in model_features:
        if feature not in future.columns:
            val = float(last_known.get(feature, 0.0) or 0.0)
            future[feature] = val

    # ------------------------------------------------------------------
    # Score: run the scrap risk model on the AI-forecasted future data
    # ------------------------------------------------------------------
    safe_features = [
        f for f in model_features
        if f in future.columns
        and f not in ["timestamp", "machine_id", "machine_id_normalized", "event_timestamp", "is_scrap"]
    ]

    X_future = future[safe_features].fillna(0.0)

    if hasattr(scrap_model, "predict_proba"):
        risk_values = scrap_model.predict_proba(X_future)[:, 1]
    else:
        risk_values = scrap_model.predict(X_future)

    future["scrap_probability"] = pd.Series(risk_values).clip(0, 1)
    future["is_scrap_actual"] = 0
    future["predicted_scrap"] = (future["scrap_probability"] >= FUTURE_RISK_THRESHOLD).astype(int)
    return future


def _row_to_timeline_point(row, is_future: bool):
    sensors = {}
    for sensor in SAFE_LIMITS:
        if sensor in row and pd.notna(row[sensor]):
            sensors[sensor] = round(float(row[sensor]), 4)

    timestamp = pd.to_datetime(row["timestamp"], utc=True)
    return {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "risk_score": round(float(row.get("scrap_probability", 0.0)), 4),
        "is_future": bool(is_future),
        "is_scrap_actual": int(float(row.get("is_scrap_actual", 0) or 0)),
        "sensors": sensors,
    }


def build_control_room_payload(machine_id: str, time_window: int = 240):
    machine_norm = _normalize_machine_id(machine_id)
    history, machine_info = _build_machine_feb_history(machine_norm)
    if history.empty:
        raise ValueError(f"No history found for machine {machine_id}")

    history = history.sort_values("timestamp").reset_index(drop=True)
    last_ts = history["timestamp"].max()
    cutoff = last_ts - timedelta(minutes=int(time_window))
    past_window = history[history["timestamp"] >= cutoff].copy()
    if past_window.empty:
        past_window = history.tail(1).copy()

    current_row = past_window.iloc[-1]
    current_sensors = {}
    for sensor in SAFE_LIMITS:
        if sensor in current_row and pd.notna(current_row[sensor]):
            current_sensors[sensor] = float(current_row[sensor])

    root_causes, breached_sensors = _compute_root_causes(current_sensors)
    base_risk = float(current_row.get("scrap_probability", 0.0) or 0.0)
    risk_penalty = min(0.35, 0.12 * len(breached_sensors))
    current_risk = min(1.0, base_risk + risk_penalty)

    if breached_sensors or current_risk >= float(ML_THRESHOLDS.get("MEDIUM", 0.60)):
        status = "HIGH"
    elif current_risk >= float(ML_THRESHOLDS.get("LOW", 0.30)):
        status = "MEDIUM"
    else:
        status = "LOW"

    future_minutes = max(15, min(120, int(round(int(time_window) / 4))))
    future_horizon = _generate_future_horizon(past_window, future_minutes=future_minutes)

    past_scrap_detected = int((past_window["is_scrap_actual"].fillna(0) >= 1).sum())
    future_scrap_predicted = int((future_horizon["scrap_probability"] >= FUTURE_RISK_THRESHOLD).sum())

    past_timeline = _downsample(past_window, max_points=320)
    future_timeline = _downsample(future_horizon, max_points=120)

    timeline = []
    for _, row in past_timeline.iterrows():
        timeline.append(_row_to_timeline_point(row, is_future=False))
    for _, row in future_timeline.iterrows():
        timeline.append(_row_to_timeline_point(row, is_future=True))

    payload = {
        "machine_info": machine_info,
        "summary_stats": {
            "past_scrap_detected": past_scrap_detected,
            "future_scrap_predicted": future_scrap_predicted,
        },
        "current_health": {
            "status": status,
            "risk_score": round(current_risk, 4),
            "root_causes": root_causes or ["Injection_pressure"],
        },
        "timeline": timeline,
        "safe_limits": _clean_limit_payload(),
    }
    return payload


def get_recent_window(machine_id, minutes=60):
    target_file = WIDE_FILE if WIDE_FILE.exists() else WIDE_FILE_FALLBACK
    if not target_file.exists():
        raise FileNotFoundError(f"Wide feature file not found: {target_file}")

    df = pd.read_parquet(target_file)

    machine_col = "machine_id_normalized" if "machine_id_normalized" in df.columns else "machine_id"
    machine_norm = _normalize_machine_id(machine_id)
    machine_series = df[machine_col].astype(str).str.replace("-", "", regex=False).str.upper()
    df = df[machine_series == machine_norm].copy()

    if df.empty:
        return pd.DataFrame()

    time_col = "event_timestamp" if "event_timestamp" in df.columns else "timestamp"
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)

    cutoff = df[time_col].max() - timedelta(minutes=minutes)
    return df[df[time_col] >= cutoff].copy()
