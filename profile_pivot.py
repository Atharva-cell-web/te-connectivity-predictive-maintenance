import time
import pandas as pd
import sys
sys.path.insert(0, r"d:\te-connectivity-3")
from backend.data_access import _resolve_machine_data_path, _load_control_model_and_features, normalize_machine_id

def test_pivot(machine_norm="M607", time_window_minutes=60, anchor_time="2026-03-14T07:00:00.000Z"):
    t0 = time.time()
    machine_path = _resolve_machine_data_path(machine_norm)
    print(f"Path: {machine_path}")
    
    t1 = time.time()
    raw = pd.read_parquet(machine_path, columns=["timestamp", "variable_name", "value", "machine_definition"], engine="pyarrow")
    print(f"Read parquet: {time.time()-t1:.3f}s")
    
    t2 = time.time()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp"])
    print(f"To datetime: {time.time()-t2:.3f}s")
    
    t3 = time.time()
    if anchor_time:
        reference_ts = pd.to_datetime(anchor_time, utc=True)
    else:
        reference_ts = raw["timestamp"].max()
    
    cutoff = reference_ts - pd.Timedelta(minutes=time_window_minutes + 15)
    raw = raw[(raw["timestamp"] >= cutoff) & (raw["timestamp"] <= reference_ts)].copy()
    print(f"Filter timeframe: {time.time()-t3:.3f}s")
    
    machine_definition = "UNKNOWN"
    if not raw.empty:
        defs = raw["machine_definition"].tail(500).dropna().astype(str).unique()
        if len(defs) > 0:
            machine_definition = defs[0]

    t4 = time.time()
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["value"])
    print(f"To numeric: {time.time()-t4:.3f}s")

    t5 = time.time()
    pivot = raw.pivot_table(index="timestamp", columns="variable_name", values="value", aggfunc="mean").reset_index()
    pivot = pivot.sort_values("timestamp").reset_index(drop=True)
    pivot = pivot.loc[:, ~pivot.columns.duplicated(keep="last")]
    print(f"Pivot: {time.time()-t5:.3f}s")
    
    print(f"TOTAL PIVOT TIME: {time.time()-t0:.3f}s")
    
test_pivot()
