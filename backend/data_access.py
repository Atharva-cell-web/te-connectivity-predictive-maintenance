import pandas as pd
from pathlib import Path
from datetime import timedelta

# Robust project root detection
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WIDE_FILE = PROJECT_ROOT / "processed" / "features" / "rolling_features_demo.parquet"

def get_recent_window(machine_id, minutes=60):
    if not WIDE_FILE.exists():
        raise FileNotFoundError(f"Wide feature file not found: {WIDE_FILE}")

    # Load data (in production, you'd load only recent rows, but this works for files)
    df = pd.read_parquet(WIDE_FILE)

    # Filter for machine
    df = df[df["machine_id_normalized"] == machine_id].copy()
    
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("event_timestamp")

    # Get last N minutes window
    cutoff = df["event_timestamp"].max() - timedelta(minutes=minutes)
    recent_df = df[df["event_timestamp"] >= cutoff]
    
    return recent_df