import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- CONFIG ----------------
SENSOR_DIR = Path("processed/sensor")
SAFE_FILE = Path("processed/safe/AI_cup_parameter_info_cleaned.csv")
OUT_FILE = Path("processed/features/rolling_live_features.csv")

WINDOWS = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min"
}
RESAMPLE_FREQ = "10S"  # 10 seconds
MIN_POINTS = 5
# ---------------------------------------

print("Loading safe limits...")
safe = pd.read_csv(SAFE_FILE)

print("Loading sensor files...")
sensor_files = list(SENSOR_DIR.glob("*_cleaned.csv"))
assert sensor_files, "No sensor files found"

out_chunks = []

for file in sensor_files:
    print(f"Processing {file.name}")
    df = pd.read_csv(file, parse_dates=["event_timestamp"])

    df = df.dropna(subset=["value_numeric"])
    df = df.sort_values("event_timestamp")

    for (machine, var), g in df.groupby(["machine_id_normalized", "variable_name"]):
        g = g.set_index("event_timestamp")

        for w_name, w in WINDOWS.items():
            rolled = g["value_numeric"].rolling(w)

            feat = pd.DataFrame({
                "machine_id": machine,
                "variable_name": var,
                "window": w_name,
                "timestamp": rolled.mean().index,
                "mean": rolled.mean(),
                "std": rolled.std(),
                "min": rolled.min(),
                "max": rolled.max(),
                "last": g["value_numeric"],
            })

            feat = feat.dropna()
            out_chunks.append(feat)

print("Concatenating output...")
final = pd.concat(out_chunks, ignore_index=True)
final.to_csv(OUT_FILE, index=False)

print("Phase 3C completed successfully")
print(final.shape)
