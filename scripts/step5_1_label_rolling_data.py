import pandas as pd
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
ROLLING_FILE = Path("processed/features/rolling_live_features.csv")
EVENT_FILE = Path("processed/features/event_level_features.csv")
OUT_FILE = Path("processed/features/rolling_training_labeled.csv")

FUTURE_WINDOW_MINUTES = 30
CHUNK_SIZE = 500_000   # safe size

print("Loading event-level outcomes...")
events = pd.read_csv(EVENT_FILE, parse_dates=["event_end"])

# Keep only scrap events
events = events[events["scrap_quantity"] > 0].copy()

# Pre-group scrap times by machine
scrap_times = {
    m: g["event_end"].to_numpy()
    for m, g in events.groupby("machine_id")
}


print("Prepared scrap timelines for machines:", list(scrap_times.keys()))

# -------------------------
# Process rolling data in chunks
# -------------------------
print("Streaming rolling data and labeling...")

first_write = True
total_rows = 0
risk_counts = {0: 0, 1: 0}

for chunk in pd.read_csv(
    ROLLING_FILE,
    chunksize=CHUNK_SIZE,
    parse_dates=["timestamp"]
):
    chunk = chunk.sort_values(["machine_id", "timestamp"])

    labels = []

    for row in chunk.itertuples(index=False):
        ts = row.timestamp
        machine_id = row.machine_id

        ev_times = scrap_times.get(machine_id)
        if ev_times is None or len(ev_times) == 0:
            labels.append(0)
            continue

        

        future_limit = ts + pd.Timedelta(minutes=FUTURE_WINDOW_MINUTES)

        risk = int(((ev_times > ts) & (ev_times <= future_limit)).any())
        labels.append(risk)

    chunk["early_scrap_risk"] = labels

    # Track stats
    total_rows += len(chunk)
    risk_counts[0] += labels.count(0)
    risk_counts[1] += labels.count(1)

    # Write incrementally
    chunk.to_csv(
        OUT_FILE,
        mode="w" if first_write else "a",
        header=first_write,
        index=False
    )

    first_write = False
    print(f"Processed {total_rows:,} rows so far...")

print("\nPhase 5.1 completed")
print("Total rows:", total_rows)
print("Risk distribution:", risk_counts)
