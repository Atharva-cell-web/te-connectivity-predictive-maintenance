import pandas as pd
from pathlib import Path

SENSOR_DIR = Path("processed/sensor")
OUTPUT_SCHEMA = Path("processed/features/rolling_feature_columns.txt")

ROLL_WINDOWS = ["5m", "15m", "30m"]

all_cols = set([
    "machine_id_normalized",
    "event_timestamp"
])

print("Scanning sensor files to build master schema...")

for f in SENSOR_DIR.glob("*_cleaned.csv"):
    print("  Reading:", f.name)
    df = pd.read_csv(f, usecols=["variable_name"], nrows=5000)

    for var in df["variable_name"].dropna().unique():
        for w in ROLL_WINDOWS:
            all_cols.update([
                f"{var}__mean_{w}",
                f"{var}__std_{w}",
                f"{var}__min_{w}",
                f"{var}__max_{w}",
                f"{var}__last_{w}",
            ])

# sort for consistency
master_cols = sorted(all_cols)

OUTPUT_SCHEMA.parent.mkdir(parents=True, exist_ok=True)
pd.Series(master_cols).to_csv(OUTPUT_SCHEMA, index=False, header=False)

print("\nMaster schema created")
print("Total columns:", len(master_cols))
print("Saved to:", OUTPUT_SCHEMA)
