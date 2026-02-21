import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# ================= CONFIG =================
SENSOR_DIR = Path("processed/sensor")
OUTPUT_FILE = Path("processed/features/rolling_features_wide.parquet")
MASTER_SCHEMA_FILE = Path("processed/features/rolling_feature_columns.txt")

ROLL_WINDOWS = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min"
}
# =========================================

print("Starting wide rolling feature rebuild (memory-safe, schema-safe)...")

# ------------------------------
# Load master schema ONCE
# ------------------------------
master_cols = pd.read_csv(
    MASTER_SCHEMA_FILE,
    header=None
)[0].tolist()

writer = None

for sensor_file in SENSOR_DIR.glob("*_cleaned.csv"):
    print(f"\nProcessing {sensor_file.name}")

    df = pd.read_csv(
        sensor_file,
        parse_dates=["event_timestamp"]
    )

    # safety
    df = df.dropna(subset=["event_timestamp", "variable_name"])

    for machine_id, mdf in df.groupby("machine_id_normalized"):
        print(f"  Machine {machine_id}")

        # ------------------------------
        # Base timeline (unique timestamps)
        # ------------------------------
        base = (
            mdf[["machine_id_normalized", "event_timestamp"]]
            .drop_duplicates()
            .sort_values("event_timestamp")
            .reset_index(drop=True)
        )

        # ------------------------------
        # Build features variable-by-variable
        # ------------------------------
        for var, vdf in mdf.groupby("variable_name"):

            # collapse duplicate timestamps FIRST
            vdf = (
                vdf.groupby("event_timestamp", as_index=False)
                .agg({"value_numeric": "mean"})
                .sort_values("event_timestamp")
                .set_index("event_timestamp")
            )

            for win_label, win in ROLL_WINDOWS.items():
                roll = vdf["value_numeric"].rolling(
                    window=win,
                    min_periods=20
                )

                feats = pd.DataFrame({
                    "event_timestamp": vdf.index,
                    f"{var}__mean_{win_label}": roll.mean().to_numpy(),
                    f"{var}__std_{win_label}": roll.std().to_numpy(),
                    f"{var}__min_{win_label}": roll.min().to_numpy(),
                    f"{var}__max_{win_label}": roll.max().to_numpy(),
                    f"{var}__last_{win_label}": vdf["value_numeric"].to_numpy(),
                })

                # merge safely on timestamp
                base = base.merge(
                    feats,
                    on="event_timestamp",
                    how="left"
                )

        # ------------------------------
        # ENFORCE MASTER SCHEMA
        # ------------------------------
        base = base.reindex(columns=master_cols)

        # ------------------------------
        # Write incrementally (Parquet-safe)
        # ------------------------------
        table = pa.Table.from_pandas(
            base,
            preserve_index=False
        )

        if writer is None:
            writer = pq.ParquetWriter(
                OUTPUT_FILE,
                table.schema,
                compression="snappy"
            )

        writer.write_table(table)

# close writer
if writer:
    writer.close()

print("\nWide rolling feature rebuild completed successfully.")
print("Saved to:", OUTPUT_FILE)
