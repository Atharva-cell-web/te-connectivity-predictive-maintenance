import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# ================= CONFIG =================
INPUT_FILE = Path("processed/features/rolling_training_labeled.parquet")
OUTPUT_FILE = Path("processed/features/rolling_training_sampled.parquet")

CONTEXT_MINUTES = 30
STABLE_SAMPLE_EVERY_SECONDS = 90
# =========================================

# -----------------------------
# FIXED PARQUET SCHEMA (CRITICAL)
# -----------------------------
PARQUET_SCHEMA = pa.schema([
    ("machine_id", pa.string()),
    ("variable_name", pa.string()),
    ("window", pa.string()),
    ("timestamp", pa.timestamp("ns", tz="UTC")),
    ("mean", pa.float64()),
    ("std", pa.float64()),
    ("min", pa.float64()),
    ("max", pa.float64()),
    ("last", pa.float64()),
    ("early_scrap_risk", pa.int64())
])

print("Opening parquet file...")
parquet_file = pq.ParquetFile(INPUT_FILE)

writer = pq.ParquetWriter(
    OUTPUT_FILE,
    PARQUET_SCHEMA,
    compression="snappy"
)

total_written = 0
print("Starting structured sampling...")

# ================= MAIN LOOP =================
for i in range(parquet_file.num_row_groups):
    print(f"Processing row group {i+1}/{parquet_file.num_row_groups}")

    df = parquet_file.read_row_group(i).to_pandas()

    # Enforce dtypes (IMPORTANT)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["machine_id"] = df["machine_id"].astype("string")
    df["variable_name"] = df["variable_name"].astype("string")
    df["window"] = df["window"].astype("string")
    df["early_scrap_risk"] = df["early_scrap_risk"].astype("int64")

    # ------------------------------
    # 1. Keep all risky rows
    # ------------------------------
    risky = df[df["early_scrap_risk"] == 1]

    # ------------------------------
    # 2. Context windows around risk
    # ------------------------------
    context_rows = []

    for _, r in risky.iterrows():
        start = r["timestamp"] - pd.Timedelta(minutes=CONTEXT_MINUTES)
        end   = r["timestamp"] + pd.Timedelta(minutes=CONTEXT_MINUTES)

        ctx = df[
            (df["machine_id"] == r["machine_id"]) &
            (df["variable_name"] == r["variable_name"]) &
            (df["timestamp"] >= start) &
            (df["timestamp"] <= end)
        ]

        context_rows.append(ctx)

    context = pd.concat(context_rows, ignore_index=True) if context_rows else df.iloc[0:0]

    # ------------------------------
    # 3. Downsample stable data
    # ------------------------------
    stable = df[df["early_scrap_risk"] == 0].copy()

    stable["ts_bucket"] = (
        stable["timestamp"].astype("int64") //
        (STABLE_SAMPLE_EVERY_SECONDS * 1_000_000_000)
    )

    stable_sampled = (
        stable
        .groupby(["machine_id", "variable_name", "window", "ts_bucket"], as_index=False)
        .first()
        .drop(columns=["ts_bucket"])
    )

    # ------------------------------
    # Combine all
    # ------------------------------
    final = (
        pd.concat([risky, context, stable_sampled], ignore_index=True)
        .drop_duplicates()
    )

    # Enforce column order
    final = final[
        ["machine_id", "variable_name", "window", "timestamp",
         "mean", "std", "min", "max", "last", "early_scrap_risk"]
    ]

    # ------------------------------
    # Write safely with fixed schema
    # ------------------------------
    table = pa.Table.from_pandas(
        final,
        schema=PARQUET_SCHEMA,
        preserve_index=False
    )

    writer.write_table(table)
    total_written += len(final)

    print(f"Written so far: {total_written:,}")

# ================= CLEANUP =================
writer.close()

print("Sampling completed.")
print("Final rows written:", total_written)
print("Saved to:", OUTPUT_FILE)
