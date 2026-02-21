import pandas as pd
from pathlib import Path

WIDE_FILE = Path("processed/features/rolling_features_wide.parquet")
LABEL_FILE = Path("processed/features/rolling_training_sampled.parquet")
OUT_FILE = Path("processed/features/rolling_features_wide_labeled.parquet")

print("Loading wide features...")
wide_df = pd.read_parquet(WIDE_FILE)

print("Loading long-format labels...")
labels = pd.read_parquet(
    LABEL_FILE,
    columns=["machine_id", "timestamp", "early_scrap_risk"]
)

labels = labels.rename(columns={
    "machine_id": "machine_id_normalized",
    "timestamp": "event_timestamp"
})

print("Aggregating labels to timestamp-level...")
labels_agg = (
    labels
    .groupby(["machine_id_normalized", "event_timestamp"], as_index=False)
    .agg({"early_scrap_risk": "max"})   # 🔑 CRITICAL LINE
)

print("Label rows after aggregation:", len(labels_agg))

print("Merging labels into wide table...")
merged = wide_df.merge(
    labels_agg,
    on=["machine_id_normalized", "event_timestamp"],
    how="left"
)

merged["early_scrap_risk"] = (
    merged["early_scrap_risk"]
    .fillna(0)
    .astype("int8")
)

print("Final rows:", len(merged))
print("Positive labels:", merged["early_scrap_risk"].sum())

merged.to_parquet(OUT_FILE, index=False)
print("Saved:", OUT_FILE)
