import pandas as pd

print("Loading event-level features...")
df = pd.read_csv("processed/features/event_level_features.csv")

df["scrap_flag"] = (df["scrap_quantity"] > 0).astype(int)

drop_cols = [
    "scrap_quantity",
    "yield_quantity",
    "machine_event_record_id"
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

output_path = "processed/features/event_level_training.csv"
df.to_csv(output_path, index=False)

print("Training data created")
print("Shape:", df.shape)
print("Scrap flag distribution:")
print(df["scrap_flag"].value_counts())
