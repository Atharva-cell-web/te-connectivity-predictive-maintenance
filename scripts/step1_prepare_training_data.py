import pandas as pd

# 1. Load event-level features
print("Loading event-level features...")
df = pd.read_csv("processed/features/event_level_features.csv")

# 2. Create target variable (scrap flag)
# 1 = scrap happened, 0 = no scrap
df["scrap_flag"] = (df["scrap_quantity"] > 0).astype(int)

# 3. Drop columns that must NOT be used as features
drop_cols = [
    "scrap_quantity",
    "yield_quantity",
    "machine_event_record_id"
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 4. Save training dataset
output_path = "processed/features/event_level_training.csv"
df.to_csv(output_path, index=False)

# 5. Print summary
print("Training data created")
print("Shape:", df.shape)
print("Scrap flag distribution:")
print(df["scrap_flag"].value_counts())
