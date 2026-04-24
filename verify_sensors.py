import pandas as pd

# Check what sensor data exists in M356 CSV grouped by DATE
raw = pd.read_csv(r"d:\te-connectivity-3\pipeline  data\new data= 16 april 2026\M356 csv file.csv")
raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
raw["date_only"] = raw["timestamp"].dt.date

print("=== M356 CSV - Variables available per DATE ===\n")

for date in sorted(raw["date_only"].dropna().unique()):
    subset = raw[raw["date_only"] == date]
    vars_list = sorted(subset["variable_name"].unique().tolist())
    print(f"{date}: {len(subset)} rows | Variables: {vars_list}")

print("\n\n=== HYDRA - Variables available per DATE ===\n")
hydra = pd.read_parquet(r"d:\te-connectivity-3\new_processed_data\HYDRA_TRAIN.parquet")
hydra["timestamp"] = pd.to_datetime(hydra["timestamp"], utc=True, errors="coerce")
hydra["date_only"] = hydra["timestamp"].dt.date

for date in sorted(hydra["date_only"].dropna().unique()):
    subset = hydra[hydra["date_only"] == date]
    vars_list = sorted(subset["variable_name"].unique().tolist())
    print(f"{date}: {len(subset)} rows | Variables: {', '.join(vars_list[:8])}{'...' if len(vars_list) > 8 else ''}")
