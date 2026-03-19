"""
check_distribution.py
Checks if sensor distributions shift across time — explains why
temporal split causes poor generalization.
Run from: D:\\te connectivity 3\\
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("new_processed_data")
EXCLUDE = {"Scrap_counter","Shot_counter","Time_on_machine","Machine_status","Alrams_array"}

print("="*60)
print("DISTRIBUTION SHIFT ANALYSIS")
print("="*60)

all_data = []
for f in sorted(PROCESSED_DIR.glob("M*_TRAIN.parquet")):
    machine_id = f.stem.replace("_TRAIN","")
    raw = pd.read_parquet(f)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp"])
    raw = raw[raw["timestamp"] <= pd.Timestamp("2026-01-11 23:59:59", tz="UTC")]
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["value"])
    if "variable_name" in raw.columns:
        raw = raw[~raw["variable_name"].isin(EXCLUDE)]
    raw["machine_id"] = machine_id
    all_data.append(raw[["timestamp","machine_id","variable_name","value"]])

df = pd.concat(all_data, ignore_index=True)
df = df.sort_values("timestamp").reset_index(drop=True)

# Split same way as training script
pivot_all = df.pivot_table(
    index=["timestamp","machine_id"],
    columns="variable_name",
    values="value",
    aggfunc="mean"
).reset_index()

split_idx = int(len(pivot_all) * 0.80)
split_time = pivot_all["timestamp"].iloc[split_idx]
train = pivot_all.iloc[:split_idx]
test  = pivot_all.iloc[split_idx:]

print(f"\nTrain period: {train['timestamp'].min().date()} → {train['timestamp'].max().date()}")
print(f"Test period:  {test['timestamp'].min().date()} → {test['timestamp'].max().date()}")
print(f"\nTrain machines: {sorted(train['machine_id'].value_counts().to_dict().items())}")
print(f"Test machines:  {sorted(test['machine_id'].value_counts().to_dict().items())}")

# Check key sensors for distribution shift
key_sensors = ["Cushion","Injection_pressure","Dosage_time","Cycle_time",
               "Cyl_tmp_z4","Cyl_tmp_z5"]

print(f"\n{'Sensor':<30} {'Train mean':>12} {'Test mean':>12} {'Shift %':>10}")
print("-"*68)
for sensor in key_sensors:
    if sensor in pivot_all.columns:
        t_mean = train[sensor].mean()
        v_mean = test[sensor].mean()
        if t_mean and t_mean != 0:
            shift = abs(v_mean - t_mean) / abs(t_mean) * 100
            print(f"{sensor:<30} {t_mean:>12.3f} {v_mean:>12.3f} {shift:>9.1f}%")

print("\n"+"="*60)
print("MACHINE OVERLAP IN TRAIN vs TEST")
print("="*60)
for m in ["M231","M356","M471","M607","M612"]:
    t_rows = len(train[train["machine_id"]==m])
    v_rows = len(test[test["machine_id"]==m])
    print(f"  {m}: train={t_rows:,} rows | test={v_rows:,} rows")

print("\n"+"="*60)
print("RECOMMENDATION")
print("="*60)
print("If shift % > 20% on key sensors → distribution shift is the problem.")
print("Solution: Use random stratified split instead of temporal split.")
print("Temporal split is only correct when test data is truly 'future'.")
print("Here, different machines/parts in train vs test period cause the shift.")