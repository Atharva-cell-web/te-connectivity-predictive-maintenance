"""
v1.0-diagnosis — Run this before any model changes.
Gives real answers about data quality and model problems.
Run from: D:\\te connectivity 3\\
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(".")
print("=" * 60)
print("DIAGNOSIS REPORT — TE Connectivity ML Model")
print("=" * 60)

# ── 1. FEATURE IMPORTANCE — check for Scrap_counter leakage ──
print("\n[1] TOP 20 FEATURES BY IMPORTANCE")
print("-" * 40)
fi = pd.read_csv(ROOT / "models" / "feature_importance.csv")
fi = fi.sort_values("importance", ascending=False)
print(fi.head(20).to_string(index=False))

scrap_counter_features = fi[fi["feature"].str.contains("Scrap_counter", case=False)]
shot_counter_features = fi[fi["feature"].str.contains("Shot_counter", case=False)]
print(f"\nScrap_counter features in top ranks: {len(scrap_counter_features)}")
if not scrap_counter_features.empty:
    print(scrap_counter_features.to_string(index=False))
print(f"\nShot_counter features in top ranks: {len(shot_counter_features)}")

# ── 2. HYDRA DATA — status code breakdown ──
print("\n[2] HYDRA DATA — STATUS CODE BREAKDOWN")
print("-" * 40)
try:
    hydra = pd.read_parquet(ROOT / "new_processed_data" / "HYDRA_TRAIN.parquet")
    print(f"Total Hydra rows: {len(hydra)}")
    print(f"Columns: {list(hydra.columns)}")
    
    # Find status column
    status_col = None
    for c in ["machine_status_code", "machine_status_name", "status_code"]:
        if c in hydra.columns:
            status_col = c
            break
    
    if status_col:
        print(f"\nStatus breakdown (using '{status_col}'):")
        status_scrap = hydra.groupby(status_col).agg(
            rows=("scrap_quantity", "count"),
            total_scrap=("scrap_quantity", "sum"),
            total_yield=("yield_quantity", "sum")
        ).reset_index()
        status_scrap["scrap_pct"] = (
            status_scrap["total_scrap"] / 
            (status_scrap["total_scrap"] + status_scrap["total_yield"]).replace(0, 1) * 100
        ).round(1)
        print(status_scrap.sort_values("total_scrap", ascending=False).to_string(index=False))
    
    # Check what scrap looks like
    scrap_rows = hydra[hydra["scrap_quantity"] > 0]
    print(f"\nRows with scrap > 0: {len(scrap_rows)} ({len(scrap_rows)/len(hydra)*100:.1f}%)")
    print(f"Total scrap parts in training: {int(hydra['scrap_quantity'].sum())}")
    print(f"Total yield parts in training: {int(hydra['yield_quantity'].sum())}")
    
except Exception as e:
    print(f"Error reading HYDRA_TRAIN.parquet: {e}")

# ── 3. LABELED TRAINING DATA — how many shots labeled as scrap ──
print("\n[3] TRAINING LABEL QUALITY")
print("-" * 40)
try:
    # Try the rolling features file which has the actual labels
    label_files = [
        ROOT / "processed" / "features" / "rolling_features_with_context.parquet",
        ROOT / "new_processed_data" / "FINAL_TRAINING_MASTER.parquet",
    ]
    for lf in label_files:
        if lf.exists():
            df = pd.read_parquet(lf)
            print(f"\nFile: {lf.name}")
            print(f"  Shape: {df.shape}")
            label_col = None
            for c in ["early_scrap_risk", "is_scrap", "is_scrap_actual"]:
                if c in df.columns:
                    label_col = c
                    break
            if label_col:
                counts = df[label_col].value_counts()
                print(f"  Label column: {label_col}")
                print(f"  Scrap (1): {counts.get(1, 0):,} rows ({counts.get(1,0)/len(df)*100:.3f}%)")
                print(f"  Normal (0): {counts.get(0, 0):,} rows")
                print(f"  Imbalance ratio: {counts.get(0,0)/max(counts.get(1,1),1):.0f}:1")
            break
except Exception as e:
    print(f"Error: {e}")

# ── 4. FEB TEST DATA — what the dashboard actually uses ──
print("\n[4] FEB TEST RESULTS — DASHBOARD DATA QUALITY")
print("-" * 40)
try:
    feb = pd.read_parquet(ROOT / "new_processed_data" / "FEB_TEST_RESULTS.parquet")
    print(f"Shape: {feb.shape}")
    print(f"Columns: {list(feb.columns)}")
    print(f"\nDate range: {feb['timestamp'].min()} to {feb['timestamp'].max()}")
    
    if "is_scrap_actual" in feb.columns:
        scrap_actual = feb["is_scrap_actual"].sum()
        print(f"Actual scrap rows: {int(scrap_actual):,} ({scrap_actual/len(feb)*100:.3f}%)")
    
    if "scrap_probability" in feb.columns:
        print(f"\nScrap probability distribution:")
        print(f"  Mean: {feb['scrap_probability'].mean():.4f}")
        print(f"  Max:  {feb['scrap_probability'].max():.4f}")
        print(f"  >0.3: {(feb['scrap_probability']>0.3).sum():,} rows")
        print(f"  >0.5: {(feb['scrap_probability']>0.5).sum():,} rows")
        print(f"  >0.7: {(feb['scrap_probability']>0.7).sum():,} rows")
    
    if "machine_id_normalized" in feb.columns:
        print(f"\nMachine breakdown:")
        print(feb.groupby("machine_id_normalized")["is_scrap_actual"].agg(
            total_rows="count",
            scrap_rows="sum"
        ).to_string())
        
except Exception as e:
    print(f"Error reading FEB_TEST_RESULTS: {e}")

# ── 5. MACHINE DATA STRUCTURE ──
print("\n[5] TEST PARQUET STRUCTURE (M231)")
print("-" * 40)
try:
    m231 = pd.read_parquet(ROOT / "new_processed_data" / "M231_TEST.parquet")
    print(f"Shape: {m231.shape}")
    print(f"Columns: {list(m231.columns)}")
    print(f"Date range: {m231['timestamp'].min()} to {m231['timestamp'].max()}")
    print(f"\nSample rows:")
    print(m231.head(5).to_string())
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE — paste full output to Claude")
print("=" * 60)