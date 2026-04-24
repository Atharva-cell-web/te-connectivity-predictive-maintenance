import polars as pl
import pandas as pd
import joblib
import os
import glob
from pathlib import Path
import time
from datetime import timedelta
import sys
import numpy as np

# Add project root to sys.path so we can import from backend
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.feature_utils import augment_temporal_signal_features

# Official Production Mappings for 1-to-1 Parity
MACHINE_MAPPING = {
    "M231": 0,
    "M356": 1,
    "M471": 2,
    "M607": 3,
    "M612": 4
}

def process_raw_data():
    print("🚀 Starting UNIFIED GOLDEN-RUN EXTRACTION (V3)...")
    start_time = time.time()
    HORIZONS = [5, 10, 15, 20, 25, 30]
    
    # 1. Setup Paths
    project_root = Path(r"D:\te-connectivity-3")
    feature_path = project_root / "models" / "production_features.pkl"
    raw_dirs = [
        r"D:\new data",
        r"D:\old data",
        r"D:\te-connectivity-3\pipeline  data\new data= 16 april 2026"
    ]
    output_dir = project_root / "new_processed_data"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 2. Load Expected Features
    if not feature_path.exists():
        print(f"❌ Error: Could not find {feature_path}")
        return
    expected_features = joblib.load(feature_path)
    # We filter out encoded columns from the EAV parser step, but we will add them back later
    parser_features = [str(f) for f in expected_features if "encoded" not in str(f) and "code" not in str(f)]
    print(f"✅ Loaded {len(expected_features)} Oracle features specification.")

    # 3. Extract Labels (Type 2: MES Excel)
    print("📋 Extracting Labels from MES Records...")
    mes_files = []
    for d in raw_dirs:
        if os.path.exists(d):
            mes_files.extend(glob.glob(os.path.join(d, "MES_Manufacturing_*.xlsx")))
    
    if not mes_files:
        print("⚠️ Warning: No MES Excel file found. Data will have no labels.")
        scrap_events = pd.DataFrame(columns=['machine', 'timestamp'])
    else:
        all_mes = []
        for mf in mes_files:
            print(f"   -> Reading: {os.path.basename(mf)}")
            df = pd.read_excel(mf)
            df['machine'] = df['machine_id'].str.replace('-', '').str.upper()
            df['timestamp'] = pd.to_datetime(df['machine_event_create_date']) + pd.to_timedelta(df['machine_event_create_time'], unit='s')
            all_mes.append(df[df['scrap_quantity'] > 0][['machine', 'timestamp']])
        
        scrap_events = pd.concat(all_mes).drop_duplicates()
        print(f"✅ Found {len(scrap_events)} total scrap events across all machines.")

    # 4. Process Telemetry (Type 1: EAV CSVs)
    all_csvs = []
    for d in raw_dirs:
        if os.path.exists(d):
            all_csvs.extend(glob.glob(os.path.join(d, "*.csv")))
    
    print(f"✅ Found {len(all_csvs)} telemetry CSV files. Processing...")
    final_pivoted_parts = []
    
    for csv_file in all_csvs:
        if "MES" in os.path.basename(csv_file): continue
        print(f"   -> Parsing: {os.path.basename(csv_file)}")
        try:
            lazy_df = (
                pl.scan_csv(csv_file, ignore_errors=True, infer_schema_length=50000)
                .select(["machine_definition", "timestamp", "variable_name", "value"])
                .filter(pl.col("variable_name").is_in(parser_features))
            )
            df_part = lazy_df.with_columns(pl.col("value").cast(pl.Float32, strict=False)).collect()
            if df_part.is_empty(): continue
            
            df_part = df_part.with_columns(pl.col("machine_definition").str.extract(r"^(M\d+)", 1).alias("machine"))
            pivoted = df_part.pivot(index=["machine", "timestamp"], on="variable_name", values="value", aggregate_function="mean")
            final_pivoted_parts.append(pivoted)
        except Exception as e:
            print(f"   ⚠️ Error processing {os.path.basename(csv_file)}: {e}")

    if not final_pivoted_parts:
        print("❌ Error: No telemetry data was successfully processed.")
        return

    print("🔄 Consolidating and Encoding Features...")
    master_df = pl.concat(final_pivoted_parts, how="diagonal")
    master_df = master_df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
    ).sort(["machine", "timestamp"])

    # 5. Golden Parity Augmentation
    print("🧠 Applying Production Math (Safety Limits, Rolling Trends, Categorical Encoding)...")
    p_df = master_df.to_pandas()
    
    # A. Add machine_id_encoded (The 275th Feature)
    p_df['machine_id_encoded'] = p_df['machine'].map(MACHINE_MAPPING).fillna(0.0).astype(float)
    
    # B. Apply 250+ Industrial features
    p_df = augment_temporal_signal_features(p_df)
    
    # C. Standardize timestamps
    p_df['timestamp'] = pd.to_datetime(p_df['timestamp'])
    scrap_events['timestamp'] = pd.to_datetime(scrap_events['timestamp'])
    
    # 6. Horizon Labeling
    print("🎯 Calculating Ground-Truth Multi-Horizon Labels...")
    for h in HORIZONS:
        p_df[f'scrap_{h}m'] = 0
    
    for machine in p_df['machine'].unique():
        machine_scrap = scrap_events[scrap_events['machine'] == machine]['timestamp'].values
        if len(machine_scrap) == 0: continue
        
        machine_mask = p_df['machine'] == machine
        machine_times = p_df.loc[machine_mask, 'timestamp'].values
        
        for h in HORIZONS:
            horizon_delta = np.timedelta64(h, 'm')
            # Vectorized approach for speed
            labels = []
            for t in machine_times:
                has_scrap = any((machine_scrap >= t) & (machine_scrap <= t + horizon_delta))
                labels.append(1 if has_scrap else 0)
            p_df.loc[machine_mask, f'scrap_{h}m'] = labels

    # 7. Final Spec Check
    # Ensure every single feature in production_features.pkl exists (even if 0.0)
    for feat in expected_features:
        if feat not in p_df.columns:
            p_df[feat] = 0.0

    # 8. Save
    out_file = output_dir / "KAGGLE_MASTER_275_LABELED.parquet"
    p_df.to_parquet(out_file, compression='zstd')
    
    duration = (time.time() - start_time) / 60
    print(f"\n✅ GOLDEN RUN SUCCESS! Dataset saved to {out_file}")
    print(f"📊 Totals: {len(p_df)} rows, {len(p_df.columns)} columns.")
    print(f"⏱️ Runtime: {duration:.2f} minutes.")

if __name__ == "__main__":
    process_raw_data()
