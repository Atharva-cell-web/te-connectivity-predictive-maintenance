import pandas as pd
import os
import gc # Garbage collector to prevent RAM crashes

# --- CONFIGURATION ---
RAW_DATA_DIR = r"D:\te connectivity 3\new_raw_data"
PROCESSED_DIR = r"D:\te connectivity 3\new_processed_data"

# Kashyap's Cutoff Date
CUTOFF_DATE = pd.to_datetime("2026-01-11 23:59:59", utc=True)

def process_machine_file(filename):
    filepath = os.path.join(RAW_DATA_DIR, filename)
    machine_id = filename.split('-')[0] # Extracts "M231" from "M231-11.csv"
    
    print(f"\n⏳ Loading {filename} (This may take a minute or two)...")
    
    # 1. Load CSV
    df = pd.read_csv(filepath)
    
    # FIX: Using the exact column name found in the new machine data
    time_col = 'timestamp' 
    print(f"   -> Converting {time_col}...")
    
    # 2. Convert timestamp to standard UTC datetime
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    
    # 3. Split the Data
    print("   -> Splitting data based on Jan 11, 2026 cutoff...")
    train_df = df[df[time_col] <= CUTOFF_DATE]
    test_df = df[df[time_col] > CUTOFF_DATE]
    
    # 4. Save as Parquet (Fast & Compressed)
    train_save_path = os.path.join(PROCESSED_DIR, f"{machine_id}_TRAIN.parquet")
    test_save_path = os.path.join(PROCESSED_DIR, f"{machine_id}_TEST.parquet")
    
    train_df.to_parquet(train_save_path, index=False)
    test_df.to_parquet(test_save_path, index=False)
    
    print(f"✅ {machine_id} Done! Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    
    # 5. Clear Memory (CRITICAL for laptops)
    del df, train_df, test_df
    gc.collect()
    
def process_hydra_data():
    print("\n⏳ Loading Hydra (Scrap) Excel file...")
    hydra_file = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.xlsx')][0]
    filepath = os.path.join(RAW_DATA_DIR, hydra_file)
    
    df = pd.read_excel(filepath)
    
    # FIX: Using the exact column name found in the new TE export
    time_col = 'machine_event_create_date' 
    print(f"   -> Converting {time_col}...")
    
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    
    train_df = df[df[time_col] <= CUTOFF_DATE]
    test_df = df[df[time_col] > CUTOFF_DATE]
    
    train_df.to_parquet(os.path.join(PROCESSED_DIR, "HYDRA_TRAIN.parquet"), index=False)
    test_df.to_parquet(os.path.join(PROCESSED_DIR, "HYDRA_TEST.parquet"), index=False)
    print(f"✅ Hydra Done! Train: {len(train_df)} rows | Test: {len(test_df)} rows")

if __name__ == "__main__":
    print("🚀 Starting Data Splitter Pipeline...")
    
    # Ensure output folder exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Process Hydra first
    process_hydra_data()
    
    # Process Machine CSVs one by one
    for file in os.listdir(RAW_DATA_DIR):
        if file.endswith(".csv"):
            process_machine_file(file)
            
    print("\n🎉 ALL DATA SUCCESSFULLY SPLIT AND SAVED!")