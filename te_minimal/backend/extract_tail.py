import pandas as pd
import glob
import os

# 1. Look for the parquet files in your data folder
# Adjust this path if your parquet files are stored somewhere else
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "new_processed_data")
OUTPUT_DIR = "verification_files"

os.makedirs(OUTPUT_DIR, exist_ok=True)

parquet_files = glob.glob(os.path.join(DATA_DIR, "M*_TEST.parquet"))

if not parquet_files:
    print(f"Could not find parquet files in '{DATA_DIR}'. Please update the DATA_DIR path in this script.")
else:
    for file in parquet_files:
        machine_id = os.path.basename(file).split('.')[0]
        print(f"Extracting tail for {machine_id}...")
        
        # Load the massive file
        df = pd.read_parquet(file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Find the absolute end and step back exactly 4 hours
        max_time = df["timestamp"].max()
        cutoff = max_time - pd.Timedelta(hours=4)
        
        # Slice off just the end of the file
        tail_df = df[df["timestamp"] >= cutoff].copy()
        tail_df = tail_df.sort_values(by="timestamp")
        
        # Save as a lightweight CSV
        output_file = os.path.join(OUTPUT_DIR, f"{machine_id}_final_4_hours.csv")
        tail_df.to_csv(output_file, index=False)
        print(f"✅ Saved {len(tail_df)} rows to {output_file}\n")

    print(f"🎉 Done! Open the '{OUTPUT_DIR}' folder to see your verification files.")