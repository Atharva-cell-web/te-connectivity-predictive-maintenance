import pandas as pd
import os
import gc

# --- CONFIGURATION ---
PROCESSED_DIR = r"D:\te connectivity 3\new_processed_data"

print("🚀 Starting Data Merger & Pivoter...")

# 1. Load Hydra Training Data (The "Answers")
print("\n⏳ Loading Hydra Scrap Events...")
hydra_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "HYDRA_TRAIN.parquet"))

# Keep only actual scrap events
scrap_events = hydra_df[hydra_df['scrap_quantity'] > 0].copy()

# --- FIX: NORMALIZE MACHINE IDs ---
# This converts "M-231 ", "m231", etc. into strictly "M231"
scrap_events['machine_id'] = scrap_events['machine_id'].astype(str).str.replace('-', '').str.upper().str.strip()

scrap_events = scrap_events[['machine_id', 'machine_event_create_date']].rename(columns={'machine_event_create_date': 'timestamp'})
scrap_events['is_scrap'] = 1
scrap_events['timestamp'] = pd.to_datetime(scrap_events['timestamp'], utc=True)
scrap_events = scrap_events.sort_values('timestamp')

print(f"   -> Found {len(scrap_events)} actual scrap events in training period.")

master_train_dfs = []

# 2. Process each Machine's Training File
for file in os.listdir(PROCESSED_DIR):
    if file.endswith("_TRAIN.parquet") and "MERGED" not in file and not file.startswith("HYDRA"):
        # Normalize the filename ID too
        machine_id_raw = file.split('_')[0] 
        machine_id = machine_id_raw.replace('-', '').upper().strip()
        
        print(f"\n⚙️ Processing {machine_id_raw} (Matching as {machine_id})...")

        # Load machine data
        df = pd.read_parquet(os.path.join(PROCESSED_DIR, file))

        # FORCE VALUES TO BE NUMBERS
        print("   -> Converting text values to numeric...")
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value'])
        
        # Drop any rows that were completely broken text and became NaN
        df = df.dropna(subset=['value'])
        # ---------------------------------------

        # PIVOT: Turn "variable_name" into columns. Take the average if multiple pings happen at the same exact timestamp.
        print("   -> Pivoting sensor data (Making it AI-ready)...")
        pivot_df = df.pivot_table(
            index='timestamp', 
            columns='variable_name', 
            values='value', 
            aggfunc='mean'
        ).reset_index()

        pivot_df = pivot_df.sort_values('timestamp')

        # Get scrap events specific to THIS machine
        machine_scrap = scrap_events[scrap_events['machine_id'] == machine_id].copy()

        if not machine_scrap.empty:
            print(f"   -> Merging {len(machine_scrap)} scrap events with sensor data...")
            # If a sensor reading happened within 5 minutes of a logged scrap event, label it as scrap (1)
            merged = pd.merge_asof(
                pivot_df,
                machine_scrap[['timestamp', 'is_scrap']],
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta('5 minutes')
            )
            merged['is_scrap'] = merged['is_scrap'].fillna(0)
        else:
            print("   -> No scrap events found for this machine in training period.")
            pivot_df['is_scrap'] = 0
            merged = pivot_df

        # Add machine ID back in
        merged['machine_id'] = machine_id
        
        # Save this machine's processed data temporarily
        merged.to_parquet(os.path.join(PROCESSED_DIR, f"{machine_id}_MERGED_TRAIN.parquet"), index=False)
        master_train_dfs.append(merged)

        # Clean memory
        del df, pivot_df, merged
        gc.collect()

# 3. Combine everything into the Final Master File
print("\n🔗 Combining all machines into Final Master Training File...")
final_train_df = pd.concat(master_train_dfs, ignore_index=True)

final_save_path = os.path.join(PROCESSED_DIR, "FINAL_TRAINING_MASTER.parquet")
final_train_df.to_parquet(final_save_path, index=False)

# Calculate some quick stats for you
total_rows = len(final_train_df)
total_scrap = len(final_train_df[final_train_df['is_scrap'] == 1])
print(f"✅ DONE! Master File Saved: {final_save_path}")
print(f"📊 Final Dataset Size: {total_rows} rows")
print(f"🧨 Total Sensor Readings Labeled as Scrap: {total_scrap}")