import pandas as pd
import os
import gc

# --- CONFIGURATION ---
PROCESSED_DIR = r"D:\te connectivity 3\new_processed_data"

# v1.2b-filter-startup-scrap
# KEY FIX: Use 5-minute point merge (original approach) BUT only on
# status 200 (Productie) rows. The window-based approach labeled 32-46%
# of rows as scrap because Hydra windows are entire shifts (up to 18 hrs).
# The machine_event_create_date is the timestamp of the actual scrap EVENT
# within the shift — this is what we merge against, not the window end.
#
# CONFIRMED CORRECT SCRAP RATE: 0.4-0.6% from real production data.
# If labeling produces >5%, something is wrong.
PRODUCTION_STATUS_CODES = [200]

print("🚀 Starting Data Merger & Pivoter...")
print("   [v1.2b] Status 200 only + 5-min point merge (corrected labeling)")

# 1. Load Hydra
print("\n⏳ Loading Hydra Scrap Events...")
hydra_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "HYDRA_TRAIN.parquet"))
print(f"   -> Total Hydra rows: {len(hydra_df)}")

# Filter to status 200 production scrap only
scrap_source = hydra_df[hydra_df['scrap_quantity'] > 0].copy()

if 'machine_status_code' in hydra_df.columns:
    before_count = len(scrap_source)
    scrap_source = scrap_source[
        scrap_source['machine_status_code'].isin(PRODUCTION_STATUS_CODES)
    ]
    print(f"   -> Status filter: {len(scrap_source)} rows kept "
          f"(removed {before_count - len(scrap_source)} non-production rows)")

    # Show what we kept
    kept_summary = scrap_source.groupby('machine_status_code')['scrap_quantity'].agg(
        rows='count', total_scrap='sum'
    )
    print(f"   -> Kept status breakdown:\n{kept_summary.to_string()}")
else:
    print("   -> WARNING: machine_status_code not found, using all scrap rows")

# Normalize machine IDs
scrap_source['machine_id_clean'] = (
    scrap_source['machine_id'].astype(str)
    .str.replace('-', '').str.upper().str.strip()
)

# Build the merge timestamp from machine_event_create_date + create_time
# This is the actual moment the scrap batch was logged — use as point event
def build_event_ts(df):
    """Combine date + seconds-of-day → UTC timestamp."""
    base = pd.to_datetime(df['machine_event_create_date'])
    if base.dt.tz is not None:
        base = base.dt.tz_convert('UTC').dt.tz_localize(None)
    if 'machine_event_create_time' in df.columns:
        offset = pd.to_timedelta(
            pd.to_numeric(df['machine_event_create_time'], errors='coerce').fillna(0),
            unit='s'
        )
        return (base + offset).dt.tz_localize('UTC')
    else:
        return base.dt.tz_localize('UTC')

scrap_source['merge_ts'] = build_event_ts(scrap_source)
scrap_source = scrap_source.sort_values('merge_ts')

print(f"\n   -> Merge timestamps built: {len(scrap_source)} scrap events")
print(f"   -> Machine IDs: {sorted(scrap_source['machine_id_clean'].unique())}")
print(f"   -> Date range: {scrap_source['merge_ts'].min()} to {scrap_source['merge_ts'].max()}")

master_train_dfs = []

# 2. Process each machine
train_files = [
    f for f in os.listdir(PROCESSED_DIR)
    if f.endswith("_TRAIN.parquet")
    and "MERGED" not in f
    and not f.startswith("HYDRA")
]
print(f"\n   -> Found {len(train_files)} TRAIN files: {train_files}")

for file in train_files:
    machine_id_raw = file.split('_')[0]
    machine_id = machine_id_raw.replace('-', '').upper().strip()

    print(f"\n⚙️  Processing {machine_id_raw} (as {machine_id})...")

    df = pd.read_parquet(os.path.join(PROCESSED_DIR, file))
    print(f"   -> Raw rows: {len(df):,}")

    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])

    print("   -> Pivoting...")
    pivot_df = df.pivot_table(
        index='timestamp',
        columns='variable_name',
        values='value',
        aggfunc='mean'
    ).reset_index()

    pivot_df['timestamp'] = pd.to_datetime(pivot_df['timestamp'], utc=True)
    pivot_df = pivot_df.sort_values('timestamp').reset_index(drop=True)
    print(f"   -> Cycle rows after pivot: {len(pivot_df):,}")

    # Get this machine's scrap events
    machine_scrap = scrap_source[
        scrap_source['machine_id_clean'] == machine_id
    ][['merge_ts', 'scrap_quantity']].copy()
    machine_scrap = machine_scrap.rename(columns={'merge_ts': 'timestamp'})
    machine_scrap['is_scrap'] = 1
    machine_scrap = machine_scrap.sort_values('timestamp')

    print(f"   -> Scrap events (status 200 only): {len(machine_scrap)}")

    if not machine_scrap.empty:
        # 5-minute nearest merge — label sensor reading as scrap if it falls
        # within 5 minutes of a logged production scrap event
        merged = pd.merge_asof(
            pivot_df,
            machine_scrap[['timestamp', 'is_scrap']],
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta('5 minutes')
        )
        merged['is_scrap'] = merged['is_scrap'].fillna(0).astype(int)
    else:
        pivot_df['is_scrap'] = 0
        merged = pivot_df

    merged['machine_id'] = machine_id

    scrap_n = int(merged['is_scrap'].sum())
    scrap_pct = scrap_n / len(merged) * 100
    print(f"   -> Scrap labels: {scrap_n:,} / {len(merged):,} ({scrap_pct:.3f}%)")

    if scrap_pct > 10:
        print(f"   -> WARNING: Scrap rate {scrap_pct:.1f}% seems high. "
              f"Expected ~0.5-2%. Check Hydra timestamps for this machine.")

    merged.to_parquet(
        os.path.join(PROCESSED_DIR, f"{machine_id}_MERGED_TRAIN.parquet"),
        index=False
    )
    master_train_dfs.append(merged)

    del df, pivot_df, merged
    gc.collect()

# 3. Combine
print(f"\n🔗 Combining {len(master_train_dfs)} machines...")
final_train_df = pd.concat(master_train_dfs, ignore_index=True)

final_save_path = os.path.join(PROCESSED_DIR, "FINAL_TRAINING_MASTER.parquet")
final_train_df.to_parquet(final_save_path, index=False)

total_rows = len(final_train_df)
total_scrap = int(final_train_df['is_scrap'].sum())
scrap_pct = total_scrap / total_rows * 100

print(f"\n✅ DONE! Saved: {final_save_path}")
print(f"📊 Rows: {total_rows:,}")
print(f"🧨 Scrap labeled: {total_scrap:,} ({scrap_pct:.3f}%)")

if scrap_pct < 0.1:
    print("⚠️  Very low scrap rate — check if Hydra timestamps align with sensor timestamps")
elif scrap_pct > 5:
    print("⚠️  High scrap rate — Hydra event timestamps may not align with sensor timestamps precisely")
else:
    print("✓  Scrap rate looks realistic (0.1% - 5% range)")

print(f"\n[v1.2b] Next steps:")
print(f"  NOTE: Skip step4b (pattern features broken — sensor name mismatch)")
print(f"  NOTE: step4c reads rolling_features_wide_labeled.parquet (old file)")
print(f"        — this pipeline path needs review before running")
print(f"  SAFE TO RUN: python scripts/step5_3b_train_lightgbm_wide.py")
print(f"  (it reads rolling_features_with_context.parquet which has the old labels)")
print(f"  To fully retrain on NEW labels, the rolling feature rebuild is needed first.")