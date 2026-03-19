"""
check_all_sources.py
Quick diagnostic — reads just first/last rows of every CSV across all 3 exports.
Runtime: 3-5 minutes
"""
import pandas as pd
from pathlib import Path
import subprocess

FOLDERS = ["old data", "new data", "new_raw_data"]

for folder in FOLDERS:
    p = Path(folder)
    if not p.exists():
        print(f"=== {folder} === NOT FOUND\n")
        continue

    print(f"\n{'='*60}")
    print(f"FOLDER: {folder}")
    print(f"{'='*60}")

    # CSV files
    csv_files = sorted(p.glob("*.csv"))
    for f in csv_files:
        print(f"\n  FILE: {f.name} ({f.stat().st_size/1e6:.0f} MB)")

        # Read first 3 rows to get columns + first timestamp
        df_head = pd.read_csv(f, nrows=3)
        print(f"  Columns: {list(df_head.columns)}")

        # Find timestamp column
        ts_col = None
        for c in df_head.columns:
            if "time" in c.lower():
                ts_col = c
                break
        if ts_col:
            first_ts = pd.to_datetime(df_head[ts_col].iloc[0], utc=True)
            print(f"  First timestamp: {first_ts}")
        else:
            print(f"  No timestamp column found")

        # Read last few rows using tail
        df_tail = pd.read_csv(f, skiprows=lambda x: x < max(0, sum(1 for _ in open(f, encoding='utf-8')) - 5))
        if ts_col and ts_col in df_tail.columns:
            last_ts = pd.to_datetime(df_tail[ts_col].iloc[-1], utc=True, errors='coerce')
            print(f"  Last timestamp:  {last_ts}")

    # Excel files
    xlsx_files = sorted(p.glob("*.xlsx"))
    for f in xlsx_files:
        print(f"\n  FILE: {f.name} ({f.stat().st_size/1e6:.1f} MB)")
        df = pd.read_excel(f, nrows=3)
        print(f"  Columns (first 8): {list(df.columns[:8])}")
        ts_col = None
        for c in df.columns:
            if "date" in c.lower() or "time" in c.lower():
                ts_col = c
                break
        if ts_col:
            df_full = pd.read_excel(f, usecols=[ts_col])
            df_full[ts_col] = pd.to_datetime(df_full[ts_col], utc=True, errors='coerce')
            print(f"  Date range: {df_full[ts_col].min().date()} to {df_full[ts_col].max().date()}")
            print(f"  Total rows: {len(df_full):,}")

print(f"\n{'='*60}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*60}")