import pandas as pd
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_FILE = PROJECT_ROOT / "processed" / "features" / "rolling_features_wide.parquet"
DEMO_FILE = PROJECT_ROOT / "processed" / "features" / "rolling_features_demo.parquet"

def create_demo_slice():
    print(f"Loading full dataset from: {FULL_FILE}")
    print("This might take a moment...")
    
    # Read the file
    df = pd.read_parquet(FULL_FILE)
    
    # Sort by time just in case
    df = df.sort_values("event_timestamp")
    
    # Get the last 4 hours of data ONLY
    # This is plenty for a demo (which usually looks at "now")
    latest_time = df["event_timestamp"].max()
    cutoff_time = latest_time - pd.Timedelta(hours=4)
    
    print(f"Slicing data from {cutoff_time} to {latest_time}")
    
    demo_df = df[df["event_timestamp"] >= cutoff_time].copy()
    
    # Save the small file
    demo_df.to_parquet(DEMO_FILE, index=False)
    
    print(f"✅ Success! Demo file created at: {DEMO_FILE}")
    print(f"Original Rows: {len(df)}")
    print(f"Demo Rows:     {len(demo_df)}")
    print(f"Size Reduction: {len(demo_df)/len(df):.1%}")

if __name__ == "__main__":
    create_demo_slice()