import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from config_limits import SAFE_LIMITS, PARAMETER_LABELS

# Resolve paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WIDE_FILE = PROJECT_ROOT / "processed" / "features" / "rolling_features_wide.parquet"
OUTPUT_DIR = PROJECT_ROOT / "plots"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

def plot_parameter_trend(machine_id, parameter_base_name):
    """
    Generates a trend plot for the last 60 minutes of data for a specific parameter.
    Overlays safe limits if they exist.
    """
    print(f"--- Generating Trend Plot for {parameter_base_name} on {machine_id} ---")

    # 1. Load Data
    if not WIDE_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {WIDE_FILE}")
    
    # Read only necessary columns to save memory
    # We use '__last_5m' as the closest proxy to the raw sensor value trace
    target_col = f"{parameter_base_name}__last_5m"
    cols = ["machine_id_normalized", "event_timestamp", target_col]
    
    try:
        df = pd.read_parquet(WIDE_FILE, columns=cols)
    except Exception:
        # Fallback if column doesn't exist (e.g. slight naming mismatch)
        print(f"❌ Column {target_col} not found in parquet.")
        return

    # 2. Filter for Machine & Time Window
    df = df[df["machine_id_normalized"] == machine_id].copy()
    df = df.sort_values("event_timestamp")
    
    if df.empty:
        print("❌ No data found for this machine.")
        return

    # Get the last 60 minutes relative to the dataset's end
    latest_ts = df["event_timestamp"].max()
    start_ts = latest_ts - pd.Timedelta(minutes=60)
    window_df = df[df["event_timestamp"] >= start_ts]

    print(f"Plotting data from {start_ts} to {latest_ts} ({len(window_df)} points)")

    # 3. Setup Plot
    plt.figure(figsize=(10, 6))
    
    # Plot the sensor data
    plt.plot(window_df["event_timestamp"], window_df[target_col], label="Actual Value", color="blue", linewidth=2)

    # 4. Overlay Safe Limits
    limits = SAFE_LIMITS.get(parameter_base_name, {})
    
    if "max" in limits and limits["max"] is not None:
        plt.axhline(y=limits["max"], color='red', linestyle='--', linewidth=2, label=f"Max Limit ({limits['max']})")
        # Add text label
        plt.text(window_df["event_timestamp"].min(), limits["max"], f" Max: {limits['max']}", color='red', verticalalignment='bottom')

    if "min" in limits and limits["min"] is not None:
        plt.axhline(y=limits["min"], color='red', linestyle='--', linewidth=2, label=f"Min Limit ({limits['min']})")
        plt.text(window_df["event_timestamp"].min(), limits["min"], f" Min: {limits['min']}", color='red', verticalalignment='top')

    # 5. Formatting
    readable_name = PARAMETER_LABELS.get(parameter_base_name, parameter_base_name)
    unit = limits.get("unit", "")
    
    plt.title(f"Trend Analysis: {readable_name} (Last 60 Min)", fontsize=14)
    plt.xlabel("Time (UTC)", fontsize=12)
    plt.ylabel(f"Value ({unit})", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 6. Save
    filename = f"trend_{machine_id}_{parameter_base_name}.png"
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path)
    plt.close()
    
    print(f"✅ Plot saved to: {save_path}")

if __name__ == "__main__":
    # Test run: Plot 'Injection_pressure' for M-231
    # (Even if risk is LOW, we verify the visualizer works)
    plot_parameter_trend("M-231", "Injection_pressure")