import pandas as pd
import numpy as np
from pathlib import Path
import json
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "processed" / "features" / "rolling_features_wide_labeled.parquet"
OUTPUT_FILE = PROJECT_ROOT / "processed" / "features" / "rolling_features_with_context.parquet"
MACHINE_DATA_DIR = PROJECT_ROOT / "new_processed_data"
ENCODINGS_FILE = PROJECT_ROOT / "models" / "part_tool_encodings.json"


def extract_tool_from_definition(machine_definition: str) -> str:
    if pd.isna(machine_definition) or not machine_definition:
        return "UNKNOWN"
    match = re.search(r'-([A-Za-z0-9]+)$', str(machine_definition))
    if match:
        return match.group(1)
    return str(machine_definition)


def load_machine_context() -> pd.DataFrame:
    print("Loading machine context (part/tool) from parquet files...")
    
    context_records = []
    machine_files = list(MACHINE_DATA_DIR.glob("M*_TEST.parquet"))
    
    print(f"Found {len(machine_files)} machine files")
    
    for machine_file in machine_files:
        try:
            machine_id = machine_file.stem.replace("_TEST", ")
            
            cols_to_load = ["timestamp", "machine_definition"]
            
            df_sample = pd.read_parquet(machine_file, engine="pyarrow")
            available_cols = df_sample.columns.tolist()
            
            # Look for part_number, tool_number, or similar columns
            part_cols = [c for c in available_cols if 'part' in c.lower()]
            tool_cols = [c for c in available_cols if 'tool' in c.lower()]
            
            # Get unique machine_definition for this machine
            if 'machine_definition' in available_cols:
                definitions = df_sample['machine_definition'].dropna().unique()
                if len(definitions) > 0:
                    machine_def = str(definitions[0])
                    tool_id = extract_tool_from_definition(machine_def)
                else:
                    machine_def = "UNKNOWN"
                    tool_id = "UNKNOWN"
            else:
                machine_def = "UNKNOWN"
                tool_id = "UNKNOWN"
            
            # Get part number if available
            part_number = "UNKNOWN"
            for col in part_cols:
                parts = df_sample[col].dropna().unique()
                if len(parts) > 0:
                    part_number = str(parts[0])
                    break
            
            # Get tool number if available (separate from machine_definition)
            for col in tool_cols:
                tools = df_sample[col].dropna().unique()
                if len(tools) > 0:
                    tool_id = str(tools[0])
                    break
            
            context_records.append({
                "machine_id": machine_id,
                "machine_definition": machine_def,
                "tool_id": tool_id,
                "part_number": part_number,
                "n_records": len(df_sample)
            })
            
            print(f"  {machine_id}: tool={tool_id}, part={part_number}")
            
        except Exception as e:
            print(f"  [WARN] Error processing {machine_file.name}: {e}")
    
    context_df = pd.DataFrame(context_records)
    print(f"[INFO] Loaded context for {len(context_df)} machines")
    
    return context_df


def create_encodings(context_df: pd.DataFrame) -> dict:
    """Create label encodings for categorical features."""
    encodings = {
        "tool_id": {},
        "part_number": {},
        "machine_id": {}
    }
    
    # Encode tool_id
    unique_tools = context_df["tool_id"].unique()
    for i, tool in enumerate(sorted(unique_tools)):
        encodings["tool_id"][tool] = i
    
    # Encode part_number
    unique_parts = context_df["part_number"].unique()
    for i, part in enumerate(sorted(unique_parts)):
        encodings["part_number"][part] = i
    
    # Encode machine_id
    unique_machines = context_df["machine_id"].unique()
    for i, machine in enumerate(sorted(unique_machines)):
        encodings["machine_id"][machine] = i
    
    print(f"[INFO] Created encodings: {len(encodings['tool_id'])} tools, "
          f"{len(encodings['part_number'])} parts, {len(encodings['machine_id'])} machines")
    
    return encodings


def add_context_features(df: pd.DataFrame, context_df: pd.DataFrame, encodings: dict) -> pd.DataFrame:
    """Add encoded part/tool features to the training data."""
    print("[INFO] Adding context features to training data...")
    
    # Check if machine_id column exists
    machine_col = None
    for col in ["machine_id", "machine_id_normalized"]:
        if col in df.columns:
            machine_col = col
            break
    
    if machine_col is None:
        print("[WARN] No machine_id column found. Adding default context features.")
        df["tool_id_encoded"] = 0
        df["part_number_encoded"] = 0
        df["machine_id_encoded"] = 0
        return df
    
    # Normalize machine IDs for matching
    df["_machine_norm"] = df[machine_col].astype(str).str.upper().str.replace("-", "").str.strip()
    context_df["_machine_norm"] = context_df["machine_id"].astype(str).str.upper().str.replace("-", "").str.strip()
    
    # Merge context
    df = df.merge(
        context_df[["_machine_norm", "tool_id", "part_number"]],
        on="_machine_norm",
        how="left"
    )
    
    # Fill missing values
    df["tool_id"] = df["tool_id"].fillna("UNKNOWN")
    df["part_number"] = df["part_number"].fillna("UNKNOWN")
    
    # Apply encodings
    df["tool_id_encoded"] = df["tool_id"].map(encodings["tool_id"]).fillna(0).astype(int)
    df["part_number_encoded"] = df["part_number"].map(encodings["part_number"]).fillna(0).astype(int)
    df["machine_id_encoded"] = df["_machine_norm"].map(
        {k.upper().replace("-", ""): v for k, v in encodings["machine_id"].items()}
    ).fillna(0).astype(int)
    
    # Drop temporary columns
    df = df.drop(columns=["_machine_norm", "tool_id", "part_number"], errors="ignore")
    
    print(f"[INFO] Added: tool_id_encoded, part_number_encoded, machine_id_encoded")
    print(f"[INFO] Unique tool encodings used: {df['tool_id_encoded'].nunique()}")
    print(f"[INFO] Unique part encodings used: {df['part_number_encoded'].nunique()}")
    
    return df


def main():
    print("=" * 60)
    print("ADDING PART/TOOL CONTEXT FEATURES")
    print("=" * 60)
    
    # Step 1: Load machine context
    context_df = load_machine_context()
    
    # Step 2: Create encodings
    encodings = create_encodings(context_df)
    
    # Save encodings for inference
    ENCODINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ENCODINGS_FILE, "w") as f:
        json.dump(encodings, f, indent=2)
    print(f"[INFO] Saved encodings to {ENCODINGS_FILE}")
    
    # Step 3: Load training data
    print(f"\n[INFO] Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE, engine="pyarrow")
    print(f"[INFO] Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Step 4: Add context features
    df = add_context_features(df, context_df, encodings)
    
    # Step 5: Save enhanced data
    print(f"\n[INFO] Saving to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, engine="pyarrow", index=False)
    print(f"[SUCCESS] Saved with {len(df.columns)} columns (added 3 context features)")
    
    print("\n" + "=" * 60)
    print("CONTEXT FEATURES ADDED")
    print("=" * 60)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Encodings: {ENCODINGS_FILE}")
    print("New columns: tool_id_encoded, part_number_encoded, machine_id_encoded")
    print("=" * 60)


if __name__ == "__main__":
    main()
