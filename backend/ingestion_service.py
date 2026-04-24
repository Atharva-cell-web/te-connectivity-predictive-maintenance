import os
import re
import pandas as pd
from pathlib import Path
import subprocess
import threading
import sys
import logging

# Import processing logic from scripts (if they were modular)
# Since they aren't, we will implement a clean modular version here for the API to use.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "pipeline  data"
PROCESSED_DIR = PROJECT_ROOT / "new_processed_data"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IngestionService")

from backend.data_access import normalize_machine_id

def _coerce_datetime(values, utc: bool = True):
    """Parse timestamps without noisy mixed-format warnings when possible."""
    try:
        return pd.to_datetime(values, utc=utc, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, utc=utc, errors="coerce")

def _resolve_target_dir(target_dir: Path | None = None) -> Path:
    """Resolve the ingestion folder once so validation and processing stay aligned."""
    return target_dir or get_latest_dated_folder() or RAW_DIR

# ── Module-level helper (accessible from ALL functions) ──────────
def _classify_file(path) -> str:
    """Classify a file as Hydra, MSE, Param, or Other based on name/content."""
    name = path.name.lower()
    if "param" in name: return "Param"
    if "mse" in name: return "MSE"

    # Adaptive Sniffing: peek columns if ambiguous
    try:
        if path.suffix.lower() == ".csv":
            peek = pd.read_csv(path, nrows=0)
            if "variable_name" in peek.columns and "value" in peek.columns:
                return "MSE"
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            peek = pd.read_excel(path, nrows=0)
            if any(c in peek.columns for c in ["machine_event_create_date", "cycle_time"]):
                return "Hydra"
    except Exception:
        pass

    if path.suffix.lower() == ".csv": return "MSE"
    if path.suffix.lower() in [".xls", ".xlsx"]: return "Hydra"
    return "Other"


def get_ingestion_history():
    """Returns a list of files currently in the raw_data directory."""
    if not RAW_DIR.exists():
        return []

    def _extract_machine_label(path: Path) -> str:
        machine = normalize_machine_id(path.name)
        return machine if machine else "AUTO"

    # Identify the target directory to list
    target_dir = get_latest_dated_folder() or RAW_DIR
    if not target_dir.exists():
        return []

    files = []
    # Scan both the target_dir AND the root RAW_DIR for backwards compatibility
    for scan_dir in {target_dir, RAW_DIR}:
        for f in scan_dir.glob("*"):
            if f.is_file() and f.suffix.lower() in [".csv", ".xls", ".xlsx", ".parquet"]:
                stats = f.stat()
                file_type = _classify_file(f)
                files.append({
                    "id": f.name,
                    "name": f.name,
                    "type": file_type,
                    "machine": _extract_machine_label(f) if file_type == "Machine" else "AUTO",
                    "size": f"{stats.st_size / 1024:.1f} KB",
                    "timestamp": pd.to_datetime(stats.st_mtime, unit='s').strftime('%Y-%m-%d %H:%M'),
                    "status": "Ready",
                    "folder": scan_dir.name
                })
    return sorted(files, key=lambda x: x['timestamp'], reverse=True)


def get_latest_dated_folder() -> Path | None:
    """Finds the folder with the highest index or latest date matching 'new data='."""
    if not RAW_DIR.exists():
        return None
    
    # Prefix requested by user: "new data="
    pattern = "new data="
    folders = [f for f in RAW_DIR.iterdir() if f.is_dir() and pattern in f.name]
    
    if not folders:
        return None
        
    # Sort by modification time to find what the user "just connected"
    # This is more robust than parsing complex date strings like "14 march 2026"
    return max(folders, key=lambda p: p.stat().st_mtime)

def get_latest_processed_artifact() -> Path | None:
    """Return the newest processed artifact the dashboard can read."""
    candidates = [
        PROCESSED_DIR / "FINAL_TRAINING_MASTER_V3.parquet",
        PROCESSED_DIR / "FEB_TEST_RESULTS.parquet",
        PROCESSED_DIR / "HYDRA_TRAIN.parquet",
        PROCESSED_DIR / "cleaned_dataset_v4.5f.parquet",
        PROCESSED_DIR / "cleaned_dataset_v4.parquet",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)

def should_refresh_latest_pipeline(target_dir: Path | None = None) -> tuple[bool, str]:
    """
    Decide whether the dashboard should rebuild processed outputs from the latest raw folder.

    We refresh when:
    - a dated raw folder exists, and
    - it is newer than the newest processed artifact, or
    - no processed artifact exists yet.
    """
    raw_dir = _resolve_target_dir(target_dir)
    if not raw_dir.exists():
        return False, "raw folder not found"

    latest_processed = get_latest_processed_artifact()
    if latest_processed is None:
        return True, "no processed artifact found"

    try:
        if raw_dir.stat().st_mtime > latest_processed.stat().st_mtime:
            return True, f"{raw_dir.name} is newer than {latest_processed.name}"
    except Exception:
        return True, "unable to compare raw and processed timestamps"

    return False, f"{latest_processed.name} is current"

def _sync_dynamic_params(path: Path):
    """
    Senior Pro Fix: Extracts threshold overrides from 'Param' files
    and syncs them to metrics/dynamic_limits.json.
    """
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        
        # Look for [parameter, min, max] pattern
        cols = [c.lower() for c in df.columns]
        target_col = next((c for c in df.columns if "param" in c.lower()), None)
        min_col = next((c for c in df.columns if "min" in c.lower()), None)
        max_col = next((c for c in df.columns if "max" in c.lower()), None)

        if not (target_col and (min_col or max_col)):
            return

        overrides = {}
        for _, row in df.iterrows():
            param = str(row[target_col]).strip()
            if not param: continue
            
            p_data = {}
            if min_col: p_data["min"] = float(pd.to_numeric(row[min_col], errors="coerce") or 0.0)
            if max_col: p_data["max"] = float(pd.to_numeric(row[max_col], errors="coerce") or 0.0)
            overrides[param] = p_data

        if overrides:
            limit_file = PROJECT_ROOT / "metrics" / "dynamic_limits.json"
            os.makedirs(limit_file.parent, exist_ok=True)
            
            # Merge with existing
            current = {}
            if limit_file.exists():
                with open(limit_file, "r") as f:
                    try: current = json.load(f)
                    except: pass
            
            current.update(overrides)
            with open(limit_file, "w") as f:
                json.dump(current, f, indent=2)
            logger.info(f"✓ Synced {len(overrides)} parameters from {path.name}")
    except Exception as e:
        logger.error(f"Param sync error for {path.name}: {e}")

def handle_upload(
    mes_files: list[tuple[bytes, str]],
    hydra_content: bytes,
    hydra_filename: str,
):
    """Saves uploaded files to a new dated subfolder using 'new data= [DATE]' format."""
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # User's preferred format: "new data= 14 march 2026"
    # We'll use full month names in lowercase to match their example exactly
    today_str = pd.Timestamp.now().strftime("%d %B %Y").lower()
    
    # Check if a folder for today already exists
    upload_folder = RAW_DIR / f"new data= {today_str}"
    
    # If the user wants separate folders for multiple uploads on same day, we add an index
    # But for now, we'll stick to the "same folder work" logic they requested
    os.makedirs(upload_folder, exist_ok=True)
    
    # Save filenames and paths
    saved_files = []

    # Save MES/Sensor files
    for mes_content, mes_filename in mes_files:
        if mes_content and mes_filename:
            # Preserve original filenames and keep all selected machine CSVs together.
            mes_path = upload_folder / mes_filename
            with open(mes_path, "wb") as f:
                f.write(mes_content)
            saved_files.append(mes_filename)

    # Save Hydra file
    if hydra_content and hydra_filename:
        hydra_path = upload_folder / hydra_filename
        with open(hydra_path, "wb") as f:
            f.write(hydra_content)
        saved_files.append(hydra_filename)

    logger.info(f"Saved {len(saved_files)} files to: {upload_folder.name}")
    return {
        "status": "success", 
        "message": f"Saved {len(saved_files)} file(s) to {upload_folder.name}",
        "folder": upload_folder.name,
        "files": saved_files
    }

def cleanup_uploaded_files(target_dir: Path):
    """Purge temporary files after successful ingestion to save space."""
    try:
        if target_dir.exists() and "uploads" in str(target_dir):
            import shutil
            shutil.rmtree(target_dir)
            logger.info(f"Purged temporary upload directory: {target_dir}")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

# Global progress tracker for the 10-step pipeline
PIPELINE_PROGRESS = {
    "status": "idle",
    "step": 0,
    "metrics": {},
    "validation_log": [],
    "error": None,
    "is_training": False,
    "training_msg": ""
}

def validate_input_files(target_dir: Path | None = None):
    """
    Performs 'Auto-Check' for file existence, naming, and schema.
    Returns (True, []) if successful, (False, errors) otherwise.
    """
    errors = []
    
    # Check 1: Existence
    target_dir = _resolve_target_dir(target_dir)
    xlsx_files = list(target_dir.glob("*.xlsx")) + list(target_dir.glob("*.xls"))
    csv_files = list(target_dir.glob("*.csv"))
    
    if not xlsx_files:
        errors.append("❌ Missing Hydra File: Please upload exactly one .xlsx or .xls file.")
    elif len(xlsx_files) > 1:
        errors.append(f"⚠️ Multiple Hydra Files found ({len(xlsx_files)}). The pipeline will only use: {xlsx_files[0].name}")

    if not csv_files:
        errors.append("❌ Missing Machine Data: No Sensor .csv files found in raw_data/.")

    # Check 2: Hydra Schema Check
    if xlsx_files:
        try:
            hydra_file = xlsx_files[0]
            # Peak just the header to save memory
            peek = pd.read_excel(hydra_file, nrows=0)
            date_col_found = any(c in peek.columns for c in ["machine_event_create_date", "timestamp", "date"])
            if not date_col_found:
                 errors.append(f"❌ Invalid Hydra Format: {hydra_file.name} is missing 'machine_event_create_date' column.")
        except Exception as e:
            errors.append(f"❌ Hydra Read Error: Could not parse {xlsx_files[0].name} ({str(e)})")

    # Check 3: CSV Schema & Naming
    for csv_path in csv_files:
        try:
            # Check Naming
            machine_id = normalize_machine_id(csv_path.name)
            if not machine_id or len(machine_id) < 2:
                errors.append(f"⚠️ Naming Warning: {csv_path.name} does not follow 'M###' pattern. Using 'AUTO' label.")
            
            # Check Schema
            peek = pd.read_csv(csv_path, nrows=0)
            required = {"variable_name", "value"} # 'timestamp' can sometimes be 'date' or inferred
            missing = required - set(peek.columns)
            if missing:
                errors.append(f"❌ Invalid CSV Schema: {csv_path.name} is missing columns: {list(missing)}")
        except Exception as e:
            errors.append(f"❌ CSV Read Error: Could not parse {csv_path.name} ({str(e)})")

    return len(errors) == 0, errors

def run_conversion_pipeline(target_dir: Path | None = None):
    """
    Triggers the real conversion and merge pipeline.
    In a production app, this would be a background task.
    """
    global PIPELINE_PROGRESS
    PIPELINE_PROGRESS["status"] = "processing"
    PIPELINE_PROGRESS["step"] = 1 # Step 1: Intake (already done by upload)
    
    try:
        # Step 1: Discovery
        target_dir = _resolve_target_dir(target_dir)
        logger.info(f"Processing target folder: {target_dir.name}")

        # Step 2: Auto-Check & Path setup
        logger.info("Running Auto-Check...")
        PIPELINE_PROGRESS["status"] = "validating"
        PIPELINE_PROGRESS["validation_log"] = [f"Checking folder: {target_dir.name}"]
        
        success, errors = validate_input_files(target_dir)
        PIPELINE_PROGRESS["validation_log"] = errors if errors else ["✅ All files passed Auto-Check."]
        
        if not success:
            critical_errors = [e for e in errors if "❌" in e]
            if critical_errors:
                PIPELINE_PROGRESS["status"] = "failed"
                PIPELINE_PROGRESS["error"] = critical_errors[0]
                return {"status": "error", "message": "Auto-Check failed", "details": errors}

        os.makedirs(PROCESSED_DIR, exist_ok=True)
        PIPELINE_PROGRESS["step"] = 2
        
        # Step 3: Pair (Identifying files - Senior Pro Data Sniffer)
        xlsx_files = list(target_dir.glob("*.xlsx")) + list(target_dir.glob("*.xls"))
        csv_files = list(target_dir.glob("*.csv"))
        
        # Smart Categorization
        hydra_files = []
        mse_files = []
        param_files = []
        
        for f in (xlsx_files + csv_files):
            f_type = _classify_file(f)
            if f_type == "Hydra": hydra_files.append(f)
            elif f_type == "MSE": mse_files.append(f)
            elif f_type == "Param": param_files.append(f)

        if not hydra_files:
            # Fallback for classification failure: use earliest XLSX
            if xlsx_files: hydra_files = [xlsx_files[0]]
            else: raise FileNotFoundError(f"Missing required Hydra data in {target_dir.name}")
            
        if not mse_files and not csv_files:
            raise FileNotFoundError(f"No Sensor data discovered in {target_dir.name}. Nothing to process.")
        
        if not mse_files: mse_files = csv_files

        hydra_file = hydra_files[0]
        
        # NEW: Process Param files if present (Sync thresholds)
        for p_file in param_files:
            try:
                _sync_dynamic_params(p_file)
            except Exception as e:
                logger.warning(f"Failed to sync params from {p_file.name}: {e}")
        PIPELINE_PROGRESS["step"] = 3
        
        # Step 4: Align (Processing Hydra)
        logger.info(f"Processing Hydra: {hydra_file.name}")
        try:
            # For Excel, we use a robust read then clean
            hydra_df = pd.read_excel(hydra_file)
        except Exception as e:
            raise ValueError(f"Failed to read Hydra file {hydra_file.name}: {str(e)}")

        if hydra_df.empty:
            raise ValueError(f"Hydra file {hydra_file.name} is empty.")

        # CLEAN HYDRA DATA: Many industrial files contain alphanumeric status codes in numeric columns
        if 'value' in hydra_df.columns:
            hydra_df['value'] = pd.to_numeric(hydra_df['value'], errors='coerce')
        
        date_col = "machine_event_create_date"
        if date_col not in hydra_df.columns:
            # Fallback to 'timestamp' if present (some files use 'timestamp')
            date_col = "timestamp" if "timestamp" in hydra_df.columns else hydra_df.columns[0]
            
            hydra_df[date_col] = _coerce_datetime(hydra_df[date_col], utc=True)
        
        # Filter out rows that failed date or value conversion
        hydra_df = hydra_df.dropna(subset=[date_col])
        
        hydra_out = PROCESSED_DIR / "HYDRA_TRAIN.parquet"
        hydra_df.to_parquet(hydra_out, index=False)
        logger.info(f"Hydra cleaned and saved to {hydra_out.name}")
        PIPELINE_PROGRESS["step"] = 4
        
        # Step 5: Aggregate (Processing Sensor Files)
        master_dfs = []
        for csv_file in csv_files:
            machine_id = normalize_machine_id(csv_file.name)
            logger.info(f"Processing Sensor: {csv_file.name}")
            
            try:
                # Use low_memory=False to avoid DtypeWarning on large industrial datasets
                df = pd.read_csv(csv_file, low_memory=False, on_bad_lines="skip")
            except Exception as e:
                logger.warning(f"Failed to read {csv_file.name}: {e}")
                continue

            if df.empty:
                continue
            
            if "timestamp" in df.columns:
                df["timestamp"] = _coerce_datetime(df["timestamp"], utc=True)
                df = df.dropna(subset=["timestamp"])
            
            # Step 6: Merge (Pivot & Align)
            # CRITICAL: Clean alphanumeric values (like '0A000') before pivoting
            # Force numeric, invalid becomes NaN, then drop NaNs
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df_numeric = df.dropna(subset=['value']).copy()
            
            if df_numeric.empty:
                logger.warning(f"No numeric sensor data found in {csv_file.name} (all rows were status codes or empty).")
                continue

            # Ensure value is float to prevent 'str to int' conversion errors in pivot_table
            df_numeric['value'] = df_numeric['value'].astype(float)
            
            pivot_df = df_numeric.pivot_table(
                index='timestamp',
                columns='variable_name',
                values='value',
                aggfunc='mean'
            ).reset_index()
            
            # Restore Machine ID for the master merge
            pivot_df['machine_id'] = machine_id
            master_dfs.append(pivot_df)
            
        if not master_dfs:
            raise ValueError("No valid numeric sensor data was discovered in the uploaded CSVs. Check file formats.")

        PIPELINE_PROGRESS["step"] = 6
        
        # Step 7: Features (Merging all)
        final_df = pd.concat(master_dfs, ignore_index=True)
        final_df = final_df.sort_values(by=["machine_id", "timestamp"]).reset_index(drop=True)
        
        master_path = PROCESSED_DIR / "FINAL_TRAINING_MASTER_V3.parquet"
        final_df.to_parquet(master_path, index=False)
        
        # VERY IMPORTANT: Split into individual machine files for the realtime dashboard!
        # Dashboard expects LONG format: (timestamp, variable_name, value, machine_definition)
        for m_id in final_df["machine_id"].unique():
            if pd.isna(m_id):
                continue
            machine_norm = normalize_machine_id(str(m_id))
            if machine_norm:
                m_df = final_df[final_df["machine_id"] == m_id].copy()
                sensor_cols = [c for c in m_df.columns if c not in ("timestamp", "machine_id")]
                long_df = m_df.melt(
                    id_vars=["timestamp", "machine_id"],
                    value_vars=sensor_cols,
                    var_name="variable_name",
                    value_name="value",
                )
                long_df["machine_definition"] = machine_norm
                long_df = long_df.dropna(subset=["value"])
                machine_path = PROCESSED_DIR / f"{machine_norm}_TEST.parquet"
                long_df.to_parquet(machine_path, index=False)
                logger.info(f"Generated Live Feed Parquet for: {machine_norm} ({len(long_df)} rows)")
                
        PIPELINE_PROGRESS["step"] = 7
        
        # Step 8: Compute real merge quality metrics
        # matchRate = % of sensor rows that had a valid (non-null) timestamp after merge
        valid_ts_count = final_df["timestamp"].notna().sum() if "timestamp" in final_df.columns else len(final_df)
        total_rows = max(len(final_df), 1)
        match_rate = round(float(valid_ts_count) / total_rows * 100.0, 1)

        # alignmentQuality = % of rows that have at least 50% non-null numeric columns
        numeric_cols_in_final = final_df.select_dtypes(include="number").columns.tolist()
        if numeric_cols_in_final:
            threshold_cols = max(1, len(numeric_cols_in_final) // 2)
            aligned_rows = (final_df[numeric_cols_in_final].notna().sum(axis=1) >= threshold_cols).sum()
            alignment_quality = round(float(aligned_rows) / total_rows, 4)
        else:
            alignment_quality = 0.0

        # Step 9-10: Finalize
        PIPELINE_PROGRESS["step"] = 10
        PIPELINE_PROGRESS["status"] = "complete"

        metrics = {
            "mesRows": len(final_df),
            "hydraRows": len(hydra_df),
            "matchRate": match_rate,
            "alignmentQuality": alignment_quality,
            "featuresCount": len(final_df.columns)
        }
        PIPELINE_PROGRESS["metrics"] = metrics
        # FINAL STEP: Trigger Automated Model Retraining in background
        def _run_background_training():
            try:
                PIPELINE_PROGRESS["is_training"] = True
                PIPELINE_PROGRESS["training_msg"] = "Retraining AI Model..."
                logger.info("Starting automated model retraining...")
                
                script_path = PROJECT_ROOT / "scripts" / "train_forecaster_v3.py"
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT)
                )
                
                if result.returncode == 0:
                    logger.info("✓ Automated retraining successful.")
                    PIPELINE_PROGRESS["training_msg"] = "AI Sync Complete"
                    # CLEANUP: Purge uploads if successful
                    cleanup_uploaded_files(target_dir)
                else:
                    logger.error(f"✗ Retraining failed: {result.stderr}")
                    PIPELINE_PROGRESS["training_msg"] = "AI Sync Failed"
            except Exception as tr_err:
                logger.error(f"Retraining process error: {tr_err}")
                PIPELINE_PROGRESS["training_msg"] = "AI Sync Error"
            finally:
                PIPELINE_PROGRESS["is_training"] = False

        threading.Thread(target=_run_background_training, daemon=True).start()
        
        return {"status": "success", "metrics": metrics}

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        PIPELINE_PROGRESS["status"] = "failed"
        return {"status": "error", "message": str(e)}

def get_pipeline_status():
    """Returns the current progress of the ingestion pipeline."""
    return PIPELINE_PROGRESS
