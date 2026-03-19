"""
build_and_train_v1_2b.py
========================
VERSION: v1.2c-tuned
CHANGES FROM v1.2b:
  - learning_rate: 0.05 → 0.02  (was stopping at round 1 due to overfitting)
  - min_child_samples: 20 → 50  (more regularization on 256k rows)
  - num_boost_round: 500 → 1000 (more rounds needed with lower lr)
  - early_stopping: 50 → 30 rounds
  - MAX_TRAIN_ROWS: 500k → 256448 (use ALL data, no downsampling needed)
  - Added Machine_status to EXCLUDE_PREFIXES (metadata, not process physics)
"""

import gc
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
MASTER_FILE    = PROJECT_ROOT / "new_processed_data" / "FINAL_TRAINING_MASTER.parquet"
CONTEXT_DIR    = PROJECT_ROOT / "new_processed_data"
WIDE_OUT       = PROJECT_ROOT / "processed" / "features" / "rolling_features_with_context.parquet"
MODEL_PATH     = PROJECT_ROOT / "models" / "lightgbm_scrap_risk_wide.pkl"
METRICS_PATH   = PROJECT_ROOT / "models" / "training_metrics_v1_2c.json"
ENCODINGS_FILE = PROJECT_ROOT / "models" / "part_tool_encodings.json"
COLS_FILE      = PROJECT_ROOT / "processed" / "features" / "rolling_feature_columns.txt"

VERSION = "v1.2c-tuned"
WINDOWS = ["5m", "15m", "30m"]
STATS   = ["mean", "std", "min", "max", "last"]
PATTERN_WINDOW = 10
PATTERN_SENSORS_CANDIDATES = [
    "Injection_pressure", "Cycle_time", "Peak_pressure_time", "Switch_pressure",
    "Cushion", "Holding_pressure",
    "Cyl_tmp_z1", "Cyl_tmp_z2", "Cyl_tmp_z3",
    "Cyl_tmp_z4", "Cyl_tmp_z5",
    "Dosage_time", "Injection_time",
]
EXCLUDE_PREFIXES = [
    "Scrap_counter", "Shot_counter", "Time_on_machine",
    "Machine_status",   # v1.2c: metadata, not process physics
]
EXCLUDE_EXACT = [
    "is_scrap", "is_scrap_actual", "early_scrap_risk", "scrap_probability",
    "timestamp", "event_timestamp", "Time", "time",
    "machine_id", "machine_id_normalized", "machine_definition",
    "part_number", "tool_number", "tool_id",
]
MAX_TRAIN_ROWS = 300_000   # v1.2c: increased, effectively uses all 256k rows


def load_master():
    print(f"\n{'='*60}\nSTEP 1 — Loading {MASTER_FILE.name}\n{'='*60}")
    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"FATAL: {MASTER_FILE} not found.\nRun scripts/02_merge_train_data.py first.")
    df = pd.read_parquet(MASTER_FILE, engine="pyarrow")
    print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")
    label_col = None
    for c in ["is_scrap", "early_scrap_risk", "is_scrap_actual"]:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError("No label column found in FINAL_TRAINING_MASTER.parquet")
    scrap_n   = int(df[label_col].sum())
    scrap_pct = scrap_n / len(df) * 100
    print(f"Label column: {label_col}\nScrap rows: {scrap_n:,} ({scrap_pct:.3f}%)")
    if scrap_pct > 10:
        print(f"WARNING: Scrap rate {scrap_pct:.1f}% is high.")
    elif scrap_pct < 0.1:
        print(f"WARNING: Scrap rate very low — check Hydra alignment.")
    else:
        print("Scrap rate looks realistic.")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df, label_col


def build_rolling_features(df, label_col):
    print(f"\n{'='*60}\nSTEP 2 — Building rolling window features\n{'='*60}")
    meta_cols = {"timestamp", "machine_id", "machine_id_normalized",
                 "machine_definition", label_col, "is_scrap",
                 "is_scrap_actual", "early_scrap_risk", "scrap_probability"}
    sensor_cols = [
        c for c in df.columns
        if c not in meta_cols
        and not any(c.startswith(p) for p in EXCLUDE_PREFIXES)
        and df[c].dtype in [np.float32, np.float64, np.int32, np.int64]
    ]
    print(f"Sensor columns to roll: {len(sensor_cols)}")
    print(f"Machines: {sorted(df['machine_id'].unique()) if 'machine_id' in df.columns else 'unknown'}")
    machine_col = "machine_id" if "machine_id" in df.columns else None
    cycle_map = {"5m": 30, "15m": 90, "30m": 180}
    rolling_dfs = []
    groups = df.groupby(machine_col) if machine_col else [("ALL", df)]
    for machine, grp in groups:
        print(f"  Rolling features for machine {machine} ({len(grp):,} rows)...")
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        keep = [machine_col, "timestamp", label_col] if machine_col else ["timestamp", label_col]
        result = grp[keep].copy()
        for sensor in sensor_cols:
            s = grp[sensor]
            for w in WINDOWS:
                n      = cycle_map[w]
                rolled = s.rolling(n, min_periods=max(3, n // 10))
                for stat in STATS:
                    col_name = f"{sensor}__{stat}_{w}"
                    if stat == "mean":   result[col_name] = rolled.mean()
                    elif stat == "std":  result[col_name] = rolled.std()
                    elif stat == "min":  result[col_name] = rolled.min()
                    elif stat == "max":  result[col_name] = rolled.max()
                    elif stat == "last": result[col_name] = s
        rolling_dfs.append(result)
        del grp, result
        gc.collect()
    combined = pd.concat(rolling_dfs, ignore_index=True)
    print(f"Rolling features built: {len(combined.columns)} columns total")
    return combined


def add_pattern_features(df):
    print(f"\n{'='*60}\nSTEP 3 — Adding pattern-aware features (fixed step4b)\n{'='*60}")
    from scipy.stats import linregress
    def compute_slope(series):
        clean = series.dropna()
        if len(clean) < 3: return 0.0
        x = np.arange(len(clean))
        try:
            slope, *_ = linregress(x, clean.values)
            return float(slope) if np.isfinite(slope) else 0.0
        except: return 0.0
    machine_col = "machine_id" if "machine_id" in df.columns else None
    available_sensors = []
    for sensor in PATTERN_SENSORS_CANDIDATES:
        if sensor in df.columns:
            available_sensors.append((sensor, sensor))
        else:
            mean_col = f"{sensor}__mean_15m"
            if mean_col in df.columns:
                available_sensors.append((sensor, mean_col))
    print(f"Pattern sensors found: {len(available_sensors)} / {len(PATTERN_SENSORS_CANDIDATES)}")
    if not available_sensors:
        print("WARNING: No pattern sensors matched. Skipping.")
        return df
    for sensor_name, source_col in available_sensors:
        print(f"  Pattern features for {sensor_name} (from {source_col})...")
        if machine_col:
            df[f"{sensor_name}__pstd_10"]   = df.groupby(machine_col)[source_col].transform(lambda x: x.rolling(PATTERN_WINDOW, min_periods=3).std())
            df[f"{sensor_name}__pmean_10"]  = df.groupby(machine_col)[source_col].transform(lambda x: x.rolling(PATTERN_WINDOW, min_periods=3).mean())
            df[f"{sensor_name}__ptrend_10"] = df.groupby(machine_col)[source_col].transform(lambda x: x.rolling(PATTERN_WINDOW, min_periods=3).apply(compute_slope, raw=False))
        else:
            df[f"{sensor_name}__pstd_10"]   = df[source_col].rolling(PATTERN_WINDOW, min_periods=3).std()
            df[f"{sensor_name}__pmean_10"]  = df[source_col].rolling(PATTERN_WINDOW, min_periods=3).mean()
            df[f"{sensor_name}__ptrend_10"] = df[source_col].rolling(PATTERN_WINDOW, min_periods=3).apply(compute_slope, raw=False)
        df[f"{sensor_name}__pdev_10"] = (df[source_col] - df[f"{sensor_name}__pmean_10"]) / (df[f"{sensor_name}__pmean_10"].abs() + 1e-6)
    pattern_cols = [c for c in df.columns if any(x in c for x in ["__pstd_", "__pmean_", "__ptrend_", "__pdev_"])]
    df[pattern_cols] = df[pattern_cols].fillna(0.0)
    print(f"Added {len(pattern_cols)} pattern-aware features")
    return df


def add_context_features(df):
    print(f"\n{'='*60}\nSTEP 4 — Adding part/tool context encodings (fixed step4c)\n{'='*60}")
    import re
    def extract_tool(machine_def):
        if pd.isna(machine_def) or not machine_def: return "UNKNOWN"
        match = re.search(r'-([A-Za-z0-9]+)$', str(machine_def))
        return match.group(1) if match else str(machine_def)
    context_records = []
    machine_files = list(CONTEXT_DIR.glob("M*_TEST.parquet"))
    print(f"Found {len(machine_files)} machine TEST files")
    for mf in machine_files:
        machine_id = mf.stem.replace("_TEST", "")
        try:
            sample = pd.read_parquet(mf, engine="pyarrow")
            avail  = sample.columns.tolist()
            machine_def = "UNKNOWN"; tool_id = "UNKNOWN"; part_number = "UNKNOWN"
            if "machine_definition" in avail:
                defs = sample["machine_definition"].dropna().unique()
                if len(defs) > 0:
                    machine_def = str(defs[0])
                    tool_id = extract_tool(machine_def)
            for col in [c for c in avail if "part" in c.lower()]:
                vals = sample[col].dropna().unique()
                if len(vals) > 0: part_number = str(vals[0]); break
            for col in [c for c in avail if "tool" in c.lower()]:
                vals = sample[col].dropna().unique()
                if len(vals) > 0: tool_id = str(vals[0]); break
            context_records.append({"machine_id": machine_id, "tool_id": tool_id, "part_number": part_number})
            print(f"  {machine_id}: tool={tool_id}, part={part_number}")
        except Exception as e:
            print(f"  WARN: {mf.name}: {e}")
    if not context_records:
        print("WARNING: No context found. Using defaults.")
        df["tool_id_encoded"] = 0; df["part_number_encoded"] = 0; df["machine_id_encoded"] = 0
        return df
    context_df = pd.DataFrame(context_records)
    encodings = {
        "tool_id":     {t: i for i, t in enumerate(sorted(context_df["tool_id"].unique()))},
        "part_number": {p: i for i, p in enumerate(sorted(context_df["part_number"].unique()))},
        "machine_id":  {m: i for i, m in enumerate(sorted(context_df["machine_id"].unique()))},
    }
    ENCODINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ENCODINGS_FILE, "w") as f:
        json.dump(encodings, f, indent=2)
    print(f"Saved encodings: {ENCODINGS_FILE.name}")
    machine_col = "machine_id" if "machine_id" in df.columns else None
    if machine_col is None:
        df["tool_id_encoded"] = 0; df["part_number_encoded"] = 0; df["machine_id_encoded"] = 0
        return df
    df["_mnorm"] = df[machine_col].astype(str).str.upper().str.replace("-", "").str.strip()
    context_df["_mnorm"] = context_df["machine_id"].astype(str).str.upper().str.replace("-", "").str.strip()
    df = df.merge(context_df[["_mnorm", "tool_id", "part_number"]], on="_mnorm", how="left")
    df["tool_id"]     = df["tool_id"].fillna("UNKNOWN")
    df["part_number"] = df["part_number"].fillna("UNKNOWN")
    df["tool_id_encoded"]     = df["tool_id"].map(encodings["tool_id"]).fillna(0).astype(int)
    df["part_number_encoded"] = df["part_number"].map(encodings["part_number"]).fillna(0).astype(int)
    df["machine_id_encoded"]  = df["_mnorm"].map({k.upper().replace("-",""): v for k, v in encodings["machine_id"].items()}).fillna(0).astype(int)
    df = df.drop(columns=["_mnorm", "tool_id", "part_number"], errors="ignore")
    print(f"Added: tool_id_encoded ({df['tool_id_encoded'].nunique()} unique), part_number_encoded ({df['part_number_encoded'].nunique()} unique), machine_id_encoded")
    return df


def save_wide_file(df):
    print(f"\n{'='*60}\nSTEP 5 — Saving {WIDE_OUT.name}\n{'='*60}")
    WIDE_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(WIDE_OUT, engine="pyarrow", index=False)
    print(f"Saved: {len(df):,} rows, {len(df.columns)} columns → {WIDE_OUT}")
    feature_cols = [c for c in df.columns if c not in
                    {"timestamp","machine_id","machine_id_normalized","machine_definition",
                     "is_scrap","is_scrap_actual","early_scrap_risk","scrap_probability"}]
    COLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COLS_FILE, "w") as f:
        f.write("\n".join(feature_cols))
    print(f"Saved column list: {COLS_FILE.name} ({len(feature_cols)} features)")


def is_excluded(col_name):
    if col_name in EXCLUDE_EXACT: return True
    for prefix in EXCLUDE_PREFIXES:
        if col_name == prefix or col_name.startswith(prefix + "__"): return True
    return False


def train_model(df, label_col):
    print(f"\n{'='*60}\nSTEP 6 — Training LightGBM ({VERSION})\n{'='*60}")
    import lightgbm as lgb
    from sklearn.metrics import (classification_report, f1_score, precision_score,
                                  recall_score, roc_auc_score, precision_recall_curve)
    from sklearn.model_selection import train_test_split
    feature_columns = [c for c in df.columns
                       if not is_excluded(c) and c != label_col
                       and df[c].dtype in [np.float32, np.float64, np.int64, np.int32]]
    removed = [c for c in df.columns if is_excluded(c) and c != label_col
               and df[c].dtype in [np.float32, np.float64, np.int64, np.int32]]
    print(f"Features selected: {len(feature_columns)}")
    print(f"Leakage/noise excluded: {len(removed)}")
    if removed:
        print(f"  Excluded: {removed[:5]}{'...' if len(removed)>5 else ''}")

    # v1.2c: use all data, no downsampling needed at 256k rows
    X = df[feature_columns].fillna(0).values.astype(np.float32)
    y = df[label_col].fillna(0).astype(int).values
    del df; gc.collect()

    n_scrap  = int((y == 1).sum())
    n_normal = int((y == 0).sum())
    print(f"X shape: {X.shape}\nScrap: {n_scrap:,} ({100*n_scrap/len(y):.2f}%) | Normal: {n_normal:,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    raw_ratio        = n_normal / max(n_scrap, 1)
    scale_pos_weight = min(raw_ratio, 50)
    print(f"Raw imbalance: {raw_ratio:.1f}x → scale_pos_weight={scale_pos_weight:.1f} (capped at 50)")

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_columns)
    valid_data = lgb.Dataset(X_test,  label=y_test,  feature_name=feature_columns,
                              reference=train_data)

    params = {
        "objective":         "binary",
        "metric":            ["auc", "binary_logloss"],
        "boosting_type":     "gbdt",
        "num_leaves":        64,
        "learning_rate":     0.02,   # v1.2c: lowered from 0.05
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "verbose":           -1,
        "scale_pos_weight":  scale_pos_weight,
        "min_child_samples": 50,     # v1.2c: raised from 20
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "n_jobs":            -1,
    }

    model = lgb.train(
        params, train_data,
        num_boost_round=1000,        # v1.2c: raised from 500
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),  # v1.2c: tightened from 50
            lgb.log_evaluation(period=50),
        ],
    )

    y_proba  = model.predict(X_test)
    y_pred05 = (y_proba >= 0.5).astype(int)
    auc      = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*60}\nRESULTS AT THRESHOLD 0.5\n{'='*60}")
    print(classification_report(y_test, y_pred05, target_names=["Normal","Scrap"], digits=4))
    print(f"AUC: {auc:.4f}")

    print(f"\n{'='*60}\nTHRESHOLD SEARCH\n{'='*60}")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    pa, ra, ta = precision_recall_curve(y_test, y_proba)
    best_f1, best_t = 0, 0.5
    last_bucket = 0
    for p, r, t in zip(pa, ra, ta):
        f1 = (2*p*r/(p+r)) if (p+r) > 0 else 0
        if f1 > best_f1: best_f1, best_t = f1, t
        bucket = int(p * 20) * 5
        if bucket >= 10 and bucket > last_bucket:
            print(f"{round(t,3):>10} {round(p*100,1):>9}% {round(r*100,1):>7}% {round(f1,3):>8}")
            last_bucket = bucket
        if p >= 0.85: break
    print(f"\nBest F1={best_f1:.3f} at threshold={best_t:.3f}")
    print(f"→ Update ML_THRESHOLDS in config_limits.py:")
    print(f"  'LOW':    0.10")
    print(f"  'MEDIUM': {round(best_t * 0.6, 2)}")
    print(f"  'HIGH':   {best_t:.2f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    importance_df = pd.DataFrame({
        "feature":    feature_columns,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(PROJECT_ROOT / "models" / "feature_importance.csv", index=False)
    print(f"\nTop 15 features:")
    print(importance_df.head(15).to_string(index=False))

    metrics = {
        "version": VERSION,
        "accuracy":  float((y_pred05 == y_test).mean()),
        "f1_score":  float(f1_score(y_test, y_pred05, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred05, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred05, zero_division=0)),
        "auc":       float(auc),
        "best_f1_threshold":    float(best_t),
        "best_f1_score":        float(best_f1),
        "scale_pos_weight_raw": float(raw_ratio),
        "scale_pos_weight_used": float(scale_pos_weight),
        "n_features":      len(feature_columns),
        "n_train_samples": len(y_train),
        "n_test_samples":  len(y_test),
        "n_train_scrap":   int((y_train == 1).sum()),
        "n_test_scrap":    int((y_test == 1).sum()),
        "note": "v1.2c: lr=0.02, min_child=50, 1000 rounds, Machine_status excluded, no downsampling."
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {METRICS_PATH}")
    return metrics


def main():
    print(f"\n{'='*60}\nBUILD & TRAIN PIPELINE — {VERSION}\n{'='*60}")
    print(f"Project root: {PROJECT_ROOT}")
    df, label_col = load_master()
    df = build_rolling_features(df, label_col); gc.collect()
    df = add_pattern_features(df); gc.collect()
    df = add_context_features(df); gc.collect()
    save_wide_file(df)
    metrics = train_model(df, label_col); gc.collect()
    print(f"\n{'='*60}\nPIPELINE COMPLETE — {VERSION}\n{'='*60}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1_score']:.4f}")
    print(f"\n  Update config_limits.py ML_THRESHOLDS to:")
    print(f"    LOW:    0.10")
    print(f"    MEDIUM: {round(metrics['best_f1_threshold']*0.6, 2)}")
    print(f"    HIGH:   {metrics['best_f1_threshold']:.2f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()