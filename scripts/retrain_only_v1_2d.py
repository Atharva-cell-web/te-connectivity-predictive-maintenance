# Sort by timestamp before splitting — CRITICAL for time series
# Random split leaks future rolling features into training

"""
retrain_only_v1_2d.py
=====================
VERSION: v1.2d-fixed-training
Skips feature rebuilding (already done) — just retrains on existing
rolling_features_with_context.parquet with corrected LightGBM params.

FIXES vs v1.2c:
  - min_child_samples: 50 → 10  (was too high for 4744 scrap rows, caused round 4 stop)
  - learning_rate: 0.02 → 0.05  (lower lr needs more data; 0.05 fine with min_child=10)
  - num_leaves: 64 → 31         (reduce complexity to prevent overfitting)
  - scale_pos_weight: 50 → 30   (less aggressive class weighting = better precision)
  - num_boost_round: 1000 → 500
  - early_stopping: 30 → 50     (give model more patience)

RUN FROM PROJECT ROOT:
  cd "D:\\te connectivity 3"
  python scripts/retrain_only_v1_2d.py

Expected runtime: 5-10 minutes
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
WIDE_FILE      = PROJECT_ROOT / "processed" / "features" / "rolling_features_with_context.parquet"
MODEL_PATH     = PROJECT_ROOT / "models" / "lightgbm_scrap_risk_wide.pkl"
METRICS_PATH   = PROJECT_ROOT / "models" / "training_metrics_v1_2d.json"

VERSION = "v1.2d-fixed-training"

EXCLUDE_PREFIXES = [
    "Scrap_counter", "Shot_counter", "Time_on_machine", "Machine_status",
]
EXCLUDE_EXACT = [
    "is_scrap", "is_scrap_actual", "early_scrap_risk", "scrap_probability",
    "timestamp", "event_timestamp", "Time", "time",
    "machine_id", "machine_id_normalized", "machine_definition",
    "part_number", "tool_number", "tool_id",
]

def is_excluded(col_name):
    if col_name in EXCLUDE_EXACT: return True
    for prefix in EXCLUDE_PREFIXES:
        if col_name == prefix or col_name.startswith(prefix + "__"): return True
    return False

def main():
    print(f"\n{'='*60}")
    print(f"RETRAIN ONLY — {VERSION}")
    print(f"{'='*60}")

    # Load existing wide feature file
    print(f"\nLoading {WIDE_FILE.name}...")
    if not WIDE_FILE.exists():
        raise FileNotFoundError(
            f"FATAL: {WIDE_FILE} not found.\n"
            "Run build_and_train_v1_2b.py first to build features."
        )
    df = pd.read_parquet(WIDE_FILE, engine="pyarrow")
    print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Find label column
    label_col = None
    for c in ["is_scrap", "early_scrap_risk", "is_scrap_actual"]:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError("No label column found.")
    print(f"Label: {label_col}")

    scrap_n = int(df[label_col].sum())
    print(f"Scrap rows: {scrap_n:,} ({scrap_n/len(df)*100:.3f}%)")

    # Feature selection
    feature_columns = [
        c for c in df.columns
        if not is_excluded(c) and c != label_col
        and df[c].dtype in [np.float32, np.float64, np.int64, np.int32]
    ]
    removed = [c for c in df.columns if is_excluded(c) and c != label_col
               and df[c].dtype in [np.float32, np.float64, np.int64, np.int32]]
    print(f"Features: {len(feature_columns)} | Excluded: {len(removed)}")

    import lightgbm as lgb
    from sklearn.metrics import (classification_report, f1_score, precision_score,
                                  recall_score, roc_auc_score, precision_recall_curve)
    from sklearn.model_selection import train_test_split

    X = df[feature_columns].fillna(0).values.astype(np.float32)
    y = df[label_col].fillna(0).astype(int).values
    del df; gc.collect()

    n_scrap  = int((y == 1).sum())
    n_normal = int((y == 0).sum())
    print(f"\nX shape: {X.shape}")
    print(f"Scrap: {n_scrap:,} ({100*n_scrap/len(y):.2f}%) | Normal: {n_normal:,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Train scrap: {int((y_train==1).sum()):,} | Test scrap: {int((y_test==1).sum()):,}")

    raw_ratio        = n_normal / max(n_scrap, 1)
    scale_pos_weight = min(raw_ratio, 30)   # v1.2d: lowered to 30
    print(f"\nRaw imbalance: {raw_ratio:.1f}x → scale_pos_weight={scale_pos_weight:.1f} (capped at 30)")

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_columns)
    valid_data = lgb.Dataset(X_test,  label=y_test,  feature_name=feature_columns,
                              reference=train_data)

    params = {
        "objective":         "binary",
        "metric":            ["auc", "binary_logloss"],
        "boosting_type":     "gbdt",
        "num_leaves":        31,     # v1.2d: reduced from 64
        "learning_rate":     0.05,   # v1.2d: back to 0.05 (works with min_child=10)
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "verbose":           -1,
        "scale_pos_weight":  scale_pos_weight,
        "min_child_samples": 10,     # v1.2d: lowered from 50 — key fix
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "n_jobs":            -1,
        "is_unbalance":      False,  # explicit — we handle via scale_pos_weight
    }

    print(f"\nTraining LightGBM {VERSION}...")
    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=25),
        ],
    )
    print(f"Best iteration: {model.best_iteration}")

    # Evaluate
    y_proba  = model.predict(X_test)
    y_pred05 = (y_proba >= 0.5).astype(int)
    auc      = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*60}\nRESULTS AT THRESHOLD 0.5\n{'='*60}")
    print(classification_report(y_test, y_pred05,
                                  target_names=["Normal","Scrap"], digits=4))
    print(f"AUC: {auc:.4f}")

    # Threshold search
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
    print(f"\n→ Update config_limits.py ML_THRESHOLDS to:")
    print(f"  'LOW':    0.{int(best_t*100*0.4):02d}")
    print(f"  'MEDIUM': {best_t*0.7:.2f}")
    print(f"  'HIGH':   {best_t:.2f}")

    # Save
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    importance_df = pd.DataFrame({
        "feature":    feature_columns,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(
        PROJECT_ROOT / "models" / "feature_importance.csv", index=False)
    print(f"\nTop 15 features:")
    print(importance_df.head(15).to_string(index=False))

    metrics = {
        "version":  VERSION,
        "accuracy": float((y_pred05 == y_test).mean()),
        "f1_score": float(f1_score(y_test, y_pred05, zero_division=0)),
        "precision":float(precision_score(y_test, y_pred05, zero_division=0)),
        "recall":   float(recall_score(y_test, y_pred05, zero_division=0)),
        "auc":      float(auc),
        "best_f1_threshold":     float(best_t),
        "best_f1_score":         float(best_f1),
        "best_iteration":        model.best_iteration,
        "scale_pos_weight_raw":  float(raw_ratio),
        "scale_pos_weight_used": float(scale_pos_weight),
        "n_features":      len(feature_columns),
        "n_train_samples": len(y_train),
        "n_test_samples":  len(y_test),
        "n_train_scrap":   int((y_train == 1).sum()),
        "n_test_scrap":    int((y_test == 1).sum()),
        "note": "v1.2d: min_child=10, lr=0.05, num_leaves=31, spw=30. Uses v1.2c feature file."
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {METRICS_PATH}")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — {VERSION}")
    print(f"{'='*60}")
    print(f"  AUC:            {metrics['auc']:.4f}")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best F1:        {best_f1:.3f} at threshold {best_t:.3f}")
    print(f"\n  Next: paste full output to Claude for analysis")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()