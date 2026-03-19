"""
retrain_only_v1_2e.py
=====================
VERSION: v1.2e-temporal-split
Fixes the "best iteration = 1" bug caused by random train/test split
on time-series data with rolling features.

ROOT CAUSE: Rolling features (mean_15m, std_30m) are computed on sorted
time windows. Random splitting puts future rows in train and past rows in
test — the test labels become inconsistent with the features the model
learned, so binary_logloss gets WORSE after round 1.

FIX: Sort by timestamp, use last 20% of time as test set (temporal split).
This is the correct evaluation strategy for any time-series ML model.

CHANGES vs v1.2d:
  - Temporal train/test split instead of random split
  - Data sorted by timestamp before split
  - scale_pos_weight back to 50 (30 was too low with temporal split)
  - min_child_samples: 10 (kept from v1.2d — this was correct)
  - num_leaves: 63
  - learning_rate: 0.05

RUN FROM PROJECT ROOT:
  cd "D:\\te connectivity 3"
  python scripts/retrain_only_v1_2e.py

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
METRICS_PATH   = PROJECT_ROOT / "models" / "training_metrics_v1_2e.json"

VERSION = "v1.2e-temporal-split"

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

    # Load wide feature file
    print(f"\nLoading {WIDE_FILE.name}...")
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

    scrap_n = int(df[label_col].sum())
    print(f"Label: {label_col} | Scrap: {scrap_n:,} ({scrap_n/len(df)*100:.3f}%)")

    # ── CRITICAL FIX: Sort by timestamp before splitting ──────────────────
    # Rolling features are computed in time order. Random split leaks future
    # data into training — test labels become inconsistent → best iteration=1.
    print(f"\nSorting by timestamp (temporal split fix)...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # Temporal split: train = first 80% of TIME, test = last 20%
    split_idx = int(len(df) * 0.80)
    split_time = df["timestamp"].iloc[split_idx]
    print(f"Split point: {split_time} (row {split_idx:,})")

    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()

    print(f"Train: {len(df_train):,} rows | scrap: {int(df_train[label_col].sum()):,} "
          f"({df_train[label_col].mean()*100:.2f}%)")
    print(f"Test:  {len(df_test):,} rows  | scrap: {int(df_test[label_col].sum()):,} "
          f"({df_test[label_col].mean()*100:.2f}%)")
    del df; gc.collect()

    # Feature selection
    feature_columns = [
        c for c in df_train.columns
        if not is_excluded(c) and c != label_col
        and df_train[c].dtype in [np.float32, np.float64, np.int64, np.int32]
    ]
    removed = [c for c in df_train.columns
               if is_excluded(c) and c != label_col
               and df_train[c].dtype in [np.float32, np.float64, np.int64, np.int32]]
    print(f"\nFeatures: {len(feature_columns)} | Excluded: {len(removed)}")
    if removed:
        print(f"Excluded sample: {removed[:3]}{'...' if len(removed)>3 else ''}")

    X_train = df_train[feature_columns].fillna(0).values.astype(np.float32)
    y_train = df_train[label_col].fillna(0).astype(int).values
    X_test  = df_test[feature_columns].fillna(0).values.astype(np.float32)
    y_test  = df_test[label_col].fillna(0).astype(int).values
    del df_train, df_test; gc.collect()

    n_scrap_train  = int((y_train == 1).sum())
    n_normal_train = int((y_train == 0).sum())
    print(f"\nX_train: {X_train.shape} | X_test: {X_test.shape}")

    raw_ratio        = n_normal_train / max(n_scrap_train, 1)
    scale_pos_weight = min(raw_ratio, 50)
    print(f"Imbalance: {raw_ratio:.1f}x → scale_pos_weight={scale_pos_weight:.1f}")

    import lightgbm as lgb
    from sklearn.metrics import (classification_report, f1_score, precision_score,
                                  recall_score, roc_auc_score, precision_recall_curve)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_columns)
    valid_data = lgb.Dataset(X_test,  label=y_test,  feature_name=feature_columns,
                              reference=train_data)

    params = {
        "objective":         "binary",
        "metric":            ["auc", "binary_logloss"],
        "boosting_type":     "gbdt",
        "num_leaves":        63,
        "learning_rate":     0.05,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "verbose":           -1,
        "scale_pos_weight":  scale_pos_weight,
        "min_child_samples": 10,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "n_jobs":            -1,
    }

    print(f"\nTraining {VERSION}...")
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
    print(f"\nBest iteration: {model.best_iteration}")

    # Evaluate
    y_proba  = model.predict(X_test)
    y_pred05 = (y_proba >= 0.5).astype(int)
    auc      = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*60}\nRESULTS AT THRESHOLD 0.5\n{'='*60}")
    print(classification_report(y_test, y_pred05,
                                  target_names=["Normal","Scrap"], digits=4))
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
    print(f"\n→ Update config_limits.py ML_THRESHOLDS to:")
    low_t   = round(best_t * 0.4, 2)
    med_t   = round(best_t * 0.7, 2)
    print(f"  'LOW':    {low_t}")
    print(f"  'MEDIUM': {med_t}")
    print(f"  'HIGH':   {best_t:.2f}")

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    # Feature importance
    importance_df = pd.DataFrame({
        "feature":    feature_columns,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(
        PROJECT_ROOT / "models" / "feature_importance.csv", index=False)
    print(f"\nTop 15 features:")
    print(importance_df.head(15).to_string(index=False))

    # Save metrics
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
        "note": (
            "v1.2e: Temporal train/test split (train=first 80% of time, "
            "test=last 20%). Fixes random-split leakage of rolling features. "
            "min_child=10, lr=0.05, num_leaves=63, spw=50."
        )
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {METRICS_PATH}")

    print(f"\n{'='*60}")
    print(f"COMPLETE — {VERSION}")
    print(f"{'='*60}")
    print(f"  AUC:            {metrics['auc']:.4f}")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best F1:        {best_f1:.3f} at threshold {best_t:.3f}")
    print(f"  Precision@best: see threshold table above")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()