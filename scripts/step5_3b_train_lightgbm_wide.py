"""
Step 5.3b: Train LightGBM scrap classifier with class balancing.
Memory-efficient version for large datasets.

VERSION: v1.1-remove-leakage
CHANGES FROM PREVIOUS VERSION:
  - Added Scrap_counter to EXCLUDE_COLS (was data leakage — machine counter
    increments AFTER scrap happens, not before. Ranked #4 feature previously,
    inflating AUC and hurting real predictive precision.)
  - Added Shot_counter to EXCLUDE_COLS (correlates with machine runtime/shift
    position, not with process physics. Not a causal predictor of scrap.)
  - Added Time_on_machine to EXCLUDE_COLS (parameter info file explicitly
    states "not process related")
  - Saved model to lightgbm_scrap_risk_wide.pkl (not _v2) to update the
    active production model used by data_access.py
  - Metrics saved to training_metrics_v1_1.json (preserves history)
  - Added version tag to metrics output for traceability
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

def get_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WIDE_FILE = PROJECT_ROOT / "processed" / "features" / "rolling_features_with_context.parquet"

# v1.1: Save to the ACTIVE model path (not _v2) so dashboard picks it up
MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_scrap_risk_wide.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "training_metrics_v1_1.json"

VERSION = "v1.1-remove-leakage"

# Pattern feature suffixes
PATTERN_SUFFIXES = ['__std_5m', '__std_15m', '__std_30m', '__mean_5m', '__mean_15m', '__mean_30m']

# ── EXCLUDE_COLS — v1.1 changes marked with comments ──
EXCLUDE_COLS = [
    # Labels and targets — never use as features
    'is_scrap', 'is_scrap_actual', 'early_scrap_risk', 'scrap_probability',

    # Timestamps and IDs — not process signals
    'timestamp', 'event_timestamp', 'Time', 'time',
    'machine_id', 'machine_id_normalized', 'machine_definition',
    'part_number', 'tool_number', 'tool_id',  # raw text versions only

    # v1.1: REMOVED — data leakage
    # Scrap_counter increments AFTER scrap is detected by the machine.
    # Rolling stats of this counter (min_15m, max_15m, last_15m) ranked
    # #4, #12, and further in feature importance. Using them means the model
    # reads the answer instead of predicting from process physics.
    'Scrap_counter',

    # v1.1: REMOVED — not a causal predictor
    # Shot_counter tracks total shots fired. Its rolling min/max encodes
    # machine runtime position within a shift, not process instability.
    # It has no causal relationship with why a specific shot becomes scrap.
    'Shot_counter',

    # v1.1: REMOVED — explicitly "not process related" per TE parameter doc
    'Time_on_machine',
]

MAX_TRAIN_ROWS = 500_000


def find_target_column(df: pd.DataFrame) -> str:
    candidates = ['early_scrap_risk', 'is_scrap', 'is_scrap_actual']
    for col in candidates:
        if col in df.columns:
            print(f"[INFO] Using target column: {col}")
            return col
    raise ValueError(f"No target column found. Available: {df.columns.tolist()[:20]}")


def downsample_stratified(df: pd.DataFrame, target_col: str, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df

    print(f"[INFO] Downsampling from {len(df):,} to {max_rows:,} rows (stratified)")
    df_pos = df[df[target_col] == 1]
    df_neg = df[df[target_col] == 0]

    ratio = max_rows / len(df)
    n_pos = max(int(len(df_pos) * ratio), min(len(df_pos), 1000))
    n_neg = max_rows - n_pos

    df_pos_sample = df_pos.sample(n=min(n_pos, len(df_pos)), random_state=42)
    df_neg_sample = df_neg.sample(n=min(n_neg, len(df_neg)), random_state=42)

    result = pd.concat([df_pos_sample, df_neg_sample], ignore_index=True).sample(frac=1, random_state=42)
    print(f"[INFO] Sampled: {len(df_pos_sample):,} scrap + {len(df_neg_sample):,} non-scrap = {len(result):,} total")
    return result


def main():
    print("=" * 60)
    print(f"TRAINING LIGHTGBM SCRAP CLASSIFIER — {VERSION}")
    print("=" * 60)

    # Step 1: Load data
    print(f"\n[INFO] Loading {WIDE_FILE}...")
    df = pd.read_parquet(WIDE_FILE, engine="pyarrow")
    print(f"[INFO] Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Step 2: Find target
    target_col = find_target_column(df)

    # Step 3: Convert to float32
    print("[INFO] Converting to float32...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col != target_col:
            df[col] = df[col].astype(np.float32)

    # Step 4: Define feature columns
    # v1.1: Any column whose NAME contains an excluded term is dropped.
    # This catches all rolling variants e.g. Scrap_counter__min_15m,
    # Scrap_counter__max_30m, Shot_counter__std_5m, Time_on_machine__mean_15m
    def is_excluded(col_name):
        for exc in EXCLUDE_COLS:
            if col_name == exc or col_name.startswith(exc + '__'):
                return True
        return False

    feature_columns = [
        c for c in df.columns
        if not is_excluded(c)
        and c != target_col
        and df[c].dtype in [np.float32, np.float64, np.int64, np.int32]
    ]

    print(f"[INFO] Features selected: {len(feature_columns)}")

    # v1.1: Print which leakage features were removed
    removed = [c for c in df.columns if is_excluded(c) and c != target_col
               and df[c].dtype in [np.float32, np.float64, np.int64, np.int32]]
    print(f"[INFO] Leakage/noise features removed: {len(removed)}")
    for r in sorted(removed):
        print(f"         - {r}")

    n_pattern = len([c for c in feature_columns if any(s in c for s in PATTERN_SUFFIXES)])
    print(f"[INFO] Pattern-aware features kept: {n_pattern}")

    # Step 5: Downsample
    df = downsample_stratified(df, target_col, MAX_TRAIN_ROWS)

    # Step 6: Prepare X and y
    print("[INFO] Preparing training data...")
    X = df[feature_columns].fillna(0).values.astype(np.float32)
    y = df[target_col].fillna(0).astype(int).values

    del df
    import gc
    gc.collect()

    print(f"[INFO] X shape: {X.shape}")
    n_scrap = (y == 1).sum()
    n_normal = (y == 0).sum()
    print(f"[INFO] Scrap: {n_scrap:,} ({100*n_scrap/len(y):.2f}%) | Normal: {n_normal:,}")

    # Step 7: Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Step 8: Class weight
    # v1.1: scale_pos_weight is still auto-calculated from actual data ratio
    # After removing leakage features the true ratio may be different.
    # We cap it at 100 to prevent the over-sensitivity that caused 3.5% precision.
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    raw_ratio = n_neg / max(n_pos, 1)
    scale_pos_weight = min(raw_ratio, 100)  # v1.1: cap at 100
    print(f"[INFO] Raw imbalance ratio: {raw_ratio:.1f}x")
    print(f"[INFO] scale_pos_weight used: {scale_pos_weight:.1f} (capped at 100)")

    # Step 9: Train LightGBM
    print("\n[INFO] Training LightGBM...")
    import lightgbm as lgb

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_columns)
    valid_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_columns, reference=train_data)

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "num_leaves": 64,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "scale_pos_weight": scale_pos_weight,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )

    # Step 10: Evaluate
    print("\n[INFO] Evaluating model...")
    from sklearn.metrics import (classification_report, f1_score,
                                  precision_score, recall_score,
                                  roc_auc_score, precision_recall_curve)

    y_pred_proba = model.predict(X_test)
    y_pred_05 = (y_pred_proba >= 0.5).astype(int)

    auc_score = roc_auc_score(y_test, y_pred_proba)

    print("\n" + "=" * 60)
    print("RESULTS AT THRESHOLD 0.5")
    print("=" * 60)
    print(classification_report(y_test, y_pred_05,
                                 target_names=["Normal", "Scrap"], digits=4))
    print(f"AUC: {auc_score:.4f}")

    # v1.1: Threshold search — find best operating point
    print("\n" + "=" * 60)
    print("THRESHOLD SEARCH — find best precision/recall tradeoff")
    print("=" * 60)
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    pa, ra, ta = precision_recall_curve(y_test, y_pred_proba)
    best_f1_threshold = 0.5
    best_f1 = 0
    last_bucket = 0
    for p, r, t in zip(pa, ra, ta):
        f1 = (2*p*r/(p+r)) if (p+r) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = t
        bucket = int(p * 20) * 5
        if bucket >= 10 and bucket > last_bucket:
            print(f"{round(t,3):>10} {round(p*100,1):>9}% {round(r*100,1):>7}% {round(f1,3):>8}")
            last_bucket = bucket
        if p >= 0.80:
            break

    print(f"\n[INFO] Best F1={best_f1:.3f} at threshold={best_f1_threshold:.3f}")
    print(f"[INFO] Recommended HIGH alert threshold: {best_f1_threshold:.2f}")
    print(f"       Update ML_THRESHOLDS in config_limits.py accordingly")

    # Step 11: Save model
    print(f"\n[INFO] Saving model to {MODEL_PATH}...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] Model saved — this is now the active dashboard model")

    # Step 12: Feature importance
    print("[INFO] Generating feature importance...")
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    importance_file = PROJECT_ROOT / "models" / "feature_importance.csv"
    importance_df.to_csv(importance_file, index=False)

    print("\n[INFO] Top 15 features after leakage removal:")
    print(importance_df.head(15).to_string(index=False))

    # Step 13: Save metrics with version tag
    metrics = {
        "version": VERSION,
        "accuracy": float((y_pred_05 == y_test).mean()),
        "f1_score": float(f1_score(y_test, y_pred_05, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred_05, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_05, zero_division=0)),
        "auc": float(auc_score),
        "best_f1_threshold": float(best_f1_threshold),
        "best_f1_score": float(best_f1),
        "scale_pos_weight_raw": float(raw_ratio),
        "scale_pos_weight_used": float(scale_pos_weight),
        "n_features": len(feature_columns),
        "n_pattern_features": n_pattern,
        "n_train_samples": len(y_train),
        "n_test_samples": len(y_test),
        "n_train_scrap": int((y_train == 1).sum()),
        "n_test_scrap": int((y_test == 1).sum()),
        "features_removed_v1_1": removed,
        "note": "Scrap_counter and Shot_counter removed (data leakage). Time_on_machine removed (not process related). scale_pos_weight capped at 100."
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE — {VERSION}")
    print("=" * 60)
    print(f"Model:   {MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"\nKey Results:")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1_score']:.4f}")
    print(f"\nNext step: v1.2-filter-startup-scrap")
    print(f"  Run 02_merge_train_data.py with status_code filter,")
    print(f"  then rebuild rolling features and retrain again.")
    print("=" * 60)


if __name__ == "__main__":
    main()