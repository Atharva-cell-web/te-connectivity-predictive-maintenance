"""
retrain_simple_model_v2.py
==========================
VERSION: v2.3-fix-early-stopping

KEY FIX vs v2.2:
  Early stopping was incorrectly using binary_logloss (which peaks at round 1
  when model barely fires) instead of AUC. LightGBM was stopping at iteration 1
  because logloss was minimal there — not because the model was best there.

  Fix:
    - metric: "auc" only (removed binary_logloss)
    - callbacks: early_stopping(..., first_metric_only=True)
  This forces early stopping to use AUC exclusively.

  Evidence from v2.2 logs:
    [25] valid auc: 0.850  ← improving
    [50] valid auc: 0.868  ← still improving
    Best=[1] valid auc: 0.736  ← wrong! logloss was lowest at round 1
  With AUC-only stopping, model should train 100-300 rounds and reach ~0.87+ AUC.

All other fixes retained from v2.2:
  - Correct Hydra (new_raw_data/, all 5 machines, status-200 only)
  - Stratified split (fixes distribution shift from temporal split)
  - Leakage features excluded

RUN FROM PROJECT ROOT:
  cd "D:\\te connectivity 3"
  python scripts/retrain_simple_model_v2.py
"""

import gc
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "new_processed_data"
MODEL_DIR     = PROJECT_ROOT / "models"
VERSION       = "v2.3-fix-early-stopping"

PRODUCTION_STATUS_CODES = [200]
EXCLUDE_SENSORS = {"Scrap_counter", "Shot_counter", "Time_on_machine",
                   "Machine_status", "Alrams_array"}
MAX_ROWS_PER_MACHINE = 250_000


def load_hydra_scrap():
    print("\n[1] Loading Hydra scrap events from new_raw_data/...")
    candidates = [f for f in (PROJECT_ROOT/"new_raw_data").glob("*.xlsx")
                  if not f.name.startswith("~$")]
    if not candidates:
        raise FileNotFoundError(
            "No Hydra xlsx found in new_raw_data/. Close Excel if file is open.")
    hydra_file = candidates[0]

    df = pd.read_excel(hydra_file)
    df.columns = df.columns.str.strip()
    print(f"    File: {hydra_file.name} | Rows: {len(df):,}")
    print(f"    Machines: {sorted(df['machine_id'].str.strip().unique())}")

    scrap = df[df["scrap_quantity"] > 0].copy()
    if "machine_status_code" in scrap.columns:
        before = len(scrap)
        scrap = scrap[scrap["machine_status_code"].isin(PRODUCTION_STATUS_CODES)]
        print(f"    Status-200 filter: {len(scrap):,} kept "
              f"(removed {before-len(scrap):,})")

    scrap["machine_id_clean"] = (scrap["machine_id"].astype(str)
                                  .str.replace("-","").str.upper().str.strip())

    def build_ts(d):
        base = pd.to_datetime(d["machine_event_create_date"])
        if base.dt.tz is not None:
            base = base.dt.tz_convert("UTC").dt.tz_localize(None)
        if "machine_event_create_time" in d.columns:
            offset = pd.to_timedelta(
                pd.to_numeric(d["machine_event_create_time"],
                              errors="coerce").fillna(0), unit="s")
            return (base + offset).dt.tz_localize("UTC")
        return base.dt.tz_localize("UTC")

    scrap["merge_ts"] = build_ts(scrap)
    scrap = scrap.sort_values("merge_ts")
    print(f"    Production scrap events: {len(scrap):,}")
    for m, g in scrap.groupby("machine_id_clean"):
        print(f"      {m}: {len(g):,} events | "
              f"{int(g['scrap_quantity'].sum()):,} parts")
    return scrap, hydra_file


def process_machine(machine_id, scrap_events, cutoff_date):
    train_file = PROCESSED_DIR / f"{machine_id}_TRAIN.parquet"
    if not train_file.exists():
        print(f"    WARN: {train_file.name} not found, skipping")
        return None

    print(f"\n[2] Processing {machine_id}...")
    raw = pd.read_parquet(train_file)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp"])
    raw = raw[raw["timestamp"] <= cutoff_date]

    if len(raw) == 0:
        print(f"    No data before cutoff, skipping")
        return None

    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["value"])
    if "variable_name" in raw.columns:
        raw = raw[~raw["variable_name"].isin(EXCLUDE_SENSORS)]

    pivot = raw.pivot_table(
        index="timestamp", columns="variable_name",
        values="value", aggfunc="mean"
    ).reset_index()
    pivot = pivot.sort_values("timestamp").reset_index(drop=True)
    print(f"    Shots: {len(pivot):,}")

    # Label first so we can stratify downsample
    m_scrap = scrap_events[
        scrap_events["machine_id_clean"] == machine_id
    ][["merge_ts"]].copy()
    m_scrap = m_scrap.rename(columns={"merge_ts": "timestamp"})
    m_scrap["is_scrap"] = 1
    m_scrap = m_scrap.sort_values("timestamp")

    if not m_scrap.empty:
        pivot = pd.merge_asof(
            pivot, m_scrap[["timestamp","is_scrap"]],
            on="timestamp", direction="nearest",
            tolerance=pd.Timedelta("5 minutes")
        )
        pivot["is_scrap"] = pivot["is_scrap"].fillna(0).astype(int)
    else:
        pivot["is_scrap"] = 0

    # Stratified downsample preserving scrap ratio
    if len(pivot) > MAX_ROWS_PER_MACHINE:
        pos = pivot[pivot["is_scrap"] == 1]
        neg = pivot[pivot["is_scrap"] == 0]
        ratio = MAX_ROWS_PER_MACHINE / len(pivot)
        n_pos = min(len(pos), max(int(len(pos)*ratio), min(len(pos), 2000)))
        n_neg = MAX_ROWS_PER_MACHINE - n_pos
        pivot = pd.concat([
            pos.sample(n=n_pos, random_state=42),
            neg.sample(n=min(n_neg, len(neg)), random_state=42)
        ], ignore_index=True).sample(frac=1, random_state=42)
        pivot = pivot.sort_values("timestamp").reset_index(drop=True)

    pivot["machine_id"] = machine_id
    scrap_n = int(pivot["is_scrap"].sum())
    print(f"    Final: {len(pivot):,} rows | scrap: {scrap_n:,} "
          f"({scrap_n/len(pivot)*100:.3f}%)")

    del raw
    gc.collect()
    return pivot


def train_model(df):
    print(f"\n[3] Training LightGBM ({VERSION})...")
    import lightgbm as lgb
    from sklearn.metrics import (classification_report, roc_auc_score,
                                  precision_recall_curve, f1_score,
                                  precision_score, recall_score)
    from sklearn.model_selection import train_test_split

    label_col = "is_scrap"
    exclude_exact = {label_col, "timestamp", "machine_id",
                     "machine_id_normalized", "is_scrap_actual",
                     "scrap_probability", "predicted_scrap"}

    feature_cols = [
        c for c in df.columns
        if c not in exclude_exact
        and not any(c == ex or c.startswith(ex+"__") for ex in EXCLUDE_SENSORS)
        and df[c].dtype in [np.float32, np.float64, np.int32, np.int64]
    ]
    print(f"    Features: {len(feature_cols)}")
    print(f"    Features list: {feature_cols}")

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df[label_col].fillna(0).astype(int).values
    del df
    gc.collect()

    n_pos = int((y==1).sum())
    n_neg = int((y==0).sum())
    print(f"    Total: {len(y):,} | Scrap: {n_pos:,} ({100*n_pos/len(y):.2f}%)")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"    Train: {len(X_train):,} | scrap: {int((y_train==1).sum()):,} "
          f"({100*(y_train==1).mean():.2f}%)")
    print(f"    Test:  {len(X_test):,}  | scrap: {int((y_test==1).sum()):,} "
          f"({100*(y_test==1).mean():.2f}%)")

    raw_ratio = n_neg / max(n_pos, 1)
    spw = min(raw_ratio, 40)
    print(f"    Imbalance: {raw_ratio:.1f}x → scale_pos_weight={spw:.1f}")

    train_data = lgb.Dataset(X_train, label=y_train,
                              feature_name=feature_cols)
    valid_data = lgb.Dataset(X_test, label=y_test,
                              feature_name=feature_cols,
                              reference=train_data)

    params = {
        "objective":         "binary",
        "metric":            "auc",       # v2.3: AUC ONLY — removes logloss conflict
        "boosting_type":     "gbdt",
        "num_leaves":        31,
        "learning_rate":     0.05,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "verbose":           -1,
        "scale_pos_weight":  spw,
        "min_child_samples": 20,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "n_jobs":            -1,
    }

    print(f"\n    Training... (watching valid AUC, early stop at 50 rounds no improvement)")
    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[valid_data],          # v2.3: only valid set (cleaner logging)
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, first_metric_only=True),
            lgb.log_evaluation(period=25),
        ],
    )
    print(f"\n    Best iteration: {model.best_iteration}")
    print(f"    Best valid AUC: {model.best_score['valid']['auc']:.4f}")

    y_proba  = model.predict(X_test)
    auc      = roc_auc_score(y_test, y_proba)
    y_pred05 = (y_proba >= 0.5).astype(int)

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
        f1v = (2*p*r/(p+r)) if (p+r) > 0 else 0
        if f1v > best_f1:
            best_f1, best_t = f1v, t
        bucket = int(p * 20) * 5
        if bucket >= 10 and bucket > last_bucket:
            print(f"{round(t,3):>10} {round(p*100,1):>9}% "
                  f"{round(r*100,1):>7}% {round(f1v,3):>8}")
            last_bucket = bucket
        if p >= 0.85:
            break

    print(f"\nBest F1={best_f1:.3f} at threshold={best_t:.3f}")
    print(f"\n→ Update config_limits.py ML_THRESHOLDS to:")
    print(f"  'LOW':    {round(best_t*0.4, 2)}")
    print(f"  'MEDIUM': {round(best_t*0.7, 2)}")
    print(f"  'HIGH':   {best_t:.2f}")

    importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print(f"\nTop 15 features:")
    print(importance_df.head(15).to_string(index=False))
    importance_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    metrics = {
        "version":        VERSION,
        "auc":            float(auc),
        "best_auc":       float(model.best_score["valid"]["auc"]),
        "best_f1":        float(best_f1),
        "best_threshold": float(best_t),
        "best_iteration": model.best_iteration,
        "n_features":     len(feature_cols),
        "spw_used":       float(spw),
        "note": "v2.3: AUC-only early stopping fix. Stratified split. "
                "All 5 machines. Status-200 scrap. No leakage features."
    }
    return model, feature_cols, metrics, best_t


def regenerate_feb(model, feature_cols, threshold, hydra_file):
    print(f"\n[4] Regenerating FEB_TEST_RESULTS.parquet...")
    df_h = pd.read_excel(hydra_file)
    df_h.columns = df_h.columns.str.strip()
    scrap_test = df_h[df_h["scrap_quantity"] > 0].copy()
    scrap_test["machine_id_clean"] = (scrap_test["machine_id"].astype(str)
                                       .str.replace("-","").str.upper().str.strip())

    def build_ts(d):
        base = pd.to_datetime(d["machine_event_create_date"])
        if base.dt.tz is not None:
            base = base.dt.tz_convert("UTC").dt.tz_localize(None)
        if "machine_event_create_time" in d.columns:
            offset = pd.to_timedelta(
                pd.to_numeric(d["machine_event_create_time"],
                              errors="coerce").fillna(0), unit="s")
            return (base + offset).dt.tz_localize("UTC")
        return base.dt.tz_localize("UTC")

    scrap_test["merge_ts"] = build_ts(scrap_test)

    all_results = []
    for test_file in sorted(PROCESSED_DIR.glob("M*_TEST.parquet")):
        machine_id = test_file.stem.replace("_TEST","")
        print(f"    Scoring {machine_id}...")

        raw = pd.read_parquet(test_file)
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        raw = raw.dropna(subset=["value"])
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True,
                                           errors="coerce")
        raw = raw.dropna(subset=["timestamp"])
        if "variable_name" in raw.columns:
            raw = raw[~raw["variable_name"].isin(EXCLUDE_SENSORS)]

        pivot = raw.pivot_table(
            index="timestamp", columns="variable_name",
            values="value", aggfunc="mean"
        ).reset_index()
        pivot = pivot.sort_values("timestamp").reset_index(drop=True)

        for col in feature_cols:
            if col not in pivot.columns:
                pivot[col] = 0.0

        X = pivot[feature_cols].fillna(0).values.astype(np.float32)
        pivot["scrap_probability"] = model.predict(X).clip(0, 1)
        pivot["predicted_scrap"]   = (
            pivot["scrap_probability"] >= threshold).astype(int)

        m_scrap = scrap_test[
            scrap_test["machine_id_clean"] == machine_id
        ][["merge_ts"]].copy()
        m_scrap = m_scrap.rename(columns={"merge_ts":"timestamp"})
        m_scrap["is_scrap_actual"] = 1
        m_scrap = m_scrap.sort_values("timestamp")

        if not m_scrap.empty:
            pivot = pd.merge_asof(
                pivot, m_scrap,
                on="timestamp", direction="nearest",
                tolerance=pd.Timedelta("5 minutes")
            )
        pivot["is_scrap_actual"] = pivot.get(
            "is_scrap_actual", pd.Series(0, index=pivot.index)
        ).fillna(0).astype(int)
        pivot["machine_id_normalized"] = machine_id

        s = int(pivot["is_scrap_actual"].sum())
        p = int(pivot["predicted_scrap"].sum())
        prec = f"{min(s,p)/p*100:.1f}%" if p > 0 else "N/A"
        print(f"      Rows: {len(pivot):,} | actual: {s:,} | "
              f"predicted: {p:,} | est. precision: {prec}")

        all_results.append(pivot)
        del raw, pivot
        gc.collect()

    feb_df = pd.concat(all_results, ignore_index=True)
    out = PROCESSED_DIR / "FEB_TEST_RESULTS.parquet"
    feb_df.to_parquet(out, index=False)

    total = len(feb_df)
    scrap = int(feb_df["is_scrap_actual"].sum())
    pred  = int(feb_df["predicted_scrap"].sum())
    print(f"\n    Saved: {total:,} rows → {out.name}")
    print(f"    Actual scrap:    {scrap:,} ({scrap/total*100:.3f}%)")
    print(f"    Predicted scrap: {pred:,}")
    if pred > 0:
        print(f"    Overall est. precision: ~{min(scrap,pred)/pred*100:.1f}%")


def main():
    print(f"\n{'='*60}")
    print(f"RETRAIN SIMPLE MODEL — {VERSION}")
    print(f"{'='*60}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    cutoff_date = pd.Timestamp("2026-01-11 23:59:59", tz="UTC")
    scrap_events, hydra_file = load_hydra_scrap()

    all_dfs = []
    for mid in ["M231","M356","M471","M607","M612"]:
        df = process_machine(mid, scrap_events, cutoff_date)
        if df is not None:
            all_dfs.append(df)
            gc.collect()

    combined = pd.concat(all_dfs, ignore_index=True)
    total_scrap = int(combined["is_scrap"].sum())
    print(f"\nCombined: {len(combined):,} rows | "
          f"scrap: {total_scrap:,} ({total_scrap/len(combined)*100:.3f}%)")
    del all_dfs
    gc.collect()

    model, feature_cols, metrics, best_t = train_model(combined)
    del combined
    gc.collect()

    joblib.dump(model,        MODEL_DIR / "lightgbm_scrap_model.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "model_features.pkl")
    with open(MODEL_DIR / "training_metrics_v2_3.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nModel saved    → models/lightgbm_scrap_model.pkl")
    print(f"Features saved → models/model_features.pkl")
    print(f"Metrics saved  → models/training_metrics_v2_3.json")

    regenerate_feb(model, feature_cols, best_t, hydra_file)

    print(f"\n{'='*60}")
    print(f"COMPLETE — {VERSION}")
    print(f"{'='*60}")
    print(f"  AUC:            {metrics['auc']:.4f}")
    print(f"  Best AUC:       {metrics['best_auc']:.4f}")
    print(f"  Best F1:        {metrics['best_f1']:.3f}")
    print(f"  Best threshold: {metrics['best_threshold']:.3f}")
    print(f"  Best iteration: {metrics['best_iteration']}")
    print(f"\nNext steps:")
    print(f"  1. Update config_limits.py ML_THRESHOLDS")
    print(f"  2. python backend/start_server.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()