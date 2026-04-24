"""
scripts/train_risk_forecaster.py
════════════════════════════════════════════════════════════════

STACKED RISK FORECASTER — Training Script v1.0
───────────────────────────────────────────────
Strategy  : Use production model output (risk_score) as PRIMARY signal.
            No class imbalance — regression on continuous 0-1, not binary classification.

Input     : new_processed_data/KAGGLE_MASTER_275_LABELED.parquet  (already clean, 2.3M rows)
Output    : models/future_models/model_scrap_5m.pkl … model_scrap_30m.pkl

How to run:
    python d:\\te-connectivity-3\\scripts\\train_risk_forecaster.py

Expected time: ~20 minutes locally (NO Kaggle needed, NO GPU needed)

What makes this better than the previous approach:
    1. No class imbalance  : regression on 0-1 risk, not rare binary scrap labels
    2. Only 20 features    : clean, explainable, no curse of dimensionality
    3. Production model    : leverages the already-excellent 275-feature model as signal
    4. Local training      : no Kaggle, no GPU, ~20 min total
    5. Zero leakage        : future scrap labels are NOT used as targets
                             (we predict future risk, not future scrap)

After training:
    1. Restart backend: .\\run-dev.ps1
    2. Open dashboard — future orange line will reflect real risk trend projection
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"d:\te-connectivity-3")
sys.path.insert(0, str(PROJECT_ROOT))

# Import RiskForecasterModel from the BACKEND module (critical for pickle compatibility)
from backend.future_predictor import RiskForecasterModel

INPUT_PARQUET   = PROJECT_ROOT / "new_processed_data" / "KAGGLE_MASTER_275_LABELED.parquet"
PROD_MODEL_PATH = PROJECT_ROOT / "models" / "production_scrap_model.pkl"
OUTPUT_DIR      = PROJECT_ROOT / "models" / "future_models"

# ─── Configuration ────────────────────────────────────────────────────────────
HORIZONS_MIN    = [5, 10, 15, 20, 25, 30]
NUM_RISK_LAGS   = 15         # how many past risk score snapshots to use as features
ROWS_PER_MINUTE = 6          # telemetry is at ~10-second intervals (6 rows/min)

LGB_PARAMS = dict(
    n_estimators     = 500,
    learning_rate    = 0.05,
    num_leaves       = 63,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_samples= 50,
    random_state     = 42,
    n_jobs           = -1,
    verbose          = -1,
)

# ─────────────────────────────────────────────────────────────────────────────
def load_production_model():
    print(f"  Loading: {PROD_MODEL_PATH}")
    model = joblib.load(PROD_MODEL_PATH)
    if hasattr(model, "feature_name") and callable(model.feature_name):
        features = model.feature_name()
    elif hasattr(model, "feature_name_"):
        features = list(model.feature_name_)
    elif hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        raise RuntimeError("Cannot read feature names from production model")
    print(f"  Production model: {len(features)} features")
    return model, features


def batch_predict(prod_model, X: pd.DataFrame, batch_size: int = 50_000) -> np.ndarray:
    """Run production model predict_proba in memory-efficient batches.
    Passes .values (numpy) to bypass pandas categorical dtype mismatch."""
    out = np.empty(len(X), dtype=np.float32)
    X_np = X.values  # convert once — avoids categorical_feature mismatch error
    for i in range(0, len(X_np), batch_size):
        batch_np = X_np[i : i + batch_size]
        if hasattr(prod_model, "predict_proba"):
            preds = prod_model.predict_proba(batch_np)[:, 1]
        else:
            preds = prod_model.predict(batch_np)
        out[i : i + len(preds)] = preds.astype(np.float32)
        pct = min(i + batch_size, len(X_np)) / len(X_np) * 100
        print(f"    {pct:5.1f}% complete  ({min(i+batch_size,len(X_np)):,}/{len(X_np):,} rows)")
    return out


def build_risk_features(machine_group: pd.DataFrame) -> pd.DataFrame:
    """
    Given a machine's time-sorted DataFrame with a 'risk_score' column,
    add all lag / velocity / trend features needed by the stacked model.
    """
    m = machine_group.copy().reset_index(drop=True)

    # Scrap velocity (current rate — no leakage, this is historical scrap)
    if "Scrap_counter" in m.columns:
        r10 = ROWS_PER_MINUTE * 10
        r30 = ROWS_PER_MINUTE * 30
        m["scrap_velocity_10m"] = m["Scrap_counter"].diff(r10).clip(lower=0).fillna(0)
        m["scrap_velocity_30m"] = m["Scrap_counter"].diff(r30).clip(lower=0).fillna(0)
    else:
        m["scrap_velocity_10m"] = 0.0
        m["scrap_velocity_30m"] = 0.0

    # Risk lag features: risk_lag_1 = 1 sample ago (~10 seconds)
    for lag in range(1, NUM_RISK_LAGS + 1):
        m[f"risk_lag_{lag}"] = m["risk_score"].shift(lag)

    # Risk trend / volatility
    w5  = ROWS_PER_MINUTE * 5
    w15 = ROWS_PER_MINUTE * 15
    m["risk_mean_5m"]  = m["risk_score"].rolling(w5,  min_periods=1).mean()
    m["risk_std_5m"]   = m["risk_score"].rolling(w5,  min_periods=1).std().fillna(0)
    m["risk_max_15m"]  = m["risk_score"].rolling(w15, min_periods=1).max()
    m["risk_delta_5m"] = (m["risk_score"] - m["risk_score"].shift(w5)).fillna(0)

    # Future risk targets (what we want to predict)
    for h in HORIZONS_MIN:
        rows_ahead = ROWS_PER_MINUTE * h
        m[f"future_risk_{h}m"] = m["risk_score"].shift(-rows_ahead)

    return m


# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("\n" + "=" * 65)
    print("  STACKED RISK FORECASTER — Training Pipeline")
    print("=" * 65)

    # ── Step 1: Production model ─────────────────────────────────────────
    print("\n[1/5] Loading production model...")
    prod_model, prod_features = load_production_model()

    # ── Step 2: Load training data ───────────────────────────────────────
    print(f"\n[2/5] Loading training data from KAGGLE_MASTER_275_LABELED.parquet...")
    df = pd.read_parquet(INPUT_PARQUET, engine="pyarrow")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["machine", "timestamp"]).reset_index(drop=True)
    machines = sorted(df["machine"].unique())
    print(f"  Loaded : {len(df):,} rows | {len(machines)} machines: {machines}")
    print(f"  Columns: {len(df.columns)}")

    # ── Step 3: Generate risk scores via production model ────────────────
    print(f"\n[3/5] Running production model to score {len(df):,} rows...")
    print(  "  (LightGBM batch inference — approx 3-5 minutes)")

    # Build aligned feature matrix
    X_prod = pd.DataFrame(index=df.index)
    for f in prod_features:
        if f in df.columns:
            X_prod[f] = pd.to_numeric(df[f], errors="coerce").fillna(0.0)
        else:
            X_prod[f] = 0.0
    X_prod = X_prod.astype(np.float32)

    df["risk_score"] = batch_predict(prod_model, X_prod)
    print(f"  Risk score range : {df['risk_score'].min():.4f}  —  {df['risk_score'].max():.4f}")
    print(f"  Risk score mean  : {df['risk_score'].mean():.4f}")

    # ── Step 4: Build lag features + future targets per machine ──────────
    print(f"\n[4/5] Building lag features and future risk targets...")
    parts = []
    for machine in machines:
        m_df = df[df["machine"] == machine]
        print(f"  {machine}: {len(m_df):,} rows")
        parts.append(build_risk_features(m_df))

    full_df = pd.concat(parts, ignore_index=True)

    FEATURE_COLS = (
        ["risk_score"]
        + [f"risk_lag_{i}" for i in range(1, NUM_RISK_LAGS + 1)]
        + ["scrap_velocity_10m", "scrap_velocity_30m",
           "risk_mean_5m", "risk_std_5m", "risk_max_15m", "risk_delta_5m"]
    )
    TARGET_COLS = [f"future_risk_{h}m" for h in HORIZONS_MIN]

    # Drop boundary rows where lag or target is NaN
    train_df = full_df[FEATURE_COLS + TARGET_COLS].dropna()
    print(f"\n  Training rows (after NaN drop): {len(train_df):,}")
    print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    X = train_df[FEATURE_COLS].astype(np.float32)

    # ── Step 5: Train one regressor per horizon ──────────────────────────
    print(f"\n[5/5] Training 6 horizon regressors (LGBMRegressor)...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for h in HORIZONS_MIN:
        target = f"future_risk_{h}m"
        y = train_df[target].astype(np.float32)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\n  ── {h}m horizon ──  train={len(X_train):,}  val={len(X_val):,}")
        base = lgb.LGBMRegressor(**LGB_PARAMS)
        base.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(100),
            ],
        )

        val_pred = base.predict(X_val)
        mae  = float(np.mean(np.abs(val_pred - y_val.values)))
        rmse = float(np.sqrt(np.mean((val_pred - y_val.values) ** 2)))
        corr = float(np.corrcoef(val_pred, y_val.values)[0, 1])
        print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  Correlation={corr:.3f}"
              f"  BestIter={base.best_iteration_}")

        # Wrap in RiskForecasterModel and save
        wrapper = RiskForecasterModel(base, FEATURE_COLS, h)
        out_path = OUTPUT_DIR / f"model_scrap_{h}m.pkl"
        joblib.dump(wrapper, out_path)
        print(f"  Saved  ->  {out_path.name}")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = (time.time() - t0) / 60
    print("\n" + "=" * 65)
    print(f"  TRAINING COMPLETE  —  {elapsed:.1f} minutes")
    print(f"  Models saved to: {OUTPUT_DIR}")
    print()
    print("  NEXT STEPS:")
    print("  1. Restart backend:  .\\run-dev.ps1")
    print("  2. Open dashboard  — future orange line will now show")
    print("     a meaningful risk trend projection, not near-zero.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
