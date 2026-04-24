"""
backend/future_predictor.py
───────────────────────────
Loads and runs the 6 multi-horizon future risk models.

Supports two model formats:
  1. Legacy format  : plain LightGBM / CalibratedClassifierCV (276 sensor features)
  2. Stacked format : RiskForecasterModel (risk history features, no class imbalance)

The RiskForecasterModel class is defined HERE so that joblib.load() can always
deserialize it without ImportError, regardless of where the training script lives.
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "future_models"

MODEL_FILES = {
    "5m":  "model_scrap_5m.pkl",
    "10m": "model_scrap_10m.pkl",
    "15m": "model_scrap_15m.pkl",
    "20m": "model_scrap_20m.pkl",
    "25m": "model_scrap_25m.pkl",
    "30m": "model_scrap_30m.pkl",
}

# Module-level cache — replaced by None to force reload after model file swaps
_FUTURE_MODELS_CACHE: Optional[dict] = None


# ─────────────────────────────────────────────────────────────────────────────
# RiskForecasterModel
# ─────────────────────────────────────────────────────────────────────────────
class RiskForecasterModel:
    """
    Wraps a LGBMRegressor so it looks like a classifier to future_predictor.py.

    Input features:
        risk_score, risk_lag_1..N, scrap_velocity_10m, scrap_velocity_30m,
        risk_mean_5m, risk_std_5m, risk_max_15m, risk_delta_5m

    Output:
        predict_proba(X)  ->  ndarray shape (n, 2)  col-1 = P(scrap)
        predict(X)        ->  ndarray shape (n,)

    The attribute `feature_names_in_` is set so that _get_model_features()
    can discover the required features automatically.
    """

    def __init__(self, model, feature_cols: list, horizon_min: int):
        self.model = model
        self.feature_cols = list(feature_cols)
        self.horizon_min = horizon_min
        # Expose as sklearn-style attribute so _get_model_features() finds it
        self.feature_names_in_ = self.feature_cols

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pred = np.clip(self.model.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - pred, pred])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(X), 0.0, 1.0)

    def is_risk_forecaster(self) -> bool:
        """Lets _generate_future_horizon detect the stacked format."""
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_model_features(model) -> list:
    """
    Robustly extract the ordered feature list from any supported model type.
    Priority: inner LightGBM booster (most accurate) → sklearn attrs.
    """
    # 1. RiskForecasterModel wrapper (our new stacked model)
    if isinstance(model, RiskForecasterModel):
        return list(model.feature_cols)

    # 2. CalibratedClassifierCV: drill into inner LightGBM
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        inner = model.calibrated_classifiers_[0].estimator
        if hasattr(inner, "booster_"):
            return inner.booster_.feature_name()
        if hasattr(inner, "feature_name_"):
            return list(inner.feature_name_)
        if hasattr(inner, "feature_names_in_"):
            return list(inner.feature_names_in_)

    # 3. Plain LGBMClassifier / LGBMRegressor
    if hasattr(model, "booster_"):
        return model.booster_.feature_name()
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)

    # 4. Sklearn fallback
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # 5. Legacy LightGBM callable API
    if hasattr(model, "feature_name"):
        fn = model.feature_name
        return fn() if callable(fn) else list(fn)

    return []


def needs_risk_history(model) -> bool:
    """Returns True if this model requires risk history features (stacked format)."""
    return isinstance(model, RiskForecasterModel) or (
        hasattr(model, "is_risk_forecaster") and model.is_risk_forecaster()
    )


def _load_future_models() -> dict:
    """Load all horizon models from disk, with module-level caching."""
    global _FUTURE_MODELS_CACHE
    if _FUTURE_MODELS_CACHE is not None:
        return _FUTURE_MODELS_CACHE

    models = {}
    for horizon, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename
        if model_path.exists():
            try:
                m = joblib.load(model_path)
                features = _get_model_features(m)
                kind = "stacked" if needs_risk_history(m) else "sensor"
                print(f"[future] Loaded {filename}: {len(features)} features [{kind}]")
                models[horizon] = m
            except Exception as e:
                print(f"[future] Failed to load {filename}: {e}")

    _FUTURE_MODELS_CACHE = models
    return models


def clear_future_model_cache():
    """Force a reload from disk (call after swapping model files)."""
    global _FUTURE_MODELS_CACHE
    _FUTURE_MODELS_CACHE = None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict_future_risk(feature_row: dict, feature_columns: list) -> dict:
    """
    Run all horizon models on feature_row.

    feature_row must contain:
      - All 275 sensor-derived features (for legacy models)
      - PLUS: risk_score, risk_lag_1..N, scrap_velocity_*, risk_* trend fields
                (for stacked RiskForecasterModel — added by _generate_future_horizon)
    """
    models = _load_future_models()
    if not models:
        return {}

    results = {}
    for horizon, model in models.items():
        m_features = _get_model_features(model)
        if not m_features:
            m_features = feature_columns

        # Build input aligned to this model's exact feature schema
        row = {f: float(feature_row.get(f, 0.0)) for f in m_features}
        X = pd.DataFrame([row], columns=list(m_features))
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        try:
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X)[0, 1])
            else:
                prob = float(model.predict(X)[0])
            results[horizon] = prob
        except Exception as e:
            print(f"[future] Model {horizon} prediction failed: {e}")
            results[horizon] = 0.0

    return results
