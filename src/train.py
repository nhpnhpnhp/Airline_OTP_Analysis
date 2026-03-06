"""
train.py — Model Training Utilities.
======================================

Hỗ trợ Logistic Regression, Random Forest, XGBoost/LightGBM.
Temporal CV, class imbalance handling.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Try importing boosting libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

def get_model_configs() -> Dict[str, dict]:
    """
    Trả về dictionary các cấu hình mô hình.
    Mỗi config gồm: model object, param_grid (cho tuning), description.
    """
    configs = {}

    # 1. Logistic Regression (Baseline)
    configs["logistic_regression"] = {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                solver="lbfgs",
            ))
        ]),
        "description": "Logistic Regression (Baseline) — Linear model with balanced class weights",
        "param_grid": {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
        },
    }

    # 2. Random Forest
    configs["random_forest"] = {
        "model": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "description": "Random Forest — Tree-based ensemble with balanced weights",
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [10, 20, 50],
        },
    }

    # 3. XGBoost
    if HAS_XGB:
        configs["xgboost"] = {
            "model": xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,  # Sẽ tính dynamic
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False,
                n_jobs=-1,
            ),
            "description": "XGBoost — Gradient Boosting with regularization",
            "param_grid": {
                "n_estimators": [200, 300, 500],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.2],
            },
        }

    # 4. LightGBM
    if HAS_LGB:
        configs["lightgbm"] = {
            "model": lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            ),
            "description": "LightGBM — Fast gradient boosting",
            "param_grid": {
                "n_estimators": [200, 300, 500],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.2],
            },
        }

    return configs


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Tính scale_pos_weight cho imbalanced data: n_negative / n_positive."""
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


# ============================================================
# TRAINING
# ============================================================

def train_single_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    configs: Optional[Dict] = None,
) -> Any:
    """
    Train 1 mô hình cụ thể.

    Args:
        X_train: features
        y_train: target
        model_name: key trong get_model_configs()
        configs: custom configs (optional)

    Returns:
        trained model object
    """
    if configs is None:
        configs = get_model_configs()

    if model_name not in configs:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(configs.keys())}")

    config = configs[model_name]
    model = config["model"]

    # Dynamic scale_pos_weight cho XGBoost
    if model_name == "xgboost" and HAS_XGB:
        spw = compute_scale_pos_weight(y_train)
        model.set_params(scale_pos_weight=spw)
        print(f"   ⚖️ scale_pos_weight = {spw:.2f}")

    print(f"\n🏋️ Training {model_name}: {config['description']}")
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   Positive rate: {y_train.mean():.4f}")

    model.fit(X_train, y_train)
    print(f"   ✅ Training complete!")
    return model


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Any]:
    """
    Train tất cả mô hình có sẵn.

    Returns:
        Dict[model_name → trained model]
    """
    configs = get_model_configs()
    trained = {}

    for name in configs:
        model = train_single_model(X_train, y_train, name, configs)
        trained[name] = model

    print(f"\n✅ Trained {len(trained)} models: {list(trained.keys())}")
    return trained


def save_model(model: Any, name: str, track: str = "A") -> Path:
    """Lưu model ra file .joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"track{track}_{name}.joblib"
    filepath = MODELS_DIR / filename
    joblib.dump(model, filepath)
    size_kb = filepath.stat().st_size / 1024
    print(f"💾 Saved: {filepath} ({size_kb:.0f} KB)")
    return filepath


def load_model(name: str, track: str = "A") -> Any:
    """Load model từ file .joblib."""
    filename = f"track{track}_{name}.joblib"
    filepath = MODELS_DIR / filename
    model = joblib.load(filepath)
    print(f"📖 Loaded: {filepath}")
    return model


if __name__ == "__main__":
    print("Available model configs:")
    configs = get_model_configs()
    for name, cfg in configs.items():
        print(f"  • {name}: {cfg['description']}")
