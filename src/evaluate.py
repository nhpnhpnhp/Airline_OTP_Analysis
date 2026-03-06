"""
evaluate.py — Module đánh giá & so sánh mô hình.
===================================================

Metrics: ROC-AUC, PR-AUC, F1, Confusion Matrix
Interpretability: Permutation Importance, SHAP
Drift analysis: performance theo năm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
    calibration_curve,
)
from sklearn.inspection import permutation_importance

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


# ============================================================
# CORE METRICS
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> dict:
    """
    Tính toán tất cả metrics đánh giá.

    Returns:
        dict với labels: accuracy, precision, recall, f1, roc_auc, pr_auc
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)

    return metrics


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict:
    """
    Đánh giá toàn diện 1 model.
    """
    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)

    print(f"\n📊 {model_name} — Evaluation Results:")
    print(f"   {'Metric':<15s} {'Value':>10s}")
    print(f"   {'-'*25}")
    for k, v in metrics.items():
        print(f"   {k:<15s} {v:>10.4f}")

    return {
        "model_name": model_name,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def compare_models(
    results: List[dict],
) -> pd.DataFrame:
    """
    So sánh tất cả model trong bảng tổng hợp.
    """
    rows = []
    for r in results:
        row = {"Model": r["model_name"]}
        row.update(r["metrics"])
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison = comparison.set_index("Model")

    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON")
    print("=" * 60)
    print(comparison.round(4).to_string())
    return comparison


# ============================================================
# VISUALIZATION
# ============================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save: bool = True,
) -> plt.Figure:
    """Vẽ confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=["On-time", "Delayed"],
        yticklabels=["On-time", "Delayed"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png", dpi=150)

    return fig


def plot_roc_curves(
    results: List[dict],
    y_test: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """Vẽ ROC curves cho tất cả models trên 1 chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        if r["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
            auc = r["metrics"].get("roc_auc", 0)
            ax.plot(fpr, tpr, label=f'{r["model_name"]} (AUC={auc:.4f})')

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "roc_curves_comparison.png", dpi=150)

    return fig


def plot_pr_curves(
    results: List[dict],
    y_test: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """Vẽ Precision-Recall curves cho tất cả models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        if r["y_proba"] is not None:
            prec, rec, _ = precision_recall_curve(y_test, r["y_proba"])
            auc = r["metrics"].get("pr_auc", 0)
            ax.plot(rec, prec, label=f'{r["model_name"]} (PR-AUC={auc:.4f})')

    baseline = y_test.mean()
    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.5,
               label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "pr_curves_comparison.png", dpi=150)

    return fig


def plot_calibration(
    results: List[dict],
    y_test: np.ndarray,
    n_bins: int = 10,
    save: bool = True,
) -> plt.Figure:
    """Calibration plot (reliability diagram)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        if r["y_proba"] is not None:
            prob_true, prob_pred = calibration_curve(
                y_test, r["y_proba"], n_bins=n_bins, strategy="uniform"
            )
            ax.plot(prob_pred, prob_true, marker="o", label=r["model_name"])

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Plot")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "calibration_plot.png", dpi=150)

    return fig


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def plot_permutation_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    n_repeats: int = 10,
    top_n: int = 20,
    save: bool = True,
) -> plt.Figure:
    """Permutation importance plot."""
    print(f"⏳ Computing permutation importance for {model_name}...")
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, random_state=42, n_jobs=-1,
        scoring="roc_auc",
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        importance_df["feature"][::-1],
        importance_df["importance_mean"][::-1],
        xerr=importance_df["importance_std"][::-1],
        color="steelblue", alpha=0.8,
    )
    ax.set_xlabel("Mean decrease in ROC-AUC")
    ax.set_title(f"Permutation Importance — {model_name} (top {top_n})")
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            FIGURES_DIR / f"perm_importance_{model_name.lower().replace(' ', '_')}.png",
            dpi=150,
        )

    return fig


def shap_analysis(
    model: Any,
    X_test: pd.DataFrame,
    model_name: str = "Model",
    max_samples: int = 1000,
    save: bool = True,
) -> Optional[plt.Figure]:
    """
    SHAP analysis (cho tree-based models).
    Returns summary plot figure.
    """
    if not HAS_SHAP:
        print("⚠️ SHAP not installed. Skipping.")
        return None

    print(f"⏳ Computing SHAP values for {model_name}...")

    # Subsample for speed
    if len(X_test) > max_samples:
        X_sample = X_test.sample(max_samples, random_state=42)
    else:
        X_sample = X_test

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Summary plot
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f"SHAP Summary — {model_name}")
        plt.tight_layout()

        if save:
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                FIGURES_DIR / f"shap_{model_name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight",
            )

        return fig
    except Exception as e:
        print(f"⚠️ SHAP failed for {model_name}: {e}")
        return None


# ============================================================
# DRIFT ANALYSIS
# ============================================================

def drift_analysis(
    model: Any,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "ARR_DEL15",
    model_name: str = "Model",
    save: bool = True,
) -> pd.DataFrame:
    """
    Phân tích performance drift qua các năm.
    Tính metrics cho mỗi năm riêng biệt.
    """
    years = sorted(df["YEAR"].unique())
    results = []

    for year in years:
        mask = df["YEAR"] == year
        X_year = df.loc[mask, feature_cols]
        y_year = df.loc[mask, target_col]

        # Drop NaN
        valid = y_year.notna()
        X_year = X_year[valid]
        y_year = y_year[valid].astype(int)

        if len(y_year) == 0:
            continue

        # Fill NaN features
        for col in X_year.columns:
            if X_year[col].dtype in ["float64", "float32", "int64", "Int64"]:
                X_year[col] = X_year[col].fillna(X_year[col].median())
            else:
                X_year[col] = X_year[col].fillna(-1)

        y_pred = model.predict(X_year)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_year)[:, 1]

        metrics = compute_metrics(y_year, y_pred, y_proba)
        metrics["year"] = year
        metrics["n_samples"] = len(y_year)
        results.append(metrics)

    drift_df = pd.DataFrame(results).set_index("year")

    if save and len(drift_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, metric in zip(axes, ["roc_auc", "f1", "pr_auc"]):
            if metric in drift_df.columns:
                ax.plot(drift_df.index, drift_df[metric], "o-", linewidth=2)
                ax.set_title(f"{metric.upper()} by Year")
                ax.set_xlabel("Year")
                ax.set_ylabel(metric)
                ax.grid(alpha=0.3)

        plt.suptitle(f"Performance Drift — {model_name}", fontsize=14)
        plt.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            FIGURES_DIR / f"drift_{model_name.lower().replace(' ', '_')}.png",
            dpi=150,
        )

    print(f"\n📊 Drift Analysis — {model_name}:")
    print(drift_df.round(4).to_string())
    return drift_df


# ============================================================
# FULL EVALUATION PIPELINE
# ============================================================

def full_evaluation(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    track: str = "A",
) -> tuple:
    """
    Pipeline đánh giá đầy đủ tất cả models.

    Returns:
        (results_list, comparison_df)
    """
    print("=" * 60)
    print(f"📊 FULL EVALUATION PIPELINE — Track {track}")
    print("=" * 60)

    results = []
    for name, model in trained_models.items():
        r = evaluate_model(model, X_test, y_test, model_name=name)
        results.append(r)

        # Confusion matrix
        plot_confusion_matrix(y_test, r["y_pred"], model_name=f"Track{track}_{name}")

    # Comparison
    comparison = compare_models(results)

    # ROC & PR curves
    plot_roc_curves(results, y_test)
    plot_pr_curves(results, y_test)

    # Calibration
    plot_calibration(results, y_test)

    return results, comparison
