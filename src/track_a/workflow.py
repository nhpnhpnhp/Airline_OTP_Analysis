"""Track A workflow: statistics, modeling, evaluation, and reporting."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    BOOSTING_CONFIG,
    CLEAN_OPERATED_DIR,
    FEATURES,
    FOREST_CONFIG,
    FIG_DIR,
    LOGREG_CONFIG,
    MODEL_DIR,
    PERMUTATION_SAMPLE,
    RANDOM_SEED,
    REPORT_DIR,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_FRACTION,
)


sns.set_theme(style="whitegrid")


def ensure_dirs() -> None:
    for path in [MODEL_DIR, REPORT_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_pickle_model(model, path: Path) -> None:
    with path.open("wb") as fh:
        pickle.dump(model, fh)


def load_track_a() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    return train, test


def load_operated_columns() -> pd.DataFrame:
    cols = ["YEAR", "DEP_TIME_BLK", "ARR_DELAY_NEW", "ARR_DEL15", "OP_CARRIER", "ROUTE"]
    return pd.read_parquet(CLEAN_OPERATED_DIR, columns=cols, engine="pyarrow")


def engineer_track_a_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df[FEATURES].copy()
    feature_names = list(out.columns)
    return out.astype("float32"), feature_names


def predict_positive_class(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            return proba[:, 1].astype(np.float32)
        return proba.astype(np.float32)
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        return (1.0 / (1.0 + np.exp(-np.clip(score, -20, 20)))).astype(np.float32)
    raise AttributeError("Model must implement predict_proba or decision_function")


def regularized_gamma_q(a: float, x: float) -> float:
    if x < 0 or a <= 0:
        return float("nan")
    if x == 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(200):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-12:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - gln)

    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi_square_test(df: pd.DataFrame) -> dict:
    contingency = pd.crosstab(df["YEAR"], df[TARGET_COL]).astype("float64")
    observed = contingency.to_numpy()
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row_sums @ col_sums / total
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    dof = int((observed.shape[0] - 1) * (observed.shape[1] - 1))
    p_value = regularized_gamma_q(dof / 2.0, chi2 / 2.0)
    cramers_v = math.sqrt(chi2 / (total * min(observed.shape[0] - 1, observed.shape[1] - 1)))
    return {
        "test": "Chi-square YEAR vs ARR_DEL15",
        "statistic": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "effect_size": cramers_v,
        "effect_name": "Cramers V",
        "n": int(total),
        "table": contingency.reset_index(),
    }


def kruskal_wallis_test(df: pd.DataFrame) -> dict:
    subset = df[["DEP_TIME_BLK", "ARR_DELAY_NEW"]].dropna().copy()
    subset["rank"] = subset["ARR_DELAY_NEW"].rank(method="average")
    group_sizes = subset.groupby("DEP_TIME_BLK").size().astype("float64")
    rank_sums = subset.groupby("DEP_TIME_BLK")["rank"].sum().astype("float64")
    n = float(len(subset))
    k = len(group_sizes)
    h = (12.0 / (n * (n + 1.0))) * ((rank_sums**2) / group_sizes).sum() - 3.0 * (n + 1.0)
    tie_counts = subset["ARR_DELAY_NEW"].value_counts()
    tie_term = ((tie_counts**3) - tie_counts).sum()
    correction = 1.0 - tie_term / (n**3 - n) if n > 1 else 1.0
    h_corrected = float(h / correction) if correction > 0 else float(h)
    dof = k - 1
    p_value = regularized_gamma_q(dof / 2.0, h_corrected / 2.0)
    epsilon_sq = max((h_corrected - k + 1.0) / (n - k), 0.0)
    return {
        "test": "Kruskal-Wallis ARR_DELAY_NEW by DEP_TIME_BLK",
        "statistic": h_corrected,
        "p_value": p_value,
        "degrees_of_freedom": int(dof),
        "effect_size": epsilon_sq,
        "effect_name": "Epsilon squared",
        "n": int(n),
        "group_summary": (
            subset.groupby("DEP_TIME_BLK")["ARR_DELAY_NEW"]
            .agg(["count", "median", "mean"])
            .reset_index()
            .sort_values("mean", ascending=False)
        ),
    }


def association_summary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    carrier = (
        df.groupby("OP_CARRIER")["ARR_DEL15"]
        .agg(["count", "mean"])
        .rename(columns={"count": "flights", "mean": "delay_rate"})
        .sort_values(["flights", "delay_rate"], ascending=[False, False])
        .head(10)
        .reset_index()
    )
    route = (
        df.groupby("ROUTE")["ARR_DEL15"]
        .agg(["count", "mean"])
        .rename(columns={"count": "flights", "mean": "delay_rate"})
        .query("flights >= 500")
        .sort_values("delay_rate", ascending=False)
        .head(10)
        .reset_index()
    )
    time_block = (
        df.groupby("DEP_TIME_BLK")
        .agg(delay_rate=("ARR_DEL15", "mean"), avg_delay_new=("ARR_DELAY_NEW", "mean"))
        .sort_values("avg_delay_new", ascending=False)
        .reset_index()
    )
    return {"carrier": carrier, "route": route, "time_block": time_block}


def stratified_validation_split(y: np.ndarray, fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    pos_val = rng.choice(pos_idx, size=max(1, int(len(pos_idx) * fraction)), replace=False)
    neg_val = rng.choice(neg_idx, size=max(1, int(len(neg_idx) * fraction)), replace=False)
    val_idx = np.concatenate([pos_val, neg_val])
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[val_idx] = False
    return np.where(train_mask)[0], np.sort(val_idx)


class LogisticRegressionGD:
    def __init__(self, learning_rate: float, epochs: int, batch_size: int, l2: float, seed: int):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        rng = np.random.default_rng(self.seed)
        self.mean_ = X.mean(axis=0).astype(np.float32)
        self.std_ = X.std(axis=0).astype(np.float32)
        self.std_[self.std_ == 0] = 1.0
        Xs = ((X - self.mean_) / self.std_).astype(np.float32)
        self.coef_ = np.zeros(Xs.shape[1], dtype=np.float32)
        prior = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.intercept_ = float(np.log(prior / (1 - prior)))
        pos_weight = (len(y) - y.sum()) / max(y.sum(), 1.0)

        for _ in range(self.epochs):
            indices = rng.permutation(len(y))
            for start in range(0, len(y), self.batch_size):
                idx = indices[start:start + self.batch_size]
                Xb = Xs[idx]
                yb = y[idx]
                weights = np.where(yb == 1.0, pos_weight, 1.0).astype(np.float32)
                logits = Xb @ self.coef_ + self.intercept_
                preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
                error = (preds - yb) * weights
                norm = np.maximum(weights.sum(), 1.0)
                grad_w = (Xb.T @ error) / norm + self.l2 * self.coef_
                grad_b = float(error.sum() / norm)
                self.coef_ -= self.learning_rate * grad_w.astype(np.float32)
                self.intercept_ -= self.learning_rate * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = ((X - self.mean_) / self.std_).astype(np.float32)
        logits = Xs @ self.coef_ + self.intercept_
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            coef=self.coef_,
            intercept=np.array([self.intercept_], dtype=np.float32),
            mean=self.mean_,
            std=self.std_,
        )


@dataclass
class TreeNode:
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None
    value: float | None = None
    gain: float = 0.0

    def to_dict(self) -> dict:
        return {
            "feature_index": self.feature_index,
            "threshold": self.threshold,
            "value": self.value,
            "gain": self.gain,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
        }


class SimpleDecisionTree:
    def __init__(self, max_depth: int, min_samples_leaf: int, max_thresholds: int, min_gain: float, max_features: int | None = None, seed: int = RANDOM_SEED):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.min_gain = min_gain
        self.max_features = max_features
        self.seed = seed

    @staticmethod
    def gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p = y.mean()
        return 1.0 - p**2 - (1.0 - p) ** 2

    def best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None, float]:
        base = self.gini(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        feature_indices = np.arange(X.shape[1])
        if self.max_features is not None and self.max_features < X.shape[1]:
            feature_indices = np.sort(self.rng.choice(feature_indices, size=self.max_features, replace=False))
        for feature_idx in feature_indices:
            values = X[:, feature_idx]
            thresholds = np.unique(np.quantile(values, np.linspace(0.05, 0.95, self.max_thresholds)))
            for threshold in thresholds:
                left_mask = values <= threshold
                left_count = int(left_mask.sum())
                right_count = len(y) - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                gain = base - (
                    (left_count / len(y)) * self.gini(y[left_mask])
                    + (right_count / len(y)) * self.gini(y[~left_mask])
                )
                if gain > best_gain:
                    best_gain = float(gain)
                    best_feature = feature_idx
                    best_threshold = float(threshold)
        return best_feature, best_threshold, best_gain

    def build(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        node = TreeNode(value=float(y.mean()))
        if depth >= self.max_depth or len(y) < self.min_samples_leaf * 2 or np.unique(y).size == 1:
            return node
        feature_idx, threshold, gain = self.best_split(X, y)
        if feature_idx is None or threshold is None or gain < self.min_gain:
            return node
        left_mask = X[:, feature_idx] <= threshold
        node.feature_index = feature_idx
        node.threshold = threshold
        node.gain = gain
        node.left = self.build(X[left_mask], y[left_mask], depth + 1)
        node.right = self.build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleDecisionTree":
        self.rng = np.random.default_rng(self.seed)
        self.tree_ = self.build(X, y, 0)
        return self

    def _predict_row(self, row: np.ndarray, node: TreeNode) -> float:
        current = node
        while current.feature_index is not None and current.threshold is not None:
            current = current.left if row[current.feature_index] <= current.threshold else current.right
        return float(current.value if current.value is not None else 0.0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(row, self.tree_) for row in X], dtype=np.float32)

    def save(self, path: Path, feature_names: list[str]) -> None:
        payload = {"feature_names": feature_names, "tree": self.tree_.to_dict()}
        path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


class RandomForestLite:
    def __init__(
        self,
        n_estimators: int,
        max_depth: int,
        min_samples_leaf: int,
        max_thresholds: int,
        min_gain: float,
        sample_size: int,
        max_features: int,
        seed: int,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.min_gain = min_gain
        self.sample_size = sample_size
        self.max_features = max_features
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestLite":
        rng = np.random.default_rng(self.seed)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        self.trees_: list[SimpleDecisionTree] = []

        pos_take = min(len(pos_idx), self.sample_size // 3)
        neg_take = min(len(neg_idx), self.sample_size - pos_take)

        for tree_num in range(self.n_estimators):
            sample_idx = np.concatenate([
                rng.choice(pos_idx, size=pos_take, replace=True),
                rng.choice(neg_idx, size=neg_take, replace=True),
            ])
            rng.shuffle(sample_idx)
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_thresholds=self.max_thresholds,
                min_gain=self.min_gain,
                max_features=self.max_features,
                seed=self.seed + tree_num,
            ).fit(X[sample_idx], y[sample_idx])
            self.trees_.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack([tree.predict_proba(X) for tree in self.trees_], axis=0)
        return preds.mean(axis=0)

    def save(self, path: Path, feature_names: list[str]) -> None:
        payload = {
            "feature_names": feature_names,
            "n_estimators": self.n_estimators,
            "trees": [tree.tree_.to_dict() for tree in self.trees_],
        }
        path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


@dataclass
class RegressionStump:
    feature_index: int | None = None
    threshold: float | None = None
    left_value: float = 0.0
    right_value: float = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.feature_index is None or self.threshold is None:
            return np.zeros(X.shape[0], dtype=np.float32)
        mask = X[:, self.feature_index] <= self.threshold
        out = np.empty(X.shape[0], dtype=np.float32)
        out[mask] = self.left_value
        out[~mask] = self.right_value
        return out

    def to_dict(self) -> dict:
        return {
            "feature_index": self.feature_index,
            "threshold": self.threshold,
            "left_value": self.left_value,
            "right_value": self.right_value,
        }


class GradientBoostingLite:
    def __init__(
        self,
        n_estimators: int,
        learning_rate: float,
        sample_size: int,
        max_thresholds: int,
        min_samples_leaf: int,
        seed: int,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.sample_size = sample_size
        self.max_thresholds = max_thresholds
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed

    def _fit_stump(self, X: np.ndarray, residual: np.ndarray) -> RegressionStump:
        best = RegressionStump()
        best_loss = float("inf")
        for feature_idx in range(X.shape[1]):
            values = X[:, feature_idx]
            thresholds = np.unique(np.quantile(values, np.linspace(0.05, 0.95, self.max_thresholds)))
            for threshold in thresholds:
                left_mask = values <= threshold
                left_count = int(left_mask.sum())
                right_count = len(values) - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                left_val = float(residual[left_mask].mean())
                right_val = float(residual[~left_mask].mean())
                pred = np.where(left_mask, left_val, right_val)
                loss = float(((residual - pred) ** 2).mean())
                if loss < best_loss:
                    best_loss = loss
                    best = RegressionStump(
                        feature_index=feature_idx,
                        threshold=float(threshold),
                        left_value=left_val,
                        right_value=right_val,
                    )
        return best

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingLite":
        rng = np.random.default_rng(self.seed)
        pos_rate = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.base_score_ = float(np.log(pos_rate / (1 - pos_rate)))
        logits = np.full(X.shape[0], self.base_score_, dtype=np.float32)
        self.stumps_: list[RegressionStump] = []

        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        pos_take = min(len(pos_idx), self.sample_size // 3)
        neg_take = min(len(neg_idx), self.sample_size - pos_take)

        for _ in range(self.n_estimators):
            sample_idx = np.concatenate([
                rng.choice(pos_idx, size=pos_take, replace=False),
                rng.choice(neg_idx, size=neg_take, replace=False),
            ])
            rng.shuffle(sample_idx)
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits[sample_idx], -20, 20)))
            residual = y[sample_idx] - probs
            stump = self._fit_stump(X[sample_idx], residual)
            update = stump.predict(X)
            logits += self.learning_rate * update
            self.stumps_.append(stump)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.full(X.shape[0], self.base_score_, dtype=np.float32)
        for stump in self.stumps_:
            logits += self.learning_rate * stump.predict(X)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))

    def save(self, path: Path, feature_names: list[str]) -> None:
        payload = {
            "feature_names": feature_names,
            "base_score": self.base_score_,
            "learning_rate": self.learning_rate,
            "stumps": [stump.to_dict() for stump in self.stumps_],
        }
        path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int8)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    pos = y_true == 1
    n_pos = pos.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = ranks[pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def confusion_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(np.int8)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def pr_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum((y_true == 1).sum(), 1)
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapz(precision, recall))


def curve_points(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)
    positives = max((y_true == 1).sum(), 1)
    negatives = max((y_true == 0).sum(), 1)
    tpr = np.concatenate(([0.0], tp / positives, [1.0]))
    fpr = np.concatenate(([0.0], fp / negatives, [1.0]))
    precision = np.concatenate(([1.0], tp / np.maximum(tp + fp, 1)))
    recall = np.concatenate(([0.0], tp / positives))
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}


def best_threshold_from_validation(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict]:
    best_threshold = 0.50
    best_metrics = confusion_metrics(y_true, y_prob, best_threshold)
    for threshold in np.arange(0.20, 0.71, 0.02):
        metrics = confusion_metrics(y_true, y_prob, float(threshold))
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def permutation_importance(model, X: np.ndarray, y: np.ndarray, base_metric: float, rng: np.random.Generator, feature_names: list[str]) -> pd.DataFrame:
    rows = []
    X_work = X.copy()
    for idx, feature in enumerate(feature_names):
        original = X_work[:, idx].copy()
        rng.shuffle(X_work[:, idx])
        shuffled_metric = roc_auc_score_manual(y, predict_positive_class(model, X_work))
        rows.append({"feature": feature, "importance_drop_auc": base_metric - shuffled_metric})
        X_work[:, idx] = original
    return pd.DataFrame(rows).sort_values("importance_drop_auc", ascending=False)


def save_curve_plots(curves: dict[str, dict[str, np.ndarray]], confusion: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, curve in curves.items():
        ax.plot(curve["fpr"], curve["tpr"], label=f"{name} (AUC={curve['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("Track A ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "track_a_roc_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, curve in curves.items():
        ax.plot(curve["recall"], curve["precision"], label=f"{name} (PR-AUC={curve['pr_auc']:.3f})")
    ax.set_title("Track A Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "track_a_pr_curves.png", dpi=180)
    plt.close(fig)

    for name, cm in confusion.items():
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"], rotation=0)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=180)
        plt.close(fig)


def save_permutation_plot(df: pd.DataFrame) -> None:
    top = df.head(10).sort_values("importance_drop_auc")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"], top["importance_drop_auc"])
    ax.set_title("Track A Permutation Importance (Top 10)")
    ax.set_xlabel("ROC-AUC Drop After Shuffle")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "track_a_permutation_importance.png", dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, float_cols: list[str] | None = None) -> str:
    out = df.copy()
    if float_cols:
        for col in float_cols:
            if col in out.columns:
                out[col] = out[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    columns = list(out.columns)
    rows = out.astype(str).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def generate_report(
    chi2_result: dict,
    kruskal_result: dict,
    association: dict[str, pd.DataFrame],
    model_table: pd.DataFrame,
    best_model_name: str,
    importance_df: pd.DataFrame,
) -> None:
    report = f"""# Track A Final Report

## 1. Executive Summary

Track A focuses on pre-flight features for predicting `ARR_DEL15` with a temporal split between 2021-2024 (train) and 2025 (test). The workflow combines compact statistical analysis, leakage-aware modeling, and final reporting artifacts suitable for the course project.

## 2. Experimental Setup

- Target: `ARR_DEL15`
- Train period: 2021-2024
- Test period: 2025
- Track A features: original pre-flight and schedule-derived numeric features from the preprocessing plan
- Modeling stack: scikit-learn estimators only
- Leakage rule: exclude operational outcome variables such as arrival outcomes, taxi-in, wheels-on, and post-arrival delay causes

## 3. Statistical Analysis

### 3.1 Chi-square Test: YEAR vs ARR_DEL15

- Statistic: {chi2_result['statistic']:.4f}
- Degrees of freedom: {chi2_result['degrees_of_freedom']}
- p-value: {chi2_result['p_value']:.6g}
- Effect size ({chi2_result['effect_name']}): {chi2_result['effect_size']:.4f}

Interpretation: OTP distribution differs across years, but the effect size should be read alongside business meaning rather than p-value alone because the sample is large.

### 3.2 Kruskal-Wallis Test: ARR_DELAY_NEW by DEP_TIME_BLK

- Statistic: {kruskal_result['statistic']:.4f}
- Degrees of freedom: {kruskal_result['degrees_of_freedom']}
- p-value: {kruskal_result['p_value']:.6g}
- Effect size ({kruskal_result['effect_name']}): {kruskal_result['effect_size']:.4f}

Interpretation: delay magnitude differs across departure time blocks, supporting time-of-day as a relevant associative factor for Track A.

## 4. Comparative Association Analysis

### 4.1 Top Carrier Associations

{markdown_table(association['carrier'], ['delay_rate'])}

### 4.2 High-Delay Routes (min 500 flights)

{markdown_table(association['route'], ['delay_rate'])}

### 4.3 Time Block Summary

{markdown_table(association['time_block'].head(10), ['delay_rate', 'avg_delay_new'])}

These findings are descriptive associations. They support feature relevance, but they are not presented as causal driver analysis.

## 5. Leakage Audit and Temporal Split

- The preprocessing stage already removed forbidden leakage columns for Track A.
- Temporal evaluation uses 2025 as a forward-looking test set, which is more realistic than a random split for OTP prediction.
- The Track A feature set remains aligned with pre-flight availability assumptions.

## 6. Track A Modeling

### 6.1 Model Comparison

{markdown_table(model_table, ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'threshold'])}

### 6.2 Best Model

The selected best model is **{best_model_name}** based on test-set discrimination and overall balance between ROC-AUC, PR-AUC, and F1.

## 7. Permutation Importance

{markdown_table(importance_df.head(10), ['importance_drop_auc'])}

Permutation importance was computed only for the selected best model to keep interpretability focused and within scope.

## 8. Track B Dependency Note

Track A does not depend on Track B implementation to finish its own modeling. The only required coordination point is keeping the same target name, temporal split, and core evaluation metrics for final cross-track comparison.

## 9. Limitations and Next Steps

- Track A uses only pre-flight information, so there is an upper limit on achievable performance.
- The ensemble model is intentionally lightweight to keep project scope realistic.
- Optional future work: SHAP for one boosting-style model, lightweight dashboard overview, and a side-by-side comparison once Track B is finalized.
"""
    (REPORT_DIR / "track_a_final_report.md").write_text(report, encoding="utf-8")


def generate_slide_deck(best_model_name: str) -> None:
    slides = """# Track A Slide Deck Draft

## Slide 1 - Title
- Airline OTP Analysis
- Track A: Pre-flight Delay Prediction

## Slide 2 - Problem Motivation
- Why OTP matters for airlines and passengers
- Why January 2021-2025 BTS data is relevant

## Slide 3 - Current Project Scope
- Preprocessing pipeline completed
- Exploratory analysis completed
- Track A modeling added as the first predictive branch

## Slide 4 - Data Pipeline
- Raw BTS CSVs
- Clean parquet outputs
- ML-ready Track A dataset

## Slide 5 - Target and Split
- Target: ARR_DEL15
- Train: 2021-2024
- Test: 2025
- Why temporal split matters

## Slide 6 - Track A Feature Design
- Schedule/time features
- Route and airport frequency features
- Historical OTP features
- Original Track A feature list only
- No post-departure leakage

## Slide 7 - Leakage Audit
- Forbidden columns removed
- Why Track A remains pre-flight only

## Slide 8 - Statistical Test 1
- Chi-square YEAR vs ARR_DEL15
- Key interpretation

## Slide 9 - Statistical Test 2
- Kruskal-Wallis ARR_DELAY_NEW by DEP_TIME_BLK
- Key interpretation

## Slide 10 - Comparative Association Findings
- Carrier
- Route
- Time block

## Slide 11 - Models
- Logistic Regression baseline
- Random Forest (scikit-learn)
- Gradient Boosting (scikit-learn)

## Slide 12 - Evaluation Metrics
- ROC-AUC
- PR-AUC
- F1
- Confusion Matrix

## Slide 13 - Model Comparison
- Table of Track A results
- Selected best model: {best_model_name}

## Slide 14 - Feature Importance
- Permutation importance for the best model

## Slide 15 - Conclusion and Next Steps
- What Track A achieved
- What remains optional
- How Track B can align without blocking Track A
"""
    (REPORT_DIR / "track_a_slide_deck.md").write_text(
        slides.format(best_model_name=best_model_name),
        encoding="utf-8",
    )


def generate_track_b_dependency_note() -> None:
    text = """# Track B Dependency Memo

## What Track B must keep stable

- Target name: `ARR_DEL15`
- Temporal split: train 2021-2024, test 2025
- Core metrics: ROC-AUC, PR-AUC, F1, confusion matrix
- Narrative scope: associative/descriptive + predictive, not causal

## What does not block Track A

- Track B model choice
- Track B tuning strategy
- Track B optional interpretability work

## What could block final cross-track comparison

- Different target definition
- Different test year
- Different metric definitions
- Different artifact naming that makes comparison ambiguous
"""
    (REPORT_DIR / "track_b_dependency_report.md").write_text(text, encoding="utf-8")


def update_repo_docs(model_table: pd.DataFrame) -> None:
    summary_lines = [
        "- Track A modeling branch with statistical analysis and evaluation artifacts",
        f"- Best available Track A workflow outputs in `{REPORT_DIR.as_posix()}/` and `{MODEL_DIR.as_posix()}/`",
        "",
        "## Track A workflow",
        "",
        "Run the Track A analysis and modeling workflow:",
        "",
        "```bash",
        "python -m src.track_a.main",
        "```",
        "",
        "Main outputs:",
        "",
        "- `reports/track_a/track_a_final_report.md`",
        "- `reports/track_a/track_a_slide_deck.md`",
        "- `reports/track_a/track_b_dependency_report.md`",
        "- `reports/track_a/model_comparison.csv`",
        "- `reports/track_a/statistical_tests.csv`",
        "- `reports/track_a/figures/`",
        "- `reports/track_a/models/`",
        "",
        "## Current Track A model comparison",
        "",
        markdown_table(model_table, ["roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"]),
    ]
    readme_path = Path("README.md")
    current = readme_path.read_text(encoding="utf-8")
    marker = "## Luu y"
    if marker in current:
        before, after = current.split(marker, 1)
        updated = before.rstrip() + "\n\n" + "\n".join(summary_lines) + "\n\n" + marker + after
    else:
        updated = current.rstrip() + "\n\n" + "\n".join(summary_lines) + "\n"
    readme_path.write_text(updated, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(RANDOM_SEED)

    train_df, test_df = load_track_a()
    operated_df = load_operated_columns()

    chi2_result = chi_square_test(operated_df)
    kruskal_result = kruskal_wallis_test(operated_df)
    association = association_summary(operated_df)

    chi2_result["table"].to_csv(REPORT_DIR / "chi_square_year_table.csv", index=False)
    kruskal_result["group_summary"].to_csv(REPORT_DIR / "kruskal_timeblock_summary.csv", index=False)
    pd.DataFrame([
        {k: v for k, v in chi2_result.items() if k not in {"table"}},
        {k: v for k, v in kruskal_result.items() if k not in {"group_summary"}},
    ]).to_csv(REPORT_DIR / "statistical_tests.csv", index=False)

    for name, frame in association.items():
        frame.to_csv(REPORT_DIR / f"association_{name}.csv", index=False)

    engineered_train, feature_names = engineer_track_a_features(train_df)
    engineered_test, _ = engineer_track_a_features(test_df)

    X = engineered_train.to_numpy(dtype=np.float32)
    y = train_df[TARGET_COL].to_numpy(dtype=np.float32)
    X_test = engineered_test.to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.int8)

    train_idx, val_idx = stratified_validation_split(y, VALIDATION_FRACTION, rng)
    X_fit, y_fit = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx].astype(np.int8)

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=max(500, LOGREG_CONFIG["epochs"] * 50),
            random_state=RANDOM_SEED,
        )),
    ])
    logreg.fit(X_fit, y_fit)
    logreg_val = predict_positive_class(logreg, X_val)
    logreg_threshold, _ = best_threshold_from_validation(y_val, logreg_val)
    logreg_test = predict_positive_class(logreg, X_test)

    forest = RandomForestClassifier(
        n_estimators=FOREST_CONFIG["n_estimators"],
        max_depth=FOREST_CONFIG["max_depth"],
        min_samples_leaf=FOREST_CONFIG["min_samples_leaf"],
        max_features=FOREST_CONFIG["max_features"],
        bootstrap=True,
        max_samples=min(FOREST_CONFIG["sample_size"], len(X_fit)),
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=RANDOM_SEED,
    )
    forest.fit(X_fit, y_fit)
    forest_val = predict_positive_class(forest, X_val)
    forest_threshold, _ = best_threshold_from_validation(y_val, forest_val)
    forest_test = predict_positive_class(forest, X_test)

    boosting = GradientBoostingClassifier(
        n_estimators=BOOSTING_CONFIG["n_estimators"],
        learning_rate=BOOSTING_CONFIG["learning_rate"],
        min_samples_leaf=BOOSTING_CONFIG["min_samples_leaf"],
        subsample=min(1.0, BOOSTING_CONFIG["sample_size"] / len(X_fit)),
        random_state=RANDOM_SEED,
    )
    boosting.fit(X_fit, y_fit)
    boosting_val = predict_positive_class(boosting, X_val)
    boosting_threshold, _ = best_threshold_from_validation(y_val, boosting_val)
    boosting_test = predict_positive_class(boosting, X_test)

    model_rows = []
    curves = {}
    confusion = {}
    model_outputs = {
        "Logistic Regression": (logreg_test, logreg_threshold),
        "Random Forest": (forest_test, forest_threshold),
        "Gradient Boosting": (boosting_test, boosting_threshold),
    }

    for model_name, (scores, threshold) in model_outputs.items():
        roc_auc = roc_auc_score_manual(y_test, scores)
        pr_auc = pr_auc_score_manual(y_test, scores)
        metrics = confusion_metrics(y_test, scores, threshold)
        curves_raw = curve_points(y_test, scores)
        curves[model_name] = {
            "fpr": curves_raw["fpr"],
            "tpr": curves_raw["tpr"],
            "precision": curves_raw["precision"],
            "recall": curves_raw["recall"],
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }
        confusion[model_name] = metrics
        model_rows.append({
            "model": model_name,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "threshold": threshold,
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        })

    model_table = pd.DataFrame(model_rows).sort_values(["roc_auc", "pr_auc", "f1"], ascending=False)
    model_table.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    save_curve_plots(curves, confusion)

    best_model_name = model_table.iloc[0]["model"]
    best_model_lookup = {
        "Logistic Regression": logreg,
        "Random Forest": forest,
        "Gradient Boosting": boosting,
    }
    best_model = best_model_lookup[best_model_name]

    sample_size = min(PERMUTATION_SAMPLE, len(X_test))
    sample_idx = rng.choice(np.arange(len(X_test)), size=sample_size, replace=False)
    base_auc = roc_auc_score_manual(y_test[sample_idx], predict_positive_class(best_model, X_test[sample_idx]))
    importance_df = permutation_importance(best_model, X_test[sample_idx].copy(), y_test[sample_idx], base_auc, rng, feature_names)
    importance_df.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)
    save_permutation_plot(importance_df)

    save_pickle_model(logreg, MODEL_DIR / "logistic_regression_track_a.pkl")
    save_pickle_model(forest, MODEL_DIR / "random_forest_track_a.pkl")
    save_pickle_model(boosting, MODEL_DIR / "gradient_boosting_track_a.pkl")

    generate_report(chi2_result, kruskal_result, association, model_table, best_model_name, importance_df)
    generate_slide_deck(best_model_name)
    generate_track_b_dependency_note()
    update_repo_docs(model_table)

    summary = {
        "best_model": best_model_name,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "validation_rows": int(len(val_idx)),
    }
    (REPORT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
