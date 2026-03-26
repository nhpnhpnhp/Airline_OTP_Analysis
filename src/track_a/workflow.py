"""Track A workflow: statistics, modeling, evaluation, and reporting."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, kruskal
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
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


def chi_square_test(df: pd.DataFrame) -> dict:
    contingency = pd.crosstab(df["YEAR"], df[TARGET_COL]).astype("float64")
    observed = contingency.to_numpy()
    chi2, p_value, dof, _ = chi2_contingency(observed, correction=False)
    total = observed.sum()
    min_dim = min(observed.shape[0] - 1, observed.shape[1] - 1)
    cramers_v = math.sqrt(chi2 / (total * min_dim)) if min_dim > 0 and total > 0 else float("nan")
    return {
        "test": "Kiểm định Chi-bình phương YEAR vs ARR_DEL15",
        "statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "effect_size": cramers_v,
        "effect_name": "Cramér's V",
        "n": int(total),
        "table": contingency.reset_index(),
    }


def kruskal_wallis_test(df: pd.DataFrame) -> dict:
    subset = df[["DEP_TIME_BLK", "ARR_DELAY_NEW"]].dropna().copy()
    groups = [group["ARR_DELAY_NEW"].to_numpy() for _, group in subset.groupby("DEP_TIME_BLK")]
    statistic, p_value = kruskal(*groups, nan_policy="omit")
    n = float(len(subset))
    k = len(groups)
    dof = k - 1
    epsilon_sq = max((float(statistic) - k + 1.0) / (n - k), 0.0) if n > k else 0.0
    return {
        "test": "Kiểm định Kruskal-Wallis ARR_DELAY_NEW theo DEP_TIME_BLK",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "effect_size": epsilon_sq,
        "effect_name": "Epsilon bình phương",
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
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=fraction,
        random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
        stratify=y,
    )
    return np.sort(train_idx), np.sort(val_idx)


def confusion_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    return {
        "threshold": threshold,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def curve_points(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, np.ndarray]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
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
        shuffled_metric = float(roc_auc_score(y, predict_positive_class(model, X_work)))
        rows.append({"feature": feature, "importance_drop_auc": base_metric - shuffled_metric})
        X_work[:, idx] = original
    return pd.DataFrame(rows).sort_values("importance_drop_auc", ascending=False)


def save_curve_plots(curves: dict[str, dict[str, np.ndarray]], confusion: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, curve in curves.items():
        ax.plot(curve["fpr"], curve["tpr"], label=f"{name} (AUC={curve['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("Đường cong ROC của Track A")
    ax.set_xlabel("Tỷ lệ dương giả")
    ax.set_ylabel("Tỷ lệ dương thật")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "track_a_roc_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, curve in curves.items():
        ax.plot(curve["recall"], curve["precision"], label=f"{name} (PR-AUC={curve['pr_auc']:.3f})")
    ax.set_title("Đường cong Precision-Recall của Track A")
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
        ax.set_title(f"Ma trận nhầm lẫn - {name}")
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"], rotation=0)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=180)
        plt.close(fig)


def save_permutation_plot(df: pd.DataFrame) -> None:
    top = df.head(10).sort_values("importance_drop_auc")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"], top["importance_drop_auc"])
    ax.set_title("Độ quan trọng hoán vị của Track A (Top 10)")
    ax.set_xlabel("Mức giảm ROC-AUC sau khi xáo trộn")
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
    report = f"""# Báo cáo cuối cùng - Track A

## 1. Tóm tắt điều hành

Track A tập trung vào các biến có sẵn trước chuyến bay để dự đoán `ARR_DEL15`, với cách chia thời gian 2021-2024 cho tập huấn luyện và 2025 cho tập kiểm tra. Workflow kết hợp phân tích thống kê gọn, mô hình hóa có kiểm soát rò rỉ dữ liệu và sinh bộ artifact cuối cùng phục vụ cho đồ án môn học.

## 2. Thiết lập thí nghiệm

- Biến mục tiêu: `ARR_DEL15`
- Giai đoạn huấn luyện: 2021-2024
- Giai đoạn kiểm tra: 2025
- Tập đặc trưng Track A: các biến số gốc trước chuyến bay và các biến dẫn xuất từ lịch bay trong kế hoạch preprocessing
- Bộ mô hình: chỉ dùng estimator từ `scikit-learn`
- Quy tắc chống leakage: loại bỏ các biến phản ánh kết quả vận hành như kết quả đến, taxi-in, wheels-on và các nguyên nhân trễ sau khi hạ cánh

## 3. Phân tích thống kê

### 3.1 Kiểm định Chi-bình phương: YEAR và ARR_DEL15

- Thống kê kiểm định: {chi2_result['statistic']:.4f}
- Bậc tự do: {chi2_result['degrees_of_freedom']}
- p-value: {chi2_result['p_value']:.6g}
- Kích thước hiệu ứng ({chi2_result['effect_name']}): {chi2_result['effect_size']:.4f}

Diễn giải: phân phối OTP có khác biệt giữa các năm, nhưng cần đọc kích thước hiệu ứng cùng với ý nghĩa nghiệp vụ thay vì chỉ nhìn p-value vì cỡ mẫu rất lớn.

### 3.2 Kiểm định Kruskal-Wallis: ARR_DELAY_NEW theo DEP_TIME_BLK

- Thống kê kiểm định: {kruskal_result['statistic']:.4f}
- Bậc tự do: {kruskal_result['degrees_of_freedom']}
- p-value: {kruskal_result['p_value']:.6g}
- Kích thước hiệu ứng ({kruskal_result['effect_name']}): {kruskal_result['effect_size']:.4f}

Diễn giải: mức độ trễ khác nhau giữa các khung giờ khởi hành, cho thấy thời điểm trong ngày là một yếu tố liên quan đáng chú ý đối với Track A.

## 4. Phân tích liên hệ mô tả

### 4.1 Các hãng có liên hệ nổi bật

{markdown_table(association['carrier'], ['delay_rate'])}

### 4.2 Các chặng có tỷ lệ trễ cao (tối thiểu 500 chuyến)

{markdown_table(association['route'], ['delay_rate'])}

### 4.3 Tóm tắt theo khung giờ

{markdown_table(association['time_block'].head(10), ['delay_rate', 'avg_delay_new'])}

Các kết quả trên chỉ mang tính mô tả và liên hệ. Chúng hỗ trợ việc chọn đặc trưng, nhưng không được trình bày như bằng chứng nhân quả.

## 5. Kiểm soát leakage và chia theo thời gian

- Giai đoạn preprocessing đã loại bỏ các cột leakage không hợp lệ cho Track A.
- Việc đánh giá trên năm 2025 phản ánh bài toán dự báo thực tế tốt hơn so với chia ngẫu nhiên.
- Tập đặc trưng Track A vẫn bám sát giả định chỉ sử dụng thông tin có trước chuyến bay.

## 6. Mô hình hóa Track A

### 6.1 So sánh mô hình

{markdown_table(model_table, ['roc_auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'f1', 'threshold'])}

### 6.2 Mô hình tốt nhất

Mô hình được chọn là **{best_model_name}**, dựa trên khả năng phân biệt trên tập kiểm tra và sự cân bằng tổng thể giữa ROC-AUC, PR-AUC, Accuracy và F1.

## 7. Độ quan trọng hoán vị

{markdown_table(importance_df.head(10), ['importance_drop_auc'])}

Phân tích permutation importance chỉ được tính cho mô hình tốt nhất để giữ phần diễn giải ở mức tập trung và phù hợp phạm vi đồ án.

## 8. Ghi chú phụ thuộc với Track B

Track A không phụ thuộc vào việc Track B hoàn thiện mô hình để kết thúc phần việc riêng của mình. Điểm cần phối hợp duy nhất là giữ thống nhất tên biến mục tiêu, cách chia theo thời gian và bộ metric cốt lõi để có thể so sánh chéo ở giai đoạn cuối.

## 9. Hạn chế và hướng tiếp theo

- Track A chỉ dùng thông tin trước chuyến bay, nên có một giới hạn tự nhiên về mức hiệu năng có thể đạt được.
- Các mô hình ensemble được giữ ở mức gọn để phạm vi triển khai phù hợp với đồ án.
- Hướng mở rộng tùy chọn: SHAP cho một mô hình boosting, dashboard tổng quan nhẹ và so sánh song song khi Track B hoàn thiện.
"""
    (REPORT_DIR / "track_a_final_report.md").write_text(report, encoding="utf-8")


def generate_slide_deck(best_model_name: str) -> None:
    slides = """# Phác thảo slide - Track A

## Slide 1 - Tiêu đề
- Airline OTP Analysis
- Track A: Dự đoán chuyến bay đến trễ từ thông tin trước chuyến bay

## Slide 2 - Động lực bài toán
- Vì sao OTP quan trọng với hãng bay và hành khách
- Vì sao dữ liệu BTS tháng 1 giai đoạn 2021-2025 là phù hợp

## Slide 3 - Phạm vi hiện tại của dự án
- Đã hoàn thành pipeline preprocessing
- Đã hoàn thành exploratory analysis
- Đã bổ sung Track A như nhánh mô hình dự báo đầu tiên

## Slide 4 - Pipeline dữ liệu
- CSV raw từ BTS
- Các đầu ra parquet đã làm sạch
- Tập dữ liệu Track A sẵn sàng cho ML

## Slide 5 - Biến mục tiêu và cách chia dữ liệu
- Biến mục tiêu: ARR_DEL15
- Huấn luyện: 2021-2024
- Kiểm tra: 2025
- Vì sao chia theo thời gian là quan trọng

## Slide 6 - Thiết kế đặc trưng cho Track A
- Đặc trưng lịch bay và thời gian
- Đặc trưng tần suất tuyến bay và sân bay
- Đặc trưng OTP lịch sử
- Chỉ dùng danh sách đặc trưng gốc của Track A
- Không dùng leakage sau giờ cất cánh

## Slide 7 - Kiểm tra leakage
- Đã loại bỏ các cột bị cấm
- Vì sao Track A vẫn giữ đúng giả định trước chuyến bay

## Slide 8 - Kiểm định thống kê 1
- Chi-bình phương giữa YEAR và ARR_DEL15
- Ý nghĩa chính của kết quả

## Slide 9 - Kiểm định thống kê 2
- Kruskal-Wallis cho ARR_DELAY_NEW theo DEP_TIME_BLK
- Ý nghĩa chính của kết quả

## Slide 10 - Các phát hiện liên hệ mô tả
- Hãng bay
- Tuyến bay
- Khung giờ

## Slide 11 - Các mô hình
- Logistic Regression làm baseline
- Random Forest (`scikit-learn`)
- Gradient Boosting (`scikit-learn`)

## Slide 12 - Bộ metric đánh giá
- ROC-AUC
- PR-AUC
- Accuracy
- F1
- Ma trận nhầm lẫn

## Slide 13 - So sánh mô hình
- Bảng kết quả của Track A
- Mô hình được chọn: {best_model_name}

## Slide 14 - Độ quan trọng đặc trưng
- Permutation importance cho mô hình tốt nhất

## Slide 15 - Kết luận và bước tiếp theo
- Track A đã hoàn thành được gì
- Phần nào vẫn là mở rộng tùy chọn
- Track B có thể căn chỉnh như thế nào mà không chặn tiến độ Track A
"""
    (REPORT_DIR / "track_a_slide_deck.md").write_text(
        slides.format(best_model_name=best_model_name),
        encoding="utf-8",
    )


def generate_track_b_dependency_note() -> None:
    text = """# Ghi chú phụ thuộc với Track B

## Những điểm Track B cần giữ ổn định

- Tên biến mục tiêu: `ARR_DEL15`
- Cách chia theo thời gian: train 2021-2024, test 2025
- Bộ metric cốt lõi: ROC-AUC, PR-AUC, Accuracy, F1, ma trận nhầm lẫn
- Phạm vi diễn giải: mô tả/liên hệ + dự báo, không khẳng định nhân quả

## Những gì không chặn tiến độ của Track A

- Lựa chọn mô hình của Track B
- Chiến lược tuning của Track B
- Phần interpretability mở rộng của Track B

## Những gì có thể làm hỏng so sánh chéo cuối kỳ

- Khác định nghĩa biến mục tiêu
- Khác năm kiểm tra
- Khác cách định nghĩa metric
- Khác quy ước đặt tên artifact khiến việc đối chiếu bị mơ hồ
"""
    (REPORT_DIR / "track_b_dependency_report.md").write_text(text, encoding="utf-8")


def update_repo_docs(model_table: pd.DataFrame) -> None:
    summary_lines = [
        "- Track A hiện có workflow mô hình hóa riêng cùng các artifact thống kê và đánh giá",
        f"- Bộ output mới nhất của Track A nằm trong `{REPORT_DIR.as_posix()}/` và `{MODEL_DIR.as_posix()}/`",
        "",
        "## Workflow Track A",
        "",
        "Chạy workflow phân tích và mô hình hóa của Track A:",
        "",
        "```bash",
        "python -m src.track_a.main",
        "```",
        "",
        "Các đầu ra chính:",
        "",
        "- `reports/track_a/track_a_final_report.md`",
        "- `reports/track_a/track_a_slide_deck.md`",
        "- `reports/track_a/track_b_dependency_report.md`",
        "- `reports/track_a/model_comparison.csv`",
        "- `reports/track_a/statistical_tests.csv`",
        "- `reports/track_a/figures/`",
        "- `reports/track_a/models/`",
        "",
        "## Bảng so sánh mô hình Track A hiện tại",
        "",
        markdown_table(model_table, ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "threshold"]),
    ]
    readme_path = Path("README.md")
    current = readme_path.read_text(encoding="utf-8")
    marker = "## Ghi chú"
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
        roc_auc = float(roc_auc_score(y_test, scores))
        pr_auc = float(average_precision_score(y_test, scores))
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
            "accuracy": metrics["accuracy"],
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
    base_auc = float(roc_auc_score(y_test[sample_idx], predict_positive_class(best_model, X_test[sample_idx])))
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
