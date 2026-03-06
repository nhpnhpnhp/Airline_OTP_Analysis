"""
features.py — Feature Engineering cho ML Models.
===================================================

Track A (Pre-flight): chỉ features biết trước thời điểm bay.
Track B (Post-pushback): cho phép thêm DEP_DELAY, TAXI_OUT, v.v.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

# Track A: Pre-flight features (KHÔNG dùng bất kỳ info sau giờ bay)
TRACK_A_FEATURES = [
    # Thời gian
    "YEAR",
    "DAY_OF_WEEK",
    "DAY_OF_MONTH",
    "IS_WEEKEND",
    "WEEK_OF_MONTH",
    "CRS_DEP_TIME_MIN",       # Giờ khởi hành dự kiến (phút)
    "CRS_ARR_TIME_MIN",       # Giờ đến dự kiến (phút)
    "CRS_ELAPSED_TIME",       # Thời gian bay dự kiến
    # Tuyến bay
    "DISTANCE",
    "DISTANCE_GROUP",
    # Categorical (sẽ encode)
    "OP_CARRIER_encoded",
    "ORIGIN_encoded",
    "DEST_encoded",
    "DEP_TIME_BLK_encoded",
    # Derived
    "CARRIER_FREQ",           # Tần suất hãng bay
    "ORIGIN_FREQ",            # Tần suất sân bay đi
    "DEST_FREQ",              # Tần suất sân bay đến
    "ROUTE_FREQ",             # Tần suất tuyến bay
    "ORIGIN_HIST_OTP",        # OTP lịch sử của sân bay đi
    "CARRIER_HIST_OTP",       # OTP lịch sử của hãng bay
]

# Track B: Post-pushback features = Track A + thông tin sau khi rời cổng
TRACK_B_EXTRA_FEATURES = [
    "DEP_DELAY",
    "DEP_DEL15",
    "TAXI_OUT",
]

TARGET = "ARR_DEL15"

# Columns that MUST NOT be used as features (leakage)
LEAKAGE_COLUMNS = [
    "ARR_DELAY", "ARR_DEL15", "ARR_TIME", "ARR_TIME_MIN",
    "ACTUAL_ELAPSED_TIME", "AIR_TIME",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
    "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    "WHEELS_OFF", "WHEELS_ON", "TAXI_IN",
    "CANCELLED", "DIVERTED", "CANCELLATION_CODE",
    "DELAY_CATEGORY", "DOMINANT_DELAY_CAUSE",
]


# ============================================================
# ENCODING FUNCTIONS
# ============================================================

def frequency_encode(df: pd.DataFrame, col: str, new_col: str = None) -> pd.DataFrame:
    """
    Frequency encoding: thay giá trị bằng tần suất xuất hiện.
    Phù hợp cho biến high-cardinality (ORIGIN, DEST, CARRIER).
    """
    df = df.copy()
    if new_col is None:
        new_col = col + "_FREQ"
    freq = df[col].value_counts(normalize=True)
    df[new_col] = df[col].map(freq).fillna(0)
    return df


def target_encode(
    df: pd.DataFrame,
    col: str,
    target: str = TARGET,
    smoothing: float = 10.0,
    new_col: str = None,
) -> pd.DataFrame:
    """
    Target encoding (smoothed) cho biến categorical.
    Sử dụng regularization để tránh overfitting.

    formula: encoded = (count * col_mean + smoothing * global_mean) / (count + smoothing)
    """
    df = df.copy()
    if new_col is None:
        new_col = col + "_TE"

    global_mean = df[target].mean()
    agg = df.groupby(col)[target].agg(["mean", "count"])
    agg["smooth"] = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
    df[new_col] = df[col].map(agg["smooth"]).fillna(global_mean)
    return df


def label_encode_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Label encode (chuyển category → số nguyên)."""
    df = df.copy()
    new_col = col + "_encoded"
    le = LabelEncoder()
    # Handle NaN
    mask = df[col].notna()
    df[new_col] = -1
    df.loc[mask, new_col] = le.fit_transform(df.loc[mask, col].astype(str))
    return df


# ============================================================
# HISTORICAL FEATURES (TRUNG BÌNH LỊCH SỬ)
# ============================================================

def compute_historical_otp(
    df: pd.DataFrame,
    group_col: str,
    target: str = TARGET,
    new_col: str = None,
) -> pd.DataFrame:
    """
    Tính OTP lịch sử trung bình (mean ARR_DEL15) cho mỗi group.
    ⚠️ PHẢI dùng chỉ training data để tính → avoid leakage.
    """
    df = df.copy()
    if new_col is None:
        new_col = group_col + "_HIST_OTP"

    hist = df.groupby(group_col)[target].mean()
    df[new_col] = df[group_col].map(hist).fillna(df[target].mean())
    return df


# ============================================================
# FEATURE ENGINEERING PIPELINE
# ============================================================

def engineer_features(df: pd.DataFrame, track: str = "A") -> pd.DataFrame:
    """
    Pipeline tạo features cho Track A hoặc Track B.

    Args:
        df: DataFrame đã clean (flights_operated)
        track: "A" (pre-flight) hoặc "B" (post-pushback)

    Returns:
        DataFrame với features đã engineer
    """
    df = df.copy()
    print(f"\n🔧 Feature Engineering — Track {track}")
    print("=" * 50)

    # 1. Encode categorical columns
    print("[1/4] Encoding categorical columns...")
    for col in ["OP_CARRIER", "ORIGIN", "DEST", "DEP_TIME_BLK"]:
        if col in df.columns:
            df = label_encode_column(df, col)
            print(f"   ✅ {col} → {col}_encoded")

    # 2. Frequency encoding
    print("[2/4] Frequency encoding...")
    for col, new_col in [
        ("OP_CARRIER", "CARRIER_FREQ"),
        ("ORIGIN", "ORIGIN_FREQ"),
        ("DEST", "DEST_FREQ"),
    ]:
        if col in df.columns:
            df = frequency_encode(df, col, new_col)

    if "ROUTE" in df.columns:
        df = frequency_encode(df, "ROUTE", "ROUTE_FREQ")

    # 3. Historical OTP
    print("[3/4] Computing historical OTP...")
    if TARGET in df.columns:
        for col, new_col in [
            ("ORIGIN", "ORIGIN_HIST_OTP"),
            ("OP_CARRIER", "CARRIER_HIST_OTP"),
        ]:
            if col in df.columns:
                df = compute_historical_otp(df, col, TARGET, new_col)

    # 4. Extra features for Track B
    if track == "B":
        print("[4/4] Adding Track B features (post-pushback)...")
        # DEP_DELAY, DEP_DEL15, TAXI_OUT đã có trong df
        # Tạo feature: dep_delay buckets
        if "DEP_DELAY" in df.columns:
            bins = [-np.inf, -5, 0, 15, 30, 60, np.inf]
            labels_dep = ["Very Early", "Early", "On Time", "Slight Delay",
                          "Moderate Delay", "Severe Delay"]
            df["DEP_DELAY_CAT"] = pd.cut(
                df["DEP_DELAY"], bins=bins, labels=labels_dep
            )
            le = LabelEncoder()
            mask = df["DEP_DELAY_CAT"].notna()
            df["DEP_DELAY_CAT_encoded"] = -1
            df.loc[mask, "DEP_DELAY_CAT_encoded"] = le.fit_transform(
                df.loc[mask, "DEP_DELAY_CAT"].astype(str)
            )
    else:
        print("[4/4] Track A — no post-pushback features.")

    print(f"\n✅ Feature engineering hoàn tất. Shape: {df.shape}")
    return df


def get_feature_target_split(
    df: pd.DataFrame,
    track: str = "A",
) -> tuple:
    """
    Lấy X (features) và y (target) cho modeling.
    Đảm bảo KHÔNG có leakage columns.

    Returns:
        (X, y, feature_names)
    """
    # Xác định feature columns
    if track == "A":
        feature_candidates = TRACK_A_FEATURES
    else:
        feature_candidates = TRACK_A_FEATURES + TRACK_B_EXTRA_FEATURES

    # Chỉ giữ các cột thực sự có trong df
    feature_cols = [c for c in feature_candidates if c in df.columns]

    # Double-check: remove any leakage
    feature_cols = [c for c in feature_cols if c not in LEAKAGE_COLUMNS]

    # Remove target from features
    feature_cols = [c for c in feature_cols if c != TARGET]

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    # Drop rows where target is NaN
    mask = y.notna()
    X = X[mask]
    y = y[mask].astype(int)

    # Fill remaining NaN in features
    for col in X.columns:
        if X[col].dtype in ["float64", "float32", "int64", "Int64"]:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(-1)

    print(f"\n📊 Feature-Target Split (Track {track}):")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   y distribution: {y.value_counts().to_dict()}")
    print(f"   Positive rate: {y.mean():.4f}")
    print(f"   Features: {feature_cols}")

    return X, y, feature_cols


def temporal_train_test_split(
    df: pd.DataFrame,
    train_years: list = [2021, 2022, 2023, 2024],
    test_years: list = [2025],
) -> tuple:
    """
    Train/test split theo thời gian (temporal).
    Train: Jan 2021–2024, Test: Jan 2025.
    """
    train = df[df["YEAR"].isin(train_years)].copy()
    test = df[df["YEAR"].isin(test_years)].copy()

    print(f"\n📅 Temporal Split:")
    print(f"   Train: years {train_years} → {len(train):,} rows")
    print(f"   Test:  years {test_years}  → {len(test):,} rows")
    return train, test


if __name__ == "__main__":
    from data_loader import load_processed
    from cleaner import run_cleaning_pipeline

    df = load_processed()
    operated, _ = run_cleaning_pipeline(df)
    df_feat = engineer_features(operated, track="A")
    X, y, features = get_feature_target_split(df_feat, track="A")
