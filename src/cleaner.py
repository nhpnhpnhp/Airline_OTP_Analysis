"""
cleaner.py — Data Cleaning Pipeline cho BTS On-Time Performance.
================================================================

Cleaning rules:
1. Tách dataset: flights_operated (CANCELLED=0) vs full_dataset
2. Xử lý missing values cho cancelled/diverted flights
3. Convert HHMM time columns thành minutes-since-midnight
4. Tạo biến thời gian từ FL_DATE
5. Consistency checks
"""

import pandas as pd
import numpy as np
from typing import Tuple


# ============================================================
# CLEANING RULES
# ============================================================

def handle_cancelled_diverted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý chuyến bay hủy/chuyển hướng:
    - Nếu CANCELLED=1: DEP_DELAY, ARR_DELAY, etc. → NaN (giữ nguyên)
    - Nếu DIVERTED=1: ARR_DELAY cũng thường NaN
    - Đặt ARR_DEL15 = NaN cho cancelled/diverted (không phải 'trễ')
    """
    df = df.copy()

    # Cancelled flights: không có thông tin delay thực tế
    mask_cancel = df["CANCELLED"] == 1
    delay_cols = [
        "DEP_TIME", "DEP_DELAY", "DEP_DEL15",
        "ARR_TIME", "ARR_DELAY", "ARR_DEL15",
        "TAXI_OUT", "TAXI_IN", "WHEELS_OFF", "WHEELS_ON",
        "ACTUAL_ELAPSED_TIME", "AIR_TIME",
        "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
        "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    ]
    for col in delay_cols:
        if col in df.columns:
            df.loc[mask_cancel, col] = np.nan

    # Diverted: ARR thông tin không chính xác
    mask_divert = df["DIVERTED"] == 1
    arr_cols = ["ARR_TIME", "ARR_DELAY", "ARR_DEL15", "TAXI_IN"]
    for col in arr_cols:
        if col in df.columns:
            df.loc[mask_divert, col] = np.nan

    n_cancel = mask_cancel.sum()
    n_divert = mask_divert.sum()
    print(f"🔧 Cancelled: {n_cancel:,} ({n_cancel/len(df)*100:.2f}%)")
    print(f"🔧 Diverted:  {n_divert:,} ({n_divert/len(df)*100:.2f}%)")
    return df


def split_datasets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tách 2 dataset:
    1. flights_operated: CANCELLED=0, DIVERTED=0 → cho delay modeling
    2. full_dataset: toàn bộ → cho OTP/cancel analysis

    Returns:
        (flights_operated, full_dataset)
    """
    full = df.copy()
    operated = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)].copy()

    print(f"📊 Full dataset:      {len(full):,} bản ghi")
    print(f"📊 Flights operated:  {len(operated):,} bản ghi")
    print(f"   (Removed {len(full) - len(operated):,} cancelled/diverted)")
    return operated, full


def convert_hhmm_to_minutes(series: pd.Series) -> pd.Series:
    """
    Convert cột HHMM (e.g., 1435 = 14:35) sang phút kể từ 0:00.
    Xử lý edge case: 2400 → 1440 (midnight).
    NaN giữ nguyên.
    """
    s = pd.to_numeric(series, errors="coerce")
    hours = (s // 100).astype("Int64")
    minutes = (s % 100).astype("Int64")

    # Xử lý 2400 → 24*60 = 1440
    result = hours * 60 + minutes
    # Clamp reasonable range
    result = result.where((result >= 0) & (result <= 1440), other=pd.NA)
    return result


def convert_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các cột _MIN (minutes since midnight) từ HHMM columns.
    """
    df = df.copy()
    time_cols = ["CRS_DEP_TIME", "DEP_TIME", "CRS_ARR_TIME", "ARR_TIME"]

    for col in time_cols:
        if col in df.columns:
            new_col = col + "_MIN"
            df[new_col] = convert_hhmm_to_minutes(df[col])
            print(f"   ⏰ {col} → {new_col}")

    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo biến thời gian từ FL_DATE:
    - day_of_week (0=Mon, 6=Sun)
    - day_of_month
    - is_weekend (Sat/Sun)
    - week_of_month
    """
    df = df.copy()

    if "FL_DATE" in df.columns:
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
        df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek       # 0=Mon
        df["DAY_OF_MONTH"] = df["FL_DATE"].dt.day
        df["IS_WEEKEND"] = (df["DAY_OF_WEEK"] >= 5).astype(int)
        df["WEEK_OF_MONTH"] = (df["DAY_OF_MONTH"] - 1) // 7 + 1
        print("   📅 Tạo: DAY_OF_WEEK, DAY_OF_MONTH, IS_WEEKEND, WEEK_OF_MONTH")

    return df


def create_route_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo biến ROUTE = ORIGIN-DEST."""
    df = df.copy()
    if "ORIGIN" in df.columns and "DEST" in df.columns:
        df["ROUTE"] = df["ORIGIN"] + "-" + df["DEST"]
        n_routes = df["ROUTE"].nunique()
        print(f"   ✈️  Tạo ROUTE ({n_routes:,} tuyến)")
    return df


def create_delay_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phân nhóm ARR_DELAY thành categories:
    - 'On-time/Early': ARR_DELAY <= 0
    - 'Slightly Late (1-15)': 0 < ARR_DELAY <= 15
    - 'Late (16-60)': 15 < ARR_DELAY <= 60
    - 'Very Late (61-120)': 60 < ARR_DELAY <= 120
    - 'Extremely Late (>120)': ARR_DELAY > 120
    """
    df = df.copy()
    if "ARR_DELAY" in df.columns:
        bins = [-np.inf, 0, 15, 60, 120, np.inf]
        labels = ["On-time/Early", "Slightly Late (1-15)",
                  "Late (16-60)", "Very Late (61-120)", "Extremely Late (>120)"]
        df["DELAY_CATEGORY"] = pd.cut(df["ARR_DELAY"], bins=bins, labels=labels)
        print("   📊 Tạo DELAY_CATEGORY (5 nhóm)")
    return df


def create_dominant_delay_cause(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xác định nguyên nhân trễ chính (có giá trị lớn nhất):
    CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY
    """
    df = df.copy()
    cause_cols = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
                  "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
    available = [c for c in cause_cols if c in df.columns]

    if available:
        # Lấy cột có giá trị max
        df["DOMINANT_DELAY_CAUSE"] = df[available].idxmax(axis=1)
        # Chỉ gán cho flights thực sự trễ
        mask_no_delay = df[available].sum(axis=1) == 0
        df.loc[mask_no_delay, "DOMINANT_DELAY_CAUSE"] = "No Delay"
        # Clean up column name
        df["DOMINANT_DELAY_CAUSE"] = (
            df["DOMINANT_DELAY_CAUSE"]
            .str.replace("_DELAY", "")
            .str.replace("_", " ")
            .str.title()
        )
        print("   🔍 Tạo DOMINANT_DELAY_CAUSE")
    return df


# ============================================================
# CONSISTENCY CHECKS
# ============================================================

def consistency_checks(df: pd.DataFrame) -> dict:
    """
    Kiểm tra tính nhất quán dữ liệu:
    - YEAR in [2021–2025]
    - MONTH = 1
    - ARR_DEL15 consistent with ARR_DELAY
    - DISTANCE > 0 cho operated flights
    """
    issues = {}

    # Year range
    if "YEAR" in df.columns:
        bad_years = df[~df["YEAR"].isin([2021, 2022, 2023, 2024, 2025])]
        if len(bad_years) > 0:
            issues["bad_years"] = len(bad_years)

    # Month = 1
    if "MONTH" in df.columns:
        bad_months = df[df["MONTH"] != 1]
        if len(bad_months) > 0:
            issues["bad_months"] = len(bad_months)

    # ARR_DEL15 consistency
    if "ARR_DELAY" in df.columns and "ARR_DEL15" in df.columns:
        mask = df["ARR_DELAY"].notna() & df["ARR_DEL15"].notna()
        expected = (df.loc[mask, "ARR_DELAY"] >= 15).astype(int)
        actual = df.loc[mask, "ARR_DEL15"].astype(int)
        inconsistent = (expected != actual).sum()
        if inconsistent > 0:
            issues["arr_del15_inconsistent"] = int(inconsistent)

    # Distance > 0
    if "DISTANCE" in df.columns:
        operated = df[df.get("CANCELLED", 0) == 0]
        zero_dist = (operated["DISTANCE"] <= 0).sum()
        if zero_dist > 0:
            issues["zero_distance"] = int(zero_dist)

    if issues:
        print(f"⚠️  Consistency issues found: {issues}")
    else:
        print("✅ Dữ liệu nhất quán!")

    return issues


# ============================================================
# PIPELINE CHÍNH
# ============================================================

def run_cleaning_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline cleaning đầy đủ.

    Returns:
        (flights_operated, full_dataset) — cả hai đã được clean & engineer basic features
    """
    print("=" * 60)
    print("🧹 DATA CLEANING PIPELINE")
    print("=" * 60)

    # 1. Xử lý cancelled/diverted
    print("\n[1/6] Xử lý cancelled/diverted...")
    df = handle_cancelled_diverted(df)

    # 2. Convert time columns
    print("\n[2/6] Convert thời gian HHMM → minutes...")
    df = convert_time_columns(df)

    # 3. Tạo time features
    print("\n[3/6] Tạo features thời gian...")
    df = create_time_features(df)

    # 4. Tạo route
    print("\n[4/6] Tạo biến ROUTE...")
    df = create_route_feature(df)

    # 5. Tạo delay category & cause
    print("\n[5/6] Tạo biến delay category & cause...")
    df = create_delay_category(df)
    df = create_dominant_delay_cause(df)

    # 6. Consistency checks
    print("\n[6/6] Kiểm tra tính nhất quán...")
    consistency_checks(df)

    # Split datasets
    print("\n" + "=" * 60)
    print("📂 TÁCH DATASET")
    operated, full = split_datasets(df)

    return operated, full


if __name__ == "__main__":
    from data_loader import load_processed

    df = load_processed()
    operated, full = run_cleaning_pipeline(df)

    # Save
    from data_loader import save_parquet
    save_parquet(operated, "flights_operated.parquet")
    save_parquet(full, "flights_full_cleaned.parquet")
