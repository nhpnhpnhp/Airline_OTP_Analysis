"""
data_loader.py — Module tải và merge dữ liệu On-Time Performance từ BTS.
==========================================================================

Hỗ trợ 2 cách:
  1. Tải thủ công: đặt CSV vào data/raw/, gọi load_and_merge()
  2. (Optional) Download tự động qua BTS form submission

Output: file parquet tổng hợp tại data/processed/flights_jan_2021_2025.parquet
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CẤU HÌNH
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

YEARS = [2021, 2022, 2023, 2024, 2025]
MONTH = 1  # January

# Các cột cần tải (có thể mở rộng)
REQUIRED_COLUMNS = [
    "YEAR", "MONTH", "FL_DATE",
    "OP_UNIQUE_CARRIER", "OP_CARRIER_AIRLINE_ID", "OP_CARRIER", "OP_CARRIER_FL_NUM",
    "ORIGIN_AIRPORT_ID", "ORIGIN", "ORIGIN_CITY_NAME",
    "DEST_AIRPORT_ID", "DEST", "DEST_CITY_NAME",
    "CRS_DEP_TIME", "DEP_TIME", "DEP_DELAY", "DEP_DEL15", "DEP_TIME_BLK",
    "TAXI_OUT",
    "WHEELS_OFF", "WHEELS_ON",
    "TAXI_IN",
    "CRS_ARR_TIME", "ARR_TIME", "ARR_DELAY", "ARR_DEL15", "ARR_TIME_BLK",
    "CANCELLED", "CANCELLATION_CODE", "DIVERTED",
    "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME", "DISTANCE",
    "DISTANCE_GROUP",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
    "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
]


# ============================================================
# HÀM CHÍNH
# ============================================================

def find_raw_csvs(raw_dir: Path = RAW_DIR) -> list[Path]:
    """
    Tìm tất cả file CSV trong thư mục raw.
    Hỗ trợ pattern:
      - T_ONTIME_REPORTING_*.csv (mặc định từ BTS)
      - jan_<year>.csv (do user đặt tên)
      - Bất kỳ *.csv nào nằm trong raw/
    """
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Không tìm thấy file CSV nào trong {raw_dir}.\n"
            f"Hãy tải dữ liệu từ BTS và đặt vào thư mục data/raw/"
        )
    print(f"📂 Tìm thấy {len(csv_files)} file CSV:")
    for f in csv_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   • {f.name} ({size_mb:.1f} MB)")
    return csv_files


def load_single_csv(filepath: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """
    Đọc 1 file CSV, tự động detect encoding.
    Nếu usecols được chỉ định, chỉ đọc các cột đó (cột không tồn tại sẽ bị bỏ qua).
    """
    # Thử đọc header trước để biết cột nào có
    sample = pd.read_csv(filepath, nrows=0)
    available_cols = set(sample.columns.str.strip())

    if usecols:
        cols_to_load = [c for c in usecols if c in available_cols]
        missing = set(usecols) - available_cols
        if missing:
            print(f"   ⚠️  Cột không có trong {filepath.name}: {missing}")
    else:
        cols_to_load = None

    df = pd.read_csv(
        filepath,
        usecols=cols_to_load,
        low_memory=False,
        dtype={
            "CANCELLATION_CODE": str,
            "OP_UNIQUE_CARRIER": str,
            "OP_CARRIER": str,
            "ORIGIN": str,
            "DEST": str,
            "DEP_TIME_BLK": str,
            "ARR_TIME_BLK": str,
        },
    )
    # Strip whitespace từ tên cột
    df.columns = df.columns.str.strip()
    return df


def load_and_merge(
    raw_dir: Path = RAW_DIR,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load tất cả CSV trong raw_dir, merge thành 1 DataFrame.
    Chuẩn hóa schema (cột thiếu → NaN).
    """
    csv_files = find_raw_csvs(raw_dir)
    if usecols is None:
        usecols = REQUIRED_COLUMNS

    dfs = []
    for f in csv_files:
        print(f"\n📥 Đang đọc {f.name}...")
        df = load_single_csv(f, usecols=usecols)
        print(f"   ✅ {len(df):,} dòng, {len(df.columns)} cột")
        dfs.append(df)

    # Merge & chuẩn hóa
    merged = pd.concat(dfs, ignore_index=True, sort=False)

    # Đảm bảo tất cả cột required đều có (điền NaN nếu thiếu)
    for col in usecols:
        if col not in merged.columns:
            merged[col] = np.nan

    # Lọc chỉ tháng 1
    if "MONTH" in merged.columns:
        merged = merged[merged["MONTH"] == MONTH].copy()

    # Lọc chỉ các năm target
    if "YEAR" in merged.columns:
        merged = merged[merged["YEAR"].isin(YEARS)].copy()

    merged = merged.reset_index(drop=True)
    print(f"\n✅ Tổng sau merge & filter: {len(merged):,} bản ghi")
    print(f"   Năm: {sorted(merged['YEAR'].unique())}")
    return merged


def save_parquet(
    df: pd.DataFrame,
    filename: str = "flights_jan_2021_2025.parquet",
    output_dir: Path = PROCESSED_DIR,
) -> Path:
    """Lưu DataFrame thành file Parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    df.to_parquet(output_path, engine="pyarrow", index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"💾 Đã lưu: {output_path} ({size_mb:.1f} MB)")
    return output_path


def load_processed(
    filename: str = "flights_jan_2021_2025.parquet",
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Đọc file parquet đã xử lý."""
    filepath = processed_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Chưa có file processed. Chạy load_and_merge() trước.\n"
            f"Expected: {filepath}"
        )
    df = pd.read_parquet(filepath, engine="pyarrow")
    print(f"📖 Đã đọc {len(df):,} bản ghi từ {filepath.name}")
    return df


# ============================================================
# DATA DICTIONARY
# ============================================================
DATA_DICTIONARY = {
    "YEAR": "Năm (2021–2025)",
    "MONTH": "Tháng (luôn = 1)",
    "FL_DATE": "Ngày bay (YYYY-MM-DD)",
    "OP_UNIQUE_CARRIER": "Mã hãng bay duy nhất",
    "OP_CARRIER": "Mã IATA hãng bay",
    "OP_CARRIER_FL_NUM": "Số hiệu chuyến bay",
    "ORIGIN_AIRPORT_ID": "ID sân bay đi (số)",
    "ORIGIN": "Mã IATA sân bay đi",
    "ORIGIN_CITY_NAME": "Tên thành phố đi",
    "DEST_AIRPORT_ID": "ID sân bay đến (số)",
    "DEST": "Mã IATA sân bay đến",
    "DEST_CITY_NAME": "Tên thành phố đến",
    "CRS_DEP_TIME": "Giờ khởi hành theo lịch (HHMM, local)",
    "DEP_TIME": "Giờ khởi hành thực tế (HHMM, local)",
    "DEP_DELAY": "Phút trễ khởi hành (âm = sớm)",
    "DEP_DEL15": "1 nếu DEP_DELAY >= 15 phút, 0 ngược lại",
    "DEP_TIME_BLK": "Khung giờ khởi hành (e.g., '0600-0659')",
    "TAXI_OUT": "Thời gian taxi-out (phút)",
    "TAXI_IN": "Thời gian taxi-in (phút)",
    "CRS_ARR_TIME": "Giờ đến theo lịch (HHMM, local)",
    "ARR_TIME": "Giờ đến thực tế (HHMM, local)",
    "ARR_DELAY": "Phút trễ đến (âm = sớm)",
    "ARR_DEL15": "1 nếu ARR_DELAY >= 15 phút, 0 ngược lại. ĐÂY LÀ TARGET CHÍNH.",
    "ARR_TIME_BLK": "Khung giờ đến (e.g., '1200-1259')",
    "CANCELLED": "1 nếu chuyến bay bị hủy",
    "CANCELLATION_CODE": "Lý do hủy: A=Carrier, B=Weather, C=NAS, D=Security",
    "DIVERTED": "1 nếu chuyến bay bị chuyển hướng",
    "CRS_ELAPSED_TIME": "Thời gian bay theo lịch (phút)",
    "ACTUAL_ELAPSED_TIME": "Thời gian bay thực tế (phút)",
    "AIR_TIME": "Thời gian trên không (phút)",
    "DISTANCE": "Khoảng cách bay (dặm)",
    "DISTANCE_GROUP": "Nhóm khoảng cách (1–11)",
    "CARRIER_DELAY": "Phút trễ do hãng bay",
    "WEATHER_DELAY": "Phút trễ do thời tiết",
    "NAS_DELAY": "Phút trễ do hệ thống không lưu",
    "SECURITY_DELAY": "Phút trễ do an ninh",
    "LATE_AIRCRAFT_DELAY": "Phút trễ do máy bay từ chuyến trước đến muộn",
}


def print_data_dictionary():
    """In data dictionary ra console."""
    print("=" * 70)
    print("DATA DICTIONARY — BTS On-Time Performance")
    print("=" * 70)
    for col, desc in DATA_DICTIONARY.items():
        print(f"  {col:<28s} {desc}")
    print("=" * 70)


# ============================================================
# QUALITY CHECKS
# ============================================================

def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo báo cáo chất lượng dữ liệu:
    - Count, missing %, dtype, nunique, sample values
    """
    report = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null": df.count(),
        "null_count": df.isnull().sum(),
        "null_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "nunique": df.nunique(),
        "sample": df.iloc[0] if len(df) > 0 else None,
    })
    return report.sort_values("null_pct", ascending=False)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BTS On-Time Performance Data Loader")
    print("=" * 60)
    print_data_dictionary()

    try:
        df = load_and_merge()
        report = quality_report(df)
        print("\n📊 QUALITY REPORT:")
        print(report.to_string())

        output = save_parquet(df)
        print(f"\n✅ Pipeline hoàn tất! File: {output}")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n📖 Hướng dẫn tải dữ liệu:")
        print("   1. Truy cập https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoession_VQ=FGJ")
        print("   2. Chọn Year, Month = January")
        print("   3. Tick tất cả các field cần thiết")
        print("   4. Download CSV → đặt vào data/raw/")
