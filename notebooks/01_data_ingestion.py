# %% [markdown]
# # 📥 01 — Data Ingestion
# **Mục tiêu**: Tải dữ liệu On-Time Performance từ BTS, merge 5 năm (Jan 2021–2025), lưu Parquet.

# %% [markdown]
# ## 1.1 Cài đặt (Google Colab)

# %%
# Uncomment nếu dùng Google Colab
# !pip install pyarrow pandas numpy tqdm -q

# %% [markdown]
# ## 1.2 Hướng dẫn tải dữ liệu từ BTS
#
# ### Cách 1: Tải thủ công (Khuyến nghị)
# 1. Truy cập: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoession_VQ=FGJ&QO_fu146_guvf=D
# 2. **Filter Geography**: All
# 3. **Year**: Chọn lần lượt 2021, 2022, 2023, 2024, 2025
# 4. **Period**: January
# 5. **Fields**: Tick chọn tất cả các trường:
#    - Time Period: YEAR, QUARTER, MONTH, DAY_OF_MONTH, DAY_OF_WEEK, FL_DATE
#    - Airline: OP_UNIQUE_CARRIER, OP_CARRIER_AIRLINE_ID, OP_CARRIER, OP_CARRIER_FL_NUM
#    - Origin: ORIGIN_AIRPORT_ID, ORIGIN, ORIGIN_CITY_NAME, ORIGIN_STATE_ABR
#    - Destination: DEST_AIRPORT_ID, DEST, DEST_CITY_NAME, DEST_STATE_ABR
#    - Departure: CRS_DEP_TIME, DEP_TIME, DEP_DELAY, DEP_DEL15, DEP_TIME_BLK
#    - Taxi: TAXI_OUT, WHEELS_OFF, WHEELS_ON, TAXI_IN
#    - Arrival: CRS_ARR_TIME, ARR_TIME, ARR_DELAY, ARR_DEL15, ARR_TIME_BLK
#    - Cancellations: CANCELLED, CANCELLATION_CODE
#    - Diversions: DIVERTED
#    - Flight Summaries: CRS_ELAPSED_TIME, ACTUAL_ELAPSED_TIME, AIR_TIME, DISTANCE, DISTANCE_GROUP
#    - Cause of Delay: CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY
# 6. Click Download → Giải nén CSV → Đặt vào data/raw/
#
# ### Cách 2: Google Colab upload
# ```python
# from google.colab import files
# uploaded = files.upload()
# ```
#
# ### Cách 3: Mount Google Drive
# ```python
# from google.colab import drive
# drive.mount('/content/drive')
# ```

# %%
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Cấu hình — điều chỉnh nếu dùng Colab
RAW_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2021, 2022, 2023, 2024, 2025]

print(f"Raw dir: {RAW_DIR.resolve()}")
print(f"Files: {list(RAW_DIR.glob('*.csv'))}")

# %% [markdown]
# ## 1.3 Load & Merge CSV Files

# %%
csv_files = sorted(RAW_DIR.glob("*.csv"))
print(f"Tìm thấy {len(csv_files)} files:")
for f in csv_files:
    size_mb = f.stat().st_size / (1024**2)
    print(f"  • {f.name} ({size_mb:.1f} MB)")

dfs = []
for f in csv_files:
    print(f"\nĐang đọc {f.name}...")
    df = pd.read_csv(f, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  Shape: {df.shape}")
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True, sort=False)
print(f"\n✅ Tổng: {len(df_all):,} bản ghi, {len(df_all.columns)} cột")

# %% [markdown]
# ## 1.4 Schema Verification

# %%
print("Các cột:")
for i, col in enumerate(df_all.columns, 1):
    print(f"  {i:2d}. {col} ({df_all[col].dtype})")

print(f"\nShape: {df_all.shape}")
if 'YEAR' in df_all.columns:
    print(f"\nPhân phối theo năm:\n{df_all['YEAR'].value_counts().sort_index()}")

# %%
# Lọc chỉ tháng 1, 2021-2025
if 'MONTH' in df_all.columns:
    df_all = df_all[df_all['MONTH'] == 1].copy()
if 'YEAR' in df_all.columns:
    df_all = df_all[df_all['YEAR'].isin(YEARS)].copy()
df_all = df_all.reset_index(drop=True)
print(f"Sau filter: {len(df_all):,} bản ghi")

# %% [markdown]
# ## 1.5 Quality Report

# %%
quality = pd.DataFrame({
    'dtype': df_all.dtypes,
    'non_null': df_all.count(),
    'null_count': df_all.isnull().sum(),
    'null_pct': (df_all.isnull().sum() / len(df_all) * 100).round(2),
    'nunique': df_all.nunique(),
})
quality.sort_values('null_pct', ascending=False)

# %%
df_all.head()

# %%
df_all.describe()

# %% [markdown]
# ## 1.6 Lưu Parquet

# %%
output_path = PROCESSED_DIR / "flights_jan_2021_2025.parquet"
df_all.to_parquet(output_path, engine='pyarrow', index=False)

csv_size = sum(f.stat().st_size for f in csv_files) / (1024**2)
parquet_size = output_path.stat().st_size / (1024**2)

print(f"CSV tổng:   {csv_size:.1f} MB")
print(f"Parquet:    {parquet_size:.1f} MB")
print(f"Tiết kiệm: {(1 - parquet_size/csv_size)*100:.0f}%")
print(f"\n✅ Đã lưu: {output_path}")

# %% [markdown]
# ---
# **Next**: `02_cleaning_eda.ipynb` → Data Cleaning & EDA
