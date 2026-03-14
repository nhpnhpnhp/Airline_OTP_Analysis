# ✈️ EXECUTIVE PLAN & PROJECT GUIDE
# US Airline On-Time Performance Analysis — January 2021–2025

---

## A. EXECUTIVE PLAN (1 trang)

**Đề tài**: Phân tích hiệu suất đúng giờ trong VTHK Hoa Kỳ tháng 1 (2021–2025) & Dự đoán trễ chuyến.

**Dữ liệu**: BTS On-Time Reporting (transtats.bts.gov) — ~500K+ bản ghi, 5 năm × tháng 1.

**Pipeline**:
1. **Ingest**: Download CSV từ BTS → merge → parquet
2. **Clean**: Handle cancelled/diverted, HHMM→minutes, time features
3. **EDA**: 20+ biểu đồ OTP, delay, causes, cancellation
4. **Statistics**: Chi-square test, Kruskal-Wallis, effect size
5. **ML**: Track A (pre-flight) + Track B (post-pushback) × 3 models
6. **Dashboard**: Streamlit với filters, KPIs, charts, prediction demo
7. **Report**: ≤100 trang + Slide 15 trang

**Kết quả kỳ vọng**:
- Xu hướng OTP post-COVID recovery
- Identify worst carriers/airports/routes/time blocks
- Delay cause decomposition
- ML prediction: ROC-AUC ≈ 0.65-0.72 (Track A), ≈ 0.90+ (Track B)
- Interactive dashboard

---

## B. TIMELINE & CHECKLIST (5 tuần)

### Tuần 1: Data Ingestion & Quality (Ngày 1–7)
- [ ] Tải CSV từ BTS cho 5 năm (Jan 2021–2025)
- [ ] Merge và chuẩn hóa schema
- [ ] Convert sang Parquet
- [ ] Data dictionary documentation
- [ ] Quality report (missing values, dtypes, ranges)
- [ ] Consistency checks (YEAR, MONTH, ARR_DEL15 logic)
- **Output**: flights_jan_2021_2025.parquet, quality report

### Tuần 2: EDA & Visualization (Ngày 8–14)
- [ ] Cleaning pipeline (cancelled/diverted, HHMM, time features)
- [ ] Tính KPI: OTP overall, by year, by carrier, by airport
- [ ] Vẽ 20 charts (xem Visualization Plan bên dưới)
- [ ] Delay cause decomposition
- [ ] Cancellation & diversion analysis
- [ ] Correlation analysis
- **Output**: 20+ charts trong reports/figures/, cleaned datasets

### Tuần 3: Statistical Analysis & Feature Engineering (Ngày 15–21)
- [ ] Chi-square test: OTP Year A vs Year B
- [ ] Kruskal-Wallis: median delay across time blocks
- [ ] Effect size calculations
- [ ] Driver analysis (lập luận carrier vs route vs time effect)
- [ ] Feature engineering Track A & Track B
- [ ] Leakage audit
- [ ] Temporal train/test split
- **Output**: Statistical test results, engineered datasets

### Tuần 4: Modeling & Evaluation (Ngày 22–28)
- [ ] Train Track A: LogReg, RF, XGBoost/LightGBM
- [ ] Train Track B: cùng 3 models
- [ ] Evaluate: ROC-AUC, F1, PR-AUC, Confusion Matrix
- [ ] Track A vs B comparison
- [ ] Permutation importance
- [ ] SHAP analysis (for boosting model)
- [ ] Drift analysis by year
- [ ] Calibration plot (optional)
- **Output**: Model comparison table, importance plots, saved models

### Tuần 5: Dashboard, Report, Slide (Ngày 29–35)
- [ ] Hoàn thiện Streamlit dashboard
- [ ] Trang Overview: KPIs + 5 charts
- [ ] Trang Predict: demo inference
- [ ] Viết báo cáo (theo outline)
- [ ] Tạo slide (15 slides)
- [ ] Review & polish toàn bộ
- [ ] README final
- **Output**: Dashboard deployed, báo cáo PDF, slide PPT

---

## C. HƯỚNG DẪN INGEST DỮ LIỆU

### Cách 1: Tải thủ công (Khuyến nghị)
1. Mở https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoession_VQ=FGJ&QO_fu146_guvf=D
2. Chọn **Year** = 2021, **Period** = January
3. Tick TẤT CẢ fields (hoặc chọn theo danh sách trong notebook 01)
4. Click **Download** → nhận file ZIP
5. Giải nén → đặt CSV vào `data/raw/jan_2021.csv`
6. Lặp lại cho 2022, 2023, 2024, 2025
7. Chạy notebook `01_data_ingestion.py`

### Cách 2: Google Colab
```python
from google.colab import files
uploaded = files.upload()  # upload 5 file CSV
```

### Lưu trữ & Format
- **Raw CSV**: KHÔNG commit (file lớn) → thêm vào .gitignore
- **Processed Parquet**: tiết kiệm ~60-70% dung lượng, load nhanh hơn
- Convert: `df.to_parquet("output.parquet", engine="pyarrow")`

---

## D. DATA CLEANING RULES (Chi tiết)

| Rule | Mô tả | Code |
|------|--------|------|
| R1 | Cancelled flights: all delay cols → NaN | `df.loc[mask_cancel, delay_cols] = np.nan` |
| R2 | Diverted flights: ARR cols → NaN | `df.loc[mask_divert, arr_cols] = np.nan` |
| R3 | HHMM → minutes since midnight | `hours*60 + minutes` |
| R4 | Tạo DAY_OF_WEEK từ FL_DATE | `df['FL_DATE'].dt.dayofweek` |
| R5 | Tạo IS_WEEKEND | `DAY_OF_WEEK >= 5` |
| R6 | Tạo ROUTE = ORIGIN-DEST | `df['ORIGIN'] + '-' + df['DEST']` |
| R7 | Delay categories (5 nhóm) | `pd.cut(ARR_DELAY, bins)` |
| R8 | Dominant delay cause | `idxmax(cause_cols)` |
| R9 | Tách: operated (cancel=0, divert=0) vs full | filter mask |
| R10 | Encode high-cardinality: frequency/label encoding | `value_counts(normalize=True)` |

---

## E. VISUALIZATION PLAN (20 Charts)

| # | Chart | Type | File | Mục đích |
|---|-------|------|------|----------|
| 1 | OTP by Year | Bar | 01_otp_by_year.png | Xu hướng OTP 2021→2025 |
| 2 | OTP by Carrier | Horizontal Bar | 02_otp_by_carrier.png | Top/bottom carriers |
| 3 | OTP by Origin Airport | Bar | 03_otp_by_origin.png | Airport performance |
| 4 | OTP by Dest Airport | Bar | 04_otp_by_dest.png | Destination effect |
| 5 | OTP by Route | Bar (2 panels) | 05_otp_by_route.png | Busiest + worst routes |
| 6 | OTP Heatmap Year×TimeBlock | Heatmap | 06_otp_heatmap.png | Time pattern |
| 7 | ARR_DELAY Distribution | Histogram | 07_delay_distribution.png | Delay shape & tail |
| 8 | ARR_DELAY Violin by Year | Violin | 08_delay_violin.png | Year comparison |
| 9 | Cancel/Divert Rate by Year | Grouped Bar | 09_cancel_divert_rate.png | Risk trend |
| 10 | Cancellations by Airport | Bar | 10_cancel_by_airport.png | Hotspot airports |
| 11 | Cancellation Reasons | Pie | 11_cancel_reasons.png | A/B/C/D breakdown |
| 12 | Delay Causes by Year | Stacked Bar | 12_delay_causes_year.png | Cause trend |
| 13 | Delay Causes by Carrier | Stacked Bar | 13_delay_causes_carrier.png | Carrier responsibility |
| 14 | OTP vs Distance | Dual-axis | 14_otp_vs_distance.png | Distance effect |
| 15 | Taxi Out vs Delay | Hexbin | 15_taxi_vs_delay.png | Taxi impact |
| 16 | OTP by Day of Week | Bar | 16_otp_by_dow.png | Weekday vs weekend |
| 17 | Daily OTP Trend (Jan) | Line | 17_daily_otp_trend.png | Day-level pattern |
| 18 | Correlation Matrix | Heatmap | 18_correlation_matrix.png | Variable relationships |
| 19 | Avg Delay by Hour | Line + fill | 19_delay_by_hour.png | Hourly pattern |
| 20 | Delay Categories | Pie | 20_delay_categories.png | Category distribution |

---

## F. MODELING SPEC — Track A & Track B

### Track A: Pre-flight (Trước giờ bay)
**Use case**: Hành khách / hãng bay muốn biết xác suất trễ TRƯỚC khi bay.

**Features được phép sử dụng**:
- YEAR, DAY_OF_WEEK, DAY_OF_MONTH, IS_WEEKEND
- CRS_DEP_TIME_MIN, CRS_ARR_TIME_MIN, CRS_ELAPSED_TIME
- DISTANCE, DISTANCE_GROUP
- OP_CARRIER (encoded), ORIGIN (encoded), DEST (encoded)
- DEP_TIME_BLK (encoded)
- CARRIER_FREQ, ORIGIN_FREQ, DEST_FREQ, ROUTE_FREQ
- ORIGIN_HIST_OTP, CARRIER_HIST_OTP

**LEAKAGE — KHÔNG ĐƯỢC dùng**:
- ❌ DEP_TIME, DEP_DELAY, DEP_DEL15
- ❌ ARR_TIME, ARR_DELAY (= target related)
- ❌ TAXI_OUT, TAXI_IN, WHEELS_OFF/ON
- ❌ AIR_TIME, ACTUAL_ELAPSED_TIME
- ❌ Delay causes (CARRIER_DELAY, etc.)
- ❌ CANCELLED, DIVERTED

### Track B: Post-pushback
**Use case**: Điều phối bay, quản lý sân bay (máy bay đã rời cổng).

**Features = Track A + thêm**:
- ✅ DEP_DELAY, DEP_DEL15
- ✅ TAXI_OUT

### Evaluation Protocol
```
┌─────────────────────────────────┐
│ Temporal Split                   │
│ Train: Jan 2021, 2022, 2023, 2024 │
│ Test:  Jan 2025                  │
└─────────────────────────────────┘

Metrics:
  • ROC-AUC (primary)
  • PR-AUC (imbalanced data)
  • F1-score (threshold = 0.5)
  • Confusion Matrix
  • Calibration Curve (optional)

Interpretability:
  • Permutation Importance (all models)
  • SHAP TreeExplainer (XGBoost/LightGBM)

Drift Check:
  • Compute metrics per year separately
  • Plot trends: is 2025 different?
```

---

## G. STATISTICAL TESTS

### Test 1: So sánh OTP giữa 2 năm (Chi-square)
```
H0: Tỷ lệ OTP năm 2021 = Tỷ lệ OTP năm 2025
H1: Tỷ lệ OTP năm 2021 ≠ Tỷ lệ OTP năm 2025

Method: Chi-square test of independence
Table: 2×2 contingency (On-time/Delayed × Year)
Report: χ², p-value, Cramér's V (effect size)
```

### Test 2: Delay across Time Blocks (Kruskal-Wallis)
```
H0: Median ARR_DELAY giống nhau giữa tất cả DEP_TIME_BLK
H1: Ít nhất 1 cặp khác biệt

Method: Kruskal-Wallis H test (non-parametric ANOVA)
Post-hoc: Dunn's test nếu significant
Report: H statistic, p-value, effect size (η²)
```

### Cách trình bày
```python
from scipy.stats import chi2_contingency, kruskal

# Chi-square example
contingency = pd.crosstab(df['YEAR_GROUP'], df['ARR_DEL15'])
chi2, p, dof, expected = chi2_contingency(contingency)
cramers_v = np.sqrt(chi2 / (contingency.sum().sum() * (min(contingency.shape) - 1)))
print(f"χ² = {chi2:.2f}, p = {p:.4f}, Cramér's V = {cramers_v:.4f}")

# Kruskal-Wallis example
groups = [group['ARR_DELAY'].dropna().values 
          for _, group in df.groupby('DEP_TIME_BLK')]
h_stat, p_val = kruskal(*groups)
print(f"H = {h_stat:.2f}, p = {p_val:.4f}")
```

---

## H. RISKS & MITIGATION

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Data size** — CSV quá nặng cho Colab | Medium | High | Dùng Parquet, chọn cột cần thiết, chunk processing |
| **Missing values** — delay causes chỉ có khi trễ >15' | High | Medium | Document rõ, chỉ phân tích subset có data |
| **Leakage** — dùng nhầm feature post-flight cho Track A | High | Critical | Strict feature list, double-check, audit script |
| **Class imbalance** — On-time >> Delayed | High | Medium | balanced class_weight, PR-AUC metric, threshold tuning |
| **Temporal drift** — 2025 khác 2021 (COVID effect) | Medium | Medium | Temporal split, drift analysis, rolling features |
| **High cardinality** — 300+ airports | Medium | Low | Frequency encoding, top-K one-hot, target encoding |
| **Overfit** — RF/XGB overfit on training years | Medium | High | Temporal CV, early stopping, regularization |
| **Colab timeout** — session disconnect | Medium | Medium | Save checkpoints, dùng Parquet cache |

---

## I. APPENDIX — Statistical Test Code Snippets

### Hypothesis Test — Chi-square
```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# So sánh OTP giữa 2021 và 2025
df_compare = df_operated[df_operated['YEAR'].isin([2021, 2025])]
contingency = pd.crosstab(df_compare['YEAR'], df_compare['ARR_DEL15'])
chi2, p_value, dof, expected = chi2_contingency(contingency)

# Effect size: Cramér's V
n = contingency.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

print(f"Chi-square test: χ²={chi2:.2f}, p={p_value:.6f}")
print(f"Effect size (Cramér's V): {cramers_v:.4f}")
if p_value < 0.05:
    print("→ BÁC BỎ H0: OTP 2021 ≠ OTP 2025 (có ý nghĩa thống kê)")
else:
    print("→ KHÔNG BÁC BỎ H0: Chưa đủ bằng chứng kết luận khác biệt")
```

### Hypothesis Test — Kruskal-Wallis
```python
from scipy.stats import kruskal

# So sánh ARR_DELAY giữa các time blocks
groups = []
for block, group in df_operated.groupby('DEP_TIME_BLK'):
    delays = group['ARR_DELAY'].dropna().values
    if len(delays) > 0:
        groups.append(delays)

h_stat, p_val = kruskal(*groups)
# Effect size: epsilon-squared
n_total = sum(len(g) for g in groups)
epsilon_sq = (h_stat - len(groups) + 1) / (n_total - len(groups))

print(f"Kruskal-Wallis: H={h_stat:.2f}, p={p_val:.6f}")
print(f"Effect size (ε²): {epsilon_sq:.4f}")
```
