# ✈️ Phân tích hiệu suất đúng giờ trong vận tải hàng không Hoa Kỳ (Tháng 1, 2021–2025) & Dự đoán trễ chuyến

> **Đồ án môn Phân tích dữ liệu** — End-to-end Data Analytics & Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Data](https://img.shields.io/badge/Data-BTS%20Transtats-orange)

---

## 📋 Mục tiêu

| # | Mục tiêu | Mô tả |
|---|---------|--------|
| 1 | **OTP Analysis** | Đo lường & phân tích On-Time Performance theo năm, hãng bay, sân bay, tuyến, khung giờ |
| 2 | **Delay Decomposition** | Phân rã nguyên nhân trễ (Carrier / Weather / NAS / Security / Late Aircraft) |
| 3 | **Risk Analysis** | Phân tích chuyến bị hủy (Cancelled) và chuyển hướng (Diverted) |
| 4 | **Delay Prediction** | Xây dựng mô hình ML dự đoán trễ chuyến (Track A: pre-flight, Track B: post-pushback) |
| 5 | **Dashboard** | Streamlit dashboard tương tác với KPI, biểu đồ, và demo prediction |

## 📂 Cấu trúc dự án

```
airline-otp-analysis/
├── README.md                          # Mô tả dự án (file này)
├── requirements.txt                   # Dependencies
├── .gitignore
│
├── data/
│   ├── raw/                           # CSV gốc từ BTS (KHÔNG commit)
│   └── processed/                     # Parquet đã xử lý
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb        # Tải & merge dữ liệu
│   ├── 02_cleaning_eda.ipynb          # Cleaning + EDA + Visualization
│   ├── 03_feature_engineering.ipynb   # Feature engineering cho ML
│   ├── 04_model_trackA.ipynb          # Track A: Pre-flight prediction
│   └── 05_model_trackB.ipynb         # Track B: Post-pushback prediction
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Load & merge raw data
│   ├── cleaner.py                     # Data cleaning pipeline
│   ├── features.py                    # Feature engineering
│   ├── train.py                       # Model training utilities
│   └── evaluate.py                    # Evaluation metrics & plots
│
├── models/                            # Saved models (.joblib)
│
├── dashboard/
│   └── app.py                         # Streamlit dashboard
│
└── reports/
    ├── figures/                        # Exported charts
    ├── report_outline.md              # Cấu trúc báo cáo
    └── slide_outline.md               # Cấu trúc slide
```

## 📊 Dữ liệu

- **Nguồn**: [Bureau of Transportation Statistics (BTS)](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoession_VQ=FGJ&QO_fu146_guvf=D)
- **Bảng**: On-Time Reporting Carrier On-Time Performance (1987–present)
- **Phạm vi**: Tháng 1 (January) của 2021, 2022, 2023, 2024, 2025
- **Quy mô**: ~500,000+ bản ghi
- **Định nghĩa OTP**: Chuyến bay đúng giờ nếu `ARR_DELAY ≤ 15 phút` (hoặc `ARR_DEL15 = 0`)

## 🚀 Cài đặt & Chạy

### 1. Clone repo
```bash
git clone <repo-url>
cd airline-otp-analysis
```

### 2. Tạo virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Tải dữ liệu
Tham khảo hướng dẫn chi tiết trong `notebooks/01_data_ingestion.ipynb`

**Cách 1 — Tải thủ công (UI):**
1. Truy cập [BTS Download Page](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoession_VQ=FGJ&QO_fu146_guvf=D)
2. Chọn Year → Month = January → Download → Lưu vào `data/raw/`
3. Lặp lại cho 2021–2025

**Cách 2 — Script tự động:**
```bash
python src/data_loader.py
```

### 4. Chạy notebooks
Mở lần lượt các notebook trong `notebooks/` trên Jupyter / Google Colab.

### 5. Chạy dashboard
```bash
streamlit run dashboard/app.py
```

## 🤖 Modeling

| Track | Mô tả | Features | Target |
|-------|--------|----------|--------|
| **A** | Pre-flight (trước giờ bay) | Carrier, Origin, Dest, CRS times, Distance, Day of week, Year... | `ARR_DEL15` |
| **B** | Post-pushback (sau pushback) | Tất cả Track A + DEP_DELAY, DEP_DEL15, TAXI_OUT... | `ARR_DEL15` |

### Mô hình so sánh
1. **Logistic Regression** (Baseline)
2. **Random Forest**
3. **XGBoost** / LightGBM

### Đánh giá
- ROC-AUC, PR-AUC, F1-score
- Confusion Matrix
- Temporal validation: Train 2021–2024, Test 2025
- SHAP / Permutation Importance

## 📝 License

MIT License — Free for academic use.
