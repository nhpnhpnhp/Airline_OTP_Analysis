# Airline OTP Analysis

Project mon hoc Phan tich du lieu ve On-Time Performance cua van tai hang khong Hoa Ky trong thang 1 giai doan 2021-2025.

## Pham vi hien tai

Project hien dang tap trung vao 3 phan da co bang chung thuc thi:

- Thu thap va to chuc du lieu raw BTS cho 2021-2025
- Tien xu ly va lam sach du lieu thanh cac tap parquet phan vung
- Exploratory analysis va risk analysis, da sinh ra hinh trong `reports/figures/`

Nhung phan chua duoc xem la hoan thien trong repo hien tai:

- Modeling / huan luyen mo hinh
- Danh gia mo hinh
- Dashboard
- Slide bao cao cuoi cung

## Cau truc hien tai

```text
Airline_OTP_Analysis/
├── archive/
│   └── exploratory/
│       ├── eda_overview.ipynb
│       ├── delay_analysis.ipynb
│       └── risk_analysis.ipynb
├── data/
│   ├── raw/
│   └── processed/
│       ├── clean_full/
│       ├── clean_operated/
│       ├── mappings/
│       ├── ml_track_a/
│       └── ml_track_b/
├── reports/
│   ├── figures/
│   ├── quality_report.md
│   └── report_outline.md
├── src/
│   └── step1_data_cleaning/
│       ├── main.py
│       ├── pipeline.py
│       ├── transformations.py
│       ├── reporting.py
│       ├── ml_preparation.py
│       ├── ml_config.py
│       ├── utils.py
│       └── config.py
├── PROJECT_GUIDE.md
├── README.md
└── requirements.txt
```

## Quy uoc to chuc

- File chinh thuc dat trong `src/`
- Notebook exploratory dat trong `archive/exploratory/`
- Output phan tich dat trong `reports/figures/`
- Khong giu file debug ca nhan, file cache, hoac notebook ML chua thuc hien trong luong chinh

## Du lieu

- Nguon: Bureau of Transportation Statistics (BTS)
- Tap raw hien co: `data/raw/T_ONTIME_REPORTING_2021.csv` den `data/raw/T_ONTIME_REPORTING_2025.csv`
- Bao cao chat luong preprocessing: `reports/quality_report.md`

## Luong chay de xuat

### 1. Preprocessing

```bash
python -m src.step1_data_cleaning.main
```

Output chinh:

- `data/processed/clean_full/`
- `data/processed/clean_operated/`
- `data/processed/mappings/`
- `data/processed/ml_track_a/`
- `data/processed/ml_track_b/`
- `reports/quality_report.md`

### 2. Exploratory notebooks

Neu can xem cac notebook phan tich thu nghiem:

- `archive/exploratory/eda_overview.ipynb`
- `archive/exploratory/delay_analysis.ipynb`
- `archive/exploratory/risk_analysis.ipynb`

## Luu y

- Thu muc `models/` va `dashboard/` da duoc loai khoi luong chinh vi chua co artifact hoan chinh.
- Cac file notebook cu cho feature engineering / modeling da duoc xoa de tranh conflict voi huong phat trien sau nay.
