# Airline OTP Analysis

Project mon hoc Phan tich du lieu ve On-Time Performance cua van tai hang khong Hoa Ky trong thang 1 giai doan 2021-2025.

## Pham vi hien tai

Project hien dang tap trung vao 4 phan da co bang chung thuc thi:

- Thu thap va to chuc du lieu raw BTS cho 2021-2025
- Tien xu ly va lam sach du lieu thanh cac tap parquet phan vung
- Exploratory analysis va risk analysis, da sinh ra hinh trong `reports/figures/`
- Statistical analysis va predictive modeling cho Track A, da sinh ra artifact trong `reports/track_a/`

Nhung phan chua duoc xem la hoan thien trong repo hien tai:

- Track B modeling
- Cross-track comparison Track A vs Track B
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
- Output modeling Track A dat trong `reports/track_a/`
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

Track A modeling branch hien da co statistical analysis, evaluation artifact va report rieng trong `reports/track_a/`.

## Track A workflow

Run the Track A analysis and modeling workflow:

```bash
python -m src.track_a.main
```

Main outputs:

- `reports/track_a/track_a_final_report.md`
- `reports/track_a/track_a_slide_deck.md`
- `reports/track_a/track_b_dependency_report.md`
- `reports/track_a/model_comparison.csv`
- `reports/track_a/statistical_tests.csv`
- `reports/track_a/figures/`
- `reports/track_a/models/`

## Current Track A model comparison

| model | roc_auc | pr_auc | precision | recall | f1 | threshold | tp | tn | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6099 | 0.2633 | 0.2310 | 0.6657 | 0.3430 | 0.4400 | 65329 | 206655 | 217484 | 32801 |
| Decision Tree | 0.5945 | 0.2278 | 0.2096 | 0.7354 | 0.3263 | 0.2400 | 72168 | 152072 | 272067 | 25962 |

## Luu y

- Thu muc `dashboard/` van chua duoc dua vao luong chinh.
- Cac file notebook cu cho feature engineering / modeling da duoc xoa de tranh conflict voi huong phat trien sau nay.
- Track A da co workflow rieng trong `src/track_a/`; Track B chua duoc xem la phan da hoan thien.
