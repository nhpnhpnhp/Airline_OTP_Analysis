# Airline OTP Analysis

Project mon hoc phan tich du lieu On-Time Performance cua van tai hang khong Hoa Ky trong thang 1 giai doan 2021-2025.

## Pham vi hien tai

Repo hien tap trung vao 4 phan da co artifact ro rang:

- Thu thap va to chuc du lieu raw BTS cho 2021-2025
- Tien xu ly va lam sach du lieu thanh cac tap parquet phan vung
- Exploratory analysis va risk analysis trong `archive/exploratory/`
- Statistical analysis va predictive modeling cho Track A trong `src/track_a/`, sinh artifact vao `reports/track_a/`

Nhung phan chua duoc xem la hoan thien:

- Track B modeling
- Cross-track comparison Track A vs Track B
- Dashboard
- Final presentation version

## Cau truc repo

```text
Airline_OTP_Analysis/
|-- archive/
|   `-- exploratory/
|-- data/
|   |-- raw/
|   `-- processed/
|       |-- clean_full/
|       |-- clean_operated/
|       |-- mappings/
|       |-- ml_track_a/
|       `-- ml_track_b/
|-- reports/
|   |-- figures/
|   `-- track_a/
|-- src/
|   |-- step1_data_cleaning/
|   `-- track_a/
|-- PROJECT_GUIDE.md
|-- README.md
`-- requirements.txt
```

## Luong chay chinh

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

### 2. Track A workflow

```bash
python -m src.track_a.main
```

Workflow nay hien:

- Doc du lieu Track A da duoc preprocessing
- Chay statistical tests cho cac bien phan tich
- Train va danh gia `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`
- Tinh metric va ve figure bang `scikit-learn`
- Chay statistical tests bang `scipy`
- Sinh report, slide markdown va model artifact vao `reports/track_a/`

Main outputs:

- `reports/track_a/track_a_final_report.md`
- `reports/track_a/track_a_slide_deck.md`
- `reports/track_a/track_b_dependency_report.md`
- `reports/track_a/model_comparison.csv`
- `reports/track_a/permutation_importance.csv`
- `reports/track_a/statistical_tests.csv`
- `reports/track_a/figures/`
- `reports/track_a/models/`

## Current Track A model comparison

| model | roc_auc | pr_auc | accuracy | precision | recall | f1 | threshold | tp | tn | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6165 | 0.2697 | 0.2190 | 0.1916 | 0.9802 | 0.3205 | 0.4800 | 96191 | 18199 | 405940 | 1939 |
| Gradient Boosting | 0.6020 | 0.2416 | 0.4379 | 0.2187 | 0.7743 | 0.3411 | 0.2000 | 75978 | 152719 | 271420 | 22152 |
| Random Forest | 0.5980 | 0.2490 | 0.4522 | 0.2179 | 0.7395 | 0.3366 | 0.5000 | 72566 | 163607 | 260532 | 25564 |

- Track A hiện có workflow mô hình hóa riêng cùng các artifact thống kê và đánh giá
- Bộ output mới nhất của Track A nằm trong `reports/track_a/` và `reports/track_a/models/`

## Workflow Track A

Chạy workflow phân tích và mô hình hóa của Track A:

```bash
python -m src.track_a.main
```

Các đầu ra chính:

- `reports/track_a/track_a_final_report.md`
- `reports/track_a/track_a_slide_deck.md`
- `reports/track_a/track_b_dependency_report.md`
- `reports/track_a/model_comparison.csv`
- `reports/track_a/statistical_tests.csv`
- `reports/track_a/figures/`
- `reports/track_a/models/`

## Bảng so sánh mô hình Track A hiện tại

| model | roc_auc | pr_auc | accuracy | precision | recall | f1 | threshold | tp | tn | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6165 | 0.2698 | 0.2190 | 0.1916 | 0.9802 | 0.3205 | 0.4800 | 96191 | 18186 | 405953 | 1939 |
| Gradient Boosting | 0.6010 | 0.2512 | 0.4122 | 0.2134 | 0.7922 | 0.3362 | 0.2000 | 77739 | 137555 | 286584 | 20391 |
| Random Forest | 0.5959 | 0.2455 | 0.4519 | 0.2174 | 0.7376 | 0.3358 | 0.5000 | 72376 | 163630 | 260509 | 25754 |

- Track A hiện có workflow mô hình hóa riêng cùng các artifact thống kê và đánh giá
- Bộ output mới nhất của Track A nằm trong `reports/track_a/` và `reports/track_a/models/`

## Workflow Track A

Chạy workflow phân tích và mô hình hóa của Track A:

```bash
python -m src.track_a.main
```

Các đầu ra chính:

- `reports/track_a/track_a_final_report.md`
- `reports/track_a/track_a_slide_deck.md`
- `reports/track_a/track_b_dependency_report.md`
- `reports/track_a/model_comparison.csv`
- `reports/track_a/statistical_tests.csv`
- `reports/track_a/figures/`
- `reports/track_a/models/`

## Bảng so sánh mô hình Track A hiện tại

| model | roc_auc | pr_auc | accuracy | precision | recall | f1 | threshold | tp | tn | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6165 | 0.2698 | 0.2190 | 0.1916 | 0.9802 | 0.3205 | 0.4800 | 96191 | 18186 | 405953 | 1939 |
| Gradient Boosting | 0.6010 | 0.2512 | 0.4122 | 0.2134 | 0.7922 | 0.3362 | 0.2000 | 77739 | 137555 | 286584 | 20391 |
| Random Forest | 0.5959 | 0.2455 | 0.4519 | 0.2174 | 0.7376 | 0.3358 | 0.5000 | 72376 | 163630 | 260509 | 25754 |

## Ghi chú

- Exploratory notebooks duoc giu trong `archive/exploratory/` thay vi nam trong luong source chinh.
- Track A da co workflow rieng trong `src/track_a/` va dang la nhanh modeling hoan chinh nhat cua repo.
- Track B hien chi moi dung o muc artifact preprocessing, chua duoc xem la workflow modeling da hoan tat.

- Track A modeling branch with statistical analysis and evaluation artifacts
- Best available Track A workflow outputs in `reports/track_a/` and `reports/track_a/models/`

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

| model | roc_auc | pr_auc | accuracy | precision | recall | f1 | threshold | tp | tn | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6165 | 0.2698 | 0.2190 | 0.1916 | 0.9802 | 0.3205 | 0.4800 | 96191 | 18186 | 405953 | 1939 |
| Gradient Boosting | 0.6010 | 0.2512 | 0.4122 | 0.2134 | 0.7922 | 0.3362 | 0.2000 | 77739 | 137555 | 286584 | 20391 |
| Random Forest | 0.5959 | 0.2455 | 0.4519 | 0.2174 | 0.7376 | 0.3358 | 0.5000 | 72376 | 163630 | 260509 | 25754 |
