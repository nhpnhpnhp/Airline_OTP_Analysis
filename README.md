# Airline OTP Analysis

Project mon hoc phan tich du lieu On-Time Performance cua van tai hang khong Hoa Ky trong thang 1 giai doan 2021-2025.

## Pham vi hien tai

Repo hien tap trung vao 4 phan da co artifact ro rang:

- Thu thap va to chuc du lieu raw BTS cho 2021-2025
- Tien xu ly va lam sach du lieu thanh cac tap parquet phan vung
- Exploratory analysis va risk analysis trong `archive/exploratory/`
- Notebook modeling cho Track A va Track B trong `src/`

Nhung phan chua duoc xem la hoan thien:

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
|   `-- quality_report.md
|-- src/
|   |-- step1_data_cleaning/
|   |-- track_a.ipynb
|   `-- track_b.ipynb
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

### 2. Track A notebook

Mo va chay `src/track_a.ipynb`.

Notebook nay hien:

- Doc du lieu `ml_track_a`
- Hien danh sach feature
- Ve heatmap tuong quan
- In tuong quan voi bien muc tieu `ARR_DEL15`
- Train va danh gia `LogisticRegression` va `LightGBM`
- Hien confusion matrix, ROC curve va feature importance
- Hien bang so sanh chi so giua cac mo hinh

### 3. Track B notebook

Mo va chay `src/track_b.ipynb`.

## Ghi chu

- `reports/track_a/` da duoc loai bo vi do la bo bao cao cu sinh tu workflow Track A truoc day.
- Hien tai Track A va Track B deu duoc trinh bay theo huong notebook de de tu viet bao cao sau.
- `data/processed/ml_track_a/` va `data/processed/ml_track_b/` van la artifact preprocessing chinh cho modeling.
