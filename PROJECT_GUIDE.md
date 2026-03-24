# PROJECT GUIDE

## Muc tieu hien tai

Phien ban repo hien tai duoc chot scope nhu sau:

1. Xay dung preprocessing pipeline on dinh cho du lieu BTS January 2021-2025
2. Tao processed datasets va quality report
3. Gom EDA thanh notebook exploratory
4. Giu cac notebook phan tich thu nghiem trong `archive/exploratory/`

Nhung phan sau duoc xem la chua bat dau hoac chua dua vao scope chinh:

- Feature engineering notebook theo huong cu
- Modeling / evaluation
- Dashboard
- Slide deck

## Trang thai repo sau khi don dep

- Giu `src/step1_data_cleaning/main.py` lam diem vao preprocessing chinh
- Gop nhom EDA script cu thanh `archive/exploratory/eda_overview.ipynb`
- Chuyen `delay_analysis.ipynb` va `risk_analysis.ipynb` thanh notebook exploratory
- Xoa notebook cu va cac file source ML/dashboard chua thuc hien

## Checklist uu tien tiep theo

### Hoan thien phan da co

- [x] Thu thap raw CSV 2021-2025
- [x] Tao pipeline preprocessing partitioned parquet
- [x] Tao quality report
- [x] Co notebook exploratory tong hop cho EDA
- [x] Tach notebook exploratory khoi source chinh
- [x] Xoa file stale / conflict-prone

### Viec nen lam tiep

- [ ] Chuan hoa them naming cho hinh exploratory trong `reports/figures/`
- [ ] Viet bao cao thuc te dua tren artifact hien co
- [ ] Neu bat dau modeling lai, tao module moi tu dau theo cau truc moi
- [ ] Neu bat dau dashboard lai, tao code moi tu output hien co thay vi tai su dung skeleton cu

## Quy uoc to chuc file

- Official source: `src/`
- Exploratory notebooks: `archive/exploratory/`
- Bao cao va artifact: `reports/`
- Du lieu raw/processed: `data/`
- Khong dat file debug, cache, draft ML cu vao luong chinh

## Ghi chu

`data/processed/ml_track_a/` va `data/processed/ml_track_b/` hien la artifact preprocessing da duoc tao ra truoc do. Chung khong duoc xem la bang chung rang phan modeling da hoan thanh.
