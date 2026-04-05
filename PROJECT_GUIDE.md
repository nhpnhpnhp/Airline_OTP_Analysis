# PROJECT GUIDE

## Muc tieu hien tai

Phien ban repo hien tai duoc chot scope nhu sau:

1. Xay dung preprocessing pipeline on dinh cho du lieu BTS January 2021-2025
2. Tao processed datasets va quality report
3. Gom EDA thanh notebook exploratory
4. Giu cac notebook phan tich thu nghiem trong `archive/exploratory/`
5. Duy tri notebook modeling rieng cho Track A va Track B trong `code/`

Nhung phan sau duoc xem la chua bat dau hoac chua dua vao scope chinh:

- Cross-track comparison day du
- Dashboard
- Slide deck

## Trang thai repo sau khi don dep

- Giu `reports/processing.ipynb` lam workflow preprocessing chinh
- Gop nhom EDA script cu thanh notebook exploratory
- Giu `code/track_a.ipynb` va `code/track_b.ipynb` lam notebook modeling
- Xoa workflow Track A cu trong `code/track_a/`
- Xoa bao cao cu trong `reports/track_a/`

## Checklist uu tien tiep theo

- [x] Thu thap raw CSV 2021-2025
- [x] Tao pipeline preprocessing partitioned parquet
- [x] Tao quality report
- [x] Co notebook exploratory tong hop cho EDA
- [x] Tach notebook exploratory khoi source chinh
- [x] Chuyen Track A sang notebook giong huong trinh bay cua Track B
- [x] Loai bo workflow va bao cao cu cua Track A
- [ ] Tu viet bao cao moi dua tren notebook Track A / Track B
- [ ] Lam cross-track comparison
- [ ] Neu bat dau dashboard lai, tao code moi tu output hien co thay vi tai su dung skeleton cu

## Quy uoc to chuc file

- Official source: `code/`
- Exploratory notebooks: `archive/exploratory/`
- Bao cao: `reports/`
- Du lieu raw/processed: `data/`
- Khong dat file debug, cache, draft ML cu vao luong chinh

## Ghi chu

`data/processed/ml_track_a/` va `data/processed/ml_track_b/` la artifact preprocessing. Trong phien ban hien tai, Track A va Track B duoc trinh bay theo notebook de ban co the tu tong hop bao cao sau.
