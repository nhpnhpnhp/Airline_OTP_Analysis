# Slide Presentation — Outline (10-15 slides)

---

## Slide 1: Trang bìa
- Tên đề tài: "Phân tích hiệu suất đúng giờ trong vận tải hàng không Hoa Kỳ tháng 1 giai đoạn 2021-2025 và dự đoán trễ chuyến"
- Tên SV, MSSV, lớp, GVHD
- Logo trường

## Slide 2: Mục tiêu & Phạm vi
- 5 mục tiêu chính (OTP analysis, delay decomposition, risk analysis, ML prediction, dashboard)
- Nguồn dữ liệu: BTS | 500K+ records | Jan 2021–2025

## Slide 3: Dữ liệu & Tiền xử lý
- Nguồn data, data quality summary
- Cleaning rules (cancelled/diverted, HHMM conversion)
- Infographic: 500K+ flights → cleaned → 2 dataset splits

## Slide 4: OTP Overview
- KPI cards: Overall OTP, Avg Delay, Cancel Rate
- Bar chart: OTP by Year (recovery trend)
- Key insight: 1 câu tóm tắt xu hướng

## Slide 5: OTP theo Hãng bay
- Top 5 & Bottom 5 carriers
- Chart: Horizontal bar chart with OTP %
- Key insight: hãng nào nổi bật

## Slide 6: OTP theo Sân bay & Tuyến bay
- Top airports, worst airports
- 2-3 chart: origin OTP, route OTP
- Key insight

## Slide 7: Heatmap — Time Pattern
- Year × Time Block heatmap (OTP %)
- Key insight: hiệu ứng buổi chiều tối

## Slide 8: Nguyên nhân trễ
- Stacked bar: delay causes by year
- Pie chart: proportion by cause
- Key insight: Carrier Delay + Late Aircraft chiếm đa số

## Slide 9: Cancellation & Operational Risk
- Cancel rate by year
- Top cancellation reasons
- Airport-level cancellation hotspots

## Slide 10: Phân tích thống kê
- 1-2 hypothesis test results (chi-square, Kruskal-Wallis)
- p-value, effect size interpretation
- Kết luận: khung giờ ảnh hưởng có ý nghĩa thống kê

## Slide 11: ML — Feature Engineering & Design
- Track A vs Track B (diagram)
- Feature list, leakage prevention
- Temporal split: Train 2021-2024, Test 2025

## Slide 12: ML — Model Results
- Bảng so sánh 3 models × 2 tracks
- ROC curves (Track A vs Track B)
- Key metric: ROC-AUC, F1

## Slide 13: ML — Interpretability
- SHAP summary plot (top features)
- Permutation importance
- Key insight: feature nào drive prediction

## Slide 14: Dashboard Demo
- Screenshot dashboard (Overview page)
- Screenshot Predict Delay page
- Hướng dẫn truy cập

## Slide 15: Kết luận & Hướng phát triển
- 4-5 key findings
- 3 khuyến nghị
- Hạn chế & hướng mở rộng
- Q&A
