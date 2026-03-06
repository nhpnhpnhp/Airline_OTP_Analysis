# Báo cáo Đồ án — Outline

## Cấu trúc báo cáo (≤ 100 trang)

---

### Chương 1: Giới thiệu & Mục tiêu (5-8 trang)
- 1.1 Đặt vấn đề: tầm quan trọng OTP trong vận tải hàng không
- 1.2 Mục tiêu nghiên cứu (OTP analysis + delay prediction)
- 1.3 Phạm vi (January 2021–2025, dữ liệu BTS)
- 1.4 Câu hỏi nghiên cứu
  - OTP thay đổi thế nào qua các năm (post-COVID recovery)?
  - Hãng/sân bay/tuyến nào có OTP tốt/xấu nhất?
  - Nguyên nhân trễ chính là gì?
  - Có thể dự đoán trễ trước giờ bay không?
- 1.5 Cấu trúc báo cáo

### Chương 2: Dữ liệu & Tiền xử lý (10-12 trang)
- 2.1 Nguồn dữ liệu (BTS Transtats)
- 2.2 Data dictionary (bảng mô tả các cột)
- 2.3 Thống kê tổng quan (shape, dtypes, missing values)
- 2.4 Data cleaning
  - Xử lý chuyến hủy/chuyển hướng
  - Convert HHMM → minutes
  - Tạo biến phụ (day_of_week, is_weekend, route)
  - Tách dataset: operated vs full
- 2.5 Quality report & consistency checks
- Bảng: Missing value summary, Data quality matrix

### Chương 3: EDA & Insights (20-25 trang)
- 3.1 OTP Overall (theo năm)
  - Chart: Bar chart OTP by Year
  - Insight: xu hướng hồi phục post-COVID
- 3.2 OTP theo Carrier
  - Chart: Top/Bottom carriers, error bars
- 3.3 OTP theo Airport (Origin & Dest)
  - Chart: Top 20 airports
- 3.4 OTP theo Route
  - Chart: Busiest routes, worst routes
- 3.5 OTP theo Khung giờ
  - Chart: Heatmap Year × Time Block
  - Insight: peak hours, late afternoon effect
- 3.6 Phân phối trễ chuyến
  - Chart: Histogram, Violin plot by year
  - Insight: tail behavior, extreme delays
- 3.7 Nguyên nhân trễ
  - Chart: Stacked bar by year, by carrier
  - Insight: Carrier delay vs Late Aircraft
- 3.8 OTP theo Distance
  - Chart: OTP vs Distance Group
- 3.9 Cancellation & Diversion
  - Chart: Rate by year, by airport, reasons pie
- 3.10 Tương quan biến
  - Chart: Correlation heatmap
  - Chart: Taxi out vs delay hexbin
- 3.11 OTP theo ngày trong tuần
  - Chart: Bar by DOW
- 3.12 Daily trend trong tháng 1
  - Chart: Line by day, color = year

### Chương 4: Phân tích Driver / Thống kê / Giả thuyết (8-10 trang)
- 4.1 So sánh OTP 2 năm (Chi-square test)
  - H0: OTP year A = OTP year B
  - H1: OTP year A ≠ OTP year B
- 4.2 So sánh median delay giữa time blocks (Kruskal-Wallis)
  - H0: Median delay giống nhau giữa các time blocks
- 4.3 Effect size (Cramér's V, Cohen's d)
- 4.4 Lập luận driver analysis
  - Carrier effect vs Route effect
  - Time-of-day effect
  - Weather proxy (WEATHER_DELAY share)

### Chương 5: Mô hình dự đoán trễ (20-25 trang)
- 5.1 Thiết kế bài toán
  - Target: ARR_DEL15
  - Track A (pre-flight) vs Track B (post-pushback)
  - Leakage prevention rules
- 5.2 Feature engineering
  - Encoding strategies (Label, Frequency, Target)
  - Historical OTP features
- 5.3 Train/Test split (temporal: 2021-2024 / 2025)
- 5.4 Class imbalance handling
- 5.5 Track A Results
  - Bảng: So sánh 3 models (ROC-AUC, F1, PR-AUC)
  - Chart: ROC curves, PR curves, Confusion matrix
  - Feature importance (Permutation)
  - SHAP summary (if boosting)
- 5.6 Track B Results — tương tự
- 5.7 Track A vs B Comparison
- 5.8 Drift Analysis (performance by year)
- 5.9 Calibration (optional)

### Chương 6: Dashboard (5-8 trang)
- 6.1 Kiến trúc dashboard (Streamlit)
- 6.2 Screenshots: Bộ lọc, KPI cards, biểu đồ
- 6.3 Trang Predict Delay
- 6.4 Hướng dẫn deploy (Streamlit Cloud)

### Chương 7: Kết luận & Khuyến nghị (5-8 trang)
- 7.1 Tóm tắt phát hiện chính
  - OTP trend
  - Top drivers of delay
  - Model performance
- 7.2 Khuyến nghị cho airlines / airports / regulators
- 7.3 Hạn chế
  - Chỉ tháng 1 (không đại diện mùa hè)
  - Không có weather data bên ngoài
  - Track A có precision hạn chế
- 7.4 Hướng phát triển
  - Mở rộng sang nhiều tháng
  - Kết hợp weather API
  - Deep learning approaches
  - Real-time prediction system

### Chương 8: Tài liệu tham khảo (2-3 trang)

### Phụ lục
- A: Data Dictionary đầy đủ
- B: Danh sách mã hãng bay (OP_CARRIER)
- C: Code snippets quan trọng
- D: Thêm biểu đồ phụ
