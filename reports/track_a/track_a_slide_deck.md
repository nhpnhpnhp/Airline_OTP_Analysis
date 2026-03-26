# Phác thảo slide - Track A

## Slide 1 - Tiêu đề
- Airline OTP Analysis
- Track A: Dự đoán chuyến bay đến trễ từ thông tin trước chuyến bay

## Slide 2 - Động lực bài toán
- Vì sao OTP quan trọng với hãng bay và hành khách
- Vì sao dữ liệu BTS tháng 1 giai đoạn 2021-2025 là phù hợp

## Slide 3 - Phạm vi hiện tại của dự án
- Đã hoàn thành pipeline preprocessing
- Đã hoàn thành exploratory analysis
- Đã bổ sung Track A như nhánh mô hình dự báo đầu tiên

## Slide 4 - Pipeline dữ liệu
- CSV raw từ BTS
- Các đầu ra parquet đã làm sạch
- Tập dữ liệu Track A sẵn sàng cho ML

## Slide 5 - Biến mục tiêu và cách chia dữ liệu
- Biến mục tiêu: ARR_DEL15
- Huấn luyện: 2021-2024
- Kiểm tra: 2025
- Vì sao chia theo thời gian là quan trọng

## Slide 6 - Thiết kế đặc trưng cho Track A
- Đặc trưng lịch bay và thời gian
- Đặc trưng tần suất tuyến bay và sân bay
- Đặc trưng OTP lịch sử
- Chỉ dùng danh sách đặc trưng gốc của Track A
- Không dùng leakage sau giờ cất cánh

## Slide 7 - Kiểm tra leakage
- Đã loại bỏ các cột bị cấm
- Vì sao Track A vẫn giữ đúng giả định trước chuyến bay

## Slide 8 - Kiểm định thống kê 1
- Chi-bình phương giữa YEAR và ARR_DEL15
- Ý nghĩa chính của kết quả

## Slide 9 - Kiểm định thống kê 2
- Kruskal-Wallis cho ARR_DELAY_NEW theo DEP_TIME_BLK
- Ý nghĩa chính của kết quả

## Slide 10 - Các phát hiện liên hệ mô tả
- Hãng bay
- Tuyến bay
- Khung giờ

## Slide 11 - Các mô hình
- Logistic Regression làm baseline
- Random Forest (`scikit-learn`)
- Gradient Boosting (`scikit-learn`)

## Slide 12 - Bộ metric đánh giá
- ROC-AUC
- PR-AUC
- Accuracy
- F1
- Ma trận nhầm lẫn

## Slide 13 - So sánh mô hình
- Bảng kết quả của Track A
- Mô hình được chọn: Logistic Regression

## Slide 14 - Độ quan trọng đặc trưng
- Permutation importance cho mô hình tốt nhất

## Slide 15 - Kết luận và bước tiếp theo
- Track A đã hoàn thành được gì
- Phần nào vẫn là mở rộng tùy chọn
- Track B có thể căn chỉnh như thế nào mà không chặn tiến độ Track A
