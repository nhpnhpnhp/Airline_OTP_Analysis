# Ghi chú phụ thuộc với Track B

## Những điểm Track B cần giữ ổn định

- Tên biến mục tiêu: `ARR_DEL15`
- Cách chia theo thời gian: train 2021-2024, test 2025
- Bộ metric cốt lõi: ROC-AUC, PR-AUC, Accuracy, F1, ma trận nhầm lẫn
- Phạm vi diễn giải: mô tả/liên hệ + dự báo, không khẳng định nhân quả

## Những gì không chặn tiến độ của Track A

- Lựa chọn mô hình của Track B
- Chiến lược tuning của Track B
- Phần interpretability mở rộng của Track B

## Những gì có thể làm hỏng so sánh chéo cuối kỳ

- Khác định nghĩa biến mục tiêu
- Khác năm kiểm tra
- Khác cách định nghĩa metric
- Khác quy ước đặt tên artifact khiến việc đối chiếu bị mơ hồ
