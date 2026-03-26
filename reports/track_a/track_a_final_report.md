# Báo cáo cuối cùng - Track A

## 1. Tóm tắt điều hành

Track A tập trung vào các biến có sẵn trước chuyến bay để dự đoán `ARR_DEL15`, với cách chia thời gian 2021-2024 cho tập huấn luyện và 2025 cho tập kiểm tra. Workflow kết hợp phân tích thống kê gọn, mô hình hóa có kiểm soát rò rỉ dữ liệu và sinh bộ artifact cuối cùng phục vụ cho đồ án môn học.

## 2. Thiết lập thí nghiệm

- Biến mục tiêu: `ARR_DEL15`
- Giai đoạn huấn luyện: 2021-2024
- Giai đoạn kiểm tra: 2025
- Tập đặc trưng Track A: các biến số gốc trước chuyến bay và các biến dẫn xuất từ lịch bay trong kế hoạch preprocessing
- Bộ mô hình: chỉ dùng estimator từ `scikit-learn`
- Quy tắc chống leakage: loại bỏ các biến phản ánh kết quả vận hành như kết quả đến, taxi-in, wheels-on và các nguyên nhân trễ sau khi hạ cánh

## 3. Phân tích thống kê

### 3.1 Kiểm định Chi-bình phương: YEAR và ARR_DEL15

- Thống kê kiểm định: 31026.8233
- Bậc tự do: 4
- p-value: 0
- Kích thước hiệu ứng (Cramér's V): 0.1129

Diễn giải: phân phối OTP có khác biệt giữa các năm, nhưng cần đọc kích thước hiệu ứng cùng với ý nghĩa nghiệp vụ thay vì chỉ nhìn p-value vì cỡ mẫu rất lớn.

### 3.2 Kiểm định Kruskal-Wallis: ARR_DELAY_NEW theo DEP_TIME_BLK

- Thống kê kiểm định: 21825.2461
- Bậc tự do: 18
- p-value: 0
- Kích thước hiệu ứng (Epsilon bình phương): 0.0090

Diễn giải: mức độ trễ khác nhau giữa các khung giờ khởi hành, cho thấy thời điểm trong ngày là một yếu tố liên quan đáng chú ý đối với Track A.

## 4. Phân tích liên hệ mô tả

### 4.1 Các hãng có liên hệ nổi bật

| OP_CARRIER | flights | delay_rate |
| --- | --- | --- |
| WN | 474178 | 0.1799 |
| DL | 336839 | 0.1710 |
| AA | 325701 | 0.2043 |
| OO | 275984 | 0.1984 |
| UA | 237279 | 0.1942 |
| YX | 118273 | 0.1588 |
| MQ | 97683 | 0.1993 |
| B6 | 87727 | 0.2734 |
| NK | 86397 | 0.2270 |
| OH | 80343 | 0.2081 |

### 4.2 Các chặng có tỷ lệ trễ cao (tối thiểu 500 chuyến)

| ROUTE | flights | delay_rate |
| --- | --- | --- |
| BOS-PBI | 784 | 0.3686 |
| ASE-DFW | 555 | 0.3658 |
| ORD-ASE | 626 | 0.3594 |
| LAX-ASE | 652 | 0.3543 |
| DFW-ASE | 550 | 0.3473 |
| DEN-ASE | 1025 | 0.3454 |
| JFK-PBI | 810 | 0.3444 |
| BOS-SJU | 633 | 0.3302 |
| ASE-ORD | 627 | 0.3301 |
| BOS-RSW | 968 | 0.3285 |

### 4.3 Tóm tắt theo khung giờ

| DEP_TIME_BLK | delay_rate | avg_delay_new |
| --- | --- | --- |
| 2100-2159 | 0.2387 | 18.7427 |
| 1900-1959 | 0.2482 | 18.7070 |
| 1800-1859 | 0.2386 | 17.6625 |
| 2200-2259 | 0.2199 | 17.5582 |
| 2000-2059 | 0.2404 | 17.5074 |
| 1700-1759 | 0.2288 | 16.9010 |
| 1600-1659 | 0.2248 | 16.3848 |
| 1500-1559 | 0.2193 | 16.1138 |
| 1400-1459 | 0.2072 | 15.3326 |
| 1300-1359 | 0.2033 | 15.1525 |

Các kết quả trên chỉ mang tính mô tả và liên hệ. Chúng hỗ trợ việc chọn đặc trưng, nhưng không được trình bày như bằng chứng nhân quả.

## 5. Kiểm soát leakage và chia theo thời gian

- Giai đoạn preprocessing đã loại bỏ các cột leakage không hợp lệ cho Track A.
- Việc đánh giá trên năm 2025 phản ánh bài toán dự báo thực tế tốt hơn so với chia ngẫu nhiên.
- Tập đặc trưng Track A vẫn bám sát giả định chỉ sử dụng thông tin có trước chuyến bay.

## 6. Mô hình hóa Track A

### 6.1 So sánh mô hình

| model | roc_auc | pr_auc | accuracy | precision | recall | f1 | threshold | tp | tn | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6165 | 0.2698 | 0.2190 | 0.1916 | 0.9802 | 0.3205 | 0.4800 | 96191 | 18186 | 405953 | 1939 |
| Gradient Boosting | 0.6010 | 0.2512 | 0.4122 | 0.2134 | 0.7922 | 0.3362 | 0.2000 | 77739 | 137555 | 286584 | 20391 |
| Random Forest | 0.5959 | 0.2455 | 0.4519 | 0.2174 | 0.7376 | 0.3358 | 0.5000 | 72376 | 163630 | 260509 | 25754 |

### 6.2 Mô hình tốt nhất

Mô hình được chọn là **Logistic Regression**, dựa trên khả năng phân biệt trên tập kiểm tra và sự cân bằng tổng thể giữa ROC-AUC, PR-AUC, Accuracy và F1.

## 7. Độ quan trọng hoán vị

| feature | importance_drop_auc |
| --- | --- |
| DAY_OF_MONTH | 0.0385 |
| DISTANCE | 0.0250 |
| CRS_ELAPSED_TIME | 0.0243 |
| ORIGIN_HIST_OTP | 0.0209 |
| CRS_DEP_TIME_MIN | 0.0113 |
| DISTANCE_GROUP | 0.0074 |
| IS_WEEKEND | 0.0036 |
| CRS_ARR_COS | 0.0034 |
| CRS_DEP_COS | 0.0030 |
| DEST_FREQ | 0.0018 |

Phân tích permutation importance chỉ được tính cho mô hình tốt nhất để giữ phần diễn giải ở mức tập trung và phù hợp phạm vi đồ án.

## 8. Ghi chú phụ thuộc với Track B

Track A không phụ thuộc vào việc Track B hoàn thiện mô hình để kết thúc phần việc riêng của mình. Điểm cần phối hợp duy nhất là giữ thống nhất tên biến mục tiêu, cách chia theo thời gian và bộ metric cốt lõi để có thể so sánh chéo ở giai đoạn cuối.

## 9. Hạn chế và hướng tiếp theo

- Track A chỉ dùng thông tin trước chuyến bay, nên có một giới hạn tự nhiên về mức hiệu năng có thể đạt được.
- Các mô hình ensemble được giữ ở mức gọn để phạm vi triển khai phù hợp với đồ án.
- Hướng mở rộng tùy chọn: SHAP cho một mô hình boosting, dashboard tổng quan nhẹ và so sánh song song khi Track B hoàn thiện.
