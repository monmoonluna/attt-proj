# Đánh giá TabPFN

TabPFN được đánh giá như một bộ phân loại bổ sung trên cùng các đặc trưng đã
chọn và cùng cỡ mẫu phân tầng: 5.000 dòng train và 2.000 dòng test cho mỗi
bộ dữ liệu. Random Forest vẫn là mô hình chính của live demo.

| Bộ dữ liệu | Random Forest F1-macro | TabPFN F1-macro | Kết luận |
|---|---:|---:|---|
| NSL-KDD | 77.84% | 77.58% | Random Forest tốt hơn |
| CICIDS2017 | 98.31% | 99.02% | TabPFN tốt hơn |
| CICDDoS2019 | 92.96% | 89.73% | Random Forest tốt hơn |
| UNSW-NB15 | 82.73% | 78.99% | Random Forest tốt hơn |

TabPFN chỉ thắng trên CICIDS2017, vì vậy chưa đáp ứng tiêu chí thay model
chính: phải có F1-macro cao hơn Random Forest trên cả bốn bộ dữ liệu.
Chi tiết đầy đủ nằm trong
[`output/tabpfn_evaluation.json`](../output/tabpfn_evaluation.json).
