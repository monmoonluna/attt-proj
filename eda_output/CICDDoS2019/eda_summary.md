# Báo cáo EDA - cicddos2019

File nguồn: `data/CICDDoS2019/cicddos2019_train.csv`

Kích thước dữ liệu (sau khi làm sạch cơ bản): 569650 dòng, 80 đặc trưng (chưa tính cột nhãn)

## 1. Phân bố nhãn

- Nhãn nhị phân: {'Attack': 512685, 'Normal': 56965}
- Tỉ lệ mất cân bằng (lớp lớn nhất / lớp nhỏ nhất): **9.00x**
- Số lớp trong nhãn gốc (multi-class): 8

## 2. Thống kê mô tả

- Chi tiết đầy đủ: `descriptive_stats.csv`
- Số đặc trưng có phương sai bằng 0 (hằng số, vô dụng cho mô hình): 12

## 3. Tương quan giữa các đặc trưng

- Biểu đồ: `correlation_heatmap.png`
- Số cặp đặc trưng có |tương quan| > 0.95: 48
- Top 10 cặp tương quan cao nhất:
    - `Flow Duration` ↔ `Fwd IAT Total`: 1.0
    - `Total Fwd Packets` ↔ `Subflow Fwd Packets`: 1.0
    - `Total Backward Packets` ↔ `Subflow Bwd Packets`: 1.0
    - `Total Length of Fwd Packets` ↔ `Subflow Fwd Bytes`: 1.0
    - `Total Length of Bwd Packets` ↔ `Subflow Bwd Bytes`: 1.0
    - `Fwd Packet Length Min` ↔ `Min Packet Length`: 1.0
    - `Fwd Packet Length Mean` ↔ `Avg Fwd Segment Size`: 1.0
    - `Bwd Packet Length Mean` ↔ `Avg Bwd Segment Size`: 1.0
    - `Fwd PSH Flags` ↔ `RST Flag Count`: 1.0
    - `Fwd Header Length` ↔ `Fwd Header Length.1`: 1.0

## 4. Mutual Information (soát rò rỉ dữ liệu)

- Chi tiết: `mutual_information.csv`, biểu đồ: `mutual_information.png`
- Entropy của nhãn (tham chiếu): 0.3178
- Đặc trưng có MI cao nhất: `Average Packet Size` = 0.2551
- ⚠️ **CẢNH BÁO**: MI của `Average Packet Size` chiếm 80% entropy của nhãn - rất đáng ngờ, cần kiểm tra xem đặc trưng này có vô tình mã hóa trực tiếp nhãn hay không trước khi dùng cho Feature Selection chính thức (tương tự trường hợp Bwd Packet Length Max từng gặp trên CICIDS2017).

## 5. PCA 2D

- Biểu đồ: `pca_scatter.png`
- 2 thành phần chính giải thích 99.7% tổng phương sai dữ liệu.
