# Báo cáo EDA - cicids2017

File nguồn: `data/cicids2017/cicids2017_cleaned.csv`

Kích thước dữ liệu (sau khi làm sạch cơ bản): 2520751 dòng, 52 đặc trưng (chưa tính cột nhãn)

## 1. Phân bố nhãn

- Nhãn nhị phân: {'Normal': 2095057, 'Attack': 425694}
- Tỉ lệ mất cân bằng (lớp lớn nhất / lớp nhỏ nhất): **4.92x**
- Số lớp trong nhãn gốc (multi-class): 7

## 2. Thống kê mô tả

- Chi tiết đầy đủ: `descriptive_stats.csv`
- Số đặc trưng có phương sai bằng 0 (hằng số, vô dụng cho mô hình): 0

## 3. Tương quan giữa các đặc trưng

- Biểu đồ: `correlation_heatmap.png`
- Số cặp đặc trưng có |tương quan| > 0.95: 23
- Top 10 cặp tương quan cao nhất:
    - `Total Fwd Packets` ↔ `Fwd Header Length`: 1.0
    - `Total Fwd Packets` ↔ `Bwd Header Length`: 1.0
    - `Total Fwd Packets` ↔ `act_data_pkt_fwd`: 1.0
    - `Total Length of Fwd Packets` ↔ `Subflow Fwd Bytes`: 1.0
    - `Fwd Header Length` ↔ `Bwd Header Length`: 1.0
    - `Fwd Header Length` ↔ `act_data_pkt_fwd`: 1.0
    - `Bwd Header Length` ↔ `act_data_pkt_fwd`: 1.0
    - `Flow Duration` ↔ `Fwd IAT Total`: 0.999
    - `Flow IAT Max` ↔ `Fwd IAT Max`: 0.999
    - `Packet Length Mean` ↔ `Average Packet Size`: 0.998

## 4. Mutual Information (soát rò rỉ dữ liệu)

- Chi tiết: `mutual_information.csv`, biểu đồ: `mutual_information.png`
- Entropy của nhãn (tham chiếu): 0.4498
- Đặc trưng có MI cao nhất: `Average Packet Size` = 0.3275
- Không phát hiện đặc trưng nào có MI bất thường cao - chưa có dấu hiệu rò rỉ rõ ràng.

## 5. PCA 2D

- Biểu đồ: `pca_scatter.png`
- 2 thành phần chính giải thích 82.2% tổng phương sai dữ liệu.
