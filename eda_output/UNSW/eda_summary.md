# Báo cáo EDA - unswnb15

File nguồn: `data/UNSW-NB15/UNSW_NB15_training-set.csv`

Kích thước dữ liệu (sau khi làm sạch cơ bản): 175341 dòng, 42 đặc trưng (chưa tính cột nhãn)

## 1. Phân bố nhãn

- Nhãn nhị phân: {'Attack': 119341, 'Normal': 56000}
- Tỉ lệ mất cân bằng (lớp lớn nhất / lớp nhỏ nhất): **2.13x**
- Số lớp trong nhãn gốc (multi-class): 10

## 2. Thống kê mô tả

- Chi tiết đầy đủ: `descriptive_stats.csv`
- Số đặc trưng có phương sai bằng 0 (hằng số, vô dụng cho mô hình): 0

## 3. Tương quan giữa các đặc trưng

- Biểu đồ: `correlation_heatmap.png`
- Số cặp đặc trưng có |tương quan| > 0.95: 13
- Top 10 cặp tương quan cao nhất:
    - `is_ftp_login` ↔ `ct_ftp_cmd`: 1.0
    - `sbytes` ↔ `sloss`: 0.998
    - `dbytes` ↔ `dloss`: 0.998
    - `swin` ↔ `dwin`: 0.991
    - `ct_srv_src` ↔ `ct_srv_dst`: 0.981
    - `dpkts` ↔ `dloss`: 0.98
    - `spkts` ↔ `sloss`: 0.978
    - `dpkts` ↔ `dbytes`: 0.976
    - `spkts` ↔ `sbytes`: 0.974
    - `ct_dst_src_ltm` ↔ `ct_srv_dst`: 0.972

## 4. Mutual Information (soát rò rỉ dữ liệu)

- Chi tiết: `mutual_information.csv`, biểu đồ: `mutual_information.png`
- Entropy của nhãn (tham chiếu): 0.6262
- Đặc trưng có MI cao nhất: `sbytes` = 0.4436
- Không phát hiện đặc trưng nào có MI bất thường cao - chưa có dấu hiệu rò rỉ rõ ràng.

## 5. PCA 2D

- Biểu đồ: `pca_scatter.png`
- 2 thành phần chính giải thích 99.1% tổng phương sai dữ liệu.
