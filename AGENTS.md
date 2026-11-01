# AGENTS.md — Quy ước & Chỉ dẫn Phát triển cho Dự án NIDS Feature Selection

Tài liệu này chứa các quy tắc, kiến trúc và chỉ dẫn vận hành dành cho AI Agent và các lập trình viên khi làm việc trên kho mã nguồn **Feature Selection for NIDS** (`attt-proj` — Mã đề tài: 2.18).

---

## 1. Tổng quan Dự án (Project Overview)

- **Mục tiêu**: Nghiên cứu và xây dựng quy trình phát hiện xâm nhập mạng (*Network Intrusion Detection System* - NIDS) kết hợp các kỹ thuật lựa chọn đặc trưng nâng cao (Filter, Wrapper, Hybrid) nhằm giảm chiều dữ liệu, tối ưu thời gian huấn luyện/suy luận mà vẫn duy trì hoặc nâng cao độ chính xác phát hiện tấn công.
- **Dữ liệu nghiên cứu**:
  - `NSL-KDD`: Dữ liệu baseline, phân loại nhị phân và đa lớp (Normal, DoS, Probe, R2L, U2R).
  - `CICIDS2017`: Dữ liệu lưu lượng mạng hiện đại (DDoS, Botnet, Brute Force, Web Attacks,...).
  - `UNSW-NB15`: Dữ liệu phân bố lưu lượng đa dạng.
  - `CICDDoS2019`: Dữ liệu chuyên sâu về các kịch bản tấn công DDoS.
- **Các phương pháp tiếp cận**:
  - **Filter**: Mutual Information, Chi-square, ANOVA F-test, Correlation.
  - **Wrapper**: Genetic Algorithm (GA), Binary Particle Swarm Optimization (BPSO).
  - **Hybrid**: Kết hợp Filter (lọc sơ bộ chiều cao) $\rightarrow$ Wrapper (tinh chọn tương tác đặc trưng).

---

## 2. Quy tắc Bất khả xâm phạm (Inviolable Rules)

1. **Nghiêm cấm rò rỉ dữ liệu (No Data Leakage)**:
   - Tất cả các bộ biến đổi (`MinMaxScaler`, `StandardScaler`, `LabelEncoder`, `OneHotEncoder`, bộ chọn đặc trưng `Filter`, `SMOTE`) **chỉ được `fit` trên tập Train**.
   - Tập Test chỉ được dùng `transform` từ tham số đã học của tập Train.
2. **Bảo toàn phân tầng nhãn (Stratified Sampling)**:
   - Khi chia tập train/val, hoặc khi lấy mẫu phụ (subsampling) để tính hàm fitness cho GA/PSO, **bắt buộc** dùng phân tầng (`stratify=y`) để tránh bỏ sót các lớp tấn công thiểu số (rare attacks).
3. **Vệ sinh Git & File dung lượng lớn**:
   - Không commit file dữ liệu lớn (`data/**/*.csv`), các file mô hình nén dung lượng lớn hoặc cache tạm vào Git. Luôn tuân thủ `.gitignore`.
4. **Chuẩn hóa Module hóa triệt để (Strict Modular Architecture — "Code ra Code, Docs ra Docs")**:
   - **Kể từ đây trở đi, mọi file mới hoặc bước refactor kế tiếp BẮT BUỘC tuân theo kiến trúc module**:
     - **Code thuật toán cốt lõi** $\rightarrow$ đặt vào `src/` (hoặc module tương ứng).
     - **Script tiện ích / CLI chạy bổ trợ** $\rightarrow$ đặt vào `scripts/`.
     - **Báo cáo, Dashboard, Notebook trình diễn** $\rightarrow$ đặt vào `reports/`.
     - **Tài liệu đề cương, đặc tả kỹ thuật, hình ảnh** $\rightarrow$ đặt vào `docs/` (và `docs/assets/`).
     - **Dữ liệu** $\rightarrow$ đặt vào `data/`.
     - **Kết quả xuất ra** $\rightarrow$ đặt vào `output/` hoặc `eda_output/`.
   - **Tuyệt đối KHÔNG tạo thêm file `.py`, `.ipynb`, `.docx` hay tài liệu linh tinh ở thư mục gốc (`root`)**. Thư mục gốc chỉ được chứa file cấu hình dự án (`AGENTS.md`, `.gitignore`, `README.md`, `requirements.txt`).

---

## 3. Kiến trúc Mã nguồn & Lộ trình Chuẩn hóa (Architecture & Target Standard)

### 3.1. Trạng thái Hiện tại (Current Working State)
> [!NOTE]
> Các file thuật toán cũ (`preprocessing.py`, `feature_selection.py`, `filter_fs.py`,...) tạm thời được giữ ở thư mục gốc để duy trì tính tương thích với các lệnh đã chạy trước đó, tránh xáo trộn trước buổi báo cáo.

```
attt-proj/
├── AGENTS.md                              # Cấu hình & Chỉ dẫn Agent
├── scripts/                               # [MÃ NGUỒN TIỆN ÍCH]
│   └── demo_cli.py                        # Script CLI chạy demo & kiểm thử
├── reports/                               # [BÁO CÁO & TRỰC QUAN HÓA]
│   ├── demo_dashboard.html                # Web Dashboard báo cáo kết quả
│   └── demo_ket_qua.ipynb                 # Notebook báo cáo thực nghiệm
├── docs/                                  # [TÀI LIỆU & ĐỀ CƯƠNG]
│   └── assets/                            # Hình ảnh sơ đồ quy trình
├── data/                                  # Dữ liệu mẫu & dữ liệu huấn luyện
├── output/                                # Kết quả chọn đặc trưng, scaler, encoders
├── eda_output/                            # Kết quả phân tích thăm dò dữ liệu
└── (Các script thuật toán gốc giữ nguyên vị trí: preprocessing.py, feature_selection.py,...)
```

### 3.2. Kiến trúc Đích chuẩn Python ML Research (Target Architecture)
Mọi đợt tái cấu trúc (refactoring) hoặc module phát triển mới phải hướng tới kiến trúc chuẩn sau:

```
attt-proj/
├── src/                                   # [CODE RA CODE] Toàn bộ mã nguồn cốt lõi
│   ├── __init__.py
│   ├── data/                              # Data loading & preprocessing pipeline
│   │   ├── __init__.py
│   │   └── preprocessor.py
│   ├── features/                          # Feature selection modules
│   │   ├── __init__.py
│   │   ├── filter_selection.py            # Mutual Info, Chi-Square, ANOVA
│   │   ├── wrapper_ga.py                  # Genetic Algorithm Wrapper
│   │   ├── wrapper_pso.py                 # Binary PSO Wrapper
│   │   └── hybrid_selection.py            # Hybrid Filter -> Metaheuristics
│   └── models/                            # Classification & Evaluation
│       ├── __init__.py
│       └── evaluator.py                   # Đánh giá RF, DT, XGBoost,...
├── scripts/                               # [SCRIPTS & CLI RUNNERS]
│   ├── demo_cli.py
│   └── run_pipeline.py
├── reports/                               # [BÁO CÁO & TRỰC QUAN]
│   ├── demo_dashboard.html
│   └── demo_ket_qua.ipynb
├── docs/                                  # [DOCS RA DOCS] Toàn bộ tài liệu
│   ├── DeCuongChiTiet_FeatureSelection_NIDS.md
│   ├── DeCuongChiTiet_FeatureSelection_NIDS.docx
│   ├── audit_report.md
│   └── assets/
├── data/                                  # Dữ liệu (NSL-KDD, CICIDS2017,...)
├── output/                                # Artifacts và kết quả JSON
└── AGENTS.md                              # Chỉ dẫn cho AI Agent
```

---

## 4. Hướng dẫn Lệnh Thực thi (CLI Runbook)

### 4.1. Tiền xử lý dữ liệu (`preprocessing.py`)
```bash
# Tiền xử lý tập dữ liệu NSL-KDD (Binary)
python preprocessing.py --dataset nsl-kdd --task binary --data-dir data/nsl-kdd --output-dir output/nsl-kdd

# Tiền xử lý đa lớp (Multi-class)
python preprocessing.py --dataset cicids2017 --task multiclass --data-dir data/cicids2017 --output-dir output/cicids2017
```

### 4.2. Khám phá Dữ liệu (`eda.py`)
```bash
python eda.py --dataset-dir output/cicids2017 --output-dir eda_output/CICIDS2017
```

### 4.3. Chọn đặc trưng Filter (`filter_fs.py`)
```bash
python filter_fs.py --dataset cicids2017 --method mutual_info --k 20
```

### 4.4. Chọn đặc trưng Wrapper & Hybrid (`feature_selection.py`, `hybrid1_fs.py`, `hybrid2_fs.py`)
```bash
# Chạy GA wrapper
python feature_selection.py --dataset cicids2017 --method ga --pop-size 30 --generations 20 --eval-classifier dt

# Chạy PSO wrapper
python feature_selection.py --dataset cicids2017 --method pso --particles 30 --iterations 20 --eval-classifier dt

# Chạy Hybrid
python hybrid1_fs.py --dataset cicids2017 --filter-k 35 --wrapper-method ga
```

### 4.5. Đánh giá Tập đặc trưng đã chọn (`eval_selected.py`)
```bash
# Đánh giá với Random Forest, XGBoost hoặc Decision Tree
python eval_selected.py --dataset cicids2017 --selected-json output/cicids2017/fs_ga/selected_features.json --classifier rf
```

### 4.6. Kiểm định chất lượng Artifacts (`audit_feature_artifacts.py`)
```bash
python audit_feature_artifacts.py --output-dir output --report-file audit_report.md
```

### 4.7. Demo Kết quả Thử nghiệm (`scripts/demo_cli.py`, `reports/`)
```bash
# Chạy demo tóm tắt kết quả terminal & live demo NSL-KDD
python scripts/demo_cli.py --dataset all --live-nslkdd

# Mở web dashboard báo cáo trực quan trên trình duyệt
open reports/demo_dashboard.html
```

---

## 5. Tiêu chuẩn Đánh giá & Chỉ số (Evaluation Metrics)

Khi đánh giá kết quả của thuật toán Feature Selection, Agent phải báo cáo đầy đủ các chỉ số:
1. **Số lượng đặc trưng được chọn ($k$)** và **Tỷ lệ giảm chiều** ($\frac{D_{gốc} - k}{D_{gốc}} \times 100\%$).
2. **Accuracy & F1-Score (Macro / Weighted)**: Đặc biệt lưu ý F1-macro trên tập dữ liệu đa lớp bị mất cân bằng.
3. **Detection Rate (DR / Recall) & False Alarm Rate (FAR)**: Hai chỉ số quan trọng hàng đầu trong an ninh mạng.
4. **Thời gian chạy**: Thời gian tìm kiếm đặc trưng ($T_{FS}$) và Thời gian suy luận trên tập Test ($T_{inference}$).

---

## 6. Tích hợp Chuẩn Superpowers cho Agent

Agent hoạt động trong repo này phải tuân thủ nghiêm ngặt các quy tắc Superpowers:
- **Tư duy trước khi code (`brainstorming`)**: Làm rõ các ràng buộc kỹ thuật, phân tích ưu/nhược điểm các thuật toán trước khi đề xuất thay đổi.
- **Lập kế hoạch chia nhỏ (`writing-plans`)**: Chia nhỏ các tác vụ thành checklist kiểm thử được trước khi can thiệp codebase.
- **Debug có hệ thống (`systematic-debugging`)**: Dựa trên trace log và phân tích root-cause, tuyệt đối không "đoán mò" hay sửa tạm bợ.
- **Xác minh trước khi kết luận (`verification-before-completion`)**: Chạy kiểm thử thực tế và đối chiếu số liệu/file output trước khi thông báo hoàn thành.

