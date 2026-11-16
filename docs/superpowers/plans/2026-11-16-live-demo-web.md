# Kế hoạch Thực hiện: Xây dựng Trang Web Demo Trực quan Phát hiện Xâm nhập Mạng (NIDS Live Demo)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng trang web demo tương tác độc lập (`reports/live_demo.html`) cho đề tài 2.18, cho phép chọn mẫu gói tin/lưu lượng mạng, bấm nút "Phát hiện", hiển thị nhãn tấn công kèm giải thích nguyên nhân (XAI - Feature Importance & Decision Path), và đối chiếu định lượng giữa mô hình đầy đủ đặc trưng vs mô hình rút gọn (Filter, GA, PSO, Hybrid).

**Architecture:** 
- Giao diện người dùng theo chuẩn Anti-AI Aesthetic (không emoji, bảng màu Slate/Navy học thuật, định dạng số tabular-nums căn phải).
- Engine suy luận và giải thích phía client (thuần JavaScript/HTML5), nạp sẵn các mẫu lưu lượng thực nghiệm từ NSL-KDD, CICIDS2017, CICDDoS2019.
- Bộ giải thích luật quyết định (Rule-based XAI & Waterfall Feature Impact) giải thích tại sao một luồng mạng bị cảnh báo.
- Bảng đối chiếu hiệu năng thời gian thực (Độ trễ suy luận, Tỷ lệ giảm chiều, F1-macro giữ vững).

**Tech Stack:** HTML5, Modern CSS (CSS Grid, Flexbox, Custom Properties), Vanilla JavaScript ES6+ (Zero external npm dependencies, chạy offline trực tiếp qua trình duyệt).

**Spec:** Yêu cầu từ người dùng kết hợp Quy chuẩn Thiết kế & Học thuật tại `AGENTS.md` (Mục 2 & Mục 5) và dữ liệu thực nghiệm tại `output/`.

---

## Global Constraints

- **Quy tắc Thư mục:** File trang web demo đặt tại `reports/live_demo.html`. Không tạo file tại root.
- **Quy chuẩn Giao diện Anti-AI (AGENTS.md Rule 5):**
  - Tuyệt đối không dùng emoji trang trí trong tiêu đề, nút bấm, bảng biểu, badge.
  - Sử dụng bảng màu Slate & Academic Navy (`#0f172a`, `#1e3a8a`, `#64748b`, `#f8fafc`).
  - Toàn bộ số liệu định lượng phải căn phải (`text-align: right`) và bật `font-variant-numeric: tabular-nums`.
  - Văn phong học thuật, khách quan, không dùng từ ngữ khoa trương AI.
- **Tính Độc lập (Portability):** Chạy được offline khi mở trực tiếp file `reports/live_demo.html` trên trình duyệt mà không cần cài đặt thêm web server.

---

## Review Focus

1. **Khả năng giải thích (Explainability):** Khi phát hiện tấn công (như SYN Flood hoặc Port Scan), hệ thống phải chỉ rõ đặc trưng nào kích hoạt cảnh báo và giá trị đo được vượt ngưỡng bình thường bao nhiêu %.
2. **Tính chân thực của dữ liệu:** Các mẫu lưu lượng và danh sách đặc trưng rút gọn phải khớp chính xác với kết quả thực nghiệm trong `output/NSL-KDD` và `output/CICIDS2017` (ví dụ GA 6 đặc trưng, Hybrid 6 đặc trưng).
3. **Độ mượt và tính tương tác:** Cho phép chuyển đổi nhanh giữa các kịch bản mẫu, chỉnh sửa trực tiếp thông số (sliders/inputs), và xem kết quả thay đổi tức thì khi bấm "Phát hiện".
4. **Đối chiếu hiệu năng:** Hiển thị rõ ràng lợi ích của việc chọn đặc trưng: giảm ~85% số chiều, giảm độ trễ suy luận trong khi F1-macro được duy trì > 99%.
5. **Liên kết điều hướng:** Có liên kết chuyển đổi qua lại mượt mà giữa `reports/demo_dashboard.html` (Dashboard tổng hợp) và `reports/live_demo.html` (Live Demo).

---

## Task Structure

### Task 1: Chuẩn bị Dữ liệu Mẫu Lưu lượng & Bộ Tham số Thực nghiệm
**Files:**
- Create: `reports/live_demo_data.js` (hoặc nhúng trực tiếp dạng module trong `reports/live_demo.html`)
- Source reference: `output/NSL-KDD/fs_ga/selected_features.json`, `output/CICIDS2017/fs_hybrid1_ga_v2/selected_features.json`

- [ ] Trích xuất các kịch bản lưu lượng chuẩn từ NSL-KDD và CICIDS2017:
  - Mẫu 1: `Normal Traffic (HTTP/Web)` — Kết nối hợp lệ, tỷ lệ lỗi 0%, lưu lượng cân bằng.
  - Mẫu 2: `DoS SYN Flood (Neptune)` — `dst_host_srv_serror_rate` = 1.0, `src_bytes` = 0, `count` cao.
  - Mẫu 3: `Probe / Port Scan (Satan)` — `dst_host_diff_srv_rate` cao, quét nhiều dịch vụ khác nhau.
  - Mẫu 4: `R2L / Password Brute Force (Guess_Passwd)` — `num_failed_logins` > 0, thời lượng kết nối kéo dài.
  - Mẫu 5: `CICIDS2017 DDoS LOIC/HOIC` — `Packet Length Variance` cao, `Flow Packets/s` đột biến.
  - Mẫu 6: `CICIDS2017 Web Attack / SQL Injection` — Kích thước gói tin bất thường, cờ điều khiển bất thường.
- [ ] Thiết lập bảng trọng số đặc trưng (Feature Weights / Rules) cho mô hình Baseline (toàn bộ đặc trưng) và các mô hình chọn đặc trưng (Filter, GA 6 đặc trưng, Hybrid 6 đặc trưng).

### Task 2: Xây dựng Giao diện Người dùng Học thuật (Academic UI Layout)
**Files:**
- Create: `reports/live_demo.html` (Phần HTML Structure & CSS Styling)

- [ ] Xây dựng Topbar & Project Header theo chuẩn đề tài 2.18 (Mã đề tài, Học phần ATTT, Đề tài Lựa chọn đặc trưng NIDS, Nút chuyển sang Dashboard phân tích).
- [ ] Xây dựng Khối Điều khiển Đầu vào (Input Control Panel):
  - Hộp chọn Kịch bản lưu lượng mẫu (Pre-set Scenario Selector).
  - Hộp chọn Bộ dữ liệu (`NSL-KDD` hoặc `CICIDS2017`).
  - Hộp chọn Phương pháp chọn đặc trưng (`Baseline`, `Filter MI`, `Wrapper GA`, `Wrapper PSO`, `Hybrid Filter+GA`).
  - Lưới hiển thị & Tùy biến thông số gói tin (Interactive Feature Inputs: các trường đặc trưng quan trọng có thể điều chỉnh giá trị).
  - Cụm nút hành động: "Phát hiện xâm nhập", "Tạo mẫu ngẫu nhiên", "Khôi phục mặc định".

### Task 3: Phát triển Engine Suy luận & Giải thích (Inference & XAI Engine)
**Files:**
- Modify: `reports/live_demo.html` (Phần JavaScript Logic)

- [ ] Hiện thực hàm suy luận `runDetection(sample, method)`:
  - Tính điểm bất thường (Anomaly Score) dựa trên các đặc trưng được chọn.
  - Xác định nhãn phân loại: `Bình thường (Normal)` vs `Tấn công (Attack: DoS, Probe, R2L, U2R, DDoS)`.
  - Tính toán độ tin cậy phân loại (Confidence Score %).
  - Ước tính thời gian suy luận (Inference Latency tính theo micro giây $\mu s$ dựa trên số lượng đặc trưng).
- [ ] Hiện thực bộ phân tích nguyên nhân `explainDecision(sample, label)`:
  - Phân tích đóng góp từng đặc trưng (Feature Contributions) theo dạng Waterfall/Bar.
  - Bật cờ cảnh báo bất thường nếu giá trị vượt ngưỡng chuẩn (ví dụ `serror_rate > 0.5`, `count > 200`).
  - Sinh diễn giải logic cây quyết định dạng văn bản khoa học ngắn gọn: "Phát hiện DoS do tỷ lệ lỗi TCP SYN đạt 100% kết hợp lưu lượng đến máy đích tập trung cao độ".

### Task 4: Xây dựng Khối Trực quan hóa & Đối chiếu Hiệu năng (Metrics & Comparison)
**Files:**
- Modify: `reports/live_demo.html`

- [ ] Hiển thị Thẻ Kết quả Chính (Detection Verdict Card):
  - Trạng thái nhãn: Bình thường (xanh dịu `#15803d` / nền `#f0fdf4`) hoặc Cảnh báo Tấn công (đỏ dịu `#b91c1c` / nền `#fef2f2`).
  - Mức độ nghiêm trọng (Risk Severity: Thấp, Trung bình, Cao, Rất cao).
  - Khuyến nghị phòng thủ mạng (Network Action: Chặn IP, Giới hạn băng thông SYN, Bật WAF).
- [ ] Bảng Đối chiếu Trực tiếp: Baseline vs Phương pháp Chọn đặc trưng:
  - Số lượng đặc trưng: 39 vs 6 đặc trưng (Giảm 84.6%).
  - Thời gian suy luận: $0.36\text{ ms}$ vs $0.05\text{ ms}$ (Nhanh hơn $7.2\times$).
  - Khả năng duy trì phát hiện: Cả 2 cùng phát hiện chính xác mẫu tấn công.
- [ ] Biểu đồ Đóng góp Đặc trưng (Feature Importance Bar Chart SVG/CSS thuần).

### Task 5: Kiểm thử Toàn diện & Tích hợp Điều hướng
**Files:**
- Modify: `reports/live_demo.html`
- Modify: `reports/demo_dashboard.html` (Bổ sung nút chuyển nhanh đến trang Live Demo)

- [ ] Kiểm thử chạy độc lập trên trình duyệt (`open reports/live_demo.html`):
  - Kiểm tra việc chuyển đổi tất cả 6 kịch bản lưu lượng.
  - Kiểm tra việc thay đổi thủ công giá trị trường đặc trưng và bấm phát hiện.
  - Kiểm tra tính toán số học, căn phải các cột dữ liệu số, zero emoji.
- [ ] Thêm nút liên kết "Mở Trình Demo Phát Hiện Trực Quan" trên đầu thanh Topbar của `reports/demo_dashboard.html`.
- [ ] Xác nhận giao diện hiển thị sắc nét trên cả desktop và màn hình trình chiếu máy chiếu giảng đường.
