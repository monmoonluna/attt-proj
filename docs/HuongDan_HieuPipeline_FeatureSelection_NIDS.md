# Hiểu bài toán và pipeline lựa chọn đặc trưng cho phát hiện tấn công mạng

Tài liệu này giải thích báo cáo **Phát hiện tấn công mạng dựa trên mô hình Feature Selection nâng cao** theo cách dành cho người mới học. Nguồn chính là [báo cáo Word](../BaoCao_FeatureSelection_NIDS.docx), đặc biệt Chương 1–3 và phụ lục. Các bảng kết quả dưới đây lấy từ báo cáo, không phải kết quả chạy lại. Phần đối chiếu mã nguồn ghi nhận trạng thái các script tại thời điểm viết tài liệu ngày 16/11/2026.


Tài liệu có **34 hình**, gồm 14 sơ đồ và tranh minh họa khái niệm cùng 20 biểu đồ dữ liệu. Các hình được đặt ngay tại phần giải thích tương ứng. Sơ đồ thể hiện quy trình hoặc ví dụ; biểu đồ kết quả lấy số liệu đã có trong báo cáo, không phải thực nghiệm chạy mới. Biểu đồ F1 dùng thang 0–1; các ô màu có ghi số để đọc chênh lệch nhỏ.

**Mục lục:** [Bài toán](#1-bài-toán-là-gì-và-giải-quyết-tổng-quan-ra-sao) · [Khái niệm](#2-các-khái-niệm-cần-hiểu-trước-khi-đọc-pipeline) · [Độ đo](#3-các-độ-đo-và-cách-đọc-con-số) · [Dữ liệu](#4-dữ-liệu-sử-dụng) · [EDA](#5-phân-tích-khám-phá-dữ-liệu-trước-khi-huấn-luyện) · [Tiền xử lý](#6-tiền-xử-lý-dữ-liệu-từng-bước) · [Thuật toán](#7-các-phương-pháp-chọn-đặc-trưng-hoạt-động-thế-nào) · [Kết quả tổng quan](#8-kết-quả-tổng-quan-nên-nhớ-trước) · [Kết quả chi tiết](#9-kết-quả-chi-tiết-và-cách-diễn-giải) · [Đối chiếu mã](#11-các-điểm-cần-phân-biệt-giữa-báo-cáo-nguyên-tắc-và-mã-hiện-tại)

## 1. Bài toán là gì và giải quyết tổng quan ra sao

### 1.1. Ta muốn máy tính làm gì

![Minh họa NIDS](assets/pipeline_guide/42-problem.png)

*Gói tin được gom thành luồng, trích đặc trưng, chọn cột rồi đưa vào bộ phân loại. Hai nhánh cuối là hai chế độ nhãn thay thế.*


Giả sử một hệ thống mạng ghi lại nhiều kết nối. Mỗi kết nối được mô tả bằng các con số: kéo dài bao lâu, có bao nhiêu gói tin, gói tin dài bao nhiêu, gửi đến cổng nào, khoảng cách giữa các gói ra sao. Máy tính cần dựa vào các thông tin đó để xác định kết nối bình thường hay tấn công.

Đây là bài toán **phân loại có giám sát**: trong dữ liệu huấn luyện, ta đã biết đáp án của từng dòng. Máy học mối quan hệ giữa thông tin đầu vào và đáp án, rồi dự đoán cho các dòng chưa dùng để huấn luyện.

Đề tài có thêm một câu hỏi nghiên cứu: **có thể chỉ dùng một phần nhỏ các thông tin đầu vào mà vẫn phân loại tốt và huấn luyện nhanh hơn không?** Việc tìm phần nhỏ đó gọi là lựa chọn đặc trưng, hay Feature Selection, viết tắt FS.

Ví dụ, CICIDS2017 trong báo cáo có 52 đặc trưng đầu vào. Thay vì luôn dùng cả 52 cột, GA chọn 5 cột cho bài toán nhị phân. Cần kiểm tra xem mô hình dùng 5 cột này mất bao nhiêu chất lượng và tiết kiệm bao nhiêu thời gian so với mô hình dùng đủ 52 cột.

### 1.2. Đầu vào và đầu ra của toàn bộ hệ thống

**Đầu vào** là các tệp CSV đã có đặc trưng thống kê của luồng và nhãn đúng. Báo cáo bắt đầu từ dữ liệu dạng bảng; phần trực tiếp bắt gói tin và trích đặc trưng từ mạng không phải phần được triển khai chính ở đây.

**Đầu ra** gồm danh sách các cột được giữ, số cột còn lại, chất lượng phân loại trên test, thời gian tìm kiếm và thời gian huấn luyện hoặc suy luận được ghi nhận. Các biểu đồ EDA, đường hội tụ và thí nghiệm bỏ/thêm đặc trưng giúp giải thích độ tin cậy của kết quả.

### 1.3. Cách giải quyết tổng quan

![Minh họa trực quan](assets/pipeline_guide/35-pipeline-illustrated.png)

*Minh họa pipeline bằng các thao tác trực quan: thu thập bảng dữ liệu, kiểm tra, chia tập, tiền xử lý, chọn cột và đánh giá. Hình mô tả quy trình nên áp dụng; các khác biệt của mã hiện tại được giải thích tại Mục 11.*


Các nhánh FS là các phương án thay thế để so sánh. Không phải chạy MI rồi GA rồi PSO rồi cả hai Hybrid nối tiếp nhau. Chỉ riêng các phương pháp lai mới có các bước nối tiếp bên trong.

**Kết luận tổng quan của báo cáo:** trên CICIDS2017 và UNSW-NB15, có thể giảm nhiều đặc trưng mà vẫn giữ chất lượng tương đối tốt. Trên CICDDoS2019, kết quả phụ thuộc mạnh vào dữ liệu, cặp đặc trưng đặc biệt và bộ phân loại. Không có một phương pháp tốt nhất cho mọi trường hợp.

## 2. Các khái niệm cần hiểu trước khi đọc pipeline

### 2.1. Gói tin, luồng và NIDS

**Gói tin** là một đơn vị dữ liệu truyền qua mạng. Một lần trao đổi thường gồm nhiều gói tin. **Luồng**, hay flow, gom các gói có chung thông tin kết nối theo quy tắc của công cụ thu thập, thường liên quan đến địa chỉ nguồn, địa chỉ đích, cổng nguồn, cổng đích và giao thức.

**NIDS**, Network Intrusion Detection System, là hệ thống phát hiện xâm nhập mạng. Trong báo cáo này, mô hình nhận thông tin thống kê của một luồng rồi phân loại. Phát hiện một luồng đáng ngờ chưa đồng nghĩa với tự động ngăn chặn cuộc tấn công.

Một số kiểu tấn công xuất hiện trong báo cáo:

| Loại | Cách hiểu ban đầu |
|:---|:---|
| DoS | Gây quá tải hoặc làm cạn tài nguyên để dịch vụ khó đáp ứng người dùng |
| DDoS | Tấn công từ chối dịch vụ từ nhiều nguồn |
| Port Scanning | Dò các cổng để tìm dịch vụ đang hoạt động |
| Brute Force | Thử nhiều thông tin đăng nhập để tìm mật khẩu đúng |
| Web Attacks | Các tấn công vào ứng dụng web |
| Bots | Lưu lượng liên quan đến máy hoặc chương trình bị điều khiển tự động |

### 2.2. Dòng dữ liệu, đặc trưng và nhãn

Mỗi **dòng** là một mẫu quan sát. Mỗi **đặc trưng** là một cột đầu vào. **Nhãn** là đáp án cần dự đoán. Bảng dưới đây chỉ là ví dụ minh họa, không lấy từ bộ dữ liệu thực nghiệm:

| Mẫu | Thời lượng | Số gói | Cổng đích | Nhãn |
|:---|---:|---:|---:|:---|
| A | 0,8 | 12 | 443 | Normal |
| B | 0,1 | 900 | 80 | Attack |

Ta ký hiệu bảng đầu vào là `X`, cột đáp án là `y`. Không đưa cột nhãn hoặc một bản sao của nhãn vào `X`, vì khi đó mô hình có thể nhìn trực tiếp đáp án.

Các đặc trưng thường gặp gồm `Flow Duration` (thời lượng luồng), `Packet Length Mean` (độ dài gói trung bình), `Packet Length Std` (mức biến động độ dài), `IAT` (thời gian giữa các gói), `Destination Port` (cổng đích), `SYN Flag Count` và `URG Flag Count` (số lần xuất hiện các cờ TCP tương ứng). Đơn vị và cách tính cụ thể phải theo định nghĩa của bộ dữ liệu.

### 2.3. Nhị phân và đa lớp

**Nhị phân** chỉ có hai đáp án: Normal và Attack. DoS, DDoS hay Bots đều được gom thành Attack.

**Đa lớp** giữ các nhóm chi tiết, chẳng hạn Normal, DoS, DDoS, Bots. Bài toán khó hơn vì mô hình phải phân biệt cả những loại tấn công có biểu hiện gần nhau, và một số lớp có rất ít mẫu.

Một mô hình làm tốt nhị phân chưa chắc làm tốt đa lớp. Nhận biết “đây là tấn công” thường dễ hơn gọi đúng loại tấn công.

### 2.4. Train, validation và test

![Minh họa NIDS](assets/pipeline_guide/43-train-validation.png)

*Chỉ train tham gia tìm tập cột; test chỉ mở ở đánh giá cuối. Các điểm số ví dụ trong tranh là giả định, không phải kết quả báo cáo.*


**Train** dùng để học. **Validation** dùng để so sánh cấu hình hoặc tập đặc trưng trong quá trình tìm kiếm. **Test** dùng để đánh giá sau khi đã chốt lựa chọn.

Trong wrapper của báo cáo, lấy một phần từ train rồi chia 70% để huấn luyện mô hình thử nghiệm và 30% để tính fitness. Sau khi tìm được tập cột, huấn luyện một mô hình cuối trên toàn bộ train với các cột đó.


Test phải được giữ độc lập. Nếu liên tục chọn phương pháp, chỉnh trọng số hoặc loại cột dựa trên test, test dần trở thành một phần của quá trình phát triển. Khi đó cần một tập kiểm tra độc lập mới để xác nhận kết luận.

### 2.5. Mất cân bằng, phân tầng và rò rỉ dữ liệu

![Minh họa NIDS](assets/pipeline_guide/44-leakage.png)

*Tham số được học trên train, lưu lại và áp dụng cho test. Tỷ lệ chia trong tranh chỉ minh họa, không phải tỷ lệ cố định của các bộ dữ liệu. Việc chọn cột được giải thích riêng ở Mục 7.*


**Mất cân bằng** nghĩa là số mẫu các lớp khác nhau nhiều. Nếu 90% là Attack, mô hình luôn đoán Attack đã có Accuracy 90%, dù không nhận biết được Normal.

**Chia phân tầng**, `stratify=y`, cố giữ tỷ lệ các lớp ở các phần dữ liệu. Với lớp rất hiếm, phải thiết kế kích thước mẫu và cách chia để lớp đó có đủ đại diện; phân tầng không tự tạo thêm dữ liệu cho lớp chỉ có một mẫu.

**Rò rỉ dữ liệu** xảy ra khi quá trình học nhận thông tin mà đáng ra chưa được biết. Ví dụ tính min/max trên cả train và test trước khi đánh giá. Quy tắc đúng là chia dữ liệu trước, học mọi tham số tiền xử lý trên train rồi áp dụng cho test. Nếu đánh giá nghiêm ngặt trên validation, các phép biến đổi cần được học trong phần train của từng lần chia validation nữa.

### 2.6. Quá khớp và lệch phân phối

**Quá khớp** là học quá sát các chi tiết riêng của dữ liệu huấn luyện, nên gặp dữ liệu mới dự đoán kém. Giảm đặc trưng có thể giúp nhưng cũng có thể làm mất tín hiệu cần thiết.

**Lệch phân phối train–test** nghĩa là dữ liệu lúc kiểm tra khác dữ liệu lúc học. CICDDoS2019 là ví dụ rõ: test chứa những loại tấn công không có ở train. Vì vậy tập cột tốt trên validation tách từ train chưa chắc tốt trên test.

## 3. Các độ đo và cách đọc con số

### 3.1. Ma trận nhầm lẫn

![Minh họa NIDS](assets/pipeline_guide/45-confusion.png)

*Bốn tình huống TP, FN, FP, TN được minh họa bằng lưu lượng mạng. Hình giải thích kết quả phân loại, không mô tả cơ chế tự động chặn kết nối.*

![Minh họa NIDS](assets/pipeline_guide/41-evaluation-illustrated.png)

*Đánh giá luồng mạng trên test, đọc TP/FP/TN/FN, đổi bộ phân loại trên cùng tập cột và ablation. Các số ví dụ trong tranh không phải kết quả thực nghiệm; công thức và số liệu chính xác nằm trong phần chữ và các biểu đồ kết quả.*


Quy ước Attack là lớp dương cho các công thức nhị phân dưới đây:

| Thực tế | Dự đoán Attack | Dự đoán Normal |
|:---|:---|:---|
| Attack | TP: phát hiện đúng tấn công | FN: bỏ sót tấn công |
| Normal | FP: cảnh báo nhầm | TN: nhận đúng bình thường |

Các chỉ số:

```text
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
FAR/FPR   = FP / (FP + TN)
```

Precision trả lời “trong các cảnh báo tấn công, bao nhiêu là đúng?”. Recall, cũng gọi là Detection Rate khi xét lớp Attack, trả lời “trong các tấn công thực, phát hiện được bao nhiêu?”. FAR/FPR đo tỷ lệ lưu lượng bình thường bị báo nhầm. Cần kiểm tra lớp dương và cách lấy trung bình trước khi gọi một giá trị Recall trong JSON là Detection Rate.

### 3.2. Vì sao dùng F1-macro

Tính F1 cho từng lớp rồi lấy trung bình đều:

`F1-macro = (F1_lớp_1 + ... + F1_lớp_C) / C`.

Mỗi lớp có cùng trọng số, kể cả lớp hiếm. **F1-weighted** cũng lấy trung bình F1 từng lớp nhưng nhân trọng số theo số mẫu, nên lớp lớn ảnh hưởng nhiều hơn. Báo cáo chủ yếu dùng F1-macro để nhìn rõ khả năng nhận biết các lớp ít mẫu.

Ví dụ minh họa: ba lớp có F1 lần lượt là 0,99; 0,90; 0,10 thì F1-macro chỉ khoảng 0,6633. Lớp cuối đang được nhận biết rất kém, dù lớp phổ biến có thể rất tốt.

F1 `0,9975` tương đương `99,75%`. Tăng từ `0,8553` lên `0,8658` là tăng **1,05 điểm phần trăm**, không phải tăng 10,5 điểm.

### 3.3. Số chiều và thời gian

Gọi `D` là số cột đầu vào sau tiền xử lý, `k` là số cột giữ lại:

`Tỷ lệ giảm chiều = (D − k) / D × 100%`.

Ví dụ 52 xuống 5 là giảm khoảng 90,38%. Giảm chiều không phải xóa 90% số dòng và cũng không có nghĩa kích thước tệp hay mọi loại thời gian đều giảm đúng tỷ lệ đó.

Cần phân biệt **T_FS** (tìm đặc trưng), **T_train** (huấn luyện mô hình cuối) và **T_inference** (dự đoán). Bảng chính của báo cáo đưa thời gian huấn luyện, không phải tổng chi phí pipeline. Wrapper có thể tốn nhiều phút tìm kiếm để đổi lấy một mô hình huấn luyện nhanh hơn. Báo cáo chưa cung cấp đầy đủ DR, FAR, F1-weighted và thời gian suy luận cho từng phương án, nên tài liệu này không tự bổ sung các con số đó.

## 4. Dữ liệu sử dụng

### 4.1. Tổng quan ba bộ dữ liệu

![Phân bố nhị phân: CICIDS2017 nhiều Normal, CICDDoS2019 nhiều Attack. Tỷ lệ làm tròn theo Bảng 3.2.](assets/pipeline_guide/22-label-balance.png)

*Phân bố nhị phân: CICIDS2017 nhiều Normal, CICDDoS2019 nhiều Attack. Tỷ lệ làm tròn theo Bảng 3.2.*

| Bộ dữ liệu | Train sau xử lý | Test | Đặc trưng đầu vào D | Nhãn gốc | Cách chia |
|:---|---:|---:|---:|:---|:---|
| CICIDS2017 | 2.016.472 | 504.118 | 52 | 7 lớp | Một tệp, phân tầng 80/20 |
| UNSW-NB15 | 175.341 | 82.332 | 42 | 10 lớp | Train/test có sẵn |
| CICDDoS2019 | 562.649 | 561.000 | 67 | 8 lớp train, 11 lớp test | Các tệp theo loại tấn công |

Số đặc trưng này là của phiên bản và quy trình trong báo cáo. Không áp dụng mặc định cho mọi bản tải về có cùng tên bộ dữ liệu. NSL-KDD có trong phạm vi dự án nhưng không nằm trong ba bộ thực nghiệm chính của báo cáo Word này.

### 4.2. CICIDS2017

Báo cáo dùng bản Kaggle “Cleaned & Preprocessed”, có 2.520.751 dòng ban đầu, 52 đặc trưng số và nhãn `Attack Type`. Bảy lớp là Normal Traffic, DoS, DDoS, Port Scanning, Brute Force, Web Attacks và Bots.

Tỷ lệ nhị phân khoảng 83,1% Normal và 16,9% Attack. Chia ngẫu nhiên giúp các phần dữ liệu tương đối giống nhau, nhưng kết quả đó chưa chứng minh mô hình hoạt động tương đương khi chuyển sang ngày thu thập hoặc hệ thống mạng khác.

Tổng train và test trong bảng là 2.520.590, ít hơn dữ liệu đầu vào 161 dòng. Đây là hai giai đoạn đếm khác nhau; không nên coi số mẫu EDA và số mẫu cuối là cùng một đại lượng.

### 4.3. UNSW-NB15

Hai tệp train và test có sẵn; mỗi tệp có 45 cột. Loại `id` và các cột nhãn còn 42 đặc trưng, gồm ba cột dạng phân loại `proto`, `service`, `state`.

Nhãn đa lớp gồm Normal và chín nhóm: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms. Train có khoảng 31,9% Normal và 68,1% Attack. Bài toán đa lớp khó hơn rõ rệt so với CICIDS2017 qua F1-macro.

### 4.4. CICDDoS2019

Bản dùng trong báo cáo có 88 cột thô, gộp từ 7 tệp train và 10 tệp test. Train ban đầu có 569.650 dòng; sau xử lý còn 562.649. Test có 561.000 dòng. Dữ liệu train nhị phân khoảng 10% Normal và 90% Attack.

Sau loại định danh, 12 cột phương sai gần 0 và `Inbound`, còn 67 đặc trưng. DNS, NTP, SNMP, TFTP có ở test nhưng chưa có ở train; Portmap chỉ có ở train.

Với nhị phân, các loại tấn công mới vẫn thuộc nhãn Attack đã biết. Với đa lớp, mô hình chưa học tên các lớp mới, nên không thể kỳ vọng bộ phân loại có giám sát thông thường tự gọi đúng tên chúng. Đây là lý do cần đọc hai bài toán riêng biệt.

## 5. Phân tích khám phá dữ liệu trước khi huấn luyện

![Minh họa NIDS](assets/pipeline_guide/46-eda.png)

*EDA khảo sát nhãn, giá trị thiếu, dòng trùng và đặc trưng đáng ngờ. Mọi bảng số ví dụ trong tranh chỉ minh họa.*


EDA là bước xem dữ liệu có cấu trúc gì và có vấn đề gì. Đầu vào là bảng dữ liệu; đầu ra là thống kê, biểu đồ và các nghi vấn cần kiểm tra.

### 5.1. Kiểm tra kích thước, kiểu dữ liệu và nhãn

Đếm dòng, cột, giá trị thiếu, vô cực, dòng trùng và số mẫu từng nhãn. Kiểm tra sau khi đổi nhãn nhị phân có thật sự còn hai lớp hay không.

Báo cáo từng gặp lỗi: chương trình chỉ so khớp chính xác `normal` hoặc `benign`, trong khi nhãn là `Normal Traffic`. Tất cả các dòng bị đổi thành Attack. Khi ấy F1 và Accuracy đều 1,0000 vì chỉ còn một đáp án. Sau khi sửa nhận diện chuỗi con, baseline CICIDS2017 còn F1-macro 0,9975. Đây là lý do phải kiểm tra nhãn trước khi tin một kết quả rất cao.

### 5.2. Tương quan giữa các cột

Hệ số tương quan `r` mô tả quan hệ tuyến tính; `|r|` gần 1 cho thấy hai cột biến thiên gần nhau. Báo cáo đếm các cặp có `|r| > 0,95`: CICIDS2017 có 23 cặp, UNSW có 13, CICDDoS2019 có 48.

Các cột như độ dài gói trung bình, độ lệch chuẩn và phương sai có thể chia sẻ thông tin. Giữ nhiều cột gần trùng thông tin làm tăng chi phí mà chưa chắc giúp phân loại. Tuy nhiên tương quan không phải phép thử đầy đủ cho mọi quan hệ phi tuyến.

### 5.3. Mutual Information và entropy nhãn

**Entropy** diễn tả mức bất định của nhãn trước khi nhìn đặc trưng. **Mutual Information**, MI, diễn tả lượng thông tin về nhãn mà một đặc trưng cung cấp. MI cao nghĩa là cột đó giúp phân biệt nhãn tốt hơn theo cách đo này.

| Bộ dữ liệu | Cột MI cao nhất | MI | Entropy nhãn | MI/entropy | Phương sai giải thích bởi PCA 2 thành phần |
|:---|:---|---:|---:|---:|---:|
| CICIDS2017 | Average Packet Size | 0,3275 | 0,4498 | 72,8% | 82,2% |
| UNSW-NB15 | sbytes | 0,4436 | 0,6262 | 70,8% | 99,1% |
| CICDDoS2019 | Average Packet Size | 0,2551 | 0,3178 | 80,3% | 99,7% |

MI bằng 80,3% entropy không có nghĩa Accuracy là 80,3%. Đây là hai đại lượng khác nhau. Ngưỡng MI vượt 80% entropy trong script là tín hiệu cần xem xét, không phải bằng chứng tự động rằng dữ liệu rò rỉ.

### 5.4. PCA dùng để quan sát

PCA tạo các trục mới là tổ hợp các cột cũ để giữ nhiều phương sai và vẽ dữ liệu trong hai chiều. Tỷ lệ 99,7% trong bảng là tỷ lệ phương sai được hai trục biểu diễn, không phải tỷ lệ phân loại đúng.

PCA ở đây phục vụ EDA. FS giữ cột gốc, còn PCA tạo cột mới. Hai kỹ thuật đều có thể giảm chiều nhưng không cùng cách làm. Tỷ lệ phương sai còn phụ thuộc thang đo đầu vào, nên không dùng riêng con số PCA để kết luận dữ liệu chỉ cần hai đặc trưng cho phân loại.

### 5.5. Rà soát artifact và ablation

**Artifact** là dấu vết của cách thu thập hoặc dựng dữ liệu mà mô hình có thể tận dụng thay cho quy luật hành vi cần học. Script audit xem độ quan trọng của Decision Tree; nếu một cột chiếm ít nhất 50%, hoặc hai cột đầu chiếm ít nhất 80%, script thử bỏ cột và huấn luyện lại.

**Ablation** là chủ động bỏ hoặc thêm một thành phần để xem kết quả thay đổi thế nào. Mức giảm F1 lớn giúp phát hiện sự phụ thuộc, nhưng vẫn cần phân tích ý nghĩa cột để kết luận nguyên nhân.

![Baseline trước và sau loại Inbound](assets/pipeline_guide/25-inbound.png)

*Đối chiếu Bảng 3.3: F1 baseline DT tăng sau khi bỏ Inbound. Đây là phép đo trên dữ liệu báo cáo.*

Trên CICDDoS2019, `Inbound` chiếm 77,3% độ quan trọng trong hai tập cột đã chọn. Train có 99,9% Attack với Inbound = 1. Đây là dấu hiệu mô hình đang dựa vào hướng thu thập lưu lượng. Loại cột này làm baseline DT tăng từ 0,8654 lên 0,9908; báo cáo loại nó và chạy lại thực nghiệm.

Ngược lại, `Average Packet Size` có MI cao nhưng độ dài gói khác nhau theo loại giao thức là lời giải thích hợp lý, nên được giữ. Audit CICIDS2017 và UNSW cũng thấy các cột quan trọng nhưng bỏ chúng không làm F1 giảm đáng kể: `Bwd Packet Length Std` cho 0,9943 → 0,9943; `sttl` cho 0,8631 → 0,8656. Các phép thử này không phát hiện vấn đề tương tự Inbound; chúng chưa chứng minh dữ liệu hoàn toàn không có artifact.

## 6. Tiền xử lý dữ liệu từng bước

![Minh họa NIDS](assets/pipeline_guide/37-data-preprocessing.png)

*Từ gói tin đến bảng luồng: kiểm tra dữ liệu, chia train/test trước khi học tham số, xử lý thiếu và mã hóa. Đây là quy trình nên áp dụng; đối chiếu Mục 11 về mã hiện tại.*


Phần này giải thích mục đích các thao tác trong báo cáo và trình tự nên áp dụng để giữ test độc lập. Những khác biệt trong mã hiện tại được ghi riêng ở Mục 11.

### 6.1. Đọc CSV và chuẩn hóa tên cột

Đọc bảng, bỏ khoảng trắng đầu/cuối tên cột. Ví dụ `" Destination Port"` trở thành `"Destination Port"`, tránh xem cùng một cột là hai tên khác nhau.

### 6.2. Xác định các phần dữ liệu

Với UNSW và CICDDoS2019, dùng tệp train/test có sẵn. Với một tệp CICIDS2017, nguyên tắc đúng là xác định nhãn, chia phân tầng 80/20 trước khi học tham số tiền xử lý.

Giữ riêng nhãn `y` và đặc trưng `X`. Các nhãn hoặc cột định danh không được vô tình trở thành đầu vào.

### 6.3. Xử lý giá trị thiếu và vô cực

Một phép tính như tốc độ trên thời lượng bằng 0 có thể tạo vô cực. Đổi vô cực thành giá trị thiếu `NaN`, rồi điền giá trị hợp lý.

Báo cáo dùng trung vị: sắp các giá trị và lấy giá trị giữa, ít bị kéo lệch bởi giá trị cực lớn hơn trung bình. Cách đúng là lưu trung vị từng cột của train và dùng chính các giá trị đó cho test. Không tính lại trung vị trên test.

### 6.4. Dòng trùng, cột gần hằng và cột định danh

Dòng trùng có thể làm một mẫu bị đếm nhiều lần. Cột gần hằng có rất ít biến thiên, thường ít giúp phân biệt lớp. Danh sách cột gần hằng cần được xác định trên train rồi áp dụng thống nhất.

Loại các cột như Flow ID, địa chỉ IP, cổng nguồn, Timestamp theo cấu hình CICDDoS2019 và `id` của UNSW. Các cột này có thể khiến mô hình ghi nhớ môi trường. `Destination Port` vẫn được giữ trong thực nghiệm, nhưng ảnh hưởng của nó được kiểm tra riêng. `Inbound` bị loại sau phân tích chất lượng dữ liệu.

### 6.5. Chuyển nhãn

Nhị phân: nhãn chứa `normal` hoặc `benign`, không phân biệt hoa thường, được xem là Normal; phần còn lại là Attack theo quy tắc của báo cáo. Phải đối chiếu quy tắc với tên nhãn thực của từng bộ dữ liệu.

Đa lớp: giữ tên nhóm tấn công, sau đó mã hóa thành số. Số 0, 1, 2 chỉ là mã; không mang ý nghĩa lớp nào “lớn hơn”. Nhãn chỉ có ở test được ánh xạ thành mã `Unknown` riêng, lưu trong `label_classes.json`. Ánh xạ này giúp chấm điểm rõ ràng, không tự tạo cho mô hình khả năng phát hiện Unknown.

### 6.6. Mã hóa cột phân loại

UNSW có `proto`, `service`, `state` dạng chữ. Label Encoding đổi mỗi giá trị thành một mã số, học ánh xạ từ train rồi áp dụng cho test. Ví dụ minh họa `tcp → 0`, `udp → 1`.

Các mã là quy ước, không có khoảng cách tự nhiên. Cây vẫn có thể tạo ngưỡng trên các mã này, nên cách mã hóa có thể ảnh hưởng kết quả. Đây là cách báo cáo sử dụng, không phải lựa chọn duy nhất cho dữ liệu phân loại.

### 6.7. Chuẩn hóa MinMax

Với từng cột, phép biến đổi là `x_mới = (x − min_train) / (max_train − min_train)`, với cột hằng cần xử lý riêng. Giá trị train thường nằm trong [0, 1]; test có thể vượt khoảng này nếu vượt cực trị train và không bật cắt ngưỡng.

Mục đích là thống nhất thang đo. Các mô hình cây thường ít cần chuẩn hóa hơn các mô hình dựa vào khoảng cách, nhưng pipeline báo cáo vẫn dùng MinMax. Tham số phải học từ train.

### 6.8. Lưu dữ liệu và tham số

Đầu ra tiền xử lý gồm `X_train`, `X_test`, `y_train`, `y_test`, danh sách cột, scaler và bộ mã hóa. Các phương án FS nhận cùng dữ liệu này để so sánh công bằng.

SMOTE là kỹ thuật sinh mẫu cho lớp ít dữ liệu. Dù xuất hiện trong quy ước dự án và mã hỗ trợ, bảng thiết lập của báo cáo không xác nhận SMOTE đã được bật cho các kết quả chính; không nên kể nó như một bước đã chạy. Nếu dùng, chỉ áp dụng trên phần huấn luyện.

## 7. Các phương pháp chọn đặc trưng hoạt động thế nào

![Hai phương pháp lai trong NIDS](assets/pipeline_guide/40-hybrids-illustrated.png)

*Hai luồng xử lý bảng đặc trưng mạng: Hybrid 1 tiền lọc rồi tìm kiếm; Hybrid 2 phối hợp MI và SHAP. Các giá trị và tên cột ví dụ trong tranh chỉ để minh họa, không phải số liệu của báo cáo.*

### 7.1. Full features làm baseline

Huấn luyện và đánh giá với toàn bộ `D` cột sau tiền xử lý. Đây là mốc để biết FS có ích hay không. Bảy phương án trong báo cáo gồm một baseline và sáu phương án FS.

### 7.2. Filter bằng Mutual Information

![Minh họa NIDS](assets/pipeline_guide/48-filter.png)

*MI chấm từng cột với nhãn rồi xếp hạng và giữ khoảng 30% số cột. Các ví dụ trong tranh không phải kết quả thực nghiệm.*


Lấy mẫu 20.000 dòng từ train, tính MI giữa mỗi cột và nhãn, sắp giảm dần rồi giữ khoảng 30% số cột. Số cột trong kết quả là 16/52, 13/42 và 20/67.

Ví dụ minh họa: ba cột A, B, C có MI lần lượt 0,30; 0,28; 0,05. Filter ưu tiên A và B. Nhưng A và B có thể chứa gần cùng thông tin, trong khi C có ích khi kết hợp với A. Chấm từng cột riêng không đánh giá đầy đủ sự phối hợp đó.

Ưu điểm là nhanh, dễ hiểu và không cần thử hàng trăm tập con. Hạn chế là phải chọn trước tỷ lệ giữ và có thể bỏ lỡ tương tác.

### 7.3. Wrapper và hàm fitness

![Minh họa NIDS](assets/pipeline_guide/47-mask.png)

*Vector 1–0–1–0 giữ Duration và Port, bỏ Packets và IAT; nhãn y nằm riêng. Đây là ví dụ lựa chọn cột, không phải tập cột thực nghiệm.*


Wrapper thử một tập cột bằng cách huấn luyện mô hình rồi chấm trên validation. Với `D` cột có khoảng `2^D` tập con; 52 cột đã có khoảng 4,5 triệu tỷ tổ hợp, nên không thể thử hết.

GA và PSO dùng vector bit để biểu diễn lựa chọn. Ví dụ `[1, 0, 1, 0, 0]` nghĩa là giữ cột thứ nhất và thứ ba.

Fitness của GA/PSO trong báo cáo:

`fitness = 0,9 × F1_macro_validation − 0,1 × k/n`.

`n` là số cột trong không gian đang tìm, `k` là số được chọn. Phần đầu thưởng chất lượng, phần sau phạt tập quá lớn. Fitness là điểm tối ưu nội bộ, không phải F1 test. Ví dụ F1 validation 0,95 và giữ 10/50 cột cho fitness `0,9 × 0,95 − 0,1 × 0,2 = 0,835`.

Tính fitness dùng 5.000 dòng với CICIDS2017 và CICDDoS2019, toàn bộ train với UNSW; 30% dữ liệu dùng tìm kiếm làm validation. Vì đây chỉ là mẫu hoặc một lần chia, tập tối ưu trên validation có thể không tối ưu trên test.

### 7.4. GA từng bước

![Minh họa NIDS](assets/pipeline_guide/38-ga-illustrated.png)

*Các cá thể là vector bit chọn cột. GA chấm chúng bằng Decision Tree trên validation, chọn cha mẹ, lai ghép, đột biến và giữ cá thể tốt. Test đứng ngoài quá trình tìm kiếm.*


GA, Genetic Algorithm, mô phỏng chọn lọc qua các thế hệ:

1. Tạo 30 vector bit ngẫu nhiên; mỗi bit có xác suất 0,5 được chọn.
2. Huấn luyện mô hình DT cho từng tập cột, tính fitness trên validation.
3. Chọn cha mẹ bằng tournament: lấy một nhóm 3 cá thể rồi ưu tiên cá thể tốt hơn.
4. Lai ghép đồng nhất với xác suất 0,8 để tạo các tổ hợp bit mới.
5. Đột biến: lật mỗi bit với xác suất `1/n`.
6. Giữ ít nhất một cá thể tốt nhất, gọi là elitism; lặp trong 30 thế hệ.
7. Lấy nghiệm tốt nhất; nếu vector không chọn cột nào thì sửa để có ít nhất một cột.

Chương trình lưu đệm kết quả của vector đã thử để tránh huấn luyện lại. GA không bảo đảm tìm được tối ưu toàn cục. Quần thể 30 và 30 thế hệ cũng không đồng nghĩa luôn có đúng 900 lần huấn luyện khác nhau vì có nghiệm lặp và cache.

### 7.5. PSO nhị phân từng bước

![Minh họa NIDS](assets/pipeline_guide/39-pso-illustrated.png)

*Hạt mang vector chọn đặc trưng, ghi nhớ pbest và gbest, rồi cập nhật bit qua vận tốc và sigmoid. Không gian tọa độ chỉ là cách hình dung quá trình tìm kiếm.*


PSO giữ 30 hạt. Mỗi hạt có **vị trí** là vector chọn cột và **vận tốc** là các số thực điều khiển xác suất thay đổi. Hạt nhớ nghiệm tốt nhất của chính nó, `pbest`; cả bầy nhớ nghiệm tốt nhất chung, `gbest`.

Ở mỗi vòng, vận tốc kết hợp hướng di chuyển cũ, hướng về `pbest` và hướng về `gbest`. Hàm sigmoid `1 / (1 + exp(−v))` đổi vận tốc thành xác suất để lấy bit mới. Sau đó tính fitness và cập nhật các nghiệm tốt nhất.

Thiết lập: 30 vòng, trọng số quán tính giảm từ 0,7 xuống 0,4, hai hệ số học bằng 1,5, vận tốc giới hạn trong [−4, 4].

**Hội tụ sớm** là bầy mất đa dạng hoặc ngừng cải thiện quá sớm. Báo cáo quan sát best fitness không đổi ngay từ đầu và thêm hai cơ chế: mỗi vòng đột biến 15% số hạt với xác suất lật bit `3/n`; sau 5 vòng không cải thiện, khởi tạo lại 30% hạt kém nhất.

Trên CICIDS2017, fitness tăng từ 0,84381 lên 0,84829 ở vòng 6 và 0,84962 ở vòng 12; số cột giảm từ 22 xuống 14. Việc tìm kiếm có tiến triển hơn không bảo đảm F1 test cao hơn: báo cáo ghi F1 test thay từ 0,9981 xuống 0,9923.

### 7.6. Hybrid 1 kết hợp MI với GA hoặc PSO


Trước tiên MI giữ 50% số cột. Sau đó GA hoặc PSO chỉ tìm trong phần đã giữ. Ví dụ từ 52 cột xuống khoảng 26 cột ứng viên, rồi tìm tập con trong 26 cột đó.

Fitness bổ sung F1 của lớp ít mẫu nhất trong train:

`fitness = 0,8 × F1_macro + 0,1 × F1_lớp_hiếm − w_feat × k/n`.

Trong đó `n` là số cột sau tiền lọc. Việc bổ sung một lớp hiếm không bảo đảm mọi lớp hiếm đều cải thiện. Nếu MI đã loại một cột cần thiết, wrapper sau đó không thể chọn lại cột ấy.

Báo cáo dùng `w_feat = 0,05` cho Hybrid 1-GA; Hybrid 1-PSO trên CICIDS2017 và UNSW dùng 0,1. Khi không gian giảm một nửa, cùng một cột bị phạt nhiều hơn nếu giữ nguyên hệ số, vì mẫu số `n` nhỏ hơn. Đây là lý do trọng số ảnh hưởng rõ đến số cột và F1.

### 7.7. Hybrid 2 kết hợp MI với SHAP từ XGBoost


XGBoost xây nhiều cây nối tiếp, mỗi cây bổ sung để giảm sai số của tập cây hiện tại. SHAP giải thích mức đóng góp của từng đặc trưng vào dự đoán của mô hình đó.

Quy trình Hybrid 2:

1. Tính MI trên mẫu 20.000 dòng train.
2. Huấn luyện XGBoost với 200 cây, độ sâu tối đa 6, và tính SHAP.
3. Lấy trung bình trị tuyệt đối SHAP của từng cột; với đa lớp, lấy qua các lớp.
4. Chuẩn hóa riêng điểm MI và điểm SHAP về [0, 1].
5. Tính `score = 0,5 × MI_chuẩn_hóa + 0,5 × SHAP_chuẩn_hóa`.
6. Giữ khoảng 30% cột điểm cao nhất.

Trị tuyệt đối giúp đo độ lớn đóng góp, không để các đóng góp dương và âm triệt tiêu. Điểm SHAP phụ thuộc mô hình đã học và không chứng minh quan hệ nhân quả. Hybrid 2 không dùng vòng tìm kiếm kiểu GA/PSO; dùng XGBoost để chọn cột cũng không bắt buộc bộ phân loại cuối phải là XGBoost.

### 7.8. Các bộ phân loại cuối

**Decision Tree** là một cây các điều kiện chia dữ liệu. **Random Forest** kết hợp nhiều cây với dữ liệu và đặc trưng được ngẫu nhiên hóa. **XGBoost** kết hợp các cây học nối tiếp. Báo cáo dùng DT cho fitness để giảm chi phí, rồi đánh giá một số tập cột bằng RF 100 cây và XGBoost 200 cây.

`eval_selected.py` nhận tập cột đã chốt và đổi bộ phân loại cuối. Nó giúp kiểm tra cùng một tập cột phù hợp với các mô hình ra sao, không phải chạy lại GA để tối ưu riêng cho RF hoặc XGBoost.

## 8. Kết quả tổng quan nên nhớ trước

| Tình huống | Kết quả đáng chú ý | Cách hiểu |
|:---|:---|:---|
| CICIDS2017 nhị phân, DT | Filter 16/52 cột: F1 0,9978; full: 0,9975 | Giảm chiều mạnh, chênh lệch chất lượng rất nhỏ |
| CICIDS2017 nhị phân, DT | GA 5/52 cột: F1 0,9959 | Tập rất nhỏ vẫn giữ chất lượng cao |
| UNSW nhị phân, DT | Hybrid 1-PSO 7/42: 0,8658; full: 0,8553 | Một số tập con tốt hơn full trong lần chạy này |
| UNSW đa lớp, DT | Hybrid 1-GA 6/42: 0,5550; full: 0,4896 | Giảm chiều có thể có ích cho bài toán khó hơn |
| CICDDoS2019 nhị phân | Thứ hạng thay đổi khi đổi DT/RF/XGBoost | Không thể xếp hạng FS chỉ bằng một bộ phân loại |
| CICDDoS2019 đa lớp | 89,4% test thuộc nhãn chưa thấy ở train | Không phù hợp để xếp hạng FS theo cách học đa lớp thông thường này |

Các kết quả chính chỉ có một seed 42. Chênh lệch nhỏ là quan sát của lần chạy, chưa đủ để khẳng định ưu thế có ý nghĩa thống kê.

## 9. Kết quả chi tiết và cách diễn giải

### 9.1. CICIDS2017 nhị phân với Decision Tree

![So sánh F1-macro trên cùng thang 0–1. Chênh lệch nhỏ cần đọc giá trị ghi bên điểm.](assets/pipeline_guide/15-f1-cic.png)

*So sánh F1-macro trên cùng thang 0–1. Chênh lệch nhỏ cần đọc giá trị ghi bên điểm.*

![Mức giảm số cột so với 52 đặc trưng đầu vào.](assets/pipeline_guide/16-reduction-cic.png)

*Mức giảm số cột so với 52 đặc trưng đầu vào.*

![Thời gian train cuối, trục logarit; chưa cộng thời gian tìm tập cột.](assets/pipeline_guide/17-train-time-cic.png)

*Thời gian train cuối, trục logarit; chưa cộng thời gian tìm tập cột.*

Nguồn: Bảng 3.5 của báo cáo. Cột giảm chiều được tính thêm từ k/D.

| Phương án | k/D | Giảm chiều % | F1-macro | Accuracy | T_train giây |
|:---|---:|---:|---:|---:|---:|
| Full | 52/52 | 0,00 | 0,9975 | 0,9986 | 150–195 |
| Filter-MI | 16/52 | 69,23 | 0,9978 | 0,9988 | 41,86 |
| GA | 5/52 | 90,38 | 0,9959 | 0,9977 | 8,09 |
| PSO cải tiến | 14/52 | 73,08 | 0,9923 | 0,9957 | 22,56 |
| Hybrid 1-GA | 6/52 | 88,46 | 0,9948 | 0,9971 | 12,34 |
| Hybrid 1-PSO | 10/52 | 80,77 | 0,9945 | 0,9969 | 47,77 |
| Hybrid 2 | 16/52 | 69,23 | 0,9956 | 0,9975 | 43,29 |

Filter tăng F1 0,03 điểm phần trăm, quá nhỏ để kết luận tốt hơn chắc chắn. GA giảm F1 0,16 điểm nhưng huấn luyện nhanh hơn khoảng 18,5–24,1 lần theo khoảng baseline. Đây là sự đánh đổi rõ giữa kích thước và chất lượng.

Thời gian tìm kiếm GA khoảng 26 giây; PSO khoảng 60–100 giây; Hybrid 1 khoảng 15–35 giây cộng khoảng 7 giây MI. Phải cộng các khoản này khi so sánh chi phí xây dựng mô hình một lần.

### 9.2. UNSW-NB15 nhị phân với Decision Tree

![F1-macro nhị phân: GA thuần chọn ít cột nhưng thấp hơn baseline.](assets/pipeline_guide/15-f1-unsw.png)

*F1-macro nhị phân: GA thuần chọn ít cột nhưng thấp hơn baseline.*

![Mức giảm số cột so với 42 đặc trưng đầu vào.](assets/pipeline_guide/16-reduction-unsw.png)

*Mức giảm số cột so với 42 đặc trưng đầu vào.*

![Chi phí train cuối; chi phí wrapper tìm kiếm được giải thích ở phần chữ bên dưới.](assets/pipeline_guide/17-train-time-unsw.png)

*Chi phí train cuối; chi phí wrapper tìm kiếm được giải thích ở phần chữ bên dưới.*

![Hai thiết lập trọng số phạt của Hybrid 1-GA và F1 tương ứng, từ Bảng 3.7.](assets/pipeline_guide/23-penalty.png)

*Hai thiết lập trọng số phạt của Hybrid 1-GA và F1 tương ứng, từ Bảng 3.7.*

Nguồn: Bảng 3.6.

| Phương án | k/D | Giảm chiều % | F1-macro | Accuracy | T_train giây |
|:---|---:|---:|---:|---:|---:|
| Full | 42/42 | 0,00 | 0,8553 | 0,8605 | 2,8–5,5 |
| Filter-MI | 13/42 | 69,05 | 0,8602 | 0,8652 | 1,14 |
| GA | 4/42 | 90,48 | 0,8274 | 0,8354 | 0,32 |
| PSO cải tiến | 15/42 | 64,29 | 0,8569 | 0,8621 | 1,00 |
| Hybrid 1-GA | 3/42 | 92,86 | 0,8632 | 0,8684 | 0,22 |
| Hybrid 1-PSO | 7/42 | 83,33 | 0,8658 | 0,8704 | 0,60 |
| Hybrid 2 | 13/42 | 69,05 | 0,8616 | 0,8665 | 1,48 |

Năm trong sáu phương án FS vượt baseline về F1 trong lần chạy. Hybrid 1-PSO cao nhất, hơn full 1,05 điểm. GA thuần chọn ít cột nhưng thấp hơn full 2,79 điểm; “ít cột hơn” chưa đủ để xem là tốt hơn.

Tìm kiếm GA khoảng 435 giây, PSO 1.640–1.800 giây, Hybrid 1-PSO khoảng 620 giây. Chi phí tìm kiếm lớn hơn nhiều thời gian train cuối của bộ dữ liệu này.

Bảng 3.7 cho thấy ảnh hưởng trọng số phạt của Hybrid 1-GA:

| Bộ dữ liệu | w_feat | k/D | F1-macro |
|:---|---:|---:|---:|
| CICIDS2017 | 0,10 | 4/52 | 0,9865 |
| CICIDS2017 | 0,05 | 6/52 | 0,9948 |
| UNSW-NB15 | 0,10 | 2/42 | 0,8272 |
| UNSW-NB15 | 0,05 | 3/42 | 0,8632 |

Giảm phạt cho phép giữ thêm cột và cải thiện F1 ở hai phép thử này. Đây không phải bằng chứng rằng cứ giảm phạt là luôn tốt.

### 9.3. CICDDoS2019 nhị phân với Decision Tree

![Sau khi loại Inbound, full có F1 cao hơn các phương án FS với Decision Tree.](assets/pipeline_guide/15-f1-ddos.png)

*Sau khi loại Inbound, full có F1 cao hơn các phương án FS với Decision Tree.*

![Mức giảm số cột so với 67 đặc trưng sau xử lý.](assets/pipeline_guide/16-reduction-ddos.png)

*Mức giảm số cột so với 67 đặc trưng sau xử lý.*

![Thời gian huấn luyện cần đọc cùng mức giảm F1, không chỉ nhìn tốc độ.](assets/pipeline_guide/17-train-time-ddos.png)

*Thời gian huấn luyện cần đọc cùng mức giảm F1, không chỉ nhìn tốc độ.*

Nguồn: Bảng 3.8, sau khi loại Inbound và chạy lại.

| Phương án | k/D | Giảm chiều % | F1-macro | Accuracy | T_train giây |
|:---|---:|---:|---:|---:|---:|
| Full | 67/67 | 0,00 | 0,9908 | 0,9966 | 7–12 |
| Filter-MI | 20/67 | 70,15 | 0,8378 | 0,9227 | 6,23 |
| GA | 6/67 | 91,04 | 0,8890 | 0,9518 | 1,18 |
| PSO cải tiến | 25/67 | 62,69 | 0,8698 | 0,9417 | 6,20 |
| Hybrid 1-GA | 4/67 | 94,03 | 0,8845 | 0,9499 | 1,23 |
| Hybrid 1-PSO | 10/67 | 85,07 | 0,8597 | 0,9360 | 1,90 |
| Hybrid 2 | 20/67 | 70,15 | 0,8847 | 0,9496 | 3,08 |

Mọi phương án FS thấp hơn full khoảng 10,18–15,30 điểm F1. Khác biệt lớn hơn hai bộ trước. Chỉ dựa vào bảng này, không thể khẳng định FS giữ chất lượng gần như nguyên vẹn trên cả ba bộ dữ liệu.

### 9.4. Kết quả đa lớp

![Mỗi ô ghi giá trị chính xác. Accuracy gần 1 vẫn có thể đi cùng F1-macro thấp hơn rõ rệt.](assets/pipeline_guide/18-multi-cic.png)

*Mỗi ô ghi giá trị chính xác. Accuracy gần 1 vẫn có thể đi cùng F1-macro thấp hơn rõ rệt.*

![Trên UNSW đa lớp, Hybrid 1-GA có F1-macro cao nhất trong lần chạy báo cáo.](assets/pipeline_guide/18-multi-unsw.png)

*Trên UNSW đa lớp, Hybrid 1-GA có F1-macro cao nhất trong lần chạy báo cáo.*

Nguồn: Bảng 3.9. Tất cả dùng Decision Tree; số cột được chọn có thể khác nhị phân vì nhãn và mục tiêu tìm kiếm thay đổi.

| Bộ dữ liệu | Phương án | k/D | F1-macro | Accuracy | T_train giây |
|:---|:---|---:|---:|---:|---:|
| CICIDS2017 | Full | 52/52 | 0,9673 | 0,9983 | 173–191 |
| CICIDS2017 | Filter-MI | 16/52 | 0,9583 | 0,9985 | 45,32 |
| CICIDS2017 | GA | 7/52 | 0,9194 | 0,9959 | 21,03 |
| CICIDS2017 | PSO | 21/52 | 0,9225 | 0,9971 | 59,22 |
| CICIDS2017 | Hybrid 1-GA | 7/52 | 0,8605 | 0,9971 | 21,59 |
| CICIDS2017 | Hybrid 1-PSO | 10/52 | 0,9557 | 0,9985 | 35,08 |
| CICIDS2017 | Hybrid 2 | 16/52 | 0,8762 | 0,9972 | 48,97 |
| UNSW-NB15 | Full | 42/42 | 0,4896 | 0,7325 | 3,5–4,0 |
| UNSW-NB15 | Filter-MI | 13/42 | 0,5117 | 0,7488 | 1,09 |
| UNSW-NB15 | GA | 9/42 | 0,5346 | 0,7536 | 0,42 |
| UNSW-NB15 | PSO | 19/42 | 0,4951 | 0,7411 | 1,59 |
| UNSW-NB15 | Hybrid 1-GA | 6/42 | 0,5550 | 0,7695 | 0,33 |
| UNSW-NB15 | Hybrid 1-PSO | 9/42 | 0,4845 | 0,7454 | 0,42 |
| UNSW-NB15 | Hybrid 2 | 13/42 | 0,4859 | 0,7374 | 0,87 |

Trên CICIDS2017, Hybrid 1-GA vẫn có Accuracy 99,71% nhưng F1-macro chỉ 86,05%. Điều đó cho thấy Accuracy cao có thể che khuất việc nhận biết kém các lớp nhỏ. Filter và Hybrid 1-PSO gần full hơn các phương án còn lại.

Trên UNSW, Hybrid 1-GA đạt 0,5550, cao hơn full 6,54 điểm, với 6/42 cột, giảm 85,71% số chiều. Cách giải thích “giảm nhiễu giúp lớp hiếm” là giả thuyết hợp lý của báo cáo, chưa được chứng minh bằng phân tích từng lớp hay thí nghiệm nhân quả.

Trên CICDDoS2019, có 501.372/561.000 mẫu test, khoảng 89,4%, thuộc nhãn chưa có ở train. F1-macro các phương án chỉ khoảng 0,084–0,116; full là 0,1010. Không dùng các số này để kết luận phương pháp FS nào tốt nhất. Mã Unknown là cách biểu diễn nhãn khi chấm điểm, không phải một lớp mà mô hình đã được huấn luyện để nhận biết.

### 9.5. Đổi bộ phân loại trên cùng tập cột

![CICIDS2017: các ô cùng tập cột nhưng khác bộ phân loại.](assets/pipeline_guide/19-models-cic.png)

*CICIDS2017: các ô cùng tập cột nhưng khác bộ phân loại.*

![UNSW-NB15: so sánh F1 giữa DT, RF và XGBoost.](assets/pipeline_guide/19-models-unsw.png)

*UNSW-NB15: so sánh F1 giữa DT, RF và XGBoost.*

![CICDDoS2019: thứ hạng thay đổi mạnh theo bộ phân loại.](assets/pipeline_guide/19-models-ddos.png)

*CICDDoS2019: thứ hạng thay đổi mạnh theo bộ phân loại.*

![Cùng 10 cột Hybrid 1-PSO nhưng đổi thứ tự đầu vào của XGBoost.](assets/pipeline_guide/24-column-order.png)

*Cùng 10 cột Hybrid 1-PSO nhưng đổi thứ tự đầu vào của XGBoost.*

Nguồn: Bảng 3.10–3.11. Dưới đây là F1-macro nhị phân; các tập cột được giữ nguyên khi đổi mô hình.

| Bộ dữ liệu | Phương án | k/D | DT | RF | XGBoost |
|:---|:---|---:|---:|---:|---:|
| CICIDS2017 | Full | 52/52 | 0,9975 | 0,9980 | 0,9985 |
| CICIDS2017 | Filter-MI | 16/52 | 0,9978 | 0,9982 | 0,9969 |
| CICIDS2017 | GA | 5/52 | 0,9959 | 0,9961 | 0,9959 |
| CICIDS2017 | Hybrid 1-PSO | 10/52 | 0,9945 | 0,9926 | 0,9931 |
| CICIDS2017 | Hybrid 2 | 16/52 | 0,9956 | 0,9965 | 0,9969 |
| UNSW-NB15 | Full | 42/42 | 0,8553 | 0,8636 | 0,8624 |
| UNSW-NB15 | Filter-MI | 13/42 | 0,8602 | 0,8613 | 0,8651 |
| UNSW-NB15 | GA | 4/42 | 0,8274 | 0,8281 | 0,8241 |
| UNSW-NB15 | Hybrid 1-PSO | 7/42 | 0,8658 | 0,8655 | 0,8652 |
| UNSW-NB15 | Hybrid 2 | 13/42 | 0,8616 | 0,8677 | 0,8677 |
| CICDDoS2019 | Full | 67/67 | 0,9908 | 0,8616 | 0,9623 |
| CICDDoS2019 | Filter-MI | 20/67 | 0,8378 | 0,8315 | 0,8787 |
| CICDDoS2019 | GA | 6/67 | 0,8890 | 0,9970 | 0,9608 |
| CICDDoS2019 | Hybrid 1-PSO | 10/67 | 0,8597 | 0,8418 | 0,8290 |
| CICDDoS2019 | Hybrid 2 | 20/67 | 0,8847 | 0,8419 | 0,9529 |

Trên CICDDoS2019, RF với 6 cột GA đạt 0,9970, trong khi RF đủ 67 cột chỉ 0,8616. DT lại có full tốt hơn GA. Vì vậy một tập cột không có chất lượng độc lập hoàn toàn với bộ phân loại.

Tập GA 6 cột gồm `Destination Port`, `Bwd Packet Length Std`, `Fwd Packets/s`, `Packet Length Std`, `SYN Flag Count`, `URG Flag Count`.

XGBoost trên CICDDoS2019 còn nhạy với thứ tự cột. Cùng tập Hybrid 1-PSO: thứ tự JSON đạt 0,8614, thứ tự gốc đạt 0,8290, đảo ngược đạt 0,8279. Báo cáo quan sát kết quả lặp lại ổn định theo từng thứ tự và đưa giả thuyết về xử lý các điểm chia ngang độ lợi. Cơ chế đó chưa được kiểm chứng. Bảng trên dùng thứ tự cột gốc.

### 9.6. Bỏ từng đặc trưng để kiểm tra CICDDoS2019

![Minh họa NIDS](assets/pipeline_guide/49-ablation.png)

*Mỗi biến thể bỏ cột được huấn luyện lại, rồi đánh giá trên cùng test. Đọc số F1 thực tế ở biểu đồ và bảng bên dưới.*


![Số liệu Bảng 3.12: tác động của bỏ Destination Port và URG Flag Count.](assets/pipeline_guide/20-ablation.png)

*Số liệu Bảng 3.12: tác động của bỏ Destination Port và URG Flag Count.*

Nguồn: Bảng 3.12. Tất cả bắt đầu từ full 67 cột, XGBoost ở bảng dưới dùng thứ tự gốc.

| Biến thể | k | F1 DT | F1 RF | F1 XGBoost |
|:---|---:|---:|---:|---:|
| Đủ cột | 67 | 0,9908 | 0,8616 | 0,9623 |
| Bỏ Destination Port | 66 | 0,8550 | 0,8586 | 0,8172 |
| Bỏ URG Flag Count | 66 | 0,8008 | 0,8383 | 0,8441 |
| Bỏ cả hai | 65 | 0,8695 | 0,8462 | 0,8042 |

DT và XGBoost phụ thuộc mạnh vào cặp cột này. Bỏ cả hai không nhất thiết làm điểm giảm nhiều hơn bỏ một, vì mô hình được huấn luyện lại và có thể đổi cấu trúc cây.

XGBoost qua 5 thứ tự cột có F1 trung bình ± độ lệch chuẩn lần lượt: full `0,9570 ± 0,0093`; bỏ cổng đích `0,8160 ± 0,0010`; bỏ URG `0,8430 ± 0,0022`; bỏ cả hai `0,8068 ± 0,0037`. Đây là biến thiên theo thứ tự cột, không phải kết quả nhiều seed cho toàn bộ nghiên cứu.

### 9.7. Thêm cột còn thiếu vào các tập yếu

![Số liệu Bảng 3.13: thêm một cột có thể cải thiện mạnh F1. XGBoost dùng thứ tự JSON.](assets/pipeline_guide/21-add-back.png)

*Số liệu Bảng 3.13: thêm một cột có thể cải thiện mạnh F1. XGBoost dùng thứ tự JSON.*

Nguồn: Bảng 3.13. XGBoost dùng thứ tự JSON, cột thêm vào được đặt cuối; chỉ so sánh trong bảng này để tránh lẫn ảnh hưởng thứ tự.

| Tập cột | k | F1 DT | F1 RF | F1 XGBoost |
|:---|---:|---:|---:|---:|
| Filter gốc | 20 | 0,8378 | 0,8315 | 0,8787 |
| Filter + URG Flag Count | 21 | 0,8849 | 0,8369 | 0,9529 |
| Hybrid 1-PSO gốc | 10 | 0,8597 | 0,8418 | 0,8614 |
| Hybrid 1-PSO + Destination Port | 11 | 0,9428 | 0,9495 | 0,9866 |

Thêm cổng đích cho Hybrid 1-PSO cải thiện cả ba mô hình. Báo cáo nghi ngờ cổng đích phản ánh cấu hình thí nghiệm. URG bằng 0 ở 99,95% Attack train và 99,93% Attack test, so với 50,2% và 57,5% ở Normal. Đây là tín hiệu phân biệt mạnh, nhưng giải thích nó bằng hành vi flood vẫn là giả thuyết.

Không nên suy ra rằng hai cột này đủ để có mô hình tốt trong mọi mạng. Ngay trong thực nghiệm, có cả hai cột vẫn chưa bảo đảm điểm cao ở mọi tập cột và mọi bộ phân loại.

## 10. Đọc các tệp trong dự án để hiểu pipeline

| Tệp hoặc thư mục | Vai trò | Cần nhìn gì |
|:---|:---|:---|
| `preprocessing.py` | Đọc, làm sạch, mã hóa, chuẩn hóa, chia dữ liệu | Nhánh có test riêng và nhánh một tệp |
| `eda.py` | Khảo sát dữ liệu | Phân bố nhãn, MI, tương quan, PCA |
| `audit_feature_artifacts.py` | Kiểm tra cột đáng ngờ | Độ quan trọng và F1 trước/sau ablation |
| `filter_fs.py` | Chọn theo điểm từng cột | Cách tính MI và số cột giữ |
| `feature_selection.py` | GA, PSO và đánh giá | Dữ liệu fitness, vector bit, cache, lịch sử hội tụ |
| `hybrid1_fs.py` | MI rồi GA/PSO | Tỷ lệ tiền lọc, trọng số phạt và lớp hiếm |
| `hybrid2_fs.py` | MI cộng SHAP | Chuẩn hóa hai điểm, xử lý đa lớp |
| `eval_selected.py` | Đổi bộ phân loại cuối | Giữ nguyên tập cột và thứ tự cột |
| `ablation_full.py` | Bỏ cột khỏi full | Đo mức phụ thuộc vào các cột |
| `output/<dataset>/` | Dữ liệu đã xử lý và kết quả | `selected_features.json`, baseline, convergence |
| `eda_output/` | Báo cáo và hình EDA | Giải thích chất lượng dữ liệu trước khi nhìn điểm |

Một `selected_features.json` trả lời tập cột đã chọn là gì và các độ đo được lưu ra sao. `baseline_full_features.json` là mốc đối chứng. `convergence.csv/png` mô tả tiến triển fitness trong tìm kiếm; đường này không phải F1 test qua từng vòng.

Khi đọc một kết quả, lần lượt xác định dataset, chế độ nhãn, dữ liệu đã loại Inbound chưa, phương pháp chọn, bộ phân loại fitness, bộ phân loại cuối, số cột, F1 và loại thời gian đang báo cáo. Với XGBoost trên CICDDoS2019, kiểm tra thêm thứ tự cột.

## 11. Các điểm cần phân biệt giữa báo cáo, nguyên tắc và mã hiện tại

### 11.1. Tiền xử lý trước khi chia dữ liệu trong nhánh một tệp

Trong `preprocessing.py`, hàm `run()` ở nhánh không có `test_path` gọi `fit_transform()` trên toàn bộ dữ liệu rồi mới `train_test_split()`. Như vậy scaler, bộ mã hóa và việc loại cột gần hằng đã nhìn dữ liệu sau này thuộc test. Đây là điểm không đáp ứng quy tắc không rò rỉ trong `AGENTS.md`.

Trình tự cần có là chia dữ liệu thô trước, fit tiền xử lý trên train, transform test. Tài liệu này chỉ giải thích và ghi nhận; các số đã trích từ Word chưa được chạy lại bằng quy trình sửa đổi. Chưa có bằng chứng để định lượng ảnh hưởng của vấn đề này lên từng điểm số.

### 11.2. Điền thiếu trên test chưa tái dùng trung vị train

`clean_missing_and_infinite()` tính trung vị ngay trên bảng nhận vào. Hàm này được gọi trong cả `fit_transform()` và `transform()`, nên nếu test có thiếu, nó dùng trung vị test thay vì trung vị train đã lưu. Hàm còn có bước điền tiến/lùi. Vì vậy không thể mô tả mã hiện tại như một bộ điền thiếu đã fit hoàn toàn trên train.

### 11.3. Phân tầng và lấy mẫu

`_safe_stratified_split()` trong `feature_selection.py` thử chia phân tầng rồi chuyển sang `stratify=None` khi gặp lớp quá hiếm. Báo cáo cũng mô tả hành vi này ở phần đa lớp. Nó trái yêu cầu bắt buộc phân tầng của dự án. Cách xử lý phù hợp cần giữ đủ mẫu từng lớp hoặc điều chỉnh thiết kế đánh giá, thay vì âm thầm bỏ phân tầng.

Các hàm lấy mẫu MI/SHAP trong `filter_fs.py` và `hybrid2_fs.py` sử dụng `X.sample(...)`; mẫu ngẫu nhiên đó không bảo đảm giữ lớp hiếm. Khi đọc kết quả đa lớp, cần nhớ mẫu tính độ quan trọng có thể không đại diện đầy đủ các lớp.

### 11.4. Môi trường chạy được mô tả không thống nhất

Mục môi trường triển khai trong Word ghi Windows, PowerShell, Python 3.13; mục cấu hình ngay sau đó ghi Apple M1 Pro, macOS, Python 3.12 Miniconda. Từ văn bản này chưa xác định chắc bảng nào được đo trên cấu hình nào. Vì vậy dùng thời gian như số liệu mô tả của báo cáo và tránh suy rộng tốc độ tuyệt đối sang máy khác.

### 11.5. Các giới hạn còn lại

Một seed chưa cho biết độ ổn định. Ngưỡng chênh lệch khoảng 0,5 điểm mà báo cáo sử dụng là quy ước diễn giải, không thay thế kiểm định thống kê. Các giải thích về hành vi URG, RF bỏ lỡ tương tác hoặc PSO mắc cực trị cục bộ đều cần thí nghiệm thêm.

Audit và ablation đã tham khảo kết quả test trong quá trình phát triển. Để xác nhận khả năng tổng quát hóa sau các quyết định đó, cần một tập đánh giá độc lập mới, tốt hơn nữa là khác thời gian hoặc môi trường thu thập.

Báo cáo chưa đánh giá đầy đủ FAR/FPR, ROC, độ trễ phát hiện trực tiếp và hoạt động thời gian thực. Thời gian suy luận trên bảng đặc trưng cũng chưa bao gồm chi phí bắt gói và tạo đặc trưng. Vì vậy các kết quả hiện tại chủ yếu chứng minh quy trình thực nghiệm trên dữ liệu có sẵn.

## 12. Tự kiểm tra xem đã hiểu báo cáo chưa

**Vì sao phải có baseline?** Để biết tập con tốt đến đâu so với việc dùng đầy đủ thông tin, chứ không chỉ so các thuật toán với nhau.

**Vì sao không chọn cột bằng test?** Vì test phải mô phỏng dữ liệu chưa thấy; dùng nó để chọn cột làm đánh giá lạc quan.

**GA/PSO có trực tiếp dự đoán Attack không?** Chúng chọn tập cột; bộ phân loại DT, RF hoặc XGBoost mới tạo dự đoán.

**Vì sao F1 fitness và F1 test khác nhau?** Chúng dùng các phần dữ liệu khác nhau, và fitness còn trừ phạt số cột hoặc cộng F1 lớp hiếm.

**Vì sao Hybrid 1 và Hybrid 2 khác nhau?** Hybrid 1 tiền lọc rồi tìm tập con bằng wrapper; Hybrid 2 phối hợp hai loại điểm để xếp hạng cột.

**Vì sao Accuracy đa lớp rất cao mà F1-macro thấp?** Lớp nhiều mẫu có thể được dự đoán tốt trong khi các lớp nhỏ bị bỏ sót.

**Vì sao bỏ một cột lại tăng điểm?** Mô hình có thể ngừng dựa vào một tín hiệu gây nhiễu hoặc không ổn định giữa train và test.

**Vì sao không kết luận GA tốt nhất từ RF 0,9970 trên CICDDoS2019?** Cùng tập cột đó với DT chỉ 0,8890, và tín hiệu còn phụ thuộc cổng đích/URG trong môi trường dữ liệu này.

**Đóng góp chính nên trình bày là gì?** Quy trình so sánh nhiều họ FS, hai cách lai, cơ chế duy trì đa dạng cho PSO, và phân tích chất lượng dữ liệu để giải thích kết quả. Mức độ hiệu quả cần luôn gắn với dataset, chế độ nhãn, bộ phân loại và các giới hạn đã nêu.
