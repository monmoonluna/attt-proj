#!/usr/bin/env python3
"""
scripts/generate_docx_report.py
===============================
Script tự động tạo file Word Báo cáo Kết quả Thực nghiệm Môn học ATTT.
Tuân thủ chuẩn trình bày tài liệu học thuật đại học:
- Trang bìa và thông tin đề tài
- Chương 1: Đặt vấn đề & Mục tiêu
- Chương 2: Bộ dữ liệu & Tiền xử lý (No Data Leakage)
- Chương 3: Các phương pháp Feature Selection (Filter, Wrapper, Hybrid)
- Chương 4: Bảng số liệu thực tế trên cả 4 dataset & Đồ thị hội tụ
- Chương 5: Thảo luận, Hạn chế kỹ thuật (Inbound artifact, PSO hội tụ sớm)
- Chương 6: Hướng phát triển tiếp theo & Kết luận
- Phụ lục: Hướng dẫn công cụ Demo
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from demo_cli import load_dataset_results

def build_report():
    doc = docx.Document()

    # Page Setup A4
    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(0.8)

    # Colors
    COLOR_NAVY = RGBColor(30, 58, 138)
    COLOR_TEXT = RGBColor(30, 41, 59)
    COLOR_MUTED = RGBColor(100, 116, 139)

    def set_para_font(p, name="Times New Roman", size=11, bold=False, italic=False, color=COLOR_TEXT, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
        p.alignment = align
        p.paragraph_format.line_spacing = 1.25
        p.paragraph_format.space_after = Pt(4)
        for run in p.runs:
            run.font.name = name
            run.font.size = Pt(size)
            run.bold = bold
            run.italic = italic
            run.font.color.rgb = color

    def add_heading_1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        set_para_font(p, size=13.5, bold=True, color=COLOR_NAVY, align=WD_ALIGN_PARAGRAPH.LEFT)
        return p

    def add_heading_2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        set_para_font(p, size=12, bold=True, color=RGBColor(51, 65, 85), align=WD_ALIGN_PARAGRAPH.LEFT)
        return p

    def add_body_p(text, italic=False, bold=False):
        p = doc.add_paragraph()
        run = p.add_run(text)
        set_para_font(p, size=11, bold=bold, italic=italic)
        return p

    def add_bullet(text, bold_prefix=""):
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.2
        if bold_prefix:
            r1 = p.add_run(bold_prefix)
            r1.bold = True
            r1.font.name = "Times New Roman"
            r1.font.size = Pt(11)
            r1.font.color.rgb = COLOR_TEXT
        r2 = p.add_run(text)
        r2.font.name = "Times New Roman"
        r2.font.size = Pt(11)
        r2.font.color.rgb = COLOR_TEXT
        return p

    def set_cell_background(cell, fill_hex):
        w_ns = nsdecls('w')
        shading_elm = parse_xml(f'<w:shd {w_ns} w:fill="{fill_hex}"/>')
        cell._tc.get_or_add_tcPr().append(shading_elm)

    def style_table(table):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        w_ns = nsdecls('w')
        for r_idx, row in enumerate(table.rows):
            for c_idx, cell in enumerate(row.cells):
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                tcPr = cell._tc.get_or_add_tcPr()
                tcMar = OxmlElement('w:tcMar')
                for m in ['top', 'bottom']:
                    node = OxmlElement(f'w:{m}')
                    node.set(qn('w:w'), '100')
                    node.set(qn('w:type'), 'dxa')
                    tcMar.append(node)
                for m in ['left', 'right']:
                    node = OxmlElement(f'w:{m}')
                    node.set(qn('w:w'), '140')
                    node.set(qn('w:type'), 'dxa')
                    tcMar.append(node)
                tcPr.append(tcMar)
                
                borders = parse_xml(f'''
                    <w:tcBorders {w_ns}>
                        <w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
                        <w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
                        <w:left w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
                        <w:right w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
                    </w:tcBorders>
                ''')
                tcPr.append(borders)
                
                for p in cell.paragraphs:
                    p.paragraph_format.line_spacing = 1.15
                    p.paragraph_format.space_after = Pt(2)
                    for run in p.runs:
                        run.font.name = "Times New Roman"
                        run.font.size = Pt(9.5)
                        if r_idx == 0:
                            run.bold = True
                            run.font.color.rgb = RGBColor(255, 255, 255)
                
                if r_idx == 0:
                    set_cell_background(cell, "1E3A8A")
                elif r_idx % 2 == 1:
                    set_cell_background(cell, "F8FAFC")

    # ----------------- COVER & HEADER -----------------
    p_top = doc.add_paragraph()
    p_top.add_run("BÁO CÁO KẾT QUẢ THỰC NGHIỆM ĐỀ TÀI MÔN HỌC")
    set_para_font(p_top, size=15, bold=True, color=COLOR_NAVY, align=WD_ALIGN_PARAGRAPH.CENTER)

    p_title = doc.add_paragraph()
    p_title.add_run("PHÁT HIỆN TẤN CÔNG MẠNG DỰA TRÊN LỰA CHỌN ĐẶC TRƯNG NÂNG CAO (NIDS)")
    set_para_font(p_title, size=13, bold=True, color=COLOR_TEXT, align=WD_ALIGN_PARAGRAPH.CENTER)

    p_meta = doc.add_paragraph()
    p_meta.add_run("Học phần: An toàn thông tin | Mã đề tài: 2.18 | Báo cáo tiến độ & kết quả thực nghiệm")
    set_para_font(p_meta, size=10.5, italic=True, color=COLOR_MUTED, align=WD_ALIGN_PARAGRAPH.CENTER)

    p_div = doc.add_paragraph()
    p_div.add_run("―" * 45)
    set_para_font(p_div, size=11, color=COLOR_MUTED, align=WD_ALIGN_PARAGRAPH.CENTER)

    # ----------------- CHƯƠNG 1 -----------------
    add_heading_1("Chương 1. TỔNG QUAN ĐỀ TÀI VÀ MỤC TIÊU NGHIÊN CỨU")

    add_body_p("Hệ thống phát hiện xâm nhập mạng (Network Intrusion Detection System - NIDS) là thành phần bảo mật then chốt trong việc bảo vệ hạ tầng mạng hiện đại trước các nguy cơ tấn công từ chối dịch vụ (DoS/DDoS), mã độc botnet, dò quét cổng và tấn công web. Tuy nhiên, lưu lượng mạng thực tế sinh ra với tốc độ rất lớn và số lượng thuộc tính trích xuất ngày càng nhiều (thường từ 40 đến hơn 80 đặc trưng flow-based). Việc xử lý toàn bộ không gian đặc trưng nguyên thủy này dẫn đến chi phí tính toán cao, tăng độ trễ cảnh báo và dễ gây hiện tượng quá khớp (overfitting) do các đặc trưng dư thừa hoặc nhiễu.")

    add_body_p("Mục tiêu trọng tâm của đề tài mã số 2.18 là nghiên cứu, cài đặt và đánh giá toàn diện các kỹ thuật lựa chọn đặc trưng (Feature Selection - FS), đặc biệt đề xuất mô hình kết hợp Hybrid (Filter kết hợp Metaheuristic Wrapper) nhằm đạt được 3 mục tiêu cụ thể:")
    add_bullet(" Giảm từ 60% đến trên 85% số chiều dữ liệu, loại bỏ tối đa các đặc trưng dư thừa.", "1. Giảm chiều dữ liệu:")
    add_bullet(" Duy trì điểm F1-score (macro) và Accuracy tương đương hoặc vượt trội so với mô hình dùng toàn bộ đặc trưng gốc.", "2. Bảo toàn chất lượng phát hiện:")
    add_bullet(" Tăng tốc độ huấn luyện mô hình từ 3 đến hơn 12 lần, rút ngắn thời gian suy luận (Inference Time) phục vụ cảnh báo thời gian thực.", "3. Tối ưu tài nguyên:")

    if os.path.exists("docs/assets/quy-trinh-nids.png"):
        doc.add_paragraph()
        doc.add_picture("docs/assets/quy-trinh-nids.png", width=Inches(5.5))
        p_cap = doc.add_paragraph("Hình 1.1: Quy trình tổng thể từ tiền xử lý, chọn đặc trưng đến đánh giá phân loại NIDS")
        set_para_font(p_cap, size=9.5, italic=True, color=COLOR_MUTED, align=WD_ALIGN_PARAGRAPH.CENTER)

    # ----------------- CHƯƠNG 2 -----------------
    add_heading_1("Chương 2. BỘ DỮ LIỆU VÀ QUY TRÌNH TIỀN XỬ LÝ (CHỐNG RÒ RỈ DỮ LIỆU)")

    add_body_p("Đề tài tiến hành thử nghiệm độc lập trên 4 bộ dữ liệu lưu lượng mạng chuẩn mực quốc tế:")
    add_bullet(" Bộ dữ liệu kế thừa kinh điển, gồm 39 đặc trưng sau tiền xử lý, dùng làm thước đo kiểm định nhanh pipeline.", "• NSL-KDD:")
    add_bullet(" Bộ dữ liệu mạng hiện đại quy mô lớn, gồm 52 đặc trưng số, bao quát các kịch bản tấn công nguy hiểm: DDoS, DoS, Brute Force, Web Attack, Botnet.", "• CICIDS2017:")
    add_bullet(" Dữ liệu phân bố đa dạng với 42 đặc trưng, phân loại các lớp tấn công tinh vi trong môi trường mạng thực tế.", "• UNSW-NB15:")
    add_bullet(" Bộ dữ liệu chuyên sâu về các dạng tấn công từ chối dịch vụ phân tán quy mô cao.", "• CICDDoS2019:")

    add_heading_2("2.1. Quy tắc nghiêm ngặt chống rò rỉ dữ liệu (No Data Leakage)")
    add_body_p("Trong nghiên cứu NIDS, lỗi phổ biến nhất làm sai lệch kết quả là rò rỉ thông tin từ tập Test vào tập Train. Pipeline tiền xử lý trong lớp NIDSPreprocessor được thiết kế tuân thủ tuyệt đối nguyên tắc:")
    add_bullet("Mọi bộ biến đổi bao gồm MinMaxScaler, StandardScaler, LabelEncoder, bộ lọc phương sai (VarianceThreshold) và bộ chọn Filter CHỈ ĐƯỢC FIT TRÊN TẬP TRAIN. Tập Test hoàn toàn chỉ áp dụng transform dựa trên các tham số đã học.")
    add_bullet("Phân tầng nhãn nghiêm ngặt (Stratified Sampling): Mọi bước chia tách Train/Val và lấy mẫu phụ (subsampling) để tính hàm thích nghi cho GA/PSO đều bắt buộc áp dụng stratify=y nhằm bảo toàn tỷ lệ các lớp tấn công thiểu số (rare attacks).")

    # ----------------- CHƯƠNG 3 -----------------
    add_heading_1("Chương 3. CÁC PHƯƠNG PHÁP LỰA CHỌN ĐẶC TRƯNG ĐÃ TRIỂN KHAI")

    add_body_p("Mã nguồn trong repository đã hiện thực hóa đầy đủ 3 nhóm phương pháp theo đúng đề cương nghiên cứu:")
    add_bullet("Mutual Information (MI), Chi-Square, ANOVA F-test và phân tích ma trận tương quan. Phương pháp này xếp hạng độc lập từng đặc trưng với tốc độ tính toán cực nhanh.", "1. Nhóm Filter:")
    add_bullet("Giải thuật di truyền (Genetic Algorithm - GA) và Tối ưu hóa bầy đàn nhị phân (Binary PSO). Không gian tìm kiếm là vector nhị phân độ dài D (bit 1 giữ, bit 0 bỏ).", "2. Nhóm Wrapper Metaheuristic:")
    add_bullet("Filter lọc trước 50% đặc trưng có điểm thấp -> GA/PSO tinh chọn sâu trên không gian đã thu hẹp. Hàm thích nghi (Fitness) được thiết kế tối ưu đồng thời: Fitness = α * F1_macro + (1 - α) * (1 - k/D).", "3. Nhóm đề xuất Hybrid:")

    # ----------------- CHƯƠNG 4 -----------------
    add_heading_1("Chương 4. KẾT QUẢ THỰC NGHIỆM VÀ PHÂN TÍCH ĐỐI CHIẾU")

    add_body_p("Dưới đây là các bảng số liệu kết quả thực nghiệm thực tế được xuất trực tiếp từ các artifacts định lượng trong repository:")

    def build_docx_table(dataset_name, caption_text):
        results = load_dataset_results("output", dataset_name)
        if not results:
            return
        p_cap = doc.add_paragraph(caption_text)
        set_para_font(p_cap, size=10, bold=True, color=COLOR_NAVY, align=WD_ALIGN_PARAGRAPH.CENTER)
        
        t = doc.add_table(rows=len(results) + 1, cols=7)
        headers = ["Phương pháp", "Số ĐT (k)", "Giảm (%)", "Accuracy", "F1-Macro", "T_train (s)", "T_infer (ms)"]
        for i, h in enumerate(headers):
            t.cell(0, i).paragraphs[0].text = h
            if i > 0:
                t.cell(0, i).paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
                
        orig_k = results[0].get("n_features_total") or results[0].get("n_features_selected") or 50
        for r_idx, r in enumerate(results):
            k = r["n_features_selected"]
            tot = r.get("n_features_total", orig_k) or orig_k
            red = ((tot - k) / tot * 100) if tot > 0 else 0
            row_cells = t.rows[r_idx + 1].cells
            
            row_cells[0].paragraphs[0].text = r.get("method_name", "Unknown")
            row_cells[1].paragraphs[0].text = str(k)
            row_cells[2].paragraphs[0].text = f"{red:.1f}%"
            row_cells[3].paragraphs[0].text = f"{r.get('accuracy', 0)*100:.2f}%"
            row_cells[4].paragraphs[0].text = f"{r.get('f1_macro', 0)*100:.2f}%"
            row_cells[5].paragraphs[0].text = f"{r.get('train_time_sec', 0):.2f}"
            row_cells[6].paragraphs[0].text = f"{r.get('inference_time_sec', 0)*1000:.1f}"
            
            for c in range(1, 7):
                row_cells[c].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
                
        style_table(t)
        doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Tables
    build_docx_table("CICIDS2017", "Bảng 4.1: So sánh hiệu năng các phương pháp trên tập dữ liệu CICIDS2017")

    add_heading_2("4.1. Phân tích điểm nhấn trên bộ dữ liệu CICIDS2017")
    add_body_p("Trên bộ dữ liệu CICIDS2017 (52 đặc trưng gốc sau làm sạch):")
    add_bullet("Mô hình Baseline sử dụng toàn bộ 52 đặc trưng đạt F1-macro 99.75% nhưng thời gian huấn luyện mất tới 151.84 giây và thời gian suy luận 208.0 ms.")
    add_bullet("Phương pháp đề xuất Hybrid 1 (Filter + GA v2) rút gọn số đặc trưng từ 52 xuống chỉ còn 6 đặc trưng (tỷ lệ giảm chiều lên tới 88.5%).")
    add_bullet("Thời gian huấn luyện giảm mạnh xuống 12.34 giây (tăng tốc gấp 12.3 lần), thời gian suy luận rút ngắn còn 70.4 ms (nhanh gấp gần 3 lần), trong khi điểm F1-macro vẫn đạt 99.48% (chênh lệch chưa tới 0.27%).")
    add_bullet("Packet Length Variance, Max Packet Length, Total Length of Fwd Packets, Init_Win_bytes_backward, Init_Win_bytes_forward, Bwd Packet Length Std. Đây đều là các đặc trưng nắm bắt hành vi độ dài gói và cơ chế cửa sổ bắt tay TCP đặc thù của các cuộc tấn công DDoS và DoS.", "• Các đặc trưng then chốt được chọn:")

    build_docx_table("CICDDoS2019", "Bảng 4.2: So sánh hiệu năng các phương pháp trên tập dữ liệu CICDDoS2019")
    build_docx_table("UNSW", "Bảng 4.3: So sánh hiệu năng các phương pháp trên tập dữ liệu UNSW-NB15")
    build_docx_table("NSL-KDD", "Bảng 4.4: So sánh hiệu năng các phương pháp trên tập dữ liệu NSL-KDD")

    if os.path.exists("output/CICIDS2017/fs_hybrid1_ga_v2/convergence.png"):
        doc.add_paragraph()
        doc.add_picture("output/CICIDS2017/fs_hybrid1_ga_v2/convergence.png", width=Inches(5.0))
        p_cap = doc.add_paragraph("Hình 4.1: Đồ thị hội tụ hàm thích nghi (Fitness) của thuật toán Hybrid 1 GA v2 trên CICIDS2017")
        set_para_font(p_cap, size=9.5, italic=True, color=COLOR_MUTED, align=WD_ALIGN_PARAGRAPH.CENTER)

    # ----------------- CHƯƠNG 5 -----------------
    add_heading_1("Chương 5. THẢO LUẬN, HẠN CHẾ VÀ BÀI HỌC PHƯƠNG PHÁP LUẬN")

    add_body_p("Trong quá trình thực nghiệm, nhóm nghiên cứu đã ghi nhận và giải quyết một số hiện tượng kỹ thuật và hạn chế quan trọng, mang giá trị bài học phương pháp luận sâu sắc:")

    add_heading_2("5.1. Hiện tượng hội tụ sớm ở thuật toán PSO (Premature Convergence)")
    add_body_p("Ở các phiên bản thử nghiệm ban đầu (PSO v1), đồ thị hội tụ cho thấy hàm fitness đứng yên hoàn toàn từ vòng lặp đầu tiên đến vòng lặp cuối cùng. Nguyên nhân là do quần thể hạt nhị phân bị suy giảm tính đa dạng quá nhanh, các vector vận tốc tiến về 0 khiến thuật toán bị mắc kẹt tại cực trị cục bộ. Nhóm đã nghiên cứu và khắc phục thành công trong phiên bản PSO v2 bằng cách đưa vào cơ chế đột biến định kỳ ngẫu nhiên (periodic random mutation) và tái khởi tạo một phần quần thể (partial re-initialization).")

    add_heading_2("5.2. Phát hiện Artifact ẩn trong bộ dữ liệu CICDDoS2019 (Cột 'Inbound')")
    add_body_p("Đây là đóng góp phương pháp luận có giá trị cao của đề tài. Khi tiến hành phân tích độ quan trọng đặc trưng (Feature Importance), nhóm phát hiện cột 'Inbound' có điểm tầm quan trọng vượt ngưỡng 50% một cách bất thường. Thực chất, đây là một artifact capture trong môi trường mô phỏng của CICDDoS2019 chứ không phải đặc tính lưu lượng tự nhiên. Nếu giữ lại, mô hình sẽ đạt điểm cao ảo do học vẹt cột này. Nhóm đã chủ động cập nhật tiền xử lý loại bỏ hoàn toàn cột 'Inbound', đảm bảo số liệu báo cáo phản ánh năng lực phát hiện thực tế.")

    add_heading_2("5.3. Hạn chế về phạm vi triển khai")
    add_bullet("Đề tài thực nghiệm trên dữ liệu tĩnh dạng flow-based offline, chưa triển khai giám sát trực tiếp trên luồng gói tin thô (raw pcap stream) trong mạng thời gian thực.")
    add_bullet("Các kịch bản tấn công tinh vi kiểu zero-day hoặc tấn công né tránh (adversarial attacks) làm biến dạng phân bố đặc trưng chưa được khảo sát chuyên sâu.")

    # ----------------- CHƯƠNG 6 -----------------
    add_heading_1("Chương 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN TIẾP THEO")

    add_heading_2("6.1. Kết luận")
    add_body_p("Đề tài đã hoàn thành xuất sắc các mục tiêu nghiên cứu ban đầu:")
    add_bullet("Xây dựng thành công quy trình NIDS hoàn chỉnh, module hóa, chống rò rỉ dữ liệu tuyệt đối.")
    add_bullet("Chứng minh phương pháp Hybrid (Filter kết hợp Metaheuristic GA/PSO) là giải pháp tối ưu vượt bậc: giảm tới 88.5% đặc trưng, tăng tốc huấn luyện 12.3 lần mà F1-score vẫn đạt 99.48%.")
    add_bullet("Bộ công cụ demo trực quan gồm Web Dashboard (demo_dashboard.html) và Live CLI (scripts/demo_cli.py) sẵn sàng phục vụ trình diễn kết quả.")

    add_heading_2("6.2. Hướng phát triển tiếp theo")
    add_bullet("1. Mở rộng đánh giá trên các mô hình phân loại sâu (Deep Learning như 1D-CNN, Bi-LSTM) và ensemble models (XGBoost, LightGBM) trên tập đặc trưng đã rút gọn.")
    add_bullet("2. Tối ưu hóa đa mục tiêu (Multi-objective Optimization) sử dụng NSGA-II để tìm tập nghiệm Pareto tối ưu hài hòa giữa Detection Rate, False Alarm Rate và thời gian suy luận.")
    add_bullet("3. Tích hợp pipeline vào hệ thống giám sát mạng thời gian thực (kết nối Zeek/Snort) để kiểm định độ trễ phát hiện trong điều kiện thực tế.")

    # ----------------- PHỤ LỤC -----------------
    add_heading_1("Phụ lục. HƯỚNG DẪN THỰC THI DEMO KẾT QUẢ")
    add_body_p("Nhóm đã chuẩn bị sẵn sàng các công cụ trực quan hóa kết quả để trình chiếu cho Giảng viên:")
    add_bullet("Mở trực tiếp file reports/demo_dashboard.html bằng trình duyệt web để xem giao diện báo cáo tương tác, chuyển đổi giữa 4 bộ dữ liệu và phóng to các biểu đồ EDA/Hội tụ.", "1. Web Dashboard: ")
    add_bullet("Mở file reports/demo_ket_qua.ipynb trong Jupyter Notebook hoặc VS Code để xem mã nguồn phân tích học thuật từng bước.", "2. Jupyter Notebook: ")
    add_bullet("Chạy lệnh: python scripts/demo_cli.py --dataset all --live-nslkdd để thực thi nạp dữ liệu và huấn luyện live trong 3 giây trên Terminal.", "3. Live Terminal Demo: ")

    out_path = os.path.join("docs", "BaoCao_KetQua_NIDS_FeatureSelection.docx")
    doc.save(out_path)
    print(f"Report successfully generated and saved to: {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    build_report()

