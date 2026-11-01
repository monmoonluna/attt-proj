#!/usr/bin/env python3
"""
scripts/generate_architecture_diagram.py
Sơ đồ kiến trúc NIDS chuẩn học thuật, tối giản, thanh lịch:
- Không rườm rà chữ, không đè chữ, không tràn khung.
- Đậm chất sơ đồ nghiên cứu khoa học (người thiết kế).
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_clean_diagram(output_path="docs/assets/quy-trinh-nids.png"):
    # Kích thước 1300 x 950 pt, DPI 300
    fig, ax = plt.subplots(figsize=(13, 9.5), dpi=300)
    ax.set_xlim(0, 1300)
    ax.set_ylim(0, 950)
    ax.axis("off")

    # Bảng màu học thuật trang nhã (Navy & Slate)
    c_bg = "#ffffff"
    c_card_bg = "#f8fafc"
    c_card_border = "#94a3b8"
    c_highlight_bg = "#f0f7ff"
    c_highlight_border = "#2563eb"
    c_text_main = "#0f172a"
    c_text_sub = "#475569"
    c_text_blue = "#1d4ed8"
    c_line = "#475569"

    fig.patch.set_facecolor(c_bg)

    def draw_box(x, y, w, h, bg=c_card_bg, border=c_card_border, lw=1.2, r=10):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=bg, edgecolor=border, linewidth=lw
        )
        ax.add_patch(rect)

    def draw_arrow(x1, y1, x2, y2, col=c_line, lw=1.3):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color=col, lw=lw,
                mutation_scale=12
            )
        )

    # 1. TIÊU ĐỀ SƠ ĐỒ
    ax.text(650, 910, "QUY TRÌNH NGHIÊN CỨU & THỰC NGHIỆM NIDS",
            fontsize=15, fontweight="bold", color=c_text_main, ha="center", va="center")
    ax.text(650, 885, "Hệ thống phát hiện xâm nhập mạng kết hợp lựa chọn đặc trưng (Feature Selection)",
            fontsize=10.5, color=c_text_sub, ha="center", va="center")

    # 2. KHỐI 1: DỮ LIỆU ĐẦU VÀO
    box_w = 960
    box_x = 170
    draw_box(box_x, 775, box_w, 75, bg=c_card_bg, border=c_card_border)
    ax.text(650, 825, "1. DỮ LIỆU MẠNG ĐẦU VÀO", fontsize=11.5, fontweight="bold", color=c_text_main, ha="center", va="center")
    ax.text(650, 798, "CICIDS2017    •    CICDDoS2019    •    UNSW-NB15    •    NSL-KDD",
            fontsize=11, color=c_text_sub, ha="center", va="center")

    draw_arrow(650, 775, 650, 725)

    # 3. KHỐI 2: TIỀN XỬ LÝ DỮ LIỆU
    draw_box(box_x, 650, box_w, 75, bg=c_card_bg, border=c_card_border)
    ax.text(650, 700, "2. TIỀN XỬ LÝ DỮ LIỆU", fontsize=11.5, fontweight="bold", color=c_text_main, ha="center", va="center")
    ax.text(650, 673, "Làm sạch (NaN / Inf / Var 0)   →   Mã hóa nhãn   →   Chuẩn hóa MinMax   →   Chia Train / Test (80 / 20)",
            fontsize=10.5, color=c_text_sub, ha="center", va="center")

    # Đường phân nhánh xuống 4 phương pháp
    # Tọa độ X trung tâm của 4 cột
    col_w = 215
    col_h = 100
    gap = (box_w - 4 * col_w) / 3 # ~33.3 pt
    c1_x = box_x
    c2_x = c1_x + col_w + gap
    c3_x = c2_x + col_w + gap
    c4_x = c3_x + col_w + gap

    col_centers = [c1_x + col_w/2, c2_x + col_w/2, c3_x + col_w/2, c4_x + col_w/2]

    # Phân nhánh từ 650 xuống ngang 605 rồi rẽ vào 4 cột
    draw_arrow(650, 650, 650, 605)
    ax.plot([col_centers[0], col_centers[3]], [605, 605], color=c_line, lw=1.2)
    for cx in col_centers:
        draw_arrow(cx, 605, cx, 565)

    # 4. KHỐI 3: 4 NHÁNH LỰA CHỌN ĐẶC TRƯNG (Y: 465 -> 565)
    y_card = 465

    # Cột 1: Baseline
    draw_box(c1_x, y_card, col_w, col_h, bg=c_card_bg, border=c_card_border)
    ax.text(col_centers[0], y_card + 75, "3.1 Baseline", fontsize=11.5, fontweight="bold", color=c_text_main, ha="center", va="center")
    ax.text(col_centers[0], y_card + 48, "Toàn bộ đặc trưng", fontsize=10, color=c_text_sub, ha="center", va="center")
    ax.text(col_centers[0], y_card + 25, "(Tập đối chứng)", fontsize=9, color="#64748b", ha="center", va="center")

    # Cột 2: Filter
    draw_box(c2_x, y_card, col_w, col_h, bg=c_card_bg, border=c_card_border)
    ax.text(col_centers[1], y_card + 75, "3.2 Filter", fontsize=11.5, fontweight="bold", color=c_text_main, ha="center", va="center")
    ax.text(col_centers[1], y_card + 48, "Mutual Information", fontsize=10, color=c_text_sub, ha="center", va="center")
    ax.text(col_centers[1], y_card + 25, "Chi-square / ANOVA", fontsize=9, color="#64748b", ha="center", va="center")

    # Cột 3: Wrapper
    draw_box(c3_x, y_card, col_w, col_h, bg=c_card_bg, border=c_card_border)
    ax.text(col_centers[2], y_card + 75, "3.3 Wrapper", fontsize=11.5, fontweight="bold", color=c_text_main, ha="center", va="center")
    ax.text(col_centers[2], y_card + 48, "Genetic Algorithm (GA)", fontsize=10, color=c_text_sub, ha="center", va="center")
    ax.text(col_centers[2], y_card + 25, "Binary PSO v2", fontsize=9, color="#64748b", ha="center", va="center")

    # Cột 4: Hybrid (Nổi bật nhẹ làm đề xuất)
    draw_box(c4_x, y_card, col_w, col_h, bg=c_highlight_bg, border=c_highlight_border, lw=1.5)
    ax.text(col_centers[3], y_card + 75, "3.4 Hybrid (Đề xuất)", fontsize=11.5, fontweight="bold", color=c_text_blue, ha="center", va="center")
    ax.text(col_centers[3], y_card + 48, "Filter (Lọc sơ bộ)", fontsize=10, color=c_text_sub, ha="center", va="center")
    ax.text(col_centers[3], y_card + 25, "→ Wrapper (GA / PSO)", fontsize=9.5, fontweight="bold", color=c_text_blue, ha="center", va="center")

    # Hội tụ 4 nhánh xuống mô hình học máy
    for cx in col_centers:
        ax.plot([cx, cx], [y_card, 425], color=c_line, lw=1.2)
    ax.plot([col_centers[0], col_centers[3]], [425, 425], color=c_line, lw=1.2)
    draw_arrow(650, 425, 650, 385)

    # 5. KHỐI 4: HUẤN LUYỆN VÀ PHÂN LOẠI
    draw_box(box_x, 310, box_w, 75, bg=c_card_bg, border=c_card_border)
    ax.text(650, 360, "4. HUẤN LUYỆN & PHÂN LOẠI (CLASSIFIERS)", fontsize=11.5, fontweight="bold", color=c_text_main, ha="center", va="center")
    ax.text(650, 333, "Decision Tree (DT)    •    Random Forest (RF)    •    XGBoost (XGB)",
            fontsize=11, color=c_text_sub, ha="center", va="center")

    draw_arrow(650, 310, 650, 260)

    # 6. KHỐI 5: ĐÁNH GIÁ & SO SÁNH HIỆU NĂNG
    draw_box(box_x, 185, box_w, 75, bg=c_card_bg, border=c_card_border)
    ax.text(650, 235, "5. ĐÁNH GIÁ & SO SÁNH HIỆU NĂNG", fontsize=11.5, fontweight="bold", color=c_text_main, ha="center", va="center")
    ax.text(650, 208, "Độ chính xác (Accuracy, F1-macro)   •   Tỷ lệ giảm chiều đặc trưng   •   Thời gian huấn luyện & suy luận",
            fontsize=10.5, color=c_text_sub, ha="center", va="center")

    # Chú thích chân trang (Footer note)
    ax.text(650, 115, "Nguyên tắc thiết kế: Tuyệt đối không rò rỉ dữ liệu (No Data Leakage)  •  Bảo toàn phân tầng nhãn (Stratified Sampling)",
            fontsize=9.5, fontstyle="italic", color="#64748b", ha="center", va="center")

    # Lưu hình ảnh chất lượng cao
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=c_bg, edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Đã xuất sơ đồ thành công tại: {output_path}")

if __name__ == "__main__":
    draw_clean_diagram()
