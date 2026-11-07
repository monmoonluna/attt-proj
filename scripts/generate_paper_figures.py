#!/usr/bin/env python3
"""
scripts/generate_paper_figures.py
=================================
Script tạo và cập nhật các biểu đồ thực nghiệm cho Báo cáo Đồ án NIDS:
- Hình 3.2: So sánh F1-macro các phương án trên CICIDS2017 (Decision Tree)
  Khắc phục:
  1. Tiêu đề không bị cắt chữ 'baseline)' ở mép phải.
  2. Số 0.9959 (GA) và 0.9956 (Hybrid 2) không bị đường nét đứt ngang cắt qua / ghi đè lên chữ.
     Sử dụng solid white bounding box (alpha=1.0) và white stroke halo che mờ đường nét đứt
     phía sau các nhãn số, đảm bảo chữ số luôn nổi rõ ràng, sắc nét và dễ đọc.
- Hình 3.7: So sánh F1-macro của 3 bộ phân loại (Decision Tree, Random Forest, XGBoost) trên CICDDoS2019
  Khắc phục: Đẩy vị trí chú giải 'Random Forest' và các nhãn lên cao, loại bỏ hoàn toàn hiện tượng
  chồng lấn giữa chữ 'Random Forest' với nhãn giá trị 0.997 trên cột GA.
- Tự động cập nhật trực tiếp vào:
  1. Thư mục ảnh: docs/assets/paper_figures/
  2. Báo cáo Word: BaoCao_FeatureSelection_NIDS.docx
  3. Web Dashboard: reports/demo_dashboard.html
"""

import os
import sys
import shutil
import numpy as np

# Đảm bảo Matplotlib chạy ổn định trong mọi môi trường
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import docx

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_FIG_DIR = os.path.join(ROOT_DIR, "docs", "assets", "paper_figures")
DOCX_PATH = os.path.join(ROOT_DIR, "BaoCao_FeatureSelection_NIDS.docx")

def generate_figure_3_2(output_path):
    """
    Vẽ Hình 3.2: F1-macro của các phương án trên CICIDS2017
    Xử lý triệt để lỗi cắt chữ tiêu đề ở mép phải và lỗi đường nét đứt cắt qua các chữ số (0.9959, 0.9956).
    """
    fig, ax = plt.subplots(figsize=(6.8, 3.5), dpi=200)

    methods = ["Full", "Filter-MI", "GA", "PSO", "Hybrid 1-GA", "Hybrid 1-PSO", "Hybrid 2"]
    f1_scores = [0.9975, 0.9978, 0.9959, 0.9923, 0.9948, 0.9945, 0.9956]
    features = [52, 16, 5, 14, 6, 10, 16]

    # Bảng màu chuẩn học thuật: Xám cho Full, Đỏ gạch cho tốt nhất (Filter-MI), Xanh navy cho các phương án khác
    colors = ["#7F7F7F", "#C44E52", "#4C72B0", "#4C72B0", "#4C72B0", "#4C72B0", "#4C72B0"]

    x = np.arange(len(methods))
    width = 0.56

    bars = ax.bar(x, f1_scores, width, color=colors, edgecolor="none", zorder=2)

    # Đường chuẩn nét đứt baseline (zorder=1 để nằm hoàn toàn phía dưới các nhãn số)
    ax.axhline(0.9975, color="#7F7F7F", linestyle="--", linewidth=1.0, zorder=1)

    # Nhãn giá trị và số đặc trưng trên đầu mỗi cột
    # Sử dụng solid white bbox (alpha=1.0, pad=0.25) và white stroke halo để đường nét đứt
    # không cắt xuyên qua các chữ số (đặc biệt là 0.9959 và 0.9956), giúp số hoàn toàn rõ ràng, dễ đọc
    for b, f1, k in zip(bars, f1_scores, features):
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.00025,
            f"{f1:.4f}\n({k} ĐT)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#1e293b",
            linespacing=1.12,
            zorder=5,
            bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none", alpha=1.0),
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")]
        )

    ax.set_ylim(0.9850, 1.0016)
    ax.set_yticks([0.9850, 0.9875, 0.9900, 0.9925, 0.9950, 0.9975, 1.0000])
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))

    ax.set_ylabel("F1-macro", fontsize=10.5, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9.5)

    # Xoay nhẹ các nhãn dài để bố cục thoáng đãng
    for tick in ax.get_xticklabels():
        txt = tick.get_text()
        if "Hybrid" in txt or "Filter" in txt:
            tick.set_rotation(16)
            tick.set_ha("right")

    # Tiêu đề đầy đủ, không bị cắt chữ 'baseline)'
    title_text = "CICIDS2017 - nhị phân, Decision Tree (đỏ: phương pháp FS tốt nhất; đường nét đứt: baseline)"
    ax.set_title(title_text, fontsize=9.2, pad=12, wrap=True)

    # Viền đồ thị phẳng (Flat academic layout)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Đã tạo thành công Hình 3.2: {output_path}")

def generate_figure_3_7(output_path):
    """
    Vẽ Hình 3.7: F1-macro của ba bộ phân loại trên CICDDoS2019
    Khắc phục: Đẩy chú giải 'Random Forest' lên cao, mở rộng khoảng trần (headroom)
    để không bị chữ 0.997 đè lên.
    """
    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=200)

    methods = ["Full", "Filter-MI", "GA", "Hybrid 1-PSO", "Hybrid 2"]

    dt_f1 = [0.9908, 0.8378, 0.8890, 0.8597, 0.8847]
    rf_f1 = [0.8616, 0.8315, 0.9970, 0.8418, 0.8419]
    xgb_f1 = [0.9623, 0.8787, 0.9608, 0.8290, 0.9529]

    x = np.arange(len(methods))
    width = 0.23

    c_dt = "#4C72B0"   # Xanh dương
    c_rf = "#55A868"   # Xanh lá cây (Random Forest)
    c_xgb = "#C44E52"  # Đỏ gạch

    rects1 = ax.bar(x - width, dt_f1, width, label="Decision Tree", color=c_dt, zorder=3)
    rects2 = ax.bar(x, rf_f1, width, label="Random Forest", color=c_rf, zorder=3)
    rects3 = ax.bar(x + width, xgb_f1, width, label="XGBoost", color=c_xgb, zorder=3)

    # Ghi nhãn giá trị thẳng đứng trên đầu mỗi cột
    for rects in [rects1, rects2, rects3]:
        for r in rects:
            h = r.get_height()
            ax.text(
                r.get_x() + r.get_width() / 2,
                h + 0.005,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8.5,
                color="#1e293b"
            )

    # Đẩy giới hạn trục Y lên 1.095 tạo khoảng không gian thoáng cho nhãn GA (0.997)
    ax.set_ylim(0.75, 1.095)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    ax.set_ylabel("F1-macro", fontsize=10.5, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10.5)

    # Đẩy chữ chú giải (Legend) 'Random Forest' lên cao (bbox_to_anchor 1.15)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        frameon=False,
        fontsize=10.2
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#64748b")
    ax.spines["bottom"].set_color("#64748b")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Đã tạo thành công Hình 3.7: {output_path}")

def update_docx_media():
    """
    Cập nhật ảnh Hình 3.2 và Hình 3.7 trực tiếp vào BaoCao_FeatureSelection_NIDS.docx
    thông qua Document Part Binary Blob (không làm thay đổi cấu trúc bảng biểu, heading, equation).
    """
    if not os.path.exists(DOCX_PATH):
        print(f"Không tìm thấy tệp {DOCX_PATH}")
        return

    doc = docx.Document(DOCX_PATH)

    # rId15: Hình 3.2 (word/media/image5.png)
    # rId20: Hình 3.7 (word/media/image10.png)
    img32_path = os.path.join(DOCS_FIG_DIR, "hinh-3-2-f1-cicids2017-binary.png")
    img37_path = os.path.join(DOCS_FIG_DIR, "hinh-3-7-f1-ba-bo-phan-loai-ddos.png")

    if "rId15" in doc.part.related_parts and os.path.exists(img32_path):
        part32 = doc.part.related_parts["rId15"]
        with open(img32_path, "rb") as f:
            part32._blob = f.read()
        print("✓ Đã cập nhật Hình 3.2 vào tệp Word (rId15 - image5.png)")

    if "rId20" in doc.part.related_parts and os.path.exists(img37_path):
        part37 = doc.part.related_parts["rId20"]
        with open(img37_path, "rb") as f:
            part37._blob = f.read()
        print("✓ Đã cập nhật Hình 3.7 vào tệp Word (rId20 - image10.png)")

    doc.save(DOCX_PATH)
    print(f"✓ Đã lưu tài liệu Báo cáo Word thành công: {DOCX_PATH}")

def main():
    os.makedirs(DOCS_FIG_DIR, exist_ok=True)

    fig32_path = os.path.join(DOCS_FIG_DIR, "hinh-3-2-f1-cicids2017-binary.png")
    fig37_path = os.path.join(DOCS_FIG_DIR, "hinh-3-7-f1-ba-bo-phan-loai-ddos.png")

    # 1. Vẽ lại biểu đồ
    generate_figure_3_2(fig32_path)
    generate_figure_3_7(fig37_path)

    # 2. Cập nhật vào báo cáo Word
    update_docx_media()

    # 3. Cập nhật Web Dashboard
    update_dash_script = os.path.join(ROOT_DIR, "scripts", "update_dashboard.py")
    if os.path.exists(update_dash_script):
        import subprocess
        python_bin = sys.executable
        res = subprocess.run([python_bin, update_dash_script], capture_output=True, text=True)
        if res.returncode == 0:
            print("✓ Đã đồng bộ lại Web Dashboard (reports/demo_dashboard.html)")
        else:
            print(f"Lỗi cập nhật dashboard: {res.stderr}")

if __name__ == "__main__":
    main()
