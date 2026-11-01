import os
import textwrap
import matplotlib
matplotlib.use('Agg')
os.environ['MPLCONFIGDIR'] = '/tmp'
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_human_academic_diagram(output_path):
    fig_w, fig_h = 16, 22.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.axis('off')

    # Color palette: Strict Academic Slate & Neutral with single Emerald Accent
    c_title = '#0f172a'       # Slate 900
    c_step_hdr = '#1e293b'    # Slate 800
    c_text = '#334155'        # Slate 700
    c_subtext = '#64748b'     # Slate 500
    c_border = '#64748b'      # Slate 500 (thin crisp border)
    c_border_light = '#cbd5e1'# Slate 300
    c_bg_box = '#ffffff'
    c_bg_subtle = '#f8fafc'   # Slate 50
    c_bg_hdr = '#f1f5f9'      # Slate 100 header bar
    c_arrow = '#475569'       # Slate 600

    # Proposed method accent (Restrained Emerald)
    c_prop_border = '#16a34a' # Emerald 600
    c_prop_bg = '#f0fdf4'     # Emerald 50
    c_prop_tag = '#15803d'

    def draw_card(x, y, w, h, bg=c_bg_box, border=c_border, lw=1.1, radius=0.4):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.02,rounding_size={radius}',
                                    fc=bg, ec=border, lw=lw, zorder=2)
        ax.add_patch(box)

    def draw_step_header(x, y, w, h_hdr, step_label, title, badge_text=""):
        hdr_box = patches.FancyBboxPatch((x, y - h_hdr), w, h_hdr,
                                        boxstyle='round,pad=0.02,rounding_size=0.4',
                                        fc=c_bg_hdr, ec=c_border, lw=1.1, zorder=3)
        ax.add_patch(hdr_box)
        rect_cover = patches.Rectangle((x, y - h_hdr), w, h_hdr/2, fc=c_bg_hdr, ec=c_bg_hdr, zorder=4)
        ax.add_patch(rect_cover)
        ax.plot([x, x + w], [y - h_hdr, y - h_hdr], color=c_border, lw=1.0, zorder=5)

        ax.text(x + 1.8, y - h_hdr/2, step_label, ha='left', va='center',
                fontsize=10.2, fontweight='bold', color='#1e3a8a', zorder=6)
        
        # Calculate offset based on step_label length
        x_title = x + (17.5 if "GIAI ĐOẠN" in step_label else 19.5)
        ax.text(x_title, y - h_hdr/2, f"|   {title}", ha='left', va='center',
                fontsize=11.2, fontweight='bold', color=c_step_hdr, zorder=6)

        if badge_text:
            ax.text(x + w - 1.8, y - h_hdr/2, badge_text, ha='right', va='center',
                    fontsize=9.5, fontweight='600', color=c_subtext, zorder=6)

    def draw_arrow(x, y_start, y_end):
        ax.annotate('', xy=(x, y_end), xytext=(x, y_start),
                    arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.5, mutation_scale=13),
                    zorder=7)

    # ==================== MAIN HEADER ====================
    ax.text(50, 102.4, "QUY TRÌNH TỔNG THỂ HỆ THỐNG PHÁT HIỆN XÂM NHẬP MẠNG (NIDS)",
            ha='center', va='center', fontsize=17, fontweight='bold', color=c_title)
    ax.text(50, 100.4, "Khung Thực Nghiệm Đánh Giá & Tối Ưu Lựa Chọn Đặc Trưng (Mã Đề Tài: 2.18)",
            ha='center', va='center', fontsize=11, color=c_subtext)

    w_main = 90
    x_main = 5

    # ==================== STEP 1: DATA INGESTION ====================
    y1 = 92.8
    h1 = 6.2
    draw_card(x_main, y1, w_main, h1)
    draw_step_header(x_main, y1 + h1, w_main, 2.4, "GIAI ĐOẠN 1", "DỮ LIỆU LƯU LƯỢNG MẠNG ĐẦU VÀO (NETWORK DATASETS)", "4 BỘ DỮ LIỆU CHUẨN")

    ds_items = [
        ("CICIDS2017", "52 đặc trưng gốc • 15 họ tấn công"),
        ("CICDDoS2019", "67 đặc trưng gốc • 8 kịch bản DDoS"),
        ("UNSW-NB15", "42 đặc trưng gốc • 10 họ tấn công"),
        ("NSL-KDD", "39 đặc trưng gốc • Kiểm chứng baseline")
    ]
    sub_w1 = w_main / 4
    for idx, (ds_name, ds_info) in enumerate(ds_items):
        cx = x_main + idx * sub_w1
        if idx > 0:
            ax.plot([cx, cx], [y1 + 0.6, y1 + h1 - 2.8], color=c_border_light, lw=0.8, zorder=5)
        ax.text(cx + sub_w1/2, y1 + 2.3, ds_name, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color=c_title, zorder=6)
        ax.text(cx + sub_w1/2, y1 + 1.1, ds_info, ha='center', va='center',
                fontsize=8.8, color=c_subtext, zorder=6)

    draw_arrow(50, y1, y1 - 2.2)

    # ==================== STEP 2: EDA & DATA LEAKAGE AUDIT ====================
    y2 = 81.8
    h2 = 8.8
    draw_card(x_main, y2, w_main, h2)
    draw_step_header(x_main, y2 + h2, w_main, 2.4, "GIAI ĐOẠN 2", "PHÂN TÍCH KHÁM PHÁ (EDA) & RÀ SOÁT RÒ RỈ DỮ LIỆU", "DATA LEAKAGE AUDIT")

    eda_cols = [
        ("Phân Tích Tương Quan & MI",
         "• Tính ma trận Pearson (|r| ≥ 0.9).\n• Định vị nhóm đặc trưng đa cộng tuyến.\n• Tính điểm Mutual Information sơ bộ."),
        ("Rà Soát & Khử Rò Rỉ (Artifact)",
         "• Phát hiện cột 'Inbound' ở CICDDoS2019 rò rỉ nhãn.\n• Bắt buộc LOẠI BỎ cột Inbound khỏi mô hình.\n• Rà soát các thuộc tính metadata (IP, Port, Time)."),
        ("Kiểm Định Mất Cân Bằng Nhãn",
         "• Đo lường tỷ lệ chênh lệch lớp cực đoan.\n• Phát hiện lỗi nhãn ở CICIDS2017 (Morning).\n• Quy định dùng F1-Macro thay cho Accuracy.")
    ]
    w_c2 = w_main / 3
    for idx, (col_t, col_body) in enumerate(eda_cols):
        cx = x_main + idx * w_c2
        if idx > 0:
            ax.plot([cx, cx], [y2 + 0.6, y2 + h2 - 2.8], color=c_border_light, lw=0.8, zorder=5)
        ax.text(cx + 1.5, y2 + h2 - 3.8, col_t, ha='left', va='top',
                fontsize=10, fontweight='bold', color=c_step_hdr, zorder=6)
        ax.text(cx + 1.5, y2 + h2 - 5.2, col_body, ha='left', va='top',
                fontsize=8.7, color=c_text, linespacing=1.35, zorder=6)

    draw_arrow(50, y2, y2 - 2.2)

    # ==================== STEP 3: PREPROCESSING PIPELINE ====================
    y3 = 70.8
    h3 = 8.8
    draw_card(x_main, y3, w_main, h3)
    draw_step_header(x_main, y3 + h3, w_main, 2.4, "GIAI ĐOẠN 3", "TIỀN XỬ LÝ DỮ LIỆU CHUẨN MỰC & KHÔNG RÒ RỈ", "PREPROCESSING PIPELINE")

    prep_cols = [
        ("Làm Sạch & Xử Lý Ngoại Lệ",
         "• Chuyển đổi Inf -> NaN, điền thiếu bằng Median.\n• Loại bỏ bản ghi trùng lặp (Drop Duplicates).\n• Loại cột có phương sai gần 0 (Var < 1e-5)."),
        ("Mã Hóa & Chuẩn Hóa Số Liệu",
         "• Loại bỏ ID, Flow ID, Timestamp, IP nguồn.\n• Mã hóa biến phân loại bằng Label Encoding.\n• Chuẩn hóa MinMaxScaler về đoạn [0, 1]."),
        ("Phân Tách Dữ Liệu Train / Test",
         "• Chia Train/Test 80/20 phân tầng (Stratified).\n• NGUYÊN TẮC: Bộ biến đổi CHỈ fit trên Train.\n• Tập Test hoàn toàn chỉ áp dụng transform.")
    ]
    w_c3 = w_main / 3
    for idx, (col_t, col_body) in enumerate(prep_cols):
        cx = x_main + idx * w_c3
        if idx > 0:
            ax.plot([cx, cx], [y3 + 0.6, y3 + h3 - 2.8], color=c_border_light, lw=0.8, zorder=5)
        ax.text(cx + 1.5, y3 + h3 - 3.8, col_t, ha='left', va='top',
                fontsize=10, fontweight='bold', color=c_step_hdr, zorder=6)
        ax.text(cx + 1.5, y3 + h3 - 5.2, col_body, ha='left', va='top',
                fontsize=8.7, color=c_text, linespacing=1.35, zorder=6)

    draw_arrow(50, y3, y3 - 2.2)

    # ==================== STEP 4: 7 FEATURE SELECTION METHODS ====================
    y4 = 44.2
    h4 = 24.2
    draw_card(x_main, y4, w_main, h4)
    draw_step_header(x_main, y4 + h4, w_main, 2.4, "GIAI ĐOẠN 4", "KHẢO SÁT 7 PHƯƠNG ÁN LỰA CHỌN ĐẶC TRƯNG", "THỰC NGHIỆM ĐỐI CHỨNG TOÀN DIỆN")

    col_gap = 1.2
    n_cols = 5
    w_fs = (w_main - 2.4 - (n_cols - 1) * col_gap) / n_cols
    y_sub = y4 + 1.2
    h_sub = h4 - 4.2

    fs_methods = [
        ("4.1 Baseline", "(Tập Đối Chứng)",
         "Sử dụng 100% đặc trưng\ngốc sau tiền xử lý\n(k = D_gốc).\n\nĐóng vai trò mốc chuẩn\n(Ground truth) để đo\nlường mức độ suy hao\nhiệu năng khi rút gọn.",
         "Mốc so sánh", False),

        ("4.2 Filter-MI", "(Mutual Information)",
         "Tính thông tin tương hỗ\nI(X; Y) đo phụ thuộc\nphi tuyến tính.\n\nXếp hạng và chọn k đặc\ntrưng cao nhất.\n\nTốc độ nhanh, độc lập\nvới mô hình phân loại.",
         "Lọc phi tuyến", False),

        ("4.3 Metaheuristics", "(GA & Binary PSO v2)",
         "Tìm kiếm toàn cục với\nmặt nạ bit nhị phân.\n\nCải tiến PSO v2: Tích hợp\ncơ chế Reset vận tốc và\nđột biến vị trí để chống kẹt\nhội tụ sớm (Local optima).",
         "Tìm kiếm tối ưu", False),

        ("4.4 Hybrid 1", "(Filter -> GA/PSO)",
         "HAI BƯỚC TUẦN TỰ:\n• Bước 1: Filter-MI lọc\n  sơ bộ 50% đặc trưng.\n• Bước 2: GA/PSO v2 tinh\n  chọn đa mục tiêu Pareto:\n  Fit = F1 + F1_min - a*(k/D).\n\nGiảm chiều sâu, F1 cao.",
         "ĐỀ XUẤT CHÍNH", True),

        ("4.5 Hybrid 2", "(MI + SHAP + XGB)",
         "PHƯƠNG PHÁP MỘT LƯỢT:\n• Tính MI trên mẫu.\n• Huấn luyện XGBoost,\n  tính SHAP qua TreeExplainer.\n• Chuẩn hóa điểm kết hợp:\n  Score = w·MI + w·SHAP.\n• Chọn Top 30% đặc trưng.",
         "Explainable AI", False)
    ]

    for i, (m_title, m_sub, m_desc, m_badge, is_prop) in enumerate(fs_methods):
        x_fs = x_main + 1.2 + i * (w_fs + col_gap)
        bg_col = c_prop_bg if is_prop else c_bg_subtle
        bc_col = c_prop_border if is_prop else c_border_light
        lw_col = 1.4 if is_prop else 0.9

        draw_card(x_fs, y_sub, w_fs, h_sub, bg=bg_col, border=bc_col, lw=lw_col, radius=0.3)

        tc = c_prop_tag if is_prop else c_step_hdr
        ax.text(x_fs + w_fs/2, y_sub + h_sub - 1.2, m_title, ha='center', va='center',
                fontsize=9.8, fontweight='bold', color=tc, zorder=6)
        ax.text(x_fs + w_fs/2, y_sub + h_sub - 2.4, m_sub, ha='center', va='center',
                fontsize=8.5, color=c_subtext, zorder=6)

        ax.plot([x_fs + 0.8, x_fs + w_fs - 0.8], [y_sub + h_sub - 3.2, y_sub + h_sub - 3.2],
                color=bc_col, lw=0.7, zorder=5)

        ax.text(x_fs + 0.8, y_sub + h_sub - 4.0, m_desc, ha='left', va='top',
                fontsize=8.2, color=c_text, linespacing=1.35, zorder=6)

        badge_bg = '#dcfce7' if is_prop else '#e2e8f0'
        badge_fg = c_prop_tag if is_prop else '#475569'
        badge_box = patches.FancyBboxPatch((x_fs + 0.8, y_sub + 0.6), w_fs - 1.6, 1.6,
                                           boxstyle='round,pad=0.02,rounding_size=0.2',
                                           fc=badge_bg, ec=bc_col, lw=0.6, zorder=5)
        ax.add_patch(badge_box)
        ax.text(x_fs + w_fs/2, y_sub + 1.4, m_badge, ha='center', va='center',
                fontsize=8.2, fontweight='bold', color=badge_fg, zorder=6)

    draw_arrow(50, y4, y4 - 2.2)

    # ==================== STEP 5: CLASSIFIERS EVALUATION ====================
    y5 = 33.2
    h5 = 8.6
    draw_card(x_main, y5, w_main, h5)
    draw_step_header(x_main, y5 + h5, w_main, 2.4, "GIAI ĐOẠN 5", "HUẤN LUYỆN & PHÂN LOẠI ĐA MÔ HÌNH", "CLASSIFIERS EVALUATION")

    clf_cols = [
        ("Decision Tree (DT)",
         "• Mô hình cây quyết định nhanh, tường minh.\n• Được nhúng trong hàm thích nghi GA/PSO để tìm\n  nghiệm tối ưu một cách hiệu quả."),
        ("Random Forest (RF)",
         "• Mô hình Ensemble tập hợp nhiều cây độc lập.\n• Đo lường tính ổn định và khả năng khái quát\n  hóa khi mở rộng sang kiến trúc ensemble."),
        ("Extreme Gradient Boosting (XGB)",
         "• Thuật toán Boosting tối ưu hàm mất mát gradient.\n• Kiểm chứng trần hiệu năng và xác thực tập đặc\n  trưng không bị thiên vị mô hình (No Classifier Bias).")
    ]
    w_c5 = w_main / 3
    for idx, (col_t, col_body) in enumerate(clf_cols):
        cx = x_main + idx * w_c5
        if idx > 0:
            ax.plot([cx, cx], [y5 + 0.6, y5 + h5 - 2.8], color=c_border_light, lw=0.8, zorder=5)
        ax.text(cx + 1.5, y5 + h5 - 3.8, col_t, ha='left', va='top',
                fontsize=10, fontweight='bold', color=c_step_hdr, zorder=6)
        ax.text(cx + 1.5, y5 + h5 - 5.2, col_body, ha='left', va='top',
                fontsize=8.6, color=c_text, linespacing=1.35, zorder=6)

    draw_arrow(50, y5, y5 - 2.2)

    # ==================== STEP 6: EVALUATION & ABLATION ====================
    y6 = 18.2
    h6 = 12.6
    draw_card(x_main, y6, w_main, h6)
    draw_step_header(x_main, y6 + h6, w_main, 2.4, "GIAI ĐOẠN 6", "ĐÁNH GIÁ KẾT QUẢ TOÀN DIỆN & PHÂN TÍCH CHUYÊN SÂU", "EVALUATION & SENSITIVITY")

    eval_cols = [
        ("1. Bộ Chỉ Số Đánh Giá",
         "• F1-Macro & Balanced Accuracy: Đánh giá\n  công bằng trên các lớp tấn công thiểu số.\n• Tỷ Lệ Giảm Chiều: (D_gốc - k) / D_gốc,\n  mục tiêu giảm > 80% số đặc trưng.\n• Thời Gian Train & Infer: Đo lường tốc độ\n  xử lý gói tin thời gian thực."),
        ("2. Phân Tích Bóc Tách (Ablation)",
         "• Đo lường mức độ phụ thuộc vào biến nhạy cảm:\n  - Bỏ 'Destination Port': F1 sụt giảm -15.18%\n    (từ 99.33% xuống còn 84.15%).\n  - Bỏ 'URG Flag': F1 sụt giảm -6.40% (92.93%).\n  - Bỏ cả hai: F1 sụt giảm -19.02% (80.31%).\n• Khẳng định vai trò của đặc trưng tầng giao vận."),
        ("3. Đánh Giá Khái Quát Hóa Chéo",
         "• Tập đặc trưng tối ưu tìm bởi DT duy trì và cải\n  thiện hiệu năng khi chuyển sang RF và XGB:\n  CICIDS2017: DT (99.59%) -> RF (99.85%).\n  UNSW-NB15: DT (86.32%) -> RF (88.45%).\n• Chứng minh tính ổn định, không bị phụ thuộc\n  vào một thuật toán học máy đơn lẻ.")
    ]
    w_c6 = w_main / 3
    for idx, (col_t, col_body) in enumerate(eval_cols):
        cx = x_main + idx * w_c6
        if idx > 0:
            ax.plot([cx, cx], [y6 + 0.6, y6 + h6 - 2.8], color=c_border_light, lw=0.8, zorder=5)
        ax.text(cx + 1.5, y6 + h6 - 3.8, col_t, ha='left', va='top',
                fontsize=10, fontweight='bold', color=c_step_hdr, zorder=6)
        ax.text(cx + 1.5, y6 + h6 - 5.2, col_body, ha='left', va='top',
                fontsize=8.5, color=c_text, linespacing=1.35, zorder=6)

    # ==================== FOOTER: INVIOLABLE PRINCIPLES ====================
    y_foot = 6.8
    h_foot = 9.0
    draw_card(x_main, y_foot, w_main, h_foot, bg='#f8fafc', border=c_border, lw=1.0)
    draw_step_header(x_main, y_foot + h_foot, w_main, 2.2, "QUY TẮC CỐT LÕI", "NGUYÊN TẮC THIẾT KẾ BẤT KHẢ XÂM PHẠM", "ACADEMIC INTEGRITY")

    foot_cols = [
        ("Nguyên Tắc 1: Không Rò Rỉ Dữ Liệu (No Data Leakage)",
         "Tất cả các phép biến đổi (MinMaxScaler, LabelEncoder, bộ chọn đặc trưng Filter) bắt buộc chỉ được fit trên tập Train. Tập Test hoàn toàn độc lập, chỉ áp dụng transform từ tham số đã học, đảm bảo tính khách quan của thực nghiệm."),
        ("Nguyên Tắc 2: Bảo Toàn Phân Tầng Nhãn (Stratified Sampling)",
         "Mọi thao tác chia tách Train/Test và lấy mẫu tính hàm thích nghi cho GA/PSO đều dùng stratify=y nhằm duy trì tỷ lệ phân bố của các lớp tấn công thiểu số cực hiếm, ngăn chặn triệt để hiện tượng bỏ sót các mối đe dọa an ninh mạng.")
    ]
    w_foot = w_main / 2
    for idx, (f_title, f_body) in enumerate(foot_cols):
        fx = x_main + idx * w_foot
        if idx > 0:
            ax.plot([fx, fx], [y_foot + 0.6, y_foot + h_foot - 2.6], color=c_border_light, lw=0.8, zorder=5)
        ax.text(fx + 1.5, y_foot + h_foot - 3.4, f_title, ha='left', va='top',
                fontsize=9.5, fontweight='bold', color=c_step_hdr, zorder=6)
        wrapped_body = textwrap.fill(f_body, width=54)
        ax.text(fx + 1.5, y_foot + h_foot - 4.6, wrapped_body, ha='left', va='top',
                fontsize=8.5, color=c_text, linespacing=1.35, zorder=6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"Human academic diagram saved successfully to: {output_path}")

if __name__ == '__main__':
    create_human_academic_diagram('docs/assets/quy-trinh-nids.png')
