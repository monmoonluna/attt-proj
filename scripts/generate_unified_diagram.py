import os
import matplotlib
matplotlib.use('Agg')
os.environ['MPLCONFIGDIR'] = '/tmp'
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_unified_diagram(output_path):
    # Dimensions
    fig_w, fig_h = 16, 21
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.axis('off')

    # Color palette
    navy_dark = '#0f172a'
    primary_blue = '#1e3a8a'
    sky_blue = '#0284c7'
    amber = '#b45309'
    green = '#15803d'
    purple = '#7e22ce'
    border_c = '#cbd5e1'
    text_dark = '#1e293b'
    text_muted = '#475569'

    # Helper: draw box
    def draw_box(x, y, w, h, bg, border, radius=1.2, lw=1.8):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.2,rounding_size={radius}',
                                    fc=bg, ec=border, lw=lw, zorder=2)
        ax.add_patch(box)

    # Helper: draw arrow
    def draw_down_arrow(x, y_start, y_end, color='#64748b', lw=2.2):
        ax.annotate('', xy=(x, y_end), xytext=(x, y_start),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=16),
                    zorder=3)

    # ==================== TITLE ====================
    ax.text(50, 101.5, "QUY TRÌNH NGHIÊN CỨU & THỰC NGHIỆM NIDS TOÀN DIỆN",
            ha='center', va='center', fontsize=22, fontweight='bold', color=navy_dark)
    ax.text(50, 99.2, "Kiến trúc 6 bước chuẩn hóa theo Đề tài: Tối ưu Lựa chọn Đặc trưng Phát hiện Xâm nhập Mạng",
            ha='center', va='center', fontsize=12.5, color=text_muted, style='italic')

    # ==================== STEP 1: INPUT DATASETS ====================
    draw_box(6, 90.5, 88, 6.8, '#f8fafc', '#94a3b8')
    ax.text(50, 95.2, "1. DỮ LIỆU MẠNG ĐẦU VÀO (NETWORK BENCHMARK DATASETS)",
            ha='center', va='center', fontsize=13.5, fontweight='bold', color=primary_blue)
    ax.text(50, 92.5, "CICIDS2017 (52 đặc trưng)   •   CICDDoS2019 (67 đặc trưng)   •   UNSW-NB15 (42 đặc trưng)   •   NSL-KDD (39 đặc trưng)",
            ha='center', va='center', fontsize=11.5, fontweight='600', color='#0369a1')

    draw_down_arrow(50, 90.3, 87.0)

    # ==================== STEP 2: EDA & ARTIFACT AUDIT ====================
    draw_box(6, 78.5, 88, 8.2, '#fffbeb', '#f59e0b')
    ax.text(50, 84.7, "2. PHÂN TÍCH KHÁM PHÁ (EDA) & RÀ SOÁT DỮ LIỆU RÒ RỈ (DATA ARTIFACTS)",
            ha='center', va='center', fontsize=13.5, fontweight='bold', color=amber)
    ax.text(50, 82.2, "• Phân tích tương quan Pearson (|r| ≥ 0.9) & tính toán điểm Thông tin tương hỗ (Mutual Information)",
            ha='center', va='center', fontsize=11, color=text_dark)
    ax.text(50, 80.2, "• Rà soát & LOẠI BỎ triệt để thuộc tính rò rỉ dữ liệu (Data Leakage Artifact): Cột 'Inbound' ở CICDDoS2019",
            ha='center', va='center', fontsize=11.2, fontweight='bold', color='#b91c1c')

    draw_down_arrow(50, 78.3, 75.0)

    # ==================== STEP 3: PREPROCESSING ====================
    draw_box(6, 66.5, 88, 8.2, '#eff6ff', '#3b82f6')
    ax.text(50, 72.7, "3. TIỀN XỬ LÝ DỮ LIỆU CHUẨN MỰC (PREPROCESSING PIPELINE)",
            ha='center', va='center', fontsize=13.5, fontweight='bold', color=primary_blue)
    ax.text(50, 70.2, "• Làm sạch: Xử lý giá trị NaN / Inf; loại bỏ bản ghi trùng lặp và cột có phương sai gần 0 (Var < 1e-5)",
            ha='center', va='center', fontsize=11, color=text_dark)
    ax.text(50, 68.2, "• Loại bỏ cột định danh (Flow ID, IP, Port nguồn, Timestamp)  • Mã hóa Label Encoding  • Chuẩn hóa MinMaxScaler về [0, 1]",
            ha='center', va='center', fontsize=11, color=text_dark)

    draw_down_arrow(50, 66.3, 62.0)

    # ==================== STEP 4: 7 FEATURE SELECTION METHODS ====================
    # Container box for FS
    draw_box(4, 38.0, 92, 23.5, '#f8fafc', '#94a3b8', radius=1.5, lw=1.5)
    ax.text(50, 59.5, "4. KHẢO SÁT 7 PHƯƠNG ÁN LỰA CHỌN ĐẶC TRƯNG (FEATURE SELECTION)",
            ha='center', va='center', fontsize=14, fontweight='bold', color=navy_dark)
    ax.text(50, 57.5, "Thực nghiệm đối chứng toàn diện: Từ Baseline đến Metaheuristics và Chiến lược Lai ghép Hybrid",
            ha='center', va='center', fontsize=11, color=text_muted, style='italic')

    # 5 Sub-boxes
    col_w = 16.8
    y_col = 39.5
    h_col = 16.5

    # 4.1 Baseline
    draw_box(6, y_col, col_w, h_col, '#ffffff', '#cbd5e1')
    ax.text(6 + col_w/2, 54.0, "4.1 Baseline", ha='center', va='center', fontsize=12, fontweight='bold', color=text_dark)
    ax.text(6 + col_w/2, 51.5, "(Tập Đối Chứng)", ha='center', va='center', fontsize=10.5, color=text_muted)
    ax.text(6 + col_w/2, 47.0, "Sử dụng 100%\nđặc trưng gốc\nsau tiền xử lý\n(k = D_gốc)", ha='center', va='center', fontsize=10.5, color=text_dark)
    ax.text(6 + col_w/2, 42.0, "Mốc chuẩn đánh\ngiá mức suy hao", ha='center', va='center', fontsize=10, color=text_muted, style='italic')

    # 4.2 Filter-MI
    draw_box(24, y_col, col_w, h_col, '#ffffff', '#cbd5e1')
    ax.text(24 + col_w/2, 54.0, "4.2 Filter-MI", ha='center', va='center', fontsize=12, fontweight='bold', color='#0284c7')
    ax.text(24 + col_w/2, 51.5, "(Mutual Information)", ha='center', va='center', fontsize=10.5, color=text_muted)
    ax.text(24 + col_w/2, 47.0, "Tính I(X; Y)\nĐo lường phụ thuộc\nphi tuyến tính.\nXếp hạng & giữ top k", ha='center', va='center', fontsize=10.5, color=text_dark)
    ax.text(24 + col_w/2, 42.0, "Tốc độ nhanh\nBỏ qua tương tác", ha='center', va='center', fontsize=10, color=text_muted, style='italic')

    # 4.3 Metaheuristics (GA / PSO)
    draw_box(42, y_col, col_w, h_col, '#ffffff', '#cbd5e1')
    ax.text(42 + col_w/2, 54.0, "4.3 Metaheuristic", ha='center', va='center', fontsize=12, fontweight='bold', color='#7e22ce')
    ax.text(42 + col_w/2, 51.5, "(GA & Binary PSO v2)", ha='center', va='center', fontsize=10.5, color=text_muted)
    ax.text(42 + col_w/2, 47.0, "Tìm kiếm toàn cục\nMặt nạ bit 0/1\nCải tiến PSO v2:\nReset chống kẹt", ha='center', va='center', fontsize=10.5, color=text_dark)
    ax.text(42 + col_w/2, 42.0, "Bắt trọn tương tác\nTốn thời gian lặp", ha='center', va='center', fontsize=10, color=text_muted, style='italic')

    # 4.4 Hybrid 1 (PROPOSED)
    draw_box(60, y_col, col_w, h_col, '#ecfdf5', '#059669', lw=2.4)
    ax.text(60 + col_w/2, 54.0, "4.4 Hybrid 1", ha='center', va='center', fontsize=12.5, fontweight='bold', color='#065f46')
    ax.text(60 + col_w/2, 51.5, "(Filter -> GA / PSO)", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#047857')
    ax.text(60 + col_w/2, 46.8, "Bước 1: Filter-MI\ngiữ top 50%\nBước 2: GA/PSO tinh\nchọn + F1 lớp hiếm", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#065f46')
    ax.text(60 + col_w/2, 41.5, "[PHƯƠNG ÁN ĐỀ XUẤT]\nCân bằng Pareto", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#059669')

    # 4.5 Hybrid 2
    draw_box(78, y_col, col_w, h_col, '#ffffff', '#cbd5e1')
    ax.text(78 + col_w/2, 54.0, "4.5 Hybrid 2", ha='center', va='center', fontsize=12, fontweight='bold', color='#d97706')
    ax.text(78 + col_w/2, 51.5, "(MI + SHAP + XGB)", ha='center', va='center', fontsize=10.5, color=text_muted)
    ax.text(78 + col_w/2, 47.0, "Không qua vòng lặp:\nĐiểm = w·MI + w·SHAP\nvới TreeExplainer\nChọn top 30% ĐT", ha='center', va='center', fontsize=10.5, color=text_dark)
    ax.text(78 + col_w/2, 42.0, "Giải thích được\n(Explainable AI)", ha='center', va='center', fontsize=10, color=text_muted, style='italic')

    draw_down_arrow(50, 37.8, 34.0)

    # ==================== STEP 5: CLASSIFIERS ====================
    draw_box(6, 25.0, 88, 8.2, '#f0fdf4', '#22c55e')
    ax.text(50, 31.2, "5. HUẤN LUYỆN & PHÂN LOẠI ĐA MÔ HÌNH (CLASSIFIERS EVALUATION)",
            ha='center', va='center', fontsize=13.5, fontweight='bold', color=green)
    ax.text(50, 28.7, "Huấn luyện trên D_train và Đánh giá trên D_test thật chỉ với các đặc trưng được chọn lọc",
            ha='center', va='center', fontsize=11, color=text_dark)
    ax.text(50, 26.7, "Decision Tree (DT)    •    Random Forest (RF)    •    Extreme Gradient Boosting (XGBoost)",
            ha='center', va='center', fontsize=11.5, fontweight='bold', color='#166534')

    draw_down_arrow(50, 24.8, 21.0)

    # ==================== STEP 6: EVALUATION & ABLATION ====================
    draw_box(6, 9.5, 88, 10.8, '#faf5ff', '#a855f7')
    ax.text(50, 18.3, "6. ĐÁNH GIÁ TOÀN DIỆN & PHÂN TÍCH BÓC TÁCH (EVALUATION & ABLATION STUDY)",
            ha='center', va='center', fontsize=13.5, fontweight='bold', color=purple)
    ax.text(50, 15.8, "• Hiệu năng cốt lõi: Accuracy, F1-macro (Nhị phân & Đa lớp), Tỷ lệ giảm chiều (%), Thời gian Train & Suy luận (Inference)",
            ha='center', va='center', fontsize=11, color=text_dark)
    ax.text(50, 13.8, "• Ablation Study: Bóc tách độ nhạy của thuộc tính 'Destination Port' (-15.2% F1) & 'URG Flag' (-6.4% F1) trên CICDDoS2019",
            ha='center', va='center', fontsize=11, fontweight='600', color='#701a75')
    ax.text(50, 11.5, "• Khái quát hóa chéo (Cross-Classifier): Kiểm chứng tính ổn định của tập đặc trưng khi chuyển từ DT sang RF / XGBoost",
            ha='center', va='center', fontsize=11, color=text_dark)

    # ==================== FOOTER INVIOLABLE RULES ====================
    draw_box(10, 2.2, 80, 4.8, '#f1f5f9', '#94a3b8', radius=0.8, lw=1.2)
    ax.text(50, 5.0, "NGUYÊN TẮC THIẾT KẾ BẤT KHẢ XÂM PHẠM (ACADEMIC RIGOR)",
            ha='center', va='center', fontsize=10.5, fontweight='bold', color=navy_dark)
    ax.text(50, 3.4, "1. Tuyệt đối không rò rỉ dữ liệu (No Data Leakage)   •   2. Bảo toàn phân tầng nhãn (Stratified Sampling trên mọi phép chia)",
            ha='center', va='center', fontsize=10, fontweight='600', color='#0369a1')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Unified diagram successfully saved to {output_path}")

if __name__ == '__main__':
    create_unified_diagram('docs/assets/quy-trinh-nids.png')
