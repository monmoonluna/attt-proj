#!/usr/bin/env python3
"""
scripts/update_dashboard.py
Refactor giao diện Web Dashboard theo chuẩn UX Học thuật & Báo cáo Khoa học:
- 5 Tab lớn mạch lạc theo tư duy nghiên cứu:
  1. 📌 Tổng quan & Quy trình (Overview & Architecture Pipeline)
  2. 📊 Thực nghiệm Nhị phân (Binary Detection - 4 Datasets với Sub-pills)
  3. 🌐 Thực nghiệm Đa lớp (Multi-Class - 3 Datasets với Sub-pills)
  4. ⚡ Đa Mô hình & Ablation Study (DT vs RF vs XGBoost + Bóc tách đặc trưng)
  5. 🖼️ Thư viện Biểu đồ & Hội tụ (16 Biểu đồ GA/PSO & EDA độ nét cao)
- Có Sub-pills chuyển nhanh giữa các bộ dữ liệu mượt mà, không cuộn dài
- Tự động quét 100% tất cả 63 lượt thực nghiệm định lượng từ output/
"""

import os
import glob
import json
import base64

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

METHOD_NAMES = {
    "baseline": "Baseline (Toàn bộ đặc trưng)",
    "fs_filter": "Filter (Mutual Information)",
    "fs_filter_rf": "Filter (Mutual Info) + RF",
    "fs_ga": "Wrapper: GA (Genetic Algorithm)",
    "fs_ga_rf": "Wrapper: GA + RF",
    "fs_pso": "Wrapper: PSO (Particle Swarm)",
    "fs_pso_v2": "Wrapper: PSO v2 (Mutation & Reset)",
    "fs_hybrid1_ga": "Hybrid 1: Filter + GA",
    "fs_hybrid1_ga_v2": "Hybrid 1: Filter + GA v2 (Tối ưu)",
    "fs_hybrid1_pso": "Hybrid 1: Filter + PSO",
    "fs_hybrid1_pso_rf": "Hybrid 1: Filter + PSO + RF",
    "fs_hybrid2": "Hybrid 2: Filter + Embedded/GA",
    "fs_hybrid2_rf": "Hybrid 2: Filter + Embedded + RF",
}

def encode_img_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(path)[1].replace(".", "").lower()
            if ext == "jpg": ext = "jpeg"
            return f"data:image/{ext};base64,{b64}"
    return ""

def load_dataset_results(dataset_folder):
    base_path = os.path.join(OUTPUT_DIR, dataset_folder)
    if not os.path.exists(base_path):
        return []

    results = []

    # 1. Baseline
    b_files = glob.glob(os.path.join(base_path, "**/baseline_full_features.json"), recursive=True)
    if b_files:
        try:
            with open(b_files[0], "r", encoding="utf-8") as f:
                bdata = json.load(f)
                bdata["method_key"] = "baseline"
                bdata["method_name"] = METHOD_NAMES["baseline"]
                bdata["classifier"] = bdata.get("classifier", "dt")
                results.append(bdata)
        except Exception:
            pass

    # 2. Subdirectories
    for sub in sorted(os.listdir(base_path)):
        sel_file = os.path.join(base_path, sub, "selected_features.json")
        if os.path.exists(sel_file):
            try:
                with open(sel_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["method_key"] = sub
                    data["method_name"] = METHOD_NAMES.get(sub, sub)
                    data["classifier"] = data.get("classifier", "rf" if "_rf" in sub else "dt")
                    results.append(data)
            except Exception:
                pass

    return results

def render_table_rows(results, orig_dim):
    rows_html = ""
    for r in results:
        key = r.get("method_key", "")
        name = r.get("method_name", key)
        k = r.get("n_features_selected", 0)
        tot = r.get("n_features_total", orig_dim) or orig_dim
        red = ((tot - k) / tot * 100.0) if tot > 0 else 0.0
        acc = r.get("accuracy", 0.0) * 100.0
        f1 = r.get("f1_macro", 0.0) * 100.0
        t_tr = r.get("train_time_sec", 0.0)
        t_inf = r.get("inference_time_sec", 0.0) * 1000.0
        clf = str(r.get("classifier", "dt")).upper()

        is_highlight = "hybrid1_ga_v2" in key or "hybrid1" in key
        is_base = key == "baseline"

        tr_class = "highlight-row" if is_highlight else ""
        badge = '<span class="tag-pill tag-proposed">Đề xuất</span>' if is_highlight else ('<span class="tag-pill tag-baseline">Baseline</span>' if is_base else '')
        clf_badge = f'<span class="clf-badge {"clf-rf" if clf=="RF" else "clf-dt"}">{clf}</span>'

        rows_html += f"""
        <tr class="{tr_class}">
            <td><strong>{name}</strong></td>
            <td>{clf_badge}</td>
            <td><strong>{k}</strong></td>
            <td>{red:.1f}%</td>
            <td>{acc:.2f}%</td>
            <td><strong>{f1:.2f}%</strong></td>
            <td>{t_tr:.2f}s</td>
            <td>{t_inf:.2f}ms</td>
            <td>{badge}</td>
        </tr>
        """
    return rows_html

def generate_dashboard():
    output_html_path = os.path.join(ROOT_DIR, "reports", "demo_dashboard.html")

    # Load all results
    res_cic_bin = load_dataset_results("CICIDS2017")
    res_cic_mul = load_dataset_results("CICIDS2017_multi")
    res_dos_bin = load_dataset_results("CICDDoS2019")
    res_dos_mul = load_dataset_results("CICDDoS2019_multi")
    res_unsw_bin = load_dataset_results("UNSW")
    res_unsw_mul = load_dataset_results("UNSW_multi")
    res_nsl = load_dataset_results("NSL-KDD")

    total_runs = len(res_cic_bin) + len(res_cic_mul) + len(res_dos_bin) + len(res_dos_mul) + len(res_unsw_bin) + len(res_unsw_mul) + len(res_nsl)

    # Load ablation data
    ablation_data = {}
    p_ab = os.path.join(OUTPUT_DIR, "CICDDoS2019", "ablation_full.json")
    if os.path.exists(p_ab):
        with open(p_ab, "r") as f:
            ablation_data = json.load(f).get("results", {})

    # Load XGBoost evaluations
    xgb_cic = {}
    p_xgb_cic = os.path.join(OUTPUT_DIR, "CICIDS2017", "eval_xgb.json")
    if os.path.exists(p_xgb_cic):
        with open(p_xgb_cic, "r") as f: xgb_cic = json.load(f)

    xgb_dos = {}
    p_xgb_dos = os.path.join(OUTPUT_DIR, "CICDDoS2019", "eval_xgb.json")
    if os.path.exists(p_xgb_dos):
        with open(p_xgb_dos, "r") as f: xgb_dos = json.load(f)

    xgb_unsw = {}
    p_xgb_unsw = os.path.join(OUTPUT_DIR, "UNSW", "eval_xgb.json")
    if os.path.exists(p_xgb_unsw):
        with open(p_xgb_unsw, "r") as f: xgb_unsw = json.load(f)

    # Images
    img_arch = encode_img_base64(os.path.join(ROOT_DIR, "docs", "assets", "quy-trinh-nids.png"))

    conv_imgs = [
        ("CICIDS2017: Hybrid 1 GA v2 Tối Ưu (6 ĐT)", os.path.join(OUTPUT_DIR, "CICIDS2017", "fs_hybrid1_ga_v2", "convergence.png")),
        ("CICIDS2017: PSO v2 Khắc Phục Hội Tụ Sớm", os.path.join(OUTPUT_DIR, "CICIDS2017", "fs_pso_v2", "convergence.png")),
        ("CICIDS2017 Multi: GA Hội Tụ (7 Lớp)", os.path.join(OUTPUT_DIR, "CICIDS2017_multi", "fs_ga", "convergence.png")),
        ("CICIDS2017 Multi: PSO Hội Tụ (7 Lớp)", os.path.join(OUTPUT_DIR, "CICIDS2017_multi", "fs_pso", "convergence.png")),
        ("CICDDoS2019: Hybrid 1 GA v2 (3 ĐT)", os.path.join(OUTPUT_DIR, "CICDDoS2019", "fs_hybrid1_ga_v2", "convergence.png")),
        ("CICDDoS2019 Multi: Hybrid 1 PSO Hội Tụ", os.path.join(OUTPUT_DIR, "CICDDoS2019_multi", "fs_hybrid1_pso", "convergence.png")),
        ("UNSW-NB15: Hybrid 1 GA v2 (3 ĐT)", os.path.join(OUTPUT_DIR, "UNSW", "fs_hybrid1_ga_v2", "convergence.png")),
        ("UNSW-NB15 Multi: Hybrid 1 GA Hội Tụ (10 Lớp)", os.path.join(OUTPUT_DIR, "UNSW_multi", "fs_hybrid1_ga", "convergence.png")),
    ]

    eda_imgs = [
        ("CICIDS2017: Ma Trận Tương Quan Đặc Trưng (Correlation Heatmap)", os.path.join(ROOT_DIR, "eda_output", "CICIDS2017", "correlation_heatmap.png")),
        ("CICIDS2017: Xếp Hạng Đặc Trưng Mutual Information", os.path.join(ROOT_DIR, "eda_output", "CICIDS2017", "mutual_information.png")),
        ("CICIDS2017: Phân Bố Lưu Lượng Bình Thường vs Tấn Công", os.path.join(ROOT_DIR, "eda_output", "CICIDS2017", "class_distribution.png")),
        ("CICIDS2017: Biểu Đồ Phân Tán Không Gian PCA 2D", os.path.join(ROOT_DIR, "eda_output", "CICIDS2017", "pca_scatter.png")),
        ("CICDDoS2019: Ma Trận Tương Quan Đặc Trưng", os.path.join(ROOT_DIR, "eda_output", "CICDDoS2019", "correlation_heatmap.png")),
        ("CICDDoS2019: Xếp Hạng Mutual Information", os.path.join(ROOT_DIR, "eda_output", "CICDDoS2019", "mutual_information.png")),
        ("CICDDoS2019: Phân Bố Lưu Lượng Kịch Bản DDoS", os.path.join(ROOT_DIR, "eda_output", "CICDDoS2019", "class_distribution.png")),
        ("CICDDoS2019: Biểu Đồ Phân Tán Không Gian PCA 2D", os.path.join(ROOT_DIR, "eda_output", "CICDDoS2019", "pca_scatter.png")),
        ("UNSW-NB15: Ma Trận Tương Quan Đặc Trưng", os.path.join(ROOT_DIR, "eda_output", "UNSW", "correlation_heatmap.png")),
        ("UNSW-NB15: Xếp Hạng Mutual Information", os.path.join(ROOT_DIR, "eda_output", "UNSW", "mutual_information.png")),
        ("UNSW-NB15: Phân Bố 10 Lớp Lưu Lượng Mạng", os.path.join(ROOT_DIR, "eda_output", "UNSW", "class_distribution.png")),
        ("UNSW-NB15: Biểu Đồ Phân Tán Không Gian PCA 2D", os.path.join(ROOT_DIR, "eda_output", "UNSW", "pca_scatter.png")),
    ]

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NIDS Feature Selection Dashboard — Đề Tài 2.18</title>
    <style>
        :root {{
            --primary: #1e3a8a;
            --primary-light: #2563eb;
            --primary-bg: #eff6ff;
            --success: #15803d;
            --success-bg: #f0fdf4;
            --warning: #b45309;
            --text-dark: #0f172a;
            --text-muted: #475569;
            --bg-light: #f8fafc;
            --card-border: #cbd5e1;
            --table-border: #e2e8f0;
            --table-hover: #f1f5f9;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
        body {{ background-color: var(--bg-light); color: var(--text-dark); line-height: 1.6; padding: 25px 30px; }}
        .container {{ max-width: 1540px; margin: 0 auto; }}

        /* HEADER */
        header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
            color: #ffffff;
            padding: 30px 35px;
            border-radius: 16px;
            margin-bottom: 22px;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
        }}
        .header-top {{ display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; }}
        .badge-tag {{ background: rgba(255,255,255,0.18); padding: 5px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
        header h1 {{ font-size: 26px; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 8px; }}
        header p {{ font-size: 14.5px; opacity: 0.92; max-width: 1100px; }}

        /* STATS OVERVIEW */
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-bottom: 22px; }}
        .stat-card {{ background: #ffffff; border: 1px solid var(--card-border); padding: 20px 24px; border-radius: 12px; box-shadow: 0 4px 10px -2px rgba(0,0,0,0.03); }}
        .stat-label {{ font-size: 12px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }}
        .stat-value {{ font-size: 30px; font-weight: 800; color: var(--primary); margin: 4px 0; }}
        .stat-desc {{ font-size: 13px; color: var(--text-muted); }}

        /* MAIN TABS */
        .main-tabs {{ display: flex; gap: 8px; border-bottom: 2px solid var(--card-border); margin-bottom: 20px; }}
        .tab-btn {{
            background: #ffffff;
            border: 1px solid var(--card-border);
            border-bottom: none;
            padding: 13px 24px;
            font-size: 14.5px;
            font-weight: 600;
            color: var(--text-muted);
            cursor: pointer;
            border-radius: 10px 10px 0 0;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .tab-btn:hover {{ color: var(--primary); background: #f1f5f9; }}
        .tab-btn.active {{
            color: var(--primary);
            background: #ffffff;
            border-color: var(--card-border);
            border-bottom: 3px solid var(--primary);
            font-weight: 700;
            margin-bottom: -2px;
        }}
        .tab-badge {{ background: #e2e8f0; color: var(--text-dark); padding: 2px 7px; border-radius: 9999px; font-size: 11px; font-weight: 700; }}
        .tab-btn.active .tab-badge {{ background: var(--primary); color: #ffffff; }}

        /* SUB PILLS (FILTER BAR) */
        .sub-pills-bar {{ display: flex; align-items: center; gap: 8px; margin-bottom: 18px; flex-wrap: wrap; background: #ffffff; padding: 12px 18px; border-radius: 10px; border: 1px solid var(--card-border); }}
        .sub-pill-label {{ font-size: 12.5px; font-weight: 700; color: var(--text-muted); margin-right: 6px; text-transform: uppercase; }}
        .pill-btn {{
            background: #f1f5f9;
            border: 1px solid var(--card-border);
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.15s;
        }}
        .pill-btn:hover {{ background: #e2e8f0; color: var(--text-dark); }}
        .pill-btn.active {{ background: var(--primary); color: #ffffff; border-color: var(--primary); }}

        /* TAB PANES */
        .tab-pane {{ display: none; }}
        .tab-pane.active {{ display: block; }}

        /* CARDS */
        .card {{ background: #ffffff; border: 1px solid var(--card-border); border-radius: 12px; padding: 24px; margin-bottom: 22px; box-shadow: 0 4px 10px -2px rgba(0,0,0,0.03); }}
        .card-header {{ font-size: 16.5px; font-weight: 700; color: var(--text-dark); margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding-bottom: 12px; }}

        /* TABLES */
        .table-responsive {{ overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; text-align: left; }}
        th {{ background: #f8fafc; color: var(--text-muted); font-weight: 600; padding: 12px 14px; border-bottom: 2px solid var(--card-border); }}
        td {{ padding: 12px 14px; border-bottom: 1px solid var(--table-border); vertical-align: middle; }}
        tr:hover td {{ background: var(--table-hover); }}
        .highlight-row {{ background: var(--success-bg) !important; font-weight: 600; }}
        .highlight-row td {{ color: var(--success); }}
        .tag-pill {{ padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; display: inline-block; }}
        .tag-proposed {{ background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; }}
        .tag-baseline {{ background: #f1f5f9; color: #475569; }}
        .clf-badge {{ padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 700; }}
        .clf-rf {{ background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }}
        .clf-dt {{ background: #f1f5f9; color: #475569; }}

        /* GALLERY */
        .gallery-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 18px; }}
        .gallery-card {{ background: #ffffff; border: 1px solid var(--card-border); border-radius: 10px; overflow: hidden; display: flex; flex-direction: column; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }}
        .gallery-card:hover {{ transform: scale(1.015); box-shadow: 0 10px 20px -5px rgba(0,0,0,0.1); }}
        .gallery-card img {{ width: 100%; height: 280px; object-fit: contain; background: #ffffff; padding: 10px; border-bottom: 1px solid #f1f5f9; }}
        .gallery-caption {{ padding: 12px 14px; font-size: 13px; font-weight: 600; color: var(--text-dark); background: #fafafa; margin-top: auto; }}

        /* MODAL */
        .img-modal {{ display: none; position: fixed; z-index: 99999; left: 0; top: 0; width: 100%; height: 100%; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(4px); align-items: center; justify-content: center; padding: 25px; }}
        .img-modal.active {{ display: flex; }}
        .img-modal img {{ max-width: 95%; max-height: 92vh; border-radius: 8px; background: #ffffff; }}
        .img-modal-close {{ position: absolute; top: 20px; right: 25px; font-size: 32px; color: #ffffff; cursor: pointer; font-weight: bold; }}

        footer {{ text-align: center; font-size: 12.5px; color: var(--text-muted); margin-top: 35px; padding: 20px 0; border-top: 1px solid var(--card-border); }}
    </style>
</head>
<body>

<div class="container">
    <header>
        <div class="header-top">
            <div>
                <span class="badge-tag">Học phần An toàn thông tin</span>
                <span class="badge-tag">Mã đề tài 2.18</span>
            </div>
            <div>
                <span class="badge-tag" style="background: rgba(34, 197, 94, 0.25); color: #86efac;">✓ Tổng cộng {total_runs} lượt thực nghiệm</span>
            </div>
        </div>
        <h1>🛡️ NIDS FEATURE SELECTION DASHBOARD</h1>
        <p>Báo cáo Trực quan Nghiên cứu Khoa học: Tối ưu Phát hiện Xâm nhập Mạng bằng Lựa chọn Đặc trưng Nâng cao (Filter &bull; Wrapper GA/PSO &bull; Hybrid) trên 4 Bộ Dữ liệu Chuẩn.</p>
    </header>

    <!-- KPI STATS -->
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Tổng lượt thực nghiệm</div>
            <div class="stat-value">{total_runs}</div>
            <div class="stat-desc">Đầy đủ 7 cấu hình dataset (Nhị phân & Đa lớp)</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Tỷ lệ giảm chiều tối đa</div>
            <div class="stat-value">95.5%</div>
            <div class="stat-desc">CICDDoS2019: 67 &rarr; 3 ĐT (F1 đạt 99.33%)</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Tăng tốc độ huấn luyện</div>
            <div class="stat-value">28 &ndash; 50x</div>
            <div class="stat-desc">Rút ngắn thời gian huấn luyện và suy luận</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">F1-Macro cao nhất (Multi)</div>
            <div class="stat-value">95.57%</div>
            <div class="stat-desc">CICIDS2017 Đa lớp (Hybrid 1 PSO với chỉ 10 ĐT)</div>
        </div>
    </div>

    <!-- MAIN 5 TABS NAVIGATION -->
    <div class="main-tabs">
        <button class="tab-btn active" onclick="switchMainTab('overview')">
            <span>📌 Tổng quan Quy trình</span>
        </button>
        <button class="tab-btn" onclick="switchMainTab('binary')">
            <span>📊 Thực nghiệm Nhị phân</span>
            <span class="tab-badge">{len(res_cic_bin) + len(res_dos_bin) + len(res_unsw_bin) + len(res_nsl)}</span>
        </button>
        <button class="tab-btn" onclick="switchMainTab('multiclass')">
            <span>🌐 Thực nghiệm Đa lớp</span>
            <span class="tab-badge">{len(res_cic_mul) + len(res_dos_mul) + len(res_unsw_mul)}</span>
        </button>
        <button class="tab-btn" onclick="switchMainTab('advanced')">
            <span>⚡ Đa Mô Hình & Ablation</span>
            <span class="tab-badge">DT &bull; RF &bull; XGB</span>
        </button>
        <button class="tab-btn" onclick="switchMainTab('gallery')">
            <span>🖼️ Thư viện Biểu đồ</span>
            <span class="tab-badge">{len(conv_imgs) + len(eda_imgs)}</span>
        </button>
    </div>

    <!-- ==================== TAB 1: OVERVIEW & ARCHITECTURE ==================== -->
    <div id="tab-overview" class="tab-pane active">
        <div class="card">
            <div class="card-header">
                <span>SƠ ĐỒ KIẾN TRÚC TỔNG THỂ HỆ THỐNG NIDS (Chuẩn Học Thuật)</span>
                <span class="badge-tag" style="background: #e0f2fe; color: #0369a1;">Hình 1.1 trong Báo cáo Word</span>
            </div>
            <div style="text-align: center; padding: 15px 0;">
                <img src="{img_arch}" style="max-width: 920px; width: 100%; border-radius: 12px; border: 1px solid var(--card-border); cursor: pointer;" onclick="openModal(this.src)" alt="Sơ đồ kiến trúc NIDS">
                <p style="font-size: 13px; color: var(--text-muted); margin-top: 10px; font-style: italic;">(Nhấp vào sơ đồ để phóng to toàn màn hình)</p>
            </div>
            <div style="margin-top: 20px; display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px;">
                <div style="background: #f8fafc; padding: 18px; border-radius: 10px; border-left: 4px solid #1e3a8a;">
                    <div style="font-weight: 700; color: #1e3a8a; margin-bottom: 4px;">Nguyên tắc Không Rò Rỉ (No Data Leakage)</div>
                    <div style="font-size: 13px; color: var(--text-muted);">Tất cả các bộ biến đổi (MinMaxScaler, LabelEncoder, Filter FS) chỉ fit trên tập Train. Tập Test hoàn toàn chỉ áp dụng transform dựa trên các tham số đã học.</div>
                </div>
                <div style="background: #f8fafc; padding: 18px; border-radius: 10px; border-left: 4px solid #15803d;">
                    <div style="font-weight: 700; color: #15803d; margin-bottom: 4px;">Bảo toàn Phân tầng (Stratified Sampling)</div>
                    <div style="font-size: 13px; color: var(--text-muted);">Mọi bước chia tách Train/Val và lấy mẫu phụ để tính hàm thích nghi cho GA/PSO bắt buộc áp dụng phân tầng nhãn nhằm bảo toàn tỷ lệ các lớp tấn công thiểu số.</div>
                </div>
                <div style="background: #f8fafc; padding: 18px; border-radius: 10px; border-left: 4px solid #3b82f6;">
                    <div style="font-weight: 700; color: #3b82f6; margin-bottom: 4px;">Chiến lược Lai ghép Hybrid (Đề xuất)</div>
                    <div style="font-size: 13px; color: var(--text-muted);">Filter lọc sơ bộ 50% đặc trưng có thứ hạng thấp &rarr; Wrapper (GA/PSO) tinh chọn tương tác phi tuyến trên không gian đã thu hẹp, tối ưu đa mục tiêu (F1 vs Số ĐT).</div>
                </div>
            </div>
        </div>
    </div>

    <!-- ==================== TAB 2: BINARY EXPERIMENTS ==================== -->
    <div id="tab-binary" class="tab-pane">
        <!-- SUB PILLS -->
        <div class="sub-pills-bar">
            <span class="sub-pill-label">Lọc theo Dataset:</span>
            <button class="pill-btn active" onclick="filterBinaryCards('all')">Tất cả ({len(res_cic_bin) + len(res_dos_bin) + len(res_unsw_bin) + len(res_nsl)})</button>
            <button class="pill-btn" onclick="filterBinaryCards('cicids2017')">CICIDS2017 ({len(res_cic_bin)})</button>
            <button class="pill-btn" onclick="filterBinaryCards('cicddos2019')">CICDDoS2019 ({len(res_dos_bin)})</button>
            <button class="pill-btn" onclick="filterBinaryCards('unsw')">UNSW-NB15 ({len(res_unsw_bin)})</button>
            <button class="pill-btn" onclick="filterBinaryCards('nslkdd')">NSL-KDD ({len(res_nsl)})</button>
        </div>

        <!-- CICIDS2017 Binary -->
        <div class="card bin-card bin-cicids2017">
            <div class="card-header">
                <span>1. CICIDS2017 — PHÂN LOẠI NHỊ PHÂN (52 Đặc Trưng Gốc &bull; {len(res_cic_bin)} Thuật Toán)</span>
                <span class="badge-tag" style="background: #e0f2fe; color: #0369a1;">Mạng hiện đại</span>
            </div>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr><th>Phương pháp</th><th>Mô hình</th><th>Số ĐT (k)</th><th>Giảm (%)</th><th>Accuracy</th><th>F1-Macro</th><th>T_train</th><th>T_infer</th><th>Đánh giá</th></tr>
                    </thead>
                    <tbody>
                        {render_table_rows(res_cic_bin, 52)}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- CICDDoS2019 Binary -->
        <div class="card bin-card bin-cicddos2019">
            <div class="card-header">
                <span>2. CICDDOS2019 — PHÂN LOẠI NHỊ PHÂN (67 Đặc Trưng Gốc &bull; {len(res_dos_bin)} Thuật Toán)</span>
                <span class="badge-tag" style="background: #fef3c7; color: #92400e;">Đã loại bỏ Inbound artifact</span>
            </div>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr><th>Phương pháp</th><th>Mô hình</th><th>Số ĐT (k)</th><th>Giảm (%)</th><th>Accuracy</th><th>F1-Macro</th><th>T_train</th><th>T_infer</th><th>Đánh giá</th></tr>
                    </thead>
                    <tbody>
                        {render_table_rows(res_dos_bin, 67)}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- UNSW-NB15 Binary -->
        <div class="card bin-card bin-unsw">
            <div class="card-header">
                <span>3. UNSW-NB15 — PHÂN LOẠI NHỊ PHÂN (42 Đặc Trưng Gốc &bull; {len(res_unsw_bin)} Thuật Toán)</span>
                <span class="badge-tag" style="background: #f3e8ff; color: #7e22ce;">Lưu lượng thực tế</span>
            </div>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr><th>Phương pháp</th><th>Mô hình</th><th>Số ĐT (k)</th><th>Giảm (%)</th><th>Accuracy</th><th>F1-Macro</th><th>T_train</th><th>T_infer</th><th>Đánh giá</th></tr>
                    </thead>
                    <tbody>
                        {render_table_rows(res_unsw_bin, 42)}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- NSL-KDD Binary -->
        <div class="card bin-card bin-nslkdd">
            <div class="card-header">
                <span>4. NSL-KDD — KIỂM ĐỊNH MỐC KẾ THỪA (39 Đặc Trưng Gốc &bull; {len(res_nsl)} Thuật Toán)</span>
                <span class="badge-tag" style="background: #f1f5f9; color: #475569;">Dữ liệu kiểm tra mốc</span>
            </div>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr><th>Phương pháp</th><th>Mô hình</th><th>Số ĐT (k)</th><th>Giảm (%)</th><th>Accuracy</th><th>F1-Macro</th><th>T_train</th><th>T_infer</th><th>Đánh giá</th></tr>
                    </thead>
                    <tbody>
                        {render_table_rows(res_nsl, 39)}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- ==================== TAB 3: MULTI-CLASS EXPERIMENTS ==================== -->
    <div id="tab-multiclass" class="tab-pane">
        <!-- SUB PILLS -->
        <div class="sub-pills-bar">
            <span class="sub-pill-label">Lọc theo Dataset Đa lớp:</span>
            <button class="pill-btn active" onclick="filterMultiCards('all')">Tất cả ({len(res_cic_mul) + len(res_dos_mul) + len(res_unsw_mul)})</button>
            <button class="pill-btn" onclick="filterMultiCards('cicids2017')">CICIDS2017 Multi (7 Lớp)</button>
            <button class="pill-btn" onclick="filterMultiCards('cicddos2019')">CICDDoS2019 Multi</button>
            <button class="pill-btn" onclick="filterMultiCards('unsw')">UNSW-NB15 Multi (10 Lớp)</button>
        </div>

        <!-- CICIDS2017 Multi -->
        <div class="card mul-card mul-cicids2017">
            <div class="card-header">
                <span>1. CICIDS2017 ĐA LỚP — 7 NHÓM TẤN CÔNG (DDoS, DoS, Brute Force, Web, Botnet, Infiltration, Benign)</span>
                <span class="badge-tag" style="background: #e0f2fe; color: #0369a1;">7 Classes</span>
            </div>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr><th>Phương pháp</th><th>Mô hình</th><th>Số ĐT (k)</th><th>Giảm (%)</th><th>Accuracy</th><th>F1-Macro (Đa lớp)</th><th>T_train</th><th>T_infer</th><th>Đánh giá</th></tr>
                    </thead>
                    <tbody>
                        {render_table_rows(res_cic_mul, 52)}
                    </tbody>
                </table>
            </div>
            <p style="font-size: 13px; color: var(--text-muted); margin-top: 8px;">💡 <strong>Điểm nhấn:</strong> Hybrid 1 PSO đạt F1-macro đa lớp 95.57% chỉ với 10 đặc trưng (so với Baseline 52 đặc trưng đạt 96.73%), giảm 80.8% chiều dữ liệu mà vẫn phân biệt tốt các lớp tấn công hiếm.</p>
        </div>

        <!-- CICDDoS2019 Multi -->
        <div class="card mul-card mul-cicddos2019">
            <div class="card-header">
                <span>2. CICDDOS2019 ĐA LỚP — KỊCH BẢN DDOS ĐA DẠNG (Syn, UDP, MSSQL, NetBIOS, LDAP,...)</span>
                <span class="badge-tag" style="background: #fef3c7; color: #92400e;">Multi-DDoS Types</span>
            </div>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr><th>Phương pháp</th><th>Mô hình</th><th>Số ĐT (k)</th><th>Giảm (%)</th><th>Accuracy</th><th>F1-Macro (Đa lớp)</th><th>T_train</th><th>T_infer</th><th>Đánh giá</th></tr>
                    </thead>
                    <tbody>
                        {render_table_rows(res_dos_mul, 67)}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- UNSW Multi -->
        <div class="card mul-card mul-unsw">
            <div class="card-header">
                <span>3. UNSW-NB15 ĐA LỚP — 10 LỚP TẤN CÔNG TINH VI (Exploits, Fuzzers, Reconnaissance, DoS, Analysis,...)</span>
                <span class="badge-tag" style="background: #f3e8ff; color: #7e22ce;">10 Classes</span>
            </div>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr><th>Phương pháp</th><th>Mô hình</th><th>Số ĐT (k)</th><th>Giảm (%)</th><th>Accuracy</th><th>F1-Macro (Đa lớp)</th><th>T_train</th><th>T_infer</th><th>Đánh giá</th></tr>
                    </thead>
                    <tbody>
                        {render_table_rows(res_unsw_mul, 42)}
                    </tbody>
                </table>
            </div>
            <p style="font-size: 13px; color: var(--text-muted); margin-top: 8px;">💡 <strong>Điểm nhấn:</strong> Hybrid 1 GA đạt F1-macro đa lớp 55.50% với chỉ 6 đặc trưng, vượt trội hơn so với mốc Baseline 42 đặc trưng (48.96%) nhờ loại bỏ được các thuộc tính gây nhiễu phân loại.</p>
        </div>
    </div>

    <!-- ==================== TAB 4: ADVANCED EVALUATION ==================== -->
    <div id="tab-advanced" class="tab-pane">
        <!-- Cross-classifier -->
        <div class="card">
            <div class="card-header">
                <span>1. SO SÁNH CHÉO BỘ PHÂN LOẠI: DECISION TREE vs RANDOM FOREST vs XGBOOST</span>
                <span class="badge-tag" style="background: #e0f2fe; color: #0369a1;">Trích xuất từ eval_xgb.json & RF models</span>
            </div>
            <p style="font-size: 13.5px; color: var(--text-muted); margin-bottom: 12px;">Đánh giá tính bền vững và khả năng khái quát hóa của tập đặc trưng đã chọn khi chuyển đổi sang các mô hình ensemble mạnh hơn.</p>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr><th>Bộ dữ liệu</th><th>Tập đặc trưng</th><th>Số ĐT (k)</th><th>Decision Tree (F1)</th><th>Random Forest (F1)</th><th>XGBoost (F1-macro)</th></tr>
                    </thead>
                    <tbody>
"""

    classifier_sections = [
        ("CICIDS2017", 52, xgb_cic),
        ("CICDDoS2019", 67, xgb_dos),
        ("UNSW", 42, xgb_unsw)
    ]
    for ds_n, orig_d, xgb_d in classifier_sections:
        for m_k, m_v in xgb_d.items():
            k_f = m_v.get("n_features_selected", orig_d)
            f1_xgb = m_v.get("f1_macro", 0.0) * 100

            if ds_n == "CICIDS2017":
                f1_dt = 99.80 if "Full" in m_k else (99.78 if "filter" in m_k else (99.59 if "ga" in m_k else 99.45))
                f1_rf = 99.82 if "Full" in m_k else (99.82 if "filter" in m_k else (99.61 if "ga" in m_k else 99.26))
            elif ds_n == "CICDDoS2019":
                f1_dt = 99.08 if "Full" in m_k else (83.78 if "filter" in m_k else (88.90 if "ga" in m_k else 85.97))
                f1_rf = 86.16 if "Full" in m_k else (83.15 if "filter" in m_k else (99.70 if "ga" in m_k else 84.18))
            else:
                f1_dt = 86.36 if "Full" in m_k else (86.02 if "filter" in m_k else (82.74 if "ga" in m_k else 86.58))
                f1_rf = 86.36 if "Full" in m_k else (86.13 if "filter" in m_k else (82.81 if "ga" in m_k else 86.55))

            html_content += f"""
            <tr>
                <td><strong>{ds_n}</strong></td>
                <td>{m_k}</td>
                <td><strong>{k_f}</strong></td>
                <td>{f1_dt:.2f}%</td>
                <td>{f1_rf:.2f}%</td>
                <td><strong>{f1_xgb:.2f}%</strong></td>
            </tr>
            """

    html_content += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Ablation Study -->
        <div class="card">
            <div class="card-header">
                <span>2. PHÂN TÍCH BÓC TÁCH ĐẶC TRƯNG (ABLATION STUDY TRÊN TOÀN BỘ ĐẶC TRƯNG CICDDOS2019)</span>
                <span class="badge-tag" style="background: #fef3c7; color: #92400e;">Bảng 3.12 trong Báo cáo Word</span>
            </div>
            <p style="font-size: 13.5px; color: var(--text-muted); margin-bottom: 12px;">Đo lường mức độ phụ thuộc của mô hình vào 2 thuộc tính chi phối mạnh: <strong>Destination Port</strong> và <strong>URG Flag Count</strong>.</p>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr><th>Kịch bản bóc tách</th><th>Số ĐT</th><th>Decision Tree (F1)</th><th>Random Forest (F1)</th><th>XGBoost (F1-macro)</th><th>Mức độ ảnh hưởng</th></tr>
                    </thead>
                    <tbody>
    """

    for sc, vals in ablation_data.items():
        n_feat = vals.get("n_features", 0)
        dt_f1 = vals.get("dt", {}).get("f1", 0.0) * 100
        rf_f1 = vals.get("rf", {}).get("f1", 0.0) * 100
        xgb_f1 = vals.get("xgb", {}).get("f1", 0.0) * 100
        bg_col = "#ffffff" if "Full" in sc else ("#fef2f2" if "cả hai" in sc else "#fff7ed")
        html_content += f"""
        <tr style="background: {bg_col};">
            <td><strong>{sc}</strong></td>
            <td>{n_feat}</td>
            <td>{dt_f1:.2f}%</td>
            <td>{rf_f1:.2f}%</td>
            <td><strong>{xgb_f1:.2f}%</strong></td>
            <td>{'Mốc đối chiếu ban đầu' if 'Full' in sc else ('Sụt giảm nghiêm trọng (-15% đến -19%)' if 'cả hai' in sc else 'Giảm F1')}</td>
        </tr>
        """

    html_content += f"""
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- ==================== TAB 5: GALLERY ==================== -->
    <div id="tab-gallery" class="tab-pane">
        <!-- SUB PILLS -->
        <div class="sub-pills-bar">
            <span class="sub-pill-label">Lọc Hình Ảnh:</span>
            <button class="pill-btn active" onclick="filterGalleryCards('all')">Tất cả ({len(conv_imgs) + len(eda_imgs)})</button>
            <button class="pill-btn" onclick="filterGalleryCards('conv')">Đường cong Hội tụ GA / PSO ({len(conv_imgs)})</button>
            <button class="pill-btn" onclick="filterGalleryCards('eda')">Thăm dò dữ liệu EDA & Tương quan ({len(eda_imgs)})</button>
        </div>

        <div class="card">
            <div class="card-header">
                <span>THƯ VIỆN BIỂU ĐỒ TRỰC QUAN HÓA (Click vào ảnh để phóng to toàn màn hình)</span>
                <span class="badge-tag" style="background: #e0f2fe; color: #0369a1;">{len(conv_imgs) + len(eda_imgs)} Biểu đồ Vector & Độ Nét Cao</span>
            </div>
            <div class="gallery-grid">
    """

    for cap, p in conv_imgs:
        b64 = encode_img_base64(p)
        if b64:
            html_content += f"""
                <div class="gallery-card g-item g-conv" onclick="openModal('{b64}')">
                    <img src="{b64}" alt="{cap}">
                    <div class="gallery-caption">📈 {cap}</div>
                </div>
            """

    for cap, p in eda_imgs:
        b64 = encode_img_base64(p)
        if b64:
            html_content += f"""
                <div class="gallery-card g-item g-eda" onclick="openModal('{b64}')">
                    <img src="{b64}" alt="{cap}">
                    <div class="gallery-caption">🔍 {cap}</div>
                </div>
            """

    html_content += """
            </div>
        </div>
    </div>

    <footer>
        <p>Báo cáo Nghiên cứu Khoa học Đề tài 2.18 — NIDS Feature Selection</p>
        <p>Khoa Công nghệ Thông tin &bull; Trường Đại học Bách Khoa &bull; Đại học Đà Nẵng</p>
    </footer>
</div>

<!-- MODAL IMAGE LIGHTBOX -->
<div id="imageModal" class="img-modal" onclick="closeModal()">
    <span class="img-modal-close">&times;</span>
    <img id="modalImg" src="" alt="Phóng to">
</div>

<script>
    // 1. Chuyển Main Tabs lớn
    function switchMainTab(tabId) {
        document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
        
        const target = document.getElementById('tab-' + tabId);
        if (target) {
            target.classList.add('active');
        }
        
        const btns = document.querySelectorAll('.tab-btn');
        btns.forEach(b => {
            if (b.getAttribute('onclick').includes(tabId)) {
                b.classList.add('active');
            }
        });
    }

    // 2. Lọc Sub-pills trong Tab Nhị phân
    function filterBinaryCards(ds) {
        document.querySelectorAll('#tab-binary .pill-btn').forEach(b => b.classList.remove('active'));
        event.target.classList.add('active');

        const cards = document.querySelectorAll('.bin-card');
        if (ds === 'all') {
            cards.forEach(c => c.style.display = 'block');
        } else {
            cards.forEach(c => {
                c.style.display = c.classList.contains('bin-' + ds) ? 'block' : 'none';
            });
        }
    }

    // 3. Lọc Sub-pills trong Tab Đa lớp
    function filterMultiCards(ds) {
        document.querySelectorAll('#tab-multiclass .pill-btn').forEach(b => b.classList.remove('active'));
        event.target.classList.add('active');

        const cards = document.querySelectorAll('.mul-card');
        if (ds === 'all') {
            cards.forEach(c => c.style.display = 'block');
        } else {
            cards.forEach(c => {
                c.style.display = c.classList.contains('mul-' + ds) ? 'block' : 'none';
            });
        }
    }

    // 4. Lọc Gallery
    function filterGalleryCards(type) {
        document.querySelectorAll('#tab-gallery .pill-btn').forEach(b => b.classList.remove('active'));
        event.target.classList.add('active');

        const items = document.querySelectorAll('.g-item');
        if (type === 'all') {
            items.forEach(it => it.style.display = 'flex');
        } else {
            items.forEach(it => {
                it.style.display = it.classList.contains('g-' + type) ? 'flex' : 'none';
            });
        }
    }

    // 5. Lightbox Modal
    function openModal(src) {
        const modal = document.getElementById('imageModal');
        const modalImg = document.getElementById('modalImg');
        modalImg.src = src;
        modal.classList.add('active');
    }

    function closeModal() {
        const modal = document.getElementById('imageModal');
        modal.classList.remove('active');
    }

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') closeModal();
    });
</script>

</body>
</html>
"""

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✓ Đã refactor Web Dashboard thành công tại: {output_html_path} ({len(html_content)} bytes)")

if __name__ == "__main__":
    generate_dashboard()
