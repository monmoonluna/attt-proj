"""
eda.py
======
Phân tích khám phá dữ liệu (Exploratory Data Analysis) cho các bộ dữ liệu NIDS,
chạy TRƯỚC bước tiền xử lý chính thức (preprocessing.py) - đúng theo mục 6.2
của đề cương.

Tái sử dụng trực tiếp NIDSPreprocessor.load() / clean_missing_and_infinite() /
build_labels() từ preprocessing.py để đảm bảo logic đọc dữ liệu và gán nhãn
NHẤT QUÁN giữa bước EDA và bước tiền xử lý chính thức (tránh lệch nhau, nguồn
gốc của lỗi "Normal Traffic" từng gặp trên CICIDS2017).

Nội dung phân tích:
    1. Phân bố nhãn gốc (multi-class) và nhãn nhị phân (Normal/Attack)
    2. Thống kê mô tả từng đặc trưng (min/max/mean/std/phân vị)
    3. Ma trận tương quan giữa các đặc trưng số (correlation heatmap)
    4. Mutual Information giữa từng đặc trưng và nhãn nhị phân - CẢNH BÁO SỚM
       các đặc trưng có dấu hiệu rò rỉ nhãn (MI bất thường cao)
    5. Trực quan hóa PCA 2D theo nhãn nhị phân

Cách dùng:
    python eda.py --input data/cicids2017_cleaned.csv --dataset cicids2017 \
        --output-dir eda_output/CICIDS2017 --sample-size 20000
"""

from __future__ import annotations

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from preprocessing import NIDSPreprocessor, DATASET_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eda")

# Ngưỡng MI để cảnh báo nghi ngờ rò rỉ dữ liệu: một đặc trưng đơn lẻ mà MI
# lớn hơn ngưỡng này (tỉ lệ so với entropy của nhãn) rất đáng ngờ, nên kiểm
# tra kỹ trước khi tin dùng.
MI_LEAKAGE_WARNING_RATIO = 0.8


def _encode_for_analysis(X: pd.DataFrame) -> pd.DataFrame:
    """Mã hóa nhanh các cột phân loại (chỉ phục vụ EDA, không lưu lại encoder)."""
    X = X.copy()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X.fillna(0)


def run_eda(input_path: str, dataset: str, output_dir: str, sample_size: int = 20000,
            random_state: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    pre = NIDSPreprocessor(dataset=dataset, random_state=random_state)

    df = pre.load(input_path)
    df = pre.clean_missing_and_infinite(df)
    X, multi_label, binary_label = pre.build_labels(df)
    X_num = _encode_for_analysis(X)

    report_lines = [f"# Báo cáo EDA - {dataset}\n", f"File nguồn: `{input_path}`\n",
                     f"Kích thước dữ liệu (sau khi làm sạch cơ bản): {df.shape[0]} dòng, "
                     f"{X.shape[1]} đặc trưng (chưa tính cột nhãn)\n"]

    # ------------------------------------------------------------------ #
    # 1. Phân bố nhãn
    # ------------------------------------------------------------------ #
    logger.info("Đang phân tích phân bố nhãn...")
    binary_counts = binary_label.value_counts()
    multi_counts = multi_label.value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    binary_counts.plot(kind="bar", ax=axes[0], color=["#4C72B0", "#C44E52"])
    axes[0].set_title("Phân bố nhãn nhị phân (Normal/Attack)")
    axes[0].set_ylabel("Số lượng mẫu")
    for i, v in enumerate(binary_counts.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    multi_counts.plot(kind="bar", ax=axes[1], color="#55A868")
    axes[1].set_title("Phân bố nhãn gốc (multi-class)")
    axes[1].set_ylabel("Số lượng mẫu")
    axes[1].tick_params(axis="x", labelrotation=60)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150)
    plt.close()

    imbalance_ratio = binary_counts.max() / max(binary_counts.min(), 1)
    report_lines.append("## 1. Phân bố nhãn\n")
    report_lines.append(f"- Nhãn nhị phân: {binary_counts.to_dict()}")
    report_lines.append(f"- Tỉ lệ mất cân bằng (lớp lớn nhất / lớp nhỏ nhất): **{imbalance_ratio:.2f}x**")
    report_lines.append(f"- Số lớp trong nhãn gốc (multi-class): {multi_label.nunique()}")
    if imbalance_ratio > 10:
        report_lines.append("- ⚠️ Mất cân bằng lớp khá nghiêm trọng (>10x) - nên cân nhắc SMOTE/class-weighting khi huấn luyện.")
    report_lines.append("")

    # ------------------------------------------------------------------ #
    # Lấy mẫu để các bước sau (correlation/MI/PCA) chạy nhanh trên dữ liệu lớn
    # ------------------------------------------------------------------ #
    if sample_size and len(X_num) > sample_size:
        idx = X_num.sample(n=sample_size, random_state=random_state).index
        X_sample = X_num.loc[idx]
        y_sample = binary_label.loc[idx] if hasattr(binary_label, "loc") else pd.Series(binary_label).loc[idx]
        logger.info("Lấy mẫu %d/%d dòng cho correlation/MI/PCA", sample_size, len(X_num))
    else:
        X_sample, y_sample = X_num, pd.Series(binary_label, index=X_num.index)

    y_sample_encoded = LabelEncoder().fit_transform(y_sample)

    # ------------------------------------------------------------------ #
    # 2. Thống kê mô tả
    # ------------------------------------------------------------------ #
    logger.info("Đang tính thống kê mô tả...")
    desc = X.describe().T
    desc.to_csv(os.path.join(output_dir, "descriptive_stats.csv"))
    n_constant = (desc["std"] == 0).sum()
    report_lines.append("## 2. Thống kê mô tả\n")
    report_lines.append(f"- Chi tiết đầy đủ: `descriptive_stats.csv`")
    report_lines.append(f"- Số đặc trưng có phương sai bằng 0 (hằng số, vô dụng cho mô hình): {n_constant}")
    report_lines.append("")

    # ------------------------------------------------------------------ #
    # 3. Ma trận tương quan
    # ------------------------------------------------------------------ #
    logger.info("Đang tính ma trận tương quan...")
    corr = X_sample.corr()
    fig_size = max(8, min(0.25 * len(corr.columns), 24))
    plt.figure(figsize=(fig_size, fig_size))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=6)
    plt.title("Ma trận tương quan giữa các đặc trưng")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # Các cặp đặc trưng tương quan rất cao (> 0.95) - ứng viên loại bớt
    high_corr_pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if abs(v) > 0.95:
                high_corr_pairs.append((cols[i], cols[j], round(float(v), 3)))
    report_lines.append("## 3. Tương quan giữa các đặc trưng\n")
    report_lines.append(f"- Biểu đồ: `correlation_heatmap.png`")
    report_lines.append(f"- Số cặp đặc trưng có |tương quan| > 0.95: {len(high_corr_pairs)}")
    if high_corr_pairs:
        report_lines.append("- Top 10 cặp tương quan cao nhất:")
        for a, b, v in sorted(high_corr_pairs, key=lambda t: -abs(t[2]))[:10]:
            report_lines.append(f"    - `{a}` ↔ `{b}`: {v}")
    report_lines.append("")

    # ------------------------------------------------------------------ #
    # 4. Mutual Information với nhãn nhị phân - soát rò rỉ dữ liệu
    # ------------------------------------------------------------------ #
    logger.info("Đang tính Mutual Information...")
    mi = mutual_info_classif(X_sample, y_sample_encoded, random_state=random_state)
    mi_series = pd.Series(mi, index=X_sample.columns).sort_values(ascending=False)
    mi_series.to_csv(os.path.join(output_dir, "mutual_information.csv"), header=["mutual_information"])

    plt.figure(figsize=(9, max(4, 0.3 * min(len(mi_series), 25))))
    mi_series.head(25).sort_values().plot(kind="barh", color="#8172B2")
    plt.title("Mutual Information giữa đặc trưng và nhãn nhị phân (top 25)")
    plt.xlabel("Mutual Information")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mutual_information.png"), dpi=150)
    plt.close()

    max_mi = mi_series.iloc[0]
    label_entropy = -sum(
        (c / len(y_sample)) * np.log(c / len(y_sample)) for c in pd.Series(y_sample_encoded).value_counts()
    )
    report_lines.append("## 4. Mutual Information (soát rò rỉ dữ liệu)\n")
    report_lines.append(f"- Chi tiết: `mutual_information.csv`, biểu đồ: `mutual_information.png`")
    report_lines.append(f"- Entropy của nhãn (tham chiếu): {label_entropy:.4f}")
    report_lines.append(f"- Đặc trưng có MI cao nhất: `{mi_series.index[0]}` = {max_mi:.4f}")
    if label_entropy > 0 and max_mi / label_entropy > MI_LEAKAGE_WARNING_RATIO:
        report_lines.append(
            f"- ⚠️ **CẢNH BÁO**: MI của `{mi_series.index[0]}` chiếm "
            f"{max_mi / label_entropy:.0%} entropy của nhãn - rất đáng ngờ, cần kiểm tra "
            f"xem đặc trưng này có vô tình mã hóa trực tiếp nhãn hay không trước khi dùng "
            f"cho Feature Selection chính thức (tương tự trường hợp Bwd Packet Length Max "
            f"từng gặp trên CICIDS2017)."
        )
    else:
        report_lines.append("- Không phát hiện đặc trưng nào có MI bất thường cao - chưa có dấu hiệu rò rỉ rõ ràng.")
    report_lines.append("")

    # ------------------------------------------------------------------ #
    # 5. PCA 2D
    # ------------------------------------------------------------------ #
    logger.info("Đang tính PCA 2D...")
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_sample)
    plt.figure(figsize=(7, 6))
    for label_val in pd.unique(y_sample):
        mask = (y_sample.values if hasattr(y_sample, "values") else y_sample) == label_val
        plt.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.4, label=str(label_val))
    plt.legend(markerscale=3)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} phương sai)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} phương sai)")
    plt.title("Trực quan hóa PCA 2D theo nhãn nhị phân")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_scatter.png"), dpi=150)
    plt.close()

    report_lines.append("## 5. PCA 2D\n")
    report_lines.append(f"- Biểu đồ: `pca_scatter.png`")
    report_lines.append(
        f"- 2 thành phần chính giải thích {sum(pca.explained_variance_ratio_):.1%} tổng phương sai dữ liệu."
    )
    report_lines.append("")

    report_path = os.path.join(output_dir, "eda_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info("Hoàn tất EDA. Báo cáo: %s", report_path)
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(description="EDA cho dữ liệu NIDS")
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=20000,
                         help="Số dòng lấy mẫu cho correlation/MI/PCA (0 = dùng toàn bộ, chậm với dữ liệu lớn)")
    return parser.parse_args()


def main():
    args = parse_args()
    run_eda(args.input, args.dataset, args.output_dir,
             sample_size=args.sample_size if args.sample_size > 0 else None)


if __name__ == "__main__":
    main()
