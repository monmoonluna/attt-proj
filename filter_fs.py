"""
filter_fs.py
============
Feature Selection bằng phương pháp Filter truyền thống - Mutual Information
(mặc định) hoặc Chi-square - đúng Kịch bản 2 trong đề cương (mục 6.6), phần
còn thiếu trước khi có thể làm Hybrid FS.

Khác với GA/PSO (Wrapper), Filter KHÔNG cần huấn luyện mô hình để đánh giá
từng tập con đặc trưng - chỉ tính 1 điểm số thống kê độc lập cho từng đặc
trưng rồi xếp hạng, nên rất nhanh (không có vòng lặp tối ưu).

Cách dùng:
    python filter_fs.py --data-dir output/CICIDS2017 --method mi \
        --top-percent 30 --classifier dt --output-dir output/CICIDS2017/fs_filter
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import MinMaxScaler

from feature_selection import load_split, evaluate_on_full_data, CLASSIFIERS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("filter_fs")


def compute_scores(X: pd.DataFrame, y: pd.Series, method: str, sample_size: int, random_state: int):
    if sample_size and len(X) > sample_size:
        X_sub = X.sample(n=sample_size, random_state=random_state)
        y_sub = y.loc[X_sub.index]
    else:
        X_sub, y_sub = X, y

    if method == "mi":
        scores = mutual_info_classif(X_sub, y_sub, random_state=random_state)
    elif method == "chi2":
        # chi2 yêu cầu giá trị không âm - dữ liệu đã qua MinMaxScaler ở preprocessing.py
        # nên thường đã không âm, nhưng scale lại cho chắc chắn.
        X_nonneg = MinMaxScaler().fit_transform(X_sub)
        scores, _ = chi2(X_nonneg, y_sub)
        scores = np.nan_to_num(scores)
    else:
        raise ValueError(f"method '{method}' không hỗ trợ. Chọn 'mi' hoặc 'chi2'.")

    return pd.Series(scores, index=X.columns).sort_values(ascending=False)


def select_top(scores: pd.Series, top_k: int | None, top_percent: float | None) -> list[str]:
    n = len(scores)
    if top_k:
        k = min(top_k, n)
    elif top_percent:
        k = max(1, round(n * top_percent / 100))
    else:
        raise ValueError("Cần truyền --top-k hoặc --top-percent")
    return scores.head(k).index.tolist()


def parse_args():
    parser = argparse.ArgumentParser(description="Filter Feature Selection (Mutual Information / Chi-square)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--method", choices=["mi", "chi2"], default="mi")
    parser.add_argument("--classifier", choices=list(CLASSIFIERS), default="dt")
    parser.add_argument("--top-k", type=int, default=None, help="Số đặc trưng muốn giữ lại (ưu tiên hơn --top-percent nếu truyền cả hai)")
    parser.add_argument("--top-percent", type=float, default=30.0, help="% số đặc trưng muốn giữ lại (mặc định 30%%)")
    parser.add_argument("--sample-size", type=int, default=20000, help="Số dòng lấy mẫu để tính điểm filter cho nhanh (0 = dùng toàn bộ)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split(args.data_dir)
    sample_size = args.sample_size if args.sample_size > 0 else None

    t0 = time.time()
    scores = compute_scores(X_train, y_train, args.method, sample_size, args.random_state)
    search_time = time.time() - t0

    selected = select_top(scores, args.top_k, args.top_percent)
    mask = np.array([1 if c in selected else 0 for c in X_train.columns], dtype=np.uint8)

    scores.to_csv(os.path.join(args.output_dir, f"{args.method}_scores.csv"), header=["score"])
    logger.info(
        "Đã chọn %d/%d đặc trưng bằng %s (thời gian tính điểm: %.2fs). Top 10: %s",
        mask.sum(), len(mask), args.method.upper(), search_time, selected[:10],
    )

    logger.info("Đang đánh giá trên toàn bộ tập train/test thật...")
    result = evaluate_on_full_data(X_train, X_test, y_train, y_test, mask, args.classifier)
    result.update({
        "method": f"filter_{args.method}",
        "classifier": args.classifier,
        "search_time_sec": search_time,
    })

    full_mask = np.ones(len(X_train.columns), dtype=np.uint8)
    logger.info("Đang đánh giá baseline (không Feature Selection) để so sánh...")
    baseline = evaluate_on_full_data(X_train, X_test, y_train, y_test, full_mask, args.classifier)

    with open(os.path.join(args.output_dir, "selected_features.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "baseline_full_features.json"), "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)

    logger.info("=" * 70)
    logger.info(
        "KẾT QUẢ SO SÁNH | Full features: %d đặc trưng, F1=%.4f, Acc=%.4f, train=%.2fs",
        baseline["n_features_total"], baseline["f1_macro"], baseline["accuracy"], baseline["train_time_sec"],
    )
    logger.info(
        "KẾT QUẢ SO SÁNH | Filter-%-4s: %d đặc trưng, F1=%.4f, Acc=%.4f, train=%.2fs",
        args.method.upper(), result["n_features_selected"], result["f1_macro"], result["accuracy"], result["train_time_sec"],
    )
    logger.info("Đã lưu kết quả vào: %s", args.output_dir)


if __name__ == "__main__":
    main()
