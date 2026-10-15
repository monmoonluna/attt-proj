"""
hybrid2_fs.py
=============
Hybrid Feature Selection 2 - Mutual Information + SHAP + XGBoost
(Explainable-AI based Feature Selection), đúng Kịch bản 5 trong đề cương
(mục 6.6), chạy song song để so sánh với Hybrid 1 (Filter + GA/PSO).

Các bước:
    1. Tính điểm Mutual Information (Filter) cho từng đặc trưng.
    2. Huấn luyện XGBoost trên toàn bộ đặc trưng.
    3. Tính SHAP value (mean |SHAP|) cho từng đặc trưng từ mô hình XGBoost -
       đo đóng góp THỰC SỰ của từng đặc trưng vào dự đoán của mô hình, khác
       với MI (chỉ đo phụ thuộc thống kê, không xét tương tác giữa các đặc trưng).
    4. Chuẩn hóa 2 điểm số (MI, SHAP) về [0,1] rồi kết hợp theo trọng số:
           combined_score = w_mi * MI_norm + w_shap * SHAP_norm
    5. Chọn top-k đặc trưng theo combined_score.
    6. Đánh giá lại trên toàn bộ tập train/test thật (dùng chung khung đánh
       giá evaluate_on_full_data với GA/PSO/Filter để so sánh công bằng).

Cách dùng:
    python hybrid2_fs.py --data-dir output/CICIDS2017 --classifier dt \
        --top-percent 30 --sample-size 20000 --output-dir output/CICIDS2017/fs_hybrid2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from feature_selection import load_split, evaluate_on_full_data, CLASSIFIERS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hybrid2_fs")


def _normalize(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / rng


def compute_mi_scores(X: pd.DataFrame, y: pd.Series, sample_size: int, random_state: int) -> pd.Series:
    if sample_size and len(X) > sample_size:
        X_sub = X.sample(n=sample_size, random_state=random_state)
        y_sub = y.loc[X_sub.index]
    else:
        X_sub, y_sub = X, y
    mi = mutual_info_classif(X_sub, y_sub, random_state=random_state)
    return pd.Series(mi, index=X.columns)


def compute_shap_scores(X: pd.DataFrame, y: pd.Series, sample_size: int, random_state: int) -> pd.Series:
    try:
        import shap
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError(
            "hybrid2_fs.py cần 2 thư viện 'xgboost' và 'shap'. Cài bằng: "
            "pip install xgboost shap --break-system-packages"
        ) from e

    if sample_size and len(X) > sample_size:
        X_sub = X.sample(n=sample_size, random_state=random_state)
        y_sub = y.loc[X_sub.index]
    else:
        X_sub, y_sub = X, y

    logger.info("Đang huấn luyện XGBoost trên %d dòng để tính SHAP...", len(X_sub))
    model = XGBClassifier(
        n_estimators=200, max_depth=6, eval_metric="logloss",
        random_state=random_state, n_jobs=-1, verbosity=0,
    )
    model.fit(X_sub, y_sub)

    logger.info("Đang tính SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)

    # Với phân loại nhị phân, shap_values có thể là 1 mảng (n_samples, n_features)
    # hoặc list 2 phần tử (1 cho mỗi lớp) tùy phiên bản XGBoost/SHAP - xử lý cả 2 trường hợp.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:   # (n_classes, n_samples, n_features) - lấy lớp cuối (Attack)
        shap_values = shap_values[-1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs_shap, index=X.columns)


def select_top(scores: pd.Series, top_k: int | None, top_percent: float | None) -> list[str]:
    n = len(scores)
    if top_k:
        k = min(top_k, n)
    elif top_percent:
        k = max(1, round(n * top_percent / 100))
    else:
        raise ValueError("Cần truyền --top-k hoặc --top-percent")
    return scores.sort_values(ascending=False).head(k).index.tolist()


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid FS 2: Mutual Information + SHAP + XGBoost")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--classifier", choices=list(CLASSIFIERS), default="dt",
                         help="Classifier dùng để ĐÁNH GIÁ CUỐI (không nhất thiết phải là xgb - "
                              "XGBoost ở đây chỉ dùng nội bộ để tính SHAP)")
    parser.add_argument("--w-mi", type=float, default=0.5, help="Trọng số của điểm MI trong combined score")
    parser.add_argument("--w-shap", type=float, default=0.5, help="Trọng số của điểm SHAP trong combined score")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-percent", type=float, default=30.0)
    parser.add_argument("--sample-size", type=int, default=20000,
                         help="Số dòng lấy mẫu để tính MI/huấn luyện XGBoost cho SHAP (0 = dùng toàn bộ)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split(args.data_dir)
    y_train_encoded = pd.Series(LabelEncoder().fit_transform(y_train), index=y_train.index)
    sample_size = args.sample_size if args.sample_size > 0 else None

    t0 = time.time()
    mi_scores = compute_mi_scores(X_train, y_train_encoded, sample_size, args.random_state)
    shap_scores = compute_shap_scores(X_train, y_train_encoded, sample_size, args.random_state)
    search_time = time.time() - t0

    mi_norm = _normalize(mi_scores)
    shap_norm = _normalize(shap_scores)
    combined = args.w_mi * mi_norm + args.w_shap * shap_norm
    combined = combined.sort_values(ascending=False)

    score_table = pd.DataFrame({
        "mutual_information": mi_scores, "mutual_information_norm": mi_norm,
        "shap_importance": shap_scores, "shap_importance_norm": shap_norm,
        "combined_score": combined,
    }).sort_values("combined_score", ascending=False)
    score_table.to_csv(os.path.join(args.output_dir, "mi_shap_scores.csv"))

    selected = select_top(combined, args.top_k, args.top_percent)
    mask = np.array([1 if c in selected else 0 for c in X_train.columns], dtype=np.uint8)

    logger.info(
        "Đã chọn %d/%d đặc trưng bằng Hybrid-2 MI+SHAP (thời gian: %.2fs). Top 10: %s",
        mask.sum(), len(mask), search_time, selected[:10],
    )

    logger.info("Đang đánh giá trên toàn bộ tập train/test thật...")
    result = evaluate_on_full_data(X_train, X_test, y_train, y_test, mask, args.classifier)
    result.update({
        "method": "hybrid2_mi_shap_xgboost",
        "classifier": args.classifier,
        "w_mi": args.w_mi,
        "w_shap": args.w_shap,
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
        "KẾT QUẢ SO SÁNH | Hybrid2 (MI+SHAP): %d đặc trưng, F1=%.4f, Acc=%.4f, train=%.2fs",
        result["n_features_selected"], result["f1_macro"], result["accuracy"], result["train_time_sec"],
    )
    logger.info("Đã lưu kết quả vào: %s", args.output_dir)


if __name__ == "__main__":
    main()
