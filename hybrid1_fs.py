"""
hybrid1_fs.py
=============
Hybrid Feature Selection 1 - Filter (Mutual Information) + Metaheuristic
(GA hoặc PSO) với hàm fitness cải tiến, đúng Kịch bản 4 trong đề cương
(mục 6.6) - đóng góp chính thứ nhất của đề tài.

Khác với chạy GA/PSO thuần túy (feature_selection.py) hay Filter/Hybrid 2
thuần túy (filter_fs.py/hybrid2_fs.py) chạy ĐỘC LẬP, Hybrid 1 GHÉP 2 bước
tuần tự trong cùng một pipeline:

    Bước 1 (Filter - MI): tính Mutual Information cho toàn bộ đặc trưng,
        giữ lại top --pre-filter-percent% đặc trưng có MI cao nhất - giảm
        KHÔNG GIAN TÌM KIẾM cho bước 2 (thay vì GA/PSO phải dò trên toàn bộ
        n đặc trưng gốc, giờ chỉ cần dò trên tập con đã được Filter rút gọn
        sẵn, giúp hội tụ nhanh hơn và giảm nguy cơ hội tụ sớm vào vùng toàn
        đặc trưng nhiễu).

    Bước 2 (Metaheuristic - GA/PSO cải tiến): chạy GA hoặc PSO (đã khắc phục
        hội tụ sớm - xem feature_selection.py) trên tập con đã lọc, với hàm
        fitness CẢI TIẾN có thêm trọng số F1 của lớp thiểu số (--w-minority),
        không chỉ tối ưu F1-macro + số lượng đặc trưng như bản GA/PSO gốc.

Cách dùng:
    python hybrid1_fs.py --data-dir output/CICIDS2017 --method ga \
        --pre-filter-percent 50 --w-minority 0.2 --classifier dt \
        --pop-size 30 --generations 30 --output-dir output/CICIDS2017/fs_hybrid1

    python hybrid1_fs.py --data-dir output/CICIDS2017 --method pso \
        --pre-filter-percent 50 --w-minority 0.2 --classifier dt \
        --swarm-size 30 --iterations 30 --output-dir output/CICIDS2017/fs_hybrid1_pso
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd

from feature_selection import (
    load_split, evaluate_on_full_data, plot_convergence, CLASSIFIERS,
    GAFeatureSelector, PSOFeatureSelector,
)
from filter_fs import compute_scores, select_top

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hybrid1_fs")


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid FS 1: Filter (MI) + Metaheuristic (GA/PSO) cải tiến")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--method", choices=["ga", "pso"], required=True)
    parser.add_argument("--classifier", choices=list(CLASSIFIERS), default="dt",
                         help="Classifier dùng để TÌM KIẾM (tính fitness hàng trăm lần) - nên giữ 'dt' cho nhanh.")
    parser.add_argument("--eval-classifier", default=None,
                         help="Classifier dùng để ĐÁNH GIÁ CUỐI - mặc định dùng lại --classifier. "
                              "Dùng để so sánh RF/XGBoost mà không làm chậm GA/PSO.")

    # Bước 1: Filter (MI) pre-filter
    parser.add_argument("--filter-method", choices=["mi", "chi2"], default="mi")
    parser.add_argument("--pre-filter-percent", type=float, default=50.0,
                         help="%% đặc trưng giữ lại sau bước Filter, trước khi đưa vào GA/PSO (mặc định 50%%)")
    parser.add_argument("--filter-sample-size", type=int, default=20000,
                         help="Số dòng lấy mẫu để tính điểm Filter cho nhanh (0 = dùng toàn bộ)")

    # Hàm fitness cải tiến (dùng chung cho cả GA và PSO ở bước 2)
    parser.add_argument("--w-acc", type=float, default=0.8)
    parser.add_argument("--w-feat", type=float, default=0.05,
                         help="Trọng số phạt số lượng đặc trưng. Mặc định 0.05 (thấp hơn GA/PSO thuần túy "
                              "vốn dùng 0.1) vì Hybrid 1 chạy trên không gian đã bị Filter rút gọn sẵn (~50%%) "
                              "- giữ nguyên 0.1 sẽ khiến tỉ lệ phạt/đặc trưng hiệu dụng tăng gấp đôi, đẩy GA "
                              "chọn quá ít đặc trưng (đã quan sát được: 2-4 đặc trưng, mất thông tin cần thiết)")
    parser.add_argument("--w-minority", type=float, default=0.1,
                         help="Trọng số F1 của lớp thiểu số trong hàm fitness - phần CẢI TIẾN so với GA/PSO gốc")
    parser.add_argument("--sample-size", type=int, default=5000,
                         help="Số dòng lấy mẫu để tính fitness trong vòng lặp GA/PSO (0 = dùng toàn bộ)")
    parser.add_argument("--min-features", type=int, default=1)

    # GA params
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--cx-pb", type=float, default=0.8)
    parser.add_argument("--mut-pb", type=float, default=None)

    # PSO params
    parser.add_argument("--swarm-size", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--w-inertia", type=float, default=0.7)
    parser.add_argument("--c1", type=float, default=1.5)
    parser.add_argument("--c2", type=float, default=1.5)
    parser.add_argument("--mutation-rate", type=float, default=0.15)
    parser.add_argument("--stagnation-limit", type=int, default=5)
    parser.add_argument("--reinit-fraction", type=float, default=0.3)

    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split(args.data_dir)
    n_total = X_train.shape[1]

    # ------------------------------------------------------------------ #
    # Bước 1: Filter (MI) - rút gọn không gian tìm kiếm trước khi chạy GA/PSO
    # ------------------------------------------------------------------ #
    filter_sample = args.filter_sample_size if args.filter_sample_size > 0 else None
    t0 = time.time()
    scores = compute_scores(X_train, y_train, args.filter_method, filter_sample, args.random_state)
    pre_filtered_cols = select_top(scores, top_k=None, top_percent=args.pre_filter_percent)
    filter_time = time.time() - t0

    logger.info(
        "Bước 1 (Filter-%s): giữ lại %d/%d đặc trưng (%.0f%%) trước khi đưa vào %s. Thời gian: %.2fs",
        args.filter_method.upper(), len(pre_filtered_cols), n_total, args.pre_filter_percent,
        args.method.upper(), filter_time,
    )

    X_train_f = X_train[pre_filtered_cols]
    X_test_f = X_test[pre_filtered_cols]

    # ------------------------------------------------------------------ #
    # Bước 2: Metaheuristic (GA/PSO) với hàm fitness cải tiến, chạy TRÊN
    # tập đặc trưng đã được Filter rút gọn ở bước 1
    # ------------------------------------------------------------------ #
    common_kwargs = dict(
        X_train=X_train_f, y_train=y_train, classifier=args.classifier,
        w_acc=args.w_acc, w_feat=args.w_feat, w_minority=args.w_minority,
        sample_size=args.sample_size if args.sample_size > 0 else None,
        min_features=args.min_features, random_state=args.random_state,
    )

    t0 = time.time()
    if args.method == "ga":
        selector = GAFeatureSelector(**common_kwargs)
        mask, best_fit, history = selector.run(
            pop_size=args.pop_size, generations=args.generations,
            cx_pb=args.cx_pb, mut_pb=args.mut_pb,
        )
        title = "Hybrid 1 (Filter-MI + GA) - Convergence curve"
    else:
        selector = PSOFeatureSelector(**common_kwargs)
        mask, best_fit, history = selector.run(
            swarm_size=args.swarm_size, iterations=args.iterations,
            w=args.w_inertia, c1=args.c1, c2=args.c2,
            mutation_rate=args.mutation_rate,
            stagnation_limit=args.stagnation_limit,
            reinit_fraction=args.reinit_fraction,
        )
        title = "Hybrid 1 (Filter-MI + PSO) - Convergence curve"
    search_time = time.time() - t0

    selected = selector.selected_feature_names(mask)
    logger.info(
        "Bước 2 (%s cải tiến) hoàn tất: %d/%d đặc trưng (trong tập đã lọc) | %d/%d so với TOÀN BỘ "
        "đặc trưng gốc | best fitness = %.5f | thời gian = %.1fs | số lần đánh giá fitness = %d",
        args.method.upper(), int(mask.sum()), len(pre_filtered_cols),
        int(mask.sum()), n_total, best_fit, search_time, selector.n_evaluations,
    )

    # ------------------------------------------------------------------ #
    # Đánh giá cuối trên toàn bộ tập train/test thật (dùng tập đã lọc ở
    # bước 1 làm "không gian đặc trưng", chỉ giữ các cột GA/PSO chọn ở bước 2)
    # ------------------------------------------------------------------ #
    eval_classifier = args.eval_classifier if args.eval_classifier else args.classifier
    logger.info("Đang đánh giá trên toàn bộ tập train/test thật (classifier: %s)...", eval_classifier)
    result = evaluate_on_full_data(X_train_f, X_test_f, y_train, y_test, mask, eval_classifier)
    result.update({
        "method": f"hybrid1_filter{args.filter_method}_{args.method}",
        "search_classifier": args.classifier,
        "classifier": eval_classifier,
        "n_features_total_original": n_total,
        "n_features_after_prefilter": len(pre_filtered_cols),
        "pre_filter_features": pre_filtered_cols,
        "pre_filter_percent": args.pre_filter_percent,
        "w_acc": args.w_acc, "w_feat": args.w_feat, "w_minority": args.w_minority,
        "filter_time_sec": filter_time,
        "search_time_sec": search_time,
        "total_time_sec": filter_time + search_time,
    })

    # So sánh với baseline dùng TOÀN BỘ đặc trưng gốc (không qua Filter, không qua GA/PSO)
    full_mask = np.ones(n_total, dtype=np.uint8)
    logger.info("Đang đánh giá baseline (không Feature Selection) để so sánh...")
    baseline = evaluate_on_full_data(X_train, X_test, y_train, y_test, full_mask, eval_classifier)

    with open(os.path.join(args.output_dir, "selected_features.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "baseline_full_features.json"), "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)

    pd.DataFrame({"generation": range(len(history)), "best_fitness": history}).to_csv(
        os.path.join(args.output_dir, "convergence.csv"), index=False
    )
    try:
        plot_convergence(history, title, os.path.join(args.output_dir, "convergence.png"))
    except Exception as e:
        logger.warning("Không vẽ được convergence curve: %s", e)

    logger.info("=" * 70)
    logger.info(
        "KẾT QUẢ SO SÁNH | Full features: %d đặc trưng, F1=%.4f, Acc=%.4f, train=%.2fs",
        baseline["n_features_total"], baseline["f1_macro"], baseline["accuracy"], baseline["train_time_sec"],
    )
    logger.info(
        "KẾT QUẢ SO SÁNH | Hybrid1 (Filter-%s+%s): %d/%d đặc trưng, F1=%.4f, Acc=%.4f, train=%.2fs, tổng thời gian=%.1fs",
        args.filter_method.upper(), args.method.upper(), result["n_features_selected"], n_total,
        result["f1_macro"], result["accuracy"], result["train_time_sec"], filter_time + search_time,
    )
    logger.info("Đã lưu toàn bộ kết quả vào: %s", args.output_dir)


if __name__ == "__main__":
    main()
