"""
ablation_full.py
================
Ablation trên TOÀN BỘ đặc trưng (Full features): bỏ đặc trưng A, bỏ đặc trưng B,
và bỏ cả hai, rồi đánh giá bằng nhiều classifier trên tập test thật.

Mục đích (CICDDoS2019): đo xem hiệu năng của CẢ bộ dữ liệu phụ thuộc bao nhiêu vào
Destination Port và URG Flag Count - hai đặc trưng liên tục xuất hiện trong các tập
cho F1 cao.

XGBoost nhạy với thứ tự cột đầu vào (xem mục 12.6.3 đề cương), nên với XGBoost script
báo cáo cả:
    - kết quả với thứ tự cột GỐC của bộ dữ liệu (cùng quy ước các bảng ở 12.6.3), và
    - trung bình / độ lệch chuẩn / khoảng giá trị qua --xgb-perms thứ tự cột ngẫu nhiên.
DT và RF chỉ chạy 1 lần với thứ tự cột gốc.

Cách dùng:
    python ablation_full.py --data-dir output/CICDDoS2019 \
        --feature-a "Destination Port" --feature-b "URG Flag Count" --xgb-perms 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from feature_selection import CLASSIFIERS, load_split

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation_full")


def fit_eval(clf_name, cols, X_train, X_test, y_train, y_test):
    clf = CLASSIFIERS[clf_name]()
    clf.fit(X_train[cols], y_train)
    pred = clf.predict(X_test[cols])
    return (f1_score(y_test, pred, average="macro", zero_division=0),
            accuracy_score(y_test, pred))


def main():
    parser = argparse.ArgumentParser(description="Ablation trên Full features")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--feature-a", required=True)
    parser.add_argument("--feature-b", required=True)
    parser.add_argument("--classifiers", nargs="+", default=["dt", "rf", "xgb"], choices=list(CLASSIFIERS))
    parser.add_argument("--xgb-perms", type=int, default=5,
                         help="Số thứ tự cột ngẫu nhiên cho XGBoost (0 = chỉ chạy thứ tự gốc)")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_split(args.data_dir)
    columns = X_train.columns.tolist()
    for f in (args.feature_a, args.feature_b):
        if f not in columns:
            raise SystemExit(f"Không có đặc trưng '{f}' trong {args.data_dir}. Kiểm tra --data-dir và tên đặc trưng.")

    variants = {
        "Full (đủ đặc trưng)": columns,
        f"Bỏ {args.feature_a}": [c for c in columns if c != args.feature_a],
        f"Bỏ {args.feature_b}": [c for c in columns if c != args.feature_b],
        f"Bỏ cả hai": [c for c in columns if c not in (args.feature_a, args.feature_b)],
    }

    results = {}
    for vname, cols in variants.items():
        results[vname] = {"n_features": len(cols)}
        for cname in args.classifiers:
            t0 = time.time()
            f1, acc = fit_eval(cname, cols, X_train, X_test, y_train, y_test)
            entry = {"f1": f1, "acc": acc}
            if cname == "xgb" and args.xgb_perms > 0:
                perm_f1 = []
                for seed in range(args.xgb_perms):
                    perm_cols = list(np.random.default_rng(seed).permutation(cols))
                    perm_f1.append(fit_eval(cname, perm_cols, X_train, X_test, y_train, y_test)[0])
                entry.update({"perm_mean": float(np.mean(perm_f1)), "perm_std": float(np.std(perm_f1)),
                              "perm_min": float(np.min(perm_f1)), "perm_max": float(np.max(perm_f1)),
                              "perm_values": [float(v) for v in perm_f1]})
            results[vname][cname] = entry
            logger.info("%-28s %-4s F1=%.4f (%.0fs)", vname, cname, f1, time.time() - t0)

    print("\n" + "=" * 100)
    print(f"Ablation Full features | dữ liệu: {args.data_dir} | A='{args.feature_a}' B='{args.feature_b}'")
    head = f"{'Biến thể':<34}{'Số ĐT':>6}"
    for c in args.classifiers:
        head += f"{c.upper() + ' F1':>12}"
    if "xgb" in args.classifiers and args.xgb_perms > 0:
        head += f"{'XGB TB±SD (min-max) qua ' + str(args.xgb_perms) + ' thứ tự cột':>56}"
    print(head)
    for vname, r in results.items():
        line = f"{vname:<34}{r['n_features']:>6}"
        for c in args.classifiers:
            line += f"{r[c]['f1']:>12.4f}"
        if "xgb" in args.classifiers and args.xgb_perms > 0:
            x = r["xgb"]
            line += f"{x['perm_mean']:>14.4f} ± {x['perm_std']:.4f} ({x['perm_min']:.4f}-{x['perm_max']:.4f})"
        print(line)
    print("=" * 100)

    out = os.path.join(args.data_dir, "ablation_full.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"feature_a": args.feature_a, "feature_b": args.feature_b, "results": results},
                  f, ensure_ascii=False, indent=2)
    logger.info("Đã lưu: %s", out)


if __name__ == "__main__":
    main()
