"""
eval_selected.py
================
Đánh giá lại các tập đặc trưng ĐÃ CHỌN (đọc từ selected_features.json do
filter_fs.py / feature_selection.py / hybrid1_fs.py / hybrid2_fs.py ghi ra)
bằng MỘT CLASSIFIER KHÁC, mà KHÔNG chạy lại bước tìm kiếm.

Lợi ích:
    - Không phải chạy lại GA/PSO (có thể mất hàng chục phút).
    - Các classifier được so sánh trên ĐÚNG CÙNG tập đặc trưng.

Cách dùng:
    python eval_selected.py --data-dir output/CICDDoS2019 --classifier xgb \
        --selected output/CICDDoS2019/fs_filter_rf output/CICDDoS2019/fs_ga_rf \
                   output/CICDDoS2019/fs_hybrid1_pso_rf output/CICDDoS2019/fs_hybrid2_rf

--selected nhận thư mục (chứa selected_features.json) hoặc đường dẫn trực tiếp tới file json.
Kết quả được in ra và lưu vào <data-dir>/eval_<classifier>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

from feature_selection import CLASSIFIERS, evaluate_on_full_data, load_split

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_selected")


def _json_path(p: str) -> str:
    return p if p.endswith(".json") else os.path.join(p, "selected_features.json")


def main():
    parser = argparse.ArgumentParser(description="Đánh giá lại các tập đặc trưng đã chọn bằng classifier khác")
    parser.add_argument("--data-dir", required=True, help="Thư mục dữ liệu đã tiền xử lý (X_train.csv, ...)")
    parser.add_argument("--classifier", choices=list(CLASSIFIERS), default="xgb")
    parser.add_argument("--selected", nargs="+", required=True,
                         help="Các thư mục kết quả (hoặc file selected_features.json) cần đánh giá lại")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_split(args.data_dir)
    columns = X_train.columns.tolist()

    results = []

    logger.info("Đánh giá baseline Full features (%s)...", args.classifier)
    full = evaluate_on_full_data(X_train, X_test, y_train, y_test,
                                  np.ones(len(columns), dtype=np.uint8), args.classifier)
    results.append(("Full features", full))

    for item in args.selected:
        path = _json_path(item)
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
        names = info["selected_features"]
        missing = [n for n in names if n not in columns]
        if missing:
            raise SystemExit(f"{path}: có đặc trưng không tồn tại trong dữ liệu {args.data_dir}: {missing}. "
                             f"Kiểm tra lại --data-dir có đúng bộ dữ liệu không.")
        mask = np.array([1 if c in names else 0 for c in columns], dtype=np.uint8)
        label = f"{os.path.basename(os.path.dirname(path))} [{info.get('method', '?')}]"
        logger.info("Đánh giá %s (%d đặc trưng)...", label, len(names))
        results.append((label, evaluate_on_full_data(X_train, X_test, y_train, y_test, mask, args.classifier)))

    print("\n" + "=" * 86)
    print(f"Classifier đánh giá: {args.classifier} | dữ liệu: {args.data_dir}")
    print(f"{'Phương pháp':<44}{'Đặc trưng':>10}{'F1-macro':>10}{'Acc':>9}{'Train(s)':>10}")
    for label, r in results:
        print(f"{label:<44}{r['n_features_selected']:>5}/{r['n_features_total']:<4}"
              f"{r['f1_macro']:>10.4f}{r['accuracy']:>9.4f}{r['train_time_sec']:>10.2f}")
    print("=" * 86)

    out_path = os.path.join(args.data_dir, f"eval_{args.classifier}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({label: {k: v for k, v in r.items() if k != "selected_features"} for label, r in results},
                  f, ensure_ascii=False, indent=2)
    logger.info("Đã lưu: %s", out_path)


if __name__ == "__main__":
    main()
