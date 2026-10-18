"""
audit_feature_artifacts.py
===========================
Rà soát một bộ dữ liệu NIDS đã tiền xử lý (output của preprocessing.py) để
phát hiện các đặc trưng "quá mạnh" một cách đáng ngờ - khả năng là artifact
của cách thu thập dữ liệu (testbed) thay vì hành vi mạng thật, giống trường
hợp Destination Port / Inbound đã phát hiện trên CICDDoS2019.

Quy trình:
    1. Huấn luyện Decision Tree trên TOÀN BỘ đặc trưng, lấy feature_importances_.
    2. Cảnh báo nếu 1 đặc trưng chiếm > --importance-threshold (mặc định 50%)
       tổng importance, hoặc nếu top-2 đặc trưng cộng lại > 80%.
    3. Với mỗi đặc trưng bị cảnh báo, tự động chạy ablation: so sánh F1 trên
       tập test THẬT khi có/không có đặc trưng đó (dùng toàn bộ đặc trưng còn
       lại, không chỉ top-k, để đánh giá khách quan nhất).
    4. In báo cáo tổng hợp cho từng đặc trưng bị cảnh báo, kèm khuyến nghị.

Cách dùng (chạy cho từng bộ dữ liệu):
    python audit_feature_artifacts.py --data-dir output/NSL-KDD
    python audit_feature_artifacts.py --data-dir output/CICIDS2017 --sample-size 50000
    python audit_feature_artifacts.py --data-dir output/UNSW

    # Rà soát nhiều bộ cùng lúc, xuất báo cáo markdown:
    python audit_feature_artifacts.py --data-dir output/NSL-KDD output/CICIDS2017 output/UNSW \
        --report-out audit_report.md
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("audit")


def load_split(data_dir: str):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))["label"]
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))["label"]
    return X_train, X_test, y_train, y_test


def audit_dataset(
    data_dir: str,
    importance_threshold: float = 0.5,
    top2_threshold: float = 0.8,
    sample_size: int | None = None,
    random_state: int = 42,
) -> dict:
    name = os.path.basename(os.path.normpath(data_dir))
    logger.info("=" * 70)
    logger.info("RÀ SOÁT: %s", name)

    X_train, X_test, y_train, y_test = load_split(data_dir)

    if sample_size and len(X_train) > sample_size:
        X_fit = X_train.sample(n=sample_size, random_state=random_state)
        y_fit = y_train.loc[X_fit.index]
    else:
        X_fit, y_fit = X_train, y_train

    # Bước 1: huấn luyện trên toàn bộ đặc trưng, lấy importance
    clf_full = DecisionTreeClassifier(random_state=random_state)
    clf_full.fit(X_fit, y_fit)
    f1_full = f1_score(y_test, clf_full.predict(X_test), average="macro", zero_division=0)

    importances = pd.Series(clf_full.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    top1_name, top1_val = importances.index[0], importances.iloc[0]
    top2_val = importances.iloc[:2].sum()

    logger.info("Full features (%d): F1 = %.4f", X_train.shape[1], f1_full)
    logger.info("Top 5 importance:\n%s", importances.head(5).to_string())

    flagged = []
    if top1_val >= importance_threshold:
        flagged.append(top1_name)
        logger.warning(
            "⚠ CẢNH BÁO: '%s' chiếm %.1f%% importance (ngưỡng %.0f%%) - khả năng là artifact",
            top1_name, top1_val * 100, importance_threshold * 100,
        )
    elif top2_val >= top2_threshold:
        flagged.extend(importances.index[:2].tolist())
        logger.warning(
            "⚠ CẢNH BÁO: top-2 đặc trưng ('%s', '%s') chiếm %.1f%% importance (ngưỡng %.0f%%)",
            importances.index[0], importances.index[1], top2_val * 100, top2_threshold * 100,
        )
    else:
        logger.info("Không phát hiện đặc trưng nào vượt ngưỡng cảnh báo - có vẻ an toàn.")

    # Bước 2: ablation tự động cho từng đặc trưng bị cảnh báo
    ablation_results = {}
    for feat in flagged:
        remaining = [c for c in X_train.columns if c != feat]
        clf_wo = DecisionTreeClassifier(random_state=random_state)
        clf_wo.fit(X_fit[remaining], y_fit)
        f1_wo = f1_score(y_test, clf_wo.predict(X_test[remaining]), average="macro", zero_division=0)
        drop = f1_full - f1_wo
        ablation_results[feat] = {"f1_with": f1_full, "f1_without": f1_wo, "drop": drop}

        verdict = (
            "artifact rõ ràng, nên loại bỏ" if drop > 0.10 else
            "cần xem xét thêm" if drop > 0.03 else
            "không đáng ngại - đặc trưng mạnh hợp lý"
        )
        logger.info(
            "  Ablation '%s': F1 có=%.4f, F1 không=%.4f, chênh lệch=%+.4f -> %s",
            feat, f1_full, f1_wo, drop, verdict,
        )

    return {
        "dataset": name,
        "n_features": X_train.shape[1],
        "f1_full": f1_full,
        "top_importances": importances.head(5).to_dict(),
        "flagged_features": flagged,
        "ablation_results": ablation_results,
    }


def write_markdown_report(results: list[dict], out_path: str):
    lines = ["# Báo cáo rà soát artifact đặc trưng\n"]
    for r in results:
        lines.append(f"## {r['dataset']}\n")
        lines.append(f"- Số đặc trưng: {r['n_features']}, F1 (full features) = {r['f1_full']:.4f}")
        lines.append("- Top 5 importance:")
        for feat, val in r["top_importances"].items():
            lines.append(f"    - `{feat}`: {val:.4f}")
        if not r["flagged_features"]:
            lines.append("- ✅ Không phát hiện artifact đáng ngờ.\n")
            continue
        lines.append("- ⚠ Đặc trưng bị cảnh báo và kết quả ablation:")
        for feat, res in r["ablation_results"].items():
            lines.append(
                f"    - `{feat}`: F1 có={res['f1_with']:.4f}, không có={res['f1_without']:.4f}, "
                f"chênh lệch={res['drop']:+.4f}"
            )
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Đã lưu báo cáo: %s", out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Rà soát artifact đặc trưng cho các bộ dữ liệu NIDS đã tiền xử lý")
    parser.add_argument("--data-dir", nargs="+", required=True, help="Một hoặc nhiều thư mục output của preprocessing.py")
    parser.add_argument("--importance-threshold", type=float, default=0.5)
    parser.add_argument("--top2-threshold", type=float, default=0.8)
    parser.add_argument("--sample-size", type=int, default=0, help="0 = dùng toàn bộ tập train")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--report-out", default=None, help="Đường dẫn file markdown để xuất báo cáo tổng hợp")
    return parser.parse_args()


def main():
    args = parse_args()
    results = []
    for data_dir in args.data_dir:
        r = audit_dataset(
            data_dir,
            importance_threshold=args.importance_threshold,
            top2_threshold=args.top2_threshold,
            sample_size=args.sample_size if args.sample_size > 0 else None,
            random_state=args.random_state,
        )
        results.append(r)

    if args.report_out:
        write_markdown_report(results, args.report_out)


if __name__ == "__main__":
    main()
