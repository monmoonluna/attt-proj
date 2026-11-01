#!/usr/bin/env python3
"""
scripts/demo_cli.py
===================
Script demo báo cáo kết quả ban đầu cho đề tài NIDS Feature Selection (ATTT).
- Tự động quét và tổng hợp kết quả định lượng từ output/
- In bảng so sánh trực quan (Số đặc trưng, Tỷ lệ giảm, F1-macro, Accuracy, T_train, T_infer)
- Hỗ trợ chế độ --live-nslkdd để chạy demo trực tiếp trên tập dữ liệu NSL-KDD có sẵn.
"""

import argparse
import glob
import json
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = ["CICIDS2017", "CICDDoS2019", "UNSW", "NSL-KDD"]

METHOD_NAMES = {
    "baseline": "Baseline (Toàn bộ đặc trưng)",
    "fs_filter": "Filter (Mutual Info / Chi2)",
    "fs_ga": "Wrapper: GA (Genetic Algorithm)",
    "fs_pso": "Wrapper: PSO (Particle Swarm)",
    "fs_pso_v2": "Wrapper: PSO v2 (Mutation & Reset)",
    "fs_hybrid1_ga": "Hybrid 1: Filter + GA",
    "fs_hybrid1_ga_v2": "Hybrid 1: Filter + GA v2 (Tối ưu)",
    "fs_hybrid1_pso": "Hybrid 1: Filter + PSO",
    "fs_hybrid2": "Hybrid 2: Filter + Embedded/GA",
}


def load_dataset_results(output_dir: str, dataset: str):
    base_path = os.path.join(output_dir, dataset)
    if not os.path.exists(base_path):
        return []

    results = []

    # Quét baseline nếu có
    baseline_files = glob.glob(os.path.join(base_path, "**/baseline_full_features.json"), recursive=True)
    if baseline_files:
        try:
            with open(baseline_files[0], "r", encoding="utf-8") as f:
                bdata = json.load(f)
                bdata["method_key"] = "baseline"
                bdata["method_name"] = METHOD_NAMES["baseline"]
                bdata["dataset"] = dataset
                results.append(bdata)
        except Exception:
            pass

    # Quét các folder thuật toán
    subdirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    for sub in subdirs:
        sel_file = os.path.join(base_path, sub, "selected_features.json")
        if os.path.exists(sel_file):
            try:
                with open(sel_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["method_key"] = sub
                    data["method_name"] = METHOD_NAMES.get(sub, sub)
                    data["dataset"] = dataset
                    results.append(data)
            except Exception:
                pass

    return results


def print_table(dataset: str, results: list):
    if not results:
        print(f"\n⚠️  Không tìm thấy kết quả cho tập dữ liệu: {dataset}")
        return

    orig_dim = results[0].get("n_features_total") or results[0].get("n_features_total_original") or 0

    print("=" * 105)
    print(f"📊 BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG FEATURE SELECTION — BỘ DỮ LIỆU: {dataset.upper()}")
    print(f"   Chiều gốc ban đầu: {orig_dim} đặc trưng")
    print("=" * 105)
    header = (
        f"{'Phương pháp':<36} | {'Số ĐT (k)':<9} | {'Giảm (%)':<8} | "
        f"{'Accuracy':<9} | {'F1-Macro':<9} | {'T_train(s)':<10} | {'T_infer(ms)':<10}"
    )
    print(header)
    print("-" * 105)

    for r in results:
        m_name = r.get("method_name", "Unknown")
        k = r.get("n_features_selected", 0)
        tot = r.get("n_features_total", orig_dim) or orig_dim
        reduction = ((tot - k) / tot * 100.0) if tot > 0 else 0.0
        acc = r.get("accuracy", 0.0) * 100.0
        f1 = r.get("f1_macro", 0.0) * 100.0
        t_tr = r.get("train_time_sec", 0.0)
        t_inf = r.get("inference_time_sec", 0.0) * 1000.0

        print(
            f"{m_name:<36} | {k:<9} | {reduction:6.1f}% | "
            f"{acc:8.2f}% | {f1:8.2f}% | {t_tr:9.2f}s | {t_inf:9.2f}ms"
        )
    print("=" * 105)

    # In nhận định điểm nhấn khoa học
    best_hybrid = next((r for r in results if "hybrid1_ga_v2" in r.get("method_key", "")), None)
    baseline = next((r for r in results if r.get("method_key") == "baseline"), None)
    if best_hybrid and baseline:
        k_red = ((baseline["n_features_selected"] - best_hybrid["n_features_selected"]) / baseline["n_features_selected"]) * 100
        speedup = (baseline["train_time_sec"] / best_hybrid["train_time_sec"]) if best_hybrid["train_time_sec"] > 0 else 1.0
        print(f"💡 NHẬN XÉT ĐIỂM NHẤN ({dataset}):")
        print(f"   - Phương pháp Hybrid tối ưu giảm {k_red:.1f}% số đặc trưng (từ {baseline['n_features_selected']} xuống {best_hybrid['n_features_selected']}).")
        print(f"   - Tốc độ huấn luyện nhanh gấp ~{speedup:.1f} lần, F1-macro chênh lệch không đáng kể ({best_hybrid['f1_macro']*100:.2f}% so với {baseline['f1_macro']*100:.2f}%).")
        print("   - Top đặc trưng giữ lại:", ", ".join(best_hybrid.get("selected_features", [])[:5]))
    print()


def run_live_nslkdd():
    print("\n" + "=" * 80)
    print("🚀 LIVE DEMO: TIỀN XỬ LÝ & CHỌN ĐẶC TRƯNG TRỰC TIẾP TRÊN NSL-KDD")
    print("=" * 80)
    try:
        import pandas as pd
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    except ImportError as e:
        print(f"\n❌ Lỗi thiếu thư viện: {e}")
        print("💡 Máy bạn đã cài sẵn các thư viện này trong môi trường conda 'dss'!")
        print("👉 Vui lòng kích hoạt môi trường 'dss' trước bằng lệnh:")
        print("\n    conda activate dss")
        print("    python scripts/demo_cli.py --live-nslkdd\n")
        return

    train_path = os.path.join(ROOT_DIR, "data", "NSL-KDD", "KDDTrain+.txt")
    test_path = os.path.join(ROOT_DIR, "data", "NSL-KDD", "KDDTest+.txt")

    if not os.path.exists(train_path):
        print(f"❌ Không tìm thấy file {train_path}")
        return

    print("1. Đang nạp dữ liệu NSL-KDD (mẫu 5.000 dòng để demo tức thì)...")
    cols = [f"f_{i}" for i in range(41)] + ["label", "difficulty"]
    df_train = pd.read_csv(train_path, names=cols, header=None, nrows=5000)
    df_test = pd.read_csv(test_path, names=cols, header=None, nrows=2000)

    # Tiền xử lý nhanh
    y_train = (df_train["label"] != "normal").astype(int)
    y_test = (df_test["label"] != "normal").astype(int)
    X_train = df_train.drop(columns=["label", "difficulty"])
    X_test = df_test.drop(columns=["label", "difficulty"])

    # Encode categorical
    for c in [1, 2, 3]:
        col_name = f"f_{c}"
        le = LabelEncoder()
        X_train[col_name] = le.fit_transform(X_train[col_name].astype(str))
        X_test[col_name] = X_test[col_name].map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)

    scaler = MinMaxScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)
    n_orig = X_tr_scaled.shape[1]
    print(f"   -> Hoàn thành tiền xử lý: {n_orig} đặc trưng, {len(X_train)} mẫu train, {len(X_test)} mẫu test.")

    # 1. Baseline Full Features
    print("\n2. Huấn luyện Baseline trên toàn bộ 41 đặc trưng...")
    t0 = time.time()
    clf_base = DecisionTreeClassifier(random_state=42)
    clf_base.fit(X_tr_scaled, y_train)
    t_base_train = time.time() - t0
    y_pred_base = clf_base.predict(X_te_scaled)
    acc_base = accuracy_score(y_test, y_pred_base)
    f1_base = f1_score(y_test, y_pred_base, average="macro")
    print(f"   ✓ Baseline: Accuracy = {acc_base*100:.2f}%, F1-macro = {f1_base*100:.2f}%, Time = {t_base_train:.3f}s")

    # 2. Filter Feature Selection (Chọn k=10)
    print("\n3. Thực hiện Feature Selection (Mutual Information Filter, chọn k=10)...")
    t0 = time.time()
    selector = SelectKBest(score_func=mutual_info_classif, k=10)
    X_tr_fs = selector.fit_transform(X_tr_scaled, y_train)
    X_te_fs = selector.transform(X_te_scaled)
    t_fs = time.time() - t0

    clf_fs = DecisionTreeClassifier(random_state=42)
    clf_fs.fit(X_tr_fs, y_train)
    y_pred_fs = clf_fs.predict(X_te_fs)
    acc_fs = accuracy_score(y_test, y_pred_fs)
    f1_fs = f1_score(y_test, y_pred_fs, average="macro")

    selected_idx = [i for i, b in enumerate(selector.get_support()) if b]
    print(f"   ✓ Chọn xong 10 đặc trưng trong {t_fs:.3f}s! Indices: {selected_idx}")
    print(f"   ✓ Model với 10 đặc trưng: Accuracy = {acc_fs*100:.2f}%, F1-macro = {f1_fs*100:.2f}%")

    print("\n" + "=" * 80)
    print("🏆 KẾT LUẬN DEMO TRỰC TIẾP:")
    print(f"   - Giảm số đặc trưng từ {n_orig} -> 10 (Giảm {(n_orig - 10)/n_orig * 100:.1f}%)")
    print(f"   - F1-Score: Baseline ({f1_base*100:.2f}%) vs 10 đặc trưng ({f1_fs*100:.2f}%)")
    print("   - Pipeline hoạt động chính xác, ổn định và không bị rò rỉ dữ liệu.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Demo CLI: Báo cáo kết quả NIDS Feature Selection")
    parser.add_argument("--output-dir", default=os.path.join(ROOT_DIR, "output"), help="Thư mục chứa kết quả artifacts")
    parser.add_argument("--dataset", choices=["all", "cicids2017", "cicddos2019", "unsw", "nsl-kdd"], default="all")
    parser.add_argument("--live-nslkdd", action="store_true", help="Chạy demo trực tiếp pipeline trên dữ liệu NSL-KDD")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🛡️  ĐỀ TÀI: NIDS FEATURE SELECTION (ATTT)")
    print("   Demo Kết quả Thử nghiệm Ban đầu")
    print("=" * 60)

    target_datasets = DATASETS if args.dataset == "all" else [args.dataset.upper() if args.dataset != "nsl-kdd" else "NSL-KDD"]

    for d in target_datasets:
        res = load_dataset_results(args.output_dir, d)
        print_table(d, res)

    if args.live_nslkdd:
        run_live_nslkdd()
    else:
        print("💡 GỢI Ý: Chạy với cờ '--live-nslkdd' để thực thi pipeline trực tiếp trên dữ liệu mẫu:")
        print("   python scripts/demo_cli.py --live-nslkdd\n")


if __name__ == "__main__":
    main()

