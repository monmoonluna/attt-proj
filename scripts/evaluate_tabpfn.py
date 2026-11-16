#!/usr/bin/env python3
"""Compare TabPFN with the current Random Forest on the four NIDS datasets.

The comparison is intentionally bounded and stratified: TabPFN is evaluated
on the same selected features, with at most --train-samples and --test-samples
rows per dataset. This avoids silently pretending that a large-data run is a
full-dataset benchmark.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
DATASETS = {
    "NSL-KDD": ("NSL-KDD", "fs_ga"),
    "CICIDS2017": ("CICIDS2017", "fs_hybrid1_ga_v2"),
    "CICDDoS2019": ("CICDDoS2019", "fs_hybrid1_ga"),
    "UNSW": ("UNSW", "fs_hybrid1_ga"),
}
logger = logging.getLogger("evaluate_tabpfn")


def load_selected_data(dataset: str, train_samples: int, test_samples: int, seed: int):
    directory, method = DATASETS[dataset]
    base = OUTPUT / directory
    selected = json.loads(
        (base / method / "selected_features.json").read_text(encoding="utf-8")
    )["selected_features"]
    X_train = pd.read_csv(base / "X_train.csv", usecols=selected).reindex(columns=selected)
    X_test = pd.read_csv(base / "X_test.csv", usecols=selected).reindex(columns=selected)
    y_train = pd.read_csv(base / "y_train.csv")["label"]
    y_test = pd.read_csv(base / "y_test.csv")["label"]

    if train_samples < len(X_train):
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=train_samples, stratify=y_train, random_state=seed
        )
    if test_samples < len(X_test):
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test, train_size=test_samples, stratify=y_test, random_state=seed
        )
    return X_train, X_test, y_train, y_test, selected


def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = model.predict(X_test)
    inference_seconds = time.perf_counter() - started
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "train_time_sec": train_seconds,
        "inference_time_sec": inference_seconds,
    }


def make_tabpfn_classifier(device: str):
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise SystemExit(
            "Thiếu TabPFN. Cài bằng: python -m pip install \"tabpfn>=6.0\""
        ) from exc

    kwargs = {"device": device, "n_estimators": 8}
    try:
        return TabPFNClassifier(**kwargs)
    except TypeError:
        # Compatibility with older TabPFN releases.
        kwargs.pop("n_estimators")
        kwargs["N_ensemble_configurations"] = 8
        return TabPFNClassifier(**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--output", default=str(OUTPUT / "tabpfn_evaluation.json"))
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    results = {}
    try:
        for dataset in DATASETS:
            logger.info("Đánh giá %s trên %d train / %d test samples", dataset, args.train_samples, args.test_samples)
            X_train, X_test, y_train, y_test, selected = load_selected_data(
                dataset, args.train_samples, args.test_samples, seed=42
            )
            rf = RandomForestClassifier(
                n_estimators=25, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1
            )
            tabpfn = make_tabpfn_classifier(args.device)
            results[dataset] = {
                "selected_features": selected,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "random_forest": evaluate_model(rf, X_train, X_test, y_train, y_test),
                "tabpfn": evaluate_model(tabpfn, X_train, X_test, y_train, y_test),
            }
            rf_f1 = results[dataset]["random_forest"]["f1_macro"]
            tab_f1 = results[dataset]["tabpfn"]["f1_macro"]
            logger.info("%s: RF F1=%.4f | TabPFN F1=%.4f", dataset, rf_f1, tab_f1)
    except Exception as exc:
        results["decision"] = {
            "replace_main_model": False,
            "status": "blocked",
            "reason": str(exc),
            "instruction": "Xác thực Hugging Face và chấp nhận điều khoản checkpoint TabPFN, sau đó chạy lại lệnh.",
            "device": args.device,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(results["decision"], ensure_ascii=True, indent=2))
        return

    wins = {
        dataset: values["tabpfn"]["f1_macro"] > values["random_forest"]["f1_macro"]
        for dataset, values in results.items()
    }
    results["decision"] = {
        "replace_main_model": all(wins.values()),
        "tabpfn_wins_all_datasets": wins,
        "rule": "Chỉ thay model chính nếu TabPFN có F1-macro cao hơn Random Forest ở cả 4 dataset.",
        "device": args.device,
        "train_samples_per_dataset": args.train_samples,
        "test_samples_per_dataset": args.test_samples,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["decision"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    main()
