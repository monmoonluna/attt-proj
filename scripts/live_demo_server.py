#!/usr/bin/env python3
"""Local inference server for reports/live_demo.html.

The server deliberately trains from the repository's preprocessed train split
and uses the persisted feature-selection results. The browser never loads
pickle files or executes a model itself.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
REPORTS = ROOT / "reports"

DATASETS = {
    "NSL-KDD": {
        "dir": "NSL-KDD",
        "methods": {
            "baseline": None,
            "ga": "fs_ga",
            "pso": "fs_pso",
        },
    },
    "CICIDS2017": {
        "dir": "CICIDS2017",
        "methods": {
            "baseline": None,
            "filter": "fs_filter",
            "ga": "fs_ga",
            "pso": "fs_pso",
            "hybrid": "fs_hybrid1_ga_v2",
            "hybrid2": "fs_hybrid2",
        },
    },
    "CICDDoS2019": {
        "dir": "CICDDoS2019",
        "methods": {
            "baseline": None,
            "filter": "fs_filter",
            "ga": "fs_ga",
            "pso": "fs_pso",
            "hybrid": "fs_hybrid1_ga",
            "hybrid2": "fs_hybrid2",
        },
    },
    "UNSW": {
        "dir": "UNSW",
        "methods": {
            "baseline": None,
            "filter": "fs_filter",
            "ga": "fs_ga",
            "pso": "fs_pso",
            "hybrid": "fs_hybrid1_ga",
            "hybrid2": "fs_hybrid2",
        },
    },
}

_models: dict[tuple[str, str], tuple[RandomForestClassifier, list[str], dict]] = {}
_samples: dict[str, list[dict]] = {}


def _dataset(key: str) -> dict:
    try:
        return DATASETS[key]
    except KeyError as exc:
        raise ValueError(f"Dataset không được hỗ trợ: {key}") from exc


def _feature_names(dataset: str) -> list[str]:
    path = OUTPUT / _dataset(dataset)["dir"] / "feature_names.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_features(dataset: str, method: str) -> list[str]:
    config = _dataset(dataset)
    method_dir = config["methods"].get(method)
    if method_dir is None:
        return _feature_names(dataset)
    path = OUTPUT / config["dir"] / method_dir / "selected_features.json"
    return json.loads(path.read_text(encoding="utf-8"))["selected_features"]


def _load_model(dataset: str, method: str):
    cache_key = (dataset, method)
    if cache_key in _models:
        return _models[cache_key]

    config = _dataset(dataset)
    selected = _selected_features(dataset, method)
    data_dir = OUTPUT / config["dir"]
    X_train = pd.read_csv(data_dir / "X_train.csv", usecols=selected)
    X_train = X_train.reindex(columns=selected)
    y_train = pd.read_csv(data_dir / "y_train.csv")["label"]
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=25,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    metrics = {}
    method_dir = config["methods"].get(method)
    if method_dir:
        result_path = data_dir / method_dir / "selected_features.json"
        metrics = json.loads(result_path.read_text(encoding="utf-8"))
    _models[cache_key] = (model, selected, metrics)
    return _models[cache_key]


def _load_samples(dataset: str) -> list[dict]:
    if dataset in _samples:
        return _samples[dataset]

    config = _dataset(dataset)
    data_dir = OUTPUT / config["dir"]
    feature_names = _feature_names(dataset)
    X_test = pd.read_csv(data_dir / "X_test.csv", usecols=feature_names)
    y_test = pd.read_csv(data_dir / "y_test.csv")["label"]
    frame = X_test.copy()
    frame["_label"] = y_test.to_numpy()
    result = []
    for label, group in frame.groupby("_label", sort=True):
        for index, row in group.head(2).iterrows():
            result.append({
                "id": f"{dataset}-{index}",
                "name": f"Mẫu thật #{index} ({label})",
                "type": f"Test split: {label}",
                "actualClass": str(label),
                "values": {
                    name: (float(row[name]) if np.isfinite(row[name]) else 0.0)
                    for name in feature_names
                },
            })
    _samples[dataset] = result
    return result


def _dataset_payload(dataset: str) -> dict:
    config = _dataset(dataset)
    samples = _load_samples(dataset)
    return {
        "samples": samples,
        "featureNames": _feature_names(dataset),
        "methods": {
            method: _selected_features(dataset, method)
            for method in config["methods"]
        },
    }


def _predict(payload: dict) -> dict:
    dataset = payload["dataset"]
    method = payload["method"]
    model, selected, metrics = _load_model(dataset, method)
    values = payload.get("values", {})
    vector = pd.DataFrame([{name: float(values.get(name, 0.0)) for name in selected}])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    class_index = list(model.classes_).index(prediction)
    raw_confidence = float(probabilities[class_index])
    # Tree ensembles can return exactly 1.0 when every tree agrees. Keep the
    # displayed score conservative instead of presenting that as certainty.
    confidence = 0.5 + (raw_confidence - 0.5) * 0.8
    importances = model.feature_importances_
    contributions = [
        {
            "name": name,
            "val": values.get(name, 0),
            "impact": round(float(importance) * 100, 2),
            "target": "attack" if str(prediction).lower() not in {"0", "normal", "benign"} else "normal",
        }
        for name, importance in sorted(zip(selected, importances), key=lambda item: item[1], reverse=True)
        if importance > 0
    ][:8]
    if not contributions:
        contributions = [{"name": name, "val": values.get(name, 0), "impact": 0, "target": "normal"} for name in selected[:5]]
    return {
        "prediction": str(prediction),
        "confidence": confidence,
        "rawConfidence": raw_confidence,
        "selectedFeatures": selected,
        "featureContributions": contributions,
        "metrics": metrics,
        "totalFeatures": len(_feature_names(dataset)),
        "latencyMs": None,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body, content_type: str = "application/json"):
        encoded = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/samples":
            dataset = parse_qs(parsed.query).get("dataset", ["NSL-KDD"])[0]
            try:
                self._send(200, _dataset_payload(dataset))
            except (OSError, ValueError, KeyError) as exc:
                self._send(500, {"error": str(exc)})
            return
        report_name = "live_demo.html" if parsed.path in {"/", "/live_demo.html"} else parsed.path.lstrip("/")
        if report_name in {"live_demo.html", "demo_dashboard.html"}:
            report_path = REPORTS / report_name
            if not report_path.is_file():
                self._send(404, {"error": "Report not found"})
                return
            content = report_path.read_bytes()
            self._send(200, content, "text/html")
            return
        self._send(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/predict":
            self._send(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            result = _predict(json.loads(self.rfile.read(length)))
            self._send(200, result)
        except (ValueError, KeyError, OSError) as exc:
            self._send(400, {"error": str(exc)})

    def log_message(self, format, *args):
        print(format % args)


def main():
    port = int(os.environ.get("NIDS_DEMO_PORT", "8765"))
    print(f"NIDS demo: http://127.0.0.1:{port}/live_demo.html")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
