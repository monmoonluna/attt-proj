"""
preprocessing.py
=================
Pipeline tiền xử lý dữ liệu cho bài toán Phát hiện tấn công mạng
(Network Intrusion Detection) - hỗ trợ NSL-KDD, CICIDS2017, UNSW-NB15, CICDDoS2019.

CICDDoS2019 gồm nhiều file CSV theo từng loại tấn công/ngày thu thập - dùng
--merge-dir / --merge-output để gộp thành 1 file trước khi chạy pipeline chính:
    python preprocessing.py --merge-dir data/CICDDoS2019/01-12 --merge-output data/cicddos2019_day1.csv
    python preprocessing.py --merge-dir data/CICDDoS2019/03-11 --merge-output data/cicddos2019_day2.csv
    python preprocessing.py --input data/cicddos2019_day1.csv --test data/cicddos2019_day2.csv \\
        --dataset cicddos2019 --label-mode binary --output-dir output/CICDDoS2019/

Các bước xử lý:
    1. Đọc dữ liệu (CSV) và chuẩn hóa tên cột.
    2. Xử lý giá trị thiếu (NaN) và giá trị vô cực (Infinity).
    3. Loại bỏ bản ghi trùng lặp và cột có phương sai gần bằng 0.
    4. Mã hóa nhãn (label) -> nhị phân (Normal/Attack) và đa lớp (attack_category).
    5. Mã hóa đặc trưng phân loại (categorical) bằng Label Encoding.
    6. Chuẩn hóa đặc trưng số bằng MinMaxScaler / StandardScaler.
    7. Xử lý mất cân bằng lớp bằng SMOTE (tùy chọn).
    8. Chia tập train/test (hoặc dùng tập test có sẵn với NSL-KDD).
    9. Lưu kết quả (đặc trưng X, nhãn y, scaler, encoder) ra đĩa để dùng
       cho bước Feature Selection và huấn luyện mô hình ở bước sau.

Cách dùng nhanh (CLI):
    python preprocessing.py --input data/KDDTrain+.csv --dataset nslkdd \
        --output-dir output/ --scaler minmax --balance smote

Cách dùng như thư viện:
    from preprocessing import NIDSPreprocessor
    pre = NIDSPreprocessor(dataset="nslkdd", scaler="minmax", balance="smote")
    X_train, X_test, y_train, y_test = pre.run("data/KDDTrain+.csv", "data/KDDTest+.csv")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preprocessing")


# --------------------------------------------------------------------------- #
# Cấu hình cho từng bộ dữ liệu (tên cột nhãn, cột categorical, ánh xạ tấn công)
# --------------------------------------------------------------------------- #

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

# Ánh xạ nhãn tấn công chi tiết -> nhóm tấn công chính (dùng cho NSL-KDD)
NSL_KDD_ATTACK_MAP = {
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "worm": "DoS", "mailbomb": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}

DATASET_CONFIGS = {
    "nslkdd": dict(
        columns=NSL_KDD_COLUMNS,
        label_col="label",
        drop_cols=["difficulty"],
        categorical_cols=["protocol_type", "service", "flag"],
        attack_map=NSL_KDD_ATTACK_MAP,
        has_header=False,
    ),
    "cicids2017": dict(
        columns=None,          # CICIDS2017 CSV đã có header sẵn
        label_col="Attack Type",   # bản "cleaned-and-preprocessed" của Eric A. Ribeiro dùng tên cột này
        drop_cols=[],
        categorical_cols=[],   # hầu hết đặc trưng CICIDS2017 là số
        attack_map=None,       # giữ nguyên nhãn gốc (nhiều loại tấn công)
        has_header=True,
    ),
    "unswnb15": dict(
        columns=None,
        label_col="attack_cat",     # cột nhãn đa lớp; "label" là cột nhị phân có sẵn
        drop_cols=["id"],
        categorical_cols=["proto", "service", "state"],
        attack_map=None,
        has_header=True,
    ),
    "cicddos2019": dict(
        columns=None,           # CSV gốc/CICFlowMeter đã có header sẵn (có thể có khoảng trắng đầu tên cột)
        label_col="Label",      # sau strip() tên cột, " Label" -> "Label"
        # Các cột định danh/không mang thông tin hành vi mạng thực sự - loại bỏ nếu có mặt,
        # tránh rò rỉ dữ liệu (IP/cổng nguồn chỉ phản ánh topology mô phỏng, không phải hành vi tấn công)
        drop_cols=[
            "Unnamed: 0", "Flow ID", "Source IP", "Src IP", "Source Port", "Src Port",
            "Destination IP", "Dst IP", "Timestamp", "SimillarHTTP",
            # "Inbound": LOẠI BỎ vì là artifact của cách bố trí testbed, không phải hành
            # vi mạng thật - xác nhận bằng ablation thực nghiệm: label Attack có
            # Inbound=1 ở 99,9% (gần như đồng nhất với nhãn) trong khi Normal thì lẫn lộn
            # (79,8%/20,1%); loại bỏ cột này giúp Hybrid2 tăng F1 từ 0,8633 lên 0,9534 trên
            # tập test (có loại tấn công khác tập train) - bằng chứng rõ ràng mô hình dựa
            # vào Inbound đang overfit vào quy ước capture của testbed, không tổng quát hóa
            # được sang tấn công mới. Xem mục 12.4 (phần phát hiện Inbound) trong đề cương.
            "Inbound",
        ],
        categorical_cols=[],    # hầu hết đặc trưng CICDDoS2019 (CICFlowMeter) là số
        attack_map=None,        # giữ nguyên nhãn gốc (BENIGN, DrDoS_DNS, Syn, UDP, Portmap, WebDDoS, ...)
        has_header=True,
    ),
}


# --------------------------------------------------------------------------- #
# Tiện ích gộp nhiều file CSV (CICDDoS2019 gốc chia theo nhiều file tấn
# công/ngày thu thập - ví dụ DrDoS_DNS.csv, Syn.csv, UDP.csv, ... mỗi ngày).
# Dùng 1 lần trước khi chạy pipeline chính: gộp toàn bộ file của 1 ngày
# thành 1 file CSV duy nhất để làm --input hoặc --test.
# --------------------------------------------------------------------------- #

def merge_csv_directory(input_dir: str, output_csv: str, pattern: str = "*.csv") -> pd.DataFrame:
    """Gộp tất cả file CSV trong một thư mục (và thư mục con) thành 1 file duy nhất.

    Các file có thể có tập cột hơi khác nhau (một số bản CICDDoS2019 thiếu/thừa
    vài cột giữa các file tấn công) - hàm này hợp nhất theo UNION cột, điền NaN
    cho cột thiếu ở từng file, sau đó nối (concat) theo dòng.
    """
    import glob

    paths = sorted(glob.glob(os.path.join(input_dir, "**", pattern), recursive=True))
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào trong {input_dir}")

    logger.info("Tìm thấy %d file CSV để gộp trong %s", len(paths), input_dir)
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            logger.warning("Bỏ qua file lỗi %s: %s", p, e)
            continue
        df.columns = [c.strip() for c in df.columns]
        df["__source_file__"] = os.path.basename(p)   # chỉ để log/kiểm tra, sẽ bị loại trước khi lưu
        frames.append(df)
        logger.info("  + %s: %d dòng, %d cột", os.path.basename(p), len(df), df.shape[1])

    merged = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("Phân bố theo file nguồn:\n%s", merged["__source_file__"].value_counts().to_string())
    merged = merged.drop(columns=["__source_file__"])  # không đưa cột debug vào file CSV cuối cùng

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
    merged.to_csv(output_csv, index=False)
    logger.info("Đã gộp thành %s: %d dòng, %d cột", output_csv, len(merged), merged.shape[1])
    return merged


@dataclass
class NIDSPreprocessor:
    """Pipeline tiền xử lý dữ liệu cho bài toán NIDS."""

    dataset: str = "nslkdd"                # "nslkdd" | "cicids2017" | "unswnb15"
    scaler: str = "minmax"                 # "minmax" | "standard" | None
    balance: Optional[str] = None          # "smote" | None
    test_size: float = 0.2
    random_state: int = 42
    nzv_threshold: float = 1e-5            # ngưỡng phương sai gần-0 để loại cột

    label_encoder_: LabelEncoder = field(default=None, init=False, repr=False)
    scaler_: object = field(default=None, init=False, repr=False)
    categorical_encoders_: dict = field(default_factory=dict, init=False, repr=False)
    feature_names_: list = field(default=None, init=False, repr=False)
    has_unknown_class_: bool = field(default=False, init=False, repr=False)
    unknown_class_names_: list = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # 1. Đọc dữ liệu
    # ------------------------------------------------------------------ #
    def load(self, path: str) -> pd.DataFrame:
        cfg = self._cfg()
        logger.info("Đang đọc dữ liệu: %s", path)
        if cfg["has_header"]:
            df = pd.read_csv(path, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
        else:
            df = pd.read_csv(path, header=None, names=cfg["columns"])
        logger.info("Kích thước dữ liệu gốc: %s", df.shape)
        return df

    # ------------------------------------------------------------------ #
    # 2. Xử lý missing / infinity
    # ------------------------------------------------------------------ #
    @staticmethod
    def clean_missing_and_infinite(df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        n_missing = df.isna().sum().sum()
        if n_missing > 0:
            logger.info("Phát hiện %d giá trị thiếu/vô cực -> điền bằng median cột", n_missing)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            df = df.ffill().bfill()
        df = df.dropna()
        logger.info("Loại bỏ %d dòng không thể phục hồi (nếu có)", n_before - len(df))
        return df

    # ------------------------------------------------------------------ #
    # 3. Loại bỏ trùng lặp & near-zero-variance
    # ------------------------------------------------------------------ #
    def remove_duplicates_and_nzv(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        n_before = len(df)
        df = df.drop_duplicates()
        logger.info("Loại bỏ %d bản ghi trùng lặp", n_before - len(df))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(
            label_col, errors="ignore"
        )
        variances = df[numeric_cols].var()
        nzv_cols = variances[variances <= self.nzv_threshold].index.tolist()
        if nzv_cols:
            logger.info("Loại bỏ %d cột phương sai gần-0: %s", len(nzv_cols), nzv_cols)
            df = df.drop(columns=nzv_cols)
        return df

    # ------------------------------------------------------------------ #
    # 4. Chuẩn hóa nhãn
    # ------------------------------------------------------------------ #
    def build_labels(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        cfg = self._cfg()
        label_col = cfg["label_col"]

        raw_label = df[label_col].astype(str).str.strip().str.replace(".", "", regex=False)

        if cfg["attack_map"] is not None:
            multi = raw_label.str.lower().map(cfg["attack_map"]).fillna("Unknown")
        else:
            multi = raw_label

        # So khớp linh hoạt (contains) thay vì khớp tuyệt đối, vì các bộ dữ liệu khác nhau
        # đặt tên nhãn "bình thường" khác nhau: "normal", "BENIGN", "Normal Traffic"...
        is_normal = multi.str.lower().str.contains("normal", na=False) | \
                    multi.str.lower().str.contains("benign", na=False)
        binary = np.where(is_normal, "Normal", "Attack")
        binary = pd.Series(binary, index=df.index, name="binary_label")

        drop_cols = [c for c in cfg["drop_cols"] if c in df.columns]
        drop_cols += [label_col]
        # với UNSW-NB15, cột "label" nhị phân gốc cũng nên loại khỏi X nếu tồn tại
        if "label" in df.columns and "label" != label_col:
            drop_cols.append("label")
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        return X, multi.rename("multi_label"), binary

    # ------------------------------------------------------------------ #
    # 5. Encode categorical
    # ------------------------------------------------------------------ #
    def encode_categorical(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        cfg = self._cfg()
        X = X.copy()
        for col in cfg["categorical_cols"]:
            if col not in X.columns:
                continue
            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.categorical_encoders_[col] = le
            else:
                le = self.categorical_encoders_[col]
                X[col] = X[col].astype(str).map(
                    lambda v: v if v in le.classes_ else le.classes_[0]
                )
                X[col] = le.transform(X[col])
        # Mọi cột object còn sót lại -> one-hot (phòng trường hợp dataset có thêm cột text)
        remaining_obj = X.select_dtypes(include=["object"]).columns.tolist()
        if remaining_obj:
            X = pd.get_dummies(X, columns=remaining_obj)
        return X

    # ------------------------------------------------------------------ #
    # 6. Scale numeric features
    # ------------------------------------------------------------------ #
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        if self.scaler is None:
            return X
        if fit:
            self.scaler_ = MinMaxScaler() if self.scaler == "minmax" else StandardScaler()
            values = self.scaler_.fit_transform(X)
        else:
            values = self.scaler_.transform(X)
        return pd.DataFrame(values, columns=X.columns, index=X.index)

    # ------------------------------------------------------------------ #
    # 7. Xử lý mất cân bằng lớp
    # ------------------------------------------------------------------ #
    def balance_classes(self, X: pd.DataFrame, y: pd.Series):
        if self.balance is None:
            return X, y
        if self.balance == "smote":
            from imblearn.over_sampling import SMOTE
            logger.info("Áp dụng SMOTE để cân bằng lớp. Phân bố trước: %s", y.value_counts().to_dict())
            sm = SMOTE(random_state=self.random_state)
            X_res, y_res = sm.fit_resample(X, y)
            logger.info("Phân bố sau SMOTE: %s", pd.Series(y_res).value_counts().to_dict())
            return X_res, y_res
        raise ValueError(f"Phương pháp cân bằng lớp không hỗ trợ: {self.balance}")

    # ------------------------------------------------------------------ #
    # Pipeline tổng: fit_transform trên tập train, transform trên tập test
    # ------------------------------------------------------------------ #
    def fit_transform(self, df: pd.DataFrame, label_mode: str = "binary"):
        cfg = self._cfg()
        df = self.clean_missing_and_infinite(df)
        df = self.remove_duplicates_and_nzv(df, cfg["label_col"])
        X, multi_label, binary_label = self.build_labels(df)

        y_raw = binary_label if label_mode == "binary" else multi_label
        self.label_encoder_ = LabelEncoder()
        y = pd.Series(self.label_encoder_.fit_transform(y_raw), index=X.index, name="label")

        X = self.encode_categorical(X, fit=True)
        self.feature_names_ = X.columns.tolist()
        X = self.scale_features(X, fit=True)
        X, y = self.balance_classes(X, y)
        return X, y

    def transform(self, df: pd.DataFrame, label_mode: str = "binary"):
        cfg = self._cfg()
        df = self.clean_missing_and_infinite(df)
        X, multi_label, binary_label = self.build_labels(df)
        y_raw = binary_label if label_mode == "binary" else multi_label

        # Xử lý nhãn lạ ở tập test (chưa từng xuất hiện ở tập train) - ví dụ các loại
        # tấn công DNS/NTP/SNMP/TFTP chỉ có ở tập test của CICDDoS2019 (mục 6.1 đề cương).
        # TRƯỚC ĐÂY: gán nhầm vào lớp đầu tiên (label_encoder_.classes_[0]) - sai về ý
        # nghĩa, làm mô hình "nhìn nhầm" tấn công mới thành lớp đã biết.
        # BÂY GIỜ: gán vào một lớp "Unknown" RIÊNG, với mã số mới (ngay sau mã lớn nhất
        # của tập train) - vừa đúng về ý nghĩa (không giả vờ là lớp đã biết), vừa biến
        # đây thành một chỉ số đánh giá khả năng tổng quát hóa sang tấn công hoàn toàn mới.
        known_classes = list(self.label_encoder_.classes_)
        label_to_code = {cls: i for i, cls in enumerate(known_classes)}
        unknown_code = len(known_classes)

        is_unseen = ~y_raw.isin(known_classes)
        if is_unseen.any():
            unseen_labels = sorted(y_raw[is_unseen].unique().tolist())
            logger.info(
                "Phát hiện %d/%d mẫu có nhãn CHƯA TỪNG xuất hiện ở tập train (%s) "
                "-> gán vào lớp \"Unknown\" (mã số %d) thay vì nhầm vào lớp đã biết.",
                int(is_unseen.sum()), len(y_raw), unseen_labels, unknown_code,
            )
            self.unknown_class_names_ = unseen_labels
            self.has_unknown_class_ = True
        else:
            self.has_unknown_class_ = False

        y = pd.Series(
            y_raw.map(lambda v: label_to_code.get(v, unknown_code)).values,
            index=X.index, name="label",
        )

        X = self.encode_categorical(X, fit=False)
        X = X.reindex(columns=self.feature_names_, fill_value=0)
        X = self.scale_features(X, fit=False)
        return X, y

    def run(self, train_path: str, test_path: Optional[str] = None, label_mode: str = "binary"):
        """Chạy toàn bộ pipeline. Nếu không có test_path riêng, tự động chia train/test."""
        df_train = self.load(train_path)

        if test_path:
            X_train, y_train = self.fit_transform(df_train, label_mode=label_mode)
            df_test = self.load(test_path)
            X_test, y_test = self.transform(df_test, label_mode=label_mode)
        else:
            X_all, y_all = self.fit_transform(df_train, label_mode=label_mode)
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=self.test_size,
                random_state=self.random_state, stratify=y_all,
            )

        logger.info("Kích thước tập train: %s | test: %s", X_train.shape, X_test.shape)
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------ #
    def _cfg(self) -> dict:
        if self.dataset not in DATASET_CONFIGS:
            raise ValueError(
                f"Dataset '{self.dataset}' không được hỗ trợ. "
                f"Chọn một trong: {list(DATASET_CONFIGS.keys())}"
            )
        return DATASET_CONFIGS[self.dataset]

    # ------------------------------------------------------------------ #
    def save(self, output_dir: str, X_train, X_test, y_train, y_test):
        os.makedirs(output_dir, exist_ok=True)
        X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
        pd.Series(y_train, name="label").to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
        pd.Series(y_test, name="label").to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

        joblib.dump(self.scaler_, os.path.join(output_dir, "scaler.pkl"))
        joblib.dump(self.label_encoder_, os.path.join(output_dir, "label_encoder.pkl"))
        joblib.dump(self.categorical_encoders_, os.path.join(output_dir, "categorical_encoders.pkl"))

        with open(os.path.join(output_dir, "feature_names.json"), "w", encoding="utf-8") as f:
            json.dump(self.feature_names_, f, ensure_ascii=False, indent=2)

        # Ánh xạ mã số -> tên lớp, kèm lớp "Unknown" nếu tập test có nhãn lạ (mục transform())
        label_classes = list(self.label_encoder_.classes_)
        class_mapping = {i: cls for i, cls in enumerate(label_classes)}
        if self.has_unknown_class_:
            class_mapping[len(label_classes)] = "Unknown"
        with open(os.path.join(output_dir, "label_classes.json"), "w", encoding="utf-8") as f:
            json.dump({
                "code_to_class": {str(k): v for k, v in class_mapping.items()},
                "has_unknown_class": self.has_unknown_class_,
                "unknown_original_labels": self.unknown_class_names_,
            }, f, ensure_ascii=False, indent=2)

        logger.info("Đã lưu toàn bộ dữ liệu và bộ mã hóa vào: %s", output_dir)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline tiền xử lý dữ liệu NIDS")
    parser.add_argument("--merge-dir", default=None,
                         help="Nếu truyền vào: CHỈ gộp toàn bộ CSV trong thư mục này thành 1 file "
                              "rồi thoát (dùng cho CICDDoS2019 gốc có nhiều file theo từng loại tấn công/ngày). "
                              "Cần đi kèm --merge-output.")
    parser.add_argument("--merge-output", default=None, help="Đường dẫn file CSV gộp sẽ được ghi ra")
    parser.add_argument("--input", required=False, help="Đường dẫn file CSV huấn luyện")
    parser.add_argument("--test", default=None, help="Đường dẫn file CSV kiểm tra (tùy chọn)")
    parser.add_argument("--dataset", default="nslkdd", choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--label-mode", default="binary", choices=["binary", "multi"])
    parser.add_argument("--scaler", default="minmax", choices=["minmax", "standard", "none"])
    parser.add_argument("--balance", default=None, choices=[None, "smote"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", default="output_preprocessed")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.merge_dir:
        if not args.merge_output:
            raise SystemExit("--merge-dir cần đi kèm --merge-output")
        merge_csv_directory(args.merge_dir, args.merge_output)
        return

    if not args.input:
        raise SystemExit("--input là bắt buộc khi không dùng --merge-dir")

    scaler = None if args.scaler == "none" else args.scaler

    pre = NIDSPreprocessor(
        dataset=args.dataset,
        scaler=scaler,
        balance=args.balance,
        test_size=args.test_size,
    )
    X_train, X_test, y_train, y_test = pre.run(
        args.input, test_path=args.test, label_mode=args.label_mode
    )
    pre.save(args.output_dir, X_train, X_test, y_train, y_test)
    logger.info("Hoàn tất tiền xử lý. Số đặc trưng cuối cùng: %d", X_train.shape[1])


if __name__ == "__main__":
    main()
