"""
feature_selection.py
=====================
Module cài đặt Feature Selection nâng cao bằng 2 thuật toán meta-heuristic:
    - Genetic Algorithm (GA)      -> lớp GAFeatureSelector
    - Binary Particle Swarm Opt.  -> lớp PSOFeatureSelector (BPSO)

Cả hai đều kế thừa từ FeatureSelectorBase, dùng chung một hàm fitness có thể
tùy chỉnh trọng số giữa độ chính xác (F1-macro) và số lượng đặc trưng được
chọn - đúng như hướng "cải tiến hàm fitness" đã nêu trong đề cương.

    fitness(mask) = w_acc * F1_macro(mask) - w_feat * (số đặc trưng đã chọn / tổng số đặc trưng)

Thiết kế để đọc trực tiếp output của preprocessing.py (X_train.csv, y_train.csv,
X_test.csv, y_test.csv), tự tách một tập con để đánh giá fitness cho nhanh
(vì GA/PSO phải huấn luyện mô hình hàng trăm/nghìn lần), sau đó đánh giá lại
bằng mô hình huấn luyện trên TOÀN BỘ tập train + tập test thật để có số liệu
báo cáo chính xác.

Cách dùng nhanh (CLI):
    python feature_selection.py --data-dir output/NSL-KDD --method ga \
        --classifier dt --pop-size 30 --generations 30 \
        --w-acc 0.9 --w-feat 0.1 --sample-size 5000 \
        --output-dir output/NSL-KDD/fs_ga

    python feature_selection.py --data-dir output/NSL-KDD --method pso \
        --classifier dt --swarm-size 30 --iterations 30 \
        --output-dir output/NSL-KDD/fs_pso

Cách dùng như thư viện:
    from feature_selection import GAFeatureSelector, load_split
    X_train, X_test, y_train, y_test = load_split("output/NSL-KDD")
    ga = GAFeatureSelector(X_train, y_train, classifier="dt")
    mask, best_fit, history = ga.run(pop_size=30, generations=30)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("feature_selection")

def _safe_stratified_split(X, y, random_state: int, train_size: Optional[float] = None, test_size: Optional[float] = None):
    """train_test_split(..., stratify=y) nhưng tự động rơi về KHÔNG phân tầng nếu có
    lớp quá hiếm (ít hơn 2 mẫu trong tập con) - tránh crash khi gặp lớp cực hiếm, ví
    dụ lớp "Unknown" hoặc các loại tấn công hiếm trong --label-mode multi."""
    try:
        return train_test_split(
            X, y, train_size=train_size, test_size=test_size,
            stratify=y, random_state=random_state,
        )
    except ValueError as e:
        if "least populated class" not in str(e):
            raise
        logger.warning(
            "Phát hiện lớp quá hiếm (ít hơn 2 mẫu) khi chia dữ liệu theo stratify - "
            "tự động chuyển sang chia KHÔNG phân tầng cho lần chia này: %s", e,
        )
        return train_test_split(
            X, y, train_size=train_size, test_size=test_size,
            stratify=None, random_state=random_state,
        )


CLASSIFIERS = {
    "dt": lambda: DecisionTreeClassifier(random_state=42),
    "rf": lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
}

try:
    from xgboost import XGBClassifier
    CLASSIFIERS["xgb"] = lambda: XGBClassifier(
        # Không ép eval_metric cố định - để XGBoost tự chọn theo số lớp của y
        # (logloss cho nhị phân, mlogloss cho đa lớp), tránh lỗi/sai lệch khi
        # dùng --label-mode multi.
        n_estimators=200, max_depth=6,
        random_state=42, n_jobs=-1, verbosity=0,
    )
except ImportError:
    pass  # xgboost chưa cài - các module filter_fs.py/hybrid2_fs.py sẽ báo lỗi rõ ràng nếu cố dùng "xgb"


# --------------------------------------------------------------------------- #
# Đọc dữ liệu output từ preprocessing.py
# --------------------------------------------------------------------------- #

def load_split(data_dir: str):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))["label"]
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))["label"]
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Lớp cơ sở: hàm fitness dùng chung cho GA và PSO
# --------------------------------------------------------------------------- #

class FeatureSelectorBase:
    """Cung cấp hàm fitness và cơ chế đánh giá dùng chung cho GA/PSO.

    Để tránh phải huấn luyện mô hình trên toàn bộ dữ liệu (có thể lên tới
    hàng triệu dòng với CICIDS2017) hàng trăm/nghìn lần trong quá trình tối
    ưu, hàm fitness được tính trên một tập con cố định (sample_size), được
    tách sẵn thành fit/validation một lần duy nhất khi khởi tạo.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        classifier: str = "dt",
        w_acc: float = 0.9,
        w_feat: float = 0.1,
        w_minority: float = 0.0,
        sample_size: Optional[int] = 5000,
        val_ratio: float = 0.3,
        min_features: int = 1,
        random_state: int = 42,
    ):
        if classifier not in CLASSIFIERS:
            raise ValueError(f"Classifier '{classifier}' không hỗ trợ. Chọn: {list(CLASSIFIERS)}")

        self.feature_names = X_train.columns.tolist()
        self.n_features = len(self.feature_names)
        self.classifier_factory: Callable = CLASSIFIERS[classifier]
        self.w_acc = w_acc
        self.w_feat = w_feat
        self.w_minority = w_minority
        self.min_features = min_features
        self.random_state = random_state
        rng = np.random.default_rng(random_state)
        self.rng = rng

        # Lớp thiểu số (minority class) - dùng cho hàm fitness cải tiến của Hybrid 1
        # (mục 6.4 đề cương: "kết hợp accuracy, số lượng đặc trưng, F1 của lớp thiểu số").
        class_counts = pd.Series(y_train).value_counts()
        self.minority_class = class_counts.idxmin()

        if sample_size and sample_size < len(X_train):
            X_sub, _, y_sub, _ = _safe_stratified_split(
                X_train, y_train, train_size=sample_size, random_state=random_state,
            )
        else:
            X_sub, y_sub = X_train, y_train

        self.X_fit, self.X_val, self.y_fit, self.y_val = _safe_stratified_split(
            X_sub, y_sub, test_size=val_ratio, random_state=random_state,
        )
        logger.info(
            "Tập đánh giá fitness: fit=%d dòng, val=%d dòng (từ sample_size=%s)",
            len(self.X_fit), len(self.X_val), sample_size,
        )

        self._X_fit_np = self.X_fit.to_numpy()
        self._X_val_np = self.X_val.to_numpy()
        self._cache: dict[bytes, float] = {}
        self.n_evaluations = 0

    # ------------------------------------------------------------------ #
    def repair(self, mask: np.ndarray) -> np.ndarray:
        """Đảm bảo mỗi nghiệm chọn ít nhất `min_features` đặc trưng."""
        mask = mask.astype(np.uint8).copy()
        if mask.sum() < self.min_features:
            idx = self.rng.choice(self.n_features, size=self.min_features, replace=False)
            mask[:] = 0
            mask[idx] = 1
        return mask

    def fitness(self, mask: np.ndarray) -> float:
        """fitness = w_acc * F1_macro + w_minority * F1(lớp thiểu số) - w_feat * (số đặc trưng/tổng số đặc trưng).

        Số hạng w_minority * F1(lớp thiểu số) là phần "cải tiến" của Hybrid 1 so với
        fitness gốc dùng cho GA/PSO thuần túy: F1-macro vốn đã coi các lớp ngang
        nhau, nhưng khi w_minority > 0, nghiệm nào giữ được khả năng nhận diện
        riêng lớp hiếm (thường là lớp Attack ít mẫu hơn, hoặc ngược lại tùy dataset)
        sẽ được thưởng thêm - tránh trường hợp FS vô tình bỏ mất đặc trưng quan
        trọng với lớp thiểu số chỉ vì nó không ảnh hưởng nhiều đến F1-macro tổng thể.
        """
        mask = self.repair(mask)
        key = mask.tobytes()
        if key in self._cache:
            return self._cache[key]

        idx = np.where(mask == 1)[0]
        clf = self.classifier_factory()
        clf.fit(self._X_fit_np[:, idx], self.y_fit)
        preds = clf.predict(self._X_val_np[:, idx])
        f1 = f1_score(self.y_val, preds, average="macro", zero_division=0)
        feat_ratio = mask.sum() / self.n_features
        value = self.w_acc * f1 - self.w_feat * feat_ratio

        if self.w_minority:
            f1_minority = f1_score(
                self.y_val, preds, labels=[self.minority_class], average="macro", zero_division=0
            )
            value += self.w_minority * f1_minority

        self._cache[key] = value
        self.n_evaluations += 1
        return value

    def selected_feature_names(self, mask: np.ndarray) -> list[str]:
        mask = self.repair(mask)
        return [name for name, bit in zip(self.feature_names, mask) if bit == 1]


# --------------------------------------------------------------------------- #
# Genetic Algorithm
# --------------------------------------------------------------------------- #

class GAFeatureSelector(FeatureSelectorBase):
    """Feature Selection bằng Genetic Algorithm (biểu diễn nhị phân)."""

    def run(
        self,
        pop_size: int = 30,
        generations: int = 30,
        cx_pb: float = 0.8,
        mut_pb: Optional[float] = None,
        tournament_k: int = 3,
        init_prob: float = 0.5,
        elitism: int = 1,
        verbose: bool = True,
    ):
        rng = self.rng
        n = self.n_features
        mut_pb = mut_pb if mut_pb is not None else 1.0 / n

        # Khởi tạo quần thể
        population = (rng.random((pop_size, n)) < init_prob).astype(np.uint8)
        fitness_vals = np.array([self.fitness(ind) for ind in population])

        history = []
        best_idx = int(np.argmax(fitness_vals))
        best_mask = population[best_idx].copy()
        best_fit = fitness_vals[best_idx]
        history.append(best_fit)

        for gen in range(1, generations + 1):
            new_population = []

            # Giữ lại `elitism` cá thể tốt nhất (elitism)
            elite_idx = np.argsort(-fitness_vals)[:elitism]
            for i in elite_idx:
                new_population.append(population[i].copy())

            while len(new_population) < pop_size:
                # Chọn lọc tournament (tournament selection)
                p1 = self._tournament_select(population, fitness_vals, tournament_k)
                p2 = self._tournament_select(population, fitness_vals, tournament_k)

                # Lai ghép đồng nhất (uniform crossover)
                if rng.random() < cx_pb:
                    mask_cx = rng.random(n) < 0.5
                    c1 = np.where(mask_cx, p1, p2)
                    c2 = np.where(mask_cx, p2, p1)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Đột biến lật bit (bit-flip mutation)
                c1 = self._mutate(c1, mut_pb)
                c2 = self._mutate(c2, mut_pb)

                new_population.append(c1)
                if len(new_population) < pop_size:
                    new_population.append(c2)

            population = np.array(new_population[:pop_size])
            fitness_vals = np.array([self.fitness(ind) for ind in population])

            gen_best_idx = int(np.argmax(fitness_vals))
            if fitness_vals[gen_best_idx] > best_fit:
                best_fit = fitness_vals[gen_best_idx]
                best_mask = population[gen_best_idx].copy()

            history.append(best_fit)
            if verbose:
                logger.info(
                    "[GA] Thế hệ %3d/%d | best fitness = %.5f | số đặc trưng = %d/%d",
                    gen, generations, best_fit, self.repair(best_mask).sum(), n,
                )

        return self.repair(best_mask), best_fit, history

    def _tournament_select(self, population, fitness_vals, k):
        idx = self.rng.choice(len(population), size=k, replace=False)
        winner = idx[np.argmax(fitness_vals[idx])]
        return population[winner]

    def _mutate(self, individual, mut_pb):
        flip = self.rng.random(self.n_features) < mut_pb
        individual = individual.copy()
        individual[flip] = 1 - individual[flip]
        return individual


# --------------------------------------------------------------------------- #
# Binary Particle Swarm Optimization
# --------------------------------------------------------------------------- #

class PSOFeatureSelector(FeatureSelectorBase):
    """Feature Selection bằng Binary Particle Swarm Optimization (BPSO).

    Vị trí hạt là vector nhị phân; vận tốc là số thực, được chuyển thành
    xác suất bit=1 qua hàm sigmoid (theo Kennedy & Eberhart, 1997).

    KHẮC PHỤC HỘI TỤ SỚM (premature convergence - quan sát được lặp lại trên
    cả 4 bộ dữ liệu đã chạy, xem mục 12.2/12.4.2 trong đề cương): khi
    (pbest - position) và (gbest - position) đều tiến về 0, quần thể mất đa
    dạng và best fitness đứng yên từ những vòng lặp đầu. Hai cơ chế được
    thêm vào để khắc phục:
        1. Đột biến ngẫu nhiên định kỳ (random mutation): mỗi vòng lặp, một
           tỉ lệ nhỏ hạt (mutation_rate) bị lật ngẫu nhiên một vài bit,
           không phụ thuộc vào pbest/gbest - duy trì đa dạng quần thể liên tục.
        2. Tái khởi tạo một phần quần thể (partial re-initialization): nếu
           gbest không cải thiện sau `stagnation_limit` vòng lặp liên tiếp,
           một tỉ lệ (reinit_fraction) các hạt tệ nhất (theo pbest_fit) được
           khởi tạo lại ngẫu nhiên hoàn toàn (vị trí lẫn vận tốc), cho một
           đợt khám phá không gian tìm kiếm mới.
    """

    def run(
        self,
        swarm_size: int = 30,
        iterations: int = 30,
        w: float = 0.7,
        w_min: float = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max: float = 4.0,
        init_prob: float = 0.5,
        mutation_rate: float = 0.15,
        bit_mutation_prob: Optional[float] = None,
        stagnation_limit: int = 5,
        reinit_fraction: float = 0.3,
        verbose: bool = True,
    ):
        rng = self.rng
        n = self.n_features
        bit_mutation_prob = bit_mutation_prob if bit_mutation_prob is not None else 3.0 / n

        positions = (rng.random((swarm_size, n)) < init_prob).astype(np.uint8)
        velocities = rng.uniform(-v_max, v_max, size=(swarm_size, n))

        fitness_vals = np.array([self.fitness(p) for p in positions])
        pbest_pos = positions.copy()
        pbest_fit = fitness_vals.copy()

        gbest_idx = int(np.argmax(pbest_fit))
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

        history = [gbest_fit]
        stagnation_counter = 0

        for it in range(1, iterations + 1):
            inertia = w - (w - w_min) * (it / iterations)  # inertia giảm dần

            r1 = rng.random((swarm_size, n))
            r2 = rng.random((swarm_size, n))
            velocities = (
                inertia * velocities
                + c1 * r1 * (pbest_pos - positions)
                + c2 * r2 * (gbest_pos - positions)
            )
            velocities = np.clip(velocities, -v_max, v_max)

            # Hàm chuyển sigmoid: xác suất bit = 1
            prob = 1.0 / (1.0 + np.exp(-velocities))
            positions = (rng.random((swarm_size, n)) < prob).astype(np.uint8)

            # --- (1) Đột biến ngẫu nhiên định kỳ: duy trì đa dạng quần thể ---
            mutate_particles = rng.random(swarm_size) < mutation_rate
            if mutate_particles.any():
                flip_mask = rng.random((mutate_particles.sum(), n)) < bit_mutation_prob
                positions[mutate_particles] = np.where(
                    flip_mask, 1 - positions[mutate_particles], positions[mutate_particles]
                )

            fitness_vals = np.array([self.fitness(p) for p in positions])

            improved = fitness_vals > pbest_fit
            pbest_pos[improved] = positions[improved]
            pbest_fit[improved] = fitness_vals[improved]

            gen_best_idx = int(np.argmax(pbest_fit))
            if pbest_fit[gen_best_idx] > gbest_fit:
                gbest_fit = pbest_fit[gen_best_idx]
                gbest_pos = pbest_pos[gen_best_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # --- (2) Tái khởi tạo một phần quần thể nếu trì trệ kéo dài ---
            if stagnation_counter >= stagnation_limit:
                n_reinit = max(1, round(swarm_size * reinit_fraction))
                worst_idx = np.argsort(pbest_fit)[:n_reinit]  # các hạt có pbest tệ nhất
                positions[worst_idx] = (rng.random((n_reinit, n)) < init_prob).astype(np.uint8)
                velocities[worst_idx] = rng.uniform(-v_max, v_max, size=(n_reinit, n))
                new_fit = np.array([self.fitness(p) for p in positions[worst_idx]])
                pbest_pos[worst_idx] = positions[worst_idx]
                pbest_fit[worst_idx] = new_fit
                stagnation_counter = 0
                if verbose:
                    logger.info(
                        "[PSO] Vòng lặp %3d: trì trệ %d vòng liên tiếp -> tái khởi tạo %d/%d hạt tệ nhất",
                        it, stagnation_limit, n_reinit, swarm_size,
                    )

            history.append(gbest_fit)
            if verbose:
                logger.info(
                    "[PSO] Vòng lặp %3d/%d | best fitness = %.5f | số đặc trưng = %d/%d",
                    it, iterations, gbest_fit, self.repair(gbest_pos).sum(), n,
                )

        return self.repair(gbest_pos), gbest_fit, history


# --------------------------------------------------------------------------- #
# Đánh giá cuối cùng trên toàn bộ tập train/test thật
# --------------------------------------------------------------------------- #

def evaluate_on_full_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    mask: np.ndarray, classifier: str = "dt",
) -> dict:
    """Huấn luyện trên TOÀN BỘ tập train (chỉ dùng đặc trưng đã chọn) và
    đánh giá trên tập test thật - dùng để báo cáo kết quả cuối cùng."""
    feature_names = X_train.columns.tolist()
    selected = [name for name, bit in zip(feature_names, mask) if bit == 1]

    clf = CLASSIFIERS[classifier]()
    t0 = time.time()
    clf.fit(X_train[selected], y_train)
    train_time = time.time() - t0

    t0 = time.time()
    preds = clf.predict(X_test[selected])
    infer_time = time.time() - t0

    return {
        "n_features_selected": len(selected),
        "n_features_total": len(feature_names),
        "selected_features": selected,
        "accuracy": accuracy_score(y_test, preds),
        "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
        "train_time_sec": train_time,
        "inference_time_sec": infer_time,
    }


def plot_convergence(history: list[float], title: str, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(range(len(history)), history, marker="o", markersize=3)
    plt.xlabel("Thế hệ / Vòng lặp")
    plt.ylabel("Best fitness")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="GA / PSO Feature Selection cho bài toán NIDS")
    parser.add_argument("--data-dir", required=True,
                         help="Thư mục chứa X_train.csv, X_test.csv, y_train.csv, y_test.csv (output của preprocessing.py)")
    parser.add_argument("--method", choices=["ga", "pso"], required=True)
    parser.add_argument("--classifier", choices=list(CLASSIFIERS), default="dt",
                         help="Classifier dùng để TÌM KIẾM (tính fitness hàng trăm lần) - nên giữ 'dt' "
                              "cho nhanh; RF/XGBoost ở đây sẽ làm GA/PSO chậm đi rất nhiều lần.")
    parser.add_argument("--eval-classifier", default=None,
                         help="Classifier dùng để ĐÁNH GIÁ CUỐI (chỉ train 1-2 lần, không ảnh hưởng tốc "
                              "độ tìm kiếm) - mặc định dùng lại --classifier nếu không truyền. Dùng tham "
                              "số này để so sánh RF/XGBoost mà không làm chậm GA/PSO, ví dụ: "
                              "--classifier dt --eval-classifier rf")

    parser.add_argument("--w-acc", type=float, default=0.9, help="Trọng số cho F1 trong hàm fitness")
    parser.add_argument("--w-feat", type=float, default=0.1, help="Trọng số phạt số lượng đặc trưng")
    parser.add_argument("--sample-size", type=int, default=5000,
                         help="Số dòng lấy mẫu để tính fitness cho nhanh (0 = dùng toàn bộ)")
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
    parser.add_argument("--mutation-rate", type=float, default=0.15,
                         help="Tỉ lệ hạt bị đột biến ngẫu nhiên mỗi vòng lặp (chống hội tụ sớm)")
    parser.add_argument("--stagnation-limit", type=int, default=5,
                         help="Số vòng lặp không cải thiện trước khi tái khởi tạo một phần quần thể")
    parser.add_argument("--reinit-fraction", type=float, default=0.3,
                         help="Tỉ lệ hạt tệ nhất được tái khởi tạo khi trì trệ")

    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split(args.data_dir)
    sample_size = args.sample_size if args.sample_size > 0 else None
    eval_classifier = args.eval_classifier if args.eval_classifier else args.classifier

    common_kwargs = dict(
        X_train=X_train, y_train=y_train, classifier=args.classifier,
        w_acc=args.w_acc, w_feat=args.w_feat,
        sample_size=sample_size, min_features=args.min_features,
    )

    t0 = time.time()
    if args.method == "ga":
        selector = GAFeatureSelector(**common_kwargs)
        mask, best_fit, history = selector.run(
            pop_size=args.pop_size, generations=args.generations,
            cx_pb=args.cx_pb, mut_pb=args.mut_pb,
        )
        title = "GA - Convergence curve"
    else:
        selector = PSOFeatureSelector(**common_kwargs)
        mask, best_fit, history = selector.run(
            swarm_size=args.swarm_size, iterations=args.iterations,
            w=args.w_inertia, c1=args.c1, c2=args.c2,
            mutation_rate=args.mutation_rate,
            stagnation_limit=args.stagnation_limit,
            reinit_fraction=args.reinit_fraction,
        )
        title = "PSO - Convergence curve"
    search_time = time.time() - t0

    logger.info(
        "Hoàn tất tìm kiếm (%s): %d đặc trưng được chọn / %d, best fitness = %.5f, thời gian = %.1fs, số lần đánh giá fitness = %d",
        args.method.upper(), int(mask.sum()), selector.n_features, best_fit, search_time, selector.n_evaluations,
    )

    # Đánh giá cuối cùng trên toàn bộ dữ liệu thật (dùng eval_classifier - có thể khác
    # classifier dùng để tìm kiếm, ví dụ tìm bằng DT cho nhanh nhưng đánh giá bằng RF/XGBoost)
    logger.info("Đang đánh giá lại trên toàn bộ tập train/test thật (classifier: %s)...", eval_classifier)
    result = evaluate_on_full_data(X_train, X_test, y_train, y_test, mask, eval_classifier)
    result.update({
        "method": args.method,
        "search_classifier": args.classifier,
        "classifier": eval_classifier,
        "best_fitness_on_subsample": best_fit,
        "search_time_sec": search_time,
        "n_fitness_evaluations": selector.n_evaluations,
    })

    # So sánh với baseline dùng toàn bộ đặc trưng (full features)
    full_mask = np.ones(selector.n_features, dtype=np.uint8)
    logger.info("Đang đánh giá baseline (không Feature Selection) để so sánh...")
    baseline = evaluate_on_full_data(X_train, X_test, y_train, y_test, full_mask, eval_classifier)

    # Lưu kết quả
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
        logger.warning("Không vẽ được convergence curve (matplotlib có thể chưa cài): %s", e)

    logger.info("=" * 70)
    logger.info(
        "KẾT QUẢ SO SÁNH | Full features: %d đặc trưng, F1=%.4f, Acc=%.4f, train=%.2fs, infer=%.4fs",
        baseline["n_features_total"], baseline["f1_macro"], baseline["accuracy"],
        baseline["train_time_sec"], baseline["inference_time_sec"],
    )
    logger.info(
        "KẾT QUẢ SO SÁNH | %s        : %d đặc trưng, F1=%.4f, Acc=%.4f, train=%.2fs, infer=%.4fs",
        args.method.upper(), result["n_features_selected"], result["f1_macro"], result["accuracy"],
        result["train_time_sec"], result["inference_time_sec"],
    )
    logger.info("Đã lưu toàn bộ kết quả vào: %s", args.output_dir)


if __name__ == "__main__":
    main()
