"""Quick rerun + tuning for 2-qubit QPSO + LSTM(+latent) + SGD."""

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

from qubit_ablation_study import (
    LSTM_VAE_Model,
    MetricsCalculator,
    QuantumInspiredFeatureSelector,
    SmartSubsampler,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class TrialCfg:
    lstm_units: int
    attention_heads: int
    dropout_rate: float
    latent_dim: int
    learning_rate: float
    batch_size: int
    num_features_to_select: int


def load_supervised_data(
    csv_path: str,
    subsample_size: int,
    seed: int,
    task: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    target_col = "Label" if "Label" in df.columns else "label"

    drop_cols = ["Connection Type", "Flow ID", "Timestamp"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    y_raw = df[target_col].astype(str).values
    if task == "binary":
        # Map all non-benign labels to "Attack".
        y_raw = np.where(y_raw == "Benign", "Benign", "Attack")
    X_df = df.drop(columns=[target_col])

    for col in X_df.columns:
        if X_df[col].dtype == "object":
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))

    X = np.nan_to_num(X_df.values, nan=0.0, posinf=0.0, neginf=0.0)

    ss = SmartSubsampler(target_size=subsample_size)
    X_sub, y_sub = ss.subsample(X, y_raw)

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y_sub)
    class_names = list(label_encoder.classes_)

    return X_sub.astype(np.float32), y_enc.astype(np.int64), class_names


def qpso_select_features(
    X: np.ndarray,
    y_raw_for_qpso: np.ndarray,
    num_qubits: int,
    num_particles: int,
    iterations: int,
    num_features_to_select: int,
) -> Tuple[np.ndarray, float, float, int]:
    qpso = QuantumInspiredFeatureSelector(
        num_qubits=num_qubits,
        num_particles=num_particles,
        iterations=iterations,
        num_features_to_select=num_features_to_select,
    )
    X_selected = qpso.fit_transform(X, y_raw_for_qpso)
    return X_selected, qpso.computation_time, qpso.memory_usage, len(qpso.best_features)


def variance_select_features(
    X: np.ndarray,
    num_features_to_select: int,
) -> Tuple[np.ndarray, float, float, int]:
    if X.ndim != 2:
        raise ValueError("Expected 2D feature matrix X for variance-based selection")
    n_features = X.shape[1]
    k = int(min(max(1, num_features_to_select), n_features))
    variances = np.var(X, axis=0)
    idx = np.argsort(variances)[-k:]
    X_selected = X[:, idx]
    return X_selected, 0.0, 0.0, int(k)


def create_sequences(X: np.ndarray, y_onehot: np.ndarray, time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i : i + time_steps])
        ys.append(y_onehot[i + time_steps - 1])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def sample_trials(seed: int, n_trials: int, feature_choices: List[int]) -> List[TrialCfg]:
    rng = random.Random(seed)
    trials: List[TrialCfg] = []
    for _ in range(n_trials):
        lstm_units = rng.choice([128, 192, 256])
        attention_heads = rng.choice([2, 4, 8])
        if lstm_units % attention_heads != 0:
            attention_heads = 4
        trials.append(
            TrialCfg(
                lstm_units=int(lstm_units),
                attention_heads=int(attention_heads),
                dropout_rate=float(rng.choice([0.2, 0.3, 0.4, 0.5])),
                latent_dim=int(rng.choice([16, 32, 64])),
                learning_rate=float(rng.choice([5e-4, 1e-3, 2e-3])),
                batch_size=int(rng.choice([32, 64, 128])),
                num_features_to_select=int(rng.choice(feature_choices)),
            )
        )
    return trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="0.ACI-IoT-2023.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    parser.add_argument("--subsample-size", type=int, default=50000)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--qpso-particles", type=int, default=20)
    parser.add_argument("--qpso-iterations", type=int, default=30)
    parser.add_argument("--feature-method", choices=["qpso", "variance"], default="qpso")
    parser.add_argument("--feature-choices", default="20,30")
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/quick_tune")
    args = parser.parse_args()

    set_seed(args.seed)

    feature_choices = [int(x.strip()) for x in args.feature_choices.split(",") if x.strip()]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.output_dir, run_id)
    ensure_dir(out_root)
    write_json(os.path.join(out_root, "run_args.json"), vars(args))

    X_sub, y_enc, class_names = load_supervised_data(
        args.csv,
        args.subsample_size,
        args.seed,
        task=args.task,
    )
    num_classes = len(class_names)
    y_onehot = to_categorical(y_enc, num_classes=num_classes).astype(np.float32)

    # For QPSO fitness, use the original labels as strings/ints; we pass encoded ints.
    y_for_qpso = y_enc

    trials = sample_trials(args.seed, args.n_trials, feature_choices)

    # Cache feature selections per feature count to avoid rerunning QPSO for every trial.
    feature_cache: Dict[int, Dict[str, Any]] = {}

    rows: List[Dict[str, Any]] = []
    best = {"test_accuracy": -1.0, "trial": None, "metrics": None}

    for idx, cfg in enumerate(trials):
        set_seed(args.seed + idx)
        tf.keras.backend.clear_session()

        if cfg.num_features_to_select not in feature_cache:
            t0 = time.time()
            if args.feature_method == "qpso":
                X_sel, qpso_time, qpso_mem, n_selected = qpso_select_features(
                    X_sub,
                    y_for_qpso,
                    num_qubits=args.qubits,
                    num_particles=args.qpso_particles,
                    iterations=args.qpso_iterations,
                    num_features_to_select=cfg.num_features_to_select,
                )
            else:
                X_sel, qpso_time, qpso_mem, n_selected = variance_select_features(
                    X_sub,
                    num_features_to_select=cfg.num_features_to_select,
                )

            # Build sequences and scale once for this feature set.
            X_seq, y_seq = create_sequences(X_sel, y_onehot, args.time_steps)
            n_features = X_seq.shape[2]
            scaler = StandardScaler()
            X_flat = X_seq.reshape(-1, n_features)
            X_scaled = scaler.fit_transform(X_flat).reshape(X_seq.shape).astype(np.float32)

            # Train/val/test split once.
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_scaled,
                y_seq,
                test_size=0.3,
                random_state=args.seed,
                stratify=np.argmax(y_seq, axis=1),
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=0.5,
                random_state=args.seed,
                stratify=np.argmax(y_temp, axis=1),
            )

            feature_cache[cfg.num_features_to_select] = {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
                "qpso_time": float(qpso_time),
                "qpso_mem": float(qpso_mem),
                "n_selected": int(n_selected),
                "prep_time": float(time.time() - t0),
                "n_features": int(n_features),
            }

        cache = feature_cache[cfg.num_features_to_select]

        input_shape = (args.time_steps, int(cache["n_features"]))
        model = LSTM_VAE_Model(
            input_shape=input_shape,
            num_classes=num_classes,
            lstm_units=cfg.lstm_units,
            attention_heads=cfg.attention_heads,
            dropout_rate=cfg.dropout_rate,
            latent_dim=cfg.latent_dim,
        )
        model.build()
        model.compile_with_sgd(learning_rate=cfg.learning_rate)

        # Faster early stopping than the base script.
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=args.patience,
                restore_best_weights=True,
                mode="max",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(1, args.patience // 2),
                min_lr=1e-6,
            ),
        ]

        t_train = time.time()
        hist = model.model.fit(
            cache["X_train"],
            cache["y_train"],
            validation_data=(cache["X_val"], cache["y_val"]),
            epochs=args.epochs,
            batch_size=cfg.batch_size,
            verbose=0,
            callbacks=callbacks,
        )
        training_time = time.time() - t_train

        y_pred, y_pred_prob = model.predict(cache["X_test"])
        y_true = np.argmax(cache["y_test"], axis=1)

        metrics_calc = MetricsCalculator(class_names)
        metrics = metrics_calc.calculate_all_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_prob=y_pred_prob,
            y_true_onehot=cache["y_test"],
            training_time=training_time,
            inference_time=model.inference_time,
            qpso_time=cache["qpso_time"],
            memory_mb=cache["qpso_mem"] + model.memory_usage,
        )

        val_best = float(np.max(hist.history.get("val_accuracy", [float("nan")])))

        row = {
            "trial": idx,
            **asdict(cfg),
            "qubits": args.qubits,
            "feature_method": args.feature_method,
            "subsample_size": args.subsample_size,
            "time_steps": args.time_steps,
            "n_features_selected": int(cache["n_selected"]),
            "val_best_accuracy": val_best,
            "test_accuracy": float(metrics["accuracy"]),
            "test_precision": float(metrics["precision"]),
            "test_recall": float(metrics["recall"]),
            "test_f1": float(metrics["f1_score"]),
            "test_mse": float(metrics["mse"]),
            "epochs_trained": int(len(hist.history.get("loss", []))),
        }
        rows.append(row)
        write_csv(os.path.join(out_root, "trials.csv"), rows)

        if row["test_accuracy"] > best["test_accuracy"]:
            best = {"test_accuracy": row["test_accuracy"], "trial": row, "metrics": metrics}
            write_json(os.path.join(out_root, "best_config.json"), row)
            write_json(os.path.join(out_root, "best_metrics.json"), metrics)

    write_json(os.path.join(out_root, "feature_cache_summary.json"), {
        str(k): {"n_selected": v["n_selected"], "prep_time": v["prep_time"], "n_features": v["n_features"]}
        for k, v in feature_cache.items()
    })


if __name__ == "__main__":
    main()
