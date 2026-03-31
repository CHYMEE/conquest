"""Phase 1: Centralized LSTM-VAE (SGD) classical ceiling via controlled hyperparameter sweep + 5-fold CV."""

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def set_global_determinism(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@dataclass(frozen=True)
class TrialConfig:
    latent_dim: int
    lstm_hidden: int
    lstm_layers: int
    dropout: float
    lr: float
    batch_size: int
    threshold_method: str


@dataclass
class FoldArtifacts:
    fold_index: int
    threshold: float
    epochs_trained: int
    train_loss_curve: List[float]
    val_loss_curve: List[float]
    train_total_loss_curve: List[float]
    val_total_loss_curve: List[float]
    train_kl_curve: List[float]
    val_kl_curve: List[float]


def _drop_known_non_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["Flow ID", "Timestamp", "Connection Type"]
    present = [c for c in drop_cols if c in df.columns]
    if present:
        df = df.drop(columns=present)
    return df


def load_binary_dataset(csv_path: str, label_col: Optional[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    if label_col is None:
        if "Label" in df.columns:
            label_col = "Label"
        elif "label" in df.columns:
            label_col = "label"
        elif "binary_label" in df.columns:
            label_col = "binary_label"
        else:
            raise ValueError("Could not infer label column. Please pass --label-col.")

    y_raw = df[label_col].astype(str)
    X_df = df.drop(columns=[label_col])
    X_df = _drop_known_non_features(X_df)

    for col in X_df.columns:
        if X_df[col].dtype == "object":
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))

    X = X_df.apply(pd.to_numeric, errors="coerce").fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

    y_bin = np.array([0 if v.lower() == "benign" else 1 for v in y_raw.values], dtype=np.int64)
    class_names = ["Benign", "anomaly"]
    return X, y_bin, class_names


def subsample_binary(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    max_per_class: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_per_class is None and max_samples is None:
        return X, y, np.arange(len(y), dtype=np.int64)

    rng = np.random.RandomState(seed)
    idx_all = np.arange(len(y), dtype=np.int64)
    kept_idx = idx_all

    if max_per_class is not None:
        kept: List[int] = []
        for cls in [0, 1]:
            cls_idx = kept_idx[y[kept_idx] == cls]
            if len(cls_idx) == 0:
                continue
            if len(cls_idx) > max_per_class:
                cls_idx = rng.choice(cls_idx, size=max_per_class, replace=False)
            kept.extend(cls_idx.tolist())
        kept_idx = np.asarray(kept, dtype=np.int64)
        rng.shuffle(kept_idx)

    if max_samples is not None and len(kept_idx) > max_samples:
        kept_idx = rng.choice(kept_idx, size=max_samples, replace=False)

    kept_idx = np.asarray(kept_idx, dtype=np.int64)
    return X[kept_idx], y[kept_idx], kept_idx


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    time_steps: int,
    order_by: Optional[np.ndarray] = None,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    if order_by is not None:
        order = np.argsort(order_by)
        X = X[order]
        y = y[order]

    if time_steps <= 1:
        return X.reshape((-1, 1, X.shape[1])), y.copy()

    stride = int(max(1, stride))

    Xs: List[np.ndarray] = []
    ys: List[int] = []
    for i in range(0, len(X) - time_steps + 1, stride):
        Xs.append(X[i : i + time_steps])
        ys.append(int(y[i + time_steps - 1]))
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


class LSTMVAE(tf.keras.Model):
    def __init__(self, time_steps: int, n_features: int, latent_dim: int, lstm_hidden: int, lstm_layers: int, dropout: float):
        super().__init__()
        self.time_steps = time_steps
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.beta = tf.Variable(1.0, trainable=False, dtype=tf.float32, name="kl_beta")

        self.encoder_lstms: List[tf.keras.layers.Layer] = []
        for layer_idx in range(lstm_layers):
            self.encoder_lstms.append(
                tf.keras.layers.LSTM(
                    lstm_hidden,
                    return_sequences=(layer_idx < lstm_layers - 1),
                    dropout=dropout,
                )
            )

        self.z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")

        self.decoder_repeat = tf.keras.layers.RepeatVector(time_steps)
        self.decoder_lstms: List[tf.keras.layers.Layer] = []
        for _ in range(lstm_layers):
            self.decoder_lstms.append(
                tf.keras.layers.LSTM(
                    lstm_hidden,
                    return_sequences=True,
                    dropout=dropout,
                )
            )

        self.decoder_out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def encode(self, x: tf.Tensor, training: bool) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        h = x
        for lstm in self.encoder_lstms:
            h = lstm(h, training=training)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        eps = tf.random.normal(shape=tf.shape(z_mean), dtype=z_mean.dtype)
        scale = tf.cast(tf.exp(0.5 * tf.cast(z_log_var, tf.float32)), z_mean.dtype)
        z = z_mean + scale * eps
        return z_mean, z_log_var, z

    def decode(self, z: tf.Tensor, training: bool) -> tf.Tensor:
        h = self.decoder_repeat(z)
        for lstm in self.decoder_lstms:
            h = lstm(h, training=training)
        return self.decoder_out(h, training=training)

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encode(inputs, training=training)
        x_hat = self.decode(z, training=training)
        return x_hat, z_mean, z_log_var

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            x_hat, z_mean, z_log_var = self(x, training=True)
            x_f = tf.cast(x, tf.float32)
            x_hat_f = tf.cast(x_hat, tf.float32)
            recon = tf.reduce_mean(tf.reduce_sum(tf.square(x_f - x_hat_f), axis=[1, 2]))
            z_mean_f = tf.cast(z_mean, tf.float32)
            z_log_var_f = tf.cast(z_log_var, tf.float32)
            kl = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_log_var_f - tf.square(z_mean_f) - tf.exp(z_log_var_f), axis=1))
            total = recon + tf.cast(self.beta, tf.float32) * kl
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x = data
        x_hat, z_mean, z_log_var = self(x, training=False)
        x_f = tf.cast(x, tf.float32)
        x_hat_f = tf.cast(x_hat, tf.float32)
        recon = tf.reduce_mean(tf.reduce_sum(tf.square(x_f - x_hat_f), axis=[1, 2]))
        z_mean_f = tf.cast(z_mean, tf.float32)
        z_log_var_f = tf.cast(z_log_var, tf.float32)
        kl = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_log_var_f - tf.square(z_mean_f) - tf.exp(z_log_var_f), axis=1))
        total = recon + tf.cast(self.beta, tf.float32) * kl

        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}


def reconstruction_errors(model: LSTMVAE, X: np.ndarray, batch_size: int) -> np.ndarray:
    errs: List[np.ndarray] = []
    for i in range(0, len(X), batch_size):
        x_b = X[i : i + batch_size]
        x_hat, _, _ = model(x_b, training=False)
        x_f = tf.cast(x_b, tf.float32)
        x_hat_f = tf.cast(x_hat, tf.float32)
        e = tf.reduce_sum(tf.square(x_f - x_hat_f), axis=[1, 2]).numpy()
        # Guard against NaN/Inf from numerical issues.
        # Use NaN here, then sanitize to a very large finite number for downstream metrics.
        e = np.where(np.isfinite(e), e, np.nan)
        errs.append(e)
    out = np.concatenate(errs, axis=0)
    # Replace non-finite with a large finite value so thresholding/ROC cannot crash.
    # This effectively treats them as extremely anomalous.
    if out.size:
        finite = out[np.isfinite(out)]
        fallback = float(np.nanmax(finite)) if finite.size else 1e12
        out = np.nan_to_num(out, nan=fallback, posinf=fallback, neginf=fallback)
    return out


def _finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.size == 0:
        return x
    return x[np.isfinite(x)]


def threshold_from_method(
    method: str,
    train_err_norm: np.ndarray,
    val_err: np.ndarray,
    val_y: np.ndarray,
) -> float:
    train_err_norm_f = _finite_1d(train_err_norm)
    # If the model produced no finite errors at all, fall back to +inf so everything is predicted benign.
    # This keeps the sweep running and records the configuration as poor (low DR).
    if train_err_norm_f.size == 0:
        return float("inf")

    if method.startswith("percentile_"):
        p = float(method.split("_", 1)[1])
        return float(np.percentile(train_err_norm_f, p))

    if method.startswith("mean_std_"):
        parts = method.split("_")
        if len(parts) != 3:
            raise ValueError("mean+K*std method must be formatted as mean_std_<K>")
        k = float(parts[2])
        return float(np.mean(train_err_norm_f) + k * np.std(train_err_norm_f))

    if method == "youden":
        val_err = np.asarray(val_err)
        val_y = np.asarray(val_y)
        mask = np.isfinite(val_err)
        val_err_f = val_err[mask]
        val_y_f = val_y[mask]

        # ROC requires at least one sample from each class.
        if val_err_f.size == 0 or np.unique(val_y_f).size < 2:
            return float(np.percentile(train_err_norm_f, 95.0))

        fpr, tpr, thr = roc_curve(val_y_f, val_err_f)
        j = tpr - fpr
        idx = int(np.argmax(j))
        candidate = float(thr[idx])
        if not np.isfinite(candidate):
            return float(np.percentile(train_err_norm_f, 95.0))
        return candidate

    raise ValueError(f"Unknown threshold method: {method}")


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = (2 * prec * rec / (prec * 1.0 + rec + 1e-12)) if (prec + rec) > 0 else 0.0

    fpr = fp / (fp + tn + 1e-12)
    fnr = fn / (fn + tp + 1e-12)

    try:
        roc_auc = roc_auc_score(y_true, score)
    except Exception:
        roc_auc = float("nan")

    try:
        pr_p, pr_r, _ = precision_recall_curve(y_true, score)
        pr_auc = auc(pr_r, pr_p)
    except Exception:
        pr_auc = float("nan")

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "dr": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }

    # Convenience percentage fields (requested as Accuracy (%), etc.)
    metrics["accuracy_pct"] = metrics["accuracy"] * 100.0
    metrics["precision_pct"] = metrics["precision"] * 100.0
    metrics["recall_pct"] = metrics["recall"] * 100.0
    metrics["dr_pct"] = metrics["dr"] * 100.0
    metrics["f1_pct"] = metrics["f1"] * 100.0
    return metrics


def loss_curve_stability(loss_curve: List[float], tail: int = 10) -> float:
    if len(loss_curve) < 3:
        return float("nan")
    tail_vals = np.asarray(loss_curve[-tail:], dtype=np.float64)
    diffs = np.diff(tail_vals)
    return float(np.std(diffs))


def write_json(path: str, payload: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_single_trial(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    cfg: TrialConfig,
    seed: int,
    epochs: int,
    patience: int,
    beta_warmup_epochs: int,
    cv_folds: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    cv_folds = int(max(2, cv_folds))
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    fold_rows: List[Dict[str, Any]] = []
    fold_metrics: List[Dict[str, float]] = []
    fold_artifacts: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_seq, y_seq)):
        set_global_determinism(seed + fold_idx)

        X_tr, X_te = X_seq[train_idx], X_seq[test_idx]
        y_tr, y_te = y_seq[train_idx], y_seq[test_idx]

        tr_train_idx, tr_val_idx = train_test_split(
            np.arange(len(y_tr)),
            test_size=0.2,
            random_state=seed + fold_idx,
            stratify=y_tr,
        )

        X_train_all, y_train_all = X_tr[tr_train_idx], y_tr[tr_train_idx]
        X_val_all, y_val_all = X_tr[tr_val_idx], y_tr[tr_val_idx]

        X_train_norm = X_train_all[y_train_all == 0]
        X_val_norm = X_val_all[y_val_all == 0]

        if len(X_train_norm) < max(cfg.batch_size, 32):
            raise ValueError("Not enough normal samples in training fold for stable VAE training.")

        time_steps = X_seq.shape[1]
        n_features = X_seq.shape[2]

        model = LSTMVAE(
            time_steps=time_steps,
            n_features=n_features,
            latent_dim=cfg.latent_dim,
            lstm_hidden=cfg.lstm_hidden,
            lstm_layers=cfg.lstm_layers,
            dropout=cfg.dropout,
        )

        opt = tf.keras.optimizers.SGD(learning_rate=cfg.lr, momentum=0.9, nesterov=True)
        model.compile(optimizer=opt)

        class _BetaWarmup(tf.keras.callbacks.Callback):
            def __init__(self, warmup_epochs: int):
                super().__init__()
                self.warmup_epochs = int(max(0, warmup_epochs))

            def on_epoch_begin(self, epoch, logs=None):
                if self.warmup_epochs <= 0:
                    self.model.beta.assign(1.0)
                    return
                t = float(min(max(epoch, 0), self.warmup_epochs)) / float(self.warmup_epochs)
                self.model.beta.assign(t)

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_total_loss",
            patience=patience,
            restore_best_weights=True,
            mode="min",
        )

        train_ds = tf.data.Dataset.from_tensor_slices(X_train_norm).batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = None
        if len(X_val_norm) > 0:
            val_ds = tf.data.Dataset.from_tensor_slices(X_val_norm).batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

        hist = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=0,
            callbacks=[es, _BetaWarmup(beta_warmup_epochs)],
        )

        epochs_trained = len(hist.history.get("total_loss", []))

        train_err_norm = reconstruction_errors(model, X_train_norm, batch_size=cfg.batch_size)
        val_err = reconstruction_errors(model, X_val_all, batch_size=cfg.batch_size)

        threshold = threshold_from_method(
            method=cfg.threshold_method,
            train_err_norm=train_err_norm,
            val_err=val_err,
            val_y=y_val_all,
        )

        test_err = reconstruction_errors(model, X_te, batch_size=cfg.batch_size)
        y_pred = (test_err > threshold).astype(np.int64)

        m = compute_binary_metrics(y_true=y_te, y_pred=y_pred, score=test_err)

        train_total_curve = [float(v) for v in hist.history.get("total_loss", [])]
        train_recon_curve = [float(v) for v in hist.history.get("recon_loss", [])]
        train_kl_curve = [float(v) for v in hist.history.get("kl_loss", [])]
        val_total_curve = [float(v) for v in hist.history.get("val_total_loss", [])]
        val_recon_curve = [float(v) for v in hist.history.get("val_recon_loss", [])]
        val_kl_curve = [float(v) for v in hist.history.get("val_kl_loss", [])]

        m.update(
            {
                "mre": float(np.mean(test_err)),
                "recon_var": float(np.var(test_err)),
                "threshold": float(threshold),
                "epochs": float(epochs_trained),
                "training_loss_curve_stability": loss_curve_stability(train_total_curve),
                "train_recon_last": float(train_recon_curve[-1]) if train_recon_curve else float("nan"),
                "train_kl_last": float(train_kl_curve[-1]) if train_kl_curve else float("nan"),
                "train_total_last": float(train_total_curve[-1]) if train_total_curve else float("nan"),
                "val_recon_last": float(val_recon_curve[-1]) if val_recon_curve else float("nan"),
                "val_kl_last": float(val_kl_curve[-1]) if val_kl_curve else float("nan"),
                "val_total_last": float(val_total_curve[-1]) if val_total_curve else float("nan"),
            }
        )

        fold_metrics.append(m)

        fold_artifacts.append(
            {
                "fold": fold_idx,
                "threshold": float(threshold),
                "epochs_trained": int(epochs_trained),
                "train_total_loss_curve": train_total_curve,
                "train_recon_loss_curve": train_recon_curve,
                "train_kl_loss_curve": train_kl_curve,
                "val_total_loss_curve": val_total_curve,
                "val_recon_loss_curve": val_recon_curve,
                "val_kl_loss_curve": val_kl_curve,
            }
        )

        fold_rows.append(
            {
                **asdict(cfg),
                "fold": fold_idx,
                **m,
            }
        )

    def mean_std(key: str) -> Tuple[float, float]:
        vals = [fm[key] for fm in fold_metrics]
        return float(np.mean(vals)), float(np.std(vals))

    summary: Dict[str, Any] = {**asdict(cfg)}
    for k in [
        "accuracy",
        "precision",
        "recall",
        "dr",
        "f1",
        "fpr",
        "fnr",
        "roc_auc",
        "pr_auc",
        "mre",
        "recon_var",
        "train_kl_last",
        "train_total_last",
        "train_recon_last",
        "val_kl_last",
        "val_total_last",
        "val_recon_last",
        "threshold",
        "epochs",
        "training_loss_curve_stability",
    ]:
        mu, sd = mean_std(k)
        summary[f"{k}_mean"] = mu
        summary[f"{k}_std"] = sd

    # Percentage summaries
    for k in ["accuracy_pct", "precision_pct", "recall_pct", "dr_pct", "f1_pct"]:
        mu, sd = mean_std(k)
        summary[f"{k}_mean"] = mu
        summary[f"{k}_std"] = sd

    summary["seed"] = seed
    return summary, fold_rows, fold_artifacts


def generate_trials(
    mode: str,
    n_trials: int,
    grid: Dict[str, List[Any]],
    seed: int,
) -> List[TrialConfig]:
    if mode == "grid":
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        trials = []
        for combo in product(*values):
            trials.append(TrialConfig(**dict(zip(keys, combo))))
        return trials

    if mode == "random":
        rng = random.Random(seed)
        trials = []
        for _ in range(n_trials):
            trials.append(
                TrialConfig(
                    latent_dim=int(rng.choice(grid["latent_dim"])),
                    lstm_hidden=int(rng.choice(grid["lstm_hidden"])),
                    lstm_layers=int(rng.choice(grid["lstm_layers"])),
                    dropout=float(rng.choice(grid["dropout"])),
                    lr=float(rng.choice(grid["lr"])),
                    batch_size=int(rng.choice(grid["batch_size"])),
                    threshold_method=str(rng.choice(grid["threshold_method"])),
                )
            )
        return trials

    raise ValueError("mode must be 'grid' or 'random'")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="0.ACI-IoT-2023.csv")
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--order-col", default=None)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--beta-warmup-epochs", type=int, default=10)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--select-metric", choices=["f1", "accuracy", "pr_auc", "recall", "dr"], default="f1")
    parser.add_argument("--max-fpr", type=float, default=None)
    parser.add_argument("--mode", choices=["grid", "random"], default="random")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/phase1")
    args = parser.parse_args()

    set_global_determinism(args.seed)

    if args.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")

    if args.quick:
        # Quick mode: aim for usable results fast.
        # Keep deterministic behavior and same output structure.
        if args.mode == "random" and args.n_trials > 8:
            args.n_trials = 8
        if args.epochs > 35:
            args.epochs = 35
        if args.patience > 6:
            args.patience = 6
        if args.time_steps > 20:
            args.time_steps = 20
        if args.stride < 5:
            args.stride = 5
        if args.beta_warmup_epochs > 10:
            args.beta_warmup_epochs = 10
        if args.cv_folds > 3:
            args.cv_folds = 3
        if args.max_samples is None:
            args.max_samples = 120000

    # Load raw so we can optionally use an ordering column before encoding/scaling
    df = pd.read_csv(args.csv)
    label_col = args.label_col
    if label_col is None:
        if "Label" in df.columns:
            label_col = "Label"
        elif "label" in df.columns:
            label_col = "label"
        elif "binary_label" in df.columns:
            label_col = "binary_label"
        else:
            raise ValueError("Could not infer label column. Please pass --label-col.")

    order_by = None
    if args.order_col is not None:
        if args.order_col not in df.columns:
            raise ValueError(f"--order-col '{args.order_col}' not found in CSV columns")
        order_by = df[args.order_col].values

    # Reuse existing loader by writing df back through the same path logic
    X, y, _ = load_binary_dataset(args.csv, label_col)
    X, y, kept_idx = subsample_binary(
        X,
        y,
        seed=args.seed,
        max_per_class=args.max_per_class,
        max_samples=args.max_samples,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    order_sub = order_by[kept_idx] if order_by is not None else None
    X_seq, y_seq = make_sequences(X_scaled, y, time_steps=args.time_steps, order_by=order_sub, stride=args.stride)

    if args.quick:
        grid = {
            "latent_dim": [16, 32],
            "lstm_hidden": [64, 128],
            "lstm_layers": [1, 2],
            "dropout": [0.0, 0.2],
            "lr": [5e-4, 1e-3],
            "batch_size": [64, 128],
            "threshold_method": [
                "percentile_70",
                "percentile_80",
                "percentile_90",
                "percentile_95",
                "mean_std_1.5",
                "mean_std_2",
                "youden",
            ],
        }
    else:
        grid = {
            "latent_dim": [8, 16, 32, 64],
            "lstm_hidden": [32, 64, 128, 256],
            "lstm_layers": [1, 2, 3],
            "dropout": [0.0, 0.2, 0.4],
            "lr": [1e-3, 5e-4, 1e-4, 5e-3],
            "batch_size": [32, 64, 128],
            "threshold_method": [
                "percentile_90",
                "percentile_92",
                "percentile_95",
                "percentile_99",
                "percentile_99.5",
                "mean_std_2",
                "mean_std_3",
                "mean_std_4",
                "youden",
            ],
        }

    trials = generate_trials(args.mode, args.n_trials, grid, seed=args.seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.output_dir, run_id)
    ensure_dir(out_root)

    write_json(os.path.join(out_root, "sweep_space.json"), grid)
    write_json(
        os.path.join(out_root, "run_args.json"),
        {
            "csv": args.csv,
            "label_col": args.label_col,
            "order_col": args.order_col,
            "time_steps": args.time_steps,
            "max_per_class": args.max_per_class,
            "max_samples": args.max_samples,
            "mode": args.mode,
            "n_trials": args.n_trials,
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
        },
    )

    summaries: List[Dict[str, Any]] = []
    fold_rows_all: List[Dict[str, Any]] = []

    best_idx = None
    best_score = -math.inf

    for idx, cfg in enumerate(trials):
        trial_dir = os.path.join(out_root, f"trial_{idx:04d}")
        ensure_dir(trial_dir)

        summary, fold_rows, fold_artifacts = run_single_trial(
            X_seq=X_seq,
            y_seq=y_seq,
            cfg=cfg,
            seed=args.seed + idx * 1000,
            epochs=args.epochs,
            patience=args.patience,
            beta_warmup_epochs=args.beta_warmup_epochs,
            cv_folds=args.cv_folds,
        )

        # Save trial summary + fold rows for this trial
        write_json(os.path.join(trial_dir, "config.json"), asdict(cfg))
        write_json(os.path.join(trial_dir, "summary.json"), summary)
        write_csv(os.path.join(trial_dir, "fold_metrics.csv"), fold_rows)
        write_json(os.path.join(trial_dir, "fold_artifacts.json"), fold_artifacts)

        summaries.append(summary)
        fold_rows_all.extend(fold_rows)

        if args.select_metric in ("recall", "dr"):
            score = float(summary.get("recall_mean", float("nan")))
        elif args.select_metric == "accuracy":
            score = float(summary.get("accuracy_mean", float("nan")))
        elif args.select_metric == "pr_auc":
            score = float(summary.get("pr_auc_mean", float("nan")))
        else:
            score = float(summary.get("f1_mean", float("nan")))

        max_fpr_ok = True
        if args.max_fpr is not None:
            max_fpr_ok = float(summary.get("fpr_mean", float("inf"))) <= float(args.max_fpr)

        if max_fpr_ok and (not math.isnan(score)) and score > best_score:
            best_score = score
            best_idx = idx

        write_csv(os.path.join(out_root, "trials_summary.csv"), summaries)
        write_csv(os.path.join(out_root, "fold_metrics.csv"), fold_rows_all)

    if best_idx is not None:
        best_cfg = trials[best_idx]
        best_summary = summaries[best_idx]
        write_json(os.path.join(out_root, "best_config.json"), asdict(best_cfg))
        write_json(os.path.join(out_root, "best_metrics.json"), best_summary)

    write_json(os.path.join(out_root, "scaler.json"), {"type": "StandardScaler"})


if __name__ == "__main__":
    main()
