# -*- coding: utf-8 -*-

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
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from tensorflow import keras
from tensorflow.keras.optimizers import SGD

from run_quick_tune_qpso_lstm_sgd import (
    create_sequences,
    load_supervised_data,
    qpso_select_features,
    variance_select_features,
)

from qubit_ablation_study import LSTM_VAE_Model


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _write_json(p: str, payload: Any) -> None:
    _ensure_dir(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(p: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    _ensure_dir(os.path.dirname(p))
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@dataclass(frozen=True)
class FLConfig:
    num_clients: int
    num_rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    client_fraction: float
    fedprox_mu: float


@dataclass(frozen=True)
class DataConfig:
    csv_path: str
    task: str
    subsample_size: int
    time_steps: int
    feature_method: str
    qubits: int
    qpso_particles: int
    qpso_iterations: int
    num_features_to_select: int


class HeterogeneityType:
    IID = "iid"
    NON_IID_LABEL = "non_iid_label"
    NON_IID_QUANTITY = "non_iid_quantity"
    DIRICHLET = "dirichlet"


class AggregatorType:
    FEDAVG = "fedavg"
    SCAFFOLD_LIKE = "scaffold_like"
    FLAME = "flame"
    FEDPROX = "fedprox"


def _partition_clients(
    y_labels: np.ndarray,
    num_clients: int,
    heterogeneity: str,
    seed: int,
    dirichlet_alpha: float,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(len(y_labels))
    idx = np.arange(n)

    if heterogeneity == HeterogeneityType.IID:
        rng.shuffle(idx)
        splits = np.array_split(idx, num_clients)
        return [s for s in splits if len(s) > 0]

    if heterogeneity == HeterogeneityType.NON_IID_LABEL:
        # Sort by label => clients get label-skewed shards
        sorted_idx = np.argsort(y_labels)
        splits = np.array_split(sorted_idx, num_clients)
        return [s for s in splits if len(s) > 0]

    if heterogeneity == HeterogeneityType.NON_IID_QUANTITY:
        rng.shuffle(idx)
        props = rng.dirichlet(np.ones(num_clients) * 0.5)
        split_pts = (np.cumsum(props) * n).astype(int)[:-1]
        splits = np.split(idx, split_pts)
        return [s for s in splits if len(s) > 0][:num_clients]

    # Dirichlet label distribution across clients
    classes = np.unique(y_labels)
    num_classes = int(len(classes))
    label_dist = rng.dirichlet([dirichlet_alpha] * num_clients, size=num_classes)
    client_idx: List[List[int]] = [[] for _ in range(num_clients)]

    for ci, c in enumerate(classes):
        c_idx = np.where(y_labels == c)[0]
        rng.shuffle(c_idx)
        frac = label_dist[ci] / np.sum(label_dist[ci])
        split_pts = (np.cumsum(frac) * len(c_idx)).astype(int)[:-1]
        chunks = np.split(c_idx, split_pts)
        for k, chunk in enumerate(chunks):
            client_idx[k].extend(chunk.tolist())

    out = []
    for ci in client_idx:
        if len(ci) == 0:
            # fallback: small random sample
            out.append(rng.choice(n, size=min(200, n), replace=False))
        else:
            out.append(np.asarray(ci, dtype=int))
    return out[:num_clients]


def _evaluate_binary(model: keras.Model, X_test: np.ndarray, y_test_onehot: np.ndarray) -> Dict[str, Any]:
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test_onehot, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # For binary task, compute AUC on Attack (class 1) when possible
    auc_attack = 0.0
    try:
        if y_prob.shape[1] >= 2:
            auc_attack = float(roc_auc_score((y_true == 1).astype(int), y_prob[:, 1]))
    except Exception:
        auc_attack = 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc_attack": auc_attack,
        "confusion_matrix": cm.tolist(),
    }


def _fit_local_model(
    global_weights: List[np.ndarray],
    input_shape: Tuple[int, int],
    num_classes: int,
    fl_cfg: FLConfig,
    X_c: np.ndarray,
    y_c: np.ndarray,
    use_fedprox: bool,
) -> Tuple[List[np.ndarray], int]:
    model_obj = LSTM_VAE_Model(
        input_shape=input_shape,
        num_classes=num_classes,
        lstm_units=256,
        attention_heads=8,
        dropout_rate=0.4,
        latent_dim=64,
    )
    model_obj.build()
    model_obj.compile_with_sgd(learning_rate=fl_cfg.learning_rate)
    local = model_obj.model
    local.set_weights(global_weights)

    if not use_fedprox or float(fl_cfg.fedprox_mu) <= 0.0:
        local.fit(X_c, y_c, epochs=fl_cfg.local_epochs, batch_size=fl_cfg.batch_size, verbose=0)
    else:
        # FedProx: minimize local loss + (mu/2) * ||w - w_global||^2
        mu = float(fl_cfg.fedprox_mu)
        # Use a frozen copy of the *trainable* variables only (global_weights includes non-trainable too)
        global_tensors = [tf.stop_gradient(tf.identity(v)) for v in local.trainable_variables]
        opt = SGD(learning_rate=fl_cfg.learning_rate)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()

        ds = tf.data.Dataset.from_tensor_slices((X_c, y_c))
        ds = ds.batch(fl_cfg.batch_size, drop_remainder=False)

        for _ in range(int(fl_cfg.local_epochs)):
            for xb, yb in ds:
                with tf.GradientTape() as tape:
                    yhat = local(xb, training=True)
                    base_loss = loss_fn(yb, yhat)
                    prox = tf.add_n(
                        [tf.reduce_sum(tf.square(v - g)) for v, g in zip(local.trainable_variables, global_tensors)]
                    )
                    loss = base_loss + (mu / 2.0) * prox
                grads = tape.gradient(loss, local.trainable_variables)
                opt.apply_gradients(zip(grads, local.trainable_variables))

    weights = local.get_weights()
    keras.backend.clear_session()
    return weights, int(len(y_c))


def _fedavg(client_weights: List[List[np.ndarray]], client_sizes: List[int]) -> List[np.ndarray]:
    total = float(sum(client_sizes))
    out: List[np.ndarray] = []
    for layer_idx in range(len(client_weights[0])):
        layer = np.zeros_like(client_weights[0][layer_idx])
        for w, s in zip(client_weights, client_sizes):
            layer += (float(s) / total) * w[layer_idx]
        out.append(layer)
    return out


def _scaffold_like(
    prev_global_weights: Optional[List[np.ndarray]],
    client_weights: List[List[np.ndarray]],
    client_sizes: List[int],
    momentum: float = 0.9,
) -> List[np.ndarray]:
    avg = _fedavg(client_weights, client_sizes)
    if prev_global_weights is None:
        return avg
    out: List[np.ndarray] = []
    for w_prev, w_avg in zip(prev_global_weights, avg):
        out.append(momentum * w_prev + (1.0 - momentum) * w_avg)
    return out


def _flame(
    client_weights: List[List[np.ndarray]],
    client_sizes: List[int],
    drop_percentile: float = 25.0,
) -> List[np.ndarray]:
    # Flatten weights and compute cosine-similarity trust score
    flats = [np.concatenate([w.flatten() for w in ws]) for ws in client_weights]
    n = len(flats)
    if n <= 2:
        return _fedavg(client_weights, client_sizes)
    flats = np.asarray(flats)
    flats = flats / (np.linalg.norm(flats, axis=1, keepdims=True) + 1e-12)
    sims = flats @ flats.T
    np.fill_diagonal(sims, 0.0)
    trust = np.mean(sims, axis=1)
    thr = np.percentile(trust, drop_percentile)
    mask = trust >= thr
    trusted_w = [w for w, m in zip(client_weights, mask) if m]
    trusted_s = [s for s, m in zip(client_sizes, mask) if m]
    if len(trusted_w) == 0:
        return _fedavg(client_weights, client_sizes)
    return _fedavg(trusted_w, trusted_s)


def run_federated(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_shape: Tuple[int, int],
    num_classes: int,
    fl_cfg: FLConfig,
    aggregator: str,
    heterogeneity: str,
    seed: int,
    dirichlet_alpha: float,
) -> Dict[str, Any]:
    set_seed(seed)

    print(
        f"\n{'='*60}\n"
        f"FL sweep: aggregator={aggregator}, heterogeneity={heterogeneity}, dirichlet_alpha={dirichlet_alpha}\n"
        f"clients={fl_cfg.num_clients}, rounds={fl_cfg.num_rounds}, local_epochs={fl_cfg.local_epochs}, "
        f"client_fraction={fl_cfg.client_fraction}\n"
        f"{'='*60}",
        flush=True,
    )

    # Initialize global model (matches Phase-1 model family)
    model = LSTM_VAE_Model(
        input_shape=input_shape,
        num_classes=num_classes,
        lstm_units=256,
        attention_heads=8,
        dropout_rate=0.4,
        latent_dim=64,
    )
    model.build()
    model.compile_with_sgd(learning_rate=fl_cfg.learning_rate)
    global_model = model.model

    y_labels = np.argmax(y_train, axis=1)
    client_splits = _partition_clients(y_labels, fl_cfg.num_clients, heterogeneity, seed=seed, dirichlet_alpha=dirichlet_alpha)

    # Client sampling each round
    n_select = max(1, int(np.ceil(fl_cfg.client_fraction * fl_cfg.num_clients)))

    round_rows: List[Dict[str, Any]] = []
    t0 = time.time()

    for r in range(fl_cfg.num_rounds):
        rng = np.random.default_rng(seed + r)
        chosen = rng.choice(len(client_splits), size=min(n_select, len(client_splits)), replace=False)

        client_weights: List[List[np.ndarray]] = []
        client_sizes: List[int] = []

        global_weights = global_model.get_weights()

        for ci in chosen:
            idx = client_splits[int(ci)]
            X_c = X_train[idx]
            y_c = y_train[idx]
            w, sz = _fit_local_model(
                global_weights=global_weights,
                input_shape=input_shape,
                num_classes=num_classes,
                fl_cfg=fl_cfg,
                X_c=X_c,
                y_c=y_c,
                use_fedprox=(aggregator == AggregatorType.FEDPROX),
            )
            client_weights.append(w)
            client_sizes.append(sz)

        prev_weights = global_model.get_weights()
        if aggregator == AggregatorType.FLAME:
            new_w = _flame(client_weights, client_sizes)
        elif aggregator == AggregatorType.SCAFFOLD_LIKE:
            new_w = _scaffold_like(prev_weights, client_weights, client_sizes)
        else:
            new_w = _fedavg(client_weights, client_sizes)
        global_model.set_weights(new_w)

        m = _evaluate_binary(global_model, X_test, y_test)
        round_rows.append({"round": r + 1, **m})

        print(
            f"Round {r+1}/{fl_cfg.num_rounds}: Acc={m['accuracy']:.4f}, "
            f"Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, F1={m['f1_score']:.4f}",
            flush=True,
        )

    final_metrics = dict(round_rows[-1]) if round_rows else _evaluate_binary(global_model, X_test, y_test)
    total_time = float(time.time() - t0)

    return {
        "aggregator": aggregator,
        "heterogeneity": heterogeneity,
        "dirichlet_alpha": float(dirichlet_alpha),
        "fl_config": asdict(fl_cfg),
        "final_metrics": final_metrics,
        "round_metrics": round_rows,
        "time_sec": total_time,
    }


def main() -> None:
    p = argparse.ArgumentParser()

    # Data pipeline settings (match Phase-1)
    p.add_argument("--csv", default="0.ACI-IoT-2023.csv")
    p.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample-size", type=int, default=80000)
    p.add_argument("--time-steps", type=int, default=10)

    p.add_argument("--feature-method", choices=["qpso", "variance"], default="qpso")
    p.add_argument("--qubits", type=int, default=6)
    p.add_argument("--qpso-particles", type=int, default=20)
    p.add_argument("--qpso-iterations", type=int, default=30)
    p.add_argument("--num-features-to-select", type=int, default=40)

    # Federated settings
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--num-rounds", type=int, default=15)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--client-fraction", type=float, default=1.0)
    p.add_argument("--fedprox-mu", type=float, default=0.001)

    p.add_argument(
        "--aggregators",
        default="fedavg",
        help="Comma-separated: fedavg,scaffold_like,flame,fedprox",
    )

    # Heterogeneity sweep
    p.add_argument(
        "--heterogeneity",
        default="iid,non_iid_label,non_iid_quantity,dirichlet",
        help="Comma-separated: iid,non_iid_label,non_iid_quantity,dirichlet",
    )
    p.add_argument("--dirichlet-alphas", default="0.1,0.3,0.5,1.0")

    p.add_argument("--output-dir", default="outputs/phase2_federated_heterogeneity")

    args = p.parse_args()
    set_seed(args.seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.output_dir, run_id)
    _ensure_dir(out_root)
    _write_json(os.path.join(out_root, "run_args.json"), vars(args))

    X_sub, y_enc, class_names = load_supervised_data(
        args.csv,
        subsample_size=args.subsample_size,
        seed=args.seed,
        task=args.task,
    )
    num_classes = len(class_names)
    y_onehot = keras.utils.to_categorical(y_enc, num_classes=num_classes).astype(np.float32)

    # Feature selection once (fast path, comparable across heterogeneity)
    if args.feature_method == "qpso":
        X_sel, qpso_time, qpso_mem, n_selected = qpso_select_features(
            X_sub,
            y_raw_for_qpso=y_enc,
            num_qubits=args.qubits,
            num_particles=args.qpso_particles,
            iterations=args.qpso_iterations,
            num_features_to_select=args.num_features_to_select,
        )
    else:
        X_sel, qpso_time, qpso_mem, n_selected = variance_select_features(
            X_sub,
            num_features_to_select=args.num_features_to_select,
        )

    X_seq, y_seq = create_sequences(X_sel, y_onehot, args.time_steps)

    # Use the same split logic as Phase-1 by importing train_test_split there is already used.
    # To keep this file self-contained, we just replicate the split quickly here.
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    n_features = int(X_seq.shape[2])
    scaler = StandardScaler()
    X_flat = X_seq.reshape(-1, n_features)
    X_scaled = scaler.fit_transform(X_flat).reshape(X_seq.shape).astype(np.float32)

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

    _write_json(
        os.path.join(out_root, "data_summary.json"),
        {
            "class_names": class_names,
            "num_classes": int(num_classes),
            "subsample_size": int(args.subsample_size),
            "n_selected": int(n_selected),
            "qpso_time_sec": float(qpso_time),
            "qpso_mem_mb": float(qpso_mem),
            "X_train": list(X_train.shape),
            "X_test": list(X_test.shape),
        },
    )

    fl_cfg = FLConfig(
        num_clients=int(args.num_clients),
        num_rounds=int(args.num_rounds),
        local_epochs=int(args.local_epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        client_fraction=float(args.client_fraction),
        fedprox_mu=float(args.fedprox_mu),
    )

    heteros = [h.strip() for h in str(args.heterogeneity).split(",") if h.strip()]
    alphas = [float(a.strip()) for a in str(args.dirichlet_alphas).split(",") if a.strip()]
    aggs = [a.strip() for a in str(args.aggregators).split(",") if a.strip()]

    results: List[Dict[str, Any]] = []

    for agg in aggs:
        for h in heteros:
            if h == HeterogeneityType.DIRICHLET:
                for a in alphas:
                    r = run_federated(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        input_shape=(int(args.time_steps), int(n_features)),
                        num_classes=num_classes,
                        fl_cfg=fl_cfg,
                        aggregator=agg,
                        heterogeneity=h,
                        seed=args.seed,
                        dirichlet_alpha=a,
                    )
                    results.append(r)
                    _write_json(os.path.join(out_root, "results.json"), results)
            else:
                r = run_federated(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    input_shape=(int(args.time_steps), int(n_features)),
                    num_classes=num_classes,
                    fl_cfg=fl_cfg,
                    aggregator=agg,
                    heterogeneity=h,
                    seed=args.seed,
                    dirichlet_alpha=alphas[0] if alphas else 0.5,
                )
                results.append(r)
                _write_json(os.path.join(out_root, "results.json"), results)

    # Flatten summary
    summary_rows: List[Dict[str, Any]] = []
    for r in results:
        fm = r.get("final_metrics", {})
        summary_rows.append(
            {
                "aggregator": r.get("aggregator"),
                "heterogeneity": r.get("heterogeneity"),
                "dirichlet_alpha": r.get("dirichlet_alpha"),
                "accuracy": fm.get("accuracy"),
                "precision": fm.get("precision"),
                "recall": fm.get("recall"),
                "f1_score": fm.get("f1_score"),
                "auc_attack": fm.get("auc_attack"),
                "time_sec": r.get("time_sec"),
            }
        )

    _write_csv(os.path.join(out_root, "summary.csv"), summary_rows)

    # Pick best by F1 then Recall then time
    def _key(x: Dict[str, Any]) -> Tuple[float, float, float]:
        return (
            float(x.get("f1_score") or 0.0),
            float(x.get("recall") or 0.0),
            -float(x.get("time_sec") or 0.0),
        )

    best = max(summary_rows, key=_key) if summary_rows else {}
    _write_json(os.path.join(out_root, "best_strategy.json"), best)


if __name__ == "__main__":
    main()
