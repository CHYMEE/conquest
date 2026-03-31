import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

from qubit_ablation_study import LSTM_VAE_Model, MetricsCalculator, QuantumInspiredFeatureSelector, SmartSubsampler


def set_seed(seed: int) -> None:
    import random

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
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _extract_far(metrics: Dict[str, Any], class_name: str) -> float:
    return _safe_float(metrics.get("far_per_class", {}).get(class_name, float("nan")))


def _extract_md_rate(metrics: Dict[str, Any], class_name: str) -> float:
    md = metrics.get("missed_detections_per_class", {}).get(class_name, {})
    return _safe_float(md.get("rate", float("nan")))


def _extract_auc_attack(metrics: Dict[str, Any]) -> float:
    auc_per = metrics.get("auc_roc_per_class", {})
    # Prefer Attack for binary, else fall back to first value.
    if "Attack" in auc_per:
        return _safe_float(auc_per.get("Attack"))
    if isinstance(auc_per, dict) and len(auc_per) > 0:
        return _safe_float(next(iter(auc_per.values())))
    return float("nan")


def load_supervised_data(csv_path: str, subsample_size: int, seed: int, task: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    target_col = "Label" if "Label" in df.columns else "label"

    drop_cols = ["Connection Type", "Flow ID", "Timestamp"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    y_raw = df[target_col].astype(str).values
    if task == "binary":
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


def create_sequences(X: np.ndarray, y_onehot: np.ndarray, time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i : i + time_steps])
        ys.append(y_onehot[i + time_steps - 1])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def qpso_select_features(
    X: np.ndarray,
    y_for_qpso: np.ndarray,
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
    X_selected = qpso.fit_transform(X, y_for_qpso)
    return X_selected, float(qpso.computation_time), float(qpso.memory_usage), int(len(qpso.best_features))


def save_plots(out_dir: str, summary_rows: List[Dict[str, Any]]) -> None:
    qubits = [int(r["qubits"]) for r in summary_rows]
    acc = [float(r["accuracy_mean"]) for r in summary_rows]
    prec = [float(r["precision_mean"]) for r in summary_rows]
    rec = [float(r["recall_mean"]) for r in summary_rows]
    f1 = [float(r["f1_mean"]) for r in summary_rows]
    total_time = [float(r["total_time_mean_sec"]) for r in summary_rows]

    acc_std = [float(r["accuracy_std"]) for r in summary_rows]
    prec_std = [float(r["precision_std"]) for r in summary_rows]
    rec_std = [float(r["recall_std"]) for r in summary_rows]
    f1_std = [float(r["f1_std"]) for r in summary_rows]
    total_time_std = [float(r["total_time_std_sec"]) for r in summary_rows]

    plt.figure(figsize=(8, 4.5))
    plt.errorbar(qubits, f1, yerr=f1_std, marker="o", capsize=4)
    plt.xticks(qubits)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Qubits")
    plt.ylabel("F1-score")
    plt.title("Qubit Ablation: F1-score vs Qubits (mean±std)\nBinary LSTM(+VAE)+SGD with QPSO")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "f1_vs_qubits.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.errorbar(qubits, acc, yerr=acc_std, marker="o", capsize=3, label="accuracy")
    plt.errorbar(qubits, prec, yerr=prec_std, marker="o", capsize=3, label="precision")
    plt.errorbar(qubits, rec, yerr=rec_std, marker="o", capsize=3, label="recall")
    plt.errorbar(qubits, f1, yerr=f1_std, marker="o", capsize=3, label="f1")
    plt.xticks(qubits)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Qubits")
    plt.ylabel("Score")
    plt.title("Qubit Ablation: Metrics vs Qubits (mean±std)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_vs_qubits.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.errorbar(qubits, total_time, yerr=total_time_std, marker="o", capsize=4)
    plt.xticks(qubits)
    plt.xlabel("Qubits")
    plt.ylabel("Total time (sec)")
    plt.title("Qubit Ablation: Total runtime vs Qubits (mean±std)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "time_vs_qubits.png"), dpi=200)
    plt.close()


def _pm(mean: float, std: float) -> str:
    if np.isnan(mean) or np.isnan(std):
        return "-"
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def save_table_tex(out_dir: str, summary_rows: List[Dict[str, Any]], best_qubits: int) -> str:
    path = os.path.join(out_dir, "ablation_table.tex")

    header = [
        "Qubits",
        "Acc (\\%)",
        "Pr (\\%)",
        "Rc (\\%)",
        "F1 (\\%)",
        "MSE",
        "FAR(Benign)",
        "FAR(Attack)",
        "MD(Benign)",
        "MD(Attack)",
        "AUC-ROC",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Robust qubit ablation for binary LSTM(+VAE)+SGD using QPSO feature selection (mean$\\pm$std over repeats). Best qubit count: %d.}\\n"
            % best_qubits
        )
        f.write("\\label{tab:qubit_ablation_improved_robust}\n")
        f.write("\\begin{tabular}{r r r r r r r r r r r}\n")
        f.write("\\hline\n")
        f.write("%s \\\\ \\hline\n" % (" & ".join(header)))

        for r in summary_rows:
            f.write(
                "%d & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\ \n"
                % (
                    int(r["qubits"]),
                    _pm(100.0 * float(r["accuracy_mean"]), 100.0 * float(r["accuracy_std"])),
                    _pm(100.0 * float(r["precision_mean"]), 100.0 * float(r["precision_std"])),
                    _pm(100.0 * float(r["recall_mean"]), 100.0 * float(r["recall_std"])),
                    _pm(100.0 * float(r["f1_mean"]), 100.0 * float(r["f1_std"])),
                    _pm(float(r["mse_mean"]), float(r["mse_std"])),
                    _pm(float(r["far_benign_mean"]), float(r["far_benign_std"])),
                    _pm(float(r["far_attack_mean"]), float(r["far_attack_std"])),
                    _pm(float(r["md_benign_mean"]), float(r["md_benign_std"])),
                    _pm(float(r["md_attack_mean"]), float(r["md_attack_std"])),
                    _pm(float(r["auc_roc_attack_mean"]), float(r["auc_roc_attack_std"])),
                )
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    return path


def pick_best(summary_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    sorted_rows = sorted(
        summary_rows,
        key=lambda r: (
            float(r["f1_mean"]),
            float(r["recall_mean"]),
            -float(r["total_time_mean_sec"]),
        ),
        reverse=True,
    )
    return sorted_rows[0]


def _mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="0.ACI-IoT-2023.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=["binary"], default="binary")
    parser.add_argument("--subsample-size", type=int, default=80000)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--qubits", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--qpso-particles", type=int, default=20)
    parser.add_argument("--qpso-iterations", type=int, default=30)
    parser.add_argument("--num-features-to-select", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache-qpso", action="store_true", default=True)
    parser.add_argument("--no-cache-qpso", dest="cache_qpso", action="store_false")

    parser.add_argument("--lstm-units", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=2)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--output-dir", default="outputs/qubit_ablation_improved")
    args = parser.parse_args()

    set_seed(args.seed)

    qubit_list = [int(x.strip()) for x in str(args.qubits).split(",") if x.strip()]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.output_dir, run_id)
    ensure_dir(out_root)
    write_json(os.path.join(out_root, "run_args.json"), vars(args))

    X_sub, y_enc, class_names = load_supervised_data(args.csv, args.subsample_size, args.seed, task=args.task)
    num_classes = len(class_names)
    y_onehot = to_categorical(y_enc, num_classes=num_classes).astype(np.float32)
    y_for_qpso = y_enc

    raw_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for q in qubit_list:
        # Cache QPSO selection + sequence scaling once per qubit.
        # Repeats are used to capture training/split variability, not QPSO variability.
        q_cache: Dict[str, Any] = {}

        if bool(args.cache_qpso):
            q_seed = int(args.seed + q * 1000)
            set_seed(q_seed)
            tf.keras.backend.clear_session()

            t_qpso0 = time.time()
            X_sel, qpso_time, qpso_mem, n_selected = qpso_select_features(
                X_sub,
                y_for_qpso,
                num_qubits=q,
                num_particles=args.qpso_particles,
                iterations=args.qpso_iterations,
                num_features_to_select=args.num_features_to_select,
            )

            X_seq, y_seq = create_sequences(X_sel, y_onehot, args.time_steps)
            n_features = X_seq.shape[2]

            scaler = StandardScaler()
            X_flat = X_seq.reshape(-1, n_features)
            X_scaled = scaler.fit_transform(X_flat).reshape(X_seq.shape).astype(np.float32)

            q_cache = {
                "X_scaled": X_scaled,
                "y_seq": y_seq,
                "n_features": int(n_features),
                "n_selected": int(n_selected),
                "qpso_time_sec": float(qpso_time),
                "qpso_mem": float(qpso_mem),
                "prep_time_sec": float(time.time() - t_qpso0),
            }

        per_repeat: List[Dict[str, Any]] = []

        for rep in range(int(max(1, args.repeats))):
            rep_seed = int(args.seed + q * 1000 + rep)
            set_seed(rep_seed)
            tf.keras.backend.clear_session()

            t0 = time.time()
            if bool(args.cache_qpso) and q_cache:
                X_scaled = q_cache["X_scaled"]
                y_seq = q_cache["y_seq"]
                n_features = int(q_cache["n_features"])
                qpso_time = float(q_cache["qpso_time_sec"])
                qpso_mem = float(q_cache["qpso_mem"])
                n_selected = int(q_cache["n_selected"])
            else:
                # Full per-repeat QPSO (slower) if caching disabled.
                X_sel, qpso_time, qpso_mem, n_selected = qpso_select_features(
                    X_sub,
                    y_for_qpso,
                    num_qubits=q,
                    num_particles=args.qpso_particles,
                    iterations=args.qpso_iterations,
                    num_features_to_select=args.num_features_to_select,
                )

                X_seq, y_seq = create_sequences(X_sel, y_onehot, args.time_steps)
                n_features = X_seq.shape[2]

                scaler = StandardScaler()
                X_flat = X_seq.reshape(-1, n_features)
                X_scaled = scaler.fit_transform(X_flat).reshape(X_seq.shape).astype(np.float32)

            X_train, X_temp, y_train, y_temp = train_test_split(
                X_scaled,
                y_seq,
                test_size=0.3,
                random_state=rep_seed,
                stratify=np.argmax(y_seq, axis=1),
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=0.5,
                random_state=rep_seed,
                stratify=np.argmax(y_temp, axis=1),
            )

            input_shape = (args.time_steps, int(n_features))
            model = LSTM_VAE_Model(
                input_shape=input_shape,
                num_classes=num_classes,
                lstm_units=args.lstm_units,
                attention_heads=args.attention_heads,
                dropout_rate=args.dropout_rate,
                latent_dim=args.latent_dim,
            )
            model.build()
            model.compile_with_sgd(learning_rate=args.learning_rate)

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
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=0,
                callbacks=callbacks,
            )
            training_time = time.time() - t_train

            y_pred, y_pred_prob = model.predict(X_test)
            y_true = np.argmax(y_test, axis=1)

            metrics_calc = MetricsCalculator(class_names)
            metrics = metrics_calc.calculate_all_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_pred_prob=y_pred_prob,
                y_true_onehot=y_test,
                training_time=training_time,
                inference_time=model.inference_time,
                qpso_time=qpso_time,
                memory_mb=qpso_mem + model.memory_usage,
            )

            total_time = time.time() - t0
            val_best = float(np.max(hist.history.get("val_accuracy", [float("nan")])))

            row = {
                "qubits": int(q),
                "repeat": int(rep),
                "seed": int(rep_seed),
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1_score": float(metrics["f1_score"]),
                "mse": float(metrics["mse"]),
                "auc_roc_attack": _extract_auc_attack(metrics),
                "far_benign": _extract_far(metrics, "Benign"),
                "far_attack": _extract_far(metrics, "Attack"),
                "md_benign": _extract_md_rate(metrics, "Benign"),
                "md_attack": _extract_md_rate(metrics, "Attack"),
                "val_best_accuracy": val_best,
                "epochs_trained": int(len(hist.history.get("loss", []))),
                "n_features_selected": int(n_selected),
                "qpso_time_sec": float(qpso_time),
                "training_time_sec": float(training_time),
                "inference_time_sec": _safe_float(metrics.get("inference_time_sec", 0.0)),
                "total_time_sec": float(total_time),
                "memory_mb": _safe_float(metrics.get("memory_mb", 0.0)),
            }

            raw_rows.append(row)
            per_repeat.append(row)
            write_csv(os.path.join(out_root, "ablation_raw.csv"), raw_rows)
            write_json(os.path.join(out_root, "ablation_raw.json"), raw_rows)

        # Aggregate mean±std for this qubit count.
        acc_m, acc_s = _mean_std([r["accuracy"] for r in per_repeat])
        pr_m, pr_s = _mean_std([r["precision"] for r in per_repeat])
        rc_m, rc_s = _mean_std([r["recall"] for r in per_repeat])
        f1_m, f1_s = _mean_std([r["f1_score"] for r in per_repeat])
        mse_m, mse_s = _mean_std([r["mse"] for r in per_repeat])
        auc_m, auc_s = _mean_std([r["auc_roc_attack"] for r in per_repeat])
        far_b_m, far_b_s = _mean_std([r["far_benign"] for r in per_repeat])
        far_a_m, far_a_s = _mean_std([r["far_attack"] for r in per_repeat])
        md_b_m, md_b_s = _mean_std([r["md_benign"] for r in per_repeat])
        md_a_m, md_a_s = _mean_std([r["md_attack"] for r in per_repeat])
        qpso_m, qpso_s = _mean_std([r["qpso_time_sec"] for r in per_repeat])
        train_m, train_s = _mean_std([r["training_time_sec"] for r in per_repeat])
        total_m, total_s = _mean_std([r["total_time_sec"] for r in per_repeat])

        summary_row = {
            "qubits": int(q),
            "repeats": int(len(per_repeat)),
            "n_features_selected_mean": float(np.mean([r["n_features_selected"] for r in per_repeat])),
            "accuracy_mean": acc_m,
            "accuracy_std": acc_s,
            "precision_mean": pr_m,
            "precision_std": pr_s,
            "recall_mean": rc_m,
            "recall_std": rc_s,
            "f1_mean": f1_m,
            "f1_std": f1_s,
            "mse_mean": mse_m,
            "mse_std": mse_s,
            "auc_roc_attack_mean": auc_m,
            "auc_roc_attack_std": auc_s,
            "far_benign_mean": far_b_m,
            "far_benign_std": far_b_s,
            "far_attack_mean": far_a_m,
            "far_attack_std": far_a_s,
            "md_benign_mean": md_b_m,
            "md_benign_std": md_b_s,
            "md_attack_mean": md_a_m,
            "md_attack_std": md_a_s,
            "qpso_time_mean_sec": qpso_m,
            "qpso_time_std_sec": qpso_s,
            "training_time_mean_sec": train_m,
            "training_time_std_sec": train_s,
            "total_time_mean_sec": total_m,
            "total_time_std_sec": total_s,
        }

        summary_rows.append(summary_row)
        write_csv(os.path.join(out_root, "ablation_summary.csv"), summary_rows)
        write_json(os.path.join(out_root, "ablation_summary.json"), summary_rows)

    best = pick_best(summary_rows)
    write_json(os.path.join(out_root, "best_qubits_summary.json"), best)

    save_plots(out_root, summary_rows)
    save_table_tex(out_root, summary_rows, best_qubits=int(best["qubits"]))

    print("Output:", out_root)
    print(
        "Best qubits:",
        best["qubits"],
        "F1(mean):",
        best["f1_mean"],
        "Recall(mean):",
        best["recall_mean"],
        "Total(mean sec):",
        best["total_time_mean_sec"],
    )


if __name__ == "__main__":
    main()
