import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _read_metrics(run_dir: str) -> Dict:
    path = os.path.join(run_dir, "best_metrics.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_run_args(run_dir: str) -> Dict:
    path = os.path.join(run_dir, "run_args.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_run_title(run_dir: str, run_args: Dict) -> str:
    feature_method = str(run_args.get("feature_method", "")).strip().lower()
    qubits = run_args.get("qubits", None)
    base = "Binary LSTM(+VAE) + SGD"
    if feature_method == "qpso":
        return f"{base} + QPSO (qubits={qubits})"
    if feature_method == "variance":
        return f"{base} (Variance Feature Selection)"
    return base


def _save_metrics_bar(run_dir: str, metrics: Dict, title_prefix: str) -> str:
    keys = ["accuracy", "precision", "recall", "f1_score"]
    vals = [float(metrics[k]) for k in keys]

    plt.figure(figsize=(7, 4))
    bars = plt.bar([k.replace("_", " ") for k in keys], vals)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title(f"{title_prefix}\nTest Metrics")

    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2.0, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    out_path = os.path.join(run_dir, "metrics_bar.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def _save_confusion_matrix(run_dir: str, metrics: Dict) -> str:
    cm = metrics.get("confusion_matrix")
    if not cm or not isinstance(cm, list) or len(cm) != 2:
        raise ValueError("best_metrics.json missing a 2x2 confusion_matrix")

    tn, fp = cm[0]
    fn, tp = cm[1]

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow([[tn, fp], [fn, tp]], cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Benign", "Pred Attack"])
    ax.set_yticklabels(["True Benign", "True Attack"])
    ax.set_title("Confusion Matrix (Test)")

    for i, row in enumerate([[tn, fp], [fn, tp]]):
        for j, val in enumerate(row):
            ax.text(j, i, str(val), ha="center", va="center", color="black", fontsize=11)

    out_path = os.path.join(run_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _fmt_seconds(sec: float) -> str:
    try:
        sec_f = float(sec)
    except Exception:
        return ""
    mins = sec_f / 60.0
    return f"{sec_f:.2f} s ({mins:.2f} min)"


def _save_tables(run_dir: str, metrics: Dict, caption: str, label: str) -> Tuple[str, str]:
    rows: List[Tuple[str, str]] = [
        ("Accuracy", f"{float(metrics.get('accuracy', 0.0)):.4f}"),
        ("Precision", f"{float(metrics.get('precision', 0.0)):.4f}"),
        ("Recall", f"{float(metrics.get('recall', 0.0)):.4f}"),
        ("F1-score", f"{float(metrics.get('f1_score', 0.0)):.4f}"),
        ("ROC-AUC (Attack)", f"{float(metrics.get('auc_roc_per_class', {}).get('Attack', 0.0)):.4f}"),
        ("Training time", _fmt_seconds(metrics.get("training_time_sec", ""))),
        ("Inference time", _fmt_seconds(metrics.get("inference_time_sec", ""))),
        ("Total time", _fmt_seconds(metrics.get("total_time_sec", ""))),
    ]

    csv_path = os.path.join(run_dir, "results_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for k, v in rows:
            w.writerow([k, v])

    tex_path = os.path.join(run_dir, "results_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\\n")
        f.write(f"\\label{{{label}}}\\n")
        f.write("\\begin{tabular}{l r}\n")
        f.write("\\hline\n")
        f.write("Metric & Value \\\\ \\hline\n")
        for k, v in rows:
            k_tex = k.replace("_", "\\_")
            v_tex = str(v).replace("_", "\\_")
            f.write(f"{k_tex} & {v_tex} \\\\ \n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    return csv_path, tex_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=os.path.join("outputs", "quick_tune_binary_variance_hi", "20251222_145849"),
        help="Path to the specific run directory containing best_metrics.json",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run dir not found: {run_dir}")

    metrics = _read_metrics(run_dir)
    run_args = _read_run_args(run_dir)
    title_prefix = _build_run_title(run_dir, run_args)

    feature_method = str(run_args.get("feature_method", "")).strip().lower()
    if feature_method == "qpso":
        caption = f"QPSO feature selection (qubits={run_args.get('qubits')}) for binary LSTM(+VAE) + SGD on ACI-IoT."
        label = "tab:qpso_baseline"
    elif feature_method == "variance":
        caption = "Non-quantum baseline (variance feature selection) for binary LSTM(+VAE) + SGD on ACI-IoT."
        label = "tab:nonquantum_baseline"
    else:
        caption = "Binary LSTM(+VAE) + SGD results on ACI-IoT."
        label = "tab:lstm_vae_sgd"

    paths = []
    paths.append(_save_metrics_bar(run_dir, metrics, title_prefix=title_prefix))
    paths.append(_save_confusion_matrix(run_dir, metrics))
    csv_path, tex_path = _save_tables(run_dir, metrics, caption=caption, label=label)
    paths.extend([csv_path, tex_path])

    print("Generated:")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
