# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_tex_table(path: str, caption: str, label: str, rows: List[Dict[str, Any]]) -> None:
    cols = ["aggregator", "heterogeneity", "dirichlet_alpha", "accuracy", "precision", "recall", "f1_score", "auc_attack", "time_sec"]

    def fmt(v: Any) -> str:
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\\n")
        f.write(f"\\label{{{label}}}\\n")
        f.write("\\begin{tabular}{l l r r r r r r r}\\n")
        f.write("\\hline\n")
        f.write("Agg. & Hetero. & $\\alpha$ & Acc & Prec & Rec & F1 & AUC & Time(s) \\\\ \\hline\n")
        for r in rows:
            agg = str(r.get("aggregator", "fedavg"))
            het = str(r.get("heterogeneity", ""))
            alpha = r.get("dirichlet_alpha", "")
            alpha_s = "" if alpha in (None, "", 0, 0.0) else fmt(alpha)
            line = (
                f"{agg.replace('_','\\\_')} & {het.replace('_','\\\_')} & {alpha_s} & "
                f"{fmt(r.get('accuracy'))} & {fmt(r.get('precision'))} & {fmt(r.get('recall'))} & {fmt(r.get('f1_score'))} & {fmt(r.get('auc_attack'))} & {fmt(r.get('time_sec'))} \\\\ \n"
            )
            f.write(line)
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def _label_for(r: Dict[str, Any]) -> str:
    het = str(r.get("heterogeneity"))
    agg = str(r.get("aggregator", "fedavg"))
    if het == "dirichlet":
        return f"{agg}:{het}(a={r.get('dirichlet_alpha')})"
    return f"{agg}:{het}"


def _plot_convergence(out_dir: str, results: List[Dict[str, Any]], metric: str, title: str, filename: str) -> str:
    plt.figure(figsize=(10, 5))
    for r in results:
        rms = r.get("round_metrics", [])
        if not rms:
            continue
        xs = [m.get("round") for m in rms]
        ys = [m.get(metric) for m in rms]
        plt.plot(xs, ys, linewidth=2, label=_label_for(r))
    plt.xlabel("Federated Round")
    plt.ylabel(metric.replace("_", " "))
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def _plot_final_bars(out_dir: str, summary_rows: List[Dict[str, Any]]) -> str:
    labels = [_label_for(r) for r in summary_rows]
    f1s = [float(r.get("f1_score", 0.0) or 0.0) for r in summary_rows]
    recs = [float(r.get("recall", 0.0) or 0.0) for r in summary_rows]

    x = list(range(len(labels)))
    width = 0.4

    plt.figure(figsize=(12, 5))
    plt.bar([i - width / 2 for i in x], f1s, width=width, label="F1")
    plt.bar([i + width / 2 for i in x], recs, width=width, label="Recall")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Final Federated Metrics by Strategy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    path = os.path.join(out_dir, "final_metrics_bar.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Federated run directory containing results.json")
    p.add_argument("--title", default="Phase-2 Federated Learning Results")
    args = p.parse_args()

    run_dir = args.run_dir
    results_path = os.path.join(run_dir, "results.json")
    if not os.path.exists(results_path):
        raise SystemExit(f"Missing results.json in: {run_dir}")

    results = _read_json(results_path)

    summary_rows: List[Dict[str, Any]] = []
    for r in results:
        fm = r.get("final_metrics", {})
        summary_rows.append(
            {
                "aggregator": r.get("aggregator", "fedavg"),
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

    summary_csv = os.path.join(run_dir, "summary_from_results.csv")
    _write_csv(summary_csv, summary_rows)

    tex_path = os.path.join(run_dir, "results_table.tex")
    _write_tex_table(
        tex_path,
        caption="Federated learning evaluation under different heterogeneity and aggregation strategies.",
        label="tab:phase2_federated_results",
        rows=summary_rows,
    )

    p1 = _plot_convergence(run_dir, results, metric="accuracy", title=f"{args.title} - Accuracy", filename="accuracy_convergence.png")
    p2 = _plot_convergence(run_dir, results, metric="f1_score", title=f"{args.title} - F1", filename="f1_convergence.png")
    p3 = _plot_final_bars(run_dir, summary_rows)

    print("Generated:")
    for pth in [summary_csv, tex_path, p1, p2, p3]:
        print(pth)


if __name__ == "__main__":
    main()
